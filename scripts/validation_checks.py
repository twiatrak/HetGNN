from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Allow running this script without requiring `pip install -e .`.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hetgnn_spectral_stability.models import SimpleNodeClassifier
from hetgnn_spectral_stability.regularizers.spectral import lambda2_scipy_normalized
from hetgnn_spectral_stability.regularizers import SSISensor


def canonical_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    u = torch.minimum(row, col)
    v = torch.maximum(row, col)
    return torch.stack([u, v], dim=0)


def undirected_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row, col = edge_index
    u = torch.minimum(row, col)
    v = torch.maximum(row, col)
    return u * num_nodes + v


def make_undirected_edge_index(num_nodes: int, edges: List[Tuple[int, int]]) -> torch.Tensor:
    row = []
    col = []
    for u, v in edges:
        row.extend([u, v])
        col.extend([v, u])
    return torch.tensor([row, col], dtype=torch.long)


def to_undirected_coalesced(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    from torch_geometric.utils import coalesce, to_undirected

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    ones = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32)
    edge_index, _ = coalesce(edge_index, ones, num_nodes=num_nodes, reduce="sum")
    return edge_index


@dataclass
class TinyGraph:
    num_nodes: int
    edge_index: torch.Tensor


def build_toy_graphs() -> Dict[str, TinyGraph]:
    # Disconnected: two triangles
    edges_disc = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    # Bridged: two cliques (0-3) and (4-7) + bridge (3,4)
    edges_clique_a = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    edges_clique_b = [(i, j) for i in range(4, 8) for j in range(i + 1, 8)]
    edges_bridge = edges_clique_a + edges_clique_b + [(3, 4)]
    # Bridge removed
    edges_bridge_removed = edges_clique_a + edges_clique_b
    # Lollipop: clique (0-4) + path (4-9)
    edges_lollipop = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    edges_lollipop += [(4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    # Cycle (0-9)
    edges_cycle = [(i, (i + 1) % 10) for i in range(10)]

    graphs = {
        "disconnected": TinyGraph(6, make_undirected_edge_index(6, edges_disc)),
        "bridged": TinyGraph(8, make_undirected_edge_index(8, edges_bridge)),
        "bridge_removed": TinyGraph(8, make_undirected_edge_index(8, edges_bridge_removed)),
        "lollipop": TinyGraph(10, make_undirected_edge_index(10, edges_lollipop)),
        "cycle": TinyGraph(10, make_undirected_edge_index(10, edges_cycle)),
    }

    for name, g in graphs.items():
        g.edge_index = to_undirected_coalesced(g.edge_index, g.num_nodes)
    return graphs


@torch.no_grad()
def build_model(in_dim: int, hidden_dim: int, out_dim: int) -> SimpleNodeClassifier:
    return SimpleNodeClassifier(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        backbone="gcn",
        dropout=0.0,
        rewire_temperature=1.0,
        rewire_hard=False,
        rewire_symmetric=True,
    )


@torch.no_grad()
def test_no_anchor_duplication(
    model: SimpleNodeClassifier,
    h1: torch.Tensor,
    edge_index_base: torch.Tensor,
    candidate_edge_index: torch.Tensor | None,
    anchor_edge_index: torch.Tensor,
    num_nodes: int,
) -> None:
    rw_ei, _, _, _ = model.rewire(
        h1,
        edge_index_base,
        candidate_edge_index=candidate_edge_index,
        anchor_edge_index=anchor_edge_index,
        sample=False,
    )

    pool = edge_index_base
    if candidate_edge_index is not None:
        pool = torch.cat([pool, candidate_edge_index], dim=1)
    pool = to_undirected_coalesced(pool, num_nodes)

    k_rw = undirected_keys(rw_ei, num_nodes)
    k_pool = undirected_keys(pool, num_nodes)

    k_pool_sorted, _ = torch.sort(k_pool)
    pos = torch.searchsorted(k_pool_sorted, k_rw).clamp(0, k_pool_sorted.numel() - 1)
    if not torch.all(k_pool_sorted[pos] == k_rw):
        raise AssertionError("rewire() produced edges outside (base ∪ candidates) pool")


@torch.no_grad()
def test_anchor_forced_on(
    model: SimpleNodeClassifier,
    h1: torch.Tensor,
    edge_index_base: torch.Tensor,
    anchor_edge_index: torch.Tensor,
    num_nodes: int,
    trials: int = 10,
    tol: float = 1e-6,
) -> None:
    akey = undirected_keys(anchor_edge_index, num_nodes)
    for _ in range(trials):
        rw_ei, rw_w, _, _ = model.rewire(
            h1,
            edge_index_base,
            anchor_edge_index=anchor_edge_index,
            sample=True,
        )
        k = undirected_keys(rw_ei, num_nodes)
        k_sorted, perm = torch.sort(k)
        pos = torch.searchsorted(k_sorted, akey).clamp(0, k_sorted.numel() - 1)
        hit = k_sorted[pos] == akey
        if hit.any():
            w_hit = rw_w[perm[pos[hit]]]
            if not torch.all(w_hit > tol):
                raise AssertionError("anchor edge weight got suppressed; enforcement broken")


def tier1_invariants() -> None:
    torch.manual_seed(7)
    num_nodes = 12
    edge_index_base = make_undirected_edge_index(
        num_nodes,
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (6, 7), (7, 8)],
    )
    edge_index_base = to_undirected_coalesced(edge_index_base, num_nodes)

    anchors = make_undirected_edge_index(num_nodes, [(0, 6), (2, 7), (3, 8)])
    anchors = canonical_undirected(to_undirected_coalesced(anchors, num_nodes))

    candidate = make_undirected_edge_index(num_nodes, [(9, 10), (10, 11), (8, 9)])

    # Merge anchors physically into base
    merged = torch.cat([edge_index_base, to_undirected_coalesced(anchors, num_nodes)], dim=1)
    edge_index_base = to_undirected_coalesced(merged, num_nodes)

    model = build_model(in_dim=8, hidden_dim=8, out_dim=3)
    x = torch.randn(num_nodes, 8)
    h1 = F.relu(model.lin_in(x))
    h1 = model.conv1(h1, edge_index_base)

    test_no_anchor_duplication(model, h1, edge_index_base, candidate, anchors, num_nodes)
    test_anchor_forced_on(model, h1, edge_index_base, anchors, num_nodes)

    print("Tier 1: invariants OK")


def tier2_spectral_sanity() -> None:
    graphs = build_toy_graphs()
    results: Dict[str, float] = {}

    for name, g in graphs.items():
        sensor = SSISensor(num_nodes=g.num_nodes, num_iters=40, num_restarts=4)
        lam2_exact = float(lambda2_scipy_normalized(g.edge_index, None, num_nodes=g.num_nodes))
        lam2_est = float(sensor.estimate(g.edge_index, None).detach().cpu())
        if lam2_exact < -1e-6 or lam2_exact > 2.0 + 1e-6:
            raise AssertionError("Exact λ₂ out of [0,2] range")
        if lam2_est < -1e-6:
            raise AssertionError(f"Estimated λ₂ is negative beyond tolerance: {lam2_est:.6f}")
        results[name] = lam2_exact
        print(f"{name:15s} λ₂ exact={lam2_exact:.6f} est={lam2_est:.6f}")

    if results["disconnected"] > 1e-6:
        raise AssertionError("Disconnected graph λ₂ should be ~0")
    if results["bridge_removed"] > results["bridged"] + 1e-4:
        raise AssertionError("Bridge removal should not increase λ₂")

    print("Tier 2: spectral sanity OK")


def tier3_end_to_end() -> None:
    torch.manual_seed(11)
    num_nodes = 20
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    edge_index_base = to_undirected_coalesced(make_undirected_edge_index(num_nodes, edges), num_nodes)

    anchors = make_undirected_edge_index(num_nodes, [(0, 10), (2, 12), (4, 14)])
    anchors = canonical_undirected(to_undirected_coalesced(anchors, num_nodes))

    # Merge anchors physically into base
    edge_index_base = to_undirected_coalesced(torch.cat([edge_index_base, anchors], dim=1), num_nodes)

    candidate = make_undirected_edge_index(num_nodes, [(1, 11), (3, 13), (5, 15), (6, 16)])

    model = build_model(in_dim=8, hidden_dim=8, out_dim=3)
    sensor = SSISensor(num_nodes=num_nodes, num_iters=20, num_restarts=4)
    x = torch.randn(num_nodes, 8)

    # Pre-healing snapshot
    with torch.no_grad():
        x1 = F.relu(model.lin_in(x))
        h1 = model.conv1(x1, edge_index_base)
        rw_ei_pre, rw_ew_pre, stats_pre, _ = model.rewire(
            h1, edge_index_base, candidate_edge_index=candidate, anchor_edge_index=anchors, sample=False
        )

    from run_nodecls_wikipedia import self_healing_step

    dual_before = float(model.dual_spectral.detach().cpu())
    lam2_pre, lam2_post = self_healing_step(
        model,
        sensor,
        x,
        edge_index_base,
        candidate_edge_index=candidate,
        anchor_shocked=anchors,
        num_nodes=num_nodes,
        target_lambda2=0.4,
        budget_ref_edges=float(edge_index_base.size(1)),
        primal_lr=5e-2,
        dual_lr=1e-2,
        num_iters=6,
        cvar_frac=0.2,
        cvar_samples=8,
        kl_beta=1.0,
        kl_temp=1.0,
        kl_conf=0.0,
        device=torch.device("cpu"),
    )

    with torch.no_grad():
        x1 = F.relu(model.lin_in(x))
        h1 = model.conv1(x1, edge_index_base)
        rw_ei_post, rw_ew_post, stats_post, _ = model.rewire(
            h1, edge_index_base, candidate_edge_index=candidate, anchor_edge_index=anchors, sample=False
        )

    l1_diff = (rw_ew_pre - rw_ew_post).abs().mean().item()
    if l1_diff <= 1e-6:
        raise AssertionError("TTA did not change edge weights")

    dual_after = float(model.dual_spectral.detach().cpu())
    if dual_after < dual_before - 1e-8:
        raise AssertionError("dual_spectral decreased unexpectedly")

    if lam2_post + 1e-6 < lam2_pre:
        raise AssertionError("λ₂ CVaR did not improve after healing")

    print(
        "Tier 3: end-to-end OK | "
        f"Δw={l1_diff:.6f} λ₂_pre={lam2_pre:.4f} λ₂_post={lam2_post:.4f} "
        f"dual={dual_before:.4f}->{dual_after:.4f} "
        f"E={float(stats_pre.expected_num_edges):.2f}->{float(stats_post.expected_num_edges):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=str, default="all", choices=["1", "2", "3", "all"])
    args = parser.parse_args()

    if args.tier in ("1", "all"):
        tier1_invariants()
    if args.tier in ("2", "all"):
        tier2_spectral_sanity()
    if args.tier in ("3", "all"):
        tier3_end_to_end()


if __name__ == "__main__":
    main()
