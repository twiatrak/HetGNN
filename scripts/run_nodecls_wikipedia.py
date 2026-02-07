from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

# Allow running this script without requiring `pip install -e .`.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hetgnn_spectral_stability.config import RegularizationConfig, TTAConfig, get_section, load_yaml_config
from hetgnn_spectral_stability.models import SimpleNodeClassifier
from hetgnn_spectral_stability.regularizers import SSISensor, dirichlet_energy
from hetgnn_spectral_stability.regularizers.spectral import lambda2_scipy_normalized


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().detach().cpu())


def edge_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    row, col = edge_index
    same = (y[row] == y[col]).float()
    return float(same.mean().detach().cpu())


@torch.no_grad()
def build_knn_candidates(x: torch.Tensor, k: int) -> torch.Tensor:
    """Build a small candidate edge pool via cosine kNN (O(n^2) — OK for Chameleon/Squirrel)."""
    x = F.normalize(x, p=2, dim=-1)
    sim = x @ x.t()
    sim.fill_diagonal_(-1.0)
    _, idx = torch.topk(sim, k=k, dim=-1)
    row = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(idx).reshape(-1)
    col = idx.reshape(-1)
    return torch.stack([row, col], dim=0)


def cvar(values: torch.Tensor, frac: float) -> torch.Tensor:
    """Compute CVaR (mean of worst frac) for a 1D tensor."""
    if values.numel() == 0:
        return torch.tensor(float("nan"), device=values.device, dtype=values.dtype)
    k = max(1, int(round(float(frac) * values.numel())))
    worst = torch.topk(values, k=k, largest=False).values
    return worst.mean()


def canonical_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    u = torch.minimum(row, col)
    v = torch.maximum(row, col)
    return torch.stack([u, v], dim=0)


def remove_undirected_pairs(
    edge_index: torch.Tensor, undirected_pairs_to_remove: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Remove undirected pairs from a (possibly directed) edge_index.

    Both inputs are treated as undirected (u<=v) for matching.
    """
    if undirected_pairs_to_remove.numel() == 0:
        return edge_index
    row, col = canonical_undirected(edge_index)
    key = row * num_nodes + col
    ru, rv = undirected_pairs_to_remove
    rkey = ru * num_nodes + rv
    rkey_sorted, _ = torch.sort(rkey)
    pos = torch.searchsorted(rkey_sorted, key)
    pos = torch.clamp(pos, 0, rkey_sorted.numel() - 1)
    hit = rkey_sorted[pos] == key
    return edge_index[:, ~hit]


@torch.no_grad()
def compute_rewired_snapshot(
    model: SimpleNodeClassifier,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    candidate_edge_index: torch.Tensor | None,
    anchor_edge_index: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Deterministic rewired graph snapshot for monitoring (uses expected gates, no dropout)."""

    x1 = model.lin_in(x)
    x1 = F.relu(x1)
    x1 = F.dropout(x1, p=model.dropout, training=False)
    h1 = model.conv1(x1, edge_index)
    h1 = F.relu(h1)
    h1 = F.dropout(h1, p=model.dropout, training=False)

    rw_edge_index, rw_edge_weight, _, _ = model.rewire(
        h1,
        edge_index,
        candidate_edge_index=candidate_edge_index,
        anchor_edge_index=anchor_edge_index,
        sample=False,
    )
    return rw_edge_index, rw_edge_weight, h1


def compute_rewired_snapshot_grad(
    model: SimpleNodeClassifier,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    candidate_edge_index: torch.Tensor | None,
    anchor_edge_index: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Differentiable rewired snapshot for spectral shielding (expected gates, no dropout)."""

    x1 = model.lin_in(x)
    x1 = F.relu(x1)
    x1 = F.dropout(x1, p=model.dropout, training=False)
    h1 = model.conv1(x1, edge_index)
    h1 = F.relu(h1)
    h1 = F.dropout(h1, p=model.dropout, training=False)

    # Shielding is meant to backprop into the gate MLP, not the feature encoder.
    h1_detached = h1.detach()

    rw_edge_index, rw_edge_weight, _, _ = model.rewire(
        h1_detached,
        edge_index,
        candidate_edge_index=candidate_edge_index,
        anchor_edge_index=anchor_edge_index,
        sample=False,
    )
    return rw_edge_index, rw_edge_weight, h1_detached


def dynamic_temperature_from_lambda2(
    lambda2_norm: float,
    tau_min: float,
    tau_max: float,
    target: float,
    slope: float,
) -> float:
    """If lambda2_norm is low (fragmenting), return higher tau; else lower tau."""
    # sigmoid((target - lambda2)/slope) -> 1 when lambda2 << target, 0 when lambda2 >> target
    z = (target - lambda2_norm) / max(slope, 1e-12)
    s = 1.0 / (1.0 + float(torch.exp(torch.tensor(-z))))
    return tau_min + (tau_max - tau_min) * s


@torch.no_grad()
def targeted_spectral_attack(
    model: SimpleNodeClassifier,
    sensor: SSISensor,
    data_x: torch.Tensor,
    edge_index: torch.Tensor,
    candidate_edge_index: torch.Tensor | None,
    anchor_edge_index: torch.Tensor | None,
    attack_pct: float = 0.2,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Targeted spectral attack: remove edges that most contribute to lambda2.
    
    Returns:
        attack_idx: indices of anchor edges to remove (sorted)
        edge_index_attacked: edge_index after removing attacked edges
    """
    if anchor_edge_index is None or anchor_edge_index.size(1) == 0:
        return torch.tensor([], device=device, dtype=torch.long), edge_index

    # Forward pass to get embeddings (deterministic, eval mode)
    x1 = model.lin_in(data_x)
    x1 = F.relu(x1)
    x1 = F.dropout(x1, p=model.dropout, training=False)
    h1 = model.conv1(x1, edge_index)
    h1 = F.relu(h1)
    h1 = F.dropout(h1, p=model.dropout, training=False)
    h1_detached = h1.detach()

    # Compute edge contribution scores via finite differences
    num_anchors = anchor_edge_index.size(1)
    num_attack = max(1, int(round(float(attack_pct) * num_anchors)))

    # Baseline λ₂ on full graph with all anchors
    rw_edge_index_full, rw_edge_weight_full, _, _ = model.rewire(
        h1_detached,
        edge_index,
        candidate_edge_index=candidate_edge_index,
        anchor_edge_index=anchor_edge_index,
        sample=False,
    )
    lam2_baseline = sensor.estimate(rw_edge_index_full, rw_edge_weight_full)

    # For each anchor edge, estimate impact on λ₂ by removing it
    scores = []
    for i in range(num_anchors):
        # Remove edge i from anchors
        keep_mask = torch.ones(num_anchors, device=anchor_edge_index.device, dtype=torch.bool)
        keep_mask[i] = False
        anchor_without_i = anchor_edge_index[:, keep_mask]

        rw_edge_index_i, rw_edge_weight_i, _, _ = model.rewire(
            h1_detached,
            edge_index,
            candidate_edge_index=candidate_edge_index,
            anchor_edge_index=anchor_without_i,
            sample=False,
        )
        lam2_without_i = sensor.estimate(rw_edge_index_i, rw_edge_weight_i)

        # Edge importance = drop in λ₂ when removed
        score = float((lam2_baseline - lam2_without_i).detach().cpu())
        scores.append(score)

    scores_tensor = torch.tensor(scores, device=anchor_edge_index.device, dtype=torch.float32)
    # Top-k indices (highest scores = most important)
    _, top_indices = torch.topk(scores_tensor, k=min(num_attack, num_anchors))
    attack_idx = torch.sort(top_indices)[0]

    attacked_pairs = anchor_edge_index[:, attack_idx]
    edge_index_attacked = remove_undirected_pairs(edge_index, attacked_pairs, num_nodes=data_x.size(0))
    return attack_idx, edge_index_attacked


def self_healing_step(
    model: SimpleNodeClassifier,
    sensor: SSISensor,
    data_x: torch.Tensor,
    edge_index_attacked: torch.Tensor,
    candidate_edge_index: torch.Tensor | None,
    anchor_shocked: torch.Tensor,
    num_nodes: int,
    target_lambda2: float,
    budget_min_ratio: float = 0.6,
    budget_max_ratio: float = 1.4,
    budget_beta: float = 10.0,
    budget_hard: bool = True,
    budget_dual_scale: float = 5.0,
    budget_ref_edges: float | None = None,
    primal_lr: float = 1e-2,
    dual_lr: float = 0.01,
    num_iters: int = 20,
    cvar_frac: float = 0.2,
    cvar_samples: int = 16,
    kl_beta: float = 1.0,
    kl_temp: float = 1.0,
    kl_conf: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """Test-time adaptation loop: adjust topology gates to restore lambda2.
    
    Returns:
        lam2_pre_healing: λ₂ CVaR before adaptation
        lam2_post_healing: λ₂ CVaR after adaptation
    """
    model.eval()

    from torch_geometric.utils import coalesce

    # Freeze everything except gate MLP
    prior_requires = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.rewire.gate_mlp.parameters():
        p.requires_grad_(True)

    opt = torch.optim.Adam(model.rewire.gate_mlp.parameters(), lr=float(primal_lr))

    def e_ref_for(edge_index_eval: torch.Tensor) -> torch.Tensor:
        row, col = edge_index_eval
        u = torch.minimum(row, col)
        v = torch.maximum(row, col)
        undirected_ei = torch.stack([u, v], dim=0)
        ones_local = torch.ones(undirected_ei.size(1), device=edge_index_eval.device, dtype=torch.float32)
        undirected_ei, _ = coalesce(undirected_ei, ones_local, num_nodes=num_nodes, reduce="sum")
        return torch.as_tensor(float(undirected_ei.size(1)), device=edge_index_eval.device, dtype=torch.float32).clamp_min(1.0)

    if budget_ref_edges is None:
        e_ref = e_ref_for(edge_index_attacked)
    else:
        e_ref = torch.as_tensor(float(budget_ref_edges), device=edge_index_attacked.device, dtype=torch.float32).clamp_min(
            1.0
        )

    # Forward pass (deterministic embedding)
    with torch.no_grad():
        x1 = model.lin_in(data_x)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=model.dropout, training=False)
        h1 = model.conv1(x1, edge_index_attacked)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=model.dropout, training=False)
        h1_detached = h1.detach()

        # Reference predictions before adaptation (expected gates)
        rw_ei0, rw_ew0, _, _ = model.rewire(
            h1_detached,
            edge_index_attacked,
            candidate_edge_index=candidate_edge_index,
            anchor_edge_index=anchor_shocked,
            sample=False,
        )
        logits0 = model.conv2(h1_detached, rw_ei0, edge_weight=rw_ew0)
        p0 = F.softmax(logits0 / float(kl_temp), dim=-1)

        if float(kl_conf) > 0.0:
            conf_mask = p0.max(dim=-1).values >= float(kl_conf)
        else:
            conf_mask = torch.ones(num_nodes, device=data_x.device, dtype=torch.bool)

    # Compute pre-healing lambda2 CVaR
    def sample_rewire() -> tuple[torch.Tensor, torch.Tensor | None]:
        rw_ei, rw_ew, _, _ = model.rewire(
            h1_detached,
            edge_index_attacked,
            candidate_edge_index=candidate_edge_index,
            anchor_edge_index=anchor_shocked,
            sample=True,
        )
        return rw_ei, rw_ew

    lam2_pre_healing = sensor.estimate_cvar(
        sample_rewire,
        samples=int(cvar_samples),
        cvar_frac=float(cvar_frac),
    )

    # TTA loop: primal update + dual ascent on spectral constraint + KL consistency
    for _ in range(num_iters):
        opt.zero_grad(set_to_none=True)

        rw_ei_det, rw_ew_det, rw_stats_det, _ = model.rewire(
            h1_detached,
            edge_index_attacked,
            candidate_edge_index=candidate_edge_index,
            anchor_edge_index=anchor_shocked,
            sample=False,
        )
        expected_e = rw_stats_det.expected_num_edges
        ratio = (expected_e / e_ref).clamp_min(0.0)
        vio_lo = (float(budget_min_ratio) - ratio).clamp_min(0.0)
        vio_hi = (ratio - float(budget_max_ratio)).clamp_min(0.0)

        lam2_vals = []
        prob_vals = []
        for _ in range(max(1, int(cvar_samples))):
            rw_ei, rw_ew, _, _ = model.rewire(
                h1_detached,
                edge_index_attacked,
                candidate_edge_index=candidate_edge_index,
                anchor_edge_index=anchor_shocked,
                sample=True,
            )
            lam2_vals.append(sensor.estimate_corrected(rw_ei, rw_ew))

            logits_s = model.conv2(h1_detached, rw_ei, edge_weight=rw_ew)
            prob_vals.append(F.softmax(logits_s / float(kl_temp), dim=-1))

        lam2_stack = torch.stack(lam2_vals, dim=0)
        lam2_cvar = sensor.cvar(lam2_stack, float(cvar_frac))
        spectral_violation = (float(target_lambda2) - lam2_cvar).clamp_min(0.0)
        damp = 1.0 / (1.0 + 10.0 * vio_hi)
        spectral_violation = spectral_violation * damp
        if bool(budget_hard):
            spectral_violation = torch.where(vio_hi > 0, torch.zeros_like(spectral_violation), spectral_violation)

        p_mean = torch.stack(prob_vals, dim=0).mean(dim=0)
        p0_masked = p0[conf_mask]
        p_mean_masked = p_mean[conf_mask]
        kl = (p0_masked * (p0_masked.log() - (p_mean_masked + 1e-12).log())).sum(dim=-1).mean()

        loss = float(kl_beta) * kl + model.dual_spectral * spectral_violation + 0.5 * spectral_violation.pow(2)
        if hasattr(model, "dual_budget_lo"):
            loss = (
                loss
                + model.dual_budget_lo * vio_lo
                + 0.5 * vio_lo.pow(2)
                + model.dual_budget_hi * vio_hi
                + 0.5 * vio_hi.pow(2)
            )
        loss = loss + float(budget_beta) * (vio_lo.pow(2) + vio_hi.pow(2))
        loss.backward()
        opt.step()

        with torch.no_grad():
            if hasattr(model, "dual_budget_lo"):
                model.dual_budget_lo.copy_(
                    (model.dual_budget_lo + float(budget_dual_scale) * dual_lr * vio_lo).clamp_min(0.0)
                )
                model.dual_budget_hi.copy_(
                    (model.dual_budget_hi + float(budget_dual_scale) * dual_lr * vio_hi).clamp_min(0.0)
                )
            model.dual_spectral.copy_(
                (model.dual_spectral + dual_lr * spectral_violation).clamp_min(0.0)
            )

    # Compute post-healing lambda2 CVaR
    lam2_post_healing = sensor.estimate_cvar(
        sample_rewire,
        samples=int(cvar_samples),
        cvar_frac=float(cvar_frac),
    )

    # Restore requires_grad
    for name, p in model.named_parameters():
        p.requires_grad_(prior_requires.get(name, True))

    return float(lam2_pre_healing.detach().cpu()), float(lam2_post_healing.detach().cpu())


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="", help="Optional YAML config with defaults.")
    pre_args, _ = pre_parser.parse_known_args()

    config_data = load_yaml_config(pre_args.config) if pre_args.config else {}
    reg_defaults = get_section(config_data, "regularization")
    tta_defaults = get_section(config_data, "tta")

    def cfg(section: dict, key: str, default):
        return section.get(key, default)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--dataset", type=str, default="chameleon", choices=["chameleon", "squirrel"])
    parser.add_argument(
        "--split",
        type=int,
        default=0,
        help="WikipediaNetwork provides 10 fixed splits; choose which column of the mask to use.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=30, help="Train on original topology before enabling rewiring.")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--backbone", type=str, default="wsage", choices=["gcn", "wsage"])

    parser.add_argument("--rewire-temp", type=float, default=1.0)
    parser.add_argument("--rewire-temp-start", type=float, default=None)
    parser.add_argument("--rewire-temp-end", type=float, default=None)
    parser.add_argument(
        "--rewire-temp-anneal-epochs",
        type=int,
        default=200,
        help="Anneal temperature over this many epochs (if start/end provided).",
    )
    parser.add_argument("--rewire-hard", action="store_true")

    parser.add_argument("--add-knn", type=int, default=0, help="If >0, add candidate edges via kNN in feature space.")

    parser.add_argument(
        "--validate-lambda2-every",
        type=int,
        default=0,
        help="If >0, compute exact lambda2 via SciPy every N epochs (slow, but OK for small graphs).",
    )

    parser.add_argument("--dynamic-temp", action="store_true", help="Enable tau = f(lambda2_norm) feedback control.")
    parser.add_argument("--tau-min", type=float, default=0.1)
    parser.add_argument("--tau-max", type=float, default=2.0)
    parser.add_argument("--lambda2-target", type=float, default=0.10, help="Target lambda2_norm(Lsym) to maintain.")
    parser.add_argument("--lambda2-slope", type=float, default=0.03, help="Sigmoid slope for tau control.")
    parser.add_argument("--tau-ema", type=float, default=0.3, help="EMA rate for tau updates (0..1).")
    parser.add_argument(
        "--dynamic-temp-every",
        type=int,
        default=5,
        help="Update tau every N epochs to reduce compute (recommended 5-10 for scalability).",
    )

    parser.add_argument(
        "--lambda2-correct",
        action="store_true",
        help="Conservative correction: calibrate lambda2 estimate using occasional exact SciPy checks.",
    )
    parser.add_argument(
        "--lambda2-correct-ema",
        type=float,
        default=0.2,
        help="EMA rate for correction factor updates.",
    )

    parser.add_argument(
        "--alpha-shield",
        type=float,
        default=0.0,
        help="Spectral shielding strength: adds a differentiable penalty to push gates to increase lambda2 when low.",
    )
    parser.add_argument(
        "--shield-every",
        type=int,
        default=5,
        help="Compute spectral shielding penalty every N epochs (reuse tau snapshot when possible).",
    )

    parser.add_argument(
        "--log-csv",
        type=str,
        default="",
        help="If set, write per-epoch metrics to <path>_train.csv and <path>_ssi.csv.",
    )
    parser.add_argument(
        "--monitor-every",
        type=int,
        default=5,
        help="Compute SSI monitor stats (lambda2/Dirichlet) every N epochs for logging.",
    )

    parser.add_argument(
        "--export-edges-at",
        type=int,
        default=0,
        help="If >0, export rewired edges at this epoch to data/exports/*.pt.",
    )

    parser.add_argument("--alpha-dir", type=float, default=cfg(reg_defaults, "alpha_dirichlet", 0.0))
    parser.add_argument("--alpha-conn", type=float, default=cfg(reg_defaults, "alpha_connectivity", 0.0))
    parser.add_argument("--conn-eps", type=float, default=cfg(reg_defaults, "connectivity_eps", 1e-2))
    parser.add_argument("--conn-iters", type=int, default=cfg(reg_defaults, "connectivity_iters", 25))
    parser.add_argument("--alpha-edge-budget", type=float, default=cfg(reg_defaults, "alpha_edge_budget", 1.0))
    parser.add_argument("--edge-budget-min-ratio", type=float, default=cfg(reg_defaults, "edge_budget_min_ratio", 0.6))
    parser.add_argument("--edge-budget-max-ratio", type=float, default=cfg(reg_defaults, "edge_budget_max_ratio", 1.4))
    parser.add_argument("--dual-lr", type=float, default=cfg(reg_defaults, "dual_lr", 1e-2), help="Dual ascent step size.")
    parser.add_argument("--cvar-samples", type=int, default=cfg(reg_defaults, "cvar_samples", 4), help="CVaR samples for lambda2 risk.")
    parser.add_argument("--cvar-frac", type=float, default=cfg(reg_defaults, "cvar_frac", 0.5), help="Tail fraction for CVaR (0..1].")
    parser.add_argument("--anchor-knn", type=int, default=0, help="If >0, anchor edges via kNN in feature space.")
    parser.add_argument("--zero-shot-shock", action="store_true", help="Run anchor shock evaluation after training.")
    parser.add_argument("--shock-drop-pct", type=float, default=0.1, help="Fraction of anchors to delete.")
    parser.add_argument("--shock-samples", type=int, default=10, help="MC samples for CVaR accuracy.")
    parser.add_argument("--shock-cvar-frac", type=float, default=0.1, help="Tail fraction for CVaR accuracy.")
    parser.add_argument("--shock-seed", type=int, default=0, help="Seed for anchor deletion.")
    parser.add_argument("--shock-adapt", action="store_true", help="Enable test-time adaptation.")
    parser.add_argument("--shock-adapt-steps", type=int, default=5, help="Adaptation steps at test time.")
    parser.add_argument("--shock-primal-steps", type=int, default=0, help="Primal steps on gate MLP at test time.")
    parser.add_argument("--shock-primal-lr", type=float, default=1e-2, help="Learning rate for primal adaptation.")
    parser.add_argument("--shock-primal-samples", type=int, default=1, help="Samples for primal risk proxy.")
    parser.add_argument("--shock-reset-duals", action="store_true", help="Reset duals before adaptation.")
    parser.add_argument("--no-shock-baseline", action="store_true", help="Skip baseline training for shock eval.")
    parser.add_argument("--shock-baseline-backbone", type=str, default="gcn", choices=["gcn", "wsage"])
    
    # Targeted spectral attack & self-healing evaluation
    parser.add_argument("--targeted-spectral-attack", action="store_true", help="Use targeted spectral attack (not random).")
    parser.add_argument("--attack-pct", type=float, default=0.15, help="Fraction of anchor edges to attack.")
    parser.add_argument("--tta-healing-steps", type=int, default=cfg(tta_defaults, "healing_steps", 20), help="TTA iterations for self-healing.")
    parser.add_argument("--tta-dual-lr", type=float, default=cfg(tta_defaults, "dual_lr", 0.01), help="Dual learning rate for TTA.")
    parser.add_argument("--tta-primal-lr", type=float, default=cfg(tta_defaults, "primal_lr", 1e-2), help="Primal learning rate for TTA gate MLP.")
    parser.add_argument("--tta-kl-beta", type=float, default=cfg(tta_defaults, "kl_beta", 1.0), help="KL consistency weight during TTA.")
    parser.add_argument("--tta-kl-temp", type=float, default=cfg(tta_defaults, "kl_temp", 1.0), help="Temperature for KL consistency.")
    parser.add_argument("--tta-kl-conf", type=float, default=cfg(tta_defaults, "kl_conf", 0.0), help="Confidence threshold for KL mask (0=all nodes).")
    parser.add_argument("--tta-cvar-samples", type=int, default=cfg(tta_defaults, "cvar_samples", 16), help="CVaR samples for TTA.")
    parser.add_argument("--tta-cvar-frac", type=float, default=cfg(tta_defaults, "cvar_frac", 0.2), help="Tail fraction for TTA CVaR.")
    parser.add_argument("--tta-budget-beta", type=float, default=cfg(tta_defaults, "budget_beta", 10.0), help="Budget penalty weight during TTA.")
    parser.add_argument(
        "--tta-budget-hard",
        action=argparse.BooleanOptionalAction,
        default=cfg(tta_defaults, "budget_hard", False),
        help="Block spectral healing when budget max is violated.",
    )
    parser.add_argument("--tta-budget-dual-scale", type=float, default=cfg(tta_defaults, "budget_dual_scale", 5.0), help="Scale for budget dual updates during TTA.")
    parser.add_argument(
        "--tta-budget-ref-base",
        action=argparse.BooleanOptionalAction,
        default=cfg(tta_defaults, "budget_ref_base", False),
        help="Use original (pre-shock) edge count as budget reference during TTA.",
    )
    parser.add_argument("--comparative-eval", action="store_true", help="Run 3-way comparative eval (baseline vs frozen vs healing).")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from torch_geometric.datasets import WikipediaNetwork
        from torch_geometric.transforms import ToUndirected
    except Exception as exc:
        raise RuntimeError(
            "This script requires PyTorch Geometric. Install torch + torch_geometric per README.md."
        ) from exc

    dataset = WikipediaNetwork(root="data/WikipediaNetwork", name=args.dataset, transform=ToUndirected())
    data = dataset[0].to(device)

    def pick_split(mask: torch.Tensor, split: int) -> torch.Tensor:
        # WikipediaNetwork masks are often [N, 10]. Some transforms/datasets use [N].
        if mask.dim() == 1:
            return mask
        if mask.dim() == 2:
            if not (0 <= split < mask.size(1)):
                raise ValueError(f"split must be in [0, {mask.size(1) - 1}], got {split}")
            return mask[:, split]
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")

    train_mask = pick_split(data.train_mask, args.split)
    val_mask = pick_split(data.val_mask, args.split)
    test_mask = pick_split(data.test_mask, args.split)

    reg = RegularizationConfig(
        alpha_dirichlet=float(args.alpha_dir),
        alpha_connectivity=float(args.alpha_conn),
        connectivity_eps=float(args.conn_eps),
        connectivity_iters=int(args.conn_iters),
        alpha_edge_budget=float(args.alpha_edge_budget),
        edge_budget_min_ratio=float(args.edge_budget_min_ratio),
        edge_budget_max_ratio=float(args.edge_budget_max_ratio),
        dual_lr=float(args.dual_lr),
        cvar_samples=int(args.cvar_samples),
        cvar_frac=float(args.cvar_frac),
    )

    tta = TTAConfig(
        healing_steps=int(args.tta_healing_steps),
        dual_lr=float(args.tta_dual_lr),
        primal_lr=float(args.tta_primal_lr),
        kl_beta=float(args.tta_kl_beta),
        kl_temp=float(args.tta_kl_temp),
        kl_conf=float(args.tta_kl_conf),
        cvar_samples=int(args.tta_cvar_samples),
        cvar_frac=float(args.tta_cvar_frac),
        budget_beta=float(args.tta_budget_beta),
        budget_hard=bool(args.tta_budget_hard),
        budget_dual_scale=float(args.tta_budget_dual_scale),
        budget_ref_base=bool(args.tta_budget_ref_base),
    )

    model = SimpleNodeClassifier(
        in_dim=dataset.num_features,
        hidden_dim=int(args.hidden),
        out_dim=dataset.num_classes,
        backbone=args.backbone,
        dropout=float(args.dropout),
        rewire_temperature=float(args.rewire_temp),
        rewire_hard=bool(args.rewire_hard),
        rewire_symmetric=True,
        reg=reg,
    ).to(device)

    sensor = SSISensor(
        num_nodes=int(data.num_nodes),
        num_iters=int(args.conn_iters),
        num_restarts=4,
        cvar_samples=int(args.cvar_samples),
        cvar_frac=float(args.cvar_frac),
    )

    candidate_edge_index = None
    if int(args.add_knn) > 0:
        candidate_edge_index = build_knn_candidates(data.x, k=int(args.add_knn))

    anchor_edge_index = None
    anchor_edge_index_undirected = None
    if int(args.anchor_knn) > 0:
        anchor_edge_index = build_knn_candidates(data.x, k=int(args.anchor_knn))

    # Build augmented topology (original + anchors) for message passing.
    edge_index_base = data.edge_index
    if anchor_edge_index is not None:
        from torch_geometric.utils import coalesce, to_undirected

        anchor_edge_index_undirected = to_undirected(anchor_edge_index, num_nodes=data.num_nodes)
        merged = torch.cat([edge_index_base, anchor_edge_index_undirected], dim=1)
        ones = torch.ones(merged.size(1), device=device, dtype=torch.float32)
        edge_index_base, _ = coalesce(merged, ones, num_nodes=data.num_nodes, reduce="sum")

        # Canonical undirected anchors for gating/masking (u<=v).
        anchor_edge_index = canonical_undirected(anchor_edge_index_undirected)
        ones_anchor = torch.ones(anchor_edge_index.size(1), device=device, dtype=torch.float32)
        anchor_edge_index, _ = coalesce(anchor_edge_index, ones_anchor, num_nodes=data.num_nodes, reduce="sum")

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_csv_path = None
    ssi_csv_path = None
    if args.log_csv:
        csv_base = Path(args.log_csv)
        train_csv_path = csv_base.with_name(f"{csv_base.stem}_train{csv_base.suffix}")
        ssi_csv_path = csv_base.with_name(f"{csv_base.stem}_ssi{csv_base.suffix}")

    if train_csv_path is not None:
        train_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with train_csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "epoch",
                    "enable_rewire",
                    "tau",
                    "keep_rate",
                    "loss",
                    "train_acc",
                    "val_acc",
                    "test_acc",
                ]
            )

    if ssi_csv_path is not None:
        ssi_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with ssi_csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "epoch",
                    "lambda2_norm_est",
                    "dirichlet_mean",
                    "lambda2_cvar",
                    "expected_num_edges",
                    "expected_e_ratio",
                    "budget_violation_lo",
                    "budget_violation_hi",
                    "spectral_violation",
                    "dual_budget_lo",
                    "dual_budget_hi",
                    "dual_spectral_pressure",
                ]
            )

    best_val = -1.0
    best_test = -1.0
    best_state = None

    # Cache for expensive monitoring within an epoch.
    cached_epoch = -1
    cached_rw: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None = None
    cached_lam2_est: float | None = None

    for epoch in range(1, int(args.epochs) + 1):
        enable_rewire = epoch > int(args.warmup_epochs)

        # Determine whether we need a deterministic monitoring snapshot this epoch.
        need_monitor = bool(
            enable_rewire
            and (
                (args.dynamic_temp and epoch % int(args.dynamic_temp_every) == 0)
                or (ssi_csv_path is not None and epoch % int(args.monitor_every) == 0)
                or (
                    int(args.validate_lambda2_every) > 0
                    and epoch % int(args.validate_lambda2_every) == 0
                )
                or (int(args.export_edges_at) > 0 and epoch == int(args.export_edges_at))
            )
        )

        if need_monitor and cached_epoch != epoch:
            cached_rw = compute_rewired_snapshot(
                model, data.x, edge_index_base, candidate_edge_index, anchor_edge_index
            )
            cached_lam2_est = float(sensor.estimate(cached_rw[0], cached_rw[1]).detach().cpu())
            cached_epoch = epoch

        # Temperature control
        if enable_rewire:
            if args.dynamic_temp and epoch % int(args.dynamic_temp_every) == 0:
                if cached_epoch != epoch or cached_lam2_est is None:
                    # Fallback (should be rare, but keep this robust).
                    cached_rw = compute_rewired_snapshot(
                        model, data.x, edge_index_base, candidate_edge_index, anchor_edge_index
                    )
                    cached_lam2_est = float(sensor.estimate(cached_rw[0], cached_rw[1]).detach().cpu())
                    cached_epoch = epoch

                lam2_used = float(sensor.corr_scale) * float(cached_lam2_est)
                desired_tau = dynamic_temperature_from_lambda2(
                    lambda2_norm=lam2_used,
                    tau_min=float(args.tau_min),
                    tau_max=float(args.tau_max),
                    target=float(args.lambda2_target),
                    slope=float(args.lambda2_slope),
                )
                ema = float(args.tau_ema)
                model.rewire.temperature = (1.0 - ema) * float(model.rewire.temperature) + ema * desired_tau
            elif args.rewire_temp_start is not None and args.rewire_temp_end is not None:
                # Fixed schedule annealing (optional fallback)
                T0 = float(args.rewire_temp_start)
                T1 = float(args.rewire_temp_end)
                denom = max(1, int(args.rewire_temp_anneal_epochs))
                t = min(1.0, (epoch - 1) / denom)
                model.rewire.temperature = (1.0 - t) * T0 + t * T1

        model.train()
        opt.zero_grad(set_to_none=True)

        logits, regs = model(
            data.x,
            edge_index_base,
            candidate_edge_index=candidate_edge_index,
            anchor_edge_index=anchor_edge_index,
            enable_rewire=enable_rewire,
            rewire_sample=None,
        )
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])

        # Spectral shielding: add a differentiable penalty that pushes gates to increase lambda2 when low.
        if (
            enable_rewire
            and float(args.alpha_shield) > 0.0
            and int(args.shield_every) > 0
            and epoch % int(args.shield_every) == 0
        ):
            rw_edge_index_s, rw_edge_weight_s, _ = compute_rewired_snapshot_grad(
                model, data.x, edge_index_base, candidate_edge_index, anchor_edge_index
            )
            lam2_s = sensor.estimate_corrected(rw_edge_index_s, rw_edge_weight_s)
            slack = (float(args.lambda2_target) - lam2_s).clamp_min(0.0)
            loss = loss + float(args.alpha_shield) * (slack * slack)

        if reg.alpha_dirichlet != 0.0:
            loss = loss + reg.alpha_dirichlet * regs["dirichlet"]

        # Primal-dual constrained losses.
        if enable_rewire:
            if "edge_budget_dual" in regs:
                loss = loss + float(reg.alpha_edge_budget) * regs["edge_budget_dual"]
            if "spectral_dual" in regs:
                loss = loss + float(reg.alpha_connectivity) * regs["spectral_dual"]

        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits, regs_eval = model(
                data.x,
                edge_index_base,
                candidate_edge_index=candidate_edge_index,
                anchor_edge_index=anchor_edge_index,
                enable_rewire=enable_rewire,
                rewire_sample=None,
            )
            train_acc = accuracy(logits[train_mask], data.y[train_mask])
            val_acc = accuracy(logits[val_mask], data.y[val_mask])
            test_acc = accuracy(logits[test_mask], data.y[test_mask])

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            keep = float(regs_eval.get("expected_keep_rate", torch.tensor(float("nan"))).cpu())
            extra = f" keep~{keep:.3f}"
            extra += f" T={getattr(model.rewire, 'temperature', float('nan')):.2f}"
            extra += f" warmup={'on' if enable_rewire else 'off'}"
            print(
                f"Epoch {epoch:04d} | loss {float(loss.detach()):.4f} | "
                f"train {train_acc:.3f} val {val_acc:.3f} test {test_acc:.3f} | best_test {best_test:.3f}" + extra
            )

        if train_csv_path is not None:
            with train_csv_path.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        epoch,
                        int(enable_rewire),
                        float(getattr(model.rewire, "temperature", float("nan"))),
                        float(regs_eval.get("expected_keep_rate", torch.tensor(float("nan"))).cpu()),
                        float(loss.detach().cpu()),
                        train_acc,
                        val_acc,
                        test_acc,
                    ]
                )

        if ssi_csv_path is not None:
            lam2_norm_est_log = float("nan")
            dir_mean = float("nan")
            if enable_rewire and int(args.monitor_every) > 0 and epoch % int(args.monitor_every) == 0:
                if cached_epoch == epoch and cached_rw is not None and cached_lam2_est is not None:
                    rw_edge_index_m, rw_edge_weight_m, h1_m = cached_rw
                    lam2_norm_est_log = float(sensor.corr_scale) * float(cached_lam2_est)
                    dir_mean = float(
                        dirichlet_energy(h1_m, rw_edge_index_m, rw_edge_weight_m, reduce="mean").detach().cpu()
                    )

            with ssi_csv_path.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        epoch,
                        lam2_norm_est_log,
                        dir_mean,
                        float(regs_eval.get("lambda2_cvar", torch.tensor(float("nan"))).detach().cpu()),
                        float(regs_eval.get("expected_num_edges", torch.tensor(float("nan"))).detach().cpu()),
                        float(regs_eval.get("expected_e_ratio", torch.tensor(float("nan"))).detach().cpu()),
                        float(regs_eval.get("budget_violation_lo", torch.tensor(float("nan"))).detach().cpu()),
                        float(regs_eval.get("budget_violation_hi", torch.tensor(float("nan"))).detach().cpu()),
                        float(regs_eval.get("spectral_violation", torch.tensor(float("nan"))).detach().cpu()),
                        float(regs_eval.get("dual_budget_lo", torch.tensor(float("nan"))).detach().cpu()),
                        float(regs_eval.get("dual_budget_hi", torch.tensor(float("nan"))).detach().cpu()),
                        float(regs_eval.get("dual_spectral", torch.tensor(float("nan"))).detach().cpu()),
                    ]
                )

        # Milestone: validate lambda2 estimate vs actual
        if int(args.validate_lambda2_every) > 0 and epoch % int(args.validate_lambda2_every) == 0 and enable_rewire:
            if cached_epoch != epoch or cached_rw is None:
                cached_rw = compute_rewired_snapshot(
                    model, data.x, edge_index_base, candidate_edge_index, anchor_edge_index
                )
                cached_epoch = epoch
            rw_edge_index, rw_edge_weight, _ = cached_rw
            if args.lambda2_correct:
                est_norm, actual, corr = sensor.calibrate_with_exact(
                    rw_edge_index,
                    rw_edge_weight,
                    ema=float(args.lambda2_correct_ema),
                )
                used = corr * est_norm
            else:
                est_norm = float(sensor.estimate(rw_edge_index, rw_edge_weight).detach().cpu())
                actual = float(lambda2_scipy_normalized(rw_edge_index, rw_edge_weight, num_nodes=data.num_nodes))
                used = float(sensor.corr_scale) * est_norm

            print(
                f"  lambda2_norm: est~{est_norm:.6f} actual~{actual:.6f} corr~{float(sensor.corr_scale):.3f} used~{used:.6f}"
            )

        # Milestone: export rewired adjacency snapshot
        if int(args.export_edges_at) > 0 and epoch == int(args.export_edges_at):
            import os
            os.makedirs("data/exports", exist_ok=True)

            # Build a deterministic snapshot (eval-mode, enable rewiring)
            if cached_epoch == epoch and cached_rw is not None:
                rw_edge_index, rw_edge_weight, h1 = cached_rw
            else:
                rw_edge_index, rw_edge_weight, h1 = compute_rewired_snapshot(
                    model, data.x, edge_index_base, candidate_edge_index, anchor_edge_index
                )

            orig_h = edge_homophily(edge_index_base, data.y)
            rw_h = edge_homophily(rw_edge_index, data.y)
            lam2_norm_est = float(sensor.estimate_corrected(rw_edge_index, rw_edge_weight).detach().cpu())
            payload = {
                "dataset": args.dataset,
                "epoch": epoch,
                "split": int(args.split),
                "tau": float(getattr(model.rewire, "temperature", float("nan"))),
                "orig_edge_index": edge_index_base.detach().cpu(),
                "rewired_edge_index": rw_edge_index.detach().cpu(),
                "rewired_edge_weight": None if rw_edge_weight is None else rw_edge_weight.detach().cpu(),
                "orig_homophily": orig_h,
                "rewired_homophily": rw_h,
                "lambda2_norm_est": lam2_norm_est,
                "dirichlet_mean": float(dirichlet_energy(h1, rw_edge_index, rw_edge_weight, reduce="mean").detach().cpu()),
            }
            out_path = f"data/exports/{args.dataset}_split{int(args.split)}_epoch{epoch}.pt"
            torch.save(payload, out_path)
            print(f"Exported rewired graph to {out_path} | homophily orig={orig_h:.3f} rewired={rw_h:.3f}")

    print("Done.")
    print("Regularization config:", asdict(reg))
    print("TTA config:", asdict(tta))
    print(f"Best val {best_val:.3f} | Best test {best_test:.3f}")

    if args.zero_shot_shock:
        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
        model.eval()

        if anchor_edge_index is None:
            raise RuntimeError("--zero-shot-shock requires --anchor-knn to define anchor edges.")

        from torch_geometric.utils import coalesce

        # **TARGETED SPECTRAL ATTACK (Optional)**
        if args.targeted_spectral_attack:
            print("\n=== Targeted Spectral Attack ===")
            attack_idx, edge_index_attacked = targeted_spectral_attack(
                model,
                sensor,
                data.x,
                edge_index_base,
                candidate_edge_index=candidate_edge_index,
                anchor_edge_index=anchor_edge_index,
                attack_pct=float(args.attack_pct),
                device=device,
            )
            print(
                f"Targeted attack identified {len(attack_idx)} critical edges "
                f"({float(len(attack_idx)) / max(1, anchor_edge_index.size(1)):.1%} of anchors)"
            )
            edge_index_shocked = edge_index_attacked
            num_anchors = anchor_edge_index.size(1)
            keep_mask = torch.ones(num_anchors, dtype=torch.bool, device=device)
            keep_mask[attack_idx] = False
            anchor_shocked = anchor_edge_index[:, keep_mask]
        else:
            # Standard random deletion
            torch.manual_seed(int(args.shock_seed))
            num_anchors = anchor_edge_index.size(1)
            keep_n = max(1, int(round((1.0 - float(args.shock_drop_pct)) * num_anchors)))
            perm = torch.randperm(num_anchors, device=anchor_edge_index.device)[:keep_n]
            anchor_shocked = anchor_edge_index[:, perm]

            drop_mask = torch.ones(num_anchors, device=anchor_edge_index.device, dtype=torch.bool)
            drop_mask[perm] = False
            anchor_dropped = anchor_edge_index[:, drop_mask]
            edge_index_shocked = remove_undirected_pairs(edge_index_base, anchor_dropped, num_nodes=data.num_nodes)

        ones = torch.ones(edge_index_shocked.size(1), device=device, dtype=torch.float32)
        edge_index_shocked, _ = coalesce(edge_index_shocked, ones, num_nodes=data.num_nodes, reduce="sum")

        def e_ref_for(edge_index_eval: torch.Tensor) -> torch.Tensor:
            row, col = edge_index_eval
            u = torch.minimum(row, col)
            v = torch.maximum(row, col)
            undirected_ei = torch.stack([u, v], dim=0)
            ones_local = torch.ones(undirected_ei.size(1), device=edge_index_eval.device, dtype=torch.float32)
            undirected_ei, _ = coalesce(undirected_ei, ones_local, num_nodes=data.num_nodes, reduce="sum")
            return torch.as_tensor(float(undirected_ei.size(1)), device=edge_index_eval.device, dtype=torch.float32).clamp_min(
                1.0
            )

        def adapt_topology(steps: int, primal_steps: int, primal_lr: float) -> None:
            gate_params = list(model.rewire.gate_mlp.parameters())
            opt_primal = torch.optim.SGD(gate_params, lr=float(primal_lr)) if primal_steps > 0 else None
            for _ in range(max(1, int(steps))):
                with torch.no_grad():
                    x1 = model.lin_in(data.x)
                    x1 = F.relu(x1)
                    x1 = F.dropout(x1, p=model.dropout, training=False)
                    h1 = model.conv1(x1, edge_index_shocked)
                    h1 = F.relu(h1)
                    h1 = F.dropout(h1, p=model.dropout, training=False)
                    h1_detached = h1.detach()

                # Primal update: adjust gate MLP to reduce violations on shocked graph.
                if opt_primal is not None and int(primal_steps) > 0:
                    for _ in range(int(primal_steps)):
                        opt_primal.zero_grad(set_to_none=True)
                        rw_edge_index_e, rw_edge_weight_e, rw_stats_e, _ = model.rewire(
                            h1_detached,
                            edge_index_shocked,
                            candidate_edge_index=candidate_edge_index,
                            anchor_edge_index=anchor_shocked,
                            sample=False,
                        )
                        expected_e = rw_stats_e.expected_num_edges
                        ratio = (expected_e / e_ref_for(edge_index_shocked)).clamp_min(0.0)
                        vio_lo = (float(args.edge_budget_min_ratio) - ratio).clamp_min(0.0)
                        vio_hi = (ratio - float(args.edge_budget_max_ratio)).clamp_min(0.0)

                        def sample_primal() -> tuple[torch.Tensor, torch.Tensor | None]:
                            rw_edge_index_s, rw_edge_weight_s, _, _ = model.rewire(
                                h1_detached,
                                edge_index_shocked,
                                candidate_edge_index=candidate_edge_index,
                                anchor_edge_index=anchor_shocked,
                                sample=True,
                            )
                            return rw_edge_index_s, rw_edge_weight_s

                        lam2_risk_e = sensor.estimate_cvar(
                            sample_primal,
                            samples=int(args.shock_primal_samples),
                            cvar_frac=float(args.cvar_frac),
                        )
                        spectral_violation_e = (float(args.conn_eps) - lam2_risk_e).clamp_min(0.0)

                        loss_primal = (
                            model.dual_budget_lo * vio_lo
                            + 0.5 * vio_lo.pow(2)
                            + model.dual_budget_hi * vio_hi
                            + 0.5 * vio_hi.pow(2)
                            + model.dual_spectral * spectral_violation_e
                            + 0.5 * spectral_violation_e.pow(2)
                        )
                        loss_primal.backward()
                        opt_primal.step()

                # Dual update: use CVaR risk under stochastic rewiring.
                rw_edge_index_e, rw_edge_weight_e, rw_stats_e, _ = model.rewire(
                    h1_detached,
                    edge_index_shocked,
                    candidate_edge_index=candidate_edge_index,
                    anchor_edge_index=anchor_shocked,
                    sample=False,
                )
                expected_e = rw_stats_e.expected_num_edges
                ratio = (expected_e / e_ref_for(edge_index_shocked)).clamp_min(0.0)
                vio_lo = (float(args.edge_budget_min_ratio) - ratio).clamp_min(0.0)
                vio_hi = (ratio - float(args.edge_budget_max_ratio)).clamp_min(0.0)

                def sample_dual() -> tuple[torch.Tensor, torch.Tensor | None]:
                    rw_edge_index_s, rw_edge_weight_s, _, _ = model.rewire(
                        h1_detached,
                        edge_index_shocked,
                        candidate_edge_index=candidate_edge_index,
                        anchor_edge_index=anchor_shocked,
                        sample=True,
                    )
                    return rw_edge_index_s, rw_edge_weight_s

                lam2_cvar = sensor.estimate_cvar(
                    sample_dual,
                    samples=int(args.cvar_samples),
                    cvar_frac=float(args.cvar_frac),
                )
                spectral_violation = (float(args.conn_eps) - lam2_cvar).clamp_min(0.0)

                with torch.no_grad():
                    model.dual_budget_lo.copy_((model.dual_budget_lo + model.dual_lr * vio_lo).clamp_min(0.0))
                    model.dual_budget_hi.copy_((model.dual_budget_hi + model.dual_lr * vio_hi).clamp_min(0.0))
                    model.dual_spectral.copy_((model.dual_spectral + model.dual_lr * spectral_violation).clamp_min(0.0))

        def eval_cvar_accuracy(
            model_eval: SimpleNodeClassifier, enable_rewire: bool, edge_index_eval: torch.Tensor
        ) -> tuple[float, float, float, float]:
            accs = []
            min_acc = float("inf")
            max_acc = float("-inf")
            for _ in range(max(1, int(args.shock_samples))):
                logits_s, _ = model_eval(
                    data.x,
                    edge_index_eval,
                    candidate_edge_index=candidate_edge_index,
                    anchor_edge_index=anchor_shocked,
                    enable_rewire=enable_rewire,
                    rewire_sample=True if enable_rewire else None,
                )
                acc = accuracy(logits_s[test_mask], data.y[test_mask])
                accs.append(acc)
                min_acc = min(min_acc, acc)
                max_acc = max(max_acc, acc)
            acc_t = torch.tensor(accs, device=data.x.device, dtype=torch.float32)
            return (
                float(acc_t.mean().cpu()),
                float(cvar(acc_t, args.shock_cvar_frac).cpu()),
                float(min_acc),
                float(max_acc),
            )

        def eval_shock_stats(
            model_eval: SimpleNodeClassifier, enable_rewire: bool
        ) -> tuple[float, float, float, float]:
            if not enable_rewire:
                lam2 = sensor.estimate_corrected(edge_index_shocked, None)
                ratio = torch.tensor(1.0, device=device)
                vio_lo = float(args.edge_budget_min_ratio) - float(ratio)
                vio_hi = float(ratio) - float(args.edge_budget_max_ratio)
                return (
                    float(lam2.detach().cpu()),
                    float(ratio),
                    float(max(0.0, vio_lo)),
                    float(max(0.0, vio_hi)),
                )

            x1 = model_eval.lin_in(data.x)
            x1 = F.relu(x1)
            x1 = F.dropout(x1, p=model_eval.dropout, training=False)
            h1 = model_eval.conv1(x1, edge_index_shocked)
            h1 = F.relu(h1)
            h1 = F.dropout(h1, p=model_eval.dropout, training=False)
            h1_detached = h1.detach()

            rw_edge_index_det, rw_edge_weight_det, rw_stats_det, _ = model_eval.rewire(
                h1_detached,
                edge_index_shocked,
                candidate_edge_index=candidate_edge_index,
                anchor_edge_index=anchor_shocked,
                sample=False,
            )
            ratio = (rw_stats_det.expected_num_edges / e_ref_for(edge_index_shocked)).clamp_min(0.0)

            def sample_eval() -> tuple[torch.Tensor, torch.Tensor | None]:
                rw_edge_index_s, rw_edge_weight_s, _, _ = model_eval.rewire(
                    h1_detached,
                    edge_index_shocked,
                    candidate_edge_index=candidate_edge_index,
                    anchor_edge_index=anchor_shocked,
                    sample=True,
                )
                return rw_edge_index_s, rw_edge_weight_s

            lam2_cvar = sensor.estimate_cvar(
                sample_eval,
                samples=int(args.cvar_samples),
                cvar_frac=float(args.cvar_frac),
            )
            vio_lo = (float(args.edge_budget_min_ratio) - ratio).clamp_min(0.0)
            vio_hi = (ratio - float(args.edge_budget_max_ratio)).clamp_min(0.0)
            return (
                float(lam2_cvar.detach().cpu()),
                float(ratio.detach().cpu()),
                float(vio_lo.detach().cpu()),
                float(vio_hi.detach().cpu()),
            )

        # Optional test-time adaptation.
        mean_self, cvar_self, min_self, max_self = eval_cvar_accuracy(
            model, enable_rewire=True, edge_index_eval=edge_index_shocked
        )
        lam2_self, ratio_self, vio_lo_self, vio_hi_self = eval_shock_stats(model, enable_rewire=True)
        if args.shock_adapt:
            if args.shock_reset_duals:
                with torch.no_grad():
                    model.dual_budget_lo.zero_()
                    model.dual_budget_hi.zero_()
                    model.dual_spectral.zero_()
            adapt_topology(int(args.shock_adapt_steps), int(args.shock_primal_steps), float(args.shock_primal_lr))
            mean_self_adapt, cvar_self_adapt, min_self_adapt, max_self_adapt = eval_cvar_accuracy(
                model, enable_rewire=True, edge_index_eval=edge_index_shocked
            )
            lam2_self_adapt, ratio_self_adapt, vio_lo_self_adapt, vio_hi_self_adapt = eval_shock_stats(
                model, enable_rewire=True
            )
        else:
            mean_self_adapt, cvar_self_adapt, min_self_adapt, max_self_adapt = (None, None, None, None)
            lam2_self_adapt, ratio_self_adapt, vio_lo_self_adapt, vio_hi_self_adapt = (None, None, None, None)

        # Baseline: train a separate backbone on the same augmented graph unless disabled.
        if args.no_shock_baseline:
            baseline_model = model
            mean_base, cvar_base, min_base, max_base = eval_cvar_accuracy(
                baseline_model, enable_rewire=False, edge_index_eval=edge_index_shocked
            )
            lam2_base, ratio_base, vio_lo_base, vio_hi_base = eval_shock_stats(baseline_model, enable_rewire=False)
        else:
            baseline_reg = RegularizationConfig()
            baseline_model = SimpleNodeClassifier(
                in_dim=dataset.num_features,
                hidden_dim=int(args.hidden),
                out_dim=dataset.num_classes,
                backbone=args.shock_baseline_backbone,
                dropout=float(args.dropout),
                rewire_temperature=float(args.rewire_temp),
                rewire_hard=bool(args.rewire_hard),
                rewire_symmetric=True,
                reg=baseline_reg,
            ).to(device)
            opt_base = torch.optim.Adam(
                baseline_model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
            )
            baseline_model.train()
            for _ in range(1, int(args.epochs) + 1):
                opt_base.zero_grad(set_to_none=True)
                logits_b, _ = baseline_model(
                    data.x, edge_index_base, enable_rewire=False, rewire_sample=None
                )
                loss_b = F.cross_entropy(logits_b[train_mask], data.y[train_mask])
                loss_b.backward()
                opt_base.step()
            baseline_model.eval()
            mean_base, cvar_base, min_base, max_base = eval_cvar_accuracy(
                baseline_model, enable_rewire=False, edge_index_eval=edge_index_shocked
            )
            lam2_base, ratio_base, vio_lo_base, vio_hi_base = eval_shock_stats(baseline_model, enable_rewire=False)

        # **COMPARATIVE EVALUATION: 3-WAY COMPARISON (Optional)**
        if args.comparative_eval:
            print("\n=== 3-Way Comparative Evaluation ===")

            # 1. GCN Baseline (no rewiring)
            print("(1/3) GCN Baseline on attacked graph...")
            mean_baseline, cvar_baseline, min_baseline, max_baseline = eval_cvar_accuracy(
                baseline_model, enable_rewire=False, edge_index_eval=edge_index_shocked
            )
            lam2_baseline_val, ratio_baseline_val, _, _ = eval_shock_stats(baseline_model, enable_rewire=False)

            # 2. Frozen Framework (rewiring without adaptation)
            print("(2/3) Frozen Self-Healing (no TTA)...")
            mean_frozen, cvar_frozen, min_frozen, max_frozen = eval_cvar_accuracy(
                model, enable_rewire=True, edge_index_eval=edge_index_shocked
            )
            lam2_frozen, ratio_frozen, _, _ = eval_shock_stats(model, enable_rewire=True)

            # 3. Self-Healing Framework (with TTA)
            print("(3/3) Self-Healing with TTA...")
            if args.shock_reset_duals:
                with torch.no_grad():
                    model.dual_budget_lo.zero_()
                    model.dual_budget_hi.zero_()
                    model.dual_spectral.zero_()

            lam2_pre_healing, lam2_post_healing = self_healing_step(
                model,
                sensor,
                data.x,
                edge_index_shocked,
                candidate_edge_index=candidate_edge_index,
                anchor_shocked=anchor_shocked,
                num_nodes=data.num_nodes,
                target_lambda2=float(args.conn_eps),
                budget_min_ratio=float(args.edge_budget_min_ratio),
                budget_max_ratio=float(args.edge_budget_max_ratio),
                budget_beta=float(tta.budget_beta),
                budget_hard=bool(tta.budget_hard),
                budget_dual_scale=float(tta.budget_dual_scale),
                budget_ref_edges=float(edge_index_base.size(1)) if tta.budget_ref_base else None,
                primal_lr=float(tta.primal_lr),
                dual_lr=float(tta.dual_lr),
                num_iters=int(tta.healing_steps),
                cvar_frac=float(tta.cvar_frac),
                cvar_samples=int(tta.cvar_samples),
                kl_beta=float(tta.kl_beta),
                kl_temp=float(tta.kl_temp),
                kl_conf=float(tta.kl_conf),
                device=device,
            )

            mean_healed, cvar_healed, min_healed, max_healed = eval_cvar_accuracy(
                model, enable_rewire=True, edge_index_eval=edge_index_shocked
            )
            lam2_healed, ratio_healed, _, _ = eval_shock_stats(model, enable_rewire=True)

            # Compute SSI Recovery Ratio
            ssi_recovery = lam2_post_healing / max(lam2_pre_healing, 1e-6)

            # Print comparative table
            print("\n" + "=" * 100)
            print(
                f"{'Method':<25} {'Accuracy':<15} {'CVaR Acc':<15} "
                f"{'λ₂ CVaR':<15} {'Edge Ratio':<15} {'SSI Recovery':<15}"
            )
            print("=" * 100)
            print(
                f"{'GCN Baseline':<25} {mean_baseline:.4f}        {cvar_baseline:.4f}        "
                f"{lam2_baseline_val:.4f}        {ratio_baseline_val:.4f}        {'N/A':<15}"
            )
            print(
                f"{'Frozen Self-Healing':<25} {mean_frozen:.4f}        {cvar_frozen:.4f}        "
                f"{lam2_frozen:.4f}        {ratio_frozen:.4f}        {'N/A':<15}"
            )
            print(
                f"{'Self-Healing (TTA)':<25} {mean_healed:.4f}        {cvar_healed:.4f}        "
                f"{lam2_healed:.4f}        {ratio_healed:.4f}        {ssi_recovery:.4f}"
            )
            print("=" * 100)

            # Robustness gap analysis
            acc_gap_baseline = mean_baseline - mean_healed
            acc_gap_frozen = mean_frozen - mean_healed
            lam2_improvement = lam2_post_healing - lam2_pre_healing

            print(f"\nResilience Gap Analysis:")
            print(f"  Baseline → Healed accuracy gap: {acc_gap_baseline:+.4f}")
            print(f"  Frozen → Healed accuracy gap: {acc_gap_frozen:+.4f}")
            print(f"  SSI Recovery Ratio (post/pre): {ssi_recovery:.4f}x")
            print(f"  SSI Improvement (post - pre): {lam2_improvement:+.6f}")
            print("=" * 100)

        msg = (
            "Zero-shot anchor shock (drop "
            f"{float(args.shock_drop_pct):.0%}): "
            f"Self-healing mean={mean_self:.3f} CVaR={cvar_self:.3f} "
            f"(min={min_self:.3f}, max={max_self:.3f}) | "
            f"Baseline({args.shock_baseline_backbone}) mean={mean_base:.3f} CVaR={cvar_base:.3f} "
            f"(min={min_base:.3f}, max={max_base:.3f})"
        )
        msg += (
            f" | lambda2_cvar self={lam2_self:.3f} base={lam2_base:.3f} "
            f"| ratio self={ratio_self:.3f} vio_lo={vio_lo_self:.3f} vio_hi={vio_hi_self:.3f}"
        )
        if mean_self_adapt is not None:
            msg += (
                " | Adapted mean="
                f"{mean_self_adapt:.3f} CVaR={cvar_self_adapt:.3f} "
                f"(min={min_self_adapt:.3f}, max={max_self_adapt:.3f})"
                f" lambda2_cvar={lam2_self_adapt:.3f} ratio={ratio_self_adapt:.3f} "
                f"vio_lo={vio_lo_self_adapt:.3f} vio_hi={vio_hi_self_adapt:.3f}"
            )
            if args.shock_reset_duals:
                msg += " duals_reset"
        print(msg)


if __name__ == "__main__":
    main()
