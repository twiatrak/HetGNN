from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _require_pyg() -> None:
    """Ensure PyTorch Geometric is available.

    Returns
    -------
    None

    Notes
    -----
    Raises a RuntimeError if torch_geometric is not installed.
    """
    try:
        import torch_geometric  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch Geometric is required for this module. "
            "Install torch + torch_geometric first. See README.md."
        ) from exc


def concrete_bernoulli_sample(
    logits: Tensor,
    temperature: float,
    hard: bool,
    training: bool,
    eps: float = 1e-10,
) -> Tensor:
    """Sample a relaxed Bernoulli gate using the Concrete distribution.

    Parameters
    ----------
    logits : Tensor
        Log-odds with shape [E].
    temperature : float
        Temperature for the Concrete distribution.
    hard : bool
        If True, use straight-through hard thresholding.
    training : bool
        If False, return the deterministic sigmoid.
    eps : float
        Numerical stability constant.

    Returns
    -------
    Tensor
        Gate values in [0, 1] with shape [E].
    """
    if not training:
        return torch.sigmoid(logits / max(temperature, eps))

    # Logistic noise: log(u) - log(1-u)
    u = torch.rand_like(logits).clamp_(eps, 1 - eps)
    g = torch.log(u) - torch.log1p(-u)
    y = torch.sigmoid((logits + g) / max(temperature, eps))

    if not hard:
        return y

    y_hard = (y >= 0.5).to(y.dtype)
    return (y_hard - y).detach() + y


@dataclass
class RewireStats:
    # Keep these as tensors so downstream regularizers can stay differentiable.
    expected_keep_rate: Tensor
    expected_num_edges: Tensor
    # Best-effort signal that anchors made it into the gated pool.
    # When anchors are explicitly injected into the pool, this will be 1.0.
    anchor_coverage: Optional[Tensor] = None


class EdgeGateMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_features: Tensor) -> Tensor:
        """Compute per-edge logits.

        Parameters
        ----------
        edge_features : Tensor
            Edge features with shape [E, F].

        Returns
        -------
        Tensor
            Logits with shape [E].
        """
        return self.net(edge_features).squeeze(-1)


class StochasticRewiring(nn.Module):
    """Differentiable stochastic edge gating with optional candidate-edge addition.

    Notes
    -----
    Works best when gates are computed from current embeddings. For scalability,
    candidate edges should be restricted (kNN, 2-hop sampling, etc.).
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        hard: bool = False,
        dropout: float = 0.0,
        symmetric: bool = False,
    ):
        super().__init__()
        _require_pyg()

        # Edge features: [h_i, h_j, |h_i-h_j|, h_i*h_j]
        self.edge_feat_dim = 4 * node_dim
        self.gate_mlp = EdgeGateMLP(self.edge_feat_dim, hidden_dim, dropout=dropout)

        self.temperature = float(temperature)
        self.hard = bool(hard)
        self.symmetric = bool(symmetric)

    def _edge_features(self, h: Tensor, edge_index: Tensor) -> Tensor:
        """Build per-edge feature vectors from node embeddings.

        Parameters
        ----------
        h : Tensor
            Node embeddings with shape [N, D].
        edge_index : Tensor
            Edge indices with shape [2, E].

        Returns
        -------
        Tensor
            Edge features with shape [E, 4 * D].
        """
        row, col = edge_index
        hi = h[row]
        hj = h[col]
        return torch.cat([hi, hj, (hi - hj).abs(), hi * hj], dim=-1)

    def _symmetrize_logits(self, edge_index: Tensor, logits: Tensor, num_nodes: int) -> Tensor:
        """Symmetrize logits for undirected graphs.

        Parameters
        ----------
        edge_index : Tensor
            Edge indices with shape [2, E].
        logits : Tensor
            Edge logits with shape [E].
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        Tensor
            Symmetrized logits with shape [E].

        Notes
        -----
        For each directed edge (i->j), the logits are averaged with (j->i)
        when both exist.
        """
        if not self.symmetric:
            return logits

        # Map edges to unique ids
        row, col = edge_index
        key = row * num_nodes + col
        rev_key = col * num_nodes + row

        # Build a map from key -> position
        # (Using sort + searchsorted to stay torch-only.)
        key_sorted, perm = torch.sort(key)
        logits_sorted = logits[perm]

        rev_pos = torch.searchsorted(key_sorted, rev_key)
        rev_pos = torch.clamp(rev_pos, 0, key_sorted.numel() - 1)
        has_rev = key_sorted[rev_pos] == rev_key

        rev_logits = torch.zeros_like(logits)
        rev_logits[has_rev] = logits_sorted[rev_pos[has_rev]]

        return 0.5 * (logits + rev_logits)

    def _canonicalize_undirected_pool(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        *,
        num_nodes: int,
        reduce: str = "mean",
    ) -> Tuple[Tensor, Tensor]:
        """Canonicalize edges into an undirected pool and coalesce duplicates.

        Parameters
        ----------
        edge_index : Tensor
            Edge indices with shape [2, E_pool].
        edge_weight : Tensor
            Edge weights with shape [E_pool].
        num_nodes : int
            Number of nodes in the graph.
        reduce : str
            Reduction mode for duplicate edges.

        Returns
        -------
        Tuple[Tensor, Tensor]
            (pool_edge_index, pool_edge_weight) after coalescing.

        Notes
        -----
        This ensures one gate per undirected pair when symmetric gating is used.
        """
        from torch_geometric.utils import coalesce

        row, col = edge_index
        u = torch.minimum(row, col)
        v = torch.maximum(row, col)
        pool_edge_index = torch.stack([u, v], dim=0)
        pool_edge_index, pool_edge_weight = coalesce(
            pool_edge_index,
            edge_weight,
            num_nodes=num_nodes,
            reduce=reduce,
        )
        return pool_edge_index, pool_edge_weight

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        candidate_edge_index: Optional[Tensor] = None,
        candidate_edge_weight: Optional[Tensor] = None,
        anchor_edge_index: Optional[Tensor] = None,
        sample: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor, RewireStats, Tensor]:
        """Rewire edges and return weighted graph plus statistics.

        Parameters
        ----------
        h : Tensor
            Node embeddings with shape [N, D].
        edge_index : Tensor
            Edge indices with shape [2, E].
        edge_weight : Optional[Tensor]
            Edge weights with shape [E].
        candidate_edge_index : Optional[Tensor]
            Candidate edge indices with shape [2, E_c].
        candidate_edge_weight : Optional[Tensor]
            Candidate edge weights with shape [E_c].
        anchor_edge_index : Optional[Tensor]
            Anchor edge indices with shape [2, E_a].
        sample : Optional[bool]
            If True, sample stochastic gates; if False, use expected gates.

        Returns
        -------
        Tuple[Tensor, Tensor, RewireStats, Tensor]
            new_edge_index: Tensor [2, E']
            new_edge_weight: Tensor [E']
            stats: RewireStats with expected rates
            edge_probs: per-edge expected keep probabilities
        """
        from torch_geometric.utils import coalesce
        from torch_geometric.utils import to_undirected

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=h.dtype)

        num_nodes = h.size(0)

        do_sample = self.training if sample is None else bool(sample)

        # Symmetry reform (2026): build a symmetric candidate pool *before* gating.
        # This ensures budget and probabilities are consistent (one gate per undirected edge).
        if self.symmetric:
            pool_edge_index = edge_index
            pool_edge_weight = edge_weight

            if candidate_edge_index is not None:
                if candidate_edge_weight is None:
                    candidate_edge_weight = torch.ones(
                        candidate_edge_index.size(1),
                        device=candidate_edge_index.device,
                        dtype=h.dtype,
                    )
                pool_edge_index = torch.cat([pool_edge_index, candidate_edge_index], dim=1)
                pool_edge_weight = torch.cat([pool_edge_weight, candidate_edge_weight], dim=0)

            # Canonical undirected pool (u<=v) and aggregate any duplicates.
            pool_edge_index, pool_edge_weight = self._canonicalize_undirected_pool(
                pool_edge_index, pool_edge_weight, num_nodes=num_nodes, reduce="mean"
            )

            feats = self._edge_features(h, pool_edge_index)
            logits = self.gate_mlp(feats)
            temp = max(float(self.temperature), 1e-12)
            probs = torch.sigmoid(logits / temp)

            gates = concrete_bernoulli_sample(
                logits=logits,
                temperature=self.temperature,
                hard=self.hard,
                training=do_sample,
            )

            anchor_coverage: Optional[Tensor] = None
            if anchor_edge_index is not None:
                # Build an indicator over the canonical pool and force gates=1 for anchors.
                row, col = pool_edge_index
                pool_key = row * num_nodes + col

                a_row, a_col = anchor_edge_index
                a_u = torch.minimum(a_row, a_col)
                a_v = torch.maximum(a_row, a_col)
                a_key = a_u * num_nodes + a_v

                pool_key_sorted, perm = torch.sort(pool_key)
                pos = torch.searchsorted(pool_key_sorted, a_key)
                pos = torch.clamp(pos, 0, pool_key_sorted.numel() - 1)
                hit = pool_key_sorted[pos] == a_key

                anchor_mask = torch.zeros_like(pool_key, dtype=torch.bool)
                anchor_mask[perm[pos[hit]]] = True

                gates = torch.where(anchor_mask, torch.ones_like(gates), gates)
                anchor_coverage = hit.to(h.dtype).mean() if a_key.numel() > 0 else torch.ones((), device=h.device, dtype=h.dtype)

            w_pool = pool_edge_weight * gates

            # Expand back to a directed representation for message passing.
            edge_index_out, w_out = to_undirected(
                pool_edge_index,
                edge_attr=w_pool,
                num_nodes=num_nodes,
                reduce="mean",
            )
            edge_index_out, w_out = coalesce(edge_index_out, w_out, num_nodes=num_nodes)

            probs_out_ei, probs_out = to_undirected(
                pool_edge_index,
                edge_attr=probs,
                num_nodes=num_nodes,
                reduce="mean",
            )
            probs_out_ei, probs_out = coalesce(probs_out_ei, probs_out, num_nodes=num_nodes)

            # Ensure alignment between returned edge_index and edge_probs.
            # (They should match; coalesce ordering is deterministic given the same inputs.)
            if probs_out_ei.size(1) != edge_index_out.size(1) or not torch.equal(probs_out_ei, edge_index_out):
                # Fall back to a safe (but less informative) proxy.
                probs_out = (w_out.detach().abs() / (w_out.detach().abs().max() + 1e-12)).clamp(0, 1)

            stats = RewireStats(
                expected_keep_rate=probs.mean(),
                expected_num_edges=probs.sum(),
                anchor_coverage=anchor_coverage,
            )

            return edge_index_out, w_out, stats, probs_out

        # Existing edges
        feats = self._edge_features(h, edge_index)
        logits = self.gate_mlp(feats)
        logits = self._symmetrize_logits(edge_index, logits, num_nodes=num_nodes)
        temp = max(float(self.temperature), 1e-12)
        probs = torch.sigmoid(logits / temp)
        gates = concrete_bernoulli_sample(
            logits=logits,
            temperature=self.temperature,
            hard=self.hard,
            training=do_sample,
        )
        w = edge_weight * gates

        # Candidate edges (optional)
        if candidate_edge_index is not None:
            if candidate_edge_weight is None:
                candidate_edge_weight = torch.ones(
                    candidate_edge_index.size(1),
                    device=candidate_edge_index.device,
                    dtype=h.dtype,
                )

            c_feats = self._edge_features(h, candidate_edge_index)
            c_logits = self.gate_mlp(c_feats)
            c_logits = self._symmetrize_logits(candidate_edge_index, c_logits, num_nodes=num_nodes)
            c_probs = torch.sigmoid(c_logits / temp)
            c_gates = concrete_bernoulli_sample(
                logits=c_logits,
                temperature=self.temperature,
                hard=self.hard,
                training=do_sample,
            )
            c_w = candidate_edge_weight * c_gates

            edge_index = torch.cat([edge_index, candidate_edge_index], dim=1)
            w = torch.cat([w, c_w], dim=0)
            probs = torch.cat([probs, c_probs], dim=0)

        edge_index, w = coalesce(edge_index, w, num_nodes=num_nodes)

        # If we intend an undirected graph, enforce symmetric weights even when
        # candidate edges are only proposed in one direction.
        if self.symmetric:
            edge_index, w = to_undirected(edge_index, edge_attr=w, num_nodes=num_nodes, reduce="mean")
            edge_index, w = coalesce(edge_index, w, num_nodes=num_nodes)

        # Note: coalesce sums duplicates; probs are no longer aligned 1:1 with edge_index.
        stats = RewireStats(
            expected_keep_rate=probs.mean(),
            expected_num_edges=probs.sum(),
            anchor_coverage=None,
        )

        # After coalesce we can't track per-edge probs exactly without extra bookkeeping.
        # Return a placeholder that matches w (best-effort: treat prob proportional to normalized weight).
        denom = (w.detach().abs().max() + 1e-12)
        edge_probs_out = (w.detach().abs() / denom).clamp(0, 1)

        return edge_index, w, stats, edge_probs_out
