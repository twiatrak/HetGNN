from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from hetgnn_spectral_stability.layers import StochasticRewiring
from hetgnn_spectral_stability.config import RegularizationConfig
from hetgnn_spectral_stability.regularizers import dirichlet_energy, estimate_lambda2_norm_min_rayleigh


def _require_pyg() -> None:
    try:
        import torch_geometric  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch Geometric is required for this model. Install torch + torch_geometric first."
        ) from exc


class WeightedSAGEConv(nn.Module):
    """A minimal GraphSAGE-like layer that supports scalar edge weights.

    h_i' = W_self h_i + W_neigh * (sum_j w_ij h_j / sum_j w_ij)
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        row, col = edge_index
        if edge_weight is None:
            edge_weight = torch.ones(row.numel(), device=x.device, dtype=x.dtype)

        msg = x[col] * edge_weight.unsqueeze(-1)
        out = torch.zeros_like(x)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)

        denom = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        denom.scatter_add_(0, row, edge_weight)
        denom = denom.clamp_min(1e-12).unsqueeze(-1)

        neigh = out / denom
        return self.lin_self(x) + self.lin_neigh(neigh)


class SimpleNodeClassifier(nn.Module):
    """A small node classifier baseline integrating stochastic rewiring + regularizers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        backbone: Literal["gcn", "wsage"] = "gcn",
        dropout: float = 0.5,
        rewire_temperature: float = 1.0,
        rewire_hard: bool = False,
        rewire_symmetric: bool = True,
        reg: Optional[RegularizationConfig] = None,
    ):
        super().__init__()
        _require_pyg()
        from torch_geometric.nn import GCNConv

        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.backbone = backbone

        if backbone == "gcn":
            self.conv1 = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
            self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True, normalize=True)
        elif backbone == "wsage":
            self.conv1 = WeightedSAGEConv(hidden_dim, hidden_dim)
            self.conv2 = WeightedSAGEConv(hidden_dim, out_dim)
        else:
            raise ValueError(f"Unknown backbone={backbone!r} (expected 'gcn' or 'wsage')")

        self.rewire = StochasticRewiring(
            node_dim=hidden_dim,
            hidden_dim=max(32, hidden_dim // 2),
            temperature=rewire_temperature,
            hard=rewire_hard,
            dropout=dropout,
            symmetric=rewire_symmetric,
        )

        self.dropout = float(dropout)
        self.reg = reg or RegularizationConfig()
        # Dual variables (Lagrangian multipliers).
        self.register_buffer("dual_budget_lo", torch.zeros((), dtype=torch.float32))
        self.register_buffer("dual_budget_hi", torch.zeros((), dtype=torch.float32))
        self.register_buffer("dual_spectral", torch.zeros((), dtype=torch.float32))
        self.dual_lr = float(self.reg.dual_lr)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        candidate_edge_index: Optional[Tensor] = None,
        anchor_edge_index: Optional[Tensor] = None,
        enable_rewire: bool = True,
        rewire_sample: Optional[bool] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        x = self.lin_in(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # First conv on original topology
        if self.backbone == "gcn":
            h1 = self.conv1(x, edge_index)
        else:
            h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Rewire based on current embeddings (optional warm-up bypass)
        if enable_rewire:
            rw_edge_index, rw_edge_weight, rw_stats, _ = self.rewire(
                h1,
                edge_index,
                candidate_edge_index=candidate_edge_index,
                anchor_edge_index=anchor_edge_index,
                anchor_mode=self.reg.anchor_mode,
                sample=rewire_sample,
            )
        else:
            rw_edge_index, rw_edge_weight = edge_index, None
            rw_stats = None

        # Second conv on rewired weighted topology
        if self.backbone == "gcn":
            logits = self.conv2(h1, rw_edge_index, edge_weight=rw_edge_weight)
        else:
            logits = self.conv2(h1, rw_edge_index, edge_weight=rw_edge_weight)

        regs: Dict[str, Tensor] = {}

        # Always expose the new constrained-risk terms (even if they are 0),
        # so training loops / loggers can rely on their presence.
        regs["edge_budget"] = torch.zeros((), device=logits.device, dtype=logits.dtype)
        regs["anchor_stability"] = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if self.reg.alpha_dirichlet != 0.0:
            dirichlet = dirichlet_energy(h1, rw_edge_index, rw_edge_weight, reduce="mean")
            if getattr(self.rewire, "symmetric", False):
                # Use undirected unique-edge convention.
                dirichlet = 0.5 * dirichlet
            regs["dirichlet"] = dirichlet

        # Primal-dual constraints: edge budget (two-sided) + spectral health.
        if enable_rewire and rw_stats is not None:
            if getattr(self.rewire, "symmetric", False):
                from torch_geometric.utils import coalesce

                num_nodes = h1.size(0)
                row, col = edge_index
                u = torch.minimum(row, col)
                v = torch.maximum(row, col)
                undirected_ei = torch.stack([u, v], dim=0)
                ones = torch.ones(undirected_ei.size(1), device=logits.device, dtype=logits.dtype)
                undirected_ei, _ = coalesce(undirected_ei, ones, num_nodes=num_nodes, reduce="sum")
                e_ref = torch.as_tensor(
                    float(undirected_ei.size(1)),
                    device=logits.device,
                    dtype=logits.dtype,
                ).clamp_min(1.0)
            else:
                e_ref = torch.as_tensor(
                    float(edge_index.size(1)),
                    device=logits.device,
                    dtype=logits.dtype,
                ).clamp_min(1.0)

            expected_e = rw_stats.expected_num_edges.to(device=logits.device, dtype=logits.dtype)
            expected_e_ratio = (expected_e / e_ref).clamp_min(0.0)
            regs["expected_num_edges"] = expected_e
            regs["expected_e_ratio"] = expected_e_ratio

            # Budget constraints (two-sided).
            budget_violation_lo = (float(self.reg.edge_budget_min_ratio) - expected_e_ratio).clamp_min(0.0)
            budget_violation_hi = (expected_e_ratio - float(self.reg.edge_budget_max_ratio)).clamp_min(0.0)
            regs["budget_violation_lo"] = budget_violation_lo
            regs["budget_violation_hi"] = budget_violation_hi

            spectral_violation = None
            if float(self.reg.alpha_connectivity) != 0.0:
                # Spectral constraint using CVaR tail risk (K samples).
                h1_detached = h1.detach()
                lam2_vals = []
                num_samples = max(1, int(self.reg.cvar_samples))
                for _ in range(num_samples):
                    rw_edge_index_s, rw_edge_weight_s, _, _ = self.rewire(
                        h1_detached,
                        edge_index,
                        candidate_edge_index=candidate_edge_index,
                        anchor_edge_index=anchor_edge_index,
                        anchor_mode=self.reg.anchor_mode,
                        sample=True,
                    )
                    lam2_s = estimate_lambda2_norm_min_rayleigh(
                        rw_edge_index_s,
                        rw_edge_weight_s,
                        num_nodes=h1.size(0),
                        num_iters=self.reg.connectivity_iters,
                    )
                    lam2_vals.append(lam2_s)
                lam2_stack = torch.stack(lam2_vals, dim=0)
                worst_k = max(1, int(round(float(self.reg.cvar_frac) * lam2_stack.numel())))
                worst_vals = torch.topk(lam2_stack, k=worst_k, largest=False).values
                lam2_cvar = worst_vals.mean()
                regs["lambda2_cvar"] = lam2_cvar

                spectral_violation = (float(self.reg.connectivity_eps) - lam2_cvar).clamp_min(0.0)
                regs["spectral_violation"] = spectral_violation

            # Dual ascent update (no gradients).
            if self.training:
                with torch.no_grad():
                    self.dual_budget_lo.copy_(
                        (self.dual_budget_lo + self.dual_lr * budget_violation_lo).clamp_min(0.0)
                    )
                    self.dual_budget_hi.copy_(
                        (self.dual_budget_hi + self.dual_lr * budget_violation_hi).clamp_min(0.0)
                    )
                    if spectral_violation is not None:
                        self.dual_spectral.copy_(
                            (self.dual_spectral + self.dual_lr * spectral_violation).clamp_min(0.0)
                        )

            regs["dual_budget_lo"] = self.dual_budget_lo.to(device=logits.device, dtype=logits.dtype)
            regs["dual_budget_hi"] = self.dual_budget_hi.to(device=logits.device, dtype=logits.dtype)
            regs["dual_spectral"] = self.dual_spectral.to(device=logits.device, dtype=logits.dtype)
            regs["edge_budget_dual"] = (
                regs["dual_budget_lo"] * budget_violation_lo
                + 0.5 * budget_violation_lo.pow(2)
                + regs["dual_budget_hi"] * budget_violation_hi
                + 0.5 * budget_violation_hi.pow(2)
            )
            if spectral_violation is not None:
                regs["spectral_dual"] = regs["dual_spectral"] * spectral_violation + 0.5 * spectral_violation.pow(2)

        # Anchor stability penalty — depends on anchor_mode.
        if enable_rewire and rw_stats is not None and float(self.reg.alpha_anchor_stability) != 0.0:
            if self.reg.anchor_mode == "forced":
                # Mode A: coverage check (anchors are hard-forced, metric is informational).
                if rw_stats.anchor_coverage is not None:
                    cov = rw_stats.anchor_coverage.to(device=logits.device, dtype=logits.dtype)
                    regs["anchor_coverage"] = cov
                    regs["anchor_stability"] = (1.0 - cov).clamp_min(0.0).pow(2)
            else:
                # Mode B: soft anchors — penalise low anchor probabilities.
                if rw_stats.anchor_probs is not None and rw_stats.anchor_probs.numel() > 0:
                    p_anchor = rw_stats.anchor_probs.to(device=logits.device, dtype=logits.dtype)
                    regs["anchor_soft_loss"] = ((1.0 - p_anchor) ** 2).mean()
                    regs["anchor_stability"] = regs["anchor_soft_loss"]
                    if rw_stats.anchor_coverage is not None:
                        regs["anchor_coverage"] = rw_stats.anchor_coverage.to(device=logits.device, dtype=logits.dtype)

        # Band-energy regularizer (heterophily-aware frequency preservation).
        # Always compute band metrics for logging; only add loss when weights > 0.
        if enable_rewire and rw_edge_index is not None:
            from hetgnn_spectral_stability.regularizers.spectral import band_energy_proxy

            be = band_energy_proxy(
                h1, rw_edge_index, rw_edge_weight, num_nodes=h1.size(0),
                cheby_order=int(self.reg.band_cheby_order),
            )
            regs["band_high_ratio"] = be

            var = h1.var(dim=0).mean()
            regs["variance"] = var

            if float(self.reg.alpha_band_high) != 0.0:
                regs["band_loss"] = F.relu(float(self.reg.band_target_high_ratio) - be).pow(2)
            if float(self.reg.alpha_variance_floor) != 0.0:
                regs["variance_floor_loss"] = F.relu(float(self.reg.variance_floor) - var).pow(2)

        # For logging
        if rw_stats is None:
            regs["expected_keep_rate"] = torch.tensor(1.0, device=logits.device)
        else:
            regs["expected_keep_rate"] = rw_stats.expected_keep_rate.to(device=logits.device, dtype=logits.dtype)

        return logits, regs
