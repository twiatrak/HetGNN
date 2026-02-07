from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


def dirichlet_energy(
    h: Tensor,
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    reduce: str = "sum",
) -> Tensor:
    """Compute Dirichlet energy for node embeddings.

    Parameters
    ----------
    h : Tensor
        Node embeddings with shape [N, D].
    edge_index : Tensor
        Edge indices with shape [2, E].
    edge_weight : Optional[Tensor]
        Optional edge weights with shape [E]. If None, unit weights are used.
    reduce : str
        Reduction mode, either "sum" or "mean".

    Returns
    -------
    Tensor
        Scalar Dirichlet energy.

    Notes
    -----
    If the graph is represented with both directions (i->j and j->i),
    energy is doubled. Use symmetric conventions accordingly.
    """
    row, col = edge_index

    if edge_weight is None:
        edge_weight = torch.ones(row.numel(), device=h.device, dtype=h.dtype)

    diff = h[row] - h[col]
    sq = (diff * diff).sum(dim=-1)
    e = edge_weight * sq

    if reduce == "sum":
        return e.sum()
    if reduce == "mean":
        return e.mean()

    raise ValueError(f"Unknown reduce={reduce!r} (expected 'sum' or 'mean')")


def estimate_lambda2_rayleigh(
    edge_index: Tensor,
    edge_weight: Optional[Tensor],
    num_nodes: int,
    num_iters: int = 25,
    eps: float = 1e-12,
) -> Tensor:
    """Estimate algebraic connectivity (lambda2) via a Rayleigh-quotient heuristic.

    Parameters
    ----------
    edge_index : Tensor
        Edge indices with shape [2, E].
    edge_weight : Optional[Tensor]
        Optional edge weights with shape [E].
    num_nodes : int
        Number of nodes in the graph.
    num_iters : int
        Iterations of smoothing/projection for the probe vector.
    eps : float
        Numerical stability constant.

    Returns
    -------
    Tensor
        Differentiable scalar estimate of lambda2.

    Notes
    -----
    With a random vector v orthogonal to 1, the Rayleigh quotient
    R(v) = (v^T L v) / (v^T v) is an upper bound on lambda2. This is a
    coarse proxy and can miss fragmentation if used alone.
    """
    device = edge_index.device
    dtype = edge_weight.dtype if edge_weight is not None else torch.float32

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device, dtype=dtype)

    row, col = edge_index

    # Start with random v, project out constant component.
    v = torch.randn(num_nodes, device=device, dtype=dtype)
    v = v - v.mean()

    # Smooth v a bit with repeated application of (I - alpha*L) to move toward low-frequency modes.
    # This is a heuristic meant to push v toward the Fiedler subspace.
    deg = torch.zeros(num_nodes, device=device, dtype=dtype)
    deg.scatter_add_(0, row, edge_weight)
    alpha = 0.5 / (deg.max().clamp_min(eps))

    for _ in range(max(1, int(num_iters))):
        # (L v)_i = sum_j w_ij (v_i - v_j)
        v_row = v[row]
        v_col = v[col]
        msg = edge_weight * (v_row - v_col)

        Lv = torch.zeros(num_nodes, device=device, dtype=dtype)
        Lv.scatter_add_(0, row, msg)

        v = v - alpha * Lv
        v = v - v.mean()
        v = v / (v.norm() + eps)

    # Rayleigh quotient for Laplacian: v^T L v = sum_{(i,j)} w_ij (v_i - v_j)^2
    v_row = v[row]
    v_col = v[col]
    num = (edge_weight * (v_row - v_col).pow(2)).sum()
    den = (v.pow(2)).sum().clamp_min(eps)
    return num / den


def estimate_lambda2_min_rayleigh(
    edge_index: Tensor,
    edge_weight: Optional[Tensor],
    num_nodes: int,
    num_restarts: int = 4,
    num_iters: int = 40,
    softmin_temperature: Optional[float] = 0.1,
    eps: float = 1e-12,
) -> Tensor:
    """Estimate lambda2 by minimizing the Rayleigh quotient over probe vectors.

    Parameters
    ----------
    edge_index : Tensor
        Edge indices with shape [2, E].
    edge_weight : Optional[Tensor]
        Optional edge weights with shape [E].
    num_nodes : int
        Number of nodes in the graph.
    num_restarts : int
        Number of random restarts for the probe vector.
    num_iters : int
        Iterations of gradient descent on the probe vector.
    softmin_temperature : Optional[float]
        If provided, use a log-mean-exp softmin across restarts.
    eps : float
        Numerical stability constant.

    Returns
    -------
    Tensor
        Differentiable scalar estimate of lambda2.

    Notes
    -----
    The estimate remains an upper bound on true lambda2, but is typically
    tighter than a single random Rayleigh probe.
    """
    device = edge_index.device
    dtype = edge_weight.dtype if edge_weight is not None else torch.float32

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device, dtype=dtype)

    row, col = edge_index

    # Step size based on max degree (stable for small graphs).
    deg = torch.zeros(num_nodes, device=device, dtype=dtype)
    deg.scatter_add_(0, row, edge_weight.detach())
    lr = (0.5 / deg.max().clamp_min(eps)).item()

    def rayleigh(v: Tensor) -> Tensor:
        v_row = v[row]
        v_col = v[col]
        num = 0.5 * (edge_weight * (v_row - v_col).pow(2)).sum()
        den = (v.pow(2)).sum().clamp_min(eps)
        return num / den

    best_vals = []
    for _ in range(max(1, int(num_restarts))):
        # v is treated as a probe vector; we update it without tracking gradients.
        v = torch.randn(num_nodes, device=device, dtype=dtype)
        v = v - v.mean()
        v = v / (v.norm() + eps)

        with torch.no_grad():
            for _ in range(max(1, int(num_iters))):
                v_row = v[row]
                v_col = v[col]
                msg = edge_weight.detach() * (v_row - v_col)

                Lv = torch.zeros(num_nodes, device=device, dtype=dtype)
                Lv.scatter_add_(0, row, msg)

                num = (edge_weight.detach() * (v_row - v_col).pow(2)).sum()
                den = (v.pow(2)).sum().clamp_min(eps)

                # grad of Rayleigh quotient wrt v: (2*Lv*den - 2*num*v)/den^2
                grad = (2.0 * Lv * den - 2.0 * num * v) / (den * den)

                v = v - lr * grad
                v = v - v.mean()
                v = v / (v.norm() + eps)

        best_vals.append(rayleigh(v))

    vals = torch.stack(best_vals, dim=0)
    if softmin_temperature is None:
        return vals.min(dim=0).values

    t = float(softmin_temperature)
    z = -vals / max(t, eps)
    k = vals.numel()
    out = (-t) * (torch.logsumexp(z, dim=0) - torch.log(torch.as_tensor(k, device=vals.device, dtype=vals.dtype)))
    return out.clamp_min(0.0)


def estimate_lambda2_norm_min_rayleigh(
    edge_index: Tensor,
    edge_weight: Optional[Tensor],
    num_nodes: int,
    num_restarts: int = 4,
    num_iters: int = 80,
    softmin_temperature: Optional[float] = 0.1,
    eps: float = 1e-12,
) -> Tensor:
    """Estimate lambda2 of the normalized Laplacian using Rayleigh minimization.

    Parameters
    ----------
    edge_index : Tensor
        Edge indices with shape [2, E].
    edge_weight : Optional[Tensor]
        Optional edge weights with shape [E].
    num_nodes : int
        Number of nodes in the graph.
    num_restarts : int
        Number of random restarts for the probe vector.
    num_iters : int
        Iterations of gradient descent on the probe vector.
    softmin_temperature : Optional[float]
        If provided, use a log-mean-exp softmin across restarts.
    eps : float
        Numerical stability constant.

    Returns
    -------
    Tensor
        Differentiable scalar estimate of normalized lambda2.

    Notes
    -----
    The normalized Laplacian is L_sym = I - D^{-1/2} A D^{-1/2}. Eigenvalues
    lie in [0, 2], which stabilizes thresholds across graphs.
    """
    device = edge_index.device
    dtype = edge_weight.dtype if edge_weight is not None else torch.float32

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device, dtype=dtype)

    row, col = edge_index

    deg = torch.zeros(num_nodes, device=device, dtype=dtype)
    # Keep degree differentiable w.r.t. edge weights (no detach here).
    deg.scatter_add_(0, row, edge_weight)
    sqrt_deg = deg.clamp_min(eps).sqrt()
    inv_sqrt_deg = sqrt_deg.reciprocal()

    # Step size heuristic.
    lr = 0.2

    def project(v: Tensor) -> Tensor:
        # Project out the nullspace direction of L_sym: sqrt(deg)
        denom = (sqrt_deg.pow(2)).sum().clamp_min(eps)
        coef = (v * sqrt_deg).sum() / denom
        v = v - coef * sqrt_deg
        return v

    def rayleigh(v: Tensor) -> Tensor:
        u = v * inv_sqrt_deg
        u_row = u[row]
        u_col = u[col]
        num = 0.5 * (edge_weight * (u_row - u_col).pow(2)).sum()
        den = (v.pow(2)).sum().clamp_min(eps)
        return num / den

    best_vals = []
    for _ in range(max(1, int(num_restarts))):
        v = torch.randn(num_nodes, device=device, dtype=dtype)
        v = project(v)
        v = v / (v.norm() + eps)

        with torch.no_grad():
            for _ in range(max(1, int(num_iters))):
                # u = D^{-1/2} v
                u = v * inv_sqrt_deg

                u_row = u[row]
                u_col = u[col]
                msg = edge_weight.detach() * (u_row - u_col)

                # We need gradient wrt v; using chain rule with u = v * inv_sqrt_deg.
                Lu = torch.zeros(num_nodes, device=device, dtype=dtype)
                Lu.scatter_add_(0, row, msg)

                # num = sum w (u_i - u_j)^2, so grad wrt u is 2 * L_u
                grad_u = 2.0 * Lu
                grad_v_num = grad_u * inv_sqrt_deg

                num = (edge_weight.detach() * (u_row - u_col).pow(2)).sum()
                den = (v.pow(2)).sum().clamp_min(eps)

                grad_den = 2.0 * v
                grad = (grad_v_num * den - num * grad_den) / (den * den)

                v = v - lr * grad
                v = project(v)
                v = v / (v.norm() + eps)

        best_vals.append(rayleigh(v))

    vals = torch.stack(best_vals, dim=0)
    if softmin_temperature is None:
        return vals.min(dim=0).values

    t = float(softmin_temperature)
    z = -vals / max(t, eps)
    k = vals.numel()
    out = (-t) * (torch.logsumexp(z, dim=0) - torch.log(torch.as_tensor(k, device=vals.device, dtype=vals.dtype)))
    return out.clamp_min(0.0)


def lambda2_scipy_normalized(
    edge_index: Tensor,
    edge_weight: Optional[Tensor],
    num_nodes: int,
) -> float:
    """Compute normalized lambda2 via SciPy sparse eigensolver.

    Parameters
    ----------
    edge_index : Tensor
        Edge indices with shape [2, E].
    edge_weight : Optional[Tensor]
        Optional edge weights with shape [E].
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    float
        Exact lambda2 of the normalized Laplacian.

    Notes
    -----
    Intended for debugging or validation on small graphs.
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32)

    row, col = edge_index.detach().cpu().numpy()
    w = edge_weight.detach().cpu().numpy().astype(np.float64)

    A = sp.coo_matrix((w, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    d = np.asarray(A.sum(axis=1)).reshape(-1)
    d = np.maximum(d, 1e-12)
    inv_sqrt_d = 1.0 / np.sqrt(d)
    D_inv_sqrt = sp.diags(inv_sqrt_d)
    Lsym = sp.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

    vals = spla.eigsh(Lsym, k=2, which="SM", return_eigenvectors=False, tol=1e-3)
    vals = np.sort(np.real(vals))
    return float(vals[1])


def lambda2_scipy(
    edge_index: Tensor,
    edge_weight: Optional[Tensor],
    num_nodes: int,
) -> float:
    """Compute unnormalized lambda2 via SciPy sparse eigensolver.

    Parameters
    ----------
    edge_index : Tensor
        Edge indices with shape [2, E].
    edge_weight : Optional[Tensor]
        Optional edge weights with shape [E].
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    float
        Exact lambda2 of the unnormalized Laplacian.

    Notes
    -----
    Intended for debugging or validation on small graphs.
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32)

    row, col = edge_index.detach().cpu().numpy()
    w = edge_weight.detach().cpu().numpy().astype(np.float64)

    A = sp.coo_matrix((w, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    d = np.asarray(A.sum(axis=1)).reshape(-1)
    L = sp.diags(d) - A

    # Smallest eigenvalues; skip the first (near zero) to get lambda2
    vals = spla.eigsh(L, k=2, which="SM", return_eigenvectors=False, tol=1e-3)
    vals = np.sort(np.real(vals))
    return float(vals[1])


class SSISensor:
    """Stable helper for estimating normalized lambda2 and its tail risk.

    Notes
    -----
    This centralizes estimation, correction, and CVaR aggregation to avoid
    duplicated logic in training and evaluation scripts.
    """

    def __init__(
        self,
        num_nodes: int,
        num_iters: int = 25,
        num_restarts: int = 4,
        cvar_samples: int = 4,
        cvar_frac: float = 0.5,
        softmin_temperature: Optional[float] = 0.1,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.num_iters = int(num_iters)
        self.num_restarts = int(num_restarts)
        self.cvar_samples = int(cvar_samples)
        self.cvar_frac = float(cvar_frac)
        self.softmin_temperature = softmin_temperature
        self.corr_scale = 1.0

    @staticmethod
    def cvar(values: Tensor, frac: float) -> Tensor:
        """Compute CVaR as the mean of the worst fraction.

        Parameters
        ----------
        values : Tensor
            1D tensor of samples.
        frac : float
            Fraction of worst samples to average.

        Returns
        -------
        Tensor
            CVaR scalar.
        """
        if values.numel() == 0:
            return torch.tensor(float("nan"), device=values.device, dtype=values.dtype)
        k = max(1, int(round(float(frac) * values.numel())))
        worst = torch.topk(values, k=k, largest=False).values
        return worst.mean()

    def estimate(self, edge_index: Tensor, edge_weight: Optional[Tensor]) -> Tensor:
        """Estimate normalized lambda2 for a single graph.

        Parameters
        ----------
        edge_index : Tensor
            Edge indices with shape [2, E].
        edge_weight : Optional[Tensor]
            Optional edge weights with shape [E].

        Returns
        -------
        Tensor
            Differentiable scalar estimate of normalized lambda2.
        """
        return estimate_lambda2_norm_min_rayleigh(
            edge_index,
            edge_weight,
            num_nodes=self.num_nodes,
            num_iters=self.num_iters,
            num_restarts=self.num_restarts,
            softmin_temperature=self.softmin_temperature,
        )

    def estimate_corrected(self, edge_index: Tensor, edge_weight: Optional[Tensor]) -> Tensor:
        """Estimate normalized lambda2 with correction scaling.

        Parameters
        ----------
        edge_index : Tensor
            Edge indices with shape [2, E].
        edge_weight : Optional[Tensor]
            Optional edge weights with shape [E].

        Returns
        -------
        Tensor
            Corrected lambda2 estimate.
        """
        return self.estimate(edge_index, edge_weight) * float(self.corr_scale)

    def estimate_cvar(
        self,
        sample_fn: Callable[[], Tuple[Tensor, Optional[Tensor]]],
        *,
        samples: Optional[int] = None,
        cvar_frac: Optional[float] = None,
        apply_correction: bool = True,
    ) -> Tensor:
        """Estimate CVaR of normalized lambda2 over sampled graphs.

        Parameters
        ----------
        sample_fn : Callable
            Function returning (edge_index, edge_weight) samples.
        samples : Optional[int]
            Number of samples to draw.
        cvar_frac : Optional[float]
            Tail fraction for CVaR.
        apply_correction : bool
            If True, apply correction scaling to each sample.

        Returns
        -------
        Tensor
            CVaR of normalized lambda2.
        """
        num_samples = max(1, int(samples if samples is not None else self.cvar_samples))
        frac = float(cvar_frac if cvar_frac is not None else self.cvar_frac)
        lam2_vals = []
        for _ in range(num_samples):
            edge_index, edge_weight = sample_fn()
            if apply_correction:
                lam2_vals.append(self.estimate_corrected(edge_index, edge_weight))
            else:
                lam2_vals.append(self.estimate(edge_index, edge_weight))
        lam2_stack = torch.stack(lam2_vals, dim=0)
        return self.cvar(lam2_stack, frac)

    def calibrate_with_exact(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        *,
        ema: float = 0.2,
    ) -> Tuple[float, float, float]:
        """Calibrate correction scale using an exact SciPy lambda2 call.

        Parameters
        ----------
        edge_index : Tensor
            Edge indices with shape [2, E].
        edge_weight : Optional[Tensor]
            Optional edge weights with shape [E].
        ema : float
            Exponential moving average factor for the correction scale.

        Returns
        -------
        Tuple[float, float, float]
            (estimate, exact, corr_scale)
        """
        est = float(self.estimate(edge_index, edge_weight).detach().cpu())
        actual = float(lambda2_scipy_normalized(edge_index, edge_weight, num_nodes=self.num_nodes))
        raw = actual / max(est, 1e-12)
        raw = float(max(0.0, min(1.0, raw)))
        self.corr_scale = (1.0 - float(ema)) * float(self.corr_scale) + float(ema) * raw
        return est, actual, float(self.corr_scale)
