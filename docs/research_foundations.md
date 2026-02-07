# Research foundations: Spectral Stability for Heterophilic + Noisy Graphs

## Problem framing

In many GNNs, the message passing operator behaves like a **low-pass filter**, which is beneficial under homophily but can destroy predictive **high-frequency** components under heterophily.

Goal: learn a **stochastic, task-conditioned graph** (via differentiable edge gates) that:
1) preserves task-relevant frequency content of node embeddings, and
2) remains globally connected / non-fragmented.

## Core objects

Let $G=(V,E)$ with $|V|=n$. Let $A$ be adjacency (possibly weighted), $D$ degree, and Laplacian $L=D-A$.

A stochastic edge mask $M$ induces a weighted adjacency $A' = A \odot M$.

We learn probabilities $\theta_{ij}\in(0,1)$ and sample $M_{ij}\sim\text{Bernoulli}(\theta_{ij})$.

### Differentiable gates (Concrete / Gumbel-Sigmoid)

For each edge, parameterize log-odds $\ell_{ij}$ and sample a relaxed gate:

$$
\tilde m_{ij} = \sigma\left(\frac{\ell_{ij} + g}{\tau}\right), \quad g = \log u - \log(1-u),\ u\sim\text{Uniform}(0,1)
$$

Straight-through hardening (optional): $m_{ij}=\mathbb{1}[\tilde m_{ij}>0.5]$ with ST gradients.

## Step A: Dirichlet energy (embedding smoothness / frequency proxy)

Dirichlet energy of embeddings $H\in\mathbb{R}^{n\times d}$:

$$
E_{dir}(H) = \sum_{(i,j)\in E} w_{ij}\,\|h_i-h_j\|^2 = 2\,\mathrm{tr}(H^\top L_w H)
$$

Key idea for heterophily: don’t force $E_{dir}$ to be minimal; instead, constrain it **relative** to task loss and/or track energy in specific spectral bands.

Practical proxies:
- Penalize collapse: keep variance of $H$ above a threshold.
- Band-energy ratio: low-vs-high frequency energy using approximate spectral projectors (Chebyshev).

## Step B: Probabilistic pruning/rewiring

Learn a distribution over sparse graphs $p_\theta(G')$ by gating edges.

Common constraints:
- Expected sparsity budget: $\sum_{e\in E} \mathbb{E}[m_e] \approx \rho|E|$
- Per-node degree budget: $\sum_{j} \mathbb{E}[m_{ij}] \le k$

Edge additions must be restricted to a candidate pool (e.g., kNN, 2-hop sampling) to avoid $O(n^2)$.

## Step C: Spectral connectivity guardrail

Fragmentation can be monitored via algebraic connectivity. In practice, tracking the
**normalized Laplacian** is often better behaved under changing edge weights:

$$
L_{sym} = I - D^{-1/2} A D^{-1/2}, \quad \lambda_2(L_{sym}) \in [0,2]
$$

Guardrail penalty:

$$
\mathcal{L}_{conn} = \max(0, \epsilon - \lambda_2(L'))^2
$$

Computational notes:
- Exact $\lambda_2$ requires eigen-solves (costly), but can be done occasionally for small graphs.
- For training-time proxies, use Rayleigh-quotient heuristics or a few Lanczos steps.

## The “Spectral Stability Index” (SSI) white space

The literature has many *methods* (pruning, rewiring, heterophily architectures) but lacks a standard
metric for: “did the learned topology remain structurally healthy during training?”

One SSI direction that matches this repo:
- **Connectivity health**: track $\lambda_2(L_{sym})$ over training (proxy + occasional exact check).
- **Frequency health**: track Dirichlet energy traces per layer.
- **Robustness health**: measure sensitivity of predictions to injected noise edges.

## Dynamic Spectral Guardrails (core white space)

Most prior work uses fixed temperature schedules (or fixed rewiring) regardless of structural health.
The “dynamic guardrail” idea makes the gate temperature a **closed-loop controller** driven by
connectivity health.

Let $\hat\lambda_2^{norm}$ be a fast proxy for $\lambda_2(L_{sym})$.

- If $\hat\lambda_2^{norm}$ is high (graph healthy), set $\tau \to \tau_{min}$ to make gating discrete.
- If $\hat\lambda_2^{norm}$ drops toward a danger threshold, increase $\tau$ to soften gates and let
	gradients explore new structures without committing to hard deletions.

In this repo, this is implemented as a sigmoid controller:

$$
	au = \tau_{min} + (\tau_{max}-\tau_{min})\,\sigma\left(\frac{\lambda^* - \hat\lambda_2^{norm}}{s}\right)
$$

Optionally smooth with EMA over epochs.

### Two practical upgrades (supervisor critique mitigations)

1) **Conservative correction for the proxy**

Even with min-Rayleigh, the proxy is still an *upper bound* and can give false confidence. When we occasionally compute the exact $\lambda_2(L_{sym})$ (SciPy), we update a scalar correction factor $c\in[0,1]$ via EMA:

$$
c \leftarrow (1-\eta)\,c + \eta\,\min\left(1,\frac{\lambda_2^{exact}}{\hat\lambda_2^{proxy}}\right)
$$

and use $c\,\hat\lambda_2^{proxy}$ for guardrails/logging.

2) **Spectral shielding (self-healing adjacency)**

Add a differentiable hinge penalty that backprops into gate parameters when connectivity is low:

$$
\mathcal{L}_{shield} = \left[\max(0,\lambda^* - c\,\hat\lambda_2^{proxy})\right]^2
$$

This is computed on an *expected-gate* rewired snapshot to keep it stable.

## Suggested objective

A simple starting point:

$$
\min_\theta\; \mathbb{E}_{M\sim\theta}\big[\mathcal{L}_{task}(f_{G\odot M}(X),Y)\big]
+ \alpha\,\mathcal{R}_{freq}(H)
+ \beta\,\mathcal{L}_{conn}(L')
$$

where $\mathcal{R}_{freq}$ can be Dirichlet-based, band-energy, or stability-to-perturbation regularizers.

## Ablations (high value)

- Remove gating (baseline GCN)
- Gating only prune (no candidate additions)
- Different gate features: raw $x$ vs hidden $h$ vs both
- Temperature schedules ($\tau$ annealing)
- Connectivity regularizer on/off
- Edge budgets (global vs per-node)

## Metrics to track beyond accuracy

- Energy traces: $E_{dir}(H^{(\ell)})$ per layer
- Estimated connectivity: $\lambda_2$ (proxy + occasional exact)
- Keep-rate and degree distribution after gating
- Robustness: add random/adversarial edges and measure degradation
