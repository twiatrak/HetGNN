# Stochastic Spectral Stability for Heterophilic GNNs

Learning task-optimized graph topologies with spectral stability guarantees under structural perturbation.

## Overview

Standard GNNs are fragile to edge deletions — removing a small fraction of critical edges can fragment the graph and collapse predictive accuracy. This framework addresses the problem by jointly learning a stochastic graph topology and enforcing algebraic connectivity constraints via primal-dual optimization.

**Key components:**

- **Stochastic rewiring** — Gumbel-Sigmoid edge gates that learn a task-conditioned topology over a candidate edge pool (original edges + KNN additions).
- **Primal-dual constraint enforcement** — Lagrangian multipliers enforce a lower bound on $\lambda_2$ (algebraic connectivity) and edge-budget guardrails during training.
- **Test-time adaptation (TTA)** — At inference, a dual-ascent loop rewires the frozen model's topology to recover from structural shocks without retraining.

## Installation

```bash
pip install -e .
# PyTorch and PyG must be installed separately (platform/CUDA-specific):
pip install torch torch-geometric
```

Requires Python $\geq$ 3.10.

## Usage

Train on Chameleon (heterophilic node classification) with spectral constraints:

```bash
python scripts/run_nodecls_wikipedia.py \
  --dataset chameleon --epochs 100 --warmup-epochs 20 \
  --add-knn 10 --anchor-knn 10 \
  --alpha-conn 1.0 --conn-eps 0.2 \
  --edge-budget-min-ratio 0.6 --edge-budget-max-ratio 1.4 \
  --dual-lr 0.02 --cvar-samples 10 --cvar-frac 0.5
```

Evaluate with targeted spectral attack and three-way comparison (baseline / frozen / TTA-healed):

```bash
python scripts/run_nodecls_wikipedia.py \
  --dataset chameleon --epochs 100 --warmup-epochs 20 \
  --add-knn 10 --anchor-knn 10 \
  --alpha-conn 1.0 --conn-eps 0.2 \
  --edge-budget-min-ratio 0.6 --edge-budget-max-ratio 1.4 \
  --dual-lr 0.02 --cvar-samples 10 --cvar-frac 0.5 \
  --zero-shot-shock --targeted-spectral-attack --attack-pct 0.15 \
  --shock-samples 200 --shock-cvar-frac 0.2 --shock-reset-duals \
  --tta-healing-steps 20 --tta-dual-lr 0.01 \
  --comparative-eval \
  --log-csv data/metrics/results.csv
```

## Project Structure

```
src/hetgnn_spectral_stability/
├── config.py                        # Centralized hyperparameter configuration
├── layers/
│   └── stochastic_rewire.py         # Gumbel-Sigmoid edge gates with temperature control
├── models/
│   └── simple_model.py              # Primal-dual GNN with augmented Lagrangian
└── regularizers/
    └── spectral.py                  # Rayleigh-quotient estimators for normalized λ₂

scripts/
├── run_nodecls_wikipedia.py         # Training + evaluation (attack, TTA, 3-way comparison)
└── validation_checks.py            # Spectral invariant and actuator sanity checks

docs/
└── research_foundations.md          # Mathematical background and derivations
```

## Method

The training objective combines task loss with spectral and budget constraints enforced via dual variables:

$$\min_\theta \; \mathbb{E}_{M \sim \theta}\!\left[\mathcal{L}_{\text{task}}\right] + \mu_{\lambda_2} \max(0,\; \epsilon - \hat\lambda_2)^2 + \mu_{\text{budget}} \cdot g(\text{edge ratio})$$

where $\hat\lambda_2$ is estimated via min-Rayleigh-quotient on the stochastic graph snapshot and $\mu$ values are updated by dual ascent.

At test time, the TTA loop freezes model weights and performs dual ascent over the topology gates to restore $\lambda_2$ after edge deletion, producing a **SSI Recovery Ratio** $= \lambda_2^{\text{post}} / \lambda_2^{\text{pre}}$ as the primary resilience metric.

## License

MIT
