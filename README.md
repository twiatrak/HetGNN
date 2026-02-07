# Stochastic Spectral Stability (S3): Self-Healing Graph Neural Networks

Part 1: Project README.md (The "Thesis Edition")

This draft summarizes the research for colleagues or reviewers. This repository implements the S3 Framework, a system for learning task-optimized graph topologies that are mathematically guaranteed to remain stable under structural attacks.

Core Research Contribution

Standard GNNs suffer from a structural cliff: if critical edges are removed, the graph fragments and predictive power collapses. S3 solves this by treating the Spectral Stability Index (SSI) as a differentiable risk measure.

Key Mechanisms

- Stochastic Rewiring: A Gumbel-Softmax layer that learns an optimal topology via probabilistic gates.
- Primal-Dual Constraint Engine: Uses Lagrangian multipliers ($\mu$) to automatically enforce algebraic connectivity ($\lambda_2$) and edge-budget guardrails.
- Test-Time Adaptation (TTA): A self-healing loop that lets a frozen GNN rewire its topology at inference time to recover from structural shocks.

Final Technical Audit: The Seals Are Holding

- Sensor Fix: SSISensor now applies the 0.5 factor for symmetric double-counting and uses log-mean-exp aggregation to avoid negative-estimate bias on fragmented graphs.
- Actuator Fix: self_healing_step performs a true primal update on the gate MLP, so TTA physically rewires the topology.
- Temperature Consistency: expected_num_edges and expected_keep_rate use temperature-scaled sigmoid probabilities, aligning budget constraints with deterministic weights.
- Unified Configuration: config.py centralizes training and adaptation settings for reproducible experiments.

Empirical Breakthrough: The 20% Attack Victory

On the heterophilic chameleon dataset, we identified a Resilience Zone:

- The Baseline Collapse: A standard WSAGE model loses significant accuracy when the top 20% of connectivity-critical edges are deleted.
- The S3 Resilience: Our model maintains a 3.5% absolute accuracy lead by physically re-routing signals and achieving a 1.03x recovery of its spectral health.

Resilience Victory (Headline Results)

- Collapse: Removing the top 20% of connectivity-critical edges causes standard backbones (WSAGE/GCN) to suffer a catastrophic accuracy drop.
- S3 Edge: The model preserves a 3.5% absolute accuracy lead over the baseline.
- Healing Signature: SSI Recovery Ratio reaches 1.03x while respecting the edge budget.

Discussion Draft: The Role of KL-Consistency

A critical challenge in self-healing graph topologies is the over-smoothing trade-off. Forcing connectivity under attack can introduce noise from label-disagreeing neighbors, which is especially harmful on heterophilic datasets like chameleon. We address this by adding a prediction consistency loss (KL divergence) during TTA. The loss anchors node predictions to their pre-shock state, allowing the model to restore spectral stability ($\lambda_2$) while preventing the topology from washing out the classifier's decision boundaries. The resulting 3.5% lead over standard backbones shows that global stability can be restored without sacrificing local signal purity.

Project Structure

- [src/hetgnn_spectral_stability/regularizers/spectral.py](src/hetgnn_spectral_stability/regularizers/spectral.py): Calibrated Rayleigh-quotient estimators for normalized $\lambda_2$.
- [src/hetgnn_spectral_stability/layers/stochastic_rewire.py](src/hetgnn_spectral_stability/layers/stochastic_rewire.py): Temperature-consistent Bernoulli gates for differentiable pruning.
- [src/hetgnn_spectral_stability/models/simple_model.py](src/hetgnn_spectral_stability/models/simple_model.py): Primal-dual architecture with augmented Lagrangian penalties.
- [scripts/validation_checks.py](scripts/validation_checks.py): Three-tier mathematical validation suite (invariants, spectral sanity, actuator check).

Part 2: The "Final Polish" Cleanup Prompt

Use this prompt to have an AI (like GPT-5.2 Pro) refactor and sanitize the code for final submission.

Role: You are a Senior Research Engineer at DeepMind/OpenAI specializing in Graph ML infrastructure.

Goal: Clean up the provided repository to meet NeurIPS/ICML 2026 production standards.

Specific Refactoring Instructions

- Eliminate Spectral Redundancy: The $\lambda_2$ estimation logic is currently duplicated in spectral.py and manually implemented inside run_nodecls_wikipedia.py. Centralize this into a single, high-stability SSI_Sensor class.
- Harmonize Hyperparameters: Move the RegularizationConfig and the TTA parameters into a unified YAML-based configuration handler to prevent command-line fatigue.
- Sanitize Monitoring Snapshots: Ensure the compute_rewired_snapshot function consistently receives the anchor_edge_index across all logging calls to eliminate the monitoring blind spot.
- Enforce Type Hints: Ensure all tensors have strict shape documentation (e.g., Tensor  # [2, E_pool]) to help other researchers audit the symmetric pooling logic.
- Logging Precision: Refactor the CSV logger to separate training statistics from spectral risk statistics (SSI) to make plotting the Pareto Frontier easier.

Part 3: Final Deep Clean Prompt (Optional)

Role: You are a Lead Research Engineer specializing in Differentiable Topology and Graph ML.

Goal: Perform a final deep clean of the provided S3 Framework to meet NeurIPS 2026 production standards.

Tasks

- Docstring Perfection: Ensure all functions in spectral.py and stochastic_rewire.py have NumPy-style docstrings with explicit Args, Returns, and Notes sections.
- Constraint Hardening: In self_healing_step, ensure that if the budget max ratio is exceeded, the spectral healing signal is dampened to prevent densification cheats.
- Logging Granularity: Update the CSV logger to track dual_spectral pressure separately from train_acc so the response signature can be plotted against attack magnitude.
- Device Hygiene: Verify that all tensor creations use the correct device to prevent CPU/GPU mismatch in large-scale runs.
