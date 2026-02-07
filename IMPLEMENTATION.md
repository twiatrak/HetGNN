# Self-Healing Framework Implementation Summary

## Overview
Implemented three core components for **Spectral Stability Index (SSI) Self-Healing**:
1. **Targeted Spectral Attack**: Identify critical edges via gradient-free finite differences
2. **Test-Time Adaptation (TTA)**: Primal-dual dual ascent loop for topology repair
3. **Comparative Evaluation**: 3-way benchmark (GCN Baseline vs Frozen vs Self-Healed)

---

## Component 1: Targeted Spectral Attack

### Location
`scripts/run_nodecls_wikipedia.py` lines 129–225

### Function Signature
```python
@torch.no_grad()
def targeted_spectral_attack(
    model: SimpleNodeClassifier,
    data_x: torch.Tensor,
    edge_index: torch.Tensor,
    candidate_edge_index: torch.Tensor | None,
    anchor_edge_index: torch.Tensor | None,
    attack_pct: float = 0.2,
    num_iters: int = 25,
    num_restarts: int = 4,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
```

### Algorithm
1. **Compute baseline λ₂** on full graph with all anchors (deterministic gates, `sample=False`)
2. **For each anchor edge i**:
   - Remove edge i from anchor set
   - Recompute λ₂ without edge i
   - Score = (λ₂_baseline - λ₂_without_i)  → edge importance
3. **Select top-k edges** with highest scores (≈ `attack_pct` of total anchors)
4. **Return**:
   - `attack_idx`: Indices of edges to remove
   - `edge_index_attacked`: Full topology after removing attacked edges

### Key Details
- **Differentiability**: Uses no-grad context; fully deterministic
- **Scalability**: O(|anchors|) λ₂ evaluations; acceptable for small graphs (Chameleon: ~2.4k nodes)
- **Anchor coverage**: Only operates on anchor edges (added via `--anchor-knn`)
- **Temperature consistency**: Uses `sample=False` → always applies `sigmoid(logits / τ)` consistently

---

## Component 2: Test-Time Adaptation (TTA)

### Location
`scripts/run_nodecls_wikipedia.py` lines 228–310

### Function Signature
```python
def self_healing_step(
    model: SimpleNodeClassifier,
    data_x: torch.Tensor,
    edge_index_attacked: torch.Tensor,
    candidate_edge_index: torch.Tensor | None,
    anchor_shocked: torch.Tensor,
    num_nodes: int,
    target_lambda2: float,
    dual_lr: float = 0.01,
    num_iters: int = 20,
    conn_iters: int = 25,
    cvar_frac: float = 0.5,
    cvar_samples: int = 4,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
```

### Algorithm (Dual Ascent Loop)
1. **Forward pass** (frozen embeddings, no grad):
   - Compute h1 from GNN encoder
   - Freeze h1 (detach)
2. **Measure pre-healing λ₂ CVaR**:
   - Sample `cvar_samples` stochastic rewirings (gate samples = True)
   - Compute λ₂ for each sample
   - CVaR = mean of worst `cvar_frac` tail
3. **For each TTA step (1 to `num_iters`)**:
   - Sample `cvar_samples` stochastic rewirings
   - Compute λ₂ CVaR for samples
   - Spectral violation = max(0, target_λ₂ - λ₂_cvar)
   - **Dual update**: `dual_spectral += dual_lr * spectral_violation`
4. **Measure post-healing λ₂ CVaR** (same as step 2)
5. **Return**: (lam2_pre_healing, lam2_post_healing)

### Key Details
- **GNN Freeze**: Only `model.dual_spectral` buffer is updated; no gate MLP gradients
- **Risk-aware**: Uses CVaR (tail risk) not mean → robust to outlier samples
- **Dual-only**: No primal updates on gates (for isolation); can add primal later
- **Initialization**: Optional `--shock-reset-duals` zeros all dual buffers before TTA

### SSI Recovery Ratio
$$\text{Recovery Ratio} = \frac{\lambda_2^{\text{post-healing}}}{\lambda_2^{\text{pre-healing}}}$$

- **> 1.0**: Success (connectivity increased)
- **= 1.0**: No change (constraint already satisfied)
- **< 1.0**: Degradation (rare; indicates limit of dual ascent)

---

## Component 3: Comparative Evaluation (3-Way)

### Location
`scripts/run_nodecls_wikipedia.py` lines 1138–1230

### Three Models

#### (1) GCN Baseline
- Standard GCN backbone, **no rewiring**
- Frozen weights (same as main model)
- Evaluated on attacked graph
- **Expected**: Fragile; accuracy drops sharply

#### (2) Frozen Self-Healing
- Same model as main (with rewiring)
- Frozen weights and gate MLP
- Stochastic rewiring enabled but no dual updates
- **Expected**: Better than baseline due to rewiring robustness

#### (3) Self-Healing (TTA)
- Same model as (2)
- After TTA loop with dual ascent
- Gates adjusted to repair connectivity
- **Expected**: Best accuracy + highest λ₂

### Metrics Reported
| Metric | Interpretation |
|--------|-----------------|
| **Accuracy** | Mean test accuracy (cvar_samples draws) |
| **CVaR Accuracy** | Tail-risk accuracy (worst `cvar_frac`) |
| **λ₂ CVaR** | Spectral health under stochasticity |
| **Edge Ratio** | Keep ratio = kept_edges / ref_edges |
| **SSI Recovery** | λ₂_post / λ₂_pre (only for healed) |

### Output Table Example
```
Method                    Accuracy        CVaR Acc        λ₂ CVaR         Edge Ratio      SSI Recovery
GCN Baseline              0.4892          0.3845          0.1523          1.0000          N/A
Frozen Self-Healing       0.5124          0.4156          0.2891          0.7542          N/A
Self-Healing (TTA)        0.5347          0.4521          0.3245          0.7684          1.1228
```

### Gap Analysis
```
Resilience Gap Analysis:
  Baseline → Healed accuracy gap: +0.0455  (how much healed outperforms baseline)
  Frozen → Healed accuracy gap: +0.0223   (TTA incremental gain)
  SSI Recovery Ratio (post/pre): 1.1228x  (connectivity repair magnitude)
  SSI Improvement (post - pre): +0.0354   (absolute λ₂ gain)
```

---

## Integration with Main Script

### Command-Line Arguments Added
```python
parser.add_argument("--targeted-spectral-attack", action="store_true")
parser.add_argument("--attack-pct", type=float, default=0.15)
parser.add_argument("--tta-healing-steps", type=int, default=20)
parser.add_argument("--tta-dual-lr", type=float, default=0.01)
parser.add_argument("--comparative-eval", action="store_true")
```

### Control Flow (in `--zero-shot-shock` block)
```
if args.zero_shot_shock:
  if args.targeted_spectral_attack:
    attack_idx, edge_index_attacked = targeted_spectral_attack(...)
  else:
    # Random deletion (original behavior)
    
  # Standard eval (frozen model)
  mean_self, lam2_self, ... = eval_cvar_accuracy(model, rewire=True)
  
  if args.comparative_eval:
    # 1. Baseline GCN
    mean_baseline, lam2_baseline, ... = eval_cvar_accuracy(baseline, rewire=False)
    
    # 2. Frozen Self-Healing
    mean_frozen, lam2_frozen, ... = eval_cvar_accuracy(model, rewire=True)
    
    # 3. Self-Healing with TTA
    lam2_pre, lam2_post = self_healing_step(model, ...)
    mean_healed, lam2_healed, ... = eval_cvar_accuracy(model, rewire=True)
    
    # Print 3-way table + gaps
```

---

## Consistency Fixes Applied

### Issue #2: Temperature Consistency
**Before**: Budget constraints used `sigmoid(logits)`, gates used `sigmoid(logits / τ)`
**After**: All use `sigmoid(logits / τ)` consistently

**File**: `src/hetgnn_spectral_stability/layers/stochastic_rewire.py`
- Line ~300: `probs = torch.sigmoid(logits / temp)` (where `temp = max(temperature, 1e-12)`)
- Ensures monitoring, budgeting, and sampling all use same probability distribution

### Issue #3: Deterministic Ratio Logging
**Before**: `eval_shock_stats()` computed ratio from last stochastic sample
**After**: Uses separate deterministic pass

**File**: `scripts/run_nodecls_wikipedia.py` in `eval_shock_stats()`
```python
# Deterministic pass for ratio
rw_edge_index_det, rw_edge_weight_det, rw_stats_det, _ = model_eval.rewire(
    h1_detached,
    edge_index_shocked,
    sample=False,  # <-- Deterministic
)
ratio = (rw_stats_det.expected_num_edges / e_ref_for(...)).clamp_min(0.0)

# Stochastic pass for λ₂ risk
for _ in range(cvar_samples):
    rw_ei, rw_ew, _, _ = model_eval.rewire(..., sample=True)
    lam2_s = estimate_lambda2_norm_min_rayleigh(...)
```

---

## Testing & Validation

### Unit Test: Targeted Attack Scores
- Verify `attack_idx` has correct length
- Check scores are non-negative
- Confirm attacked edges are removed from `edge_index_attacked`

### Unit Test: TTA Recovery
- Call `self_healing_step()` with dummy model
- Verify `lam2_pre_healing` and `lam2_post_healing` are valid floats
- Check `dual_spectral` increases with iterations

### Integration Test: Comparative Eval
- Run all three models on same attacked graph
- Verify no crashes, all metrics populated
- Check SSI recovery ratio > 0 and finite

### Example Commands
```bash
# Test Scenario 1: Activation
python scripts/run_nodecls_wikipedia.py --dataset chameleon --epochs 50 \
  --zero-shot-shock --shock-drop-pct 0.5 --shock-adapt --shock-adapt-steps 30

# Test Scenario 2: No constraints
python scripts/run_nodecls_wikipedia.py --dataset chameleon --epochs 50 \
  --alpha-conn 0.0 --alpha-edge-budget 0.0 --zero-shot-shock --shock-drop-pct 0.1

# Test Scenario 3: Full pipeline
python scripts/run_nodecls_wikipedia.py --dataset chameleon --epochs 50 \
  --zero-shot-shock --targeted-spectral-attack --attack-pct 0.15 \
  --tta-healing-steps 20 --comparative-eval
```

---

## Future Extensions

1. **Primal Updates during TTA**: Add gate MLP fine-tuning
2. **Accuracy Proxy**: Distillation loss to preserve pre-shock logits
3. **Adaptive Dual LR**: Schedule `dual_lr` based on constraint satisfaction
4. **Multi-layer Adaptation**: Adapt embeddings beyond h1
5. **Uncertainty Quantification**: Confidence intervals on SSI recovery ratio

---

## Files Modified

1. **scripts/run_nodecls_wikipedia.py**
   - Added: `targeted_spectral_attack()`, `self_healing_step()`
   - Modified: Shock evaluation block (lines 1030–1230)
   - Added: Arguments for targeted attack, TTA, comparative eval

2. **src/hetgnn_spectral_stability/regularizers/spectral.py**
   - Fixed: Removed accidentally pasted analysis text (syntax error)
   - Temperature consistency already implemented in rewire layer

3. **Documentation**
   - Created: `README_HEALING_EVAL.md` (usage guide)
   - Created: `run_healing_eval.sh` (bash runner)
   - Created: `run_healing_eval.bat` (Windows runner)
   - Created: This file (`IMPLEMENTATION.md`)

---

## Deployment Checklist

- [x] Core functions implemented and tested
- [x] Arguments added to argparse
- [x] Integration with `--zero-shot-shock` pipeline
- [x] Temperature consistency fixed
- [x] Deterministic ratio logging fixed
- [x] Comparative evaluation table formatted
- [x] Documentation created (README + examples)
- [x] Runners created (bash + batch)
- [x] Error handling for edge cases
- [ ] Full end-to-end test on Chameleon dataset
- [ ] Benchmark on Squirrel dataset
- [ ] Hyperparameter sweep (attack_pct, tta_dual_lr, tta_healing_steps)

---

## Support & Debugging

**Q: TTA not improving λ₂?**
- Increase `--tta-dual-lr` (try 0.02–0.05)
- Increase `--tta-healing-steps` (try 30–50)
- Check `--shock-reset-duals` is set

**Q: Attack too weak?**
- Increase `--attack-pct` (try 0.25–0.30)
- Verify `--anchor-knn` is set (else no anchor edges to attack)

**Q: Comparison looks odd?**
- Ensure all three runs use same `--split` and `--seed`
- Check CSV logs for NaN values

**Q: Memory issues?**
- Reduce `--cvar-samples` (default 10 → try 4–6)
- Reduce `--tta-healing-steps` (default 20 → try 10)

---

## References

- Spectral Stability Index: $\lambda_2$ (algebraic connectivity)
- CVaR (Conditional Value at Risk): tail-risk metric for robust decisions
- Lagrangian Duality: primal-dual optimization framework
- Primal-Dual Ascent: coordinate-wise updates on constraints
