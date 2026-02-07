# Quick Start: Self-Healing Spectral Stability

## TL;DR - Run the Full Pipeline in 3 Commands

### Windows (PowerShell)
```powershell
# Command 1: Test if adaptation activates under strong shock
python scripts/run_nodecls_wikipedia.py --dataset chameleon --epochs 100 --warmup-epochs 20 --add-knn 10 --anchor-knn 10 --alpha-conn 1.0 --conn-eps 0.2 --edge-budget-min-ratio 0.6 --edge-budget-max-ratio 1.4 --dual-lr 0.02 --cvar-samples 10 --cvar-frac 0.5 --zero-shot-shock --shock-drop-pct 0.5 --shock-samples 200 --shock-cvar-frac 0.2 --shock-adapt --shock-adapt-steps 30 --shock-primal-steps 5 --shock-primal-lr 0.01 --shock-primal-samples 8 --shock-reset-duals --log-csv data/metrics/scenario1.csv

# Command 2: Check if rewiring (not constraints) is the bottleneck
python scripts/run_nodecls_wikipedia.py --dataset chameleon --epochs 100 --warmup-epochs 20 --add-knn 10 --anchor-knn 10 --alpha-conn 0.0 --alpha-edge-budget 0.0 --zero-shot-shock --shock-drop-pct 0.1 --shock-samples 200 --shock-cvar-frac 0.2 --log-csv data/metrics/scenario2.csv

# Command 3: Full 3-way comparison with targeted attack
python scripts/run_nodecls_wikipedia.py --dataset chameleon --epochs 100 --warmup-epochs 20 --add-knn 10 --anchor-knn 10 --alpha-conn 1.0 --conn-eps 0.2 --edge-budget-min-ratio 0.6 --edge-budget-max-ratio 1.4 --dual-lr 0.02 --cvar-samples 10 --cvar-frac 0.5 --zero-shot-shock --targeted-spectral-attack --attack-pct 0.15 --shock-samples 200 --shock-cvar-frac 0.2 --shock-reset-duals --tta-healing-steps 20 --tta-dual-lr 0.01 --comparative-eval --log-csv data/metrics/scenario3.csv
```

### Linux/Mac (Bash)
```bash
./run_healing_eval.sh chameleon 100 20
```

---

## What Each Test Does

| # | Test | Purpose | Key Metric |
|---|------|---------|-----------|
| **1** | Strong shock + adaptation | Is dual ascent working? | λ₂_post > λ₂_pre? |
| **2** | No constraints | Is rewiring or constraints the bottleneck? | Acc(no-const) vs Acc(baseline) |
| **3** | Targeted attack + 3-way | How good is self-healing? | SSI Recovery Ratio |

---

## Understanding the Output

### Scenario 1: Adaptation Activation
```
Epoch 0030 | λ₂_cvar self=0.194 | dual_spectral=0.023
...
Epoch 0100 | λ₂_cvar self=0.283 | dual_spectral=0.156
```
✓ **Good**: λ₂_cvar increases, duals increase → adaptation works
✗ **Bad**: λ₂_cvar flat, duals near-zero → constraints not active

**Fix**: Increase `--shock-drop-pct 0.5` → `0.7` or `--tta-dual-lr 0.01` → `0.05`

---

### Scenario 2: No Constraints
```
Self-healing (α_conn=0.0) accuracy: 0.512
GCN baseline accuracy: 0.489
```
✓ **Good**: No-constraint model ≈ baseline (rewiring is fair)
✗ **Bad**: No-constraint model >> baseline (unfair comparison)

---

### Scenario 3: Comparative Eval
```
====================================================================================================
Method                    Accuracy        CVaR Acc        λ₂ CVaR         Edge Ratio      SSI Recovery
====================================================================================================
GCN Baseline              0.4892          0.3845          0.1523          1.0000          N/A
Frozen Self-Healing       0.5124          0.4156          0.2891          0.7542          N/A
Self-Healing (TTA)        0.5347          0.4521          0.3245          0.7684          1.1228
====================================================================================================

Resilience Gap Analysis:
  Baseline → Healed accuracy gap: +0.0455  ✓ (healed wins)
  Frozen → Healed accuracy gap: +0.0223   ✓ (TTA helps)
  SSI Recovery Ratio (post/pre): 1.1228x  ✓ (λ₂ improved 12.28%)
```

✓ **Ideal**: All gaps positive, SSI recovery > 1.0

---

## Key Parameters

### Targeted Spectral Attack
- `--targeted-spectral-attack`: Enable (vs random deletion)
- `--attack-pct 0.15`: Attack 15% of anchors (try 0.10–0.30)

### Test-Time Adaptation
- `--tta-healing-steps 20`: Dual ascent iterations (try 10–50)
- `--tta-dual-lr 0.01`: Dual step size (try 0.005–0.05)
- `--shock-reset-duals`: Zero duals before adaptation (recommended)

### Shock Configuration
- `--shock-drop-pct 0.5`: Drop 50% of anchors (0.1 = weak, 0.5 = strong)
- `--shock-adapt`: Enable test-time adaptation
- `--shock-adapt-steps`: How many dual updates during main training

### Monitoring
- `--cvar-samples 10`: Stochastic samples for λ₂ CVaR (more = more robust)
- `--cvar-frac 0.5`: Tail fraction (0.5 = worst 50%, 0.2 = worst 20%)

---

## Common Issues & Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| TTA not improving λ₂ | λ₂_pre ≈ λ₂_post in Scenario 1 | ↑ `--tta-dual-lr` to 0.02–0.05 |
| Adaptation not active | dual_spectral stays near 0 | ↑ `--shock-drop-pct` to 0.5 or ↓ `--conn-eps` to 0.1 |
| Healed accuracy drops | Self-healing < frozen | This is expected (no accuracy proxy); add distillation loss |
| Attack too weak | SSI recovery near 1.0 | ↑ `--attack-pct` to 0.20–0.30 |
| Out of memory | GPU error | ↓ `--cvar-samples` to 4–6 or ↓ `--tta-healing-steps` to 10 |

---

## Expected Results (Chameleon Dataset)

| Metric | Baseline | Frozen | Healed |
|--------|----------|--------|--------|
| **Accuracy** | 0.48–0.52 | 0.51–0.54 | 0.52–0.56 |
| **λ₂ CVaR** | 0.12–0.18 | 0.25–0.35 | 0.28–0.40 |
| **SSI Recovery** | N/A | N/A | 1.05–1.20 |

---

## Next Steps

1. **Run all 3 scenarios** → Confirm each works
2. **Analyze CSVs** → Plot λ₂ and accuracy trends
3. **Sweep hyperparams** → Find optimal `--tta-dual-lr` and `--attack-pct`
4. **Ablations**:
   - Remove `--shock-reset-duals` → Does it still work?
   - Reduce `--tta-healing-steps` → Diminishing returns?
5. **Add accuracy proxy** → Implement distillation loss for Scenario 3

---

## Files Generated

After running all 3 scenarios:
- `data/metrics/scenario1.csv` → Activation diagnostics
- `data/metrics/scenario2.csv` → No-constraints baseline
- `data/metrics/scenario3.csv` → 3-way comparison + SSI recovery

---

## For More Details

- **Full implementation**: See `IMPLEMENTATION.md`
- **Usage guide**: See `README_HEALING_EVAL.md`
- **Automated pipeline**: See `run_healing_eval.sh` (Linux) or `run_healing_eval.bat` (Windows)

---

## Citation

If this framework helps your research:

```bibtex
@article{hetgnn2025,
  title={Stochastic Spectral Stability: Self-Healing Graph Topology 
         Under Structural Shocks via Primal-Dual Optimization},
  year={2025},
  note={Spectral resilience + test-time adaptation framework}
}
```

Good luck! 🚀
