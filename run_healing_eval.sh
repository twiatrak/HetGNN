#!/bin/bash
# Self-Healing Framework: Complete Evaluation Pipeline
# Run all three diagnostic scenarios for the Stochastic Spectral Stability framework

set -e

DATASET=${1:-chameleon}
EPOCHS=${2:-100}
WARMUP=${3:-20}
SEED=${4:-42}

echo "=================================="
echo "Self-Healing SSI Framework"
echo "=================================="
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS, Warmup: $WARMUP"
echo "Seed: $SEED"
echo ""

# Common parameters
COMMON_ARGS="
--dataset $DATASET
--epochs $EPOCHS
--warmup-epochs $WARMUP
--add-knn 10
--anchor-knn 10
--alpha-conn 1.0
--conn-eps 0.2
--edge-budget-min-ratio 0.6
--edge-budget-max-ratio 1.4
--dual-lr 0.02
--cvar-samples 10
--cvar-frac 0.5
--shock-samples 200
--shock-cvar-frac 0.2
--shock-reset-duals
--seed $SEED
"

# ============================================================
# Scenario 1: Is adaptation activating? (Run A)
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Scenario 1: Adaptation Activation (Stronger Shock)            ║"
echo "║ Question: Can primal-dual mechanism repair λ₂ when threatened? ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Command:"
echo "python scripts/run_nodecls_wikipedia.py \\"
echo "  $COMMON_ARGS \\"
echo "  --zero-shot-shock \\"
echo "  --shock-drop-pct 0.5 \\"
echo "  --shock-adapt \\"
echo "  --shock-adapt-steps 30 \\"
echo "  --shock-primal-steps 5 \\"
echo "  --shock-primal-lr 0.01 \\"
echo "  --shock-primal-samples 8 \\"
echo "  --log-csv data/metrics/scenario1_activation.csv"
echo ""
read -p "Press Enter to run Scenario 1..." -t 30 || true
python scripts/run_nodecls_wikipedia.py \
  $COMMON_ARGS \
  --zero-shot-shock \
  --shock-drop-pct 0.5 \
  --shock-adapt \
  --shock-adapt-steps 30 \
  --shock-primal-steps 5 \
  --shock-primal-lr 0.01 \
  --shock-primal-samples 8 \
  --log-csv data/metrics/scenario1_activation.csv

# ============================================================
# Scenario 2: Is rewiring the accuracy bottleneck? (Run B)
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Scenario 2: Rewiring vs Constraint Bottleneck                 ║"
echo "║ Question: Does rewiring model itself (no constraints) match    ║"
echo "║          baseline accuracy, or is constraint pressure the gap? ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Command:"
echo "python scripts/run_nodecls_wikipedia.py \\"
echo "  $COMMON_ARGS \\"
echo "  --alpha-conn 0.0 \\"
echo "  --alpha-edge-budget 0.0 \\"
echo "  --zero-shot-shock \\"
echo "  --shock-drop-pct 0.1 \\"
echo "  --log-csv data/metrics/scenario2_no_constraints.csv"
echo ""
read -p "Press Enter to run Scenario 2..." -t 30 || true
python scripts/run_nodecls_wikipedia.py \
  $COMMON_ARGS \
  --alpha-conn 0.0 \
  --alpha-edge-budget 0.0 \
  --zero-shot-shock \
  --shock-drop-pct 0.1 \
  --log-csv data/metrics/scenario2_no_constraints.csv

# ============================================================
# Scenario 3: Targeted Spectral Attack + Comparative Eval
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Scenario 3: Targeted Spectral Attack + 3-Way Comparison       ║"
echo "║ Questions:                                                     ║"
echo "║ 1) Can we identify critical edges for spectral health?        ║"
echo "║ 2) How does baseline GCN vs frozen vs healing compare?         ║"
echo "║ 3) What is the SSI Recovery Ratio post-healing?                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Command:"
echo "python scripts/run_nodecls_wikipedia.py \\"
echo "  $COMMON_ARGS \\"
echo "  --zero-shot-shock \\"
echo "  --targeted-spectral-attack \\"
echo "  --attack-pct 0.15 \\"
echo "  --tta-healing-steps 20 \\"
echo "  --tta-dual-lr 0.01 \\"
echo "  --comparative-eval \\"
echo "  --log-csv data/metrics/scenario3_targeted_attack.csv"
echo ""
read -p "Press Enter to run Scenario 3..." -t 30 || true
python scripts/run_nodecls_wikipedia.py \
  $COMMON_ARGS \
  --zero-shot-shock \
  --targeted-spectral-attack \
  --attack-pct 0.15 \
  --tta-healing-steps 20 \
  --tta-dual-lr 0.01 \
  --comparative-eval \
  --log-csv data/metrics/scenario3_targeted_attack.csv

# ============================================================
# Summary
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Evaluation Complete!                                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to:"
echo "  - data/metrics/scenario1_activation.csv"
echo "  - data/metrics/scenario2_no_constraints.csv"
echo "  - data/metrics/scenario3_targeted_attack.csv"
echo ""
echo "Next steps:"
echo "1. Analyze SSI Recovery Ratios (Scenario 3)"
echo "2. Compare Scenario 1: λ₂_post-adapt > λ₂_pre-adapt?"
echo "3. Compare Scenario 2: Rewiring model accuracy vs baseline?"
echo ""
echo "For full details, see: README_HEALING_EVAL.md"
