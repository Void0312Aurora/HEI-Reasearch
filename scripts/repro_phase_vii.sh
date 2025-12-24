#!/bin/bash
# Aurora Phase VII Reproducibility Script
# Target: Active Rank < 50 (Phase V baseline) in 2k steps
# Validates: Holdout generalization, no overfitting

set -e  # Exit on error

echo "====================================="
echo "Aurora Phase VII Reproducibility Test"
echo "====================================="
echo ""

# Configuration (from aurora_recipe_v1.yaml)
STEPS=2000
SEED=42
DATASET="cilin"
SEMANTIC_PATH="checkpoints/semantic_edges_wiki.pkl"
CHECKPOINT="checkpoints/aurora_v2_final.pkl"  # Phase IV stable checkpoint
OUTPUT_CHECKPOINT="checkpoints/aurora_repro_test.pkl"
SPLIT="train"

# Triplet Loss Configuration
TRIPLET_FLAG="--triplet"
NUM_CANDIDATES=50
K_REP=0.1
K_SEM=1.0
STAGE_RATIO=0.0  # No staging

echo "Configuration:"
echo "  Steps: $STEPS"
echo "  Seed: $SEED"
echo "  Hard Mining Candidates: $NUM_CANDIDATES"
echo "  Starting Checkpoint: $CHECKPOINT"
echo ""

# Check prerequisites
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint $CHECKPOINT not found!"
    echo "Please ensure Phase IV checkpoint exists."
    exit 1
fi

if [ ! -f "$SEMANTIC_PATH" ]; then
    echo "ERROR: Semantic edges $SEMANTIC_PATH not found!"
    echo "Please build semantic edges first."
    exit 1
fi

echo "[1/3] Training on 90% Train Split..."
python -u scripts/train_aurora_v2.py \
    --steps $STEPS \
    --stage_ratio $STAGE_RATIO \
    --k_sem $K_SEM \
    --k_rep $K_REP \
    $TRIPLET_FLAG \
    --num_candidates $NUM_CANDIDATES \
    --checkpoint $CHECKPOINT \
    --split $SPLIT \
    --dataset $DATASET \
    --semantic_path $SEMANTIC_PATH \
    --device cuda \
    --seed $SEED \
    --save_path $OUTPUT_CHECKPOINT

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

echo ""
echo "[2/3] Evaluating on 10% Holdout Split..."
python scripts/eval_aurora.py \
    --checkpoint $OUTPUT_CHECKPOINT \
    --semantic_path $SEMANTIC_PATH \
    --split holdout | tee eval_repro.log

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed!"
    exit 1
fi

echo ""
echo "[3/3] Verifying Active Rank < 50..."

# Extract Active Rank from eval output
ACTIVE_RANK=$(grep "Active Subgraph Mean Rank" eval_repro.log | awk '{print $5}')

if [ -z "$ACTIVE_RANK" ]; then
    echo "ERROR: Could not extract Active Rank from evaluation output!"
    exit 1
fi

echo "  Active Rank: $ACTIVE_RANK"

# Check if Active Rank < 50 (Phase V baseline)
if (( $(echo "$ACTIVE_RANK < 50.0" | bc -l) )); then
    echo "  ✓ PASSED: Active Rank < 50 (Phase V baseline achieved)"
else
    echo "  ✗ FAILED: Active Rank >= 50 (Expected improvement not achieved)"
    exit 1
fi

echo ""
echo "====================================="
echo "Reproducibility Test: PASSED"
echo "====================================="
echo ""
echo "Summary:"
echo "  - Training completed: $STEPS steps"
echo "  - Holdout Active Rank: $ACTIVE_RANK"
echo "  - Checkpoint saved: $OUTPUT_CHECKPOINT"
echo ""
echo "Next steps:"
echo "  - For Phase VII full validation: Run 3-seed robustness (seeds 42,43,44)"
echo "  - For Active < 20: Implement semi-hard curriculum or memory bank"
