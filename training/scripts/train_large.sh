#!/bin/bash
# 大规模训练启动脚本

set -e

# 配置
PROJECT_ROOT="/home/void0312/HEI-Research"
DATA_PATH="${PROJECT_ROOT}/HEI/data/wiki/wikipedia-zh-20250901.json"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/large_scale_$(date +%Y%m%d_%H%M%S)"

# 模型配置
DIM_Q=128
DIM_Z=32
VOCAB_SIZE=32000

# 训练配置
BATCH_SIZE=32
GRAD_ACCUM=16
MAX_SEQ_LEN=256
NUM_EPOCHS=5
MAX_STEPS=-1

# 优化器配置
LR=3e-4
WARMUP_STEPS=2000
WEIGHT_DECAY=0.1

# 其他配置
NUM_WORKERS=4
SAVE_EVERY=2000
LOG_EVERY=50

# 激活环境
cd "${PROJECT_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PINNs
export PYTHONPATH="${PROJECT_ROOT}/HEI:$PYTHONPATH"

echo "========================================"
echo "  大规模训练"
echo "========================================"
echo "输出目录: ${OUTPUT_DIR}"
echo "数据路径: ${DATA_PATH}"
echo "有效批量大小: $((BATCH_SIZE * GRAD_ACCUM))"
echo "========================================"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 单卡训练
python HEI/training/large_scale_trainer.py \
    --dim_q ${DIM_Q} \
    --dim_z ${DIM_Z} \
    --vocab_size ${VOCAB_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation ${GRAD_ACCUM} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --num_epochs ${NUM_EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --lr ${LR} \
    --warmup_steps ${WARMUP_STEPS} \
    --weight_decay ${WEIGHT_DECAY} \
    --data "${DATA_PATH}" \
    --num_workers ${NUM_WORKERS} \
    --output_dir "${OUTPUT_DIR}" \
    --save_every ${SAVE_EVERY} \
    --log_every ${LOG_EVERY} \
    --amp_dtype bfloat16 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "========================================"
echo "训练完成！"
echo "模型保存在: ${OUTPUT_DIR}"
echo "========================================"

