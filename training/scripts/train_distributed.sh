#!/bin/bash
# 分布式训练启动脚本（多GPU）

set -e

# 配置
PROJECT_ROOT="/home/void0312/HEI-Research"
DATA_PATH="${PROJECT_ROOT}/HEI/data/wiki/wikipedia-zh-20250901.json"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/distributed_$(date +%Y%m%d_%H%M%S)"

# GPU配置
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "检测到 ${NUM_GPUS} 个GPU"

# 模型配置
DIM_Q=256
DIM_Z=64
VOCAB_SIZE=50000

# 训练配置 - 分布式时每GPU的batch_size
BATCH_SIZE_PER_GPU=16
GRAD_ACCUM=8
MAX_SEQ_LEN=512
NUM_EPOCHS=3
MAX_STEPS=100000

# 优化器配置
LR=1e-4
WARMUP_STEPS=5000
WEIGHT_DECAY=0.1

# 其他配置
NUM_WORKERS=4
SAVE_EVERY=5000
LOG_EVERY=100

# 激活环境
cd "${PROJECT_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PINNs
export PYTHONPATH="${PROJECT_ROOT}/HEI:$PYTHONPATH"

EFFECTIVE_BATCH=$((BATCH_SIZE_PER_GPU * GRAD_ACCUM * NUM_GPUS))

echo "========================================"
echo "  分布式大规模训练"
echo "========================================"
echo "GPU数量: ${NUM_GPUS}"
echo "输出目录: ${OUTPUT_DIR}"
echo "每GPU批量大小: ${BATCH_SIZE_PER_GPU}"
echo "梯度累积: ${GRAD_ACCUM}"
echo "有效批量大小: ${EFFECTIVE_BATCH}"
echo "========================================"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 分布式训练
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    HEI/training/large_scale_trainer.py \
    --dim_q ${DIM_Q} \
    --dim_z ${DIM_Z} \
    --vocab_size ${VOCAB_SIZE} \
    --batch_size ${BATCH_SIZE_PER_GPU} \
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
echo "分布式训练完成！"
echo "模型保存在: ${OUTPUT_DIR}"
echo "========================================"

