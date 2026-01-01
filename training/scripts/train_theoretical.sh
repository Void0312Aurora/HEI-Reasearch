#!/bin/bash
#
# 理论对齐训练启动脚本
#
# 使用方法:
#   bash HEI/training/scripts/train_theoretical.sh
#
# 或者自定义参数:
#   bash HEI/training/scripts/train_theoretical.sh --batch_size 32 --epochs 20

set -e

# 切换到项目目录
cd /home/void0312/HEI-Research

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PINNs

# 设置Python路径
export PYTHONPATH=/home/void0312/HEI-Research/HEI:$PYTHONPATH

# 默认参数
DATA_PATH="HEI/data/wiki/wikipedia-zh-20250901.json"
OUTPUT_DIR="checkpoints/theoretical"
DIM_Q=128
DIM_Z=32
NUM_CHARTS=8
T_OFFLINE=20
T_ONLINE=10
BATCH_SIZE=768
EPOCHS=10
LR=1e-4
PHASE1_EPOCHS=10
PHASE2_EPOCHS=20
PHASE3_EPOCHS=40
NUM_WORKERS=8

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "============================================================"
echo "理论对齐训练 (基于公理A1-A5和动力学模板L1-L3)"
echo "============================================================"
echo ""
echo "核心原则:"
echo "  1. SoulEntity是核心，不是Transformer"
echo "  2. 语言是端口(L3)，不是系统本身"
echo "  3. 训练目标是变分自由能F"
echo "  4. 多步接触动力学演化"
echo "  5. 三阶段渐进式训练"
echo ""
echo "配置:"
echo "  数据: $DATA_PATH"
echo "  输出: $OUTPUT_DIR"
echo "  dim_q: $DIM_Q"
echo "  T_offline: $T_OFFLINE"
echo "  T_online: $T_ONLINE"
echo "  batch_size: $BATCH_SIZE"
echo ""
echo "三阶段训练:"
echo "  阶段1: 离线动力学 ($PHASE1_EPOCHS epochs)"
echo "  阶段2: 图册联络 ($PHASE2_EPOCHS epochs)"
echo "  阶段3: 接口对齐 ($PHASE3_EPOCHS epochs)"
echo ""
echo "============================================================"

# 运行训练
python HEI/training/theoretical_trainer.py \
    --data "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dim_q $DIM_Q \
    --dim_z $DIM_Z \
    --num_charts $NUM_CHARTS \
    --T_offline $T_OFFLINE \
    --T_online $T_ONLINE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --phase1_epochs $PHASE1_EPOCHS \
    --phase2_epochs $PHASE2_EPOCHS \
    --phase3_epochs $PHASE3_EPOCHS \
    --num_workers $NUM_WORKERS \
    "$@"

echo ""
echo "============================================================"
echo "训练完成"
echo "模型保存至: $OUTPUT_DIR"
echo "============================================================"

