#!/bin/bash
#
# 可量化训练基准测试
#
# 使用方法:
#   bash HEI/training/scripts/run_benchmark.sh [phase] [options]
#
# 示例:
#   bash HEI/training/scripts/run_benchmark.sh all          # 运行所有阶段
#   bash HEI/training/scripts/run_benchmark.sh 1            # 只运行Phase 1
#   bash HEI/training/scripts/run_benchmark.sh 2 --quick    # 快速模式运行Phase 2

set -e

# 切换到项目目录
cd /home/void0312/HEI-Research

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PINNs

# 设置Python路径
export PYTHONPATH=/home/void0312/HEI-Research/HEI:$PYTHONPATH

# 默认参数
PHASE="${1:-all}"
DATA_PATH="HEI/data/wiki/wikipedia-zh-20250901.json"
OUTPUT_DIR="benchmark_results/$(date +%Y%m%d_%H%M%S)"

# 移除第一个参数，剩余的作为额外参数传递
shift || true

echo "============================================================"
echo "可量化训练基准测试"
echo "============================================================"
echo ""
echo "评估目标:"
echo "  1. 计算曲线: tokens/s vs (B, L, evolution_steps)"
echo "  2. 内存曲线: peak VRAM vs 同一组自变量"
echo "  3. 能力曲线: PPL/E_pred vs (dim_q, 几何约束权重)"
echo ""
echo "运行阶段: $PHASE"
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "============================================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行基准测试
python HEI/training/benchmark_trainer.py \
    --phase "$PHASE" \
    --data "$DATA_PATH" \
    --output "$OUTPUT_DIR" \
    "$@"

echo ""
echo "============================================================"
echo "基准测试完成"
echo "结果保存至: $OUTPUT_DIR"
echo ""
echo "生成的文件:"
ls -la "$OUTPUT_DIR"
echo "============================================================"

