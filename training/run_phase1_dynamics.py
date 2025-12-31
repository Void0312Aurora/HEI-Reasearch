"""
阶段1：离线动力学训练

目标：训练H^c的参数，使接触动力学满足Lyapunov性质

理论依据：
- A1: 系统在无输入时能自演化
- A2: 演化应收敛到有意义的吸引子
- Lyapunov: F应在演化过程中单调下降

成功标准：
- Lyapunov违反率 < 5%
- 吸引子数量 > 1（避免collapse）
- F在演化过程中单调下降

运行：
python HEI/training/run_phase1_dynamics.py --epochs 100 --save_dir checkpoints/phase1
"""

import os
import sys
import argparse
import json
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.lyapunov_training import (
    LyapunovTrainingConfig,
    OfflineDynamicsTrainer,
)


def train_phase1(args):
    """阶段1训练主循环"""
    
    print("=" * 70)
    print("阶段1：离线动力学训练")
    print("=" * 70)
    print("\n理论目标：")
    print("  - A1: 系统在无输入时能自演化")
    print("  - A2: 演化应收敛到有意义的吸引子")  
    print("  - Lyapunov: F应在演化过程中单调下降")
    print("\n成功标准：")
    print("  - Lyapunov违反率 < 5%")
    print("  - 吸引子数量 > 1")
    print("  - F持续下降")
    print("=" * 70)
    
    # 配置
    config = LyapunovTrainingConfig(
        dim_q=args.dim_q,
        dim_z=args.dim_z,
        T_offline=args.T_offline,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_lyapunov=args.lambda_lyap,
        lambda_converge=args.lambda_conv,
        lambda_diversity=args.lambda_div,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\n配置：")
    print(f"  dim_q: {config.dim_q}")
    print(f"  T_offline: {config.T_offline}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  λ_lyapunov: {config.lambda_lyapunov}")
    print(f"  λ_converge: {config.lambda_converge}")
    print(f"  λ_diversity: {config.lambda_diversity}")
    print(f"  device: {config.device}")
    
    # 创建训练器
    trainer = OfflineDynamicsTrainer(config)
    trainer = trainer.to(config.device)
    
    # 优化器
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps_per_epoch
    )
    
    # 保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, 'training_log.jsonl')
    
    # 训练历史
    best_violation_rate = 1.0
    
    print(f"\n开始训练... (epochs: {args.epochs})")
    print("-" * 70)
    
    for epoch in range(args.epochs):
        epoch_stats = {
            'loss': 0, 'violation_rate': 0, 'mean_decrease': 0,
            'F_initial': 0, 'F_final': 0, 'attractors': 0
        }
        
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")
        for step in pbar:
            optimizer.zero_grad()
            
            diagnostics = trainer.train_step()
            loss = diagnostics['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # 累积统计
            epoch_stats['loss'] += loss.item()
            epoch_stats['violation_rate'] += diagnostics['lyap_violation_rate'].item()
            epoch_stats['mean_decrease'] += diagnostics['lyap_mean_decrease'].item()
            epoch_stats['F_initial'] += diagnostics['lyap_F_initial'].item()
            epoch_stats['F_final'] += diagnostics['lyap_F_final'].item()
            epoch_stats['attractors'] += diagnostics['div_estimated_attractors'].item()
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.2f}",
                'Viol': f"{diagnostics['lyap_violation_rate'].item():.1%}",
                'dF': f"{diagnostics['lyap_mean_decrease'].item():.3f}",
            })
        
        # 计算epoch平均
        n = args.steps_per_epoch
        for k in epoch_stats:
            epoch_stats[k] /= n
        
        # 打印epoch总结
        print(f"\nEpoch {epoch+1} 总结：")
        print(f"  损失: {epoch_stats['loss']:.4f}")
        print(f"  Lyapunov违反率: {epoch_stats['violation_rate']:.2%}")
        print(f"  F下降量: {epoch_stats['mean_decrease']:.4f}")
        print(f"  F: {epoch_stats['F_initial']:.2f} → {epoch_stats['F_final']:.2f}")
        print(f"  估计吸引子数: {epoch_stats['attractors']:.1f}")
        
        # 保存日志
        log_entry = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            **epoch_stats
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # 保存最佳模型
        if epoch_stats['violation_rate'] < best_violation_rate:
            best_violation_rate = epoch_stats['violation_rate']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'violation_rate': best_violation_rate,
                'config': config,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  ★ 新最佳模型! 违反率: {best_violation_rate:.2%}")
        
        # 检查是否达到成功标准
        if epoch_stats['violation_rate'] < 0.05 and epoch_stats['attractors'] > 1:
            print("\n" + "=" * 70)
            print("✓ 达到成功标准！阶段1训练完成")
            print("=" * 70)
            break
        
        print("-" * 70)
    
    # 保存最终模型
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': trainer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_violation_rate': epoch_stats['violation_rate'],
        'config': config,
    }, os.path.join(args.save_dir, 'final_model.pt'))
    
    print("\n" + "=" * 70)
    print("阶段1训练完成")
    print(f"最佳Lyapunov违反率: {best_violation_rate:.2%}")
    print(f"模型保存至: {args.save_dir}")
    print("=" * 70)
    
    # 返回是否成功
    return best_violation_rate < 0.05


def main():
    parser = argparse.ArgumentParser(description='阶段1：离线动力学训练')
    
    # 模型参数
    parser.add_argument('--dim_q', type=int, default=64, help='状态空间维度')
    parser.add_argument('--dim_z', type=int, default=16, help='上下文维度')
    
    # 演化参数
    parser.add_argument('--T_offline', type=int, default=50, help='离线演化步数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='每轮步数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    
    # 损失权重
    parser.add_argument('--lambda_lyap', type=float, default=10.0, help='Lyapunov约束权重')
    parser.add_argument('--lambda_conv', type=float, default=1.0, help='收敛约束权重')
    parser.add_argument('--lambda_div', type=float, default=0.1, help='多样性约束权重')
    
    # 保存
    parser.add_argument('--save_dir', type=str, default='checkpoints/phase1', help='保存目录')
    
    args = parser.parse_args()
    
    success = train_phase1(args)
    
    if success:
        print("\n下一步：运行阶段2训练")
        print("python HEI/training/run_phase2_atlas.py --checkpoint checkpoints/phase1/best_model.pt")
    else:
        print("\n阶段1未达成功标准，建议：")
        print("  1. 增加训练轮数 --epochs 200")
        print("  2. 增加Lyapunov权重 --lambda_lyap 50.0")
        print("  3. 增加演化步数 --T_offline 100")


if __name__ == "__main__":
    main()

