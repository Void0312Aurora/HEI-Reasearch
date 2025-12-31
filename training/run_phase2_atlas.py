"""
阶段2：图册联络训练 (Atlas & Connection Training)

目标：训练L2层的图册结构和联络，确保：
- 多图册覆盖状态空间
- 图册间转移的一致性
- 联络平行移动满足正交性

理论依据：
- L2: 李代数胚联络 / 平行移动
- 几何基础：多图册结构、协调相容性、联络正交性

成功标准：
- 闭环误差 < 0.1 (图册转移一致性)
- 联络正交化损失 < 0.01
- 图册熵 > 0.5 (多图册激活)

运行：
python HEI/training/run_phase2_atlas.py --checkpoint checkpoints/phase1/best_model.pt --epochs 50
"""

import os
import sys
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from he_core.soul_entity import SoulEntity, create_soul_entity
from he_core.state import ContactState
from he_core.atlas import AtlasRouter, TransitionMap
from he_core.connection import Connection


@dataclass
class AtlasTrainingConfig:
    """阶段2训练配置"""
    dim_q: int = 64
    dim_z: int = 16
    num_charts: int = 4
    batch_size: int = 32
    learning_rate: float = 1e-4
    
    # 损失权重
    lambda_consistency: float = 1.0   # 图册一致性
    lambda_orthogonal: float = 0.1    # 联络正交性
    lambda_entropy: float = 0.01      # 图册熵（多样性）
    lambda_stability: float = 0.1     # 离线稳定性保持
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class AtlasTrainer(nn.Module):
    """阶段2：图册联络训练器"""
    
    def __init__(self, config: AtlasTrainingConfig, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.config = config
        
        # 创建SoulEntity
        self.entity = create_soul_entity({
            'dim_q': config.dim_q,
            'dim_z': config.dim_z,
            'hyperbolic_c': 1.0,
            'device': config.device,
        })
        
        # 加载阶段1的权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"加载阶段1权重: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
            # 提取entity的权重
            entity_state = {}
            for k, v in ckpt['model_state_dict'].items():
                if k.startswith('entity.'):
                    entity_state[k[7:]] = v  # 去掉'entity.'前缀
            if entity_state:
                self.entity.load_state_dict(entity_state, strict=False)
                print("  ✓ 权重加载成功")
        
        # 图册组件
        self.atlas_router = AtlasRouter(
            dim_q=config.dim_q,
            num_charts=config.num_charts
        )
        
        # 转移映射 - 简化版：直接在q空间做映射
        # 每对图册之间有一个转移映射
        self.transition_maps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.dim_q, config.dim_q),
                nn.Tanh(),
                nn.Linear(config.dim_q, config.dim_q)
            )
            for _ in range(config.num_charts * (config.num_charts - 1))
        ])
        
        # 随机初始化（非恒等），训练目标是让闭环误差趋近0
        for tm in self.transition_maps:
            nn.init.orthogonal_(tm[0].weight, gain=0.5)
            nn.init.normal_(tm[0].bias, std=0.1)
            nn.init.orthogonal_(tm[2].weight, gain=0.5)
            nn.init.normal_(tm[2].bias, std=0.1)
        
        # 联络
        self.connection = Connection(
            dim_q=config.dim_q,
            hidden_dim=64
        )
        
    def random_init_state(self, batch_size: int) -> ContactState:
        """初始化状态"""
        # 创建ContactState并手动设置q, p, s
        state = ContactState(
            dim_q=self.config.dim_q,
            batch_size=batch_size,
            device=self.config.device
        )
        # 用小随机值初始化
        state._data = torch.randn(batch_size, 2 * self.config.dim_q + 1, device=self.config.device) * 0.1
        return state
    
    def compute_chart_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """计算图册分布熵（衡量多图册利用度）"""
        # weights: (batch, num_charts)
        # 使用softmax确保归一化
        p = F.softmax(weights, dim=-1) + 1e-8
        entropy = -(p * p.log()).sum(dim=-1).mean()
        return entropy
    
    def compute_consistency_loss(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算图册一致性损失"""
        batch_size = q.shape[0]
        
        # 获取图册权重
        chart_weights = self.atlas_router(q)  # (batch, num_charts)
        
        # 找最活跃的两个图册
        top2 = chart_weights.topk(2, dim=-1)
        chart_i = top2.indices[:, 0]  # (batch,)
        chart_j = top2.indices[:, 1]  # (batch,)
        
        # 计算闭环误差: q → T_ij(q) → T_ji(T_ij(q)) ≈ q
        # 简化：取第一个样本的图册
        if batch_size < 2:
            return {
                'consistency_loss': torch.tensor(0.0, device=q.device),
                'chart_entropy': self.compute_chart_entropy(chart_weights)
            }
        
        # 构建闭环: 选择一对图册进行测试
        # 使用第一个样本的top2图册
        i, j = chart_i[0].item(), chart_j[0].item()
        
        if i == j:
            j = (i + 1) % self.config.num_charts
        
        # 获取对应的转移映射
        map_idx_ij = i * (self.config.num_charts - 1) + (j if j < i else j - 1)
        map_idx_ji = j * (self.config.num_charts - 1) + (i if i < j else i - 1)
        
        map_idx_ij = min(map_idx_ij, len(self.transition_maps) - 1)
        map_idx_ji = min(map_idx_ji, len(self.transition_maps) - 1)
        
        T_ij = self.transition_maps[map_idx_ij]
        T_ji = self.transition_maps[map_idx_ji]
        
        # 闭环: q → T_ij(q) → T_ji(T_ij(q))
        q_in_j = T_ij(q)
        q_back = T_ji(q_in_j)
        
        # 闭环误差
        loop_error = (q_back - q).pow(2).mean()
        
        # 图册熵
        chart_entropy = self.compute_chart_entropy(chart_weights)
        
        return {
            'consistency_loss': loop_error,
            'chart_entropy': chart_entropy,
            'chart_weights': chart_weights
        }
    
    def compute_connection_orthogonality(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        计算联络正交性损失
        
        理论要求：平行移动应保持向量范数
        ||transport(v)|| ≈ ||v|| 对于正交联络
        """
        # 创建微小位移的目标点
        q_to = q + 0.1 * torch.randn_like(q)
        
        # 计算平行移动后的向量
        v_transported = self.connection(q, q_to, v)  # (batch, dim_q)
        
        # 正交性损失: 范数应保持
        norm_before = v.norm(dim=-1)
        norm_after = v_transported.norm(dim=-1)
        orthogonal_loss = (norm_after - norm_before).pow(2).mean()
        
        return orthogonal_loss
    
    def train_step(self) -> Dict[str, torch.Tensor]:
        """单步训练"""
        # 初始化状态
        state = self.random_init_state(self.config.batch_size)
        
        # 重置entity内部状态
        self.entity.reset(batch_size=self.config.batch_size)
        self.entity.state = state  # 设置初始状态
        
        q = state.q
        v = state.p  # 使用p作为速度向量
        
        # 1. 图册一致性损失
        consistency_results = self.compute_consistency_loss(q)
        L_consistency = consistency_results['consistency_loss']
        chart_entropy = consistency_results['chart_entropy']
        
        # 2. 联络正交性损失
        L_orthogonal = self.compute_connection_orthogonality(q, v)
        
        # 3. 图册熵损失（鼓励使用多个图册）
        # 熵越大越好，所以取负
        L_entropy = -chart_entropy
        
        # 4. 离线稳定性损失（保持阶段1学到的性质）
        # 让系统演化一步，检查F是否增加
        try:
            z = self.entity.z.expand(self.config.batch_size, -1)
            H_c_before = self.entity.internal_gen(state, z).detach()
            
            # 演化一步
            result = self.entity.step(u_dict=None, dt=0.1)
            new_state = self.entity.state  # 使用更新后的内部状态
            
            H_c_after = self.entity.internal_gen(new_state, z)
            
            L_stability = torch.relu(H_c_after - H_c_before).mean()
            
            # 数值稳定性检查
            if torch.isnan(L_stability) or torch.isinf(L_stability):
                L_stability = torch.tensor(0.0, device=q.device)
        except Exception:
            L_stability = torch.tensor(0.0, device=q.device)
        
        # 数值稳定性检查
        def safe_tensor(t, default=0.0):
            if torch.isnan(t).any() or torch.isinf(t).any():
                return torch.tensor(default, device=t.device)
            return t
        
        L_consistency = safe_tensor(L_consistency)
        L_orthogonal = safe_tensor(L_orthogonal)
        L_entropy = safe_tensor(L_entropy)
        chart_entropy = safe_tensor(chart_entropy)
        L_stability = safe_tensor(L_stability)
        
        # 总损失
        loss = (
            self.config.lambda_consistency * L_consistency +
            self.config.lambda_orthogonal * L_orthogonal +
            self.config.lambda_entropy * L_entropy +
            self.config.lambda_stability * L_stability
        )
        
        return {
            'loss': loss,
            'consistency_loss': L_consistency,
            'orthogonal_loss': L_orthogonal,
            'chart_entropy': chart_entropy,
            'stability_loss': L_stability,
        }


def train_phase2(args):
    """阶段2训练主循环"""
    
    print("=" * 70)
    print("阶段2：图册联络训练")
    print("=" * 70)
    print("\n理论目标：")
    print("  - L2: 李代数胚联络 / 平行移动")
    print("  - 多图册覆盖状态空间")
    print("  - 图册间转移一致性")
    print("  - 联络平行移动正交性")
    print("\n成功标准：")
    print("  - 闭环误差 < 0.1")
    print("  - 联络正交化损失 < 0.01")
    print("  - 图册熵 > 0.5")
    print("=" * 70)
    
    config = AtlasTrainingConfig(
        dim_q=args.dim_q,
        dim_z=args.dim_z,
        num_charts=args.num_charts,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    print(f"\n配置：")
    print(f"  dim_q: {config.dim_q}")
    print(f"  num_charts: {config.num_charts}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  device: {config.device}")
    
    trainer = AtlasTrainer(config, checkpoint_path=args.checkpoint)
    trainer = trainer.to(config.device)
    
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps_per_epoch
    )
    
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, 'training_log.jsonl')
    
    best_consistency = float('inf')
    
    print(f"\n开始训练... (epochs: {args.epochs})")
    print("-" * 70)
    
    for epoch in range(args.epochs):
        epoch_stats = {
            'loss': 0, 'consistency': 0, 'orthogonal': 0,
            'entropy': 0, 'stability': 0
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
            
            epoch_stats['loss'] += loss.item()
            epoch_stats['consistency'] += diagnostics['consistency_loss'].item()
            epoch_stats['orthogonal'] += diagnostics['orthogonal_loss'].item()
            epoch_stats['entropy'] += diagnostics['chart_entropy'].item()
            epoch_stats['stability'] += diagnostics['stability_loss'].item()
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.3f}",
                'Cons': f"{diagnostics['consistency_loss'].item():.4f}",
                'Ent': f"{diagnostics['chart_entropy'].item():.2f}",
            })
        
        n = args.steps_per_epoch
        for k in epoch_stats:
            epoch_stats[k] /= n
        
        print(f"\nEpoch {epoch+1} 总结：")
        print(f"  损失: {epoch_stats['loss']:.4f}")
        print(f"  闭环误差: {epoch_stats['consistency']:.4f}")
        print(f"  联络正交性: {epoch_stats['orthogonal']:.6f}")
        print(f"  图册熵: {epoch_stats['entropy']:.3f}")
        print(f"  稳定性损失: {epoch_stats['stability']:.4f}")
        
        log_entry = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            **epoch_stats
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        if epoch_stats['consistency'] < best_consistency:
            best_consistency = epoch_stats['consistency']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'consistency_loss': best_consistency,
                'config': config,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  ★ 新最佳模型! 闭环误差: {best_consistency:.4f}")
        
        # 检查成功标准（至少训练10个epoch）
        min_epochs = 10
        if epoch >= min_epochs - 1:
            if (epoch_stats['consistency'] < 0.1 and 
                epoch_stats['orthogonal'] < 0.01 and 
                epoch_stats['entropy'] > 0.5):
                print("\n" + "=" * 70)
                print("✓ 达到成功标准！阶段2训练完成")
                print("=" * 70)
                break
        
        print("-" * 70)
    
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': trainer.state_dict(),
        'final_consistency': epoch_stats['consistency'],
        'config': config,
    }, os.path.join(args.save_dir, 'final_model.pt'))
    
    print("\n" + "=" * 70)
    print("阶段2训练完成")
    print(f"最佳闭环误差: {best_consistency:.4f}")
    print(f"模型保存至: {args.save_dir}")
    print("=" * 70)
    
    return best_consistency < 0.1


def main():
    parser = argparse.ArgumentParser(description='阶段2：图册联络训练')
    
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--dim_z', type=int, default=16)
    parser.add_argument('--num_charts', type=int, default=4)
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='阶段1的checkpoint路径')
    parser.add_argument('--save_dir', type=str, default='checkpoints/phase2')
    
    args = parser.parse_args()
    
    success = train_phase2(args)
    
    if success:
        print("\n下一步：运行阶段3训练")
        print("python HEI/training/run_phase3_interface.py --checkpoint checkpoints/phase2/best_model.pt")
    else:
        print("\n阶段2未达成功标准，建议：")
        print("  1. 增加训练轮数 --epochs 100")
        print("  2. 调整图册数量 --num_charts 8")


if __name__ == "__main__":
    main()

