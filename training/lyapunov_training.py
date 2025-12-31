"""
基于Lyapunov的新训练范式

理论依据（自监督.md）：
"若神经动力学可表为对变分自由能的梯度下降，则自由能是该动力学的 Lyapunov 函数"

核心思想：
1. 让系统学会"自己降低F"，而不是让优化器降低F
2. 动力学演化应该内在地驱动F下降
3. 语言只是接口，不应主导动力学结构

阶段1：离线动力学训练 - 学习有意义的吸引结构
阶段2：图册联络训练 - 学习跨区域一致性
阶段3：接口对齐训练 - 在稳定动力学上加语言
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

from he_core.soul_entity import SoulEntity, create_soul_entity
from he_core.state import ContactState


@dataclass
class LyapunovTrainingConfig:
    """Lyapunov训练配置"""
    # 状态空间
    dim_q: int = 64
    dim_z: int = 16
    
    # 演化参数
    T_offline: int = 50      # 离线演化步数（足够长以观察收敛）
    T_online: int = 20       # 在线演化步数
    dt: float = 0.1          # 时间步长
    
    # Lyapunov约束
    lyapunov_margin: float = 0.01  # F[t+1] < F[t] - margin
    
    # 损失权重
    lambda_lyapunov: float = 10.0  # Lyapunov约束最重要
    lambda_converge: float = 1.0   # 收敛约束
    lambda_diversity: float = 0.1  # 多样性约束
    
    # 训练参数
    batch_size: int = 32
    learning_rate: float = 1e-4
    device: str = 'cuda'


class LyapunovLoss(nn.Module):
    """
    Lyapunov损失：确保F在动力学演化下单调下降
    
    理论依据：
    - F应该是接触动力学的Lyapunov函数
    - dF/dt ≤ -margin < 0
    """
    
    def __init__(self, margin: float = 0.01):
        super().__init__()
        self.margin = margin
    
    def forward(self, F_trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            F_trajectory: (batch, T) 演化过程中的F值序列
            
        Returns:
            losses: 包含各种Lyapunov相关损失
        """
        batch_size, T = F_trajectory.shape
        device = F_trajectory.device
        
        # 1. 单调下降约束：F[t+1] < F[t] - margin
        # violations = relu(F[t+1] - F[t] + margin)
        F_diff = F_trajectory[:, 1:] - F_trajectory[:, :-1]  # (batch, T-1)
        violations = torch.relu(F_diff + self.margin)
        L_monotonic = violations.mean()
        
        # 2. 下降率统计
        mean_decrease = -F_diff.mean()  # 正值表示在下降
        violation_rate = (F_diff > 0).float().mean()  # 违反比例
        
        # 3. 收敛损失：最终F应该尽量低
        L_final = F_trajectory[:, -1].mean()
        
        # 4. 收敛速度：前期下降应该快，后期趋于稳定
        # 早期下降
        early_decrease = (F_trajectory[:, 0] - F_trajectory[:, T//2]).mean()
        # 后期稳定
        late_variance = F_trajectory[:, T//2:].var(dim=1).mean()
        L_convergence_quality = -early_decrease + late_variance
        
        return {
            'L_monotonic': L_monotonic,
            'L_final': L_final,
            'L_convergence_quality': L_convergence_quality,
            'mean_decrease': mean_decrease,
            'violation_rate': violation_rate,
            'F_initial': F_trajectory[:, 0].mean(),
            'F_final': F_trajectory[:, -1].mean(),
        }


class AttractorDiversityLoss(nn.Module):
    """
    吸引子多样性损失：不同初始状态应收敛到不同吸引子
    
    理论依据：
    - 系统应有多个稳定吸引子（对应不同技能/概念区域）
    - 不应全部collapse到单一点
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, final_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            final_states: (batch, dim_q) 演化终态
            
        Returns:
            losses: 多样性相关损失
        """
        batch_size = final_states.shape[0]
        
        # 1. 计算终态之间的距离矩阵
        # dist[i,j] = ||q_i - q_j||
        diff = final_states.unsqueeze(1) - final_states.unsqueeze(0)  # (B, B, D)
        dist_matrix = diff.norm(dim=-1)  # (B, B)
        
        # 2. 多样性：鼓励终态之间有足够距离
        # 排除对角线
        mask = 1 - torch.eye(batch_size, device=final_states.device)
        mean_dist = (dist_matrix * mask).sum() / mask.sum()
        
        # 多样性损失：距离越大越好（取负）
        L_diversity = -mean_dist
        
        # 3. 估计吸引子数量（聚类）
        # 使用简单的距离阈值
        threshold = mean_dist * 0.5
        close_pairs = (dist_matrix < threshold).float() * mask
        avg_neighbors = close_pairs.sum(dim=1).mean()
        estimated_clusters = batch_size / (1 + avg_neighbors)
        
        return {
            'L_diversity': L_diversity,
            'mean_distance': mean_dist,
            'estimated_attractors': estimated_clusters,
        }


class OfflineDynamicsTrainer(nn.Module):
    """
    阶段1：离线动力学训练器
    
    目标：学习H^c的参数，使接触动力学能自发降低F
    
    训练不涉及任何外部输入，纯粹验证A1/A2公理
    """
    
    def __init__(self, config: LyapunovTrainingConfig):
        super().__init__()
        self.config = config
        
        # 创建SoulEntity
        entity_config = {
            'dim_q': config.dim_q,
            'dim_z': config.dim_z,
            'stiffness': 0.1,
            'contact_stiffness': 0.1,
            'hyperbolic_c': 0.1,
        }
        self.entity = create_soul_entity(entity_config)
        
        # Lyapunov损失
        self.lyapunov_loss = LyapunovLoss(margin=config.lyapunov_margin)
        self.diversity_loss = AttractorDiversityLoss()
        
    def random_init_state(self, batch_size: int) -> ContactState:
        """随机初始化状态"""
        device = self.config.device
        dim_q = self.config.dim_q
        
        # 创建ContactState
        state = ContactState(dim_q=dim_q, batch_size=batch_size, device=device)
        
        # 随机初始化
        # q: 均匀分布在有界区域
        state.q = torch.randn(batch_size, dim_q, device=device) * 2.0
        
        # p: 初始动量较小
        state.p = torch.randn(batch_size, dim_q, device=device) * 0.1
        
        # s: 初始能量账本
        state.s = torch.zeros(batch_size, 1, device=device)
        
        return state
    
    def compute_offline_free_energy(self, state: ContactState) -> torch.Tensor:
        """
        计算离线自由能
        
        离线阶段没有预测误差，F由内在能量决定：
        F = H^c (接触哈密顿量)
        
        这符合理论：离线态的自由能应该由内在动力学状态决定
        """
        # 计算H^c
        z = self.entity.z.expand(state.batch_size, -1)
        H_c = self.entity.internal_gen(state, z)  # (batch, 1)
        
        # KL正则化（自主上下文z的先验约束）
        kl = 0.5 * (z ** 2).sum(dim=1, keepdim=True)  # 简化的KL
        
        F = H_c + 0.01 * kl
        return F.squeeze(-1)  # (batch,)
    
    def evolve_offline(self, init_state: ContactState) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        离线演化：无外部输入，纯接触动力学
        
        训练策略：
        1. 演化过程用no_grad（避免就地操作问题）
        2. 记录轨迹的关键点
        3. 用记录的点重新计算F（带梯度）
        
        Returns:
            trajectory_q: q状态轨迹
            F_trajectory: F值序列
        """
        batch_size = init_state.batch_size
        device = init_state.q.device
        
        # 收集轨迹点（用于后续带梯度的计算）
        trajectory_q = [init_state.q.detach().clone()]
        trajectory_p = [init_state.p.detach().clone()]
        trajectory_s = [init_state.s.detach().clone()]
        
        # 演化过程不需要梯度
        with torch.no_grad():
            self.entity.state = ContactState(
                dim_q=self.config.dim_q, 
                batch_size=batch_size, 
                device=str(device)
            )
            self.entity.state.q = init_state.q.clone()
            self.entity.state.p = init_state.p.clone()
            self.entity.state.s = init_state.s.clone()
            
            for t in range(self.config.T_offline):
                result = self.entity.step(
                    u_dict=None,
                    dt=self.config.dt
                )
                
                trajectory_q.append(self.entity.state.q.clone())
                trajectory_p.append(self.entity.state.p.clone())
                trajectory_s.append(self.entity.state.s.clone())
        
        # 现在用收集的轨迹点计算F（带梯度）
        # 这里的梯度流向的是H^c的参数
        F_values = []
        for t in range(len(trajectory_q)):
            # 创建带梯度的state
            state_t = ContactState(
                dim_q=self.config.dim_q,
                batch_size=batch_size,
                device=str(device)
            )
            state_t.q = trajectory_q[t].requires_grad_(True)
            state_t.p = trajectory_p[t].requires_grad_(True)
            state_t.s = trajectory_s[t].requires_grad_(True)
            
            F_t = self.compute_offline_free_energy(state_t)
            F_values.append(F_t)
        
        F_trajectory = torch.stack(F_values, dim=1)  # (batch, T+1)
        
        return trajectory_q, F_trajectory
    
    def train_step(self) -> Dict[str, torch.Tensor]:
        """单步训练"""
        batch_size = self.config.batch_size
        
        # 1. 随机初始化状态
        init_state = self.random_init_state(batch_size)
        
        # 2. 离线演化
        trajectory_q, F_trajectory = self.evolve_offline(init_state)
        
        # 3. 计算Lyapunov损失
        lyap_losses = self.lyapunov_loss(F_trajectory)
        
        # 4. 计算多样性损失
        final_q = trajectory_q[-1]  # 已经是tensor
        div_losses = self.diversity_loss(final_q)
        
        # 5. 总损失
        L_total = (
            self.config.lambda_lyapunov * lyap_losses['L_monotonic'] +
            self.config.lambda_converge * lyap_losses['L_final'] +
            self.config.lambda_diversity * div_losses['L_diversity']
        )
        
        # 汇总诊断信息
        diagnostics = {
            'loss': L_total,
            **{f'lyap_{k}': v for k, v in lyap_losses.items()},
            **{f'div_{k}': v for k, v in div_losses.items()},
            'trajectory_length': len(trajectory_q),
        }
        
        return diagnostics


def test_offline_dynamics():
    """测试离线动力学训练"""
    print("=" * 60)
    print("阶段1测试：离线动力学训练")
    print("=" * 60)
    print("\n理论目标：")
    print("  - A1: 系统在无输入时能自演化")
    print("  - A2: 演化应收敛到有意义的吸引子")
    print("  - Lyapunov: F应在演化过程中单调下降")
    print()
    
    config = LyapunovTrainingConfig(
        dim_q=32,
        dim_z=8,
        T_offline=20,
        batch_size=16,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    trainer = OfflineDynamicsTrainer(config)
    trainer = trainer.to(config.device)
    
    optimizer = torch.optim.Adam(trainer.parameters(), lr=config.learning_rate)
    
    print(f"设备: {config.device}")
    print(f"演化步数: {config.T_offline}")
    print(f"批量大小: {config.batch_size}")
    print()
    
    # 训练几步
    print("开始训练...")
    for step in range(10):
        optimizer.zero_grad()
        diagnostics = trainer.train_step()
        diagnostics['loss'].backward()
        optimizer.step()
        
        if step % 2 == 0:
            print(f"\nStep {step}:")
            print(f"  总损失: {diagnostics['loss'].item():.4f}")
            print(f"  Lyapunov违反率: {diagnostics['lyap_violation_rate'].item():.2%}")
            print(f"  F下降量: {diagnostics['lyap_mean_decrease'].item():.4f}")
            print(f"  初始F: {diagnostics['lyap_F_initial'].item():.4f}")
            print(f"  最终F: {diagnostics['lyap_F_final'].item():.4f}")
            print(f"  终态多样性: {diagnostics['div_mean_distance'].item():.4f}")
            print(f"  估计吸引子数: {diagnostics['div_estimated_attractors'].item():.1f}")
    
    print("\n" + "=" * 60)
    print("✓ 离线动力学测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_offline_dynamics()

