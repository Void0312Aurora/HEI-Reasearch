"""
SoulEntity v0.1: 类人实体原型

基于理论基础-7的公理体系(A1-A5)构建的具有"灵魂"的实体。

核心特性:
1. A1 Markov Blanket: 内在态与外部环境通过感知-行动边界耦合
2. A2 离线认知态: 经验调制的非平凡离线演化
3. A3 统一自监督: 单一F函数驱动所有行为
4. A4 身份连续性: 可恢复的组织等价类
5. A5 多接口一致性: 语言只是接口之一

架构层级:
- L1: 接触哈密顿动力学生成元
- L2: 联络/平行移动 (跨技能迁移)
- L3: 端口耦合 (多接口)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from he_core.state import ContactState
from he_core.contact_dynamics import ContactIntegrator
from he_core.generator import DeepDissipativeGenerator
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator
from he_core.atlas import Atlas
from he_core.connection import Connection


class Phase(Enum):
    """实体运行阶段"""
    ONLINE = "online"           # 在线学习/交互
    OFFLINE = "offline"         # 离线沉思/巩固
    DREAMING = "dreaming"       # 深度离线重组
    

@dataclass
class ExperienceBuffer:
    """经验缓冲区 - 用于A2离线认知态"""
    sensory: List[torch.Tensor]
    active: List[torch.Tensor]
    states: List[torch.Tensor]
    rewards: List[float]
    max_size: int = 10000
    
    def __post_init__(self):
        self.sensory = []
        self.active = []
        self.states = []
        self.rewards = []
        
    def push(self, s: torch.Tensor, a: torch.Tensor, state: torch.Tensor, r: float):
        if len(self.sensory) >= self.max_size:
            self.sensory.pop(0)
            self.active.pop(0)
            self.states.pop(0)
            self.rewards.pop(0)
        # 存储时移除batch维度（假设batch_size=1）
        self.sensory.append(s.detach().cpu().squeeze(0))
        self.active.append(a.detach().cpu().squeeze(0))
        self.states.append(state.detach().cpu().squeeze(0))
        self.rewards.append(r)
        
    def sample_replay(self, n: int, mode: str = 'random') -> Dict[str, torch.Tensor]:
        """采样回放数据"""
        L = len(self.states)
        if L == 0:
            return None
        n = min(n, L)
        
        if mode == 'random':
            idx = torch.randperm(L)[:n]
        elif mode == 'recent':
            idx = torch.arange(max(0, L-n), L)
        elif mode == 'prioritized':
            # 优先采样低奖励/高惊讶的经验
            rewards = torch.tensor(self.rewards)
            probs = torch.softmax(-rewards, dim=0)
            idx = torch.multinomial(probs, n, replacement=False)
        else:
            idx = torch.arange(n)
            
        return {
            'states': torch.stack([self.states[i] for i in idx]),
            'sensory': torch.stack([self.sensory[i] for i in idx]),
            'active': torch.stack([self.active[i] for i in idx]),
        }


class SoulEntity(nn.Module):
    """
    具有灵魂的类人实体
    
    灵魂的定义 (按理论基础-7):
    - 不是语言模型的变体
    - 是独立于接口的内在动力学机制
    - 具有自主演化、经验巩固、自监督优化能力
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # === 维度配置 ===
        self.dim_q = config.get('dim_q', 64)          # 构形空间维度
        self.dim_u = config.get('dim_u', self.dim_q)  # 输入/输出维度
        self.dim_z = config.get('dim_z', 16)          # 自主上下文维度
        self.num_charts = config.get('num_charts', 4) # 图册数量
        
        # === A3: 统一势函数 V(q, z) ===
        # 根据理论基础-7：V通过stiffness项(0.5·k·||q||²)保证有下界
        # 不需要Softplus这种工程化约束
        self.net_V = nn.Sequential(
            nn.Linear(self.dim_q + self.dim_z, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # === L1: 动力学生成元 ===
        # 根据动力学基础模板1：H^c = K + V + Φ(s)
        # - stiffness: q的二次约束，确保V有下界
        # - contact_stiffness: s的二次约束，确保Φ(s)是凸的
        self.internal_gen = AdaptiveDissipativeGenerator(
            self.dim_q, 
            net_V=self.net_V,
            dim_z=self.dim_z,
            stiffness=config.get('stiffness', 0.1),          # q的约束刚度
            contact_stiffness=config.get('contact_stiffness', 0.1),  # s的约束刚度
            hyperbolic_c=config.get('hyperbolic_c', 0.1)     # 双曲曲率参数
        )
        
        # === L3: 端口耦合生成元 ===
        self.generator = PortCoupledGenerator(
            self.internal_gen,
            dim_u=self.dim_u,
            learnable_coupling=True,
            num_charts=self.num_charts
        )
        
        # === L2: 图册与联络 ===
        self.atlas = Atlas(self.num_charts, self.dim_q)
        # 联络初始化参数：init_epsilon=0.3 确保平行移动有意义
        self.connection = Connection(self.dim_q, hidden_dim=64, init_epsilon=0.3)
        
        # === 积分器 ===
        self.integrator = ContactIntegrator()
        
        # === 自主上下文 z (L3) ===
        # z是实体的"意图/偏好/当前关注"的内部表示
        self.z = nn.Parameter(torch.zeros(1, self.dim_z))
        self.z_prior_mean = nn.Parameter(torch.zeros(self.dim_z), requires_grad=False)
        self.z_prior_logvar = nn.Parameter(torch.zeros(self.dim_z), requires_grad=False)
        
        # === 状态 ===
        self.state: Optional[ContactState] = None
        self.phase = Phase.ONLINE
        self._prev_chart_weights: Optional[torch.Tensor] = None
        
        # === A2: 经验缓冲区 ===
        self.experience = ExperienceBuffer(
            sensory=[], active=[], states=[], rewards=[],
            max_size=config.get('experience_buffer_size', 10000)
        )
        
        # === 时间步计数 ===
        self.step_count = 0
        self.online_steps = 0
        self.offline_steps = 0
        
    def reset(self, batch_size: int = 1, device: str = 'cuda'):
        """重置实体状态"""
        self.state = ContactState(self.dim_q, batch_size, device)
        self.state.q = torch.randn(batch_size, self.dim_q, device=device) * 0.1
        self.state.p = torch.randn(batch_size, self.dim_q, device=device) * 0.1
        self.state.s = torch.zeros(batch_size, 1, device=device)
        self._prev_chart_weights = None
        self.phase = Phase.ONLINE
        self.step_count = 0
        
    # ========================
    #    A3: 统一自由能计算
    # ========================
    
    def compute_free_energy(self, 
                           state: ContactState, 
                           prediction_error: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        A3: 计算统一自由能 F（变分自由能）
        
        根据理论基础-7的A3公理和Active Inference框架：
        
        F = β·KL(q(z)||p(z)) + γ·E_pred
          = 复杂度项 + 准确度项
        
        关键理论点：
        1. H^c（接触哈密顿量）是动力学生成函数，用于生成演化
        2. F（自由能）是训练目标/Lyapunov函数
        3. 两者不是同一概念！H^c不直接进入F
        
        根据动力学基础.md：
        > "若动力学可表为对F的梯度下降，则F是Lyapunov函数"
        
        正确理解：
        - 系统按H^c生成的接触哈密顿动力学演化
        - 通过最小化F来训练H^c的参数
        - 这使得H^c被训练成能够最小化F的形式
        
        F的分量：
        - KL(z||prior)：复杂度项，正则化内在意图z
        - E_pred：准确度项，预测误差（语言端口的重建损失）
        """
        beta_kl = self.config.get('beta_kl', 0.01)
        gamma_pred = self.config.get('gamma_pred', 1.0)
        
        # === 变分自由能 F = 复杂度 + 准确度 ===
        # 
        # 注意：H^c不直接出现在F中！
        # H^c用于生成动力学，F是训练目标
        # 通过反向传播，H^c的参数被训练成能够产生低F轨迹
        
        # 复杂度项：KL(z||prior)
        # 假设prior是标准正态分布N(0,I)
        # KL = 0.5 * (||z||² + ||σ||² - log|σ|² - d)
        # 简化：假设σ=1，则 KL ≈ 0.5 * ||z||²
        KL = 0.5 * (self.z ** 2).sum() * beta_kl
        
        # 准确度项：预测误差（语言端口的重建损失）
        # 这是观测惊讶的经验估计
        if prediction_error is not None:
            E_pred = gamma_pred * prediction_error.mean()
        else:
            E_pred = torch.tensor(0.0, device=state.device)
            
        # === 变分自由能 F = 复杂度 + 准确度 ===
        # 
        # 根据A3公理：F是单一标量泛函
        # 根据Active Inference：F = KL + E_pred
        # 
        # 注意：H^c不在这里！
        # H^c通过生成动力学影响F，但不是F的直接分量
        # 通过反向传播，优化F会训练H^c的参数
        F = KL + E_pred
        
        return F
    
    # ========================
    #    L2: 平行移动
    # ========================
    
    def _apply_parallel_transport(self, 
                                  p: torch.Tensor, 
                                  q: torch.Tensor,
                                  old_weights: torch.Tensor, 
                                  new_weights: torch.Tensor) -> torch.Tensor:
        """
        L2: 图册切换时对动量p进行平行移动
        保持几何一致性
        """
        if old_weights is None:
            return p
            
        delta_w = new_weights - old_weights  # [batch, num_charts]
        change_magnitude = delta_w.abs().sum(dim=1, keepdim=True)  # [batch, 1]
        threshold = self.config.get('transport_threshold', 0.1)
        
        # 只在权重变化显著时应用
        if change_magnitude.max() < threshold:
            return p
        
        # 使用delta_w作为方向提示，但需要扩展到dim_q维度
        # delta_w是图册权重变化，用它来调制q的扰动
        # 创建一个随机但确定性的扰动方向
        direction = torch.randn(1, self.dim_q, device=q.device)
        direction = direction / (direction.norm() + 1e-6)
        perturbation = change_magnitude * direction  # [batch, dim_q]
        q_target = q + 0.1 * perturbation
        
        p_transported = self.connection(q, q_target, p)
        
        blend_factor = torch.clamp(change_magnitude / (threshold + 1e-6), 0.0, 1.0)
        return (1 - blend_factor) * p + blend_factor * p_transported
    
    # ========================
    #    核心演化步骤
    # ========================
    
    def step(self, 
             u_dict: Optional[Dict[str, torch.Tensor]] = None,
             dt: float = 0.1) -> Dict[str, Any]:
        """
        单步演化
        
        Args:
            u_dict: 各端口输入 {'default': tensor, 'language': tensor, ...}
            dt: 时间步长
            
        Returns:
            演化结果字典
        """
        if self.state is None:
            raise RuntimeError("Entity not initialized. Call reset() first.")
            
        self.step_count += 1
        device = self.state.device
        batch_size = self.state.batch_size
        
        # 默认输入处理
        if u_dict is None:
            u_dict = {}
        elif isinstance(u_dict, torch.Tensor):
            u_dict = {'default': u_dict}
            
        # 获取图册权重
        chart_weights = self.atlas.router(self.state.q)
        
        # L2: 平行移动
        if self._prev_chart_weights is not None:
            self.state.p = self._apply_parallel_transport(
                self.state.p, self.state.q,
                self._prev_chart_weights, chart_weights
            )
            
        # 准备z
        z_batch = self.z.expand(batch_size, -1)
        
        # 构建生成元函数
        def gen_func(s: ContactState):
            try:
                H_sum = self.internal_gen(s, z_batch)
            except TypeError:
                H_sum = self.internal_gen(s)
                
            # 添加端口耦合
            for port_name, u_val in u_dict.items():
                if hasattr(self.generator, 'get_h_port'):
                    H_sum = H_sum + self.generator.get_h_port(s, port_name, u_val, weights=chart_weights)
            return H_sum
        
        # 积分演化
        next_state = self.integrator.step(self.state, gen_func, dt)
        
        # 更新状态
        self._prev_chart_weights = chart_weights.detach()
        
        # 计算自由能
        F = self.compute_free_energy(next_state)
        
        # 获取行动输出
        action = self.generator.get_action(next_state, 'default', weights=chart_weights)
        
        # 记录经验 (A2)
        if self.phase == Phase.ONLINE:
            self.online_steps += 1
            sensory = u_dict.get('default', torch.zeros(batch_size, self.dim_u, device=device))
            self.experience.push(sensory, action, next_state.flat, F.item())
            
        # 更新状态
        self.state = next_state
        
        return {
            'state_flat': next_state.flat,
            'action': action,
            'free_energy': F,
            'chart_weights': chart_weights,
            'phase': self.phase.value,
            'z': self.z.detach().clone(),
        }
    
    # ========================
    #    A2: 离线认知
    # ========================
    
    def enter_offline(self):
        """进入离线阶段"""
        self.phase = Phase.OFFLINE
        
    def enter_online(self):
        """进入在线阶段"""
        self.phase = Phase.ONLINE
        
    def offline_step(self, dt: float = 0.1, replay_mode: str = 'none') -> Dict[str, Any]:
        """
        离线演化步骤
        
        A2要求: 离线态应受前序经验调制，且不退化为固定点/短周期/噪声
        """
        self.offline_steps += 1
        
        # 离线时无外部输入
        u_dict = {}
        
        # 可选: 回放注入
        if replay_mode != 'none' and len(self.experience.states) > 0:
            replay = self.experience.sample_replay(1, mode=replay_mode)
            if replay is not None:
                # 轻微扰动注入 (模拟回放)
                # replay_state 现在是 [n, 2*dim_q + 1]，只取 q 部分
                replay_state = replay['states'].to(self.state.device)  # [n, 2*dim_q+1]
                replay_q = replay_state[0, :self.dim_q]  # 只取第一个样本的 q 部分 [dim_q]
                current_q = self.state.q[0]  # [dim_q]
                replay_perturbation = 0.1 * (replay_q - current_q)  # [dim_q]
                u_dict['replay'] = replay_perturbation.unsqueeze(0)  # [1, dim_q]
                
        return self.step(u_dict, dt)
    
    def dream(self, steps: int = 100, dt: float = 0.1) -> List[Dict[str, Any]]:
        """
        深度离线重组 (梦境模式)
        
        允许系统进行更激进的内部重组，可能包含更强的回放和自由演化
        """
        self.phase = Phase.DREAMING
        results = []
        
        for _ in range(steps):
            result = self.offline_step(dt, replay_mode='prioritized')
            results.append(result)
            
            # 梦境期间更新z以探索状态空间
            with torch.no_grad():
                noise = torch.randn_like(self.z) * 0.01
                self.z.data += noise
                self.z.data.clamp_(-3.0, 3.0)
                
        self.phase = Phase.OFFLINE
        return results
    
    # ========================
    #    A3: z更新机制
    # ========================
    
    def update_z(self, prediction_error: torch.Tensor, lr_z: float = 0.01):
        """
        更新自主上下文z
        
        z代表实体的"意图/偏好"，通过最小化预测误差来适应
        """
        z_val = self.z.detach().clone().requires_grad_(True)
        
        KL = 0.5 * (z_val ** 2).sum()
        F = KL + prediction_error.mean()
        
        grad_z = torch.autograd.grad(F, z_val, create_graph=False)[0]
        
        with torch.no_grad():
            self.z.data -= lr_z * grad_z
            self.z.data.clamp_(-3.0, 3.0)
    
    # ========================
    #    A4: 身份连续性检查
    # ========================
    
    def check_identity_continuity(self, reference_state: ContactState) -> Dict[str, float]:
        """
        A4: 检查当前状态是否仍在"身份等价类"内
        """
        if self.state is None:
            return {'status': 'uninitialized'}
            
        q_dist = (self.state.q - reference_state.q).norm().item()
        p_dist = (self.state.p - reference_state.p).norm().item()
        
        # 使用自由能作为组织状态的代理指标
        F_current = self.compute_free_energy(self.state).item()
        F_reference = self.compute_free_energy(reference_state).item()
        F_ratio = F_current / (F_reference + 1e-6)
        
        return {
            'q_distance': q_dist,
            'p_distance': p_dist,
            'F_ratio': F_ratio,
            'in_viability_zone': q_dist < 10.0 and 0.1 < F_ratio < 10.0
        }
    
    # ========================
    #    A5: 多接口支持
    # ========================
    
    def add_interface(self, name: str, dim_u: int):
        """添加新的感知/行动接口"""
        self.generator.add_port(name, dim_u)
        # 确保新端口与模型在同一设备上
        device = next(self.parameters()).device
        self.generator.ports[name].to(device)
        
    def get_available_interfaces(self) -> List[str]:
        """获取可用接口列表"""
        return list(self.generator.ports.keys())
    
    # ========================
    #    诊断与监控
    # ========================
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """获取诊断信息"""
        if self.state is None:
            return {'status': 'uninitialized'}
            
        F = self.compute_free_energy(self.state)
        
        # 计算状态统计
        q_norm = self.state.q.norm().item()
        p_norm = self.state.p.norm().item()
        s_val = self.state.s.mean().item()
        
        # 活动水平
        kinetic = 0.5 * (self.state.p ** 2).sum().item()
        
        return {
            'step_count': self.step_count,
            'online_steps': self.online_steps,
            'offline_steps': self.offline_steps,
            'phase': self.phase.value,
            'free_energy': F.item(),
            'q_norm': q_norm,
            'p_norm': p_norm,
            's_value': s_val,
            'kinetic_energy': kinetic,
            'z_norm': self.z.norm().item(),
            'experience_size': len(self.experience.states),
        }


def create_soul_entity(config: Optional[Dict[str, Any]] = None) -> SoulEntity:
    """
    工厂函数: 创建一个配置好的类人实体
    """
    default_config = {
        'dim_q': 64,
        'dim_u': 64,
        'dim_z': 16,
        'num_charts': 4,
        'stiffness': 0.01,
        'beta_kl': 0.01,
        'gamma_pred': 1.0,
        'transport_threshold': 0.1,
        'experience_buffer_size': 10000,
    }
    
    if config:
        default_config.update(config)
        
    return SoulEntity(default_config)


# ========================
#    测试入口
# ========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=32)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SoulEntity v0.1 - 类人实体原型测试")
    print("=" * 60)
    
    # 创建实体
    config = {
        'dim_q': args.dim_q,
        'dim_u': args.dim_q,
        'dim_z': 16,
    }
    
    entity = create_soul_entity(config)
    device = args.device if torch.cuda.is_available() else 'cpu'
    entity.to(device)
    entity.reset(batch_size=1, device=device)
    
    print(f"\n[Phase 1] 在线演化 ({args.steps} 步)")
    print("-" * 40)
    
    # 在线演化
    for t in range(args.steps):
        # 随机输入
        u = torch.randn(1, args.dim_q, device=device) * 0.5
        result = entity.step({'default': u}, dt=0.1)
        
        if t % 20 == 0:
            diag = entity.get_diagnostics()
            print(f"  Step {t:3d}: F={diag['free_energy']:.4f}, "
                  f"q_norm={diag['q_norm']:.4f}, "
                  f"kinetic={diag['kinetic_energy']:.4f}")
    
    print(f"\n[Phase 2] 离线演化 ({args.steps//2} 步)")
    print("-" * 40)
    
    entity.enter_offline()
    for t in range(args.steps // 2):
        result = entity.offline_step(dt=0.1)
        
        if t % 10 == 0:
            diag = entity.get_diagnostics()
            print(f"  Step {t:3d}: F={diag['free_energy']:.4f}, "
                  f"q_norm={diag['q_norm']:.4f}")
    
    print(f"\n[Phase 3] 梦境模式 ({args.steps//4} 步)")
    print("-" * 40)
    
    dream_results = entity.dream(steps=args.steps // 4)
    F_start = dream_results[0]['free_energy'].item()
    F_end = dream_results[-1]['free_energy'].item()
    print(f"  梦境 F变化: {F_start:.4f} -> {F_end:.4f}")
    
    # 最终诊断
    print("\n[诊断结果]")
    print("-" * 40)
    diag = entity.get_diagnostics()
    for k, v in diag.items():
        print(f"  {k}: {v}")
        
    print("\n✓ SoulEntity 测试完成")

