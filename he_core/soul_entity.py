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
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import math

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
    """经验缓冲区 - 优化版 (Tensor-based)"""
    max_size: int = 10000
    dim_s: int = 0
    dim_a: int = 0
    dim_state: int = 0
    device: str = 'cpu' # 默认 CPU 以节省显存，但支持 GPU
    
    def __post_init__(self):
        self.ptr = 0
        self.size = 0
        self.sensory = None
        self.active = None
        self.states = None
        self.rewards = None

    def _init_buffers(self, s: torch.Tensor, a: torch.Tensor, state: torch.Tensor):
        self.dim_s = s.shape[1]
        self.dim_a = a.shape[1] if a is not None else 0
        self.dim_state = state.shape[1]
        device = s.device # Use the device of incoming data to avoid transfers if possible
        
        self.sensory = torch.zeros((self.max_size, self.dim_s), device=device)
        self.active = torch.zeros((self.max_size, self.dim_a), device=device) if self.dim_a > 0 else None
        self.states = torch.zeros((self.max_size, self.dim_state), device=device)
        self.rewards = torch.zeros((self.max_size,), device=device)
        self.device = device

    def push(self, s: torch.Tensor, a: torch.Tensor, state: torch.Tensor, r: float):
        if self.sensory is None:
            self._init_buffers(s, a, state)
            
        batch_size = s.shape[0]
        
        # 1. 准备数据
        r_tensor = torch.full((batch_size,), r, device=self.device)
        
        # 2. 写入 (环形缓冲区)
        indices = torch.arange(self.ptr, self.ptr + batch_size, device=self.device) % self.max_size
        
        self.sensory[indices] = s.detach().to(self.device)
        if self.active is not None and a is not None:
            self.active[indices] = a.detach().to(self.device)
        self.states[indices] = state.detach().to(self.device)
        self.rewards[indices] = r_tensor
        
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample_replay(self, n: int, mode: str = 'random') -> Dict[str, torch.Tensor]:
        if self.size == 0:
            return None
        n = min(n, self.size)
        
        if mode == 'random':
            idx = torch.randint(0, self.size, (n,), device=self.device)
        elif mode == 'recent':
            # 简化：随机采样
            idx = torch.randint(0, self.size, (n,), device=self.device)
        elif mode == 'prioritized':
            # 简化的 prioritized
            current_rewards = self.rewards[:self.size]
            probs = torch.softmax(-current_rewards, dim=0)
            idx = torch.multinomial(probs, n, replacement=True)
        else:
            idx = torch.arange(n, device=self.device)
            
        return {
            'states': self.states[idx],
            'sensory': self.sensory[idx],
            'active': self.active[idx] if self.active is not None else None,
        }


class LowerBoundedPotential(nn.Module):
    """
    Lower-bounded scalar potential V(x) >= 0.

    Theory alignment:
    - 理论基础-7/快慢变量/最小族.md 要求势能有下界，否则 A3 的 Lyapunov 语义会失效。
    - 仅靠 q 的 stiffness 项只能约束远场，无法阻止可学习势能出现“常数偏置漂移”并把 F 推向 -∞。
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.backbone(x)
        return raw.square()


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
        self.router_context_dim = int(config.get("router_context_dim", 0) or 0)
        
        # === A3: 统一势函数 V(q, z) ===
        # 理论基础-7/快慢变量/最小族.md 明确要求：V(q) 必须有下界（否则 A3 的 Lyapunov 语义失效）。
        # 仅靠 stiffness 项只能保证 ||q||→∞ 时趋于 +∞，但无法阻止网络把 V 整体平移到 -∞（常数偏置漂移）。
        # 这里用“下界化的可学习残差”实现 V >= 0（再叠加 stiffness 提供远场约束），避免训练中出现 F 向负无界发散。
        self.net_V = LowerBoundedPotential(self.dim_q + self.dim_z)
        
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
            alpha_min=float(config.get("alpha_min", 0.2)),
            alpha_max=float(config.get("alpha_max", 1.0)),
            hyperbolic_c=config.get('hyperbolic_c', 0.1)     # 双曲曲率参数
        )
        
        # === L3: 端口耦合生成元 ===
        self.generator = PortCoupledGenerator(
            self.internal_gen,
            dim_u=self.dim_u,
            learnable_coupling=True,
            num_charts=self.num_charts,
            top_k=int(config.get('port_top_k', 0) or 0),
            topk_impl=str(config.get("port_topk_impl", "grouped") or "grouped"),
        )
        # A2: 离线回放注入端口（经验调制）
        self.add_interface('replay', self.dim_q)
        
        # === L2: 图册与联络 ===
        self.atlas = Atlas(self.num_charts, self.dim_q, router_context_dim=self.router_context_dim)
        # 联络初始化参数：init_epsilon=0.3 确保平行移动有意义
        self.connection = Connection(
            self.dim_q,
            hidden_dim=int(config.get('connection_hidden_dim', 64) or 64),
            init_epsilon=float(config.get('connection_init_epsilon', 0.3) or 0.3),
            rank=int(config.get('connection_rank', 0) or 0),
        )
        
        # Cache config values for speed
        self._sanitize_nonfinite = bool(self.config.get("sanitize_nonfinite", True))
        self._strict_nonfinite = bool(self.config.get("strict_nonfinite", False))
        self._q_clip_norm = float(self.config.get("q_clip_norm", 0.0) or 0.0)
        self._p_clip_norm = float(self.config.get("p_clip_norm", 0.0) or 0.0)
        self._s_clip_abs = float(self.config.get("s_clip_abs", 0.0) or 0.0)
        transport_threshold = self.config.get("transport_threshold", 0.1)
        if transport_threshold is None:
            transport_threshold = 0.1
        # Important: allow explicit 0.0 to DISABLE transport (do not use `or` fallback here).
        self._transport_threshold = float(transport_threshold)
        # Router inertia timescale (seconds). If >0, chart weights evolve as a slow
        # relaxation toward the instantaneous router output:
        #   w̄_{t+1} = w̄_t + α (w_raw(q_t) - w̄_t),   α = dt/(τ+dt)
        self._router_tau = float(self.config.get("router_tau", 0.0) or 0.0)
        
        # === 积分器 ===
        integrator_method = config.get('integrator_method', 'euler')
        integrator_damping = float(config.get('damping', 0.1))
        integrator_substeps = int(config.get("integrator_substeps", 1) or 1)
        integrator_gamma_clip = float(config.get("integrator_gamma_clip", 0.0) or 0.0)
        self.integrator = ContactIntegrator(
            method=integrator_method,
            dim_q=self.dim_q,
            damping=integrator_damping,
            substeps=integrator_substeps,
            gamma_clip=integrator_gamma_clip,
        )
        
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
        
        F = V(q,z) + β·KL(q(z)||p(z)) + γ·E_pred
          = 势能项 + 复杂度项 + 准确度项
        
        关键理论点：
        1. H^c（接触哈密顿量）是动力学生成函数，用于生成演化
        2. F（自由能）是训练目标/Lyapunov函数
        3. 两者不是同一概念！F使用势能项作为内在一致性代理，而非完整H^c
        
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
        
        # === 变分自由能 F = 势能 + 复杂度 + 准确度 ===
        # 
        # 注意：F使用V(q,z)作为内在标量代理，而不是完整H^c！
        # H^c用于生成动力学，F是训练目标
        # 通过反向传播，H^c的参数被训练成能够产生低F轨迹
        
        # 势能项：V(q,z)（含刚度项，确保下界）
        z_batch = self.z.expand(state.batch_size, -1)
        V_inp = torch.cat([state.q, z_batch], dim=1)
        V = self.net_V(V_inp)
        if os.getenv("HEI_DEBUG_NAN", "0") == "1":
            if not torch.isfinite(V).all():
                v = V.detach()
                raise RuntimeError(
                    f"NaN/Inf in net_V output: shape={tuple(v.shape)} max_abs={float(v.abs().max().item())}"
                )
        stiffness = getattr(self.internal_gen, 'stiffness', 0.0)
        if stiffness > 0:
            V = V + 0.5 * stiffness * (state.q ** 2).sum(dim=1, keepdim=True)
        V_term = V.mean()

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
            # NOTE: `torch.tensor(0.0, device="cuda")` performs a host->device copy and
            # is not capturable under CUDA graph capture. Use a device-native zero.
            E_pred = V_term.new_zeros(())

        # === 变分自由能 F = 势能 + 复杂度 + 准确度 ===
        # 
        # 根据A3公理：F是单一标量泛函
        # 根据A3公理：F是单一标量泛函（这里选V+KL+E_pred作为可检验代理）
        # 
        # 注意：H^c不在这里！
        # H^c通过生成动力学影响F，但不是F的直接分量
        # 通过反向传播，优化F会训练H^c的参数
        F = V_term + KL + E_pred

        if os.getenv("HEI_DEBUG_NAN", "0") == "1":
            if not torch.isfinite(F).all():
                raise RuntimeError(
                    f"NaN/Inf in free_energy: V_term={float(V_term.detach().item())} KL={float(KL.detach().item())} "
                    f"E_pred={(float(E_pred.detach().item()) if torch.is_tensor(E_pred) else E_pred)}"
                )
        
        return F

    # ========================
    #    Functional step (no in-place)
    # ========================

    def forward_tensor(self,
                       state_flat: torch.Tensor,
                       u_dict: Optional[torch.Tensor | Dict[str, torch.Tensor]],
                       dt: float,
                       prev_chart_weights: Optional[torch.Tensor] = None,
                       prediction_error: Optional[torch.Tensor] = None,
                       detach_next_prev_weights: bool = True,
                       compute_action: bool = True,
                       strict_nonfinite_override: Optional[bool] = None,
                       sanitize_nonfinite_override: Optional[bool] = None,
                       return_finite_mask: bool = False,
                       skip_free_energy: bool = False,
                       router_context: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Functional (side-effect free) step for differentiable rollouts.

        Unlike `step()`, this method does not mutate `self.state` and avoids any
        in-place writes on `state_flat` views (critical for backprop through
        online→offline boundaries).
        """
        batch_size = state_flat.shape[0]
        device = state_flat.device
        strict_nonfinite = self._strict_nonfinite if strict_nonfinite_override is None else bool(strict_nonfinite_override)
        sanitize_nonfinite = self._sanitize_nonfinite if sanitize_nonfinite_override is None else bool(sanitize_nonfinite_override)
        finite_mask = torch.ones(batch_size, device=device, dtype=torch.bool)

        # Compatibility layer
        if u_dict is None:
            u_dict = {}
        elif isinstance(u_dict, torch.Tensor):
            u_dict = {'default': u_dict}

        s_in = ContactState(self.dim_q, batch_size, device, state_flat)

        if os.getenv("HEI_DEBUG_NAN", "0") == "1":
            if not torch.isfinite(state_flat).all():
                x = state_flat.detach()
                raise RuntimeError(
                    f"NaN/Inf entering forward_tensor: shape={tuple(x.shape)} max_abs={float(x.abs().max().item())}"
                )

        # Router input uses clone to prevent view-version conflicts.
        chart_weights_raw = self.atlas.router(s_in.q.clone(), context=router_context)

        finite_mask = finite_mask & torch.isfinite(chart_weights_raw).all(dim=1)
        if strict_nonfinite and (not sanitize_nonfinite):
            # Avoid per-step GPU→CPU sync: assert asynchronously on device.
            torch._assert_async(torch.isfinite(chart_weights_raw).all(), "NaN/Inf in chart_weights_raw")
        if sanitize_nonfinite:
            chart_weights_raw = torch.nan_to_num(chart_weights_raw, nan=0.0, posinf=0.0, neginf=0.0)
            row_sum = chart_weights_raw.sum(dim=1, keepdim=True)
            # If the router collapsed to all-zeros due to NaN cleanup, fall back to uniform.
            bad = row_sum <= 1e-8
            uniform = torch.full_like(chart_weights_raw, 1.0 / float(self.num_charts))
            chart_weights_raw = torch.where(bad.expand_as(chart_weights_raw), uniform, chart_weights_raw)
            chart_weights_raw = chart_weights_raw / chart_weights_raw.sum(dim=1, keepdim=True).clamp(min=1e-8)

        if os.getenv("HEI_DEBUG_NAN", "0") == "1":
            if not torch.isfinite(chart_weights_raw).all():
                w = chart_weights_raw.detach()
                raise RuntimeError(
                    f"NaN/Inf in chart_weights: shape={tuple(w.shape)} min={float(w.min().item())} max={float(w.max().item())}"
                )

        # Slow router state (w̄) as an endogenous carrier (fast–slow alignment).
        chart_weights = chart_weights_raw
        tau = self._router_tau
        if prev_chart_weights is not None and tau > 0.0:
            alpha = float(dt) / (tau + float(dt))
            # Blend in log-space (geometric mean) to avoid excessive overlap/mixing which
            # can destabilize port coupling while still providing inertia:
            #   w̄ ∝ (w_prev)^(1-α) · (w_raw)^α
            eps = 1e-8
            log_prev = torch.log(prev_chart_weights.clamp(min=eps))
            log_raw = torch.log(chart_weights_raw.clamp(min=eps))
            log_mix = (1.0 - alpha) * log_prev + alpha * log_raw
            w = torch.exp(log_mix)
            chart_weights = w / w.sum(dim=1, keepdim=True).clamp(min=eps)

        finite_mask = finite_mask & torch.isfinite(chart_weights).all(dim=1)
        if strict_nonfinite and (not sanitize_nonfinite):
            torch._assert_async(torch.isfinite(chart_weights).all(), "NaN/Inf in chart_weights")
        if sanitize_nonfinite:
            chart_weights = torch.nan_to_num(chart_weights, nan=0.0, posinf=0.0, neginf=0.0)
            row_sum = chart_weights.sum(dim=1, keepdim=True)
            bad = row_sum <= 1e-8
            uniform = torch.full_like(chart_weights, 1.0 / float(self.num_charts))
            chart_weights = torch.where(bad.expand_as(chart_weights), uniform, chart_weights)
            chart_weights = chart_weights / chart_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # L2: parallel transport (functional, no in-place on s_in)
        if prev_chart_weights is not None and self._transport_threshold > 0.0:
            p_transported = self._apply_parallel_transport(
                s_in.p, s_in.q,
                prev_chart_weights, chart_weights
            )
            state_flat = torch.cat([s_in.q, p_transported, s_in.s], dim=1)
            s_in = ContactState(self.dim_q, batch_size, device, state_flat)

        # Prepare z
        z_batch = self.z.expand(batch_size, -1)

        def gen_func(s: ContactState):
            try:
                H_sum = self.internal_gen(s, z_batch)
            except TypeError:
                H_sum = self.internal_gen(s)

            for port_name, u_val in u_dict.items():
                if hasattr(self.generator, 'get_h_port'):
                    H_sum = H_sum + self.generator.get_h_port(s, port_name, u_val, weights=chart_weights)
            return H_sum

        s_next = self.integrator.step(s_in, gen_func, dt)

        # === Numerical stability / viability projection ===
        # Prevent rare but catastrophic state explosions from poisoning training.
        # This acts as a soft "viability zone" projection: it should be inactive
        # under normal operation and only engage when values become extreme.
        s_next_flat = s_next.flat
        finite_mask = finite_mask & torch.isfinite(s_next_flat).all(dim=1)
        if strict_nonfinite and (not sanitize_nonfinite):
            torch._assert_async(torch.isfinite(s_next_flat).all(), "NaN/Inf in state after integrator")
        if sanitize_nonfinite:
            # Avoid `if not tensor.all():` which triggers a GPU→CPU sync every step.
            s_next_flat = torch.nan_to_num(s_next_flat, nan=0.0, posinf=0.0, neginf=0.0)

        d = self.dim_q
        q_next = s_next_flat[:, :d]
        p_next = s_next_flat[:, d : 2 * d]
        s_next_s = s_next_flat[:, 2 * d :]

        q_clip = self._q_clip_norm
        if q_clip > 0:
            q_norm = q_next.norm(dim=1, keepdim=True)
            scale = torch.clamp(q_clip / (q_norm + 1e-6), max=1.0)
            q_next = q_next * scale

        p_clip = self._p_clip_norm
        if p_clip > 0:
            p_norm = p_next.norm(dim=1, keepdim=True)
            scale = torch.clamp(p_clip / (p_norm + 1e-6), max=1.0)
            p_next = p_next * scale

        s_clip = self._s_clip_abs
        if s_clip > 0:
            s_next_s = torch.clamp(s_next_s, -s_clip, s_clip)

        s_next = ContactState(self.dim_q, batch_size, device, torch.cat([q_next, p_next, s_next_s], dim=1))

        if os.getenv("HEI_DEBUG_NAN", "0") == "1":
            if not torch.isfinite(s_next.flat).all():
                x = s_next.flat.detach()
                raise RuntimeError(
                    f"NaN/Inf after integrator: shape={tuple(x.shape)} max_abs={float(x.abs().max().item())}"
                )

        if skip_free_energy:
            F = torch.zeros(batch_size, device=device, dtype=s_next.flat.dtype)
        else:
            F = self.compute_free_energy(s_next, prediction_error=prediction_error)
        action = None
        if compute_action:
            action = self.generator.get_action(s_next, 'default', weights=chart_weights)

        next_prev = chart_weights.detach() if detach_next_prev_weights else chart_weights

        out_dict = {
            'next_state_flat': s_next.flat,
            'chart_weights': chart_weights,
            'next_prev_chart_weights': next_prev,
            'free_energy': F,
            'action': action,
        }
        if return_finite_mask:
            out_dict["finite_mask"] = finite_mask
        return out_dict
    
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
        change_magnitude = delta_w.abs().sum(dim=1)  # [batch]
        change_magnitude = torch.nan_to_num(change_magnitude, nan=0.0, posinf=0.0, neginf=0.0)
        threshold = self._transport_threshold
        if threshold <= 0:
            return p
        
        # NOTE: avoid boolean indexing (variable-sized tensors), which is slow and not capturable
        # under CUDA graphs. Keep everything dense and mask the update.
        cm = change_magnitude.unsqueeze(1)  # [B,1]
        apply = (cm >= threshold).to(dtype=p.dtype)  # [B,1] in {0,1}

        # 使用 q 自身作为确定性扰动方向，避免随机噪声破坏可复现实验/训练稳定性
        direction = q / (q.norm(dim=1, keepdim=True) + 1e-6)
        perturbation = cm * direction  # [B, dim_q]
        q_target = q + 0.1 * perturbation

        p_transported = self.connection(q, q_target, p)
        blend_factor = apply * torch.clamp(cm / (threshold + 1e-6), 0.0, 1.0)
        return (1.0 - blend_factor) * p + blend_factor * p_transported
    
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

        # Normalize inputs for experience tracking
        if u_dict is None:
            u_dict_norm: Dict[str, torch.Tensor] = {}
        elif isinstance(u_dict, torch.Tensor):
            u_dict_norm = {'default': u_dict}
        else:
            u_dict_norm = u_dict

        out = self.forward_tensor(
            state_flat=self.state.flat,
            u_dict=u_dict_norm,
            dt=dt,
            prev_chart_weights=self._prev_chart_weights,
            prediction_error=None,
            detach_next_prev_weights=True,
        )
        next_state = ContactState(self.dim_q, self.state.batch_size, self.state.device, out['next_state_flat'])
        chart_weights = out['chart_weights']
        self._prev_chart_weights = out['next_prev_chart_weights']
        F = out['free_energy']
        action = out['action']
        
        # 记录经验 (A2)
        if self.phase == Phase.ONLINE:
            self.online_steps += 1
            sensory = u_dict_norm.get('language', u_dict_norm.get('default', torch.zeros(batch_size, self.dim_u, device=device)))
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
        prev_phase = self.phase
        self.phase = Phase.OFFLINE
        
        # 离线时无外部输入
        u_dict = {}
        
        # 可选: 回放注入
        if replay_mode != 'none' and int(self.experience.size) > 0:
            replay = self.experience.sample_replay(1, mode=replay_mode)
            if replay is not None:
                # 轻微扰动注入 (模拟回放)
                # replay_state 现在是 [n, 2*dim_q + 1]，只取 q 部分
                replay_state = replay['states'].to(self.state.device)  # [n, 2*dim_q+1]
                replay_q = replay_state[0, :self.dim_q]  # 只取第一个样本的 q 部分 [dim_q]
                current_q = self.state.q[0]  # [dim_q]
                replay_perturbation = 0.1 * (replay_q - current_q)  # [dim_q]
                u_dict['replay'] = replay_perturbation.unsqueeze(0)  # [1, dim_q]
                
        result = self.step(u_dict, dt)
        self.phase = prev_phase
        return result
    
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
    
    def update_z(self, prediction_error: torch.Tensor, lr_z: float = 0.01, state: Optional[ContactState] = None):
        """
        更新自主上下文z
        
        z代表实体的"意图/偏好"，通过最小化预测误差来适应
        """
        z_val = self.z.detach().clone().requires_grad_(True)

        beta_kl = float(self.config.get("beta_kl", 0.01))
        gamma_pred = float(self.config.get("gamma_pred", 1.0))

        KL = 0.5 * (z_val ** 2).sum() * beta_kl
        F = KL + gamma_pred * prediction_error.mean()

        # Align with A3 semantics: include viability term V(q,z) when state is available.
        if state is None:
            state = self.state
        if state is not None:
            q = state.q.detach()
            z_batch = z_val.expand(q.shape[0], -1)
            V_inp = torch.cat([q, z_batch], dim=1)
            V = self.net_V(V_inp)
            stiffness = getattr(self.internal_gen, "stiffness", 0.0)
            if stiffness > 0:
                V = V + 0.5 * stiffness * (q ** 2).sum(dim=1, keepdim=True)
            F = F + V.mean()

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
            'experience_size': int(self.experience.size),
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
        'router_context_dim': 0,
        'connection_rank': 0,
        'connection_hidden_dim': 64,
        'connection_init_epsilon': 0.3,
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
