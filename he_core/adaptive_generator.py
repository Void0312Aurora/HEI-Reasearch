
import torch
import torch.nn as nn
import math
from he_core.state import ContactState
from he_core.generator import DeepDissipativeGenerator

class AdaptiveDissipativeGenerator(DeepDissipativeGenerator):
    """
    Theoretical Closure: State-Dependent Damping with Bounded Convex Contact Potential.
    
    根据理论基础-7的动力学基础（模板1）：
    H^c(q,p,S) = H_0(q,p) + Φ(S) + U(q,S)
    
    几何基础-7的核心要求（第二节）：
    - "双曲度量作为表示偏置"
    - T(q,p) = 1/2 <p, g^(-1)(q), p>，其中 g 是黎曼度量
    - "双曲 + 辛/接触并不冲突：一个定义距离/曲率偏置，一个定义相空间的几何动力学语言"
    
    本实现将双曲度量集成到动能项：
    - K(p) = 1/2 <p, g_H^(-1)(q), p>
    - g_H(q) = λ(q)² I，其中 λ(q) = 2/(1 - c||q||²) 是共形因子（Poincaré模型）
    - 使用正则化版本避免边界问题：λ(q) = 1 + c||q||²/(1 + c||q||²)
    
    关键修正：
    1. Φ(S) 必须是 S 的凸函数（二次项确保）
    2. α(q) 必须有上界（确保 Φ_min 有下界）
    3. V(q) 必须有下界（势能物理约束）
    4. 动能使用位置依赖的双曲度量（层级偏置）
    """
    def __init__(self, dim_q: int, net_V: nn.Module = None, dim_z: int = 0, 
                 stiffness: float = 0.0, contact_stiffness: float = 0.1,
                 alpha_max: float = 1.0, alpha_min: float = 0.2, hyperbolic_c: float = 0.1):
        super().__init__(dim_q, alpha=0.0, net_V=net_V, stiffness=stiffness)
        self.dim_z = dim_z
        
        # 接触刚度：确保Φ(s)是强凸的
        # λ > 0 保证 Φ''(s) = λ > 0
        self.contact_stiffness = contact_stiffness
        
        # 耗散系数上界：确保Φ有下界
        # Φ_min = -α_max²/(2λ) 是有限的
        self.alpha_max = alpha_max

        # 耗散系数下界：确保 dH/ds 有正的下界，从而在有界外力下动量不会无界增长。
        # 这比单纯“>0”更强：允许 alpha(q) 过小会导致接近无耗散，p 在持续驱动下线性/超线性变大，
        # 进而把群元推进到 cosh/sinh 溢出区间，引发 NaN。
        self.alpha_min = float(alpha_min)
        
        # === 双曲度量参数（几何基础第二节）===
        # 曲率参数 c > 0，控制双曲效应强度
        # c 越大，层级压缩越强
        self.hyperbolic_c = hyperbolic_c
        
        # Learnable Damping Field α(q) ∈ (0, α_max]
        # 使用sigmoid确保正且有界：α = α_max · sigmoid(net(q))
        self.net_Alpha = nn.Sequential(
            nn.Linear(dim_q, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
            # 不再使用Softplus，而是在forward中使用sigmoid + scale
        )
        
        self._init_stable_alpha()
        
    def _init_stable_alpha(self):
        """
        Theoretical Stability:
        Initialize Alpha(q) to be nearly constant (nabla Alpha ~ 0) but positive.
        This prevents the -s * grad(Alpha) term from pumping energy during early training.
        """
        for name, param in self.net_Alpha.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1e-5) # Almost zero gradient
            elif 'bias' in name:
                # For hidden layers, bias=0.
                # For the output layer, bias sets the initial damping level.
                # With alpha = alpha_max * sigmoid(alpha_raw), choose a small positive alpha
                # so early training is dissipative (prevents immediate momentum blow-up).
                if param.shape[0] == 1: # Output layer (assuming latent=64)
                    # Initialize near a mild, but strictly positive, dissipation level.
                    # With alpha(q) = alpha_min + (alpha_max-alpha_min)*sigmoid(raw),
                    # choose target_alpha in (alpha_min, alpha_max).
                    alpha_min = float(self.alpha_min)
                    alpha_max = float(self.alpha_max)
                    if not (alpha_max > alpha_min):
                        alpha_max = alpha_min + 1.0
                        self.alpha_max = alpha_max
                    target_alpha = alpha_min + 0.1 * (alpha_max - alpha_min)
                    # Invert sigmoid: ratio = (alpha-target_min)/(alpha_max-alpha_min)
                    ratio = (target_alpha - alpha_min) / (alpha_max - alpha_min)
                    ratio = max(min(float(ratio), 0.99), 0.01)
                    init_bias = math.log(ratio / (1.0 - ratio))
                    nn.init.constant_(param, init_bias)
                else:
                    nn.init.zeros_(param)
        
    def _compute_metric_factor(self, q: torch.Tensor) -> torch.Tensor:
        """
        计算双曲度量的共形因子（几何基础第二节）
        
        理论背景：
        - Poincaré球模型中，度量张量 g = λ(q)² I
        - λ(q) = 2/(1 - c||q||²)，在边界 ||q|| → 1/√c 时 λ → ∞
        
        为避免边界问题，使用正则化版本：
        - λ(q) = 1 + c·||q||²/(1 + c·||q||²)
        - 这在 ||q|| → ∞ 时趋于 2，提供温和的层级压缩
        
        度量逆（用于动能）：
        - g^(-1) = λ(q)^(-2) I
        - K(p) = 1/2 <p, g^(-1), p> = 1/2 λ^(-2) ||p||²
        
        物理意义：
        - 远离原点时，λ 增大 → g^(-1) 减小 → 相同 ||p|| 产生更小的动能
        - 这鼓励系统向原点收敛（层级根节点）
        - 同时远离原点的变化被"放大"（层级敏感性）
        """
        c = self.hyperbolic_c
        q_norm_sq = (q ** 2).sum(dim=-1, keepdim=True)  # ||q||²
        
        # 正则化共形因子
        # λ = 1 + c·||q||²/(1 + c·||q||²) ∈ [1, 2)
        lambda_q = 1.0 + c * q_norm_sq / (1.0 + c * q_norm_sq)
        
        return lambda_q
    
    def forward(self, state: ContactState, z: torch.Tensor = None) -> torch.Tensor:
        """
        计算接触哈密顿量 H^c
        
        H^c = K(p, q) + V(q,z) + Φ(s)
        
        其中（根据几何基础第二节和最小族模板0）：
        - K(p,q) = 0.5·<p, g_H^(-1)(q), p> = 0.5·λ(q)^(-2)·||p||²  (双曲动能)
        - V(q,z) = V_learned(q,z) + 0.5·stiffness·||q||²  (势能，有下界)
        - Φ(s) = 0.5·λ·s² + α(q)·s  (接触势，凸函数，有下界)
        
        双曲度量的作用（几何基础第二节）：
        - "层级结构在双曲空间中可以以较低维度、较小失真同时表达'相似性 + 层级性'"
        - 远离原点的状态有更小的有效动能，提供向心偏置
        - 这是表示层的几何偏置，与接触动力学共存
        """
        q, p, s = state.q, state.p, state.s
        
        # === 双曲动能 K(p,q) = 0.5·λ(q)^(-2)·||p||² ===
        # 根据最小族.md: T(q,p) = 1/2 <p, g^(-1)(q), p>
        # g = λ² I → g^(-1) = λ^(-2) I
        lambda_q = self._compute_metric_factor(q)
        metric_inv = lambda_q ** (-2)  # λ^(-2)
        K = 0.5 * metric_inv * (p**2).sum(dim=1, keepdim=True)
        
        # === 势能 V(q, z) ===
        if self.dim_z > 0 and z is not None:
            # Expand z if necessary to match q batch size
            if z.shape[0] != q.shape[0]:
                z = z.expand(q.shape[0], -1)
            inp = torch.cat([q, z], dim=1)
            V = self.net_V(inp)
        else:
            V = self.net_V(q)
            
        # 谐波约束 (q的二次势，确保有下界)
        if self.stiffness > 0:
            V_conf = 0.5 * self.stiffness * (q**2).sum(dim=1, keepdim=True)
            V = V + V_conf
        
        # === 接触势 Φ(s) ===
        # 
        # 关键约束：
        # 1. 凸性：Φ''(s) = λ > 0 ✓
        # 2. 有界性：α ∈ [ε, α_max] 确保 Φ_min >= -α_max²/(2λ)
        #
        # 最小值点：s* = -α/λ
        # 最小值：Φ(s*) = -α²/(2λ)
        
        # 自适应耗散系数 α(q) ∈ (0, α_max]
        # 与理论基础-7中“耗散系数需有上界且用于Lyapunov语义”的约束一致。
        alpha_raw = self.net_Alpha(q)
        alpha_min = float(self.alpha_min)
        alpha_max = float(self.alpha_max)
        if not (alpha_max > alpha_min):
            alpha_max = alpha_min + 1.0
        alpha_q = alpha_min + (alpha_max - alpha_min) * torch.sigmoid(alpha_raw)
        
        # 关键修正（数值/动力学一致性）：
        # 若使用二次项 0.5·λ·s²，则 dot_s = p·dH/dp - H 会包含 -0.5·λ·s²，
        # 在常见情形下会把 s 推向 -∞，从而使 dH/ds = Φ'(s)+... → -∞，导致动量被强烈“负阻尼”泵爆。
        # 为保持 Φ 的凸性但避免 dH/ds 无界，使用“渐近线性”的凸势：
        #   Φ(s) = λ( sqrt(1+s²) - 1 ) + α(q)·s
        # 其导数 Φ'(s) = λ * s/sqrt(1+s²) 有界于 [-λ, λ]。
        lambda_s = self.contact_stiffness
        Phi_s = lambda_s * (torch.sqrt(1.0 + s * s) - 1.0) + alpha_q * s
        
        # 计算 Φ 的理论下界用于调试
        # Phi_min = -alpha_q**2 / (2 * lambda_s)
        # 由于 alpha_q <= alpha_max + 1e-3，Phi_min >= -(alpha_max+1e-3)²/(2λ)
        
        # 总哈密顿量
        H = K + V + Phi_s
        
        return H
