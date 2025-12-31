"""
双曲几何模块 - 使用Lorentz模型避免边界问题

根据理论基础-7/几何基础.md 第二节：
"在Q上选取负曲率（双曲型）黎曼度量 g_H"

关键设计决策（避免边界问题）：
1. 主要使用Lorentz模型（双曲面）而非Poincaré球
2. Lorentz模型无边界，数值稳定性更好
3. 使用群积分器进行演化，保持几何结构

动机：
- 层级结构在双曲空间中可以以较低维度、较小失真同时表达"相似性 + 层级性"
- 这是一种明确的容量/层级偏置，适用于"技能/概念/生成元"存在树状或近树状组织时

参考：
- Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical Representations", NeurIPS 2017
- 理论基础-7/快慢变量/最小族.md 中的群积分器思想
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ========================
#    基础操作
# ========================

def clamp_norm(x: torch.Tensor, max_norm: float = 1.0 - 1e-5) -> torch.Tensor:
    """将向量范数限制在max_norm内（用于Poincaré ball）"""
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return x * (max_norm / norm).clamp(max=1.0)


def artanh(x: torch.Tensor) -> torch.Tensor:
    """反双曲正切函数 artanh(x) = 0.5 * log((1+x)/(1-x))"""
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return 0.5 * torch.log((1 + x) / (1 - x))


def tanh(x: torch.Tensor) -> torch.Tensor:
    """双曲正切函数"""
    return torch.tanh(x)


# ========================
#    Poincaré Ball Model
# ========================

class PoincareBall:
    """
    Poincaré球模型
    
    空间: B^n = {x ∈ R^n : ||x|| < 1}
    度量: ds² = (2 / (1 - ||x||²))² * ||dx||²
    
    关键操作：
    - 莫比乌斯加法 (Möbius addition)
    - 指数映射 (Exponential map)
    - 对数映射 (Logarithmic map)
    - 测地线距离 (Geodesic distance)
    """
    
    def __init__(self, c: float = 1.0):
        """
        Args:
            c: 曲率参数 (c > 0 表示双曲空间，c=1 是标准Poincaré球)
        """
        self.c = c
        self.eps = 1e-7
    
    def _lambda_x(self, x: torch.Tensor) -> torch.Tensor:
        """共形因子 λ_x = 2 / (1 - c||x||²)"""
        c = self.c
        norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
        return 2.0 / (1 - c * norm_sq).clamp(min=self.eps)
    
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius加法: x ⊕ y
        
        (1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y
        ─────────────────────────────────────────
        1 + 2c<x,y> + c²||x||²||y||²
        """
        c = self.c
        x_sq = (x ** 2).sum(dim=-1, keepdim=True)
        y_sq = (y ** 2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        
        num = (1 + 2*c*xy + c*y_sq) * x + (1 - c*x_sq) * y
        denom = 1 + 2*c*xy + (c**2)*x_sq*y_sq
        
        return num / denom.clamp(min=self.eps)
    
    def mobius_scalar_mul(self, r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Möbius标量乘法: r ⊗ x = tanh(r * artanh(||x||)) * x/||x||
        """
        c = self.c
        sqrt_c = c ** 0.5
        
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        norm_c = sqrt_c * norm
        
        # tanh(r * artanh(sqrt(c) * ||x||)) / (sqrt(c) * ||x||) * x
        return tanh(r * artanh(norm_c)) / norm_c.clamp(min=self.eps) * x
    
    def exp_map(self, v: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        指数映射 exp_x(v): 切空间 T_x B → B
        
        如果 x 为 None，则从原点映射 (x = 0)
        """
        c = self.c
        sqrt_c = c ** 0.5
        
        if x is None:
            # 从原点映射: exp_0(v) = tanh(sqrt(c)||v||) * v/(sqrt(c)||v||)
            norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
            return tanh(sqrt_c * norm) / (sqrt_c * norm) * v
        else:
            # 从x映射: exp_x(v) = x ⊕ (tanh(λ_x||v||/2) * v/(sqrt(c)||v||))
            lambda_x = self._lambda_x(x)
            norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
            
            second_term = tanh(sqrt_c * lambda_x * norm / 2) / (sqrt_c * norm) * v
            return self.mobius_add(x, second_term)
    
    def log_map(self, y: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        对数映射 log_x(y): B → T_x B
        
        如果 x 为 None，则映射到原点 (x = 0)
        """
        c = self.c
        sqrt_c = c ** 0.5
        
        if x is None:
            # 到原点: log_0(y) = artanh(sqrt(c)||y||) * y/(sqrt(c)||y||)
            norm = y.norm(dim=-1, keepdim=True).clamp(min=self.eps)
            return artanh(sqrt_c * norm) / (sqrt_c * norm) * y
        else:
            # 到x: log_x(y) = 2/(sqrt(c)λ_x) * artanh(sqrt(c)||-x⊕y||) * (-x⊕y)/||-x⊕y||
            neg_x = -x
            diff = self.mobius_add(neg_x, y)
            norm = diff.norm(dim=-1, keepdim=True).clamp(min=self.eps)
            lambda_x = self._lambda_x(x)
            
            return 2 / (sqrt_c * lambda_x) * artanh(sqrt_c * norm) / norm * diff
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        测地线距离 d(x, y)
        
        d(x, y) = (2/sqrt(c)) * artanh(sqrt(c)||-x⊕y||)
        """
        c = self.c
        sqrt_c = c ** 0.5
        
        neg_x = -x
        diff = self.mobius_add(neg_x, y)
        norm = diff.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        
        return 2 / sqrt_c * artanh(sqrt_c * norm)
    
    def project(self, x: torch.Tensor, max_norm: float = 1.0 - 1e-5) -> torch.Tensor:
        """将点投影回Poincaré球内"""
        return clamp_norm(x, max_norm)


# ========================
#    Lorentz Model (更稳定)
# ========================

class LorentzModel:
    """
    Lorentz模型 (双曲面模型)
    
    空间: H^n = {x ∈ R^{n+1} : <x,x>_L = -1, x_0 > 0}
    Lorentz内积: <x,y>_L = -x_0*y_0 + x_1*y_1 + ... + x_n*y_n
    
    优点：数值稳定性更好，无边界问题
    """
    
    def __init__(self, c: float = 1.0):
        self.c = c
        self.eps = 1e-7
    
    def lorentz_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Lorentz内积: -x_0*y_0 + sum(x_i*y_i for i>=1)"""
        # x, y: (*, n+1)
        return -x[..., 0:1] * y[..., 0:1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    
    def lorentz_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Lorentz范数: sqrt(-<x,x>_L)"""
        return (-self.lorentz_inner(x, x)).clamp(min=self.eps).sqrt()
    
    def project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """
        将(n+1)维向量投影到双曲面上
        
        给定 x = (x_0, x_1, ..., x_n)
        计算 x_0' = sqrt(1 + ||x_{1:}||²)
        """
        x_space = x[..., 1:]
        x_0 = (1 + (x_space ** 2).sum(dim=-1)).sqrt()
        return torch.cat([x_0.unsqueeze(-1), x_space], dim=-1)
    
    def exp_map(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        指数映射 exp_x(v): T_x H → H
        
        exp_x(v) = cosh(||v||_L) * x + sinh(||v||_L) * v/||v||_L
        """
        v_norm = self.lorentz_norm(v).clamp(min=self.eps)
        return torch.cosh(v_norm) * x + torch.sinh(v_norm) * v / v_norm
    
    def log_map(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        对数映射 log_x(y): H → T_x H
        """
        xy = self.lorentz_inner(x, y)
        v = y + xy * x
        v_norm = self.lorentz_norm(v).clamp(min=self.eps)
        dist = self.distance(x, y)
        
        return dist / v_norm * v
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        测地线距离 d(x, y) = arcosh(-<x,y>_L)
        """
        xy = -self.lorentz_inner(x, y)
        return torch.acosh(xy.clamp(min=1.0 + self.eps))
    
    def to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """从Lorentz模型转换到Poincaré球"""
        return x[..., 1:] / (1 + x[..., 0:1])
    
    def from_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """从Poincaré球转换到Lorentz模型"""
        x_sq = (x ** 2).sum(dim=-1, keepdim=True)
        x_0 = (1 + x_sq) / (1 - x_sq)
        x_space = 2 * x / (1 - x_sq)
        return torch.cat([x_0, x_space], dim=-1)
    
    def parallel_transport(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        平行移动：将切向量v从x点移动到y点
        
        根据理论基础-7/几何基础.md第五节：
        "平行移动/联络的角色...把特征从一个局部框架搬运到另一个框架"
        
        Lorentz模型的平行移动公式：
        PT_{x→y}(v) = v - <y, v>_L / (1 - <x,y>_L) * (x + y)
        """
        xy = self.lorentz_inner(x, y)
        yv = self.lorentz_inner(y, v)
        
        # 避免除零
        denom = (1 - xy).clamp(min=self.eps)
        
        return v - yv / denom * (x + y)


# ========================
#    群积分器 (Lie Group Integrator)
# ========================

class HyperbolicIntegrator(nn.Module):
    """
    双曲空间群积分器
    
    根据理论基础-7/快慢变量/最小族.md：
    "接触化是把耗散做成几何公理，而不是训练技巧"
    
    关键设计：
    1. 使用Lorentz模型避免边界问题
    2. 使用指数映射/对数映射进行状态更新
    3. 保持双曲面约束（<x,x>_L = -1）
    4. 支持平行移动进行动量传输
    
    这是一个结构保持的积分器，保证状态始终在双曲面上。
    """
    
    def __init__(self, dim: int, c: float = 1.0):
        """
        Args:
            dim: 空间维度（不含时间维度）
            c: 曲率参数
        """
        super().__init__()
        self.dim = dim
        self.dim_full = dim + 1  # Lorentz模型需要n+1维
        self.manifold = LorentzModel(c)
        
        # 原点（双曲面上的参考点）
        # 在Lorentz模型中，原点是 (1, 0, 0, ..., 0)
        self.register_buffer('origin', self._create_origin(dim))
    
    def _create_origin(self, dim: int) -> torch.Tensor:
        """创建双曲面原点"""
        origin = torch.zeros(dim + 1)
        origin[0] = 1.0  # 时间分量
        return origin
    
    def euclidean_to_hyperbolic(self, q: torch.Tensor) -> torch.Tensor:
        """
        将欧氏状态映射到双曲空间
        
        使用指数映射从原点出发：
        h = exp_o(q_tangent)
        
        其中 q_tangent 是原点切空间中的向量
        """
        batch_size = q.shape[0]
        device = q.device
        
        # 构造切向量（在原点的切空间中）
        # 切空间：{v : <v, origin>_L = 0}，即 v_0 = 0
        v = F.pad(q, (1, 0), value=0.0)  # (batch, dim+1)
        
        # 从原点指数映射
        origin = self.origin.to(device).unsqueeze(0).expand(batch_size, -1)
        h = self.manifold.exp_map(v, origin)
        
        return h
    
    def hyperbolic_to_euclidean(self, h: torch.Tensor) -> torch.Tensor:
        """
        将双曲状态映射回欧氏空间
        
        使用对数映射到原点：
        q = log_o(h)
        
        返回空间分量（去除时间分量）
        """
        batch_size = h.shape[0]
        device = h.device
        
        origin = self.origin.to(device).unsqueeze(0).expand(batch_size, -1)
        v = self.manifold.log_map(h, origin)
        
        # 返回空间分量
        return v[..., 1:]
    
    def step(self, 
             q: torch.Tensor, 
             p: torch.Tensor, 
             velocity_func, 
             dt: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在双曲空间中进行一步演化
        
        使用测地线积分（结构保持）：
        1. q_h = euclidean_to_hyperbolic(q)
        2. v = velocity_func(q) → 切向量
        3. q_h_new = exp_{q_h}(dt * v)
        4. q_new = hyperbolic_to_euclidean(q_h_new)
        5. 平行移动动量
        
        Args:
            q: 欧氏位置 (batch, dim)
            p: 欧氏动量 (batch, dim)
            velocity_func: 速度场函数 q → v
            dt: 时间步长
            
        Returns:
            q_new, p_new: 更新后的状态
        """
        batch_size = q.shape[0]
        device = q.device
        
        # 映射到双曲空间
        q_h = self.euclidean_to_hyperbolic(q)
        
        # 计算速度（在欧氏空间）
        v_euc = velocity_func(q)
        
        # 构造切向量
        v = F.pad(v_euc, (1, 0), value=0.0)
        
        # 确保v在切空间内：<v, q_h>_L = 0
        # 投影：v_tangent = v - <v, q_h>_L * q_h
        vq = self.manifold.lorentz_inner(v, q_h)
        v_tangent = v - vq * q_h
        
        # 沿测地线演化
        q_h_new = self.manifold.exp_map(dt * v_tangent, q_h)
        
        # 平行移动动量
        p_extended = F.pad(p, (1, 0), value=0.0)
        # 投影到切空间
        pq = self.manifold.lorentz_inner(p_extended, q_h)
        p_tangent = p_extended - pq * q_h
        
        # 平行移动
        p_transported = self.manifold.parallel_transport(p_tangent, q_h, q_h_new)
        
        # 映射回欧氏空间
        q_new = self.hyperbolic_to_euclidean(q_h_new)
        p_new = p_transported[..., 1:]  # 空间分量
        
        return q_new, p_new
    
    def hyperbolic_distance(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """计算两个欧氏状态在双曲空间中的距离"""
        h1 = self.euclidean_to_hyperbolic(q1)
        h2 = self.euclidean_to_hyperbolic(q2)
        return self.manifold.distance(h1, h2)
    
    def geodesic_midpoint(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """计算测地线中点"""
        h1 = self.euclidean_to_hyperbolic(q1)
        h2 = self.euclidean_to_hyperbolic(q2)
        
        # log_h1(h2) 得到方向
        v = self.manifold.log_map(h2, h1)
        
        # 走一半
        h_mid = self.manifold.exp_map(0.5 * v, h1)
        
        return self.hyperbolic_to_euclidean(h_mid)


# ========================
#    双曲嵌入层
# ========================

class HyperbolicEmbedding(nn.Module):
    """
    双曲空间嵌入层
    
    根据几何基础.md：
    "双曲度量作为表示偏置...层级结构在双曲空间中可以以较低维度、
     较小失真同时表达'相似性 + 层级性'"
    
    用法：
    1. 作为位置编码的增强
    2. 作为层级概念的表示空间
    3. 作为技能/图册的组织结构
    """
    
    def __init__(self, dim: int, model: str = 'poincare', c: float = 1.0):
        """
        Args:
            dim: 嵌入维度
            model: 'poincare' 或 'lorentz'
            c: 曲率参数
        """
        super().__init__()
        self.dim = dim
        self.model_type = model
        self.c = c
        
        if model == 'poincare':
            self.manifold = PoincareBall(c)
        else:
            self.manifold = LorentzModel(c)
            self.dim_full = dim + 1  # Lorentz需要额外一维
            
    def euclidean_to_hyperbolic(self, x: torch.Tensor) -> torch.Tensor:
        """将欧氏空间向量映射到双曲空间"""
        if self.model_type == 'poincare':
            return self.manifold.exp_map(x * 0.1)  # 缩放以避免边界
        else:
            # 扩展维度并投影
            x_extended = F.pad(x, (1, 0), value=0.0)  # 添加时间维度
            return self.manifold.project_to_hyperboloid(x_extended)
    
    def hyperbolic_to_euclidean(self, x: torch.Tensor) -> torch.Tensor:
        """将双曲空间向量映射回欧氏空间"""
        if self.model_type == 'poincare':
            return self.manifold.log_map(x)
        else:
            return self.manifold.to_poincare(x)
    
    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算双曲距离"""
        return self.manifold.distance(x, y)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入映射到双曲空间
        
        Args:
            x: 欧氏空间输入 (batch, dim)
            
        Returns:
            双曲空间表示 (batch, dim) 或 (batch, dim+1) for Lorentz
        """
        return self.euclidean_to_hyperbolic(x)


# ========================
#    双曲距离正则化
# ========================

class HyperbolicRegularizer(nn.Module):
    """
    双曲几何正则化器
    
    确保状态空间遵循双曲几何结构：
    1. 层级距离保持
    2. 相似性聚类
    3. 边界约束（Poincaré）
    """
    
    def __init__(self, dim: int, model: str = 'poincare', c: float = 1.0):
        super().__init__()
        self.embedding = HyperbolicEmbedding(dim, model, c)
        
    def hierarchy_preserving_loss(self, 
                                   q_parent: torch.Tensor, 
                                   q_child: torch.Tensor,
                                   margin: float = 0.1) -> torch.Tensor:
        """
        层级保持损失：确保子概念距原点比父概念更远
        
        根据Poincaré嵌入论文：
        "越抽象的概念越靠近原点，越具体的概念越靠近边界"
        """
        h_parent = self.embedding(q_parent)
        h_child = self.embedding(q_child)
        
        # 计算到原点的距离
        origin = torch.zeros_like(h_parent)
        
        d_parent = self.embedding.hyperbolic_distance(origin, h_parent)
        d_child = self.embedding.hyperbolic_distance(origin, h_child)
        
        # 子概念应该比父概念更远离原点
        loss = F.relu(d_parent - d_child + margin)
        
        return loss.mean()
    
    def clustering_loss(self, 
                        q_similar: torch.Tensor,
                        q_different: torch.Tensor,
                        margin: float = 0.5) -> torch.Tensor:
        """
        聚类损失：相似概念在双曲空间中应该更近
        """
        h_sim = self.embedding(q_similar)
        h_diff = self.embedding(q_different)
        
        # 相似样本应该接近
        d_sim = self.embedding.hyperbolic_distance(h_sim[:-1], h_sim[1:])
        
        # 不同样本应该远离
        d_diff = self.embedding.hyperbolic_distance(h_sim, h_diff)
        
        # Triplet-like loss
        loss = F.relu(d_sim - d_diff + margin)
        
        return loss.mean()
    
    def boundary_regularization(self, q: torch.Tensor) -> torch.Tensor:
        """
        边界正则化：防止状态过于靠近Poincaré球边界
        """
        h = self.embedding(q)
        
        if self.embedding.model_type == 'poincare':
            norm = h.norm(dim=-1)
            # 惩罚过于靠近边界的点
            loss = F.relu(norm - 0.9) ** 2
            return loss.mean()
        else:
            return torch.tensor(0.0, device=q.device)


# ========================
#    测试
# ========================

if __name__ == "__main__":
    print("双曲几何模块测试\n")
    
    # 测试Poincaré球
    print("=== Poincaré Ball ===")
    pb = PoincareBall(c=1.0)
    
    x = torch.randn(4, 32) * 0.1
    y = torch.randn(4, 32) * 0.1
    
    x_proj = pb.project(x)
    print(f"投影后范数: {x_proj.norm(dim=-1)}")
    
    d = pb.distance(x_proj, pb.project(y))
    print(f"测地线距离: {d.squeeze()}")
    
    # Möbius运算
    z = pb.mobius_add(x_proj, pb.project(y))
    print(f"Möbius加法结果范数: {z.norm(dim=-1)}")
    
    # 测试指数/对数映射
    v = torch.randn(4, 32) * 0.01
    y_exp = pb.exp_map(v, x_proj)
    v_log = pb.log_map(y_exp, x_proj)
    print(f"exp-log一致性误差: {(v - v_log).norm():.6f}")
    
    # 测试嵌入层
    print("\n=== Hyperbolic Embedding ===")
    emb = HyperbolicEmbedding(32, 'poincare')
    q = torch.randn(4, 32)
    h = emb(q)
    print(f"嵌入维度: {h.shape}")
    print(f"嵌入范数: {h.norm(dim=-1)}")
    
    # 测试正则化器
    print("\n=== Hyperbolic Regularizer ===")
    reg = HyperbolicRegularizer(32)
    
    q1 = torch.randn(4, 32) * 0.1  # "父"概念
    q2 = torch.randn(4, 32) * 0.5  # "子"概念
    
    h_loss = reg.hierarchy_preserving_loss(q1, q2)
    print(f"层级保持损失: {h_loss.item():.6f}")
    
    b_loss = reg.boundary_regularization(q)
    print(f"边界正则化损失: {b_loss.item():.6f}")
    
    print("\n✓ 双曲几何模块测试通过")

