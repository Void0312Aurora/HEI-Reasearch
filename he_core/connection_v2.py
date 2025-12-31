"""
L2 Connection v2: 增强版联络/平行移动

理论基础 (理论基础-7/逻辑/生成元.md):
- L2 迁移/规范生成元：跨技能/跨图册的一致性迁移
- 联络可视为 Lie algebroid 短正合列的分裂(splitting)
- 曲率反映"绕圈不回到原位"的几何缺陷

增强功能:
1. 完整的平行移动算子
2. 曲率估计和正则化
3. Holonomy loop 测试
4. 非阿贝尔性探针
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
import math


class ConnectionV2(nn.Module):
    """
    增强版 L2 联络
    
    实现完整的平行移动和曲率估计:
    - 平行移动: PT(q_from, q_to, v) = exp(-∫ A dq) @ v
    - 曲率: F = dA + A ∧ A
    - Holonomy: 沿闭合路径的平行移动残差
    """
    
    def __init__(self, dim_q: int, hidden_dim: int = 64, num_charts: int = 4):
        super().__init__()
        self.dim_q = dim_q
        self.hidden_dim = hidden_dim
        self.num_charts = num_charts
        
        # === 联络1-形式 A_mu(q) ===
        # 输出: dim_q 个 so(dim_q) 矩阵 (反对称)
        # 每个方向 mu 对应一个反对称矩阵 A_mu
        # 总参数: dim_q * (dim_q * (dim_q-1) / 2)
        self.net_A = nn.Sequential(
            nn.Linear(dim_q, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim_q * dim_q * dim_q)  # A_mu^{ab}
        )
        
        # === 图册间转换矩阵 ===
        # 每对图册之间的转换
        self.chart_transitions = nn.Parameter(
            torch.eye(dim_q).unsqueeze(0).unsqueeze(0).expand(num_charts, num_charts, -1, -1).clone()
        )
        
        # === 曲率正则化强度 ===
        self.curvature_weight = nn.Parameter(torch.tensor(0.1))
        
        # 初始化为近平坦联络
        self._init_flat()
        
    def _init_flat(self):
        """初始化为近平坦联络 (零曲率)"""
        for name, param in self.net_A.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, std=1e-5)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def get_connection_form(self, q: torch.Tensor) -> torch.Tensor:
        """
        获取联络1-形式 A_mu(q)
        
        Args:
            q: 位置 [batch, dim_q]
            
        Returns:
            A: 联络形式 [batch, dim_q, dim_q, dim_q]
               A[b, mu, a, b] = 第mu方向的 (a,b) 分量
        """
        batch_size = q.shape[0]
        
        # 获取原始输出
        A_raw = self.net_A(q)  # [batch, dim_q^3]
        A = A_raw.view(batch_size, self.dim_q, self.dim_q, self.dim_q)
        
        # 对每个mu方向，确保反对称性 (so(n) 结构)
        A_skew = 0.5 * (A - A.transpose(-1, -2))
        
        return A_skew
    
    def compute_curvature(self, q: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
        """
        计算曲率张量 F_{mu,nu}
        
        F = dA + A ∧ A
        
        数值近似:
        F_{mu,nu} ≈ (A_nu(q + delta_mu) - A_nu(q - delta_mu)) / (2*delta)
                  - (A_mu(q + delta_nu) - A_mu(q - delta_nu)) / (2*delta)
                  + [A_mu, A_nu]
        
        Returns:
            curvature: [batch, dim_q, dim_q, dim_q, dim_q]
                       F[b, mu, nu, a, b] = 曲率张量
        """
        batch_size = q.shape[0]
        device = q.device
        
        # 获取当前点的联络
        A = self.get_connection_form(q)  # [B, dim_q, dim_q, dim_q]
        
        # 简化: 只计算曲率的 Frobenius 范数作为标量度量
        # 完整曲率计算太昂贵，用李括号的范数近似
        
        curvature_norm = torch.zeros(batch_size, device=device)
        
        for mu in range(min(self.dim_q, 4)):  # 只检查前几个方向
            for nu in range(mu + 1, min(self.dim_q, 4)):
                A_mu = A[:, mu]  # [B, dim_q, dim_q]
                A_nu = A[:, nu]  # [B, dim_q, dim_q]
                
                # 李括号 [A_mu, A_nu]
                commutator = torch.bmm(A_mu, A_nu) - torch.bmm(A_nu, A_mu)
                
                curvature_norm += commutator.pow(2).sum(dim=[-1, -2])
                
        return curvature_norm
    
    def parallel_transport(self, 
                          q_from: torch.Tensor, 
                          q_to: torch.Tensor, 
                          v: torch.Tensor,
                          num_steps: int = 10) -> torch.Tensor:
        """
        沿直线路径进行平行移动
        
        PT(q_from -> q_to, v) = ∏_i exp(-A_i * dq_i)  @ v
        
        Args:
            q_from: 起点 [batch, dim_q]
            q_to: 终点 [batch, dim_q]
            v: 要移动的向量 [batch, dim_q]
            num_steps: 离散化步数
            
        Returns:
            v_transported: 移动后的向量 [batch, dim_q]
        """
        batch_size = q_from.shape[0]
        device = q_from.device
        
        # 路径参数化
        dq = (q_to - q_from) / num_steps  # [batch, dim_q]
        
        v_current = v.clone()
        q_current = q_from.clone()
        
        for step in range(num_steps):
            # 获取当前位置的联络
            A = self.get_connection_form(q_current)  # [B, dim_q, dim_q, dim_q]
            
            # 计算 A_mu * dq^mu (求和)
            # A: [B, mu, i, j], dq: [B, mu]
            # -> A_contracted: [B, i, j]
            A_contracted = torch.einsum('bmij,bm->bij', A, dq)
            
            # 一阶近似: exp(-A*dq) ≈ I - A*dq
            I = torch.eye(self.dim_q, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            transport_matrix = I - A_contracted
            
            # 移动向量
            v_current = torch.bmm(transport_matrix, v_current.unsqueeze(-1)).squeeze(-1)
            
            # 更新位置
            q_current = q_current + dq
            
        return v_current
    
    def holonomy_loop(self, 
                      q_center: torch.Tensor, 
                      radius: float = 0.1,
                      num_vertices: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算围绕中心点的 holonomy (几何相位)
        
        沿闭合路径移动单位向量，返回后应该回到原位
        偏差即 holonomy 误差
        
        Args:
            q_center: 中心点 [batch, dim_q]
            radius: 闭合路径半径
            num_vertices: 路径顶点数
            
        Returns:
            holonomy_error: [batch] holonomy 误差范数
            final_vector: [batch, dim_q] 最终向量
        """
        batch_size = q_center.shape[0]
        device = q_center.device
        
        # 初始单位向量
        v_init = torch.zeros(batch_size, self.dim_q, device=device)
        v_init[:, 0] = 1.0
        
        # 构建闭合路径 (正方形或多边形)
        path_vertices = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            offset = torch.zeros(batch_size, self.dim_q, device=device)
            if self.dim_q >= 2:
                offset[:, 0] = radius * math.cos(angle)
                offset[:, 1] = radius * math.sin(angle)
            else:
                offset[:, 0] = radius * math.cos(angle)
            path_vertices.append(q_center + offset)
        path_vertices.append(path_vertices[0])  # 闭合
        
        # 沿路径移动
        v_current = v_init.clone()
        for i in range(len(path_vertices) - 1):
            v_current = self.parallel_transport(
                path_vertices[i], 
                path_vertices[i + 1], 
                v_current,
                num_steps=5
            )
        
        # 计算误差
        holonomy_error = (v_current - v_init).norm(dim=1)
        
        return holonomy_error, v_current
    
    def chart_transport(self,
                       v: torch.Tensor,
                       from_chart: int,
                       to_chart: int) -> torch.Tensor:
        """
        跨图册移动向量
        
        使用学习的图册间转换矩阵
        """
        T = self.chart_transitions[from_chart, to_chart]  # [dim_q, dim_q]
        return torch.matmul(v, T.T)
    
    def forward(self, 
                q_from: torch.Tensor, 
                q_to: torch.Tensor, 
                v: torch.Tensor) -> torch.Tensor:
        """
        标准接口: 平行移动
        """
        return self.parallel_transport(q_from, q_to, v, num_steps=5)
    
    def curvature_loss(self, q_samples: torch.Tensor) -> torch.Tensor:
        """
        曲率正则化损失
        
        鼓励近平坦联络
        """
        curvature = self.compute_curvature(q_samples)
        return self.curvature_weight * curvature.mean()
    
    def holonomy_loss(self, q_samples: torch.Tensor, radius: float = 0.1) -> torch.Tensor:
        """
        Holonomy 正则化损失
        
        鼓励小闭环返回原位
        """
        error, _ = self.holonomy_loop(q_samples, radius)
        return error.mean()


class NonAbelianProbe(nn.Module):
    """
    非阿贝尔性探针
    
    测试不同顺序的迁移组合是否等价
    [A, B] ≠ 0 表示非阿贝尔
    """
    
    def __init__(self, connection: ConnectionV2):
        super().__init__()
        self.connection = connection
        
    def measure_commutativity(self, 
                              q_center: torch.Tensor,
                              direction_a: torch.Tensor,
                              direction_b: torch.Tensor,
                              magnitude: float = 0.1) -> torch.Tensor:
        """
        测量 A->B 和 B->A 的差异
        
        Returns:
            commutator_norm: [batch] 非交换性强度
        """
        batch_size = q_center.shape[0]
        device = q_center.device
        
        # 初始向量
        v = torch.randn(batch_size, self.connection.dim_q, device=device)
        v = v / v.norm(dim=1, keepdim=True)
        
        # 路径 A -> B
        q_a = q_center + magnitude * direction_a
        q_ab = q_a + magnitude * direction_b
        
        v_ab = self.connection.parallel_transport(q_center, q_a, v)
        v_ab = self.connection.parallel_transport(q_a, q_ab, v_ab)
        
        # 路径 B -> A
        q_b = q_center + magnitude * direction_b
        q_ba = q_b + magnitude * direction_a
        
        v_ba = self.connection.parallel_transport(q_center, q_b, v)
        v_ba = self.connection.parallel_transport(q_b, q_ba, v_ba)
        
        # 非交换性 = 差异范数
        commutator_norm = (v_ab - v_ba).norm(dim=1)
        
        return commutator_norm


# ============================================
#   测试
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("L2 Connection v2 测试")
    print("=" * 60)
    
    dim_q = 8
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    conn = ConnectionV2(dim_q).to(device)
    
    # 测试1: 平行移动
    print("\n[Test 1] 平行移动")
    q_from = torch.randn(batch_size, dim_q, device=device)
    q_to = q_from + torch.randn(batch_size, dim_q, device=device) * 0.5
    v = torch.randn(batch_size, dim_q, device=device)
    
    v_transported = conn.parallel_transport(q_from, q_to, v)
    print(f"  输入向量范数: {v.norm(dim=1).mean():.4f}")
    print(f"  输出向量范数: {v_transported.norm(dim=1).mean():.4f}")
    
    # 测试2: 曲率估计
    print("\n[Test 2] 曲率估计")
    q_samples = torch.randn(batch_size, dim_q, device=device)
    curvature = conn.compute_curvature(q_samples)
    print(f"  曲率范数均值: {curvature.mean():.6f}")
    
    # 测试3: Holonomy Loop
    print("\n[Test 3] Holonomy Loop")
    q_center = torch.randn(batch_size, dim_q, device=device)
    holonomy_error, _ = conn.holonomy_loop(q_center, radius=0.1)
    print(f"  Holonomy 误差: {holonomy_error.mean():.6f}")
    
    # 测试4: 非阿贝尔性
    print("\n[Test 4] 非阿贝尔性探针")
    probe = NonAbelianProbe(conn)
    dir_a = torch.zeros(batch_size, dim_q, device=device)
    dir_b = torch.zeros(batch_size, dim_q, device=device)
    dir_a[:, 0] = 1.0
    dir_b[:, 1] = 1.0
    
    comm_norm = probe.measure_commutativity(q_center, dir_a, dir_b)
    print(f"  非交换性强度: {comm_norm.mean():.6f}")
    
    # 测试5: 损失函数
    print("\n[Test 5] 正则化损失")
    curv_loss = conn.curvature_loss(q_samples)
    holo_loss = conn.holonomy_loss(q_samples)
    print(f"  曲率损失: {curv_loss.item():.6f}")
    print(f"  Holonomy 损失: {holo_loss.item():.6f}")
    
    print("\n✓ L2 Connection v2 测试完成")

