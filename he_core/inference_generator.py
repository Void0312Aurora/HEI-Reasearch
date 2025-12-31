"""
L4 推断生成元 (Inference Generator)

理论基础 (理论基础-7/自监督.md):
- L4 推断生成元: 不确定性推断与条件化
- 基于 Markov category 的概率论抽象
- 支持自由能最小化的变分推断

核心功能:
1. 变分推断: q(z|x) 近似 p(z|x)
2. 预测误差计算: 用于A3统一势函数
3. 主动推断: 行动选择最小化期望自由能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class InferenceState:
    """推断状态"""
    mean: torch.Tensor       # 后验均值 [batch, dim_z]
    logvar: torch.Tensor     # 后验对数方差 [batch, dim_z]
    sample: torch.Tensor     # 采样值 [batch, dim_z]
    kl_divergence: torch.Tensor  # KL散度 [batch]


class VariationalEncoder(nn.Module):
    """
    变分编码器
    
    将观测映射到潜变量的后验分布
    q(z|x) = N(mu(x), sigma(x))
    """
    
    def __init__(self, dim_input: int, dim_z: int, hidden_dim: int = 128):
        super().__init__()
        self.dim_z = dim_z
        
        self.encoder = nn.Sequential(
            nn.Linear(dim_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.fc_mean = nn.Linear(hidden_dim, dim_z)
        self.fc_logvar = nn.Linear(hidden_dim, dim_z)
        
        # 初始化为接近先验
        nn.init.zeros_(self.fc_mean.weight)
        nn.init.zeros_(self.fc_mean.bias)
        nn.init.constant_(self.fc_logvar.bias, -2.0)  # 小方差
        
    def forward(self, x: torch.Tensor) -> InferenceState:
        """
        编码观测为后验分布
        
        Args:
            x: 观测 [batch, dim_input]
            
        Returns:
            InferenceState: 后验分布和采样
        """
        h = self.encoder(x)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h).clamp(-10, 2)  # 防止数值问题
        
        # 重参数化采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mean + std * eps
        
        # KL散度 (vs 标准正态先验)
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1)
        
        return InferenceState(
            mean=mean,
            logvar=logvar,
            sample=sample,
            kl_divergence=kl
        )


class GenerativeDecoder(nn.Module):
    """
    生成解码器
    
    从潜变量重建观测
    p(x|z) = N(mu(z), sigma_obs)
    """
    
    def __init__(self, dim_z: int, dim_output: int, hidden_dim: int = 128):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(dim_z, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim_output)
        )
        
        # 观测噪声 (可学习)
        self.log_sigma_obs = nn.Parameter(torch.zeros(1))
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解码潜变量为观测
        
        Args:
            z: 潜变量 [batch, dim_z]
            
        Returns:
            mean: 重建均值 [batch, dim_output]
            sigma: 观测噪声 [1]
        """
        mean = self.decoder(z)
        sigma = self.log_sigma_obs.exp()
        return mean, sigma


class WorldModel(nn.Module):
    """
    世界模型
    
    预测下一时刻的潜变量
    p(z_t+1 | z_t, a_t) = N(mu(z_t, a_t), sigma_trans)
    """
    
    def __init__(self, dim_z: int, dim_action: int, hidden_dim: int = 128):
        super().__init__()
        self.dim_z = dim_z
        
        self.transition = nn.Sequential(
            nn.Linear(dim_z + dim_action, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.fc_mean = nn.Linear(hidden_dim, dim_z)
        self.fc_logvar = nn.Linear(hidden_dim, dim_z)
        
        # 初始化为近恒等变换
        nn.init.zeros_(self.fc_mean.weight)
        nn.init.zeros_(self.fc_mean.bias)
        nn.init.constant_(self.fc_logvar.bias, -3.0)
        
    def forward(self, z: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测下一时刻状态
        
        Args:
            z: 当前潜变量 [batch, dim_z]
            action: 行动 [batch, dim_action]
            
        Returns:
            mean: 预测均值 [batch, dim_z]
            logvar: 预测对数方差 [batch, dim_z]
        """
        h = self.transition(torch.cat([z, action], dim=1))
        mean = z + self.fc_mean(h)  # 残差连接
        logvar = self.fc_logvar(h).clamp(-10, 2)
        return mean, logvar


class InferenceGenerator(nn.Module):
    """
    L4 推断生成元
    
    整合变分推断、世界模型和主动推断
    实现自由能最小化的完整推断机制
    """
    
    def __init__(self, 
                 dim_obs: int,
                 dim_z: int,
                 dim_action: int,
                 hidden_dim: int = 128,
                 beta: float = 1.0):
        super().__init__()
        
        self.dim_obs = dim_obs
        self.dim_z = dim_z
        self.dim_action = dim_action
        self.beta = beta  # KL权重
        
        # 组件
        self.encoder = VariationalEncoder(dim_obs, dim_z, hidden_dim)
        self.decoder = GenerativeDecoder(dim_z, dim_obs, hidden_dim)
        self.world_model = WorldModel(dim_z, dim_action, hidden_dim)
        
        # 先验偏好 (用于主动推断)
        self.prior_preference = nn.Parameter(torch.zeros(dim_z))
        self.prior_precision = nn.Parameter(torch.ones(dim_z))
        
    def encode(self, obs: torch.Tensor) -> InferenceState:
        """编码观测"""
        return self.encoder(obs)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码潜变量"""
        mean, _ = self.decoder(z)
        return mean
    
    def compute_elbo(self, 
                     obs: torch.Tensor, 
                     inference_state: Optional[InferenceState] = None) -> Dict[str, torch.Tensor]:
        """
        计算证据下界 (ELBO)
        
        ELBO = E_q[log p(x|z)] - beta * KL(q(z|x) || p(z))
        
        Args:
            obs: 观测 [batch, dim_obs]
            inference_state: 可选的预计算推断状态
            
        Returns:
            dict: elbo, reconstruction_loss, kl_loss
        """
        if inference_state is None:
            inference_state = self.encoder(obs)
            
        z = inference_state.sample
        
        # 重建损失
        recon_mean, sigma = self.decoder(z)
        recon_loss = 0.5 * ((obs - recon_mean).pow(2) / sigma.pow(2) + 
                           2 * torch.log(sigma)).sum(dim=1)
        
        # KL 散度
        kl_loss = inference_state.kl_divergence
        
        # ELBO
        elbo = -recon_loss - self.beta * kl_loss
        
        return {
            'elbo': elbo.mean(),
            'reconstruction_loss': recon_loss.mean(),
            'kl_loss': kl_loss.mean(),
            'z': z,
        }
    
    def compute_free_energy(self, 
                            obs: torch.Tensor,
                            inference_state: Optional[InferenceState] = None) -> torch.Tensor:
        """
        计算变分自由能 F
        
        F = -ELBO = E_q[-log p(x|z)] + beta * KL(q(z|x) || p(z))
        
        这是A3统一自监督势的概率论版本
        """
        result = self.compute_elbo(obs, inference_state)
        return -result['elbo']
    
    def predict_next(self, 
                     z: torch.Tensor, 
                     action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测下一时刻状态
        """
        return self.world_model(z, action)
    
    def compute_expected_free_energy(self,
                                     z: torch.Tensor,
                                     actions: torch.Tensor,
                                     horizon: int = 1) -> torch.Tensor:
        """
        计算期望自由能 (EFE)
        
        用于主动推断的行动选择
        
        EFE = E_q[H[p(o|z')]] + E_q[KL(q(z'|o) || q(z'))]
            = Ambiguity + Risk
            
        简化版: EFE ≈ -E[log p(z' | preference)]
        
        Args:
            z: 当前潜变量 [batch, dim_z]
            actions: 候选行动 [batch, num_actions, dim_action] 或 [batch, dim_action]
            horizon: 预测步数
            
        Returns:
            efe: 期望自由能 [batch] 或 [batch, num_actions]
        """
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
            
        batch_size, num_actions, _ = actions.shape
        device = z.device
        
        efe = torch.zeros(batch_size, num_actions, device=device)
        
        for a_idx in range(num_actions):
            z_current = z
            
            for t in range(horizon):
                # 预测下一状态
                z_mean, z_logvar = self.world_model(z_current, actions[:, a_idx])
                
                # 采样
                z_std = (0.5 * z_logvar).exp()
                z_next = z_mean + z_std * torch.randn_like(z_std)
                
                # 计算与偏好的偏离 (风险项)
                preference = self.prior_preference.unsqueeze(0)
                precision = self.prior_precision.unsqueeze(0)
                risk = 0.5 * (precision * (z_next - preference).pow(2)).sum(dim=1)
                
                # 模糊性项 (预测不确定性)
                ambiguity = 0.5 * z_logvar.sum(dim=1)
                
                efe[:, a_idx] += risk + ambiguity
                
                z_current = z_next
                
        return efe.squeeze(-1) if num_actions == 1 else efe
    
    def select_action(self,
                      z: torch.Tensor,
                      action_candidates: torch.Tensor,
                      temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        主动推断行动选择
        
        选择最小化EFE的行动
        
        Args:
            z: 当前潜变量 [batch, dim_z]
            action_candidates: 候选行动 [batch, num_actions, dim_action]
            temperature: softmax 温度
            
        Returns:
            action: 选择的行动 [batch, dim_action]
            action_probs: 行动概率 [batch, num_actions]
        """
        # 计算EFE
        efe = self.compute_expected_free_energy(z, action_candidates)
        
        # 概率 = softmax(-EFE / temperature)
        action_probs = torch.softmax(-efe / temperature, dim=-1)
        
        # 采样行动
        action_indices = torch.multinomial(action_probs, 1).squeeze(-1)
        
        # 获取选择的行动
        batch_size = z.shape[0]
        actions = action_candidates[torch.arange(batch_size), action_indices]
        
        return actions, action_probs
    
    def update_beliefs(self,
                       obs: torch.Tensor,
                       prior_mean: Optional[torch.Tensor] = None,
                       prior_logvar: Optional[torch.Tensor] = None) -> InferenceState:
        """
        贝叶斯信念更新
        
        结合先验和观测更新后验
        """
        # 从观测获取似然参数
        likelihood_state = self.encoder(obs)
        
        if prior_mean is None:
            return likelihood_state
            
        # 贝叶斯融合 (高斯情况)
        prior_var = (prior_logvar.exp() if prior_logvar is not None 
                    else torch.ones_like(likelihood_state.logvar))
        likelihood_var = likelihood_state.logvar.exp()
        
        # 后验精度 = 先验精度 + 似然精度
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / likelihood_var)
        posterior_mean = posterior_var * (prior_mean / prior_var + 
                                          likelihood_state.mean / likelihood_var)
        posterior_logvar = posterior_var.log()
        
        # 重采样
        std = posterior_var.sqrt()
        sample = posterior_mean + std * torch.randn_like(std)
        
        # 计算更新后的KL
        kl = -0.5 * (1 + posterior_logvar - posterior_mean.pow(2) - posterior_var).sum(dim=1)
        
        return InferenceState(
            mean=posterior_mean,
            logvar=posterior_logvar,
            sample=sample,
            kl_divergence=kl
        )


# ============================================
#   与 SoulEntity 集成的适配器
# ============================================

class InferenceIntegrator(nn.Module):
    """
    将 InferenceGenerator 与 SoulEntity 的几何动力学集成
    
    桥接概率推断和几何演化
    """
    
    def __init__(self, 
                 inference_gen: InferenceGenerator,
                 dim_q: int):
        super().__init__()
        self.inference_gen = inference_gen
        self.dim_q = dim_q
        
        # q -> obs 映射
        self.q_to_obs = nn.Linear(dim_q, inference_gen.dim_obs)
        
        # z -> q 映射 (用于驱动动力学)
        self.z_to_force = nn.Linear(inference_gen.dim_z, dim_q)
        
    def get_inference_force(self, 
                           q: torch.Tensor,
                           target_obs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取基于推断的驱动力
        
        如果有目标观测，则驱动q向能产生该观测的方向移动
        """
        if target_obs is None:
            # 自主模式: 使用偏好作为目标
            z_target = self.inference_gen.prior_preference.unsqueeze(0).expand(q.shape[0], -1)
        else:
            # 有观测: 编码目标
            state = self.inference_gen.encode(target_obs)
            z_target = state.mean
            
        # 当前q的观测
        obs = self.q_to_obs(q)
        current_state = self.inference_gen.encode(obs)
        z_current = current_state.mean
        
        # 力 = z_target - z_current (梯度下降方向)
        dz = z_target - z_current
        force = self.z_to_force(dz)
        
        return force
    
    def compute_prediction_error(self, 
                                 q: torch.Tensor,
                                 next_q: torch.Tensor,
                                 action: torch.Tensor) -> torch.Tensor:
        """
        计算预测误差
        
        用于A3统一势函数
        """
        # 编码当前和下一状态
        obs = self.q_to_obs(q)
        next_obs = self.q_to_obs(next_q)
        
        current_state = self.inference_gen.encode(obs)
        next_state = self.inference_gen.encode(next_obs)
        
        # 预测
        pred_mean, pred_logvar = self.inference_gen.predict_next(current_state.mean, action)
        
        # 预测误差
        error = (next_state.mean - pred_mean).pow(2).sum(dim=1)
        
        return error


# ============================================
#   测试
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("L4 Inference Generator 测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dim_obs = 32
    dim_z = 16
    dim_action = 8
    batch_size = 16
    
    inf_gen = InferenceGenerator(dim_obs, dim_z, dim_action).to(device)
    
    # 测试1: 编码
    print("\n[Test 1] 变分编码")
    obs = torch.randn(batch_size, dim_obs, device=device)
    state = inf_gen.encode(obs)
    print(f"  均值: {state.mean.shape}")
    print(f"  方差: {state.logvar.exp().mean():.4f}")
    print(f"  KL: {state.kl_divergence.mean():.4f}")
    
    # 测试2: 解码
    print("\n[Test 2] 解码重建")
    recon = inf_gen.decode(state.sample)
    recon_error = (obs - recon).pow(2).mean()
    print(f"  重建误差: {recon_error:.4f}")
    
    # 测试3: ELBO
    print("\n[Test 3] ELBO计算")
    elbo_result = inf_gen.compute_elbo(obs)
    print(f"  ELBO: {elbo_result['elbo']:.4f}")
    print(f"  重建损失: {elbo_result['reconstruction_loss']:.4f}")
    print(f"  KL损失: {elbo_result['kl_loss']:.4f}")
    
    # 测试4: 自由能
    print("\n[Test 4] 变分自由能")
    F = inf_gen.compute_free_energy(obs)
    print(f"  F = {F:.4f}")
    
    # 测试5: 世界模型预测
    print("\n[Test 5] 世界模型预测")
    action = torch.randn(batch_size, dim_action, device=device)
    z_next_mean, z_next_logvar = inf_gen.predict_next(state.mean, action)
    print(f"  预测均值: {z_next_mean.shape}")
    print(f"  预测不确定性: {z_next_logvar.exp().mean():.4f}")
    
    # 测试6: 期望自由能
    print("\n[Test 6] 期望自由能 (EFE)")
    num_actions = 5
    action_candidates = torch.randn(batch_size, num_actions, dim_action, device=device)
    efe = inf_gen.compute_expected_free_energy(state.mean, action_candidates)
    print(f"  EFE shape: {efe.shape}")
    print(f"  EFE mean: {efe.mean():.4f}")
    
    # 测试7: 行动选择
    print("\n[Test 7] 主动推断行动选择")
    selected_action, probs = inf_gen.select_action(state.mean, action_candidates)
    print(f"  选择的行动: {selected_action.shape}")
    print(f"  行动概率: {probs[0]}")
    
    # 测试8: 集成器
    print("\n[Test 8] SoulEntity集成器")
    dim_q = 32
    integrator = InferenceIntegrator(inf_gen, dim_q).to(device)
    
    q = torch.randn(batch_size, dim_q, device=device)
    force = integrator.get_inference_force(q)
    print(f"  推断力: {force.shape}")
    print(f"  力的范数: {force.norm(dim=1).mean():.4f}")
    
    print("\n✓ L4 Inference Generator 测试完成")

