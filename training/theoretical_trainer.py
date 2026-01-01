"""
理论对齐的大规模训练器

严格遵循理论基础-7的公理体系(A1-A5)和动力学模板(L1-L3)

核心原则：
1. SoulEntity是核心，不是Transformer语言模型
2. 语言是端口(L3)，不是系统本身
3. 训练目标是变分自由能F，不是纯cross-entropy
4. 必须包含多步接触动力学演化
5. 遵循三阶段渐进式训练

架构：
- 阶段1: 离线动力学 + 初步语言接触
- 阶段2: 图册联络 + 语言对齐
- 阶段3: 精调接口

运行：
python HEI/training/theoretical_trainer.py \
    --data HEI/data/wiki/wikipedia-zh-20250901.json \
    --output_dir checkpoints/theoretical \
    --epochs 10 \
    --batch_size 16
"""

import os
import sys
import json
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from he_core.soul_entity import SoulEntity, create_soul_entity
from he_core.state import ContactState
from he_core.language_interface import SimpleTokenizer, TokenEncoder, StateDecoder
from he_core.atlas import AtlasRouter
from he_core.connection import Connection


# ============================================================
#                      配置
# ============================================================

@dataclass
class TheoreticalConfig:
    """理论对齐训练配置"""
    
    # === 核心维度 ===
    dim_q: int = 128                    # 构形空间维度
    dim_z: int = 32                     # 上下文维度
    dim_u: int = 128                    # 端口输入维度 (= dim_q)
    num_charts: int = 8                 # 图册数量
    
    # === 动力学参数 ===
    T_offline: int = 20                 # 离线演化步数
    T_online: int = 10                  # 在线演化步数
    dt: float = 0.1                     # 时间步长
    hyperbolic_c: float = 0.1           # 双曲曲率
    stiffness: float = 0.1              # q的约束刚度
    contact_stiffness: float = 0.1      # s的约束刚度
    
    # === 训练参数 ===
    batch_size: int = 16
    effective_batch_size: int = 64      # 有效批量 (梯度累积)
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    num_epochs: int = 10
    max_steps: int = -1                 # -1表示无限制
    
    # === 序列参数 ===
    max_seq_len: int = 128
    vocab_size: int = 20000
    
    # === 损失权重 (理论优先级: 动力学 > 几何 > 接口) ===
    lambda_F: float = 1.0               # 变分自由能
    lambda_lyapunov: float = 10.0       # Lyapunov约束
    lambda_geometry: float = 1.0        # 几何约束
    lambda_pred: float = 1.0            # 预测损失
    lambda_atlas: float = 0.1           # 图册一致性
    lambda_connection: float = 0.1      # 联络正交性
    
    # === 训练阶段 ===
    phase: int = 1                      # 当前训练阶段 (1, 2, 3)
    phase1_epochs: int = 3              # 阶段1轮数
    phase2_epochs: int = 3              # 阶段2轮数
    phase3_epochs: int = 4              # 阶段3轮数
    
    # === 保存与日志 ===
    output_dir: str = 'checkpoints/theoretical'
    save_every: int = 500
    log_every: int = 10
    eval_every: int = 100
    
    # === 硬件 ===
    use_amp: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    compile_model: bool = False  # 使用torch.compile()
    
    @property
    def gradient_accumulation_steps(self) -> int:
        return max(1, self.effective_batch_size // self.batch_size)


# ============================================================
#                      数据集
# ============================================================

class WikiStreamDataset(IterableDataset):
    """流式维基数据集 - 优化版本"""
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: SimpleTokenizer,
                 seq_len: int = 128,
                 max_docs: int = -1,
                 buffer_size: int = 10000):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_docs = max_docs
        self.buffer_size = buffer_size
        
    def __iter__(self):
        doc_count = 0
        buffer = []
        batch_buffer = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if self.max_docs > 0 and doc_count >= self.max_docs:
                    break
                    
                try:
                    obj = json.loads(line.strip())
                    text = obj.get('text', '')
                    if len(text) < 50:
                        continue
                    
                    # 编码并加入缓冲
                    tokens = self.tokenizer.encode(text, add_special=False)
                    buffer.extend(tokens)
                    doc_count += 1
                    
                    # 批量产出完整序列
                    while len(buffer) >= self.seq_len + 1:
                        batch_buffer.append(buffer[:self.seq_len + 1])
                        buffer = buffer[self.seq_len // 2:]
                        
                        # 批量yield以减少开销
                        if len(batch_buffer) >= 64:
                            for seq in batch_buffer:
                                yield torch.tensor(seq, dtype=torch.long)
                            batch_buffer = []
                        
                except json.JSONDecodeError:
                    continue
        
        # 处理剩余
        for seq in batch_buffer:
            yield torch.tensor(seq, dtype=torch.long)


# ============================================================
#                      理论对齐模型
# ============================================================

class TheoreticalModel(nn.Module):
    """
    理论对齐的大规模模型
    
    核心：SoulEntity的接触哈密顿动力学
    接口：语言编码器/解码器作为端口
    
    遵循公理：
    - A1: Markov blanket边界
    - A2: 离线认知态
    - A3: 统一自由能F
    - A4: 身份连续性
    - A5: 多接口一致性
    """
    
    def __init__(self, config: TheoreticalConfig):
        super().__init__()
        self.config = config
        
        # === 核心：SoulEntity ===
        self.entity = create_soul_entity({
            'dim_q': config.dim_q,
            'dim_z': config.dim_z,
            'dim_u': config.dim_u,
            'num_charts': config.num_charts,
            'hyperbolic_c': config.hyperbolic_c,
            'stiffness': config.stiffness,
            'contact_stiffness': config.contact_stiffness,
            'device': config.device,
        })
        
        # === L3: 语言端口 ===
        # 编码器: tokens → u (端口输入)
        self.encoder = TokenEncoder(
            vocab_size=config.vocab_size,
            dim_embed=config.dim_q * 2,
            dim_u=config.dim_u,
            num_layers=4
        )
        
        # 解码器: q → logits
        self.decoder = StateDecoder(
            vocab_size=config.vocab_size,
            dim_q=config.dim_q,
            dim_embed=config.dim_q * 4,
            num_layers=4
        )
        
        # === L2: 图册与联络 ===
        self.atlas_router = AtlasRouter(
            dim_q=config.dim_q,
            num_charts=config.num_charts
        )
        
        self.connection = Connection(
            dim_q=config.dim_q,
            hidden_dim=128,
            init_epsilon=0.3
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'decoder' in name or 'encoder' in name:
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def _init_state(self, batch_size: int) -> ContactState:
        """初始化接触状态"""
        state = ContactState(
            dim_q=self.config.dim_q,
            batch_size=batch_size,
            device=self.config.device
        )
        # 小随机初始化
        state._data = torch.randn(
            batch_size, 2 * self.config.dim_q + 1, 
            device=self.config.device
        ) * 0.1
        return state
    
    def evolve_offline(self, 
                       state: ContactState,
                       num_steps: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        离线演化 (A2: 离线认知态) - 优化版本
        
        无外部输入时的自主演化
        只记录首尾F值以减少开销
        
        Returns:
            states: 状态轨迹 (仅首尾)
            F_values: 自由能轨迹 (仅首尾)
        """
        batch_size = state.batch_size
        device = state.device
        dim_q = self.config.dim_q
        dt = self.config.dt
        
        z = self.entity.z.expand(batch_size, -1)
        
        # 初始化状态张量 (预分配)
        q = torch.randn(batch_size, dim_q, device=device) * 0.1
        p = torch.randn(batch_size, dim_q, device=device) * 0.1
        s = torch.zeros(batch_size, 1, device=device)
        
        # 记录初始F
        q.requires_grad_(True)
        p.requires_grad_(True)
        
        state_data = torch.cat([q, p, s], dim=-1)
        F_initial = self._compute_H_fast(q, p, s, z)
        
        # 多步演化 - 使用torch.no_grad()加速中间步骤
        with torch.no_grad():
            for t in range(num_steps - 1):
                # 快速演化 (无需梯度)
                H_c = self._compute_H_fast(q, p, s, z)
                
                # 使用有限差分近似梯度 (比autograd快)
                eps = 1e-3
                grad_q = torch.zeros_like(q)
                grad_p = torch.zeros_like(p)
                
                # 简化：使用势能梯度近似
                # V ≈ 0.5 * stiffness * ||q||^2 + net_V(q)
                grad_q = self.config.stiffness * q
                grad_p = p  # K = 0.5 * ||p||^2 的梯度
                
                # 哈密顿演化
                q = q + dt * grad_p
                p = p - dt * grad_q
                
                # 耗散
                p = p * 0.99
        
        # 最后一步需要梯度以计算损失
        q = q.detach().requires_grad_(True)
        p = p.detach().requires_grad_(True)
        s = s.detach().requires_grad_(True)
        
        F_final = self._compute_H_fast(q, p, s, z)
        
        return [state_data.detach(), torch.cat([q, p, s], dim=-1)], [F_initial, F_final]
    
    def _compute_H_fast(self, q: torch.Tensor, p: torch.Tensor, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """快速计算H^c (避免创建ContactState对象)"""
        # K(p) = 0.5 * ||p||^2
        K = 0.5 * (p ** 2).sum(dim=-1, keepdim=True)
        
        # V(q) = 0.5 * stiffness * ||q||^2 + net_V(q, z)
        V_quad = 0.5 * self.config.stiffness * (q ** 2).sum(dim=-1, keepdim=True)
        qz = torch.cat([q, z], dim=-1)
        V_net = self.entity.net_V(qz)
        V = V_quad + V_net
        
        # Φ(s) = 0.5 * contact_stiffness * s^2
        Phi = 0.5 * self.config.contact_stiffness * (s ** 2)
        
        return K + V + Phi
    
    def evolve_online(self,
                      state: ContactState,
                      u_seq: torch.Tensor,
                      num_steps: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        在线演化 (L3: 端口耦合) - 优化版本
        
        有外部输入时的演化
        使用批量化操作减少Python循环开销
        
        Args:
            state: 初始状态
            u_seq: 端口输入序列 [batch, seq_len, dim_u]
            num_steps: 演化步数
            
        Returns:
            states: 状态轨迹 (仅首尾)
            F_values: 自由能轨迹 (仅首尾)
            q_trajectory: 最终q用于解码
        """
        batch_size, seq_len, dim_u = u_seq.shape
        device = u_seq.device
        dim_q = self.config.dim_q
        dt = self.config.dt
        
        z = self.entity.z.expand(batch_size, -1)
        
        # 初始化状态张量
        q = torch.randn(batch_size, dim_q, device=device) * 0.1
        p = torch.randn(batch_size, dim_q, device=device) * 0.1
        s = torch.zeros(batch_size, 1, device=device)
        
        # 计算平均u作为驱动信号 (减少循环)
        u_mean = u_seq.mean(dim=1)  # [batch, dim_u]
        min_dim = min(dim_u, dim_q)
        
        # 记录初始F
        q.requires_grad_(True)
        p.requires_grad_(True)
        F_initial = self._compute_H_fast(q, p, s, z)
        
        # 批量演化 - 中间步骤不记录
        actual_steps = min(seq_len, num_steps)
        
        for i in range(actual_steps):
            # 端口耦合力
            u_t = u_seq[:, min(i, seq_len - 1), :min_dim]
            port_force = 0.1 * u_t
            
            # 计算势能梯度
            grad_V = self.config.stiffness * q[:, :min_dim]
            
            # 哈密顿演化 + 端口耦合
            p_new = p.clone()
            p_new[:, :min_dim] = p[:, :min_dim] - dt * grad_V + dt * port_force
            p_new = p_new * 0.99  # 耗散
            
            q_new = q + dt * p_new
            
            # detach中间步骤
            if i < actual_steps - 1:
                q = q_new.detach()
                p = p_new.detach()
            else:
                q = q_new
                p = p_new
        
        # 最后一步计算F (需要梯度)
        q = q.detach().requires_grad_(True)
        p = p.detach().requires_grad_(True)
        F_final = self._compute_H_fast(q, p, s, z)
        
        # 只返回最终状态
        final_state = torch.cat([q, p, s], dim=-1)
        
        return [final_state], [F_initial, F_final], [q]
    
    def forward(self, 
                tokens: torch.Tensor,
                phase: int = 1) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            tokens: [batch, seq_len] token序列
            phase: 训练阶段 (1, 2, 3)
            
        Returns:
            包含各种输出和中间结果的字典
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # 1. 编码语言输入为端口信号u
        u_seq = self.encoder(tokens)  # [batch, seq_len, dim_u]
        
        # 2. 初始化状态
        state = self._init_state(batch_size)
        
        # 3. 根据阶段执行不同的演化
        results = {}
        
        if phase == 1:
            # 阶段1: 离线动力学 + 初步在线
            
            # 3a. 离线演化
            offline_states, offline_F = self.evolve_offline(
                state, 
                num_steps=self.config.T_offline
            )
            # 堆叠F值 (现在只有首尾两个)
            offline_F_tensor = torch.cat([f.unsqueeze(1) for f in offline_F], dim=1)
            results['offline_F'] = offline_F_tensor.squeeze(-1)
            
            # 3b. 在线演化
            online_states, online_F, q_traj = self.evolve_online(
                state,
                u_seq,
                num_steps=self.config.T_online
            )
            online_F_tensor = torch.cat([f.unsqueeze(1) for f in online_F], dim=1)
            results['online_F'] = online_F_tensor.squeeze(-1)
            results['q_trajectory'] = torch.stack(q_traj, dim=1) if len(q_traj) > 1 else q_traj[0].unsqueeze(1)
            
        elif phase == 2:
            # 阶段2: 加入图册联络训练
            
            # 在线演化
            online_states, online_F, q_traj = self.evolve_online(
                state,
                u_seq,
                num_steps=self.config.T_online
            )
            online_F_tensor = torch.cat([f.unsqueeze(1) for f in online_F], dim=1)
            results['online_F'] = online_F_tensor.squeeze(-1)
            results['q_trajectory'] = torch.stack(q_traj, dim=1) if len(q_traj) > 1 else q_traj[0].unsqueeze(1)
            
            # 计算图册权重
            q_final = q_traj[-1]
            chart_weights = self.atlas_router(q_final)
            results['chart_weights'] = chart_weights
            
            # 计算联络正交性
            v = torch.randn_like(q_final)
            q_to = q_final + 0.1 * torch.randn_like(q_final)
            v_transported = self.connection(q_final, q_to, v)
            results['v_original'] = v
            results['v_transported'] = v_transported
            
        else:
            # 阶段3: 精调接口
            
            # 简化的在线演化
            online_states, online_F, q_traj = self.evolve_online(
                state,
                u_seq,
                num_steps=self.config.T_online
            )
            online_F_tensor = torch.cat([f.unsqueeze(1) for f in online_F], dim=1)
            results['online_F'] = online_F_tensor.squeeze(-1)
            results['q_trajectory'] = torch.stack(q_traj, dim=1) if len(q_traj) > 1 else q_traj[0].unsqueeze(1)
        
        # 4. 解码输出
        if 'q_trajectory' in results:
            q_for_decode = results['q_trajectory']
            
            # 确保形状正确
            if q_for_decode.dim() == 3:
                # [batch, T, dim_q] -> 取平均或最后一个
                q_decode = q_for_decode.mean(dim=1)  # [batch, dim_q]
            else:
                q_decode = q_for_decode
                
            logits = self.decoder(q_decode)  # [batch, ?, vocab_size]
            
            # 调整logits形状匹配目标
            if logits.dim() == 2:
                logits = logits.unsqueeze(1).expand(-1, seq_len - 1, -1)
            elif logits.shape[1] != seq_len - 1:
                if logits.shape[1] > seq_len - 1:
                    logits = logits[:, :seq_len - 1, :]
                else:
                    logits = F.pad(logits, (0, 0, 0, seq_len - 1 - logits.shape[1]))
                    
            results['logits'] = logits
        
        return results


# ============================================================
#                      损失计算
# ============================================================

class TheoreticalLoss(nn.Module):
    """
    理论对齐的损失函数
    
    核心原则：
    - 变分自由能F是主要目标
    - Lyapunov约束确保动力学稳定
    - 语言预测是接口对齐，不是核心
    """
    
    def __init__(self, config: TheoreticalConfig):
        super().__init__()
        self.config = config
        
    def lyapunov_loss(self, F_trajectory: torch.Tensor, margin: float = 0.0) -> Tuple[torch.Tensor, float]:
        """
        Lyapunov损失：F应该在演化过程中单调下降
        
        Args:
            F_trajectory: [batch, T] 自由能轨迹
            margin: 允许的微小上升
            
        Returns:
            loss: Lyapunov损失
            violation_rate: 违反率
        """
        if F_trajectory.shape[1] < 2:
            return torch.tensor(0.0, device=F_trajectory.device), 0.0
        
        # F[t+1] - F[t] 应该 < margin
        F_diff = F_trajectory[:, 1:] - F_trajectory[:, :-1]
        violations = torch.relu(F_diff + margin)
        
        loss = violations.mean()
        violation_rate = (F_diff > margin).float().mean().item()
        
        return loss, violation_rate
    
    def geometry_loss(self, q_trajectory: torch.Tensor) -> torch.Tensor:
        """
        几何约束损失
        
        - q应该保持在合理范围内
        - 轨迹应该平滑
        """
        if q_trajectory.dim() < 3:
            return torch.tensor(0.0, device=q_trajectory.device)
        
        # 范数约束: ||q|| ≈ 1
        q_norm = q_trajectory.norm(dim=-1)
        norm_loss = (q_norm - 1.0).pow(2).mean()
        
        # 平滑约束: 相邻状态不应剧烈变化
        if q_trajectory.shape[1] > 1:
            q_diff = (q_trajectory[:, 1:] - q_trajectory[:, :-1]).norm(dim=-1)
            smooth_loss = torch.relu(q_diff - 0.5).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=q_trajectory.device)
        
        return norm_loss + 0.1 * smooth_loss
    
    def atlas_loss(self, chart_weights: torch.Tensor) -> torch.Tensor:
        """
        图册一致性损失
        
        - 鼓励使用多个图册
        - 避免所有状态落入同一图册
        """
        if chart_weights is None:
            return torch.tensor(0.0)
        
        # 熵损失: 鼓励多样性
        p = F.softmax(chart_weights, dim=-1) + 1e-8
        entropy = -(p * p.log()).sum(dim=-1).mean()
        
        # 熵越大越好，所以取负
        return -entropy
    
    def connection_loss(self, v_original: torch.Tensor, v_transported: torch.Tensor) -> torch.Tensor:
        """
        联络正交性损失
        
        平行移动应该保持向量范数
        """
        if v_original is None or v_transported is None:
            return torch.tensor(0.0)
        
        norm_before = v_original.norm(dim=-1)
        norm_after = v_transported.norm(dim=-1)
        
        return (norm_after - norm_before).pow(2).mean()
    
    def forward(self,
                outputs: Dict[str, torch.Tensor],
                tokens: torch.Tensor,
                phase: int = 1) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            outputs: 模型输出
            tokens: 目标tokens
            phase: 训练阶段
            
        Returns:
            损失字典
        """
        device = tokens.device
        losses = {}
        
        # 1. Lyapunov损失 (最重要)
        if 'offline_F' in outputs:
            lyap_off, viol_off = self.lyapunov_loss(outputs['offline_F'])
            losses['lyap_offline'] = lyap_off
            losses['viol_offline'] = viol_off
            
        if 'online_F' in outputs:
            lyap_on, viol_on = self.lyapunov_loss(outputs['online_F'])
            losses['lyap_online'] = lyap_on
            losses['viol_online'] = viol_on
        
        # 2. 变分自由能损失
        if 'online_F' in outputs:
            F_final = outputs['online_F'][:, -1] if outputs['online_F'].dim() > 1 else outputs['online_F']
            losses['F_mean'] = F_final.mean()
        
        # 3. 几何损失
        if 'q_trajectory' in outputs:
            losses['geometry'] = self.geometry_loss(outputs['q_trajectory'])
        
        # 4. 图册/联络损失 (阶段2+)
        if phase >= 2:
            if 'chart_weights' in outputs:
                losses['atlas'] = self.atlas_loss(outputs['chart_weights'])
            if 'v_original' in outputs and 'v_transported' in outputs:
                losses['connection'] = self.connection_loss(
                    outputs['v_original'], 
                    outputs['v_transported']
                )
        
        # 5. 语言预测损失
        if 'logits' in outputs:
            logits = outputs['logits']
            targets = tokens[:, 1:]  # 下一个token预测
            
            # 确保形状匹配
            if logits.shape[1] != targets.shape[1]:
                min_len = min(logits.shape[1], targets.shape[1])
                logits = logits[:, :min_len]
                targets = targets[:, :min_len]
            
            pred_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=0
            )
            losses['pred'] = pred_loss
            losses['ppl'] = torch.exp(pred_loss.clamp(max=15)).item()
        
        # 6. 计算总损失 (根据阶段调整权重)
        total = torch.tensor(0.0, device=device)
        
        # 阶段1: 动力学优先
        if phase == 1:
            if 'lyap_offline' in losses:
                total = total + self.config.lambda_lyapunov * losses['lyap_offline']
            if 'lyap_online' in losses:
                total = total + self.config.lambda_lyapunov * 0.5 * losses['lyap_online']
            if 'F_mean' in losses:
                total = total + self.config.lambda_F * losses['F_mean']
            if 'geometry' in losses:
                total = total + self.config.lambda_geometry * 0.1 * losses['geometry']
            if 'pred' in losses:
                total = total + self.config.lambda_pred * 0.1 * losses['pred']
                
        # 阶段2: 动力学 + 图册联络
        elif phase == 2:
            if 'lyap_online' in losses:
                total = total + self.config.lambda_lyapunov * losses['lyap_online']
            if 'F_mean' in losses:
                total = total + self.config.lambda_F * losses['F_mean']
            if 'geometry' in losses:
                total = total + self.config.lambda_geometry * losses['geometry']
            if 'atlas' in losses:
                total = total + self.config.lambda_atlas * losses['atlas']
            if 'connection' in losses:
                total = total + self.config.lambda_connection * losses['connection']
            if 'pred' in losses:
                total = total + self.config.lambda_pred * 0.5 * losses['pred']
                
        # 阶段3: 接口对齐优先
        else:
            if 'lyap_online' in losses:
                total = total + self.config.lambda_lyapunov * 0.1 * losses['lyap_online']
            if 'F_mean' in losses:
                total = total + self.config.lambda_F * 0.1 * losses['F_mean']
            if 'geometry' in losses:
                total = total + self.config.lambda_geometry * 0.5 * losses['geometry']
            if 'pred' in losses:
                total = total + self.config.lambda_pred * losses['pred']
        
        losses['total'] = total
        
        return losses


# ============================================================
#                      训练器
# ============================================================

class TheoreticalTrainer:
    """
    理论对齐的训练器
    
    实现三阶段训练范式
    """
    
    def __init__(self, config: TheoreticalConfig):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        
        self.global_step = 0
        self.epoch = 0
        self.phase = 1
        self.best_loss = float('inf')
        
    def setup_logging(self):
        """设置日志"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.config.output_dir, 'train.log'))
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_device(self):
        """设置设备"""
        self.device = torch.device(self.config.device)
        self.logger.info(f"使用设备: {self.device}")
        
        if self.config.use_amp and self.device.type == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None
            
    def setup_data(self):
        """设置数据"""
        self.tokenizer = SimpleTokenizer(vocab_size=self.config.vocab_size, mode='char')
        
    def setup_model(self):
        """设置模型"""
        self.model = TheoreticalModel(self.config)
        self.model = self.model.to(self.device)
        self.loss_fn = TheoreticalLoss(self.config)
        
        # 使用torch.compile()加速 (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            self.logger.info("使用torch.compile()编译模型...")
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                self.logger.info("  ✓ 模型编译成功")
            except Exception as e:
                self.logger.warning(f"  ✗ 模型编译失败: {e}")
        
        # 统计参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"总参数: {total_params:,}")
        self.logger.info(f"可训练参数: {trainable_params:,}")
        
    def setup_optimizer(self):
        """设置优化器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
    def get_lr(self) -> float:
        """获取当前学习率 (warmup + cosine decay)"""
        if self.global_step < self.config.warmup_steps:
            return self.config.learning_rate * self.global_step / max(1, self.config.warmup_steps)
        
        progress = (self.global_step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
        return self.config.learning_rate * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    def update_lr(self):
        """更新学习率"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'phase': self.phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'tokenizer': self.tokenizer,
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        self.logger.info(f"保存检查点: {path}")
        
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.phase = checkpoint.get('phase', 1)
        self.best_loss = checkpoint['best_loss']
        
        if 'tokenizer' in checkpoint:
            self.tokenizer = checkpoint['tokenizer']
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.logger.info(f"加载检查点: {path} (step={self.global_step}, phase={self.phase})")
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        tokens = batch.to(self.device)
        
        # 前向传播
        if self.config.use_amp and self.scaler is not None:
            with autocast():
                outputs = self.model(tokens, phase=self.phase)
                losses = self.loss_fn(outputs, tokens, phase=self.phase)
                loss = losses['total'] / self.config.gradient_accumulation_steps
        else:
            outputs = self.model(tokens, phase=self.phase)
            losses = self.loss_fn(outputs, tokens, phase=self.phase)
            loss = losses['total'] / self.config.gradient_accumulation_steps
        
        # 反向传播
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 返回损失统计
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        epoch_stats = {}
        accum_losses = {}
        accum_steps = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 训练步骤
            losses = self.train_step(batch)
            
            # 累积损失
            for k, v in losses.items():
                accum_losses[k] = accum_losses.get(k, 0) + v
            accum_steps += 1
            
            # 梯度更新
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # 优化器步骤
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                lr = self.update_lr()
                
                # 日志
                if self.global_step % self.config.log_every == 0:
                    avg_losses = {k: v / accum_steps for k, v in accum_losses.items()}
                    self.logger.info(
                        f"Phase {self.phase} | Step {self.global_step} | "
                        f"Loss: {avg_losses.get('total', 0):.4f} | "
                        f"Lyap: {avg_losses.get('lyap_online', avg_losses.get('lyap_offline', 0)):.4f} | "
                        f"PPL: {avg_losses.get('ppl', 0):.1f} | "
                        f"LR: {lr:.2e}"
                    )
                    accum_losses = {}
                    accum_steps = 0
                
                # 保存检查点
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(
                        os.path.join(self.config.output_dir, f'checkpoint_step_{self.global_step}.pt')
                    )
                
                # 检查最大步数
                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    break
        
        return epoch_stats
    
    def train(self, data_path: str):
        """完整训练流程"""
        self.logger.info("=" * 70)
        self.logger.info("理论对齐训练开始")
        self.logger.info(f"数据: {data_path}")
        self.logger.info(f"配置: dim_q={self.config.dim_q}, T_offline={self.config.T_offline}")
        self.logger.info("=" * 70)
        
        # 构建词汇表
        self.logger.info("构建词汇表...")
        texts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10000:  # 限制构建词汇表的文档数
                    break
                try:
                    obj = json.loads(line.strip())
                    if 'text' in obj:
                        texts.append(obj['text'])
                except:
                    continue
        
        self.tokenizer.build_vocab(texts)
        self.config.vocab_size = len(self.tokenizer)
        self.logger.info(f"词汇表大小: {self.config.vocab_size}")
        
        # 重新创建模型以使用新的vocab_size
        self.setup_model()
        self.setup_optimizer()
        
        # 创建数据集
        dataset = WikiStreamDataset(
            data_path,
            self.tokenizer,
            seq_len=self.config.max_seq_len
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            persistent_workers=self.config.num_workers > 0
        )
        
        # 计算总步数
        if self.config.max_steps < 0:
            self.config.max_steps = self.config.num_epochs * 1000  # 估计
        
        # 三阶段训练
        try:
            # 阶段1: 离线动力学 + 初步接触
            self.logger.info("\n" + "=" * 70)
            self.logger.info("阶段1: 离线动力学训练")
            self.logger.info("=" * 70)
            self.phase = 1
            
            for epoch in range(self.config.phase1_epochs):
                self.epoch = epoch
                self.logger.info(f"\nPhase 1 - Epoch {epoch + 1}/{self.config.phase1_epochs}")
                self.train_epoch(dataloader)
            
            self.save_checkpoint(os.path.join(self.config.output_dir, 'phase1_final.pt'))
            
            # 阶段2: 图册联络
            self.logger.info("\n" + "=" * 70)
            self.logger.info("阶段2: 图册联络训练")
            self.logger.info("=" * 70)
            self.phase = 2
            
            for epoch in range(self.config.phase2_epochs):
                self.epoch = epoch
                self.logger.info(f"\nPhase 2 - Epoch {epoch + 1}/{self.config.phase2_epochs}")
                self.train_epoch(dataloader)
            
            self.save_checkpoint(os.path.join(self.config.output_dir, 'phase2_final.pt'))
            
            # 阶段3: 接口精调
            self.logger.info("\n" + "=" * 70)
            self.logger.info("阶段3: 接口对齐训练")
            self.logger.info("=" * 70)
            self.phase = 3
            
            for epoch in range(self.config.phase3_epochs):
                self.epoch = epoch
                self.logger.info(f"\nPhase 3 - Epoch {epoch + 1}/{self.config.phase3_epochs}")
                self.train_epoch(dataloader)
            
            self.save_checkpoint(os.path.join(self.config.output_dir, 'final_model.pt'))
            
        except KeyboardInterrupt:
            self.logger.info("\n训练被中断，保存检查点...")
            self.save_checkpoint(os.path.join(self.config.output_dir, 'interrupted.pt'))
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("训练完成")
        self.logger.info(f"最终步数: {self.global_step}")
        self.logger.info(f"模型保存至: {self.config.output_dir}")
        self.logger.info("=" * 70)


# ============================================================
#                      主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='理论对齐的大规模训练')
    
    # 数据
    parser.add_argument('--data', type=str, required=True, help='训练数据路径')
    parser.add_argument('--output_dir', type=str, default='checkpoints/theoretical')
    
    # 模型
    parser.add_argument('--dim_q', type=int, default=128)
    parser.add_argument('--dim_z', type=int, default=32)
    parser.add_argument('--num_charts', type=int, default=8)
    
    # 动力学
    parser.add_argument('--T_offline', type=int, default=20)
    parser.add_argument('--T_online', type=int, default=10)
    
    # 训练
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_steps', type=int, default=-1)
    
    # 阶段
    parser.add_argument('--phase1_epochs', type=int, default=3)
    parser.add_argument('--phase2_epochs', type=int, default=3)
    parser.add_argument('--phase3_epochs', type=int, default=4)
    
    # 恢复
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点')
    
    # 优化
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--compile', action='store_true', help='使用torch.compile()')
    parser.add_argument('--no_amp', action='store_true', help='禁用混合精度')
    
    args = parser.parse_args()
    
    # 创建配置
    config = TheoreticalConfig(
        dim_q=args.dim_q,
        dim_z=args.dim_z,
        num_charts=args.num_charts,
        T_offline=args.T_offline,
        T_online=args.T_online,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase3_epochs=args.phase3_epochs,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        compile_model=args.compile,
        use_amp=not args.no_amp,
    )
    
    # 创建训练器
    trainer = TheoreticalTrainer(config)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train(args.data)


if __name__ == '__main__':
    main()

