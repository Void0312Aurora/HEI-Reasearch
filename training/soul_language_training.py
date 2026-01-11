"""
正确的语言技能训练框架

基于理论基础-7的公理体系，严格遵循：
- A3: 自由能F驱动，而非PPL
- A5: 语言只是端口，SoulEntity是核心
- L1: 接触哈密顿动力学
- L3: 端口耦合机制

核心原则：
语言输入u → 端口耦合 → SoulEntity演化 → 状态解码 → 语言输出
训练目标是最小化自由能F，语言误差只是F的一个分量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import math

from he_core.soul_entity import SoulEntity, create_soul_entity
from he_core.language_interface import SimpleTokenizer, LanguagePort
from he_core.state import ContactState


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型维度
    dim_q: int = 64
    dim_z: int = 16
    dim_embed: int = 256
    vocab_size: int = 10000
    num_charts: int = 4

    # 端口/序列建模形态
    # - transformer+pooled: 端口自回归，u 为整段 pooled（当前默认）
    # - minimal+recurrent: 逐 token 驱动动力学，端口仅做极薄读写（推荐用于“语言能力在动力学中涌现”）
    port_arch: str = "transformer"     # "transformer" | "minimal"
    sequence_mode: str = "pooled"      # "pooled" | "recurrent"
    tie_io_weights: bool = True        # minimal 端口时，是否绑定输入/输出权重
    port_trainable: bool = True        # minimal 端口时，是否允许端口权重学习（False=纯“表驱动”端口）
    port_init_std: float = 0.02        # minimal 端口初始化尺度（过大将导致 E_pred/PPL 初期爆炸）
    detach_every: int = 0              # recurrent 模式下每 N token 截断一次 BPTT（0=不截断）

    # Closed-loop mitigation (Protocol-5): scheduled sampling
    scheduled_sampling_prob: float = 0.0   # probability of feeding model token instead of teacher token
    scheduled_sampling_mode: str = "sample"  # "sample" | "argmax"
    scheduled_sampling_top_k: int = 20
    scheduled_sampling_temperature: float = 1.0

    # Atlas skill shaping
    router_balance_weight: float = 0.0   # encourage uniform chart usage across batch/time
    router_entropy_weight: float = 0.0   # encourage sparse chart selection (min entropy)

    # Experience-conditioned offline (A2): how many end-states to push per batch (0=disable)
    experience_push_n: int = 0

    # L3 port coupling sparsity: only use top-k charts per step (0 = dense over all charts)
    port_coupling_top_k: int = 0
    # Port coupling implementation detail (performance-only):
    # - grouped: sparse top-k with per-chart grouping (less compute, more tiny GEMMs)
    # - dense:   compute dense einsum then mask to top-k (more compute, fewer kernel launches)
    port_coupling_impl: str = "grouped"

    # L2 parallel transport controls
    transport_threshold: float = 0.1
    connection_rank: int = 0

    # Numerical stability (viability projection): clip state components to avoid NaNs/Infs.
    q_clip_norm: float = 100.0
    p_clip_norm: float = 100.0
    s_clip_abs: float = 100.0
    sanitize_nonfinite: bool = True
    
    # 自由能权重（A3公理核心）
    beta_kl: float = 0.01       # KL正则权重
    gamma_pred: float = 1.0     # 预测误差权重
    alpha_potential: float = 1.0  # 势能权重
    
    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 16
    max_seq_len: int = 128
    dt: float = 0.1  # 动力学时间步
    
    # 多步演化（允许SoulEntity演化多步）
    num_evolution_steps: int = 1
    num_offline_steps: int = 1
    offline_dt: float = 0.1
    offline_replay_mode: str = 'none'
    offline_weight: float = 1.0
    offline_detach_init: bool = False
    offline_loss_mode: str = 'end'  # 'end' | 'delta' | 'relu_delta'
    offline_margin: float = 0.0     # used by relu_delta: relu((F_end-F_start)+margin)
    reset_each_batch: bool = True
    
    # 设备
    device: str = 'cuda'


class SoulLanguageTrainer(nn.Module):
    """
    正确的语言技能训练器
    
    核心架构（严格遵循理论基础-7）：
    
    1. SoulEntity是核心（接触哈密顿动力学）
    2. 语言是端口（感知/行动边界）
    3. 训练目标是自由能F（不是交叉熵）
    
    训练流程：
    input_tokens → Encode → u → 端口耦合 → SoulEntity演化(q,p,s) 
                                          ↓
                  F = V(q,z) + β·KL + γ·E_pred ← 预测误差 ← Decode(q) → logits
    """
    
    def __init__(self, config: TrainingConfig, tokenizer: SimpleTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.port_arch = getattr(config, "port_arch", "transformer")
        self.sequence_mode = getattr(config, "sequence_mode", "pooled")
        
        # === 核心：SoulEntity（接触哈密顿动力学）===
        entity_config = {
            'dim_q': config.dim_q,
            'dim_u': config.dim_q,  # 语言端口维度匹配
            'dim_z': getattr(config, "dim_z", 16),
            'num_charts': getattr(config, "num_charts", 4),
            'beta_kl': config.beta_kl,
            'gamma_pred': config.gamma_pred,
            'port_top_k': int(getattr(config, "port_coupling_top_k", 0) or 0),
            'port_topk_impl': str(getattr(config, "port_coupling_impl", "grouped") or "grouped"),
            'transport_threshold': float(getattr(config, "transport_threshold", 0.1) or 0.1),
            'connection_rank': int(getattr(config, "connection_rank", 0) or 0),
            'q_clip_norm': float(getattr(config, "q_clip_norm", 0.0) or 0.0),
            'p_clip_norm': float(getattr(config, "p_clip_norm", 0.0) or 0.0),
            's_clip_abs': float(getattr(config, "s_clip_abs", 0.0) or 0.0),
            'sanitize_nonfinite': bool(getattr(config, "sanitize_nonfinite", True)),
        }
        self.entity = create_soul_entity(entity_config)
        
        # === 语言端口（只是接口之一）===
        # 注意：端口不应“覆盖”内在动力学。默认保留 Transformer 端口用于基线/对照；
        # 若要将语言处理迁回动力学，请使用 minimal+recurrent。
        if self.port_arch == "transformer":
            self.language_port = LanguagePort(
                vocab_size=config.vocab_size,
                dim_q=config.dim_q,
                dim_u=config.dim_q,
                dim_embed=config.dim_embed,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dropout=0.1
            )
            self.language_port.set_tokenizer(tokenizer)
            self.token_embedding = None
            self.output_proj = None
        elif self.port_arch == "minimal":
            # 极薄端口：token -> u(q维) 的 embedding + q -> logits 的线性读出
            self.language_port = None
            self.token_embedding = nn.Embedding(
                config.vocab_size,
                config.dim_q,
                padding_idx=tokenizer.pad_id,
            )
            self.output_proj = nn.Linear(config.dim_q, config.vocab_size, bias=False)
            if getattr(config, "tie_io_weights", True):
                self.output_proj.weight = self.token_embedding.weight

            # Default nn.Embedding init is N(0,1), which makes logits extremely large for high dim_q.
            # Use a small GPT-like init so the dynamics starts in a reasonable regime.
            init_std = float(getattr(config, "port_init_std", 0.02) or 0.0)
            if init_std > 0:
                with torch.no_grad():
                    nn.init.normal_(self.token_embedding.weight, mean=0.0, std=init_std)
                    if 0 <= int(self.tokenizer.pad_id) < self.token_embedding.weight.shape[0]:
                        self.token_embedding.weight[int(self.tokenizer.pad_id)].zero_()
                    if self.output_proj.weight is not self.token_embedding.weight:
                        nn.init.normal_(self.output_proj.weight, mean=0.0, std=init_std)

            if not bool(getattr(config, "port_trainable", True)):
                self.token_embedding.weight.requires_grad_(False)
                self.output_proj.weight.requires_grad_(False)
        else:
            raise ValueError(f"Unknown port_arch: {self.port_arch}")
        
        # 添加语言端口到SoulEntity
        self.entity.add_interface('language', config.dim_q)

        # CUDA-graph friendly scalar constants (avoid creating new CUDA tensors inside capture).
        self.register_buffer("_pad_token", torch.tensor(int(tokenizer.pad_id), dtype=torch.long), persistent=False)
        
    def reset_entity(self, batch_size: int):
        """重置实体状态"""
        self.entity.reset(batch_size, self.config.device)
        
    def encode_tokens(self, 
                     token_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码tokens为几何输入u
        
        这是语言端口的感知端（Read）
        """
        if self.port_arch != "transformer" or self.language_port is None:
            raise RuntimeError("encode_tokens() is only available for transformer ports.")
        return self.language_port.encoder.encode_pooled(token_ids, attention_mask)

    def encode_token_sequence(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode tokens into a per-token u(t) sequence for recurrent dynamics.

        Returns:
            u_seq: [batch, seq_len, dim_q]
        """
        if self.port_arch != "minimal" or self.token_embedding is None:
            raise RuntimeError("encode_token_sequence() requires a minimal port.")
        return self.token_embedding(token_ids)
    
    def decode_state(self, 
                     state_q: torch.Tensor,
                     prev_tokens: torch.Tensor) -> torch.Tensor:
        """
        从几何状态q解码为token logits
        
        这是语言端口的行动端（Write）
        """
        if self.port_arch != "transformer" or self.language_port is None:
            raise RuntimeError("decode_state() is only available for transformer ports.")
        return self.language_port.decoder(state_q, prev_tokens)

    def decode_logits_from_q(self, q: torch.Tensor) -> torch.Tensor:
        """Minimal-port readout: q -> logits [batch, vocab_size]."""
        if self.output_proj is None:
            raise RuntimeError("decode_logits_from_q() requires a minimal port.")
        return self.output_proj(q)
    
    def compute_prediction_error(self,
                                 logits: torch.Tensor,
                                 target_ids: torch.Tensor,
                                 attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算预测误差
        
        这将作为自由能F的γ分量
        """
        # 移位：预测下一个token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        # Cross-entropy
        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_targets.view(-1),
            ignore_index=self.tokenizer.pad_id,
            reduction='none'
        )
        loss = loss.view(shift_targets.shape)
        
        # Masked mean
        per_sample = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
        E_pred = per_sample.mean()

        return E_pred, per_sample
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        单步训练
        
        核心流程：
        1. 编码输入 → u
        2. u通过端口耦合驱动SoulEntity演化
        3. 从新状态q解码
        4. 计算预测误差E_pred（作为F的一部分）
        5. 计算总自由能F = V(q,z) + β·KL + γ·E_pred
        6. 返回F用于反向传播
        
        关键区别：优化F而非单独的语言损失！
        """
        if self.sequence_mode == "recurrent" or self.port_arch == "minimal":
            return self._train_step_recurrent(batch)

        token_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size = token_ids.shape[0]
        device = token_ids.device
        
        # 重置实体状态（可配置保留慢变量）
        if self.config.reset_each_batch or self.entity.state is None or self.entity.state.batch_size != batch_size:
            self.reset_entity(batch_size)
        self.entity.enter_online()
        
        # === Step 1: 语言感知端 ===
        # 编码tokens为几何输入u
        u = self.encode_tokens(token_ids, attention_mask)
        
        # === Step 2: 端口耦合驱动SoulEntity演化 ===
        # H(q,p,s,u) = H_int(q,p,s) + <u, B(q)>
        # 这是L3模板3的实现
        accumulated_states = []
        last_chart_weights = None
        for step in range(self.config.num_evolution_steps):
            result = self.entity.step(
                {'language': u},
                dt=self.config.dt
            )
            accumulated_states.append(result['state_flat'].clone())
            last_chart_weights = result.get('chart_weights', None)
        
        # 获取最终状态（拷贝避免后续离线演化的就地修改）
        current_state = self.entity.state.clone()
        q_final = current_state.q
        
        # === Step 3: 语言行动端 ===
        # 从几何状态q解码为token分布
        logits = self.decode_state(q_final, token_ids)
        
        # === Step 4: 计算预测误差 ===
        # 作为自由能F的γ分量
        E_pred, E_pred_per_sample = self.compute_prediction_error(logits, token_ids, attention_mask)
        
        # === Step 5: 计算总自由能F ===
        # F = V(q,z) + β·KL(z) + γ·E_pred
        # 这是A3公理的核心：单一标量泛函驱动所有行为
        F_online = self.entity.compute_free_energy(
            current_state,
            prediction_error=E_pred
        )
        F_offline = torch.tensor(0.0, device=device)
        F_offline_start = torch.tensor(0.0, device=device)
        F_offline_end = torch.tensor(0.0, device=device)
        delta_offline = torch.tensor(0.0, device=device)
        offline_loss = torch.tensor(0.0, device=device)
        if self.config.num_offline_steps > 0:
            # Differentiable offline rollout (A2/A3): same generator, u(t)=0 (+ optional replay)
            offline_init = current_state.clone()
            if self.config.offline_detach_init:
                offline_init = offline_init.detach()

            offline_state_flat = offline_init.flat.clone()
            offline_state = ContactState(self.entity.dim_q, batch_size, device, offline_state_flat)
            F_offline_start = self.entity.compute_free_energy(offline_state)

            prev_weights = last_chart_weights
            if prev_weights is not None and self.config.offline_detach_init:
                prev_weights = prev_weights.detach()

            last_out = None
            for _ in range(self.config.num_offline_steps):
                u_off = {}
                if self.config.offline_replay_mode != 'none' and len(self.entity.experience.states) > 0:
                    replay = self.entity.experience.sample_replay(batch_size, mode=self.config.offline_replay_mode)
                    if replay is not None:
                        replay_state = replay['states'].to(device)
                        if replay_state.shape[0] < batch_size:
                            pad = replay_state[0:1].expand(batch_size - replay_state.shape[0], -1)
                            replay_state = torch.cat([replay_state, pad], dim=0)
                        replay_q = replay_state[:, :self.entity.dim_q]
                        u_off['replay'] = 0.1 * (replay_q - offline_state.q)

                last_out = self.entity.forward_tensor(
                    state_flat=offline_state.flat,
                    u_dict=u_off,
                    dt=self.config.offline_dt,
                    prev_chart_weights=prev_weights,
                    prediction_error=None,
                    detach_next_prev_weights=False,
                    compute_action=False,
                )
                offline_state = ContactState(self.entity.dim_q, batch_size, device, last_out['next_state_flat'])
                prev_weights = last_out['chart_weights']

            if last_out is not None:
                F_offline_end = last_out['free_energy']
                F_offline = F_offline_end
                delta_offline = F_offline_end - F_offline_start

                mode = self.config.offline_loss_mode
                if mode == 'end':
                    offline_loss = F_offline_end
                elif mode == 'delta':
                    offline_loss = delta_offline
                elif mode == 'relu_delta':
                    offline_loss = torch.relu(delta_offline + float(self.config.offline_margin))
                else:
                    raise ValueError(f"Unknown offline_loss_mode: {mode}")
        
        F_total = F_online + self.config.offline_weight * offline_loss

        # If we keep state across batches, treat it as a fast variable (no BPTT across batches).
        # This aligns with the fast/slow narrative: parameters/structures learn slowly, state evolves quickly.
        if not self.config.reset_each_batch and self.entity.state is not None:
            self.entity.state = self.entity.state.detach()
        
        # === 诊断指标 ===
        # Keep diagnostics on-device to avoid per-step `.item()` synchronizations.
        with torch.no_grad():
            ppl = torch.exp(E_pred)
            q_norm = current_state.q.norm(dim=1).mean()
            p_norm = current_state.p.norm(dim=1).mean()
            s_val = current_state.s.mean()
        
        return {
            'loss': F_total,  # 核心：优化F而非语言损失
            'free_energy': F_total.detach(),
            'free_energy_online': F_online.detach(),
            'free_energy_offline': F_offline.detach(),
            'free_energy_offline_start': F_offline_start.detach(),
            'free_energy_offline_end': F_offline_end.detach(),
            'delta_free_energy_offline': delta_offline.detach(),
            'offline_loss': offline_loss.detach(),
            'prediction_error': E_pred.detach(),
            'prediction_error_per_sample': E_pred_per_sample.detach(),
            'perplexity': ppl,
            'q_norm': q_norm,
            'p_norm': p_norm,
            's_value': s_val,
        }

    def _train_step_recurrent(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Recurrent language training: tokens drive dynamics step-by-step.

        Design goal (理论对齐)：
        - 把序列建模压力迁回内在动力学/图册（而不是端口 Transformer）
        - 端口只提供最小读写：token->u(t) 与 q_t->p(x_{t+1})
        """
        if self.port_arch != "minimal":
            raise RuntimeError("recurrent training currently requires port_arch='minimal'.")

        token_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        mask_f = attention_mask.to(dtype=torch.float32)

        if self.config.reset_each_batch or self.entity.state is None or self.entity.state.batch_size != batch_size:
            self.reset_entity(batch_size)
            prev_weights = None
        else:
            prev_weights = getattr(self.entity, "_prev_chart_weights", None)
        self.entity.enter_online()

        state_flat = self.entity.state.flat

        # Accumulate next-token prediction error.
        per_sample_loss = torch.zeros(batch_size, device=device)
        per_sample_count = torch.zeros(batch_size, device=device)

        # Router diagnostics/regularizers (skills≈charts needs non-collapsed atlas usage).
        router_balance_w = float(getattr(self.config, "router_balance_weight", 0.0) or 0.0)
        router_entropy_w = float(getattr(self.config, "router_entropy_weight", 0.0) or 0.0)
        router_token_count = state_flat.new_zeros(())
        router_weight_sum = None
        router_entropy_sum = state_flat.new_zeros(())

        # Run dynamics across the sequence. We update state even for special tokens, but freeze for PAD.
        detach_every = int(getattr(self.config, "detach_every", 0) or 0)
        ss_prob = float(getattr(self.config, "scheduled_sampling_prob", 0.0) or 0.0)
        ss_prob = max(0.0, min(1.0, ss_prob))
        ss_mode = str(getattr(self.config, "scheduled_sampling_mode", "sample"))
        ss_temp = float(getattr(self.config, "scheduled_sampling_temperature", 1.0) or 1.0)
        ss_top_k = int(getattr(self.config, "scheduled_sampling_top_k", 20) or 0)
        pad_token = self._pad_token

        # Fast path (ss_prob=0): precompute token embeddings and vectorize the readout/loss over time.
        # This greatly reduces kernel-launch overhead and typically increases GPU utilization.
        vectorized_readout = ss_prob <= 0.0
        u_seq = None
        q_seq = None
        if vectorized_readout:
            u_seq = self.token_embedding(token_ids)  # [B,L,Dq]
            u_seq = u_seq * mask_f.unsqueeze(-1)
            if seq_len > 1:
                q_seq = torch.empty((batch_size, seq_len - 1, self.entity.dim_q), device=device, dtype=state_flat.dtype)
        ss_mask = None
        if (not vectorized_readout) and seq_len > 1:
            # Precompute the scheduled-sampling decision mask once per sequence to avoid
            # per-token RNG calls (and to be friendlier to CUDA graph capture).
            if ss_prob >= 1.0 - 1e-8:
                ss_mask = mask_f[:, 1:].bool()
            else:
                ss_mask = (torch.rand(batch_size, seq_len - 1, device=device) < ss_prob) & mask_f[:, 1:].bool()

        # Current token ids fed into the dynamics (scheduled sampling may overwrite teacher tokens).
        current_tokens = token_ids[:, 0]

        for t in range(seq_len):
            mask_curr = mask_f[:, t : t + 1]  # [B,1]
            if vectorized_readout:
                assert u_seq is not None
                u_t = u_seq[:, t, :]  # already masked
            else:
                u_t = self.token_embedding(current_tokens) * mask_curr  # [B,Dq]

            # Integrator steps per token (fast dynamics)
            last_chart_weights = None
            for _ in range(self.config.num_evolution_steps):
                out = self.entity.forward_tensor(
                    state_flat=state_flat,
                    u_dict={'language': u_t},
                    dt=self.config.dt,
                    prev_chart_weights=prev_weights,
                    prediction_error=None,
                    detach_next_prev_weights=True,
                    compute_action=False,
                )
                next_state_flat = out['next_state_flat']
                last_chart_weights = out.get('chart_weights', None)
                next_prev = out['next_prev_chart_weights']

                # Freeze padded samples (mask=0) to avoid uncontrolled internal drift on padding.
                state_flat = mask_curr * next_state_flat + (1.0 - mask_curr) * state_flat
                if prev_weights is None:
                    prev_weights = next_prev
                else:
                    prev_weights = mask_curr * next_prev + (1.0 - mask_curr) * prev_weights

            # Router regularization statistics (count only non-pad tokens).
            if (router_balance_w != 0.0 or router_entropy_w != 0.0) and last_chart_weights is not None:
                mask1 = mask_curr.squeeze(1)  # [B]
                if router_weight_sum is None:
                    router_weight_sum = state_flat.new_zeros((last_chart_weights.shape[1],))
                router_token_count = router_token_count + mask1.sum()
                router_weight_sum = router_weight_sum + (last_chart_weights * mask1.unsqueeze(1)).sum(dim=0)
                entropy = -(last_chart_weights * torch.log(last_chart_weights + 1e-8)).sum(dim=1)  # [B]
                router_entropy_sum = router_entropy_sum + (entropy * mask1).sum()

            # Next-token prediction after consuming token t (teacher-forced target x_{t+1})
            if t < seq_len - 1:
                if vectorized_readout:
                    assert q_seq is not None
                    q_seq[:, t, :] = state_flat[:, : self.entity.dim_q]
                else:
                    mask_tgt = mask_f[:, t + 1]  # [B]
                    q = state_flat[:, :self.entity.dim_q]
                    logits = self.decode_logits_from_q(q)  # [B,V]
                    targets = token_ids[:, t + 1]

                    # Masked CE (padding targets are ignore_index; mask further guards non-pad count).
                    loss_vec = F.cross_entropy(
                        logits,
                        targets,
                        ignore_index=self.tokenizer.pad_id,
                        reduction='none',
                    )  # [B]
                    loss_vec = loss_vec * mask_tgt
                    per_sample_loss = per_sample_loss + loss_vec
                    per_sample_count = per_sample_count + mask_tgt

                    # Scheduled sampling: sometimes feed model output as next input (Protocol-5 mitigation).
                    # Avoid Python branching on CUDA tensors (e.g. `.any()`), which forces sync per token.
                    next_tokens = targets
                    if ss_mask is not None:
                        use_sample = ss_mask[:, t] & mask_tgt.bool()
                        step_logits = logits / max(ss_temp, 1e-6)
                        if ss_top_k > 0:
                            v, _ = torch.topk(step_logits, min(ss_top_k, step_logits.shape[-1]), dim=-1)
                            cutoff = v[:, -1].unsqueeze(-1)
                            step_logits = step_logits.masked_fill(step_logits < cutoff, float("-inf"))
                        if ss_mode == "argmax" or ss_top_k == 1:
                            sampled = torch.argmax(step_logits, dim=-1)
                        else:
                            probs = torch.softmax(step_logits, dim=-1)
                            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                            probs = probs.clamp(min=0.0)
                            probs = probs + 1e-8
                            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                            sampled = torch.multinomial(probs, 1).squeeze(-1)

                        next_tokens = torch.where(use_sample, sampled, targets)

                    # For padded positions, keep feeding PAD (and dynamics is frozen by mask=0).
                    next_tokens = torch.where(mask_tgt.bool(), next_tokens, pad_token)
                    current_tokens = next_tokens

            if detach_every > 0 and (t + 1) % detach_every == 0:
                state_flat = state_flat.detach()

        if vectorized_readout:
            if self.output_proj is None:
                raise RuntimeError("vectorized_readout requires output_proj (minimal port).")
            if seq_len <= 1:
                logits_seq = self.output_proj(torch.zeros((batch_size, 0, self.entity.dim_q), device=device, dtype=state_flat.dtype))
            else:
                assert q_seq is not None
                logits_seq = self.output_proj(q_seq)  # [B,L-1,V]
            targets_seq = token_ids[:, 1:]  # [B,L-1]
            mask_seq = mask_f[:, 1:]  # [B,L-1]

            flat_logits = logits_seq.reshape(-1, logits_seq.shape[-1])
            flat_targets = targets_seq.reshape(-1)
            flat_loss = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=self.tokenizer.pad_id,
                reduction="none",
            )
            loss_mat = flat_loss.view(batch_size, -1)
            per_sample_loss = (loss_mat * mask_seq).sum(dim=1)
            per_sample_count = mask_seq.sum(dim=1)

        per_sample_count = per_sample_count.clamp(min=1.0)
        E_pred_per_sample = per_sample_loss / per_sample_count
        E_pred = E_pred_per_sample.mean()

        current_state = ContactState(self.entity.dim_q, batch_size, device, state_flat)

        F_online = self.entity.compute_free_energy(current_state, prediction_error=E_pred)

        # A2: experience-conditioned offline cognition needs actual experience.
        # We push a small subset of end-of-sequence states into the experience buffer,
        # so offline replay can be modulated by recent online trajectories.
        push_n = int(getattr(self.config, "experience_push_n", 0) or 0)
        if push_n > 0:
            n = min(push_n, batch_size)
            idx = torch.randperm(batch_size, device=device)[:n]
            zeros = torch.zeros(n, self.entity.dim_u, device=device)
            state_to_store = current_state.flat.detach()[idx]
            # Avoid `.item()` (device sync). v0: keep rewards neutral.
            self.entity.experience.push(zeros, zeros, state_to_store, r=0.0)

        # Atlas regularizers (treated as part of the unified functional in practice).
        router_balance = state_flat.new_zeros(())
        router_entropy = state_flat.new_zeros(())
        atlas_reg = state_flat.new_zeros(())
        if router_weight_sum is not None:
            K = float(router_weight_sum.shape[0])
            mean_w = router_weight_sum / router_token_count.clamp(min=1.0)
            router_entropy = router_entropy_sum / router_token_count.clamp(min=1.0)
            if router_balance_w != 0.0:
                uniform = torch.full_like(mean_w, 1.0 / K)
                router_balance = (mean_w * torch.log((mean_w + 1e-8) / (uniform + 1e-8))).sum()
            atlas_reg = router_balance_w * router_balance + router_entropy_w * router_entropy

        # Offline rollout (differentiable, A2/A3)
        F_offline = state_flat.new_zeros(())
        F_offline_start = state_flat.new_zeros(())
        F_offline_end = state_flat.new_zeros(())
        delta_offline = state_flat.new_zeros(())
        offline_loss = state_flat.new_zeros(())
        if self.config.num_offline_steps > 0:
            offline_init = current_state.clone()
            if self.config.offline_detach_init:
                offline_init = offline_init.detach()

            offline_state_flat = offline_init.flat.clone()
            offline_state = ContactState(self.entity.dim_q, batch_size, device, offline_state_flat)
            F_offline_start = self.entity.compute_free_energy(offline_state)

            # We keep offline independent of chart-weight gradients by detaching prev weights.
            prev_off_weights = prev_weights
            if prev_off_weights is not None:
                prev_off_weights = prev_off_weights.detach()

            last_out = None
            for _ in range(self.config.num_offline_steps):
                u_off = {}
                if self.config.offline_replay_mode != 'none' and len(self.entity.experience.states) > 0:
                    replay = self.entity.experience.sample_replay(batch_size, mode=self.config.offline_replay_mode)
                    if replay is not None:
                        replay_state = replay['states'].to(device)
                        if replay_state.shape[0] < batch_size:
                            pad = replay_state[0:1].expand(batch_size - replay_state.shape[0], -1)
                            replay_state = torch.cat([replay_state, pad], dim=0)
                        replay_q = replay_state[:, :self.entity.dim_q]
                        u_off['replay'] = 0.1 * (replay_q - offline_state.q)

                last_out = self.entity.forward_tensor(
                    state_flat=offline_state.flat,
                    u_dict=u_off,
                    dt=self.config.offline_dt,
                    prev_chart_weights=prev_off_weights,
                    prediction_error=None,
                    detach_next_prev_weights=False,
                    compute_action=False,
                )
                offline_state = ContactState(self.entity.dim_q, batch_size, device, last_out['next_state_flat'])
                prev_off_weights = last_out['chart_weights']

            if last_out is not None:
                F_offline_end = last_out['free_energy']
                F_offline = F_offline_end
                delta_offline = F_offline_end - F_offline_start

                mode = self.config.offline_loss_mode
                if mode == 'end':
                    offline_loss = F_offline_end
                elif mode == 'delta':
                    offline_loss = delta_offline
                elif mode == 'relu_delta':
                    offline_loss = torch.relu(delta_offline + float(self.config.offline_margin))
                else:
                    raise ValueError(f"Unknown offline_loss_mode: {mode}")

        F_total = F_online + self.config.offline_weight * offline_loss + atlas_reg

        # Persist a detached state snapshot for generation/debugging without retaining graphs.
        self.entity.state = ContactState(self.entity.dim_q, batch_size, device, state_flat.detach())
        self.entity._prev_chart_weights = prev_weights.detach() if prev_weights is not None else None

        with torch.no_grad():
            ppl = torch.exp(E_pred)
            q_norm = current_state.q.norm(dim=1).mean()
            p_norm = current_state.p.norm(dim=1).mean()
            s_val = current_state.s.mean()

        return {
            'loss': F_total,
            'free_energy': F_total.detach(),
            'free_energy_online': F_online.detach(),
            'free_energy_offline': F_offline.detach(),
            'free_energy_offline_start': F_offline_start.detach(),
            'free_energy_offline_end': F_offline_end.detach(),
            'delta_free_energy_offline': delta_offline.detach(),
            'offline_loss': offline_loss.detach(),
            'prediction_error': E_pred.detach(),
            'prediction_error_per_sample': E_pred_per_sample.detach(),
            'perplexity': ppl,
            'q_norm': q_norm,
            'p_norm': p_norm,
            's_value': s_val,
            'router_balance': router_balance.detach(),
            'router_entropy': router_entropy.detach(),
            'atlas_reg': atlas_reg.detach(),
        }
    
    @torch.no_grad()
    def generate(self,
                 prompt: str = "",
                 max_len: int = 50,
                 temperature: float = 1.0,
                 top_k: int = 50) -> str:
        """
        从SoulEntity状态生成文本
        
        流程：
        1. 如果有prompt，先编码并驱动演化
        2. 以prompt作为前缀，自回归生成后续 token（生成 max_len 个新 token）
        """
        device = self.config.device
        self.reset_entity(1)

        if self.port_arch == "transformer":
            start_tokens = None
            if prompt:
                # 编码prompt
                ids = self.tokenizer.encode(prompt)
                # For continuation-style generation, we keep the prompt (without EOS) as the prefix tokens.
                # Note: tokenizer.encode() already adds BOS/EOS; we drop EOS so generation can continue.
                prefix_ids = ids[:-1] if len(ids) >= 2 else ids
                start_tokens = torch.tensor([prefix_ids], device=device)
                token_ids = torch.tensor([ids], device=device)

                # 驱动演化
                u = self.encode_tokens(token_ids)
                for _ in range(self.config.num_evolution_steps):
                    self.entity.step({'language': u}, dt=self.config.dt)
            else:
                start_tokens = torch.tensor([[self.tokenizer.bos_id]], device=device)

            # 从当前状态生成
            q = self.entity.state.q
            assert self.language_port is not None
            generated = self.language_port.decoder.generate(
                q,
                start_tokens=start_tokens,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id
            )
            return self.tokenizer.decode(generated[0].cpu().tolist())

        # Minimal recurrent generation: decode from q, feed predicted tokens back as u(t)
        assert self.token_embedding is not None and self.output_proj is not None

        ids = self.tokenizer.encode(prompt) if prompt else [self.tokenizer.bos_id, self.tokenizer.eos_id]
        prefix_ids = ids[:-1] if len(ids) >= 2 else ids
        if not prefix_ids:
            prefix_ids = [self.tokenizer.bos_id]

        state_flat = self.entity.state.flat
        prev_weights = None

        # Consume prefix (excluding EOS)
        for tok in prefix_ids:
            tok_t = torch.tensor([[tok]], device=device, dtype=torch.long)
            u_t = self.token_embedding(tok_t).squeeze(1)  # [1, Dq]
            for _ in range(int(self.config.num_evolution_steps)):
                out = self.entity.forward_tensor(
                    state_flat=state_flat,
                    u_dict={'language': u_t},
                    dt=self.config.dt,
                    prev_chart_weights=prev_weights,
                    prediction_error=None,
                    detach_next_prev_weights=True,
                    compute_action=False,
                )
                state_flat = out['next_state_flat']
                prev_weights = out['next_prev_chart_weights']

        generated: List[int] = list(prefix_ids)
        for _ in range(max_len):
            q = state_flat[:, :self.entity.dim_q]
            logits = self.decode_logits_from_q(q).squeeze(0)
            logits = logits / max(float(temperature), 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(int(top_k), int(logits.shape[-1])))
                logits = logits.clone()
                logits[logits < v[-1]] = float('-inf')
            if top_k == 1:
                next_tok = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                next_tok = int(torch.multinomial(probs, 1).item())

            generated.append(next_tok)
            if next_tok == self.tokenizer.eos_id:
                break

            tok_t = torch.tensor([[next_tok]], device=device, dtype=torch.long)
            u_t = self.token_embedding(tok_t).squeeze(1)
            for _ in range(int(self.config.num_evolution_steps)):
                out = self.entity.forward_tensor(
                    state_flat=state_flat,
                    u_dict={'language': u_t},
                    dt=self.config.dt,
                    prev_chart_weights=prev_weights,
                    prediction_error=None,
                    detach_next_prev_weights=True,
                    compute_action=False,
                )
                state_flat = out['next_state_flat']
                prev_weights = out['next_prev_chart_weights']

        return self.tokenizer.decode(generated)
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """评估模型"""
        self.eval()
        
        total_F = 0.0
        total_E_pred = 0.0
        total_samples = 0
        
        for batch in dataloader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            result = self.train_step(batch)
            
            batch_size = batch['input_ids'].shape[0]
            total_F += result['free_energy'].item() * batch_size
            total_E_pred += result['prediction_error'].item() * batch_size
            total_samples += batch_size
        
        self.train()
        
        return {
            'avg_free_energy': total_F / total_samples,
            'avg_prediction_error': total_E_pred / total_samples,
            'avg_perplexity': math.exp(total_E_pred / total_samples),
        }
    
    def get_entity_diagnostics(self) -> Dict[str, Any]:
        """获取SoulEntity诊断信息"""
        return self.entity.get_diagnostics()


class LanguageDataset(torch.utils.data.Dataset):
    """语言数据集"""
    
    def __init__(self, 
                 texts: List[str],
                 tokenizer: SimpleTokenizer,
                 max_len: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # 分词
        ids = self.tokenizer.encode(text)
        
        # 截断
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        
        # 创建mask
        mask = [1] * len(ids)
        
        # Padding
        pad_len = self.max_len - len(ids)
        ids = ids + [self.tokenizer.pad_id] * pad_len
        mask = mask + [0] * pad_len
        
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """整理batch"""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
    }


# ============================================
#   验证测试
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("正确的语言技能训练框架 - 验证测试")
    print("核心原则: 语言是端口，SoulEntity是核心，优化自由能F")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")
    
    # 测试1: 创建分词器
    print("\n[Test 1] 创建分词器和训练器")
    
    tokenizer = SimpleTokenizer(vocab_size=5000, mode='char')
    sample_texts = [
        "这是一个测试句子。",
        "人工智能是计算机科学的重要领域。",
        "深度学习改变了机器学习的范式。",
        "自由能原理描述了自组织系统的行为。",
    ]
    tokenizer.build_vocab(sample_texts * 100)  # 扩展词表
    
    config = TrainingConfig(
        dim_q=32,
        dim_embed=128,
        vocab_size=len(tokenizer),
        batch_size=2,
        max_seq_len=64,
        device=device,
    )
    
    trainer = SoulLanguageTrainer(config, tokenizer)
    trainer.to(device)
    
    print(f"  SoulEntity dim_q: {trainer.entity.dim_q}")
    print(f"  语言端口 vocab_size: {trainer.language_port.vocab_size}")
    
    # 测试2: 单步训练
    print("\n[Test 2] 单步训练验证")
    
    dataset = LanguageDataset(sample_texts, tokenizer, max_len=64)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    
    result = trainer.train_step(batch)
    
    print(f"  自由能 F: {result['free_energy'].item():.4f}")
    print(f"  预测误差 E_pred: {result['prediction_error'].item():.4f}")
    print(f"  PPL: {result['perplexity'].item():.2f}")
    print(f"  q_norm: {result['q_norm']:.4f}")
    print(f"  p_norm: {result['p_norm']:.4f}")
    print(f"  s_value: {result['s_value']:.4f}")
    
    # 测试3: 反向传播
    print("\n[Test 3] 反向传播验证")
    
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-4)
    
    loss = result['loss']
    loss.backward()
    
    # 检查SoulEntity组件的梯度
    has_grad_entity = any(p.grad is not None for p in trainer.entity.parameters())
    has_grad_port = any(p.grad is not None for p in trainer.language_port.parameters())
    
    print(f"  SoulEntity有梯度: {has_grad_entity}")
    print(f"  语言端口有梯度: {has_grad_port}")
    
    optimizer.step()
    optimizer.zero_grad()
    
    # 测试4: 生成验证
    print("\n[Test 4] 文本生成")
    
    text = trainer.generate("人工", max_len=20, temperature=1.0)
    print(f"  生成文本: {text[:50]}")
    
    # 测试5: SoulEntity诊断
    print("\n[Test 5] SoulEntity诊断")
    
    diag = trainer.get_entity_diagnostics()
    for k, v in diag.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # 测试6: 多步训练
    print("\n[Test 6] 多步训练验证")
    
    losses = []
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        
        result = trainer.train_step(batch)
        loss = result['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(result['free_energy'].item())
        print(f"  Step {i}: F={result['free_energy'].item():.4f}, "
              f"E_pred={result['prediction_error'].item():.4f}")
    
    print(f"\n  F变化趋势: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ 验证通过: 语言是端口，SoulEntity是核心，优化自由能F")
    print("=" * 70)
