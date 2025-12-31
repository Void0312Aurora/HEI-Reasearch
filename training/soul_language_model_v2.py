"""
SoulLanguageModel V2: 改进版灵魂语言模型

问题诊断:
1. V1架构中SoulEntity中间层破坏了序列信息
2. 几何动力学与语言建模目标脱节
3. 数据规模不足

改进方案:
1. 分离语言模型和灵魂核心，使用交叉注意力连接
2. 语言模型保留完整的序列建模能力
3. SoulEntity提供"意图/状态"向量，影响但不阻断语言生成
4. 添加验证集和早停机制

架构:
  Input Tokens
       ↓
  [Transformer Encoder] ──→ sequence hidden states
       ↓                           ↓
  [Pooling] ──→ u             [Cross-Attn with Soul State]
       ↓                           ↓
  [SoulEntity] ──→ q ────────→ soul context
       ↓                           ↓
  Free Energy F              [Transformer Decoder]
                                   ↓
                              Output Logits
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from he_core.state import ContactState
from he_core.language_interface import PositionalEncoding


@dataclass
class ModelConfig:
    """模型配置"""
    vocab_size: int = 8000
    dim_embed: int = 256
    dim_hidden: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dim_q: int = 64          # 几何状态维度
    dim_z: int = 16          # 上下文维度
    dropout: float = 0.1
    max_len: int = 256
    
    # 损失权重
    lm_weight: float = 1.0       # 语言模型损失
    soul_weight: float = 0.01    # 灵魂一致性损失


class SoulCore(nn.Module):
    """
    轻量级灵魂核心
    
    提供持久的内部状态，影响语言生成
    但不阻断序列信息流
    """
    
    def __init__(self, dim_input: int, dim_q: int, dim_z: int):
        super().__init__()
        self.dim_q = dim_q
        self.dim_z = dim_z
        
        # u → (q, p, s) 状态初始化
        self.state_net = nn.Sequential(
            nn.Linear(dim_input, dim_q * 2),
            nn.Tanh(),
            nn.Linear(dim_q * 2, dim_q * 2 + 1)
        )
        
        # 势能网络 V(q, z)
        self.V_net = nn.Sequential(
            nn.Linear(dim_q + dim_z, dim_q),
            nn.Tanh(),
            nn.Linear(dim_q, 1)
        )
        
        # 上下文 z (可学习)
        self.z = nn.Parameter(torch.randn(dim_z) * 0.01)
        
    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            u: 输入向量 [batch, dim_input]
            
        Returns:
            q: 几何状态 [batch, dim_q]
            F: 自由能 [batch]
        """
        batch_size = u.shape[0]
        device = u.device
        
        # 初始化状态
        state = self.state_net(u)
        q = state[:, :self.dim_q]
        
        # 计算自由能
        z = self.z.unsqueeze(0).expand(batch_size, -1)
        qz = torch.cat([q, z], dim=-1)
        V = self.V_net(qz).squeeze(-1)
        
        # KL正则
        kl = 0.5 * (z.pow(2).sum(dim=-1))
        
        F = V + 0.01 * kl
        
        return q, F


class SoulLanguageModelV2(nn.Module):
    """
    改进版灵魂语言模型
    
    - 保留完整的Transformer语言建模能力
    - SoulCore提供上下文调制，而非阻断信息流
    - 支持teacher forcing训练
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token嵌入
        self.embedding = nn.Embedding(config.vocab_size, config.dim_embed, padding_idx=0)
        self.pos_encoding = PositionalEncoding(config.dim_embed, config.dropout, config.max_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_embed,
            nhead=config.num_heads,
            dim_feedforward=config.dim_hidden,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # 灵魂核心
        self.soul = SoulCore(config.dim_embed, config.dim_q, config.dim_z)
        
        # Soul状态投影到嵌入空间
        self.soul_proj = nn.Linear(config.dim_q, config.dim_embed)
        
        # Transformer解码器 (带soul交叉注意力)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.dim_embed,
            nhead=config.num_heads,
            dim_feedforward=config.dim_hidden,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.num_layers)
        
        # 输出层
        self.output = nn.Linear(config.dim_embed, config.vocab_size)
        
        # 共享嵌入权重
        self.output.weight = self.embedding.weight
        
        # 因果mask缓存
        self._causal_mask = None
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
                
    def _get_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """获取因果注意力mask"""
        if self._causal_mask is None or self._causal_mask.size(0) < size:
            mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
            self._causal_mask = mask
        return self._causal_mask[:size, :size].to(device)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] 目标token
            
        Returns:
            dict: logits, loss, soul_loss, F
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. 嵌入
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        # 2. 编码
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
            
        enc_out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 3. Soul核心处理
        # 使用第一个token的隐藏状态作为soul输入
        soul_input = enc_out[:, 0, :]  # [batch, dim_embed]
        soul_state, F = self.soul(soul_input)
        
        # 将soul状态投影并扩展为memory
        soul_memory = self.soul_proj(soul_state).unsqueeze(1)  # [batch, 1, dim_embed]
        
        # 4. 组合memory: 编码器输出 + soul状态
        memory = torch.cat([enc_out, soul_memory], dim=1)  # [batch, seq_len+1, dim_embed]
        
        # 5. 解码 (自回归)
        tgt = x  # 使用相同的输入作为target
        tgt_mask = self._get_causal_mask(seq_len, device)
        
        dec_out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # 6. 输出logits
        logits = self.output(dec_out)  # [batch, seq_len, vocab_size]
        
        result = {
            'logits': logits,
            'soul_state': soul_state,
            'F': F.mean(),
        }
        
        # 7. 计算损失
        if labels is not None:
            # 语言模型损失
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            lm_loss = F_cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=0
            )
            
            # 总损失
            total_loss = self.config.lm_weight * lm_loss + self.config.soul_weight * F.mean()
            
            result['loss'] = total_loss
            result['lm_loss'] = lm_loss
            result['ppl'] = torch.exp(lm_loss)
            
        return result
    
    @torch.no_grad()
    def generate(self,
                 prompt_ids: torch.Tensor,
                 max_new_tokens: int = 50,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9) -> torch.Tensor:
        """
        自回归生成
        
        Args:
            prompt_ids: [batch, prompt_len]
            max_new_tokens: 最大生成token数
            
        Returns:
            generated_ids: [batch, prompt_len + generated_len]
        """
        self.eval()
        device = prompt_ids.device
        batch_size = prompt_ids.shape[0]
        
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            # 截断到最大长度
            if generated.size(1) > self.config.max_len:
                context = generated[:, -self.config.max_len:]
            else:
                context = generated
                
            # 前向
            outputs = self.forward(context)
            logits = outputs['logits'][:, -1, :]  # 最后一个位置
            
            # 温度
            logits = logits / temperature
            
            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
                
            # Top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # 添加到序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查EOS
            if (next_token == 3).all():  # EOS token
                break
                
        return generated


def F_cross_entropy(logits, targets, ignore_index=0):
    """交叉熵损失"""
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


# ============================================
#   测试
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("SoulLanguageModel V2 测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = ModelConfig(
        vocab_size=5000,
        dim_embed=128,
        dim_hidden=256,
        num_layers=2,
        num_heads=4,
        dim_q=32,
        dim_z=8,
    )
    
    model = SoulLanguageModelV2(config).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 测试前向
    batch_size = 4
    seq_len = 32
    
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model(input_ids, attention_mask, labels)
    
    print(f"\nForward pass:")
    print(f"  logits: {outputs['logits'].shape}")
    print(f"  loss: {outputs['loss'].item():.4f}")
    print(f"  lm_loss: {outputs['lm_loss'].item():.4f}")
    print(f"  ppl: {outputs['ppl'].item():.2f}")
    print(f"  F: {outputs['F'].item():.4f}")
    print(f"  soul_state: {outputs['soul_state'].shape}")
    
    # 测试生成
    print(f"\nGeneration test:")
    prompt = torch.tensor([[2, 100, 200]], device=device)  # BOS + 2 tokens
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"  prompt: {prompt[0].tolist()}")
    print(f"  generated: {generated[0].tolist()}")
    
    print("\n✓ SoulLanguageModel V2 测试完成")

