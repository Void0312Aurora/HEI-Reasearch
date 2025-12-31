"""
SoulLanguageModel V3: 解决过拟合和模式坍塌

改进:
1. Label Smoothing - 防止过度自信
2. Repetition Penalty - 训练时惩罚重复
3. Entropy Regularization - 鼓励预测多样性
4. 改进的生成策略 - repetition penalty + 更好的采样
5. 更好的评估指标 - Distinct-n
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from he_core.language_interface import PositionalEncoding


@dataclass
class ModelConfigV3:
    """模型配置"""
    vocab_size: int = 8000
    dim_embed: int = 256
    dim_hidden: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dim_q: int = 64
    dim_z: int = 16
    dropout: float = 0.2  # 增加dropout
    max_len: int = 256
    
    # 正则化
    label_smoothing: float = 0.1      # Label smoothing
    entropy_weight: float = 0.01      # 熵正则化权重
    repetition_penalty: float = 1.2   # 生成时的重复惩罚


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss
    
    防止模型过度自信，改善泛化
    """
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch * seq_len, vocab_size]
            targets: [batch * seq_len]
        """
        # 创建平滑分布
        smooth_dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 2))
        smooth_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        smooth_dist[:, self.ignore_index] = 0
        
        # 标记忽略位置
        mask = targets != self.ignore_index
        
        # 计算KL散度
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(smooth_dist * log_probs, dim=-1)
        
        # 只计算有效位置
        loss = loss * mask.float()
        return loss.sum() / mask.sum().clamp(min=1)


class SoulCoreV3(nn.Module):
    """轻量级灵魂核心"""
    
    def __init__(self, dim_input: int, dim_q: int, dim_z: int):
        super().__init__()
        self.dim_q = dim_q
        self.dim_z = dim_z
        
        self.state_net = nn.Sequential(
            nn.Linear(dim_input, dim_q * 2),
            nn.Tanh(),
            nn.Linear(dim_q * 2, dim_q)
        )
        
        self.V_net = nn.Sequential(
            nn.Linear(dim_q + dim_z, dim_q),
            nn.Tanh(),
            nn.Linear(dim_q, 1)
        )
        
        self.z = nn.Parameter(torch.randn(dim_z) * 0.01)
        
    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = u.shape[0]
        q = self.state_net(u)
        
        z = self.z.unsqueeze(0).expand(batch_size, -1)
        qz = torch.cat([q, z], dim=-1)
        V = self.V_net(qz).squeeze(-1)
        
        kl = 0.5 * z.pow(2).sum(dim=-1)
        F = V + 0.01 * kl
        
        return q, F


class SoulLanguageModelV3(nn.Module):
    """
    改进版灵魂语言模型
    
    新增:
    - Label Smoothing
    - Entropy Regularization
    - 改进的生成策略
    """
    
    def __init__(self, config: ModelConfigV3):
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
        self.soul = SoulCoreV3(config.dim_embed, config.dim_q, config.dim_z)
        
        # Soul状态投影
        self.soul_proj = nn.Linear(config.dim_q, config.dim_embed)
        
        # Transformer解码器
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
        self.output.weight = self.embedding.weight  # 共享权重
        
        # Label Smoothing Loss
        self.loss_fn = LabelSmoothingLoss(
            config.vocab_size, 
            smoothing=config.label_smoothing,
            ignore_index=0
        )
        
        self._causal_mask = None
        self._init_weights()
        
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
                
    def _get_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask.size(0) < size:
            mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
            self._causal_mask = mask
        return self._causal_mask[:size, :size].to(device)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 嵌入
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        # 编码
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
            
        enc_out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Soul核心
        soul_input = enc_out[:, 0, :]
        soul_state, F_energy = self.soul(soul_input)
        soul_memory = self.soul_proj(soul_state).unsqueeze(1)
        
        # 组合memory
        memory = torch.cat([enc_out, soul_memory], dim=1)
        
        # 解码
        tgt = x
        tgt_mask = self._get_causal_mask(seq_len, device)
        dec_out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # 输出logits
        logits = self.output(dec_out)
        
        result = {
            'logits': logits,
            'soul_state': soul_state,
            'F': F_energy.mean(),
        }
        
        if labels is not None:
            # 移位
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Label Smoothing Loss
            lm_loss = self.loss_fn(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            # 熵正则化 - 鼓励多样性
            probs = F.softmax(shift_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            
            # 我们希望熵较高（更不确定），所以惩罚低熵
            mask = (shift_labels != 0).float()
            entropy_loss = -entropy * mask
            entropy_loss = entropy_loss.sum() / mask.sum().clamp(min=1)
            
            # 总损失
            total_loss = lm_loss + self.config.entropy_weight * entropy_loss
            
            # 计算标准PPL（用于比较）
            with torch.no_grad():
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=0
                )
                ppl = torch.exp(ce_loss)
            
            result['loss'] = total_loss
            result['lm_loss'] = lm_loss
            result['entropy_loss'] = entropy_loss
            result['ppl'] = ppl
            result['entropy'] = (entropy * mask).sum() / mask.sum().clamp(min=1)
            
        return result
    
    @torch.no_grad()
    def generate(self,
                 prompt_ids: torch.Tensor,
                 max_new_tokens: int = 50,
                 temperature: float = 0.9,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.2,
                 no_repeat_ngram_size: int = 3) -> torch.Tensor:
        """
        改进的生成策略
        
        新增:
        - repetition_penalty: 惩罚已生成的token
        - no_repeat_ngram_size: 禁止重复n-gram
        """
        self.eval()
        device = prompt_ids.device
        batch_size = prompt_ids.shape[0]
        
        generated = prompt_ids.clone()
        
        for step in range(max_new_tokens):
            # 截断到最大长度
            if generated.size(1) > self.config.max_len:
                context = generated[:, -self.config.max_len:]
            else:
                context = generated
                
            # 前向
            outputs = self.forward(context)
            logits = outputs['logits'][:, -1, :].clone()  # [batch, vocab]
            
            # 1. 应用温度
            logits = logits / temperature
            
            # 2. 应用重复惩罚
            for b in range(batch_size):
                for prev_token in generated[b].unique():
                    if prev_token != 0:  # 不惩罚padding
                        logits[b, prev_token] /= repetition_penalty
                        
            # 3. 禁止重复n-gram
            if no_repeat_ngram_size > 0 and generated.size(1) >= no_repeat_ngram_size:
                for b in range(batch_size):
                    # 获取最近的n-1个token
                    prefix = tuple(generated[b, -(no_repeat_ngram_size-1):].tolist())
                    
                    # 检查历史中是否有相同前缀
                    for i in range(generated.size(1) - no_repeat_ngram_size + 1):
                        prev_prefix = tuple(generated[b, i:i+no_repeat_ngram_size-1].tolist())
                        if prev_prefix == prefix:
                            # 禁止下一个token
                            next_token = generated[b, i+no_repeat_ngram_size-1].item()
                            logits[b, next_token] = float('-inf')
            
            # 4. Top-k过滤
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
                
            # 5. Top-p (nucleus) 过滤
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
            if (next_token == 3).all():
                break
                
        return generated


def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    """
    计算Distinct-n指标
    
    衡量生成文本的多样性
    Distinct-n = unique n-grams / total n-grams
    """
    total_ngrams = 0
    unique_ngrams = set()
    
    for text in texts:
        chars = list(text)
        for i in range(len(chars) - n + 1):
            ngram = tuple(chars[i:i+n])
            unique_ngrams.add(ngram)
            total_ngrams += 1
            
    if total_ngrams == 0:
        return 0.0
        
    return len(unique_ngrams) / total_ngrams


def compute_repetition_rate(texts: List[str], n: int = 2) -> float:
    """
    计算重复率
    
    重复率 = 重复n-gram数 / 总n-gram数
    """
    total_ngrams = 0
    ngram_counts = Counter()
    
    for text in texts:
        chars = list(text)
        for i in range(len(chars) - n + 1):
            ngram = tuple(chars[i:i+n])
            ngram_counts[ngram] += 1
            total_ngrams += 1
            
    if total_ngrams == 0:
        return 0.0
        
    repeated = sum(count - 1 for count in ngram_counts.values() if count > 1)
    return repeated / total_ngrams


# ============================================
#   测试
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("SoulLanguageModel V3 测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = ModelConfigV3(
        vocab_size=5000,
        dim_embed=128,
        dim_hidden=256,
        num_layers=2,
        num_heads=4,
        dim_q=32,
        dim_z=8,
        label_smoothing=0.1,
        entropy_weight=0.01,
    )
    
    model = SoulLanguageModelV3(config).to(device)
    
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
    print(f"  loss: {outputs['loss'].item():.4f}")
    print(f"  lm_loss: {outputs['lm_loss'].item():.4f}")
    print(f"  entropy_loss: {outputs['entropy_loss'].item():.4f}")
    print(f"  ppl: {outputs['ppl'].item():.2f}")
    print(f"  entropy: {outputs['entropy'].item():.4f}")
    
    # 测试生成
    print(f"\nGeneration test (with repetition penalty):")
    prompt = torch.tensor([[2, 100, 200]], device=device)
    
    # 无惩罚
    gen_no_penalty = model.generate(prompt, max_new_tokens=30, repetition_penalty=1.0)
    
    # 有惩罚
    gen_with_penalty = model.generate(prompt, max_new_tokens=30, repetition_penalty=1.5)
    
    print(f"  Without penalty: {gen_no_penalty[0].tolist()[-20:]}")
    print(f"  With penalty:    {gen_with_penalty[0].tolist()[-20:]}")
    
    # 测试多样性指标
    print(f"\nDiversity metrics test:")
    test_texts = [
        "这是一个测试文本用于计算多样性指标",
        "省省省区区区县县县市市市",  # 低多样性
    ]
    
    for text in test_texts:
        d1 = compute_distinct_n([text], n=1)
        d2 = compute_distinct_n([text], n=2)
        rep = compute_repetition_rate([text], n=2)
        print(f"  '{text[:20]}...'")
        print(f"    Distinct-1: {d1:.4f}, Distinct-2: {d2:.4f}, Repetition: {rep:.4f}")
    
    print("\n✓ SoulLanguageModel V3 测试完成")

