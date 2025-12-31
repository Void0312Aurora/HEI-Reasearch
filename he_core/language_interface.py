"""
语言接口 (Language Interface)

A5公理要求: 语言只是可能的感知/行动接口之一
本模块实现 token ↔ 几何状态 的双向映射

核心组件:
1. Tokenizer: 文本 ↔ token IDs
2. TokenEncoder: token IDs → 几何空间 u
3. StateDecoder: 几何状态 q → token 分布
4. LanguagePort: 整合编码解码的语言端口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import json
import os
from collections import Counter
import re


class SimpleTokenizer:
    """
    简单分词器
    
    支持字符级和词级分词
    专门针对中文优化
    """
    
    def __init__(self, 
                 vocab_size: int = 10000,
                 mode: str = 'char',  # 'char' 或 'word'
                 special_tokens: List[str] = None):
        self.vocab_size = vocab_size
        self.mode = mode
        
        # 特殊token
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>']
        
        self.token_to_id = {}
        self.id_to_token = {}
        
        # 初始化特殊token
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            
        self.pad_id = self.token_to_id['<PAD>']
        self.unk_id = self.token_to_id['<UNK>']
        self.bos_id = self.token_to_id['<BOS>']
        self.eos_id = self.token_to_id['<EOS>']
        
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """从文本构建词表"""
        counter = Counter()
        
        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)
            
        # 按频率排序
        sorted_tokens = sorted(counter.items(), key=lambda x: -x[1])
        
        # 构建词表
        idx = len(self.special_tokens)
        for token, freq in sorted_tokens:
            if freq < min_freq:
                break
            if idx >= self.vocab_size:
                break
            if token not in self.token_to_id:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1
                
        print(f"Built vocabulary: {len(self.token_to_id)} tokens")
        
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if self.mode == 'char':
            # 字符级分词 (适合中文)
            return list(text.replace(' ', ''))
        else:
            # 简单词级分词 (中文用字，英文用词)
            tokens = []
            current_word = []
            for char in text:
                if '\u4e00' <= char <= '\u9fff':  # 中文字符
                    if current_word:
                        tokens.append(''.join(current_word))
                        current_word = []
                    tokens.append(char)
                elif char.isalnum():
                    current_word.append(char)
                else:
                    if current_word:
                        tokens.append(''.join(current_word))
                        current_word = []
            if current_word:
                tokens.append(''.join(current_word))
            return tokens
            
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """文本 → token IDs"""
        tokens = self._tokenize(text)
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
            
        return ids
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """token IDs → 文本"""
        tokens = []
        for idx in ids:
            if skip_special and idx in [self.pad_id, self.bos_id, self.eos_id]:
                continue
            tokens.append(self.id_to_token.get(idx, '<UNK>'))
        return ''.join(tokens)
    
    def save(self, path: str):
        """保存词表"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'mode': self.mode,
                'token_to_id': self.token_to_id,
            }, f, ensure_ascii=False, indent=2)
            
    def load(self, path: str):
        """加载词表"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab_size = data['vocab_size']
            self.mode = data.get('mode', 'char')
            self.token_to_id = data['token_to_id']
            self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
            
    def __len__(self):
        return len(self.token_to_id)


class TokenEncoder(nn.Module):
    """
    Token 编码器
    
    将 token IDs 映射到几何空间的输入向量 u
    token IDs → Embedding → Projection → u ∈ ℝ^dim_u
    """
    
    def __init__(self, 
                 vocab_size: int,
                 dim_embed: int = 256,
                 dim_u: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim_embed = dim_embed
        self.dim_u = dim_u
        
        # Token 嵌入
        self.embedding = nn.Embedding(vocab_size, dim_embed, padding_idx=0)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(dim_embed, dropout)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embed,
            nhead=4,
            dim_feedforward=dim_embed * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 投影到几何空间
        self.projection = nn.Sequential(
            nn.Linear(dim_embed, dim_embed),
            nn.GELU(),
            nn.Linear(dim_embed, dim_u)
        )
        
    def forward(self, 
                token_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码 tokens 为几何输入
        
        Args:
            token_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (1 = 有效, 0 = padding)
            
        Returns:
            u: [batch, seq_len, dim_u] 或 [batch, dim_u] (pooled)
        """
        # 嵌入
        x = self.embedding(token_ids)  # [B, L, D]
        x = self.pos_encoder(x)
        
        # 创建 padding mask
        if attention_mask is not None:
            # Transformer 期望 True = 忽略
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
            
        # Transformer 编码
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # 投影
        u = self.projection(x)
        
        return u
    
    def encode_pooled(self, 
                      token_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码并池化为单一向量
        """
        u_seq = self.forward(token_ids, attention_mask)
        
        if attention_mask is not None:
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            u_pooled = (u_seq * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            u_pooled = u_seq.mean(dim=1)
            
        return u_pooled


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class StateDecoder(nn.Module):
    """
    状态解码器
    
    将几何状态 q 映射到 token 分布
    q ∈ ℝ^dim_q → logits ∈ ℝ^vocab_size
    
    支持自回归生成
    """
    
    def __init__(self,
                 vocab_size: int,
                 dim_q: int,
                 dim_embed: int = 256,
                 num_layers: int = 2,
                 max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim_q = dim_q
        self.dim_embed = dim_embed
        self.max_len = max_len
        
        # 状态投影
        self.state_proj = nn.Linear(dim_q, dim_embed)
        
        # Token 嵌入 (用于自回归)
        self.token_embedding = nn.Embedding(vocab_size, dim_embed, padding_idx=0)
        self.pos_encoder = PositionalEncoding(dim_embed, dropout, max_len)
        
        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_embed,
            nhead=4,
            dim_feedforward=dim_embed * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 输出层
        self.output_proj = nn.Linear(dim_embed, vocab_size)
        
        # 因果 mask 缓存
        self._causal_mask = None
        
    def _get_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """获取因果注意力mask"""
        if self._causal_mask is None or self._causal_mask.size(0) < size:
            mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
            self._causal_mask = mask
        return self._causal_mask[:size, :size].to(device)
        
    def forward(self,
                state: torch.Tensor,
                prev_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        解码状态为 token logits
        
        Args:
            state: 几何状态 [batch, dim_q]
            prev_tokens: 之前的 tokens [batch, prev_len] (用于自回归)
            
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size = state.shape[0]
        device = state.device
        
        # 状态作为 memory
        memory = self.state_proj(state).unsqueeze(1)  # [B, 1, D]
        
        if prev_tokens is None:
            # 只有状态，输出单个 token 分布
            # 用 BOS token 作为查询
            bos = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            tgt = self.token_embedding(bos)
            tgt = self.pos_encoder(tgt)
            
            out = self.transformer(tgt, memory)
            logits = self.output_proj(out)
        else:
            # 自回归模式
            tgt = self.token_embedding(prev_tokens)
            tgt = self.pos_encoder(tgt)
            
            # 因果 mask
            tgt_mask = self._get_causal_mask(prev_tokens.size(1), device)
            
            out = self.transformer(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(out)
            
        return logits
    
    @torch.no_grad()
    def generate(self,
                 state: torch.Tensor,
                 max_len: int = 50,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 bos_id: int = 2,
                 eos_id: int = 3) -> torch.Tensor:
        """
        自回归生成
        
        Args:
            state: 几何状态 [batch, dim_q]
            max_len: 最大生成长度
            temperature: 采样温度
            top_k: top-k 采样
            
        Returns:
            generated: [batch, gen_len]
        """
        batch_size = state.shape[0]
        device = state.device
        
        # 初始化
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len):
            logits = self.forward(state, generated)[:, -1, :]  # [B, V]
            
            # 温度缩放
            logits = logits / temperature
            
            # Top-k 过滤
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
                
            # 采样
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # 添加到序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查 EOS
            finished = finished | (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break
                
        return generated


class LanguagePort(nn.Module):
    """
    语言端口
    
    整合编码器和解码器，提供完整的语言接口
    作为 SoulEntity 的感知/行动接口之一
    """
    
    def __init__(self,
                 vocab_size: int,
                 dim_q: int,
                 dim_u: Optional[int] = None,
                 dim_embed: int = 256,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim_q = dim_q
        self.dim_u = dim_u or dim_q
        
        # 编码器: tokens → u
        self.encoder = TokenEncoder(
            vocab_size=vocab_size,
            dim_embed=dim_embed,
            dim_u=self.dim_u,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        # 解码器: q → tokens
        self.decoder = StateDecoder(
            vocab_size=vocab_size,
            dim_q=dim_q,
            dim_embed=dim_embed,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
        # 分词器 (需要单独设置)
        self.tokenizer: Optional[SimpleTokenizer] = None
        
    def set_tokenizer(self, tokenizer: SimpleTokenizer):
        """设置分词器"""
        self.tokenizer = tokenizer
        
    def encode_text(self, 
                    texts: Union[str, List[str]],
                    max_len: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码文本为几何输入
        
        Args:
            texts: 单个或多个文本
            max_len: 最大长度
            
        Returns:
            u: [batch, dim_u] 编码向量
            token_ids: [batch, seq_len] token IDs
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set. Call set_tokenizer() first.")
            
        if isinstance(texts, str):
            texts = [texts]
            
        device = next(self.parameters()).device
        
        # 分词
        all_ids = [self.tokenizer.encode(t) for t in texts]
        
        # Padding
        max_seq_len = min(max_len, max(len(ids) for ids in all_ids))
        padded_ids = []
        attention_masks = []
        
        for ids in all_ids:
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            mask = [1] * len(ids)
            
            # Pad
            pad_len = max_seq_len - len(ids)
            ids = ids + [self.tokenizer.pad_id] * pad_len
            mask = mask + [0] * pad_len
            
            padded_ids.append(ids)
            attention_masks.append(mask)
            
        token_ids = torch.tensor(padded_ids, device=device)
        attention_mask = torch.tensor(attention_masks, device=device)
        
        # 编码
        u = self.encoder.encode_pooled(token_ids, attention_mask)
        
        return u, token_ids
    
    def decode_state(self,
                     state: torch.Tensor,
                     max_len: int = 50,
                     temperature: float = 1.0) -> List[str]:
        """
        解码几何状态为文本
        
        Args:
            state: [batch, dim_q]
            max_len: 最大生成长度
            temperature: 采样温度
            
        Returns:
            texts: 生成的文本列表
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set. Call set_tokenizer() first.")
            
        # 生成 token IDs
        generated = self.decoder.generate(
            state, 
            max_len=max_len,
            temperature=temperature,
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id
        )
        
        # 解码为文本
        texts = []
        for ids in generated:
            text = self.tokenizer.decode(ids.cpu().tolist())
            texts.append(text)
            
        return texts
    
    def forward(self,
                token_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            token_ids: 输入 tokens [batch, seq_len]
            attention_mask: 注意力 mask
            state: 如果提供，用于解码
            
        Returns:
            dict: u (编码), logits (解码, 如果 state 提供)
        """
        result = {}
        
        # 编码
        u = self.encoder.encode_pooled(token_ids, attention_mask)
        result['u'] = u
        
        # 如果有状态，则解码
        if state is not None:
            logits = self.decoder(state, token_ids)
            result['logits'] = logits
            
        return result
    
    def compute_loss(self,
                     token_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     state: torch.Tensor,
                     target_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算语言模型损失
        
        用于训练: 给定状态，预测下一个 token
        
        Args:
            token_ids: 输入 tokens [batch, seq_len]
            attention_mask: mask
            state: 几何状态 [batch, dim_q]
            target_ids: 目标 tokens [batch, seq_len]
            
        Returns:
            dict: loss, perplexity
        """
        # 解码
        logits = self.decoder(state, token_ids)  # [B, L, V]
        
        # 计算损失
        # 移位: 预测下一个 token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        # Cross-entropy
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_targets.view(-1),
            ignore_index=0,  # padding
            reduction='none'
        )
        loss = loss.view(shift_targets.shape)
        
        # Masked mean
        loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
        
        # Perplexity
        ppl = torch.exp(loss)
        
        return {'loss': loss, 'perplexity': ppl}


# ============================================
#   测试
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Language Interface 测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试1: 分词器
    print("\n[Test 1] 分词器")
    tokenizer = SimpleTokenizer(vocab_size=5000, mode='char')
    
    # 构建词表
    sample_texts = [
        "数学是研究数量、结构以及空间等概念的学科。",
        "人工智能是计算机科学的一个分支。",
        "深度学习是机器学习的一种方法。",
    ]
    tokenizer.build_vocab(sample_texts)
    
    # 编解码测试
    text = "数学是一门美丽的科学"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"  原文: {text}")
    print(f"  编码: {ids[:10]}...")
    print(f"  解码: {decoded}")
    
    # 测试2: Token 编码器
    print("\n[Test 2] Token 编码器")
    vocab_size = len(tokenizer)
    dim_u = 64
    
    encoder = TokenEncoder(vocab_size, dim_embed=128, dim_u=dim_u).to(device)
    
    token_ids = torch.tensor([ids], device=device)
    u = encoder.encode_pooled(token_ids)
    print(f"  u shape: {u.shape}")
    print(f"  u norm: {u.norm():.4f}")
    
    # 测试3: 状态解码器
    print("\n[Test 3] 状态解码器")
    dim_q = 64
    
    decoder = StateDecoder(vocab_size, dim_q, dim_embed=128).to(device)
    
    state = torch.randn(1, dim_q, device=device)
    generated = decoder.generate(
        state, 
        max_len=20,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id
    )
    gen_text = tokenizer.decode(generated[0].cpu().tolist())
    print(f"  生成的文本: {gen_text[:50]}...")
    
    # 测试4: 语言端口
    print("\n[Test 4] 语言端口")
    port = LanguagePort(vocab_size, dim_q, dim_embed=128).to(device)
    port.set_tokenizer(tokenizer)
    
    # 编码
    u, _ = port.encode_text("这是一个测试")
    print(f"  编码 u: {u.shape}")
    
    # 解码
    texts = port.decode_state(state, max_len=15)
    print(f"  解码文本: {texts[0][:30]}...")
    
    print("\n✓ Language Interface 测试完成")

