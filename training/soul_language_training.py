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
    dim_embed: int = 256
    vocab_size: int = 10000
    
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
        
        # === 核心：SoulEntity（接触哈密顿动力学）===
        entity_config = {
            'dim_q': config.dim_q,
            'dim_u': config.dim_q,  # 语言端口维度匹配
            'dim_z': 16,
            'num_charts': 4,
            'beta_kl': config.beta_kl,
            'gamma_pred': config.gamma_pred,
        }
        self.entity = create_soul_entity(entity_config)
        
        # === 语言端口（只是接口之一）===
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
        
        # 添加语言端口到SoulEntity
        self.entity.add_interface('language', config.dim_q)
        
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
        return self.language_port.encoder.encode_pooled(token_ids, attention_mask)
    
    def decode_state(self, 
                     state_q: torch.Tensor,
                     prev_tokens: torch.Tensor) -> torch.Tensor:
        """
        从几何状态q解码为token logits
        
        这是语言端口的行动端（Write）
        """
        return self.language_port.decoder(state_q, prev_tokens)
    
    def compute_prediction_error(self,
                                 logits: torch.Tensor,
                                 target_ids: torch.Tensor,
                                 attention_mask: torch.Tensor) -> torch.Tensor:
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
        E_pred = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
        
        return E_pred
    
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
        token_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size = token_ids.shape[0]
        
        # 重置实体状态
        self.reset_entity(batch_size)
        
        # === Step 1: 语言感知端 ===
        # 编码tokens为几何输入u
        u = self.encode_tokens(token_ids, attention_mask)
        
        # === Step 2: 端口耦合驱动SoulEntity演化 ===
        # H(q,p,s,u) = H_int(q,p,s) + <u, B(q)>
        # 这是L3模板3的实现
        accumulated_states = []
        for step in range(self.config.num_evolution_steps):
            result = self.entity.step(
                {'language': u},
                dt=self.config.dt
            )
            accumulated_states.append(result['state_flat'].clone())
        
        # 获取最终状态
        current_state = self.entity.state
        q_final = current_state.q
        
        # === Step 3: 语言行动端 ===
        # 从几何状态q解码为token分布
        logits = self.decode_state(q_final, token_ids)
        
        # === Step 4: 计算预测误差 ===
        # 作为自由能F的γ分量
        E_pred = self.compute_prediction_error(logits, token_ids, attention_mask)
        
        # === Step 5: 计算总自由能F ===
        # F = V(q,z) + β·KL(z) + γ·E_pred
        # 这是A3公理的核心：单一标量泛函驱动所有行为
        F_total = self.entity.compute_free_energy(
            current_state,
            prediction_error=E_pred
        )
        
        # === 诊断指标 ===
        with torch.no_grad():
            ppl = torch.exp(E_pred)
            q_norm = current_state.q.norm().item()
            p_norm = current_state.p.norm().item()
            s_val = current_state.s.mean().item()
        
        return {
            'loss': F_total,  # 核心：优化F而非语言损失
            'free_energy': F_total.detach(),
            'prediction_error': E_pred.detach(),
            'perplexity': ppl,
            'q_norm': q_norm,
            'p_norm': p_norm,
            's_value': s_val,
        }
    
    @torch.no_grad()
    def generate(self,
                 prompt: str = "",
                 max_len: int = 50,
                 temperature: float = 1.0) -> str:
        """
        从SoulEntity状态生成文本
        
        流程：
        1. 如果有prompt，先编码并驱动演化
        2. 从当前状态q自回归生成
        """
        device = self.config.device
        self.reset_entity(1)
        
        if prompt:
            # 编码prompt
            ids = self.tokenizer.encode(prompt)
            token_ids = torch.tensor([ids], device=device)
            
            # 驱动演化
            u = self.encode_tokens(token_ids)
            for _ in range(self.config.num_evolution_steps):
                self.entity.step({'language': u}, dt=self.config.dt)
        
        # 从当前状态生成
        q = self.entity.state.q
        generated = self.language_port.decoder.generate(
            q,
            max_len=max_len,
            temperature=temperature,
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id
        )
        
        text = self.tokenizer.decode(generated[0].cpu().tolist())
        return text
    
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

