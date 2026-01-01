"""
大规模训练系统

特性：
- 混合精度训练 (AMP)
- 梯度累积
- 分布式数据并行 (DDP)
- 高效数据加载
- 完整的checkpoint和恢复机制
- 学习率调度
- 详细日志记录
"""

import os
import sys
import json
import math
import time
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple, Iterator
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from he_core.soul_entity import SoulEntity, create_soul_entity
from he_core.state import ContactState
from he_core.language_interface import TokenEncoder, StateDecoder, SimpleTokenizer


# ============================================================
#                      配置
# ============================================================

@dataclass
class LargeScaleConfig:
    """大规模训练配置"""
    # 模型参数
    dim_q: int = 128
    dim_z: int = 32
    dim_u: int = 128
    vocab_size: int = 32000
    
    # 训练参数
    batch_size: int = 64
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = field(init=False)
    
    max_seq_len: int = 512
    num_epochs: int = 10
    max_steps: int = -1  # -1表示由epoch决定
    
    # 优化器
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # 学习率调度
    warmup_steps: int = 2000
    lr_decay_steps: int = -1  # -1表示使用总步数
    
    # 混合精度
    use_amp: bool = True
    amp_dtype: str = 'bfloat16'  # 'float16' or 'bfloat16'
    
    # 损失权重
    lambda_lm: float = 1.0
    lambda_stability: float = 0.01
    lambda_geometry: float = 0.001
    
    # 数据
    data_path: str = ''
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # 分布式
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1
    
    # 保存和日志
    output_dir: str = 'checkpoints/large_scale'
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 10
    
    # 设备
    device: str = 'cuda'
    
    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps * self.world_size
        
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LargeScaleConfig':
        # 排除计算字段
        exclude_fields = {'effective_batch_size'}
        return cls(**{k: v for k, v in d.items() 
                     if k in cls.__dataclass_fields__ and k not in exclude_fields})


# ============================================================
#                      数据集
# ============================================================

class StreamingTextDataset(IterableDataset):
    """
    流式文本数据集 - 支持大规模数据
    
    特点：
    - 流式读取，不需要全部加载到内存
    - 支持JSON Lines格式
    - 动态tokenize
    """
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer: SimpleTokenizer,
                 seq_len: int = 512,
                 rank: int = 0,
                 world_size: int = 1):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        
        # 统计文件行数（用于估计epoch大小）
        self.num_lines = self._count_lines()
        
    def _count_lines(self) -> int:
        """快速统计行数"""
        count = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """流式迭代"""
        worker_info = torch.utils.data.get_worker_info()
        
        # 计算当前worker/rank应该处理的数据
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        
        # 结合分布式rank
        total_workers = num_workers * self.world_size
        global_worker_id = self.rank * num_workers + worker_id
        
        buffer = []
        line_idx = 0
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 只处理属于当前worker的行
                if line_idx % total_workers != global_worker_id:
                    line_idx += 1
                    continue
                line_idx += 1
                
                # 解析JSON
                try:
                    if self.data_path.endswith('.json'):
                        obj = json.loads(line.strip())
                        text = obj.get('text', '')
                    else:
                        text = line.strip()
                except:
                    continue
                
                if not text:
                    continue
                
                # Tokenize
                tokens = self.tokenizer.encode(text)
                buffer.extend(tokens)
                
                # 当buffer足够时，生成训练样本
                while len(buffer) >= self.seq_len + 1:
                    sample = buffer[:self.seq_len + 1]
                    buffer = buffer[self.seq_len:]
                    yield torch.tensor(sample, dtype=torch.long)
    
    def __len__(self) -> int:
        # 估计值
        return self.num_lines // self.world_size


# ============================================================
#                      模型
# ============================================================

class SimpleDecoder(nn.Module):
    """简化的解码器：直接MLP映射"""
    
    def __init__(self, dim_q: int, vocab_size: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_q, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LargeScaleModel(nn.Module):
    """
    大规模训练模型
    
    架构：
    - Embedding + Transformer Encoder
    - 状态投影
    - 简单解码器
    """
    
    def __init__(self, config: LargeScaleConfig):
        super().__init__()
        self.config = config
        
        # Token嵌入
        self.embedding = nn.Embedding(config.vocab_size, config.dim_q * 2)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.dim_q * 2)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_q * 2,
            nhead=8,
            dim_feedforward=config.dim_q * 8,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 投影到状态空间
        self.to_state = nn.Linear(config.dim_q * 2, config.dim_q)
        
        # 解码器
        self.decoder = SimpleDecoder(config.dim_q, config.vocab_size, config.dim_q * 4)
        
        # LayerNorm
        self.ln_f = nn.LayerNorm(config.dim_q)
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            tokens: (batch, seq_len)
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            aux_loss: 辅助损失
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # 位置索引
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 嵌入
        x = self.embedding(tokens) + self.pos_embedding(positions)
        
        # 因果mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        
        # Transformer编码
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        
        # 投影到状态空间
        q = self.to_state(x)  # (batch, seq_len, dim_q)
        q = self.ln_f(q)
        
        # 解码
        logits = self.decoder(q)  # (batch, seq_len, vocab_size)
        
        # 计算辅助损失
        # 1. 几何正则化：q范数应该适中
        q_norm = q.norm(dim=-1)
        aux_geometry = (q_norm - 1.0).pow(2).mean()
        
        # 2. 稳定性：相邻状态不应变化太剧烈
        if seq_len > 1:
            q_diff = (q[:, 1:] - q[:, :-1]).norm(dim=-1)
            aux_stability = torch.relu(q_diff - 0.5).mean()
        else:
            aux_stability = torch.tensor(0.0, device=device)
        
        return {
            'logits': logits,
            'aux_geometry': aux_geometry,
            'aux_stability': aux_stability,
        }


# ============================================================
#                      训练器
# ============================================================

class LargeScaleTrainer:
    """大规模训练器"""
    
    def __init__(self, config: LargeScaleConfig):
        self.config = config
        self.setup_logging()
        self.setup_distributed()
        self.setup_device()
        self.setup_data()  # 先加载数据获取vocab_size
        self._create_model()  # 再创建模型
        self.setup_optimizer()
        
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    def setup_logging(self):
        """设置日志"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(self.config.output_dir, 'train.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        if self.config.local_rank == 0:
            self.logger.info(f"配置: {json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False)}")
    
    def setup_distributed(self):
        """设置分布式训练"""
        if self.config.distributed:
            dist.init_process_group(backend='nccl')
            self.config.local_rank = dist.get_rank()
            self.config.world_size = dist.get_world_size()
            torch.cuda.set_device(self.config.local_rank)
            self.config.device = f'cuda:{self.config.local_rank}'
    
    def setup_device(self):
        """设置设备"""
        self.device = torch.device(self.config.device)
        
        # 设置AMP dtype
        if self.config.amp_dtype == 'bfloat16':
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        
        self.scaler = GradScaler(enabled=self.config.use_amp and self.config.amp_dtype == 'float16')
    
    def setup_model(self):
        """设置模型（在数据加载后调用以获取正确的vocab_size）"""
        pass  # 延迟到数据加载后
    
    def _create_model(self):
        """创建模型"""
        self.model = LargeScaleModel(self.config).to(self.device)
        
        if self.config.distributed:
            self.model = DDP(self.model, device_ids=[self.config.local_rank])
        
        # 统计参数量
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if self.config.local_rank == 0:
            self.logger.info(f"模型参数: {num_params:,} (可训练: {trainable_params:,})")
    
    def setup_optimizer(self):
        """设置优化器"""
        # 分组参数（不同的weight decay）
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'embedding' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2)
        )
    
    def setup_data(self):
        """设置数据"""
        # 初始化tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=self.config.vocab_size)
        
        if self.config.data_path and os.path.exists(self.config.data_path):
            # 构建词汇表
            if self.config.local_rank == 0:
                self.logger.info(f"加载数据: {self.config.data_path}")
            
            # 读取部分数据构建词汇表
            texts = []
            with open(self.config.data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 100000:  # 使用前10万行构建词汇表
                        break
                    try:
                        if self.config.data_path.endswith('.json'):
                            obj = json.loads(line.strip())
                            texts.append(obj.get('text', ''))
                        else:
                            texts.append(line.strip())
                    except:
                        continue
            
            self.tokenizer.build_vocab(texts)
            self.config.vocab_size = len(self.tokenizer)
            
            if self.config.local_rank == 0:
                self.logger.info(f"词汇表大小: {self.config.vocab_size}")
            
            # 创建数据集
            self.train_dataset = StreamingTextDataset(
                self.config.data_path,
                self.tokenizer,
                self.config.max_seq_len,
                self.config.local_rank,
                self.config.world_size
            )
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                prefetch_factor=self.config.prefetch_factor,
                pin_memory=True
            )
            
            if self.config.local_rank == 0:
                self.logger.info(f"数据集大小: ~{len(self.train_dataset)} 样本")
        else:
            if self.config.local_rank == 0:
                self.logger.warning("未找到数据文件，使用合成数据")
            self.train_loader = None
    
    def get_lr(self, step: int) -> float:
        """计算学习率（cosine with warmup）"""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * step / self.config.warmup_steps
        
        decay_steps = self.config.lr_decay_steps if self.config.lr_decay_steps > 0 else self.config.max_steps
        if decay_steps <= 0:
            decay_steps = 100000
        
        decay_ratio = (step - self.config.warmup_steps) / (decay_steps - self.config.warmup_steps)
        decay_ratio = min(1.0, decay_ratio)
        
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        
        # 准备数据
        tokens = batch.to(self.device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        
        # 混合精度上下文
        amp_ctx = autocast(dtype=self.amp_dtype) if self.config.use_amp else nullcontext()
        
        with amp_ctx:
            # 前向传播
            outputs = self.model(inputs)
            logits = outputs['logits']
            
            # 语言建模损失
            lm_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=0
            )
            
            # 辅助损失
            aux_loss = (
                self.config.lambda_geometry * outputs['aux_geometry'] +
                self.config.lambda_stability * outputs['aux_stability']
            )
            
            # 总损失
            loss = self.config.lambda_lm * lm_loss + aux_loss
            
            # 梯度累积
            loss = loss / self.config.gradient_accumulation_steps
        
        # 反向传播
        if self.config.use_amp and self.config.amp_dtype == 'float16':
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'lm_loss': lm_loss.item(),
            'aux_geometry': outputs['aux_geometry'].item(),
            'aux_stability': outputs['aux_stability'].item(),
            'ppl': math.exp(min(lm_loss.item(), 20)),
        }
    
    def optimizer_step(self):
        """优化器步骤"""
        # 梯度裁剪
        if self.config.use_amp and self.config.amp_dtype == 'float16':
            self.scaler.unscale_(self.optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.grad_clip
        )
        
        # 更新参数
        if self.config.use_amp and self.config.amp_dtype == 'float16':
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
        
        return grad_norm.item()
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        if self.config.local_rank != 0:
            return
        
        model_state = self.model.module.state_dict() if self.config.distributed else self.model.state_dict()
        
        checkpoint = {
            'model': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict() if self.config.use_amp else None,
            'config': self.config.to_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'tokenizer': self.tokenizer,
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"保存检查点: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        model = self.model.module if self.config.distributed else self.model
        model.load_state_dict(checkpoint['model'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if checkpoint['scaler'] and self.config.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        if 'tokenizer' in checkpoint:
            self.tokenizer = checkpoint['tokenizer']
        
        self.logger.info(f"加载检查点: {path} (step={self.global_step}, epoch={self.epoch})")
    
    def train(self):
        """训练主循环"""
        if self.config.local_rank == 0:
            self.logger.info("=" * 60)
            self.logger.info("开始大规模训练")
            self.logger.info(f"有效批量大小: {self.config.effective_batch_size}")
            self.logger.info(f"混合精度: {self.config.use_amp} ({self.config.amp_dtype})")
            self.logger.info("=" * 60)
        
        start_time = time.time()
        accum_loss = 0.0
        accum_steps = 0
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            
            if self.train_loader is None:
                # 合成数据
                data_iter = iter(self._synthetic_data())
            else:
                data_iter = iter(self.train_loader)
            
            if self.config.local_rank == 0:
                self.logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(data_iter):
                # 更新学习率
                lr = self.get_lr(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # 训练步骤
                metrics = self.train_step(batch)
                accum_loss += metrics['loss']
                accum_steps += 1
                
                # 梯度累积
                if accum_steps >= self.config.gradient_accumulation_steps:
                    grad_norm = self.optimizer_step()
                    self.global_step += 1
                    
                    # 日志
                    if self.global_step % self.config.log_every == 0 and self.config.local_rank == 0:
                        avg_loss = accum_loss / accum_steps
                        elapsed = time.time() - start_time
                        samples_per_sec = self.global_step * self.config.effective_batch_size / elapsed
                        
                        self.logger.info(
                            f"Step {self.global_step} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"PPL: {metrics['ppl']:.2f} | "
                            f"LR: {lr:.2e} | "
                            f"Grad: {grad_norm:.2f} | "
                            f"Speed: {samples_per_sec:.1f} samples/s"
                        )
                    
                    # 保存检查点
                    if self.global_step % self.config.save_every == 0:
                        path = os.path.join(self.config.output_dir, f'checkpoint_{self.global_step}.pt')
                        self.save_checkpoint(path)
                        
                        if avg_loss < self.best_loss:
                            self.best_loss = avg_loss
                            best_path = os.path.join(self.config.output_dir, 'best_model.pt')
                            self.save_checkpoint(best_path)
                    
                    accum_loss = 0.0
                    accum_steps = 0
                
                # 检查最大步数
                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    break
            
            # Epoch结束保存
            path = os.path.join(self.config.output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            self.save_checkpoint(path)
            
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break
        
        # 最终保存
        final_path = os.path.join(self.config.output_dir, 'final_model.pt')
        self.save_checkpoint(final_path)
        
        total_time = time.time() - start_time
        if self.config.local_rank == 0:
            self.logger.info("=" * 60)
            self.logger.info(f"训练完成！总时间: {total_time / 3600:.2f} 小时")
            self.logger.info(f"最佳损失: {self.best_loss:.4f}")
            self.logger.info("=" * 60)
    
    def _synthetic_data(self) -> Iterator[torch.Tensor]:
        """生成合成数据（用于测试）"""
        while True:
            yield torch.randint(
                0, self.config.vocab_size,
                (self.config.batch_size, self.config.max_seq_len + 1)
            )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='大规模训练')
    
    # 模型
    parser.add_argument('--dim_q', type=int, default=128)
    parser.add_argument('--dim_z', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=32000)
    
    # 训练
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=-1)
    
    # 优化器
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    
    # 混合精度
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--amp_dtype', type=str, default='bfloat16')
    
    # 数据
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 保存
    parser.add_argument('--output_dir', type=str, default='checkpoints/large_scale')
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=10)
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None)
    
    # 分布式
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    
    # 创建配置
    config = LargeScaleConfig(
        dim_q=args.dim_q,
        dim_z=args.dim_z,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_seq_len=args.max_seq_len,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp,
        amp_dtype=args.amp_dtype,
        data_path=args.data,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        distributed='WORLD_SIZE' in os.environ,
        local_rank=args.local_rank,
    )
    
    # 创建训练器
    trainer = LargeScaleTrainer(config)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()

