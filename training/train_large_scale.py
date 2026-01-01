"""
大规模训练脚本

特性：
- 混合精度训练 (AMP)
- 分布式数据并行 (DDP)
- 梯度累积
- 断点续训
- WandB/TensorBoard 日志
- 动态学习率调度
- 完整的训练监控

运行方式：
# 单GPU
python HEI/training/train_large_scale.py --config configs/large_scale.yaml

# 多GPU (DDP)
torchrun --nproc_per_node=4 HEI/training/train_large_scale.py --config configs/large_scale.yaml

# 从检查点恢复
python HEI/training/train_large_scale.py --resume checkpoints/latest.pt
"""

import os
import sys
import json
import yaml
import math
import time
import argparse
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from he_core.soul_entity import create_soul_entity
from he_core.state import ContactState
from he_core.language_interface import TokenEncoder, StateDecoder, SimpleTokenizer


# ============================================================
#                    配置管理
# ============================================================

@dataclass
class ModelConfig:
    """模型配置"""
    dim_q: int = 128
    dim_z: int = 32
    dim_u: int = 128
    dim_embed: int = 512
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 512
    dropout: float = 0.1
    hyperbolic_c: float = 1.0


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练
    epochs: int = 100
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # 学习率
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    
    # 损失权重
    lambda_lm: float = 1.0
    lambda_stability: float = 0.1
    lambda_geometry: float = 0.01
    lambda_atlas: float = 0.01
    
    # 混合精度
    use_amp: bool = True
    amp_dtype: str = "float16"  # float16 or bfloat16
    
    # 分布式
    use_ddp: bool = False
    
    # 检查点
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 100
    
    # 路径
    output_dir: str = "checkpoints/large_scale"
    data_path: str = "HEI/data/wiki/wikipedia-zh-20250901.json"


@dataclass
class FullConfig:
    """完整配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'FullConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # 处理科学计数法字符串
        def convert_numbers(d):
            if isinstance(d, dict):
                return {k: convert_numbers(v) for k, v in d.items()}
            elif isinstance(d, str):
                # 尝试转换科学计数法
                try:
                    if 'e' in d.lower():
                        return float(d)
                except:
                    pass
                return d
            else:
                return d
        
        data = convert_numbers(data)
        model_cfg = ModelConfig(**data.get('model', {}))
        train_cfg = TrainingConfig(**data.get('training', {}))
        return cls(model=model_cfg, training=train_cfg)
    
    def to_yaml(self, path: str):
        data = {
            'model': asdict(self.model),
            'training': asdict(self.training)
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


# ============================================================
#                    数据集
# ============================================================

class LargeScaleDataset(Dataset):
    """大规模文本数据集，支持流式加载"""
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer: SimpleTokenizer,
                 max_seq_len: int = 512,
                 max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        
        print(f"加载数据集: {data_path}")
        
        # 加载JSON Lines格式
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(tqdm(f, desc="读取文档")):
                    if max_samples and i >= max_samples:
                        break
                    try:
                        obj = json.loads(line.strip())
                        if 'text' in obj and len(obj['text']) > 50:
                            # 预tokenize
                            tokens = tokenizer.encode(obj['text'])
                            # 切分为固定长度
                            for j in range(0, len(tokens) - max_seq_len, max_seq_len // 2):
                                self.samples.append(tokens[j:j + max_seq_len + 1])
                    except json.JSONDecodeError:
                        continue
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - max_seq_len, max_seq_len // 2):
                self.samples.append(tokens[i:i + max_seq_len + 1])
        
        print(f"数据集大小: {len(self.samples)} 样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # 确保长度正确
        if len(tokens) < self.max_seq_len + 1:
            tokens = tokens + [0] * (self.max_seq_len + 1 - len(tokens))
        return torch.tensor(tokens[:self.max_seq_len + 1], dtype=torch.long)


# ============================================================
#                    模型
# ============================================================

class SoulLanguageModel(nn.Module):
    """基于SoulEntity的语言模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # SoulEntity核心
        self.entity = create_soul_entity({
            'dim_q': config.dim_q,
            'dim_z': config.dim_z,
            'hyperbolic_c': config.hyperbolic_c,
            'device': 'cpu',  # 稍后移动到正确设备
        })
        
        # 语言编码器
        self.encoder = TokenEncoder(
            vocab_size=config.vocab_size,
            dim_embed=config.dim_embed,
            dim_u=config.dim_u,
            num_layers=config.num_encoder_layers,
            dropout=config.dropout
        )
        
        # 语言解码器
        self.decoder = StateDecoder(
            vocab_size=config.vocab_size,
            dim_q=config.dim_q,
            dim_embed=config.dim_embed,
            num_layers=config.num_decoder_layers,
            max_len=config.max_seq_len,
            dropout=config.dropout
        )
        
        # 状态投影
        self.state_proj = nn.Linear(config.dim_u, config.dim_q)
        
    def forward(self, 
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: (batch, seq_len) 输入token
            labels: (batch, seq_len) 目标token
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 编码输入序列
        u_seq = self.encoder(input_ids)  # (batch, seq_len, dim_u)
        
        # 聚合编码到状态空间
        u_pooled = u_seq.mean(dim=1)  # (batch, dim_u)
        q_state = self.state_proj(u_pooled)  # (batch, dim_q)
        
        # 解码
        logits = self.decoder(q_state)  # (batch, seq_len, vocab_size)
        
        # 确保seq_len匹配
        if logits.shape[1] != seq_len:
            if logits.shape[1] > seq_len:
                logits = logits[:, :seq_len, :]
            else:
                logits = F.pad(logits, (0, 0, 0, seq_len - logits.shape[1]))
        
        result = {'logits': logits, 'q_state': q_state}
        
        # 计算损失
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=0
            )
            result['loss'] = loss
            result['ppl'] = torch.exp(loss.clamp(max=15))
        
        # 几何正则
        q_norm = q_state.norm(dim=-1)
        result['geometry_loss'] = (q_norm - 1.0).pow(2).mean()
        
        return result


# ============================================================
#                    训练器
# ============================================================

class LargeScaleTrainer:
    """大规模训练器"""
    
    def __init__(self, config: FullConfig, local_rank: int = 0):
        self.config = config
        self.local_rank = local_rank
        self.global_rank = 0
        self.world_size = 1
        
        # 设备
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
        
        # 分布式初始化
        self.is_distributed = config.training.use_ddp and dist.is_initialized()
        if self.is_distributed:
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        
        self.is_main_process = self.global_rank == 0
        
        # 混合精度
        self.use_amp = config.training.use_amp and torch.cuda.is_available()
        if self.use_amp:
            dtype = torch.float16 if config.training.amp_dtype == "float16" else torch.bfloat16
            self.amp_dtype = dtype
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # 输出目录
        self.output_dir = Path(config.training.output_dir)
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志
        self.log_file = self.output_dir / 'training_log.jsonl' if self.is_main_process else None
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_ppl = float('inf')
        
    def setup_model(self):
        """设置模型"""
        self.model = SoulLanguageModel(self.config.model)
        self.model.to(self.device)
        
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        if self.is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")
    
    def setup_data(self):
        """设置数据"""
        # Tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=self.config.model.vocab_size)
        
        # 构建词汇表
        if self.is_main_process:
            print("构建词汇表...")
        
        texts = []
        with open(self.config.training.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 100000:  # 只用前10万文档构建词汇表
                    break
                try:
                    obj = json.loads(line.strip())
                    if 'text' in obj:
                        texts.append(obj['text'][:5000])
                except:
                    continue
        
        self.tokenizer.build_vocab(texts)
        
        if self.is_main_process:
            print(f"词汇表大小: {len(self.tokenizer)}")
        
        # 数据集
        self.train_dataset = LargeScaleDataset(
            self.config.training.data_path,
            self.tokenizer,
            self.config.model.max_seq_len,
            max_samples=1000000  # 限制样本数以加快测试
        )
        
        # 采样器
        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True
            )
        else:
            self.train_sampler = None
        
        # 数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        # 计算总步数
        steps_per_epoch = len(self.train_loader) // self.config.training.gradient_accumulation_steps
        self.total_steps = steps_per_epoch * self.config.training.epochs
        
        if self.is_main_process:
            print(f"每epoch步数: {steps_per_epoch}")
            print(f"总训练步数: {self.total_steps}")
    
    def get_lr(self) -> float:
        """获取当前学习率（带warmup的余弦退火）"""
        cfg = self.config.training
        
        if self.global_step < cfg.warmup_steps:
            # Warmup
            return cfg.learning_rate * self.global_step / cfg.warmup_steps
        else:
            # 余弦退火
            progress = (self.global_step - cfg.warmup_steps) / (self.total_steps - cfg.warmup_steps)
            return cfg.min_learning_rate + 0.5 * (cfg.learning_rate - cfg.min_learning_rate) * \
                   (1 + math.cos(math.pi * progress))
    
    def update_lr(self):
        """更新学习率"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """单步训练"""
        input_ids = batch[:, :-1].to(self.device)
        labels = batch[:, 1:].to(self.device)
        
        # 混合精度前向
        if self.use_amp:
            with autocast(dtype=self.amp_dtype):
                outputs = self.model(input_ids, labels)
                loss = outputs['loss']
                geo_loss = outputs['geometry_loss']
                total_loss = loss + self.config.training.lambda_geometry * geo_loss
                total_loss = total_loss / self.config.training.gradient_accumulation_steps
        else:
            outputs = self.model(input_ids, labels)
            loss = outputs['loss']
            geo_loss = outputs['geometry_loss']
            total_loss = loss + self.config.training.lambda_geometry * geo_loss
            total_loss = total_loss / self.config.training.gradient_accumulation_steps
        
        # 反向传播
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        return {
            'loss': loss.item(),
            'ppl': outputs['ppl'].item(),
            'geo_loss': geo_loss.item(),
        }
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        if self.is_distributed:
            self.train_sampler.set_epoch(epoch)
        
        cfg = self.config.training
        accumulation_steps = cfg.gradient_accumulation_steps
        
        epoch_loss = 0
        epoch_ppl = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", disable=not self.is_main_process)
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            # 训练步
            metrics = self.train_step(batch)
            
            epoch_loss += metrics['loss']
            epoch_ppl += metrics['ppl']
            num_batches += 1
            
            # 梯度累积
            if (step + 1) % accumulation_steps == 0:
                # 梯度裁剪
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    cfg.max_grad_norm
                )
                
                # 优化器步进
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # 更新学习率
                lr = self.update_lr()
                
                # 日志
                if self.global_step % cfg.log_every == 0 and self.is_main_process:
                    avg_loss = epoch_loss / num_batches
                    avg_ppl = epoch_ppl / num_batches
                    
                    log_entry = {
                        'step': self.global_step,
                        'epoch': epoch + 1,
                        'loss': avg_loss,
                        'ppl': avg_ppl,
                        'lr': lr,
                        'geo_loss': metrics['geo_loss'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    with open(self.log_file, 'a') as f:
                        f.write(json.dumps(log_entry) + '\n')
                    
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.3f}',
                        'ppl': f'{avg_ppl:.1f}',
                        'lr': f'{lr:.2e}'
                    })
                
                # 保存检查点
                if self.global_step % cfg.save_every == 0 and self.is_main_process:
                    self.save_checkpoint('latest.pt')
        
        return epoch_loss / num_batches, epoch_ppl / num_batches
    
    def train(self):
        """完整训练"""
        if self.is_main_process:
            print("=" * 70)
            print("开始大规模训练")
            print("=" * 70)
            print(f"设备: {self.device}")
            print(f"分布式: {self.is_distributed} (world_size={self.world_size})")
            print(f"混合精度: {self.use_amp}")
            print(f"梯度累积: {self.config.training.gradient_accumulation_steps}")
            print("=" * 70)
        
        for epoch in range(self.epoch, self.config.training.epochs):
            self.epoch = epoch
            
            avg_loss, avg_ppl = self.train_epoch(epoch)
            
            if self.is_main_process:
                print(f"\nEpoch {epoch+1} 完成:")
                print(f"  平均损失: {avg_loss:.4f}")
                print(f"  平均PPL: {avg_ppl:.2f}")
                
                # 保存最佳模型
                if avg_ppl < self.best_ppl:
                    self.best_ppl = avg_ppl
                    self.save_checkpoint('best.pt')
                    print(f"  ★ 新最佳模型! PPL: {avg_ppl:.2f}")
                
                # 保存epoch检查点
                self.save_checkpoint(f'epoch_{epoch+1}.pt')
                
                print("-" * 70)
        
        if self.is_main_process:
            print("\n" + "=" * 70)
            print("训练完成!")
            print(f"最佳PPL: {self.best_ppl:.2f}")
            print("=" * 70)
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        path = self.output_dir / filename
        
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_ppl': self.best_ppl,
            'config': asdict(self.config.model),
            'training_config': asdict(self.config.training),
        }
        
        torch.save(checkpoint, path)
        print(f"  保存检查点: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        print(f"加载检查点: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch'] + 1
        self.best_ppl = checkpoint['best_ppl']
        
        print(f"  恢复自: epoch {checkpoint['epoch']}, step {self.global_step}")
        print(f"  最佳PPL: {self.best_ppl:.2f}")


# ============================================================
#                    主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='大规模训练')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点')
    parser.add_argument('--local_rank', type=int, default=0, help='DDP local rank')
    
    # 快速配置覆盖
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    # 分布式初始化
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        config = FullConfig.from_yaml(args.config)
    else:
        config = FullConfig()
    
    # 命令行覆盖
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.data:
        config.training.data_path = args.data
    if args.output:
        config.training.output_dir = args.output
    
    if 'WORLD_SIZE' in os.environ:
        config.training.use_ddp = True
    
    # 创建训练器
    trainer = LargeScaleTrainer(config, local_rank)
    trainer.setup_model()
    trainer.setup_data()
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 保存配置
    if trainer.is_main_process:
        config.to_yaml(trainer.output_dir / 'config.yaml')
    
    # 开始训练
    trainer.train()
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

