"""
使用中文维基百科训练SoulEntity的语言技能

基于理论基础-7的正确架构：
- 语言只是端口（A5公理）
- 训练目标是自由能F（A3公理）
- SoulEntity的接触哈密顿动力学是核心（L1模板）

运行：
python HEI/training/train_wiki_soul.py --epochs 5 --batch_size 8

数据：
使用 HEI/data/wiki/wikipedia-zh-20250901.json
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from he_core.soul_entity import create_soul_entity
from he_core.language_interface import SimpleTokenizer
from training.soul_language_training import (
    SoulLanguageTrainer, 
    TrainingConfig, 
    LanguageDataset,
    collate_fn
)


def load_wiki_data(path: str, max_samples: int = None) -> List[str]:
    """
    加载维基百科数据
    
    每行是一个JSON对象，包含 'text' 字段
    """
    texts = []
    print(f"加载数据: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="读取行")):
            if max_samples and i >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                if len(text) > 50:  # 过滤太短的文本
                    texts.append(text)
            except:
                continue
    
    print(f"加载了 {len(texts)} 条文本")
    return texts


def create_dataloaders(texts: List[str],
                       tokenizer: SimpleTokenizer,
                       config: TrainingConfig,
                       val_ratio: float = 0.1):
    """创建训练和验证数据加载器"""
    
    # 分割数据
    n_val = int(len(texts) * val_ratio)
    train_texts = texts[n_val:]
    val_texts = texts[:n_val]
    
    print(f"训练集: {len(train_texts)}, 验证集: {len(val_texts)}")
    
    train_dataset = LanguageDataset(train_texts, tokenizer, config.max_seq_len)
    val_dataset = LanguageDataset(val_texts, tokenizer, config.max_seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


class FreeEnergyTracker:
    """自由能追踪器（A3验证）"""
    
    def __init__(self):
        self.F_history = []
        self.E_pred_history = []
        self.V_history = []
        
    def update(self, result: Dict[str, Any]):
        self.F_history.append(result['free_energy'].item())
        self.E_pred_history.append(result['prediction_error'].item())
        
    def get_stats(self) -> Dict[str, float]:
        if len(self.F_history) == 0:
            return {}
        return {
            'avg_F': sum(self.F_history) / len(self.F_history),
            'avg_E_pred': sum(self.E_pred_history) / len(self.E_pred_history),
            'F_trend': self.F_history[-1] - self.F_history[0] if len(self.F_history) > 1 else 0,
        }
    
    def reset(self):
        self.F_history = []
        self.E_pred_history = []


def train_epoch(trainer: SoulLanguageTrainer,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                epoch: int,
                config: TrainingConfig) -> Dict[str, float]:
    """训练一个epoch"""
    
    trainer.train()
    tracker = FreeEnergyTracker()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(config.device) for k, v in batch.items()}
        
        # 前向传播
        result = trainer.train_step(batch)
        
        # 反向传播（优化自由能F）
        loss = result['loss']
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 追踪
        tracker.update(result)
        
        # 更新进度条
        pbar.set_postfix({
            'F': f"{result['free_energy'].item():.4f}",
            'E_pred': f"{result['prediction_error'].item():.4f}",
            'PPL': f"{result['perplexity'].item():.2f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # 定期打印诊断
        if batch_idx > 0 and batch_idx % 500 == 0:
            diag = trainer.get_entity_diagnostics()
            print(f"\n  [诊断] q_norm={diag['q_norm']:.4f}, "
                  f"p_norm={diag['p_norm']:.4f}, "
                  f"kinetic={diag['kinetic_energy']:.4f}")
    
    return tracker.get_stats()


@torch.no_grad()
def evaluate(trainer: SoulLanguageTrainer,
             dataloader: DataLoader,
             config: TrainingConfig) -> Dict[str, float]:
    """评估"""
    
    trainer.eval()
    tracker = FreeEnergyTracker()
    
    for batch in tqdm(dataloader, desc="评估"):
        batch = {k: v.to(config.device) for k, v in batch.items()}
        result = trainer.train_step(batch)
        tracker.update(result)
    
    stats = tracker.get_stats()
    stats['perplexity'] = torch.exp(torch.tensor(stats['avg_E_pred'])).item()
    
    return stats


def generate_samples(trainer: SoulLanguageTrainer,
                    prompts: List[str] = None,
                    n_samples: int = 3) -> List[str]:
    """生成样本"""
    
    if prompts is None:
        prompts = ["", "中国", "科学"]
    
    samples = []
    for prompt in prompts[:n_samples]:
        text = trainer.generate(prompt, max_len=50, temperature=0.8)
        samples.append(f"[{prompt or '无提示'}] {text}")
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="训练SoulEntity语言技能")
    
    # 数据参数
    parser.add_argument('--data_path', type=str, 
                       default='HEI/data/wiki/wikipedia-zh-20250901.json',
                       help='维基百科数据路径')
    parser.add_argument('--max_samples', type=int, default=100000,
                       help='最大样本数')
    
    # 模型参数
    parser.add_argument('--dim_q', type=int, default=64,
                       help='构形空间维度')
    parser.add_argument('--dim_embed', type=int, default=256,
                       help='嵌入维度')
    parser.add_argument('--vocab_size', type=int, default=8000,
                       help='词表大小')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_seq_len', type=int, default=128)
    
    # 自由能权重（A3公理）
    parser.add_argument('--beta_kl', type=float, default=0.01,
                       help='KL正则权重')
    parser.add_argument('--gamma_pred', type=float, default=1.0,
                       help='预测误差权重')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints/soul_wiki')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SoulEntity 语言技能训练")
    print("=" * 70)
    print("核心原则:")
    print("  1. 语言只是端口 (A5公理)")
    print("  2. 训练目标是自由能F (A3公理)")
    print("  3. SoulEntity动力学是核心 (L1模板)")
    print("=" * 70)
    
    # 加载数据
    texts = load_wiki_data(args.data_path, args.max_samples)
    if len(texts) < 100:
        print("警告: 数据量太少，可能影响训练效果")
    
    # 创建分词器
    print("\n构建词表...")
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size, mode='char')
    tokenizer.build_vocab(texts, min_freq=2)
    
    # 创建配置
    config = TrainingConfig(
        dim_q=args.dim_q,
        dim_embed=args.dim_embed,
        vocab_size=len(tokenizer),
        beta_kl=args.beta_kl,
        gamma_pred=args.gamma_pred,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(texts, tokenizer, config)
    
    # 创建训练器
    print("\n创建SoulLanguageTrainer...")
    trainer = SoulLanguageTrainer(config, tokenizer)
    trainer.to(args.device)
    
    # 打印模型信息
    n_params = sum(p.numel() for p in trainer.parameters())
    n_entity_params = sum(p.numel() for p in trainer.entity.parameters())
    n_port_params = sum(p.numel() for p in trainer.language_port.parameters())
    
    print(f"\n模型参数:")
    print(f"  总参数: {n_params:,}")
    print(f"  SoulEntity参数: {n_entity_params:,}")
    print(f"  语言端口参数: {n_port_params:,}")
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    best_val_F = float('inf')
    
    print("\n开始训练...")
    print("-" * 70)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        
        # 训练
        train_stats = train_epoch(trainer, train_loader, optimizer, scheduler, epoch, config)
        
        # 评估
        val_stats = evaluate(trainer, val_loader, config)
        
        # 打印结果
        print(f"\n训练: avg_F={train_stats['avg_F']:.4f}, F_trend={train_stats['F_trend']:.4f}")
        print(f"验证: avg_F={val_stats['avg_F']:.4f}, PPL={val_stats['perplexity']:.2f}")
        
        # 生成样本
        print("\n生成样本:")
        samples = generate_samples(trainer)
        for s in samples:
            print(f"  {s[:60]}...")
        
        # SoulEntity诊断
        diag = trainer.get_entity_diagnostics()
        print(f"\nSoulEntity状态:")
        print(f"  q_norm={diag['q_norm']:.4f}, p_norm={diag['p_norm']:.4f}")
        print(f"  kinetic={diag['kinetic_energy']:.4f}, experience_size={diag['experience_size']}")
        
        # 保存最佳模型
        if val_stats['avg_F'] < best_val_F:
            best_val_F = val_stats['avg_F']
            save_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_F': val_stats['avg_F'],
                'config': config.__dict__,
            }, save_path)
            print(f"  ✓ 保存最佳模型 (F={best_val_F:.4f})")
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"最佳验证F: {best_val_F:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

