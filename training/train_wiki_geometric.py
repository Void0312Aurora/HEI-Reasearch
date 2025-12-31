"""
使用中文维基百科训练SoulEntity（完整几何约束版本）

基于理论基础-7的完整架构：
- A3: 变分自由能驱动
- 几何基础第二节: 双曲度量（Lorentz模型，无边界问题）
- 几何基础第三节: 图册一致性
- 几何基础第五节: 联络正交性

运行：
python HEI/training/train_wiki_geometric.py --epochs 5 --batch_size 16
"""

import os
import sys
import json
import argparse
import time
from typing import List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from he_core.language_interface import SimpleTokenizer
from training.geometric_language_training import (
    GeometricLanguageTrainer, 
    GeometricTrainingConfig,
)


class WikiDataset(Dataset):
    """维基百科数据集"""
    
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_len: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        ids = self.tokenizer.encode(text, add_special=True)
        
        # 截断/填充
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.tokenizer.pad_id] * (self.max_len - len(ids))
        
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor([1 if i != self.tokenizer.pad_id else 0 for i in ids], dtype=torch.long),
        }


def load_wiki_data(path: str, max_samples: int = None) -> List[str]:
    """加载维基百科数据"""
    texts = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                if text and len(text) >= 50:
                    texts.append(text[:500])  # 限制长度
            except:
                continue
    
    print(f"加载了 {len(texts)} 条文本")
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                       default='/home/void0312/HEI-Research/HEI/data/wiki/wikipedia-zh-20250901.json')
    parser.add_argument('--max_samples', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--dim_embed', type=int, default=256)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("=" * 60)
    print("几何语言训练 - 维基百科中文")
    print("=" * 60)
    print(f"理论基础: 理论基础-7/几何基础.md")
    print(f"  - 双曲度量: Lorentz模型 (无边界问题)")
    print(f"  - 图册一致性: 多片结构过渡约束")
    print(f"  - 联络正交性: 平行移动保体积")
    print("=" * 60)
    
    # 加载数据
    print("\n[1] 加载数据...")
    texts = load_wiki_data(args.data_path, args.max_samples)
    
    # 划分训练/验证集
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    print(f"训练集: {len(train_texts)}, 验证集: {len(val_texts)}")
    
    # 构建分词器
    print("\n[2] 构建分词器...")
    tokenizer = SimpleTokenizer(vocab_size=10000, mode='char')
    tokenizer.build_vocab(train_texts, min_freq=2)
    print(f"词表大小: {len(tokenizer)}")
    
    # 配置
    config = GeometricTrainingConfig(
        dim_q=args.dim_q,
        dim_embed=args.dim_embed,
        vocab_size=len(tokenizer),
        max_seq_len=args.max_len,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        # 几何约束权重
        lambda_atlas=0.1,
        lambda_conn=0.01,
        lambda_hyp=0.1,
    )
    
    # 创建数据集
    train_dataset = WikiDataset(train_texts, tokenizer, args.max_len)
    val_dataset = WikiDataset(val_texts, tokenizer, args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 创建模型
    print("\n[3] 创建模型...")
    trainer = GeometricLanguageTrainer(config, tokenizer)
    trainer = trainer.to(args.device)
    
    param_count = sum(p.numel() for p in trainer.parameters())
    print(f"模型参数: {param_count:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    
    # 训练
    print("\n[4] 开始训练...")
    print("-" * 80)
    
    best_val_F = float('inf')
    
    for epoch in range(args.epochs):
        trainer.train()
        epoch_stats = {
            'loss': 0, 'F': 0, 'L_geo': 0, 'E_pred': 0,
            'L_atlas': 0, 'L_conn': 0, 'L_hyp': 0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            result = trainer.train_step(batch)
            
            result['loss'].backward()
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # 累积统计
            epoch_stats['loss'] += result['loss'].item()
            epoch_stats['F'] += result['free_energy'].item()
            epoch_stats['L_geo'] += result['geometric_loss'].item()
            epoch_stats['E_pred'] += result['prediction_error'].item()
            epoch_stats['L_atlas'] += result['L_atlas']
            epoch_stats['L_conn'] += result['L_conn']
            epoch_stats['L_hyp'] += result['L_hyp']
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{result['loss'].item():.3f}",
                'F': f"{result['free_energy'].item():.3f}",
                'Geo': f"{result['geometric_loss'].item():.4f}",
                'PPL': f"{result['perplexity'].item():.1f}",
            })
        
        # 计算epoch平均
        n_batches = len(train_loader)
        for k in epoch_stats:
            epoch_stats[k] /= n_batches
        
        # 验证
        trainer.eval()
        val_result = trainer.evaluate(val_loader)
        
        print(f"\nEpoch {epoch+1} 总结:")
        print(f"  训练: Loss={epoch_stats['loss']:.4f}, F={epoch_stats['F']:.4f}, "
              f"L_geo={epoch_stats['L_geo']:.6f}")
        print(f"         L_atlas={epoch_stats['L_atlas']:.6f}, "
              f"L_conn={epoch_stats['L_conn']:.6f}, L_hyp={epoch_stats['L_hyp']:.6f}")
        print(f"  验证: F={val_result['avg_free_energy']:.4f}, "
              f"PPL={val_result['avg_perplexity']:.2f}, "
              f"L_geo={val_result['avg_geometric_loss']:.6f}")
        
        # 生成样本
        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            print("\n  生成样本:")
            for prompt in ["中国", "历史", ""]:
                text = trainer.generate(prompt=prompt, max_len=30, temperature=0.8)
                print(f"    '{prompt}' → {text[:50]}...")
        
        # 几何诊断
        trainer.reset_entity(4)
        diagnostics = trainer.get_geometric_diagnostics()
        print(f"\n  几何诊断:")
        print(f"    活跃图册数: {diagnostics['active_charts']:.2f}")
        print(f"    曲率代理: {diagnostics['curvature_proxy']:.6f}")
        print(f"    双曲距离: mean={diagnostics['hyperbolic_distances'].mean().item():.4f}")
        
        # 保存最佳模型
        if val_result['avg_free_energy'] < best_val_F:
            best_val_F = val_result['avg_free_energy']
            print(f"  ★ 新最佳模型! F={best_val_F:.4f}")
        
        print("-" * 80)
    
    print("\n训练完成!")
    print(f"最佳验证F: {best_val_F:.4f}")


if __name__ == "__main__":
    main()


