"""
SoulLanguageModel V2 训练脚本

改进:
1. 验证集支持，早停机制
2. 更大的数据规模
3. 学习率warmup
4. 梯度累积
5. 更好的评估指标
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Optional
from tqdm import tqdm
import time
from datetime import datetime
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.soul_language_model_v2 import SoulLanguageModelV2, ModelConfig
from he_core.language_interface import SimpleTokenizer


class WikiDataset(Dataset):
    """优化的Wiki数据集"""
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: SimpleTokenizer,
                 max_len: int = 256,
                 max_samples: int = 100000):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        print(f"Loading data from {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading"):
                if len(self.samples) >= max_samples:
                    break
                try:
                    doc = json.loads(line.strip())
                    text = doc.get('text', '')
                    if len(text) < 50:
                        continue
                    
                    # 分割成段落，每段作为一个样本
                    for para in text.split('\n'):
                        para = para.strip()
                        if 50 < len(para) < 1000:
                            # 预分词
                            ids = tokenizer.encode(para)
                            if 10 < len(ids) <= max_len:
                                self.samples.append(ids)
                                if len(self.samples) >= max_samples:
                                    break
                except:
                    continue
                    
        print(f"Loaded {len(self.samples)} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ids = self.samples[idx]
        
        # 随机截取子序列（数据增强）
        if len(ids) > self.max_len:
            start = random.randint(0, len(ids) - self.max_len)
            ids = ids[start:start + self.max_len]
        
        # Padding
        input_ids = ids[:-1]
        target_ids = ids[1:]
        
        pad_len = self.max_len - 1 - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [0] * pad_len
            target_ids = target_ids + [0] * pad_len
            
        attention_mask = [1 if i != 0 else 0 for i in input_ids]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(target_ids, dtype=torch.long),
        }


def build_tokenizer(data_path: str, vocab_size: int, sample_size: int = 20000) -> SimpleTokenizer:
    """构建分词器"""
    print(f"Building tokenizer (vocab_size={vocab_size})")
    
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, mode='char')
    texts = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Sampling"):
            if len(texts) >= sample_size:
                break
            try:
                doc = json.loads(line.strip())
                text = doc.get('text', '')
                if len(text) > 50:
                    texts.append(text[:2000])
            except:
                continue
                
    tokenizer.build_vocab(texts, min_freq=3)
    return tokenizer


def get_lr(step: int, warmup_steps: int, max_lr: float, total_steps: int) -> float:
    """Warmup + Cosine decay学习率"""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())


@torch.no_grad()
def evaluate(model, dataloader, device) -> Dict[str, float]:
    """评估"""
    model.eval()
    total_loss = 0
    total_lm_loss = 0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask, labels)
        
        # 计算有效token数
        mask = (labels != 0).float()
        num_tokens = mask.sum().item()
        
        total_loss += outputs['loss'].item() * num_tokens
        total_lm_loss += outputs['lm_loss'].item() * num_tokens
        total_tokens += num_tokens
        
    avg_loss = total_loss / max(total_tokens, 1)
    avg_lm_loss = total_lm_loss / max(total_tokens, 1)
    ppl = torch.exp(torch.tensor(avg_lm_loss)).item()
    
    model.train()
    
    return {
        'loss': avg_loss,
        'lm_loss': avg_lm_loss,
        'ppl': ppl,
    }


def generate_samples(model, tokenizer, prompts: List[str], device, max_len: int = 50):
    """生成样本"""
    model.eval()
    
    print("\n生成样本:")
    for prompt in prompts:
        try:
            # 编码prompt
            ids = tokenizer.encode(prompt, add_special=False)
            prompt_tensor = torch.tensor([ids], device=device)
            
            # 生成
            generated = model.generate(
                prompt_tensor,
                max_new_tokens=max_len,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            # 解码
            text = tokenizer.decode(generated[0].cpu().tolist())
            print(f"  {prompt} -> {text[:80]}...")
        except Exception as e:
            print(f"  {prompt} -> 生成失败: {e}")
            
    model.train()


def train(args):
    """主训练函数"""
    
    print("=" * 60)
    print("SoulLanguageModel V2 训练")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = torch.device(args.device)
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 分词器
    tokenizer_path = os.path.join(args.checkpoint_dir, 'tokenizer_v2.json')
    if os.path.exists(tokenizer_path) and args.resume:
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = SimpleTokenizer()
        tokenizer.load(tokenizer_path)
    else:
        tokenizer = build_tokenizer(args.data_path, args.vocab_size)
        tokenizer.save(tokenizer_path)
        
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # 模型配置
    config = ModelConfig(
        vocab_size=len(tokenizer),
        dim_embed=args.dim_embed,
        dim_hidden=args.dim_hidden,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_q=args.dim_q,
        dim_z=args.dim_z,
        dropout=args.dropout,
        max_len=args.max_len,
        lm_weight=1.0,
        soul_weight=args.soul_weight,
    )
    
    # 创建模型
    print("\nCreating model...")
    model = SoulLanguageModelV2(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 数据集
    print("\nLoading dataset...")
    full_dataset = WikiDataset(
        args.data_path,
        tokenizer,
        max_len=args.max_len,
        max_samples=args.max_samples
    )
    
    # 分割训练/验证集
    val_size = min(int(len(full_dataset) * 0.1), 5000)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )
    
    # 训练状态
    global_step = 0
    best_val_ppl = float('inf')
    patience_counter = 0
    total_steps = len(train_loader) * args.epochs
    
    # 恢复检查点
    if args.resume:
        ckpt_path = os.path.join(args.checkpoint_dir, 'latest_v2.pt')
        if os.path.exists(ckpt_path):
            print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            global_step = ckpt.get('global_step', 0)
            best_val_ppl = ckpt.get('best_val_ppl', float('inf'))
    
    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_lm_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            # 学习率warmup
            lr = get_lr(global_step, args.warmup_steps, args.lr, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # 前向
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            epoch_lm_loss += outputs['lm_loss'].item()
            num_batches += 1
            global_step += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ppl': f"{outputs['ppl'].item():.2f}",
                'lr': f"{lr:.2e}"
            })
            
            # 定期评估
            if global_step % args.eval_every == 0:
                val_metrics = evaluate(model, val_loader, device)
                print(f"\n  Step {global_step}: val_ppl={val_metrics['ppl']:.2f}")
                
                # 早停检查
                if val_metrics['ppl'] < best_val_ppl:
                    best_val_ppl = val_metrics['ppl']
                    patience_counter = 0
                    
                    # 保存最佳模型
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'best_val_ppl': best_val_ppl,
                    }, os.path.join(args.checkpoint_dir, 'best_v2.pt'))
                else:
                    patience_counter += 1
                    
                if patience_counter >= args.patience:
                    print(f"  早停: {args.patience}次评估无改进")
                    break
                    
            # 定期保存
            if global_step % args.save_every == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'best_val_ppl': best_val_ppl,
                }, os.path.join(args.checkpoint_dir, 'latest_v2.pt'))
        
        if patience_counter >= args.patience:
            break
            
        # Epoch结束
        avg_loss = epoch_loss / num_batches
        avg_lm_loss = epoch_lm_loss / num_batches
        avg_ppl = torch.exp(torch.tensor(avg_lm_loss)).item()
        
        print(f"\nEpoch {epoch+1} 完成:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Train PPL: {avg_ppl:.2f}")
        
        # 验证
        val_metrics = evaluate(model, val_loader, device)
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val PPL: {val_metrics['ppl']:.2f}")
        
        # 生成样本
        generate_samples(
            model, tokenizer,
            ["数学是", "人工智能", "中国历史"],
            device
        )
        
    # 最终保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_val_ppl': val_metrics['ppl'],
    }, os.path.join(args.checkpoint_dir, 'final_v2.pt'))
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳Val PPL: {best_val_ppl:.2f}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description='SoulLanguageModel V2 训练')
    
    # 数据
    parser.add_argument('--data_path', type=str,
                       default='/home/void0312/HEI-Research/HEI/data/wiki/wikipedia-zh-20250901.json')
    parser.add_argument('--max_samples', type=int, default=100000)
    parser.add_argument('--max_len', type=int, default=128)
    
    # 模型
    parser.add_argument('--vocab_size', type=int, default=8000)
    parser.add_argument('--dim_embed', type=int, default=256)
    parser.add_argument('--dim_hidden', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--dim_z', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--soul_weight', type=float, default=0.01)
    
    # 训练
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--patience', type=int, default=5)
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str,
                       default='/home/void0312/HEI-Research/HEI/checkpoints/soul_lm_v2')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=1000)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"数据文件不存在: {args.data_path}")
        return
        
    train(args)


if __name__ == "__main__":
    main()

