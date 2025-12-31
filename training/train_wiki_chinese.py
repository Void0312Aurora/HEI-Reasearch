"""
中文维基百科训练脚本

使用 HEI/data/wiki/wikipedia-zh-20250901.json 进行大规模训练

用法:
    python train_wiki_chinese.py --epochs 10 --batch_size 32

注意:
    - 需要在 PINNs conda 环境下运行
    - 需要 GPU (推荐 8GB+ 显存)
"""

import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional
from tqdm import tqdm
import time
from datetime import datetime

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from he_core.soul_entity import SoulEntity, create_soul_entity
from he_core.language_interface import LanguagePort, SimpleTokenizer
from training.language_trainer import TrainingConfig, SoulLanguageModel


class WikiChineseDataset(Dataset):
    """
    维基百科中文数据集
    
    支持流式加载和内存缓存
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: SimpleTokenizer,
                 max_len: int = 256,
                 max_samples: int = 100000,
                 cache: bool = True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        print(f"Loading Wikipedia data from {data_path}")
        print(f"Max samples: {max_samples}")
        
        count = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading"):
                if count >= max_samples:
                    break
                    
                try:
                    doc = json.loads(line.strip())
                    text = doc.get('text', '')
                    
                    # 过滤太短的文档
                    if len(text) < 100:
                        continue
                        
                    # 分割成段落
                    paragraphs = text.split('\n')
                    for para in paragraphs:
                        para = para.strip()
                        if len(para) > 50 and len(para) < 2000:
                            if cache:
                                # 预分词并缓存
                                ids = tokenizer.encode(para)
                                if len(ids) > 10:
                                    self.samples.append(ids)
                                    count += 1
                            else:
                                self.samples.append(para)
                                count += 1
                                
                            if count >= max_samples:
                                break
                except Exception as e:
                    continue
                    
        print(f"Loaded {len(self.samples)} samples")
        self.is_tokenized = cache
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.is_tokenized:
            ids = self.samples[idx]
        else:
            ids = self.tokenizer.encode(self.samples[idx])
            
        # 截断
        if len(ids) > self.max_len:
            # 随机起始位置
            start = torch.randint(0, len(ids) - self.max_len, (1,)).item()
            ids = ids[start:start + self.max_len]
            
        # 输入和目标
        input_ids = ids[:-1]
        target_ids = ids[1:]
        
        # Padding
        pad_len = self.max_len - 1 - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
            target_ids = target_ids + [self.tokenizer.pad_id] * pad_len
            
        attention_mask = [1 if i != self.tokenizer.pad_id else 0 for i in input_ids]
        
        return {
            'input_ids': torch.tensor(input_ids),
            'target_ids': torch.tensor(target_ids),
            'attention_mask': torch.tensor(attention_mask),
        }


def build_tokenizer_from_wiki(data_path: str, 
                               vocab_size: int = 8000,
                               sample_size: int = 10000) -> SimpleTokenizer:
    """
    从维基数据构建分词器
    """
    print(f"Building tokenizer from {data_path}")
    
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, mode='char')
    
    texts = []
    count = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Sampling for tokenizer"):
            if count >= sample_size:
                break
            try:
                doc = json.loads(line.strip())
                text = doc.get('text', '')
                if len(text) > 50:
                    texts.append(text[:1000])  # 取前1000字符
                    count += 1
            except:
                continue
                
    tokenizer.build_vocab(texts, min_freq=5)
    
    return tokenizer


def train(args):
    """主训练函数"""
    
    print("=" * 60)
    print("SoulEntity 中文维基百科训练")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据路径: {args.data_path}")
    print(f"设备: {args.device}")
    
    device = torch.device(args.device)
    
    # 配置
    config = TrainingConfig(
        dim_q=args.dim_q,
        dim_u=args.dim_q,
        dim_z=args.dim_z,
        dim_embed=args.dim_embed,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_seq_len=args.max_len,
        device=args.device,
        num_workers=args.num_workers,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # 构建或加载分词器
    tokenizer_path = os.path.join(args.checkpoint_dir, 'tokenizer.json')
    
    if args.resume and os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        tokenizer.load(tokenizer_path)
    else:
        tokenizer = build_tokenizer_from_wiki(
            args.data_path, 
            vocab_size=config.vocab_size,
            sample_size=args.tokenizer_samples
        )
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")
        
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # 创建模型
    print("\nCreating model...")
    model = SoulLanguageModel(config).to(device)
    model.language_port.set_tokenizer(tokenizer)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * args.max_samples // config.batch_size,
        eta_min=config.learning_rate * 0.01
    )
    
    # 加载检查点
    global_step = 0
    start_epoch = 0
    
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'latest.pt')
        if os.path.exists(checkpoint_path):
            print(f"Resuming from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_step = checkpoint.get('global_step', 0)
            start_epoch = checkpoint.get('epoch', 0)
            
    # 创建数据集
    print("\nLoading dataset...")
    dataset = WikiChineseDataset(
        args.data_path,
        tokenizer,
        max_len=config.max_seq_len,
        max_samples=args.max_samples,
        cache=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_ppl = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in pbar:
            # 移到设备
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 前向
            output = model(input_ids, attention_mask, target_ids)
            
            # 损失
            lm_loss = output['loss']
            free_energy = output['free_energy']
            
            total_loss = (
                config.lm_loss_weight * lm_loss +
                config.kl_loss_weight * free_energy
            )
            
            # 反向
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # 统计
            epoch_loss += total_loss.item()
            epoch_ppl += output['perplexity'].item()
            num_batches += 1
            global_step += 1
            
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'ppl': f"{output['perplexity'].item():.2f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # 定期保存
            if global_step % config.save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, global_step, epoch,
                    os.path.join(args.checkpoint_dir, f'step_{global_step}.pt')
                )
                save_checkpoint(
                    model, optimizer, scheduler, global_step, epoch,
                    os.path.join(args.checkpoint_dir, 'latest.pt')
                )
                
        # Epoch 结束
        avg_loss = epoch_loss / num_batches
        avg_ppl = epoch_ppl / num_batches
        
        print(f"\nEpoch {epoch+1} 完成:")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  平均困惑度: {avg_ppl:.2f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, scheduler, global_step, epoch,
                os.path.join(args.checkpoint_dir, 'best.pt')
            )
            print("  >> 保存最佳模型")
            
        # 生成样本
        generate_samples(model, tokenizer, device)
        
    print("\n训练完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def save_checkpoint(model, optimizer, scheduler, global_step, epoch, path):
    """保存检查点"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
    }, path)


def generate_samples(model, tokenizer, device):
    """生成样本"""
    model.eval()
    
    prompts = ["数学是", "人工智能", "中国历史"]
    
    print("\n生成样本:")
    for prompt in prompts:
        try:
            with torch.no_grad():
                text = model.generate(prompt, max_len=50)
                print(f"  {prompt} -> {text[:60]}...")
        except Exception as e:
            print(f"  {prompt} -> 生成失败: {e}")
            
    model.train()


def main():
    parser = argparse.ArgumentParser(description='SoulEntity 中文训练')
    
    # 数据
    parser.add_argument('--data_path', type=str, 
                       default='/home/void0312/HEI-Research/HEI/data/wiki/wikipedia-zh-20250901.json',
                       help='维基数据路径')
    parser.add_argument('--max_samples', type=int, default=50000,
                       help='最大样本数')
    parser.add_argument('--tokenizer_samples', type=int, default=10000,
                       help='构建分词器的样本数')
    
    # 模型
    parser.add_argument('--dim_q', type=int, default=128,
                       help='几何状态维度')
    parser.add_argument('--dim_z', type=int, default=32,
                       help='上下文维度')
    parser.add_argument('--dim_embed', type=int, default=256,
                       help='嵌入维度')
    parser.add_argument('--vocab_size', type=int, default=8000,
                       help='词表大小')
    parser.add_argument('--max_len', type=int, default=128,
                       help='最大序列长度')
    
    # 训练
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--epochs', type=int, default=5,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='/home/void0312/HEI-Research/HEI/checkpoints/soul_lm',
                       help='检查点目录')
    parser.add_argument('--resume', action='store_true',
                       help='从检查点恢复')
    parser.add_argument('--log_every', type=int, default=100,
                       help='日志间隔')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='保存间隔')
    
    args = parser.parse_args()
    
    # 检查数据文件
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        return
        
    train(args)


if __name__ == "__main__":
    main()

