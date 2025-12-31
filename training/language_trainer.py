"""
语言训练框架 (Language Training Framework)

整合 SoulEntity 与语言数据的大规模训练

核心功能:
1. 中文语料数据加载 (Wiki, CLUE)
2. 几何-语言联合训练
3. 多阶段训练策略
4. 分布式训练支持
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Iterator
import random
from dataclasses import dataclass
from tqdm import tqdm
import time

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from he_core.soul_entity import SoulEntity, create_soul_entity
from he_core.language_interface import LanguagePort, SimpleTokenizer


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型
    dim_q: int = 128
    dim_u: int = 128
    dim_z: int = 32
    dim_embed: int = 256
    vocab_size: int = 10000
    
    # 训练
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_seq_len: int = 256
    
    # 损失权重
    lm_loss_weight: float = 1.0      # 语言模型损失
    recon_loss_weight: float = 0.1   # 重建损失
    kl_loss_weight: float = 0.01     # KL 损失
    dynamics_loss_weight: float = 0.1  # 动力学一致性损失
    
    # 数据
    data_path: str = ""
    tokenizer_path: str = ""
    checkpoint_dir: str = "checkpoints"
    
    # 设备
    device: str = "cuda"
    num_workers: int = 4
    
    # 日志
    log_every: int = 100
    save_every: int = 1000
    eval_every: int = 500


class ChineseTextDataset(Dataset):
    """
    中文文本数据集
    
    支持:
    - Wikipedia JSON 格式
    - CLUE 纯文本格式
    - 通用 JSONL 格式
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: SimpleTokenizer,
                 max_len: int = 256,
                 data_format: str = 'auto'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        # 检测格式
        if data_format == 'auto':
            if data_path.endswith('.json'):
                data_format = 'wiki_json'
            elif data_path.endswith('.jsonl'):
                data_format = 'jsonl'
            else:
                data_format = 'text'
                
        # 加载数据
        print(f"Loading data from {data_path} (format: {data_format})")
        
        if data_format == 'wiki_json':
            self._load_wiki_json(data_path)
        elif data_format == 'jsonl':
            self._load_jsonl(data_path)
        else:
            self._load_text(data_path)
            
        print(f"Loaded {len(self.samples)} samples")
        
    def _load_wiki_json(self, path: str, max_samples: int = 100000):
        """加载 Wikipedia JSON 格式"""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(self.samples) >= max_samples:
                    break
                try:
                    doc = json.loads(line.strip())
                    text = doc.get('text', '')
                    if len(text) > 50:  # 过滤太短的
                        # 分割成段落
                        paragraphs = text.split('\n')
                        for para in paragraphs:
                            para = para.strip()
                            if len(para) > 50:
                                self.samples.append(para)
                except:
                    continue
                    
    def _load_jsonl(self, path: str, max_samples: int = 100000):
        """加载 JSONL 格式"""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(self.samples) >= max_samples:
                    break
                try:
                    doc = json.loads(line.strip())
                    text = doc.get('text', doc.get('content', ''))
                    if len(text) > 20:
                        self.samples.append(text)
                except:
                    continue
                    
    def _load_text(self, path: str, max_samples: int = 100000):
        """加载纯文本格式"""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(self.samples) >= max_samples:
                    break
                line = line.strip()
                if len(line) > 20:
                    self.samples.append(line)
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # 分词
        ids = self.tokenizer.encode(text)
        
        # 截断/padding
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            
        # 创建输入和目标
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


class SoulLanguageModel(nn.Module):
    """
    灵魂语言模型
    
    整合 SoulEntity 和 LanguagePort
    实现几何动力学与语言的联合模型
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # 核心实体
        entity_config = {
            'dim_q': config.dim_q,
            'dim_u': config.dim_u,
            'dim_z': config.dim_z,
            'num_charts': 4,
            'stiffness': 0.01,
        }
        self.entity = create_soul_entity(entity_config)
        
        # 语言端口
        self.language_port = LanguagePort(
            vocab_size=config.vocab_size,
            dim_q=config.dim_q,
            dim_u=config.dim_u,
            dim_embed=config.dim_embed,
        )
        
        # 状态初始化网络 (从编码初始化状态)
        self.state_init = nn.Sequential(
            nn.Linear(config.dim_u, config.dim_q * 2),
            nn.Tanh(),
            nn.Linear(config.dim_q * 2, config.dim_q * 2 + 1)  # q, p, s
        )
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                target_ids: Optional[torch.Tensor] = None,
                num_steps: int = 5) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        流程:
        1. 编码输入 tokens → u
        2. 初始化状态 q,p,s
        3. 运行几何动力学
        4. 解码状态 → token logits
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 1. 编码
        u_pooled = self.language_port.encoder.encode_pooled(input_ids, attention_mask)
        
        # 2. 初始化状态
        state_flat = self.state_init(u_pooled)
        
        # 3. 运行动力学 (简化版本，避免 in-place 操作)
        # 直接使用状态初始化网络的输出作为最终状态
        # 完整动力学需要更多调整以支持梯度
        final_q = state_flat[:, :self.config.dim_q]  # 取 q 部分
        
        # 创建一个简单的状态对象用于自由能计算
        from he_core.state import ContactState
        final_state = ContactState(self.config.dim_q, batch_size, device, state_flat)
        
        # 4. 解码
        output = {'q': final_q}
        
        if target_ids is not None:
            # 计算语言模型损失
            loss_dict = self.language_port.compute_loss(
                input_ids, attention_mask,
                final_q, target_ids
            )
            output.update(loss_dict)
            
        # 计算自由能
        output['free_energy'] = self.entity.compute_free_energy(final_state)
        
        return output
    
    def generate(self,
                 prompt: str,
                 max_len: int = 50,
                 temperature: float = 1.0) -> str:
        """
        生成文本
        """
        # 编码 prompt
        u, _ = self.language_port.encode_text(prompt)
        
        # 初始化状态
        state_flat = self.state_init(u)
        
        # 演化
        self.entity.reset(1, str(u.device))
        self.entity.state = type(self.entity.state)(
            self.config.dim_q, 1, u.device, state_flat
        )
        
        for _ in range(5):
            self.entity.step({'default': u}, dt=0.1)
            
        # 解码
        texts = self.language_port.decode_state(
            self.entity.state.q,
            max_len=max_len,
            temperature=temperature
        )
        
        return texts[0]


class LanguageTrainer:
    """
    语言训练器
    
    实现完整的训练循环
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 构建分词器
        self.tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        
        # 构建模型
        self.model = SoulLanguageModel(config).to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs * 10000,
            eta_min=config.learning_rate * 0.01
        )
        
        # 日志
        self.global_step = 0
        self.best_loss = float('inf')
        
    def build_tokenizer(self, texts: List[str]):
        """构建分词器"""
        self.tokenizer.build_vocab(texts)
        self.model.language_port.set_tokenizer(self.tokenizer)
        
    def prepare_data(self, data_path: str) -> DataLoader:
        """准备数据"""
        dataset = ChineseTextDataset(
            data_path,
            self.tokenizer,
            max_len=self.config.max_seq_len
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        return loader
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        
        # 移到设备
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # 前向
        output = self.model(input_ids, attention_mask, target_ids)
        
        # 计算总损失
        lm_loss = output['loss']
        free_energy = output['free_energy']
        
        total_loss = (
            self.config.lm_loss_weight * lm_loss +
            self.config.kl_loss_weight * free_energy
        )
        
        # 反向
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        return {
            'loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'free_energy': free_energy.item(),
            'ppl': output['perplexity'].item(),
            'lr': self.scheduler.get_last_lr()[0],
        }
    
    def train(self, train_loader: DataLoader):
        """训练循环"""
        print("Starting training...")
        print(f"  Total epochs: {self.config.num_epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Steps per epoch: {len(train_loader)}")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            epoch_ppl = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch in pbar:
                metrics = self.train_step(batch)
                
                epoch_loss += metrics['loss']
                epoch_ppl += metrics['ppl']
                
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'ppl': f"{metrics['ppl']:.2f}",
                    'lr': f"{metrics['lr']:.2e}"
                })
                
                # 定期日志
                if self.global_step % self.config.log_every == 0:
                    self._log_metrics(metrics)
                    
                # 定期保存
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()
                    
            # Epoch 结束
            avg_loss = epoch_loss / len(train_loader)
            avg_ppl = epoch_ppl / len(train_loader)
            
            print(f"\nEpoch {epoch+1} completed:")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Avg PPL: {avg_ppl:.2f}")
            
            # 生成样本
            self._generate_samples()
            
    def _log_metrics(self, metrics: Dict[str, float]):
        """记录指标"""
        # 可以集成 wandb 或 tensorboard
        pass
    
    def _generate_samples(self):
        """生成样本"""
        self.model.eval()
        
        prompts = ["数学是", "人工智能", "今天天气"]
        
        print("\n生成样本:")
        for prompt in prompts:
            try:
                with torch.no_grad():
                    text = self.model.generate(prompt, max_len=30)
                    print(f"  {prompt} -> {text}")
            except Exception as e:
                print(f"  生成失败: {e}")
                
        self.model.train()
        
    def save_checkpoint(self, path: Optional[str] = None):
        """保存检查点"""
        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self.config.checkpoint_dir,
                f"step_{self.global_step}.pt"
            )
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
        }, path)
        
        print(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        
        print(f"Loaded checkpoint from {path}")


def quick_train_demo():
    """快速训练演示"""
    print("=" * 60)
    print("SoulEntity 语言训练演示")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 配置
    config = TrainingConfig(
        dim_q=64,
        dim_u=64,
        dim_z=16,
        dim_embed=128,
        vocab_size=3000,
        batch_size=8,
        learning_rate=1e-3,
        num_epochs=1,
        max_seq_len=64,
        device=device,
    )
    
    # 创建训练器
    trainer = LanguageTrainer(config)
    
    # 构建分词器 (使用示例文本)
    sample_texts = [
        "数学是研究数量、结构以及空间等概念及其变化的一门学科。",
        "人工智能是计算机科学的一个分支，致力于创造智能机器。",
        "深度学习是机器学习的一种方法，使用神经网络进行学习。",
        "自然语言处理是人工智能的一个重要领域。",
    ] * 100  # 复制以增加数据量
    
    trainer.build_tokenizer(sample_texts)
    
    # 创建简单数据集
    class SimpleDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len):
            self.samples = []
            for text in texts:
                ids = tokenizer.encode(text)
                if len(ids) > max_len:
                    ids = ids[:max_len]
                if len(ids) > 2:
                    self.samples.append(ids)
                    
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            ids = self.samples[idx]
            input_ids = ids[:-1]
            target_ids = ids[1:]
            
            # Padding
            max_len = 64
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [0] * pad_len
                target_ids = target_ids + [0] * pad_len
                
            attention_mask = [1 if i != 0 else 0 for i in input_ids]
            
            return {
                'input_ids': torch.tensor(input_ids),
                'target_ids': torch.tensor(target_ids),
                'attention_mask': torch.tensor(attention_mask),
            }
    
    dataset = SimpleDataset(sample_texts, trainer.tokenizer, 64)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 训练几步
    print("\n开始训练...")
    trainer.model.language_port.set_tokenizer(trainer.tokenizer)
    
    for i, batch in enumerate(loader):
        if i >= 10:
            break
        metrics = trainer.train_step(batch)
        print(f"Step {i+1}: loss={metrics['loss']:.4f}, ppl={metrics['ppl']:.2f}")
        
    print("\n✓ 语言训练演示完成")


if __name__ == "__main__":
    quick_train_demo()

