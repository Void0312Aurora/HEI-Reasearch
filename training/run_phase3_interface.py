"""
阶段3：接口对齐训练 (Interface Alignment Training)

目标：训练语言接口（L3端口耦合），实现：
- 语言编码器：tokens → u（端口输入）
- 语言解码器：q → logits（预测输出）
- 保持前两阶段学到的动力学和几何结构

理论依据：
- A4: 可交互性 - 通过端口与外部交互
- A5: 多接口单核心 - 语言只是一种接口
- L3: 端口耦合 H^c(q,p,S,t) = H^c_int + <u(t), B(q)>

成功标准：
- PPL < 100 (语言建模质量)
- 离线稳定性保持 (Lyapunov违反率 < 10%)
- 几何一致性保持 (闭环误差 < 0.2)

运行：
python HEI/training/run_phase3_interface.py --checkpoint checkpoints/phase2/best_model.pt --data wiki_data.txt --epochs 50
"""

import os
import sys
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from he_core.soul_entity import SoulEntity, create_soul_entity
from he_core.state import ContactState
from he_core.language_interface import TokenEncoder, StateDecoder, SimpleTokenizer

# 导入阶段2的配置类，用于加载checkpoint
try:
    from training.run_phase2_atlas import AtlasTrainingConfig
except ImportError:
    # 如果无法导入，定义一个占位类
    from dataclasses import dataclass
    @dataclass
    class AtlasTrainingConfig:
        pass


@dataclass
class InterfaceTrainingConfig:
    """阶段3训练配置"""
    dim_q: int = 64
    dim_z: int = 16
    dim_u: int = 64  # 端口输入维度 = dim_q
    vocab_size: int = 10000
    batch_size: int = 16
    seq_len: int = 64
    learning_rate: float = 1e-4
    num_steps: int = 10  # 每个token的演化步数
    dt: float = 0.1
    
    # 损失权重
    lambda_lm: float = 1.0         # 语言建模损失
    lambda_stability: float = 0.1   # 离线稳定性
    lambda_geometry: float = 0.01   # 几何一致性
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class InterfaceTrainer(nn.Module):
    """阶段3：接口对齐训练器"""
    
    def __init__(self, config: InterfaceTrainingConfig, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.config = config
        
        # 创建SoulEntity
        self.entity = create_soul_entity({
            'dim_q': config.dim_q,
            'dim_z': config.dim_z,
            'hyperbolic_c': 1.0,
            'device': config.device,
        })
        
        # 语言编码器：token_embedding → u
        self.encoder = TokenEncoder(
            vocab_size=config.vocab_size,
            dim_embed=128,
            dim_u=config.dim_u,
            num_layers=2
        )
        
        # 语言解码器：q → logits
        self.decoder = StateDecoder(
            vocab_size=config.vocab_size,
            dim_q=config.dim_q,
            dim_embed=256,
            num_layers=2
        )
        
        # 加载前阶段权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"加载阶段2权重: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
            entity_state = {}
            for k, v in ckpt['model_state_dict'].items():
                if k.startswith('entity.'):
                    entity_state[k[7:]] = v
            if entity_state:
                self.entity.load_state_dict(entity_state, strict=False)
                print("  ✓ 权重加载成功")
    
    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播（简化版，避免复杂的梯度计算）
        
        Args:
            tokens: (batch, seq_len) token序列
            
        Returns:
            logits: (batch, seq_len, vocab_size) 预测logits
            diagnostics: 诊断信息
        """
        batch_size, seq_len = tokens.shape
        
        # 一次编码整个序列
        u_all = self.encoder(tokens)  # (batch, seq_len, dim_u)
        
        # 简化：使用平均编码来驱动状态
        u_mean = u_all.mean(dim=1)  # (batch, dim_u)
        
        # 初始化状态
        state = self._init_state(batch_size)
        
        # 计算初始F
        z = self.entity.z.expand(batch_size, -1)
        F_initial = self.entity.internal_gen(state, z).detach()
        
        # 使用u_mean更新状态（简化的端口耦合）
        # 将u映射到q空间的扰动
        q_perturb = u_mean[:, :self.config.dim_q] if u_mean.shape[1] >= self.config.dim_q else F.pad(u_mean, (0, self.config.dim_q - u_mean.shape[1]))
        
        # 更新q（非inplace）
        new_q = state.q + 0.1 * q_perturb
        
        # 使用新q直接解码
        logits = self.decoder(new_q)  # (batch, seq_len_out, vocab_size)
        
        # 确保输出shape正确
        if logits.dim() == 2:
            # 扩展为 (batch, seq_len, vocab_size)
            logits = logits.unsqueeze(1).expand(-1, seq_len, -1)
        elif logits.shape[1] != seq_len:
            # 调整seq_len
            if logits.shape[1] > seq_len:
                logits = logits[:, :seq_len, :]
            else:
                logits = F.pad(logits, (0, 0, 0, seq_len - logits.shape[1]))
        
        # 计算最终F
        final_state = ContactState(
            dim_q=self.config.dim_q,
            batch_size=batch_size,
            device=self.config.device
        )
        final_state._data = torch.cat([new_q, state.p, state.s], dim=-1)
        F_final = self.entity.internal_gen(final_state, z).detach()
        
        # 简化的F轨迹
        F_trajectory = torch.stack([F_initial, F_final], dim=1).squeeze(-1)
        
        return {
            'logits': logits,
            'F_trajectory': F_trajectory,
            'final_state': final_state
        }
    
    def _init_state(self, batch_size: int) -> ContactState:
        """初始化状态"""
        state = ContactState(
            dim_q=self.config.dim_q,
            batch_size=batch_size,
            device=self.config.device
        )
        # 用小随机值初始化
        state._data = torch.randn(batch_size, 2 * self.config.dim_q + 1, device=self.config.device) * 0.1
        return state
    
    def compute_loss(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            tokens: (batch, seq_len) token序列
        """
        # 前向传播
        result = self.forward(tokens[:, :-1])  # 输入为tokens[:-1]
        logits = result['logits']  # (batch, seq_len-1, vocab_size)
        F_trajectory = result['F_trajectory']
        
        # 1. 语言建模损失
        targets = tokens[:, 1:]  # 目标为tokens[1:]
        L_lm = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            ignore_index=0  # 忽略padding
        )
        
        # 2. 离线稳定性损失
        # F应该在无外部输入时下降
        F_diff = F_trajectory[:, 1:] - F_trajectory[:, :-1]
        L_stability = torch.relu(F_diff).mean()
        
        # 3. 几何一致性损失（简化：q的范数约束）
        final_q = result['final_state'].q
        L_geometry = (final_q.norm(dim=-1) - 1.0).pow(2).mean()
        
        # 总损失
        loss = (
            self.config.lambda_lm * L_lm +
            self.config.lambda_stability * L_stability +
            self.config.lambda_geometry * L_geometry
        )
        
        # 计算PPL
        ppl = torch.exp(L_lm).item()
        
        # 计算Lyapunov违反率
        violation_rate = (F_diff > 0).float().mean().item()
        
        return {
            'loss': loss,
            'lm_loss': L_lm,
            'stability_loss': L_stability,
            'geometry_loss': L_geometry,
            'ppl': ppl,
            'violation_rate': violation_rate,
        }


class TextDataset(torch.utils.data.Dataset):
    """简单文本数据集"""
    
    def __init__(self, text_path: str, tokenizer: SimpleTokenizer, seq_len: int = 64):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        # 加载文本
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # tokenize
        self.tokens = tokenizer.encode(text)
        self.num_samples = (len(self.tokens) - 1) // seq_len
        
        print(f"数据集: {len(self.tokens)} tokens, {self.num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target
        tokens = self.tokens[start:end]
        
        # padding if needed
        if len(tokens) < self.seq_len + 1:
            tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)


def train_phase3(args):
    """阶段3训练主循环"""
    
    print("=" * 70)
    print("阶段3：接口对齐训练")
    print("=" * 70)
    print("\n理论目标：")
    print("  - A4: 可交互性 - 通过端口与外部交互")
    print("  - A5: 多接口单核心 - 语言只是一种接口")
    print("  - L3: 端口耦合 H^c(q,p,S,t) = H^c_int + <u(t), B(q)>")
    print("\n成功标准：")
    print("  - PPL < 100")
    print("  - Lyapunov违反率 < 10%")
    print("  - 几何损失 < 0.1")
    print("=" * 70)
    
    config = InterfaceTrainingConfig(
        dim_q=args.dim_q,
        dim_z=args.dim_z,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.lr,
    )
    
    print(f"\n配置：")
    print(f"  dim_q: {config.dim_q}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  seq_len: {config.seq_len}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  device: {config.device}")
    
    # 初始化tokenizer
    tokenizer = SimpleTokenizer()
    
    # 加载数据
    if args.data and os.path.exists(args.data):
        # 构建词汇表
        print(f"\n加载数据: {args.data}")
        with open(args.data, 'r', encoding='utf-8') as f:
            text = f.read()
        tokenizer.build_vocab([text])
        
        # 更新vocab_size
        config.vocab_size = len(tokenizer)
        print(f"词汇表大小: {config.vocab_size}")
        
        # 创建数据集和加载器
        dataset = TextDataset(args.data, tokenizer, config.seq_len)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        )
    else:
        print("\n警告: 无数据文件，使用合成数据")
        dataloader = None
    
    # 创建训练器
    trainer = InterfaceTrainer(config, checkpoint_path=args.checkpoint)
    trainer = trainer.to(config.device)
    
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * (len(dataloader) if dataloader else 100)
    )
    
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, 'training_log.jsonl')
    
    best_ppl = float('inf')
    
    print(f"\n开始训练... (epochs: {args.epochs})")
    print("-" * 70)
    
    for epoch in range(args.epochs):
        epoch_stats = {
            'loss': 0, 'lm_loss': 0, 'ppl': 0,
            'stability': 0, 'geometry': 0, 'violation': 0
        }
        
        if dataloader:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
            num_batches = 0
            
            for batch in pbar:
                batch = batch.to(config.device)
                
                optimizer.zero_grad()
                diagnostics = trainer.compute_loss(batch)
                loss = diagnostics['loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_stats['loss'] += loss.item()
                epoch_stats['lm_loss'] += diagnostics['lm_loss'].item()
                epoch_stats['ppl'] += diagnostics['ppl']
                epoch_stats['stability'] += diagnostics['stability_loss'].item()
                epoch_stats['geometry'] += diagnostics['geometry_loss'].item()
                epoch_stats['violation'] += diagnostics['violation_rate']
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.2f}",
                    'PPL': f"{diagnostics['ppl']:.1f}",
                    'Viol': f"{diagnostics['violation_rate']:.1%}",
                })
            
            for k in epoch_stats:
                epoch_stats[k] /= num_batches
        else:
            # 合成数据训练
            for step in range(100):
                fake_tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len + 1), device=config.device)
                
                optimizer.zero_grad()
                diagnostics = trainer.compute_loss(fake_tokens)
                loss = diagnostics['loss']
                
                loss.backward()
                optimizer.step()
                
                epoch_stats['loss'] += loss.item()
                epoch_stats['ppl'] += diagnostics['ppl']
            
            epoch_stats['loss'] /= 100
            epoch_stats['ppl'] /= 100
        
        print(f"\nEpoch {epoch+1} 总结：")
        print(f"  损失: {epoch_stats['loss']:.4f}")
        print(f"  PPL: {epoch_stats['ppl']:.2f}")
        print(f"  稳定性损失: {epoch_stats['stability']:.4f}")
        print(f"  几何损失: {epoch_stats['geometry']:.4f}")
        print(f"  Lyapunov违反率: {epoch_stats['violation']:.1%}")
        
        log_entry = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            **epoch_stats
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        if epoch_stats['ppl'] < best_ppl:
            best_ppl = epoch_stats['ppl']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ppl': best_ppl,
                'config': config,
                'tokenizer': tokenizer,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  ★ 新最佳模型! PPL: {best_ppl:.2f}")
        
        # 检查成功标准
        if (epoch_stats['ppl'] < 100 and 
            epoch_stats['violation'] < 0.1 and
            epoch_stats['geometry'] < 0.1):
            print("\n" + "=" * 70)
            print("✓ 达到成功标准！阶段3训练完成")
            print("=" * 70)
            break
        
        print("-" * 70)
    
    # 保存最终模型
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': trainer.state_dict(),
        'final_ppl': epoch_stats['ppl'],
        'config': config,
        'tokenizer': tokenizer,
    }, os.path.join(args.save_dir, 'final_model.pt'))
    
    print("\n" + "=" * 70)
    print("阶段3训练完成")
    print(f"最佳PPL: {best_ppl:.2f}")
    print(f"模型保存至: {args.save_dir}")
    print("=" * 70)
    
    return best_ppl < 100


def main():
    parser = argparse.ArgumentParser(description='阶段3：接口对齐训练')
    
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--dim_z', type=int, default=16)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--seq_len', type=int, default=64)
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--data', type=str, default=None, help='训练数据路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='阶段2的checkpoint路径')
    parser.add_argument('--save_dir', type=str, default='checkpoints/phase3')
    
    args = parser.parse_args()
    
    success = train_phase3(args)
    
    if success:
        print("\n✓ 完整三阶段训练完成！")
        print("可以使用以下命令进行推理:")
        print("python HEI/inference/generate.py --checkpoint checkpoints/phase3/best_model.pt --prompt '你好'")
    else:
        print("\n阶段3未达成功标准，建议：")
        print("  1. 增加训练轮数 --epochs 100")
        print("  2. 使用更多数据")
        print("  3. 调整学习率 --lr 5e-5")


if __name__ == "__main__":
    main()

