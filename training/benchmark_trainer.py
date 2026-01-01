#!/usr/bin/env python3
"""
可量化训练评估框架

实现temp-01.md中描述的6阶段评估方案：
- Phase 0: 测量基座（步时、tokens/s、VRAM、梯度范数）
- Phase 1: 算力估计微基准（B、L、evolution_steps扫描）
- Phase 2: dim_q能力曲线（欧氏vs双曲对照）
- Phase 3: 几何约束权重消融

产出：CSV/JSONL日志，包含所有可观测量
"""

import os
import sys
import json
import time
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from he_core.soul_entity import create_soul_entity
from he_core.state import ContactState
from he_core.atlas import AtlasRouter
from he_core.connection import Connection
from he_core.language_interface import SimpleTokenizer, TokenEncoder, StateDecoder


# ============================================================
#                      配置
# ============================================================

@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    
    # === 模型维度 ===
    dim_q: int = 64
    dim_z: int = 16
    dim_u: int = 128
    dim_embed: int = 256  # 默认 4 * dim_q
    vocab_size: int = 10000
    num_charts: int = 8
    
    # === 几何参数 ===
    hyperbolic_c: float = 1.0
    stiffness: float = 0.1
    contact_stiffness: float = 0.01
    dt: float = 0.01
    
    # === 训练参数 ===
    batch_size: int = 8
    max_seq_len: int = 128
    evolution_steps: int = 5
    learning_rate: float = 1e-4
    
    # === 几何损失权重 ===
    lambda_atlas: float = 0.1
    lambda_conn: float = 0.1
    lambda_hyp: float = 0.01
    
    # === 实验控制 ===
    num_steps: int = 200  # 每个配置跑的步数
    warmup_steps: int = 10  # 预热步数（不计入统计）
    log_every: int = 10
    
    # === 硬件 ===
    use_amp: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'BenchmarkConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================
#                      数据集
# ============================================================

class SyntheticDataset(IterableDataset):
    """合成数据集（用于基准测试）"""
    
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 100000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __iter__(self):
        for _ in range(self.num_samples):
            yield torch.randint(0, self.vocab_size, (self.seq_len + 1,))


class WikiStreamDataset(IterableDataset):
    """维基流式数据集"""
    
    def __init__(self, data_path: str, tokenizer: SimpleTokenizer, 
                 seq_len: int = 128, max_docs: int = -1):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_docs = max_docs
        
    def __iter__(self):
        doc_count = 0
        buffer = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if self.max_docs > 0 and doc_count >= self.max_docs:
                    break
                try:
                    obj = json.loads(line.strip())
                    text = obj.get('text', '')
                    if len(text) < 50:
                        continue
                    tokens = self.tokenizer.encode(text, add_special=False)
                    buffer.extend(tokens)
                    doc_count += 1
                    
                    while len(buffer) >= self.seq_len + 1:
                        yield torch.tensor(buffer[:self.seq_len + 1], dtype=torch.long)
                        buffer = buffer[self.seq_len // 2:]
                except json.JSONDecodeError:
                    continue


# ============================================================
#                      模型
# ============================================================

class BenchmarkModel(nn.Module):
    """
    基准测试模型
    
    支持欧氏/双曲模式切换
    """
    
    def __init__(self, config: BenchmarkConfig, euclidean_mode: bool = False):
        super().__init__()
        self.config = config
        self.euclidean_mode = euclidean_mode
        
        # === 核心：SoulEntity ===
        self.entity = create_soul_entity({
            'dim_q': config.dim_q,
            'dim_z': config.dim_z,
            'dim_u': config.dim_u,
            'num_charts': config.num_charts,
            'hyperbolic_c': 1e-6 if euclidean_mode else config.hyperbolic_c,  # 近欧氏
            'stiffness': config.stiffness,
            'contact_stiffness': config.contact_stiffness,
            'device': config.device,
        })
        
        # === 语言端口 ===
        self.encoder = TokenEncoder(
            vocab_size=config.vocab_size,
            dim_embed=config.dim_embed,
            dim_u=config.dim_u,
        )
        
        self.decoder = StateDecoder(
            vocab_size=config.vocab_size,
            dim_q=config.dim_q,
            dim_embed=config.dim_embed,
        )
        
        # === 几何模块 ===
        self.atlas_router = AtlasRouter(
            dim_q=config.dim_q,
            num_charts=config.num_charts,
        )
        
        self.connection = Connection(dim_q=config.dim_q)
        
    def get_param_breakdown(self) -> Dict[str, int]:
        """获取各模块参数量"""
        breakdown = {}
        
        # 端口参数
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        breakdown['encoder'] = encoder_params
        breakdown['decoder'] = decoder_params
        breakdown['port_total'] = encoder_params + decoder_params
        
        # 主体参数
        entity_params = sum(p.numel() for p in self.entity.parameters())
        breakdown['entity'] = entity_params
        
        # 几何模块
        atlas_params = sum(p.numel() for p in self.atlas_router.parameters())
        conn_params = sum(p.numel() for p in self.connection.parameters())
        breakdown['atlas'] = atlas_params
        breakdown['connection'] = conn_params
        breakdown['geometric_total'] = atlas_params + conn_params
        
        # 总计
        breakdown['total'] = sum(p.numel() for p in self.parameters())
        
        return breakdown
    
    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # 1. 编码
        u_seq = self.encoder(tokens)  # [batch, seq_len, dim_u]
        
        # 2. 初始化状态
        q = torch.randn(batch_size, self.config.dim_q, device=device) * 0.1
        p = torch.randn(batch_size, self.config.dim_q, device=device) * 0.1
        s = torch.zeros(batch_size, 1, device=device)
        
        z = self.entity.z.expand(batch_size, -1)
        
        # 3. 多步演化
        q_trajectory = []
        F_values = []
        
        for step in range(self.config.evolution_steps):
            # 端口耦合
            u_t = u_seq[:, min(step, seq_len - 1), :self.config.dim_q]
            
            # 计算F
            qz = torch.cat([q, z], dim=-1)
            V = 0.5 * self.config.stiffness * (q ** 2).sum(dim=-1, keepdim=True)
            V = V + self.entity.net_V(qz)
            K = 0.5 * (p ** 2).sum(dim=-1, keepdim=True)
            Phi = 0.5 * self.config.contact_stiffness * (s ** 2)
            F = K + V + Phi
            F_values.append(F)
            
            # 演化
            dt = self.config.dt
            grad_V = self.config.stiffness * q
            
            p_new = p - dt * grad_V + dt * 0.1 * u_t
            p_new = p_new * 0.99  # 耗散
            q_new = q + dt * p_new
            
            q_trajectory.append(q_new)
            q = q_new.detach()
            p = p_new.detach()
        
        # 4. 解码
        q_final = q_trajectory[-1] if q_trajectory else q
        logits = self.decoder(q_final)
        
        # 扩展到序列长度
        if logits.dim() == 2:
            logits = logits.unsqueeze(1).expand(-1, seq_len - 1, -1)
        
        # 5. 几何诊断
        chart_weights = self.atlas_router(q_final)
        
        # 双曲距离（Lorentz模型）
        if not self.euclidean_mode:
            c = self.config.hyperbolic_c
            q_norm_sq = (q_final ** 2).sum(dim=-1)
            hyp_dist = torch.acosh(1 + 2 * c * q_norm_sq / (1 - c * q_norm_sq).clamp(min=1e-6))
        else:
            hyp_dist = (q_final ** 2).sum(dim=-1).sqrt()
        
        return {
            'logits': logits,
            'F_values': F_values,
            'q_final': q_final,
            'chart_weights': chart_weights,
            'hyp_dist': hyp_dist,
            'q_norm': q_final.norm(dim=-1).mean(),
            'p_norm': p.norm(dim=-1).mean(),
        }


# ============================================================
#                      损失计算
# ============================================================

class BenchmarkLoss(nn.Module):
    """基准测试损失"""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.config = config
        
    def forward(self, outputs: Dict, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算损失"""
        batch_size, seq_len = tokens.shape
        
        losses = {}
        
        # 1. 语言预测损失
        logits = outputs['logits']
        targets = tokens[:, 1:seq_len]
        
        if logits.shape[1] != targets.shape[1]:
            min_len = min(logits.shape[1], targets.shape[1])
            logits = logits[:, :min_len]
            targets = targets[:, :min_len]
        
        E_pred = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=0
        )
        losses['E_pred'] = E_pred
        losses['PPL'] = torch.exp(E_pred.detach())
        
        # 2. Lyapunov损失
        F_values = outputs['F_values']
        if len(F_values) >= 2:
            lyap_violations = []
            for i in range(1, len(F_values)):
                diff = F_values[i] - F_values[i-1]
                violation = F.relu(diff).mean()
                lyap_violations.append(violation)
            L_lyap = sum(lyap_violations) / len(lyap_violations)
        else:
            L_lyap = torch.tensor(0.0, device=tokens.device)
        losses['L_lyap'] = L_lyap
        
        # 3. 图册损失（熵正则化）
        chart_weights = outputs['chart_weights']
        chart_entropy = -(chart_weights * (chart_weights + 1e-8).log()).sum(dim=-1).mean()
        L_atlas = -chart_entropy  # 鼓励使用多个图册
        losses['L_atlas'] = L_atlas
        losses['chart_entropy'] = chart_entropy
        
        # 4. 联络正交性损失
        # 简化：鼓励q_final具有正交结构
        q_final = outputs['q_final']
        L_conn = (q_final @ q_final.T - torch.eye(batch_size, device=tokens.device)).abs().mean()
        losses['L_conn'] = L_conn
        
        # 5. 双曲正则化
        hyp_dist = outputs['hyp_dist']
        L_hyp = F.relu(hyp_dist - 5.0).mean()  # 防止距离过大
        losses['L_hyp'] = L_hyp
        
        # 6. 总损失
        total = E_pred
        total = total + self.config.lambda_atlas * L_atlas
        total = total + self.config.lambda_conn * L_conn
        total = total + self.config.lambda_hyp * L_hyp
        losses['total'] = total
        
        # 诊断量
        losses['q_norm'] = outputs['q_norm']
        losses['p_norm'] = outputs['p_norm']
        losses['hyp_dist_mean'] = hyp_dist.mean()
        
        return losses


# ============================================================
#                      基准测试运行器
# ============================================================

class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, output_dir: str = 'benchmark_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.results = []
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('benchmark')
        logger.setLevel(logging.INFO)
        
        # 控制台
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(ch)
        
        # 文件
        fh = logging.FileHandler(self.output_dir / 'benchmark.log')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        
        return logger
    
    def run_single_config(self, 
                          config: BenchmarkConfig,
                          data_path: Optional[str] = None,
                          euclidean_mode: bool = False,
                          config_name: str = 'default') -> Dict:
        """运行单个配置的基准测试"""
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"配置: {config_name}")
        self.logger.info(f"  dim_q={config.dim_q}, B={config.batch_size}, L={config.max_seq_len}")
        self.logger.info(f"  evolution_steps={config.evolution_steps}")
        self.logger.info(f"  euclidean_mode={euclidean_mode}")
        self.logger.info(f"{'='*60}")
        
        device = torch.device(config.device)
        
        # 创建模型
        model = BenchmarkModel(config, euclidean_mode=euclidean_mode)
        model = model.to(device)
        loss_fn = BenchmarkLoss(config)
        
        # 参数分解
        param_breakdown = model.get_param_breakdown()
        self.logger.info(f"参数分布:")
        for name, count in param_breakdown.items():
            self.logger.info(f"  {name}: {count:,}")
        
        # 创建数据
        if data_path and os.path.exists(data_path):
            tokenizer = SimpleTokenizer()
            tokenizer.build_vocab_from_file(data_path, max_vocab=config.vocab_size)
            dataset = WikiStreamDataset(data_path, tokenizer, config.max_seq_len)
        else:
            dataset = SyntheticDataset(config.vocab_size, config.max_seq_len)
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=2,
            pin_memory=True,
        )
        
        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scaler = GradScaler() if config.use_amp else None
        
        # 统计量
        step_times = []
        tokens_per_sec_list = []
        peak_vram_list = []
        loss_history = {
            'total': [], 'E_pred': [], 'PPL': [],
            'L_lyap': [], 'L_atlas': [], 'L_conn': [], 'L_hyp': [],
            'q_norm': [], 'p_norm': [], 'hyp_dist_mean': [], 'chart_entropy': []
        }
        grad_norms = {'encoder': [], 'decoder': [], 'entity': [], 'atlas': [], 'connection': []}
        
        # 重置VRAM统计
        torch.cuda.reset_peak_memory_stats()
        
        # 训练循环
        data_iter = iter(dataloader)
        for step in range(config.num_steps + config.warmup_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            batch = batch.to(device)
            
            # 同步计时开始
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Forward + Backward
            optimizer.zero_grad()
            
            if config.use_amp:
                with autocast():
                    outputs = model(batch)
                    losses = loss_fn(outputs, batch)
                    loss = losses['total']
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch)
                losses = loss_fn(outputs, batch)
                loss = losses['total']
                loss.backward()
                optimizer.step()
            
            # 同步计时结束
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            
            # 跳过预热
            if step < config.warmup_steps:
                continue
            
            actual_step = step - config.warmup_steps
            
            # 记录步时和吞吐
            step_times.append(elapsed * 1000)  # ms
            tokens = config.batch_size * (config.max_seq_len - 1)
            tokens_per_sec_list.append(tokens / elapsed)
            
            # 记录VRAM
            if actual_step % config.log_every == 0:
                peak_vram = torch.cuda.max_memory_allocated() / 1e9
                peak_vram_list.append(peak_vram)
                torch.cuda.reset_peak_memory_stats()
            
            # 记录损失
            for key in loss_history:
                if key in losses:
                    loss_history[key].append(losses[key].item() if torch.is_tensor(losses[key]) else losses[key])
            
            # 记录梯度范数
            if actual_step % config.log_every == 0:
                for name, module in [
                    ('encoder', model.encoder),
                    ('decoder', model.decoder),
                    ('entity', model.entity),
                    ('atlas', model.atlas_router),
                    ('connection', model.connection),
                ]:
                    total_norm = 0.0
                    for p in module.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    grad_norms[name].append(total_norm ** 0.5)
            
            # 日志
            if actual_step % config.log_every == 0:
                self.logger.info(
                    f"Step {actual_step:4d} | "
                    f"Loss: {losses['total'].item():.4f} | "
                    f"PPL: {losses['PPL'].item():.1f} | "
                    f"ms/step: {elapsed*1000:.1f} | "
                    f"tok/s: {tokens/elapsed:.0f} | "
                    f"VRAM: {peak_vram:.2f}GB"
                )
        
        # 汇总结果
        result = {
            'config_name': config_name,
            'config': config.to_dict(),
            'euclidean_mode': euclidean_mode,
            'param_breakdown': param_breakdown,
            
            # 性能指标
            'step_time_ms_mean': sum(step_times) / len(step_times),
            'step_time_ms_std': (sum((t - sum(step_times)/len(step_times))**2 for t in step_times) / len(step_times)) ** 0.5,
            'tokens_per_sec_mean': sum(tokens_per_sec_list) / len(tokens_per_sec_list),
            'peak_vram_gb_max': max(peak_vram_list) if peak_vram_list else 0,
            
            # 损失指标（最后10步平均）
            'final_loss': sum(loss_history['total'][-10:]) / 10 if loss_history['total'] else 0,
            'final_PPL': sum(loss_history['PPL'][-10:]) / 10 if loss_history['PPL'] else 0,
            'final_E_pred': sum(loss_history['E_pred'][-10:]) / 10 if loss_history['E_pred'] else 0,
            'final_L_lyap': sum(loss_history['L_lyap'][-10:]) / 10 if loss_history['L_lyap'] else 0,
            'final_L_atlas': sum(loss_history['L_atlas'][-10:]) / 10 if loss_history['L_atlas'] else 0,
            'final_L_conn': sum(loss_history['L_conn'][-10:]) / 10 if loss_history['L_conn'] else 0,
            'final_L_hyp': sum(loss_history['L_hyp'][-10:]) / 10 if loss_history['L_hyp'] else 0,
            
            # 状态诊断
            'final_q_norm': sum(loss_history['q_norm'][-10:]) / 10 if loss_history['q_norm'] else 0,
            'final_p_norm': sum(loss_history['p_norm'][-10:]) / 10 if loss_history['p_norm'] else 0,
            'final_hyp_dist': sum(loss_history['hyp_dist_mean'][-10:]) / 10 if loss_history['hyp_dist_mean'] else 0,
            'final_chart_entropy': sum(loss_history['chart_entropy'][-10:]) / 10 if loss_history['chart_entropy'] else 0,
            
            # 梯度分布
            'grad_norm_encoder': sum(grad_norms['encoder']) / len(grad_norms['encoder']) if grad_norms['encoder'] else 0,
            'grad_norm_decoder': sum(grad_norms['decoder']) / len(grad_norms['decoder']) if grad_norms['decoder'] else 0,
            'grad_norm_entity': sum(grad_norms['entity']) / len(grad_norms['entity']) if grad_norms['entity'] else 0,
            'grad_norm_atlas': sum(grad_norms['atlas']) / len(grad_norms['atlas']) if grad_norms['atlas'] else 0,
            'grad_norm_connection': sum(grad_norms['connection']) / len(grad_norms['connection']) if grad_norms['connection'] else 0,
        }
        
        self.results.append(result)
        
        # 清理
        del model, optimizer
        torch.cuda.empty_cache()
        
        return result
    
    def run_phase1_compute(self, data_path: Optional[str] = None):
        """Phase 1: 算力估计微基准"""
        self.logger.info("\n" + "="*70)
        self.logger.info("Phase 1: 算力估计微基准")
        self.logger.info("="*70)
        
        base_config = BenchmarkConfig()
        
        # 1A: 扫描 B 和 L
        self.logger.info("\n--- 1A: B × L 扫描 ---")
        for B in [1, 2, 4, 8, 16]:
            for L in [64, 128, 256]:
                try:
                    config = BenchmarkConfig(
                        batch_size=B,
                        max_seq_len=L,
                        num_steps=100,
                    )
                    self.run_single_config(
                        config, 
                        data_path,
                        config_name=f'B{B}_L{L}'
                    )
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        self.logger.warning(f"OOM: B={B}, L={L}")
                        torch.cuda.empty_cache()
                    else:
                        raise
        
        # 1B: 扫描 evolution_steps
        self.logger.info("\n--- 1B: evolution_steps 扫描 ---")
        for steps in [1, 2, 4, 8]:
            config = BenchmarkConfig(
                batch_size=8,
                max_seq_len=128,
                evolution_steps=steps,
                num_steps=100,
            )
            self.run_single_config(
                config,
                data_path,
                config_name=f'evo{steps}'
            )
        
        self._save_results('phase1_compute.json')
    
    def run_phase2_dim_q(self, data_path: Optional[str] = None):
        """Phase 2: dim_q能力曲线"""
        self.logger.info("\n" + "="*70)
        self.logger.info("Phase 2: dim_q能力曲线 (欧氏 vs 双曲)")
        self.logger.info("="*70)
        
        for dim_q in [16, 24, 32, 48, 64]:
            dim_embed = 4 * dim_q  # 比例缩放
            
            # 双曲组
            config = BenchmarkConfig(
                dim_q=dim_q,
                dim_embed=dim_embed,
                num_steps=200,
            )
            self.run_single_config(
                config,
                data_path,
                euclidean_mode=False,
                config_name=f'dim{dim_q}_hyp'
            )
            
            # 欧氏组
            self.run_single_config(
                config,
                data_path,
                euclidean_mode=True,
                config_name=f'dim{dim_q}_euc'
            )
        
        self._save_results('phase2_dim_q.json')
    
    def run_phase3_ablation(self, data_path: Optional[str] = None):
        """Phase 3: 几何约束权重消融"""
        self.logger.info("\n" + "="*70)
        self.logger.info("Phase 3: 几何约束权重消融")
        self.logger.info("="*70)
        
        ablation_configs = [
            ('baseline', 0.0, 0.0, 0.0),
            ('+atlas', 0.1, 0.0, 0.0),
            ('+conn', 0.0, 0.1, 0.0),
            ('+hyp', 0.0, 0.0, 0.01),
            ('full', 0.1, 0.1, 0.01),
        ]
        
        for name, la, lc, lh in ablation_configs:
            config = BenchmarkConfig(
                lambda_atlas=la,
                lambda_conn=lc,
                lambda_hyp=lh,
                num_steps=200,
            )
            self.run_single_config(
                config,
                data_path,
                config_name=name
            )
        
        self._save_results('phase3_ablation.json')
    
    def _save_results(self, filename: str):
        """保存结果"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"结果已保存至: {filepath}")
        
        # 同时保存CSV摘要
        csv_path = filepath.with_suffix('.csv')
        if self.results:
            import csv
            keys = ['config_name', 'step_time_ms_mean', 'tokens_per_sec_mean', 
                    'peak_vram_gb_max', 'final_PPL', 'final_E_pred',
                    'grad_norm_encoder', 'grad_norm_entity']
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
                writer.writeheader()
                for r in self.results:
                    writer.writerow(r)
            self.logger.info(f"CSV摘要已保存至: {csv_path}")


# ============================================================
#                      主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='可量化训练基准测试')
    parser.add_argument('--phase', type=str, choices=['1', '2', '3', 'all'], default='all',
                        help='运行哪个阶段')
    parser.add_argument('--data', type=str, default=None, help='数据路径')
    parser.add_argument('--output', type=str, default='benchmark_results', help='输出目录')
    parser.add_argument('--quick', action='store_true', help='快速模式（减少步数）')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(output_dir=args.output)
    
    if args.phase == '1' or args.phase == 'all':
        runner.run_phase1_compute(args.data)
    
    if args.phase == '2' or args.phase == 'all':
        runner.run_phase2_dim_q(args.data)
    
    if args.phase == '3' or args.phase == 'all':
        runner.run_phase3_ablation(args.data)
    
    print("\n" + "="*70)
    print("基准测试完成!")
    print(f"结果保存至: {args.output}")
    print("="*70)


if __name__ == '__main__':
    main()

