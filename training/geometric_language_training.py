"""
符合理论基础-7的完整几何语言训练框架

根据几何基础.md，训练必须包含：

1. 双曲度量 (第二节)
   - 层级结构偏置
   - 相似性+层级性表达

2. 图册一致性 (第三节)  
   - 多片结构
   - 过渡一致性约束

3. 联络正交性 (第五节)
   - 平行移动保体积
   - 跨图一致性

训练目标:
F_total = F_variational + λ_atlas·L_atlas + λ_conn·L_conn + λ_hyp·L_hyp

其中:
- F_variational = β·KL + γ·E_pred (变分自由能)
- L_atlas = 图册过渡一致性损失
- L_conn = 联络正交性损失
- L_hyp = 双曲几何正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import math

from he_core.soul_entity import SoulEntity, create_soul_entity
from he_core.language_interface import SimpleTokenizer, LanguagePort
from he_core.state import ContactState
from he_core.hyperbolic import HyperbolicEmbedding, HyperbolicRegularizer, HyperbolicIntegrator, LorentzModel


@dataclass
class GeometricTrainingConfig:
    """
    几何训练配置
    
    包含完整的几何约束权重（根据几何基础.md）
    """
    # 模型维度
    dim_q: int = 64
    dim_embed: int = 256
    vocab_size: int = 10000
    
    # === A3: 变分自由能权重 ===
    beta_kl: float = 0.01       # KL正则权重
    gamma_pred: float = 1.0     # 预测误差权重
    
    # === 几何约束权重 (理论基础-7核心) ===
    lambda_atlas: float = 0.1    # 图册一致性损失权重
    lambda_conn: float = 0.1     # 联络正交性损失权重（增大以使L2生成元有效）
    lambda_hyp: float = 0.01     # 双曲层级正则化权重（减小，因为双曲度量已集成到动能）
    
    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 16
    max_seq_len: int = 128
    dt: float = 0.1
    num_evolution_steps: int = 1
    
    # 双曲几何参数
    hyperbolic_model: str = 'poincare'  # 'poincare' or 'lorentz'
    hyperbolic_c: float = 1.0  # 曲率参数
    
    # 设备
    device: str = 'cuda'


class GeometricLanguageTrainer(nn.Module):
    """
    完整几何语言训练器
    
    严格遵循理论基础-7/几何基础.md：
    
    1. 双曲度量 (第二节)
       "在Q上选取负曲率（双曲型）黎曼度量 g_H"
       
    2. 图册一致性 (第三节)
       "在重叠区 U_i ∩ U_j 上必须满足过渡一致性"
       
    3. 联络正交性 (第五节)
       "Gauge Equivariant...用以保证对局部gauge选择的等变性与一致性"
    """
    
    def __init__(self, config: GeometricTrainingConfig, tokenizer: SimpleTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # === 核心：SoulEntity ===
        entity_config = {
            'dim_q': config.dim_q,
            'dim_u': config.dim_q,
            'dim_z': 16,
            'num_charts': 4,
            'beta_kl': config.beta_kl,
            'gamma_pred': config.gamma_pred,
            'stiffness': 0.1,
            'contact_stiffness': 0.1,
        }
        self.entity = create_soul_entity(entity_config)
        
        # === 语言端口 ===
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
        self.entity.add_interface('language', config.dim_q)
        
        # === 双曲几何模块 (第二节) ===
        # 使用Lorentz模型避免边界问题
        self.hyperbolic_embedding = HyperbolicEmbedding(
            dim=config.dim_q,
            model='lorentz',  # 始终使用Lorentz避免边界
            c=config.hyperbolic_c
        )
        self.hyperbolic_regularizer = HyperbolicRegularizer(
            dim=config.dim_q,
            model='lorentz',  # 始终使用Lorentz避免边界
            c=config.hyperbolic_c
        )
        
        # === 群积分器 (基于最小族.md) ===
        self.hyperbolic_integrator = HyperbolicIntegrator(
            dim=config.dim_q,
            c=config.hyperbolic_c
        )
        
        # 用于收集训练过程中的几何诊断信息
        self._geometric_diagnostics = {}
        
    def reset_entity(self, batch_size: int):
        """重置实体状态"""
        self.entity.reset(batch_size, self.config.device)
        
    def encode_tokens(self, 
                     token_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码tokens为几何输入u"""
        return self.language_port.encoder.encode_pooled(token_ids, attention_mask)
    
    def decode_state(self, 
                     state_q: torch.Tensor,
                     prev_tokens: torch.Tensor) -> torch.Tensor:
        """从几何状态q解码为token logits"""
        return self.language_port.decoder(state_q, prev_tokens)
    
    def compute_prediction_error(self,
                                 logits: torch.Tensor,
                                 target_ids: torch.Tensor,
                                 attention_mask: torch.Tensor) -> torch.Tensor:
        """计算预测误差（作为F的γ分量）"""
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_targets.view(-1),
            ignore_index=self.tokenizer.pad_id,
            reduction='none'
        )
        loss = loss.view(shift_targets.shape)
        E_pred = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
        
        return E_pred
    
    # ========================
    #    几何损失计算
    # ========================
    
    def compute_atlas_consistency_loss(self) -> torch.Tensor:
        """
        图册一致性损失 (几何基础 第三节)
        
        根据文档：
        "在重叠区 U_i ∩ U_j 上必须满足过渡一致性"
        
        实现：
        L_atlas = Σ_{i,j} ||φ_{ij}(x_i) - x_j||²
        """
        atlas = self.entity.atlas
        total_loss = torch.tensor(0.0, device=self.config.device)
        count = 0
        
        # 遍历所有图对
        for i in range(atlas.num_charts):
            for j in range(atlas.num_charts):
                if i != j:
                    # 检查是否存在过渡映射
                    key = f"{i}_{j}"
                    if key in atlas.transitions:
                        loss_ij = atlas.compute_consistency_loss(i, j)
                        total_loss = total_loss + loss_ij
                        count += 1
        
        # 如果没有过渡映射，创建一些基本的
        if count == 0:
            # 初始化一些过渡映射
            for i in range(min(atlas.num_charts, 2)):
                for j in range(min(atlas.num_charts, 2)):
                    if i != j:
                        atlas.add_transition(i, j)
            
            # 重新计算
            for i in range(min(atlas.num_charts, 2)):
                for j in range(min(atlas.num_charts, 2)):
                    if i != j:
                        key = f"{i}_{j}"
                        if key in atlas.transitions:
                            loss_ij = atlas.compute_consistency_loss(i, j)
                            total_loss = total_loss + loss_ij
                            count += 1
        
        return total_loss / max(count, 1)
    
    def compute_connection_orthogonality_loss(self) -> torch.Tensor:
        """
        联络正交性损失 (几何基础 第五节)
        
        根据文档：
        "Gauge Equivariant...把'沿边平行移动特征'作为消息传递步骤的一部分，
         用以保证对局部gauge选择的等变性与一致性"
        
        实现：
        L_conn = ||T^T @ T - I||² (确保平移矩阵正交)
        """
        connection = self.entity.connection
        q = self.entity.state.q
        
        return connection.orthogonality_loss(q)
    
    def compute_hyperbolic_regularization(self, q: torch.Tensor) -> torch.Tensor:
        """
        双曲层级正则化 (几何基础 第二节)
        
        理论背景：
        - 双曲度量已集成到动能项 K(p,q) = 1/2 λ(q)^(-2) ||p||²
        - 这提供了自然的向心偏置（远离原点的状态有更小的有效动能）
        - 此正则化是辅助性的，鼓励状态形成有意义的层级结构
        
        根据几何基础.md：
        "层级结构在双曲空间中可以以较低维度、较小失真同时表达'相似性 + 层级性'"
        
        实现：
        L_hyp = 层级多样性损失（鼓励不同样本占据不同层级）
        
        注意：由于双曲度量已在动力学中起作用，此正则化权重应较小
        """
        if q.shape[0] <= 1:
            return torch.tensor(0.0, device=q.device)
        
        # 使用欧氏范数作为"层级深度"的代理
        # （由于动能中使用了 λ(q)^(-2)，范数大的状态天然有更小的有效动能）
        q_norms = q.norm(dim=-1)  # (batch,)
        
        # 层级多样性：不同样本应在不同"层级"（范数）
        # 使用标准差作为多样性度量
        norm_std = q_norms.std()
        norm_mean = q_norms.mean()
        
        # 目标：
        # 1. 范数应有适度的方差（不全collapse）
        # 2. 范数均值应在合理范围（不太小也不太大）
        target_std = 1.0
        target_mean = 2.0
        
        std_loss = (norm_std - target_std).abs()
        mean_loss = (norm_mean - target_mean).abs()
        
        return std_loss + 0.1 * mean_loss
    
    def compute_geometric_loss(self, q: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算完整的几何损失
        
        L_geo = λ_atlas · L_atlas + λ_conn · L_conn + λ_hyp · L_hyp
        
        返回：
            total_loss: 总几何损失
            diagnostics: 各分量值
        """
        # 图册一致性
        L_atlas = self.compute_atlas_consistency_loss()
        
        # 联络正交性
        L_conn = self.compute_connection_orthogonality_loss()
        
        # 双曲正则化
        L_hyp = self.compute_hyperbolic_regularization(q)
        
        # 加权求和
        L_geo = (
            self.config.lambda_atlas * L_atlas +
            self.config.lambda_conn * L_conn +
            self.config.lambda_hyp * L_hyp
        )
        
        diagnostics = {
            'L_atlas': L_atlas.item() if isinstance(L_atlas, torch.Tensor) else L_atlas,
            'L_conn': L_conn.item() if isinstance(L_conn, torch.Tensor) else L_conn,
            'L_hyp': L_hyp.item() if isinstance(L_hyp, torch.Tensor) else L_hyp,
            'L_geo_total': L_geo.item() if isinstance(L_geo, torch.Tensor) else L_geo,
        }
        
        return L_geo, diagnostics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        单步训练（包含完整几何约束）
        
        训练目标:
        F_total = F_variational + L_geometric
                = (β·KL + γ·E_pred) + (λ_atlas·L_atlas + λ_conn·L_conn + λ_hyp·L_hyp)
        
        这是理论基础-7的完整实现：
        - A3: 变分自由能 F_variational
        - 几何基础第二节: 双曲正则化 L_hyp  
        - 几何基础第三节: 图册一致性 L_atlas
        - 几何基础第五节: 联络正交性 L_conn
        """
        token_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size = token_ids.shape[0]
        
        # 重置实体状态
        self.reset_entity(batch_size)
        
        # === Step 1: 语言感知端 ===
        u = self.encode_tokens(token_ids, attention_mask)
        
        # === Step 2: 端口耦合驱动SoulEntity演化 ===
        accumulated_states = []
        for step in range(self.config.num_evolution_steps):
            result = self.entity.step(
                {'language': u},
                dt=self.config.dt
            )
            accumulated_states.append(result['state_flat'].clone())
        
        current_state = self.entity.state
        q_final = current_state.q
        
        # === Step 3: 语言行动端 ===
        logits = self.decode_state(q_final, token_ids)
        
        # === Step 4: 计算预测误差 ===
        E_pred = self.compute_prediction_error(logits, token_ids, attention_mask)
        
        # === Step 5: 计算变分自由能 ===
        F_variational = self.entity.compute_free_energy(
            current_state,
            prediction_error=E_pred
        )
        
        # === Step 6: 计算几何损失 (核心新增) ===
        L_geo, geo_diagnostics = self.compute_geometric_loss(q_final)
        
        # === Step 7: 总损失 ===
        F_total = F_variational + L_geo
        
        # === 诊断指标 ===
        with torch.no_grad():
            ppl = torch.exp(E_pred)
            q_norm = q_final.norm().item()
            p_norm = current_state.p.norm().item()
            s_val = current_state.s.mean().item()
            
            # 双曲空间诊断
            h = self.hyperbolic_embedding(q_final)
            h_norm = h.norm(dim=-1).mean().item()
            
            # 图册诊断
            chart_weights = self.entity.atlas.router(q_final)
            chart_entropy = -(chart_weights * torch.log(chart_weights + 1e-8)).sum(dim=-1).mean().item()
        
        return {
            'loss': F_total,
            'free_energy': F_variational.detach(),
            'geometric_loss': L_geo.detach(),
            'prediction_error': E_pred.detach(),
            'perplexity': ppl,
            'q_norm': q_norm,
            'p_norm': p_norm,
            's_value': s_val,
            'hyperbolic_norm': h_norm,
            'chart_entropy': chart_entropy,
            **geo_diagnostics,
        }
    
    @torch.no_grad()
    def generate(self,
                 prompt: str = "",
                 max_len: int = 50,
                 temperature: float = 1.0) -> str:
        """从SoulEntity状态生成文本"""
        device = self.config.device
        self.reset_entity(1)
        
        if prompt:
            ids = self.tokenizer.encode(prompt)
            token_ids = torch.tensor([ids], device=device)
            u = self.encode_tokens(token_ids)
            for _ in range(self.config.num_evolution_steps):
                self.entity.step({'language': u}, dt=self.config.dt)
        
        q = self.entity.state.q
        generated = self.language_port.decoder.generate(
            q,
            max_len=max_len,
            temperature=temperature,
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id
        )
        
        return self.tokenizer.decode(generated[0].cpu().tolist())
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """评估模型"""
        self.eval()
        
        totals = {
            'F': 0.0, 'E_pred': 0.0, 'L_geo': 0.0,
            'L_atlas': 0.0, 'L_conn': 0.0, 'L_hyp': 0.0
        }
        total_samples = 0
        
        for batch in dataloader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            result = self.train_step(batch)
            
            batch_size = batch['input_ids'].shape[0]
            totals['F'] += result['free_energy'].item() * batch_size
            totals['E_pred'] += result['prediction_error'].item() * batch_size
            totals['L_geo'] += result['geometric_loss'].item() * batch_size
            totals['L_atlas'] += result['L_atlas'] * batch_size
            totals['L_conn'] += result['L_conn'] * batch_size
            totals['L_hyp'] += result['L_hyp'] * batch_size
            total_samples += batch_size
        
        self.train()
        
        return {
            'avg_free_energy': totals['F'] / total_samples,
            'avg_prediction_error': totals['E_pred'] / total_samples,
            'avg_perplexity': math.exp(totals['E_pred'] / total_samples),
            'avg_geometric_loss': totals['L_geo'] / total_samples,
            'avg_L_atlas': totals['L_atlas'] / total_samples,
            'avg_L_conn': totals['L_conn'] / total_samples,
            'avg_L_hyp': totals['L_hyp'] / total_samples,
        }
    
    def get_geometric_diagnostics(self) -> Dict[str, Any]:
        """获取几何诊断信息"""
        q = self.entity.state.q
        
        # 双曲嵌入分析
        h = self.hyperbolic_embedding(q)
        origin = torch.zeros_like(h)
        distances = self.hyperbolic_embedding.hyperbolic_distance(origin, h)
        
        # 图册激活分析
        chart_weights = self.entity.atlas.router(q)
        active_charts = (chart_weights > 0.1).sum(dim=-1).float().mean()
        
        # 联络分析
        curvature = self.entity.connection.get_curvature_proxy(q)
        
        return {
            'hyperbolic_distances': distances.detach().cpu(),
            'chart_weights': chart_weights.detach().cpu(),
            'active_charts': active_charts.item(),
            'curvature_proxy': curvature.mean().item(),
        }


# ========================
#    测试
# ========================

if __name__ == "__main__":
    print("几何语言训练框架测试\n")
    
    # 创建简单tokenizer用于测试
    test_texts = [
        "这是一个测试文本",
        "双曲几何用于层级表示",
        "图册一致性确保跨技能迁移",
        "联络正交性保证体积守恒",
    ]
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(test_texts, min_freq=1)
    
    # 配置
    config = GeometricTrainingConfig(
        dim_q=32,
        dim_embed=64,
        vocab_size=len(tokenizer),
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    # 创建训练器
    trainer = GeometricLanguageTrainer(config, tokenizer)
    trainer = trainer.to(config.device)
    
    print(f"设备: {config.device}")
    print(f"词表大小: {config.vocab_size}")
    print(f"模型参数: {sum(p.numel() for p in trainer.parameters()):,}")
    
    # 测试前向传播
    print("\n=== 前向传播测试 ===")
    trainer.reset_entity(2)
    
    # 模拟输入
    ids = tokenizer.encode(test_texts[0])
    token_ids = torch.tensor([ids, ids], device=config.device)
    attention_mask = torch.ones_like(token_ids)
    
    batch = {
        'input_ids': token_ids,
        'attention_mask': attention_mask,
    }
    
    result = trainer.train_step(batch)
    
    print(f"总损失: {result['loss'].item():.4f}")
    print(f"变分自由能 F: {result['free_energy'].item():.4f}")
    print(f"几何损失 L_geo: {result['geometric_loss'].item():.4f}")
    print(f"  - 图册一致性 L_atlas: {result['L_atlas']:.6f}")
    print(f"  - 联络正交性 L_conn: {result['L_conn']:.6f}")
    print(f"  - 双曲正则化 L_hyp: {result['L_hyp']:.6f}")
    print(f"预测误差 E_pred: {result['prediction_error'].item():.4f}")
    print(f"困惑度 PPL: {result['perplexity'].item():.2f}")
    print(f"双曲范数: {result['hyperbolic_norm']:.4f}")
    print(f"图册熵: {result['chart_entropy']:.4f}")
    
    # 测试几何诊断
    print("\n=== 几何诊断 ===")
    diagnostics = trainer.get_geometric_diagnostics()
    print(f"活跃图册数: {diagnostics['active_charts']:.2f}")
    print(f"曲率代理: {diagnostics['curvature_proxy']:.6f}")
    print(f"双曲距离均值: {diagnostics['hyperbolic_distances'].mean().item():.4f}")
    
    # 测试梯度
    print("\n=== 梯度测试 ===")
    result['loss'].backward()
    
    grad_norms = {}
    for name, param in trainer.named_parameters():
        if param.grad is not None:
            grad_norms[name.split('.')[0]] = param.grad.norm().item()
    
    print("各模块梯度范数:")
    unique_modules = set()
    for name, param in trainer.named_parameters():
        if param.grad is not None:
            module = name.split('.')[0]
            if module not in unique_modules:
                unique_modules.add(module)
                print(f"  {module}: {param.grad.norm().item():.6f}")
    
    print("\n✓ 几何语言训练框架测试通过")

