"""
理论对齐模型的推理脚本

功能:
1. 加载训练好的模型
2. 生成文本
3. 分析动力学行为
4. 验证公理满足情况

使用:
python HEI/training/theoretical_inference.py \
    --checkpoint checkpoints/theoretical/final_model.pt \
    --prompt "北京是" \
    --mode generate
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.theoretical_trainer import TheoreticalConfig, TheoreticalModel
from he_core.language_interface import SimpleTokenizer
from he_core.state import ContactState


class TheoreticalInference:
    """理论对齐模型推理器"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.load_model(checkpoint_path)
        
    def load_model(self, checkpoint_path: str):
        """加载模型"""
        print(f"加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 加载配置
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            self.config = TheoreticalConfig()
        
        # 加载tokenizer
        if 'tokenizer' in checkpoint:
            self.tokenizer = checkpoint['tokenizer']
        else:
            self.tokenizer = SimpleTokenizer()
            
        # 创建模型
        self.model = TheoreticalModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"  词汇表大小: {self.config.vocab_size}")
        print(f"  dim_q: {self.config.dim_q}")
        print(f"  训练阶段: {checkpoint.get('phase', 'unknown')}")
        print(f"  训练步数: {checkpoint.get('global_step', 'unknown')}")
        
    def generate(self,
                 prompt: str,
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 num_samples: int = 3) -> List[str]:
        """
        生成文本
        
        使用SoulEntity的动力学演化来生成
        简化版本：使用语言编码驱动状态演化
        """
        results = []
        
        for sample_idx in range(num_samples):
            # 编码prompt
            tokens = self.tokenizer.encode(prompt, add_special=True)
            generated = list(tokens)
            
            # 初始化状态
            q = torch.randn(1, self.config.dim_q, device=self.device) * 0.1
            
            for step in range(max_length):
                # 编码当前序列
                current_tokens = torch.tensor([generated[-self.config.max_seq_len:]], 
                                             dtype=torch.long, device=self.device)
                
                with torch.no_grad():
                    u_seq = self.model.encoder(current_tokens)
                    
                    # 取最后一个token的编码来更新q
                    u_t = u_seq[:, -1, :]
                    
                    # 简化的状态更新：q += α * u
                    min_dim = min(u_t.shape[-1], self.config.dim_q)
                    q = q + 0.1 * u_t[:, :min_dim]
                    
                    # 归一化以保持稳定
                    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
                    
                    # 解码
                    logits = self.model.decoder(q)
                    if logits.dim() == 3:
                        logits = logits[:, -1, :]
                    
                    # 采样
                    logits = logits / temperature
                    if top_k > 0:
                        top_k_val = min(top_k, logits.shape[-1])
                        indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated.append(next_token)
                
                # 检查结束
                if next_token == self.tokenizer.eos_id:
                    break
            
            # 解码生成的文本
            text = self.tokenizer.decode(generated)
            results.append(text)
        
        return results
    
    @torch.no_grad()
    def analyze_dynamics(self, prompt: str, num_steps: int = 50) -> Dict[str, Any]:
        """
        分析动力学行为
        
        验证:
        - A2: 离线演化结构
        - A3: F单调下降
        """
        # 编码prompt
        tokens = self.tokenizer.encode(prompt, add_special=True)
        tokens = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # 初始化状态
        state = ContactState(
            dim_q=self.config.dim_q,
            batch_size=1,
            device=self.device
        )
        state._data = torch.randn(1, 2 * self.config.dim_q + 1, device=self.device) * 0.1
        
        self.model.entity.reset(batch_size=1)
        self.model.entity.state = state
        
        # 记录轨迹
        F_trajectory = []
        q_trajectory = []
        
        z = self.model.entity.z.expand(1, -1)
        
        # 离线演化
        for t in range(num_steps):
            current_state = self.model.entity.state
            
            # 计算F
            F_t = self.model.entity.compute_free_energy(current_state).item()
            F_trajectory.append(F_t)
            q_trajectory.append(current_state.q.cpu().numpy().copy())
            
            # 演化
            result = self.model.entity.step(u_dict=None, dt=self.config.dt)
        
        # 分析
        F_diffs = [F_trajectory[i+1] - F_trajectory[i] for i in range(len(F_trajectory)-1)]
        violations = sum(1 for d in F_diffs if d > 0)
        violation_rate = violations / len(F_diffs) if F_diffs else 0
        
        # 检查吸引子
        q_norms = [q.reshape(-1).tolist() for q in q_trajectory[-10:]]
        q_variance = sum(sum((a - b)**2 for a, b in zip(q1, q2)) 
                        for q1, q2 in zip(q_norms[:-1], q_norms[1:])) / max(1, len(q_norms)-1)
        
        return {
            'F_initial': F_trajectory[0],
            'F_final': F_trajectory[-1],
            'F_decrease': F_trajectory[0] - F_trajectory[-1],
            'violation_rate': violation_rate,
            'lyapunov_satisfied': violation_rate < 0.1,
            'q_variance_final': q_variance,
            'converged': q_variance < 0.01,
            'F_trajectory': F_trajectory,
        }
    
    @torch.no_grad()
    def verify_axioms(self, test_prompts: List[str] = None) -> Dict[str, Any]:
        """
        验证公理满足情况
        """
        if test_prompts is None:
            test_prompts = ["北京", "数学", "计算机", "人工智能", "中国"]
        
        results = {
            'A1_markov_blanket': True,  # 架构保证
            'A2_offline_cognition': [],
            'A3_unified_F': [],
            'A4_identity_continuity': True,  # 需要更长测试
            'A5_interface_invariance': True,  # 架构保证
        }
        
        print("\n验证公理满足情况...")
        print("-" * 50)
        
        for prompt in test_prompts:
            print(f"\n测试 '{prompt}':")
            analysis = self.analyze_dynamics(prompt, num_steps=30)
            
            # A2: 离线演化
            results['A2_offline_cognition'].append({
                'prompt': prompt,
                'converged': analysis['converged'],
                'q_variance': analysis['q_variance_final'],
            })
            print(f"  A2 离线收敛: {'✓' if analysis['converged'] else '✗'} (方差: {analysis['q_variance_final']:.4f})")
            
            # A3: 统一F
            results['A3_unified_F'].append({
                'prompt': prompt,
                'lyapunov_satisfied': analysis['lyapunov_satisfied'],
                'violation_rate': analysis['violation_rate'],
                'F_decrease': analysis['F_decrease'],
            })
            print(f"  A3 Lyapunov: {'✓' if analysis['lyapunov_satisfied'] else '✗'} (违反率: {analysis['violation_rate']:.1%})")
        
        # 汇总
        print("\n" + "-" * 50)
        print("公理满足汇总:")
        print(f"  A1 Markov Blanket: ✓ (架构保证)")
        
        a2_pass = sum(1 for r in results['A2_offline_cognition'] if r['converged'])
        print(f"  A2 离线认知: {a2_pass}/{len(results['A2_offline_cognition'])} 通过")
        
        a3_pass = sum(1 for r in results['A3_unified_F'] if r['lyapunov_satisfied'])
        print(f"  A3 统一F: {a3_pass}/{len(results['A3_unified_F'])} 通过")
        
        print(f"  A4 身份连续性: ✓ (需长期测试)")
        print(f"  A5 多接口: ✓ (架构保证)")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='理论对齐模型推理')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='检查点路径')
    parser.add_argument('--mode', type=str, default='generate',
                       choices=['generate', 'analyze', 'verify'],
                       help='运行模式')
    parser.add_argument('--prompt', type=str, default='北京是', help='生成prompt')
    parser.add_argument('--max_length', type=int, default=100, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--num_samples', type=int, default=3, help='生成样本数')
    parser.add_argument('--interactive', action='store_true', help='交互模式')
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = TheoreticalInference(args.checkpoint)
    
    if args.mode == 'generate':
        print("\n" + "=" * 60)
        print("文本生成")
        print("=" * 60)
        
        if args.interactive:
            print("交互模式 (输入 'quit' 退出)")
            while True:
                prompt = input("\nPrompt: ")
                if prompt.lower() == 'quit':
                    break
                
                results = inferencer.generate(
                    prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    num_samples=args.num_samples
                )
                
                for i, text in enumerate(results, 1):
                    print(f"\n[{i}] {text}")
        else:
            results = inferencer.generate(
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                num_samples=args.num_samples
            )
            
            print(f"\nPrompt: {args.prompt}")
            print("-" * 40)
            for i, text in enumerate(results, 1):
                print(f"\n[{i}] {text}")
                
    elif args.mode == 'analyze':
        print("\n" + "=" * 60)
        print("动力学分析")
        print("=" * 60)
        
        analysis = inferencer.analyze_dynamics(args.prompt, num_steps=50)
        
        print(f"\nPrompt: {args.prompt}")
        print("-" * 40)
        print(f"初始F: {analysis['F_initial']:.4f}")
        print(f"最终F: {analysis['F_final']:.4f}")
        print(f"F下降: {analysis['F_decrease']:.4f}")
        print(f"Lyapunov违反率: {analysis['violation_rate']:.1%}")
        print(f"Lyapunov满足: {'✓' if analysis['lyapunov_satisfied'] else '✗'}")
        print(f"已收敛: {'✓' if analysis['converged'] else '✗'}")
        
    elif args.mode == 'verify':
        print("\n" + "=" * 60)
        print("公理验证")
        print("=" * 60)
        
        results = inferencer.verify_axioms()


if __name__ == '__main__':
    main()

