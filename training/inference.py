"""
推理脚本 - 测试模型生成质量

用法：
python HEI/training/inference.py --checkpoint checkpoints/test_large/best_model.pt --prompt "中国"
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.large_scale_trainer import LargeScaleModel, LargeScaleConfig


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """加载模型"""
    print(f"加载模型: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 恢复配置
    config = LargeScaleConfig.from_dict(checkpoint['config'])
    config.device = device
    
    # 创建模型
    model = LargeScaleModel(config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 恢复tokenizer
    tokenizer = checkpoint.get('tokenizer')
    
    print(f"  词汇表大小: {config.vocab_size}")
    print(f"  dim_q: {config.dim_q}")
    print(f"  训练步数: {checkpoint.get('global_step', 'unknown')}")
    
    return model, tokenizer, config


@torch.no_grad()
def generate(model, tokenizer, config, prompt: str, 
             max_new_tokens: int = 100,
             temperature: float = 0.8,
             top_k: int = 50,
             top_p: float = 0.9) -> str:
    """生成文本"""
    device = config.device
    
    # 编码prompt
    if tokenizer:
        tokens = tokenizer.encode(prompt)
    else:
        # 如果没有tokenizer，使用简单的字符编码
        tokens = [ord(c) % config.vocab_size for c in prompt]
    
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # 生成
    for _ in range(max_new_tokens):
        # 截断到最大长度
        if tokens.shape[1] > config.max_seq_len:
            tokens = tokens[:, -config.max_seq_len:]
        
        # 前向传播
        outputs = model(tokens)
        logits = outputs['logits'][:, -1, :]  # 取最后一个位置
        
        # 温度调节
        logits = logits / temperature
        
        # Top-k 过滤
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Top-p (nucleus) 过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # 采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 添加到序列
        tokens = torch.cat([tokens, next_token], dim=1)
        
        # 检查结束符
        if tokenizer and next_token.item() == tokenizer.token_to_id.get('<EOS>', -1):
            break
    
    # 解码
    if tokenizer:
        generated = tokenizer.decode(tokens[0].tolist())
    else:
        generated = ''.join([chr(t) if 32 <= t < 127 else '?' for t in tokens[0].tolist()])
    
    return generated


def interactive_mode(model, tokenizer, config):
    """交互式对话模式"""
    print("\n" + "=" * 60)
    print("交互式生成模式")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\n输入> ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("退出")
                break
            
            if not prompt:
                continue
            
            print("\n生成中...")
            result = generate(model, tokenizer, config, prompt)
            print(f"\n输出: {result}")
            
        except KeyboardInterrupt:
            print("\n退出")
            break


def test_samples(model, tokenizer, config):
    """测试样本生成"""
    test_prompts = [
        "中国",
        "数学是",
        "人工智能",
        "北京",
        "计算机科学",
    ]
    
    print("\n" + "=" * 60)
    print("测试样本生成")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\n{'='*40}")
        print(f"Prompt: {prompt}")
        print("-" * 40)
        
        # 生成多个样本
        for i in range(3):
            result = generate(model, tokenizer, config, prompt, 
                            max_new_tokens=50, 
                            temperature=0.8)
            print(f"  [{i+1}] {result[:100]}...")
    
    print("\n" + "=" * 60)


def compute_perplexity(model, tokenizer, config, text: str) -> float:
    """计算给定文本的困惑度"""
    device = config.device
    
    if tokenizer:
        tokens = tokenizer.encode(text)
    else:
        tokens = [ord(c) % config.vocab_size for c in text]
    
    if len(tokens) < 2:
        return float('inf')
    
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    
    outputs = model(tokens[:, :-1])
    logits = outputs['logits']
    
    # 计算交叉熵
    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        tokens[:, 1:].reshape(-1)
    )
    
    return torch.exp(loss).item()


def main():
    parser = argparse.ArgumentParser(description='模型推理测试')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint路径')
    parser.add_argument('--prompt', type=str, default=None, help='生成prompt')
    parser.add_argument('--max_tokens', type=int, default=100, help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.8, help='温度')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k采样')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p采样')
    parser.add_argument('--interactive', action='store_true', help='交互模式')
    parser.add_argument('--test', action='store_true', help='运行测试样本')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer, config = load_model(args.checkpoint, args.device)
    
    if args.test:
        test_samples(model, tokenizer, config)
    elif args.interactive:
        interactive_mode(model, tokenizer, config)
    elif args.prompt:
        result = generate(model, tokenizer, config, args.prompt,
                         max_new_tokens=args.max_tokens,
                         temperature=args.temperature,
                         top_k=args.top_k,
                         top_p=args.top_p)
        print(f"\nPrompt: {args.prompt}")
        print(f"Output: {result}")
    else:
        # 默认运行测试
        test_samples(model, tokenizer, config)


if __name__ == '__main__':
    main()

