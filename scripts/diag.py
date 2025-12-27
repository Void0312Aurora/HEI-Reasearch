import sys
import os
import torch
import numpy as np

# 路径 Hack
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.model.injector import AuroraInjector
from aurora.data.data_pipeline import GlobalEntropyStats
from aurora.engine.readout import ReadoutMechanism
from aurora.physics import geometry

def diagnose():
    print("=== Aurora v3.1 深度诊断程序 ===")
    
    # 1. 检查文件
    model_path = "data/aurora_v3_1.pth"
    if not os.path.exists(model_path):
        print(f"[FAIL] 模型文件不存在: {model_path}")
        return
    else:
        size = os.path.getsize(model_path) / (1024*1024)
        print(f"[PASS] 发现模型文件: {model_path} ({size:.2f} MB)")

    # 2. 加载资源
    stats_path = "data/entropy_stats.json"
    entropy_stats = GlobalEntropyStats(stats_path)
    vocab = sorted(list(entropy_stats.stats.keys()))
    char_to_id = {c:i for i,c in enumerate(vocab)}
    id_to_char = {i:c for i,c in enumerate(vocab)}
    print(f"[INFO] 词表大小: {len(vocab)}")

    # 3. 加载模型 (Experiment 9: DIM=16)
    DIM = 16
    injector = AuroraInjector(len(vocab), 32, 64, DIM)
    
    try:
        # 尝试加载参数
        state_dict = torch.load(model_path, map_location='cpu')
        injector.load_state_dict(state_dict)
        print("[PASS] 模型参数加载成功")
    except Exception as e:
        print(f"[FAIL] 模型加载失败: {e}")
        return

    injector.eval()
    readout = ReadoutMechanism(injector, entropy_stats, id_to_char)

    # 4. 核心测试：自洽性检查 (Self-Consistency Check)
    # 如果模型是好的，injector("你") 产生的坐标，应该离 "你" 的基态最近
    
    test_chars = ["你", "好", "是", "的", "A", "。"]
    print("\n--- 语义基态测试 (Semantic Ground State Test) ---")
    print(f"{'Char':<4} | {'Norm':<6} | {'Self-Dist':<10} | {'Top-1 Prediction':<10} | {'Prob':<6}")
    print("-" * 60)
    
    for char in test_chars:
        if char not in char_to_id:
            print(f"{char:<4} | [OOV]  | -")
            continue
            
        cid = torch.tensor([[char_to_id[char]]], dtype=torch.long)
        r_tgt = torch.tensor([[entropy_stats.get_radial_target(char)]], dtype=torch.float)
        
        with torch.no_grad():
            # 1. 注入得到坐标 q
            m, q, J, p = injector(cid, r_tgt)
            q_vec = q.view(1, DIM)
            
            # 2. 检查半径 (Exp 8 应该在 0.8~0.95 左右)
            r_curr = torch.norm(q_vec).item()
            
            # 3. 读出 (看看它认为自己是谁)
            # 使用高 beta 来模拟尖锐采样
            probs = readout.read_prob(q_vec, beta=50.0)
            top1_idx = probs.argmax().item()
            top1_char = id_to_char[top1_idx]
            top1_prob = probs[0, top1_idx].item()
            
            # 4. 计算到自己的距离 (应该是 0)
            # readout.prototypes 存储了所有基态位置
            # 如果 readout 逻辑没问题，q 应该完全等于 prototype[cid]
            # 距离应该极小
            dist_to_self = -torch.log(probs[0, char_to_id[char]] + 1e-9).item() # 近似距离度量
            
            print(f"{char:<4} | {r_curr:.4f} | {dist_to_self:.4f}     | {top1_char:<10}     | {top1_prob:.4f}")

    print("\n=== 诊断结论 ===")
    print("1. 如果 'Top-1' 也是乱码: 说明模型权重是随机的 (Step 10000 没存进去)。")
    print("2. 如果 'Top-1' 是正确的: 说明模型是好的，是 Chat 脚本的动力学参数(T/Friction)有问题。")
    print("3. 如果 'Norm' 极小 (<0.1): 说明发生了'原点坍缩'。")

if __name__ == "__main__":
    diagnose()