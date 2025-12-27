"""
Aurora Interactive Chat (CCD v3.1).
===================================

Demonstrates "Natural Language Interaction" via geometric field dynamics.
Pipeline:
1. Input Stream -> Injector -> Physics Evolution -> LTM.
2. Generation -> Readout -> Sampling -> Self-Injection.

Ref: Axiom 3.3 (Readout) & 4.3 (Memory).
"""

import sys
import os
import torch
import torch.nn.functional as F
import json
import logging
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data.data_pipeline import GlobalEntropyStats
from aurora.model.injector import AuroraInjector
from aurora.engine.forces import ForceField
from aurora.engine.integrator import LieIntegrator
from aurora.engine.memory import HyperbolicMemory
from aurora.engine.readout import ReadoutMechanism
from aurora.physics import geometry

# Config (Experiment 9: Increased Dimension)
DIM = 16
EMBED_DIM = 32
HIDDEN_DIM = 64
MEMORY_SIZE = 10000
CONTEXT_K = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Physics Hyperparameters (Phase 38) ---
TEMPERATURE = 0.0   # 热力学温度 (kT): 越高越随机，越低越固执
FRICTION = 2      # 阻尼系数: 防止动量无限累积
DT = 0.05            # 积分步长
MASS_SCALE = 1.0    # 质量缩放

def main():
    print(f"Initializing Aurora v3.1 Engine on {DEVICE}...")
    
    # 1. Load Resources
    stats_path = "data/entropy_stats.json"
    if not os.path.exists(stats_path):
        print("Error: data/entropy_stats.json missing.")
        return
        
    entropy_stats = GlobalEntropyStats(stats_path)
    vocab_list = sorted(list(entropy_stats.stats.keys()))
    char_to_id = {c:i for i,c in enumerate(vocab_list)}
    id_to_char = {i:c for i,c in enumerate(vocab_list)}
    VOCAB_SIZE = len(vocab_list)
    
    # 2. Model
    injector = AuroraInjector(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, DIM).to(DEVICE)
    model_path = "data/aurora_v3_1.pth"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        injector.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("Warning: No trained model found. Using random weights.")
    
    injector.eval()
        
    # 3. Engines
    # Match Training Config (Exp 8)
    ff = ForceField(G=1.0, lambda_gauge=1.5, k_geo=1.0, mu=1.0, lambda_quartic=0.01)
    # Note: We don't use integrator.step directly because we need custom Langevin logic
    ltm = HyperbolicMemory(DIM, MEMORY_SIZE, device=DEVICE)
    readout = ReadoutMechanism(injector, entropy_stats, id_to_char)
    
    # State Buffers
    active_q = []
    active_m = []
    active_J = []
    
    MAX_ATOMS = 30 
    
    print("\n--- Aurora v3.1 Chat (Physics Enhanced) ---")
    print(f"T={TEMPERATURE}, Friction={FRICTION}")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        # 1. Ingest Input (Excitation Phase)
        last_q = None
        last_p = None
        
        with torch.no_grad():
            for char in user_input:
                if char not in char_to_id: continue
                    
                cid = char_to_id[char]
                r_tgt = entropy_stats.get_radial_target(char)
                
                # Inject
                cid_t = torch.tensor([[cid]], dtype=torch.long, device=DEVICE)
                r_t = torch.tensor([[r_tgt]], dtype=torch.float, device=DEVICE)
                
                m, q_rest, J, p_init = injector(cid_t, r_t)
                
                q_c = q_rest.view(1, DIM)
                p_c = p_init.view(1, DIM)
                m_c = m.view(1, 1)
                J_c = J.view(1, DIM, DIM)
                
                # Manage Context Buffer
                if len(active_q) >= MAX_ATOMS:
                    active_q.pop(0); active_m.pop(0); active_J.pop(0)
                    
                active_q.append(q_c)
                active_m.append(m_c)
                active_J.append(J_c)
                
                last_q = q_c
                last_p = p_c
                
        # 2. Generate Reply (Relaxation Phase)
        print("Aurora: ", end='', flush=True)
        
        gen_len = 0
        max_len = 60
        
        # Start state
        curr_q = last_q if last_q is not None else torch.zeros(1, DIM, device=DEVICE)
        curr_p = last_p if last_p is not None else torch.randn(1, DIM, device=DEVICE)
        curr_m = torch.tensor([[1.0]], device=DEVICE) # Default mass for probe
        curr_J = torch.zeros(1, DIM, DIM, device=DEVICE)
        
        while gen_len < max_len:
             # === 公理 3.3.2 兼容生成循环 ===
             
             # --- A. READOUT (在当前位置读出) ---
             probs = readout.read_prob(curr_q, beta=50.0)
             dist = torch.distributions.Categorical(probs)
             idx = dist.sample()
             char = id_to_char[idx.item()]
             
             print(char, end='', flush=True)
             gen_len += 1
             
             if char == '\n': break
             
             # --- B. 获取观测本征态 (读出字符的基态参数) ---
             cid = torch.tensor([[idx]], dtype=torch.long, device=DEVICE)
             r = torch.tensor([[entropy_stats.get_radial_target(char)]], dtype=torch.float, device=DEVICE)
             
             m_c, q_c, J_c, p_c = injector(cid, r)
             q_c = q_c.view(1, DIM)
             p_c = p_c.view(1, DIM)
             
             # --- C. 软投影 (公理 3.3.2) ---
             # 位置修正: 朝基态移动，但不完全重置
             # q^+ = Exp_q(α * Log_q(q_c))
             alpha = 0.3  # 柔性系数：越小越保守，越大越激进
             
             # 计算修正向量: 从当前位置指向基态的切向量
             correction = geometry.log_map(curr_q, q_c)  # 返回 (1, DIM) 切向量
             
             # 软位置修正: 沿修正方向移动 α 比例
             curr_q = geometry.exp_map(curr_q, alpha * correction)
             
             # --- D. 动量流动 (公理 2.2 测地线惯性) ---
             # 用训练好的动量 p 驱动粒子流动
             # q_{t+1} = Exp_q(p * dt)
             dt = 0.1  # 时间步长
             curr_q = geometry.exp_map(curr_q, curr_p * dt)
             
             # --- E. 动量反冲 (公理 3.3.2) ---
             # p^+ = p^- - ΔE_emit + Δp_recoil
             # 简化: p 减去修正消耗的"能量"，并加上新字符的动量贡献
             recoil_strength = 0.5
             curr_p = curr_p - alpha * correction  # 修正消耗
             curr_p = curr_p + recoil_strength * p_c  # 新字符动量贡献
             
             # 动量安全截断
             curr_p = torch.clamp(curr_p, min=-1.0, max=1.0)
             
             # 更新内部状态
             curr_m = m_c.view(1, 1)
             curr_J = J_c.view(1, DIM, DIM)
             
             # --- F. 上下文缓冲区更新 ---
             if len(active_q) >= MAX_ATOMS:
                active_q.pop(0); active_m.pop(0); active_J.pop(0)
             
             active_q.append(q_c) 
             active_m.append(m_c.view(1, 1))
             active_J.append(J_c.view(1, DIM, DIM))
             
        print() 
        
if __name__ == "__main__":
    main()
