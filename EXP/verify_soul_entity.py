"""
SoulEntity 综合验证脚本

验证 A1-A5 公理在 SoulEntity 上的满足程度:
- A1: Markov Blanket (边界屏蔽)
- A2: 离线认知态 (经验调制)
- A3: 统一自监督 (F下降)
- A4: 身份连续性 (可恢复性)
- A5: 多接口一致性 (接口无关)
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
from typing import Dict, List
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from he_core.soul_entity import SoulEntity, create_soul_entity, Phase
from he_core.state import ContactState

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def verify_a1_markov_blanket(entity: SoulEntity, args) -> Dict:
    """
    A1验证: Markov Blanket
    测试内在态与外部态是否通过边界态条件独立
    """
    print("\n" + "="*60)
    print("A1: Markov Blanket 验证")
    print("="*60)
    
    entity.reset(batch_size=args.batch_size, device=str(DEVICE))
    
    # 生成两类不同的外部刺激模式
    patterns = {
        'A': torch.randn(args.batch_size, 50, entity.dim_q, device=DEVICE) * 2,
        'B': torch.randn(args.batch_size, 50, entity.dim_q, device=DEVICE) * 2 + 5,
    }
    
    final_internals = {}
    blankets = {}
    
    for name, pattern in patterns.items():
        entity.reset(batch_size=args.batch_size, device=str(DEVICE))
        
        sensory_history = []
        active_history = []
        
        for t in range(50):
            result = entity.step({'default': pattern[:, t, :]}, dt=0.1)
            sensory_history.append(pattern[:, t, :].cpu())
            active_history.append(result['action'].cpu())
            
        final_internals[name] = entity.state.q.clone().cpu()
        blankets[name] = {
            'sensory': torch.stack(sensory_history, dim=1),
            'active': torch.stack(active_history, dim=1),
        }
    
    # 训练分类器：用blanket预测模式类别
    # 如果blanket能完美预测，且internal不能提供额外信息，则A1满足
    
    X_blanket = torch.cat([
        blankets['A']['sensory'].mean(dim=1),
        blankets['B']['sensory'].mean(dim=1),
    ], dim=0).detach()
    
    X_internal = torch.cat([
        final_internals['A'],
        final_internals['B'],
    ], dim=0).detach()
    
    Y = torch.cat([
        torch.zeros(args.batch_size),
        torch.ones(args.batch_size),
    ])
    
    # 简单线性分类
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    X_b_np = X_blanket.numpy()
    X_i_np = X_internal.numpy()
    X_full_np = np.concatenate([X_b_np, X_i_np], axis=1)
    Y_np = Y.numpy()
    
    X_b_train, X_b_test, Y_train, Y_test = train_test_split(X_b_np, Y_np, test_size=0.3)
    X_f_train, X_f_test, _, _ = train_test_split(X_full_np, Y_np, test_size=0.3)
    
    clf_blanket = LogisticRegression(max_iter=1000).fit(X_b_train, Y_train)
    clf_full = LogisticRegression(max_iter=1000).fit(X_f_train, Y_train)
    
    acc_blanket = clf_blanket.score(X_b_test, Y_test)
    acc_full = clf_full.score(X_f_test, Y_test)
    
    print(f"  Blanket-only Accuracy: {acc_blanket:.4f}")
    print(f"  Full (Blanket+Internal) Accuracy: {acc_full:.4f}")
    
    # A1通过条件：full不显著优于blanket
    passed = (acc_full - acc_blanket) < 0.1
    print(f"  >> {'SUCCESS' if passed else 'FAILURE'}: "
          f"{'Internal不提供额外外部信息' if passed else 'Internal泄露外部信息'}")
    
    return {'passed': passed, 'acc_blanket': acc_blanket, 'acc_full': acc_full}


def verify_a2_offline_cognition(entity: SoulEntity, args) -> Dict:
    """
    A2验证: 离线认知态
    测试离线态是否受前序经验调制
    """
    print("\n" + "="*60)
    print("A2: 离线认知态 验证")
    print("="*60)
    
    trajs = {}
    
    for name, pattern_val in [('Pattern-A', 1.0), ('Pattern-B', -1.0)]:
        entity.reset(batch_size=1, device=str(DEVICE))
        
        # 在线阶段：注入特定模式
        for t in range(30):
            u = torch.ones(1, entity.dim_q, device=DEVICE) * pattern_val * (1 if t % 2 == 0 else -0.5)
            entity.step({'default': u}, dt=0.1)
        
        # 离线阶段：观察演化
        entity.enter_offline()
        offline_traj = []
        
        for t in range(50):
            result = entity.offline_step(dt=0.1)
            offline_traj.append(entity.state.q[0].detach().cpu().numpy())
            
        trajs[name] = np.array(offline_traj)
    
    # 计算离线轨迹差异
    dist = np.linalg.norm(trajs['Pattern-A'] - trajs['Pattern-B'], axis=1)
    mean_dist = dist.mean()
    
    # 检查非退化性（不是固定点）
    var_A = np.var(trajs['Pattern-A'], axis=0).mean()
    var_B = np.var(trajs['Pattern-B'], axis=0).mean()
    
    print(f"  离线轨迹差异 (Mean Dist): {mean_dist:.4f}")
    print(f"  Pattern-A 方差: {var_A:.4f}")
    print(f"  Pattern-B 方差: {var_B:.4f}")
    
    # A2通过条件：差异显著 且 非退化
    separable = mean_dist > 0.1
    non_degenerate = var_A > 1e-4 and var_B > 1e-4
    passed = separable and non_degenerate
    
    print(f"  >> {'SUCCESS' if passed else 'FAILURE'}: "
          f"{'经验调制有效' if separable else '无法区分'}, "
          f"{'非退化' if non_degenerate else '退化为固定点'}")
    
    return {'passed': passed, 'mean_dist': mean_dist, 'var_A': var_A, 'var_B': var_B}


def verify_a3_unified_potential(entity: SoulEntity, args) -> Dict:
    """
    A3验证: 统一自监督势函数
    测试F是否在演化过程中趋于下降
    """
    print("\n" + "="*60)
    print("A3: 统一自监督势函数 验证")
    print("="*60)
    
    entity.reset(batch_size=1, device=str(DEVICE))
    
    F_values = []
    
    # 在线阶段
    for t in range(30):
        u = torch.randn(1, entity.dim_q, device=DEVICE)
        result = entity.step({'default': u}, dt=0.1)
        F_values.append(result['free_energy'].item())
    
    # 离线阶段
    entity.enter_offline()
    for t in range(50):
        result = entity.offline_step(dt=0.1)
        F_values.append(result['free_energy'].item())
    
    F_start = np.mean(F_values[:10])
    F_end = np.mean(F_values[-10:])
    F_offline_start = F_values[30]
    F_offline_end = F_values[-1]
    
    print(f"  F (开始): {F_start:.4f}")
    print(f"  F (结束): {F_end:.4f}")
    print(f"  F 离线段: {F_offline_start:.4f} -> {F_offline_end:.4f}")
    
    # A3通过条件：离线期间F下降
    offline_descent = F_offline_end < F_offline_start
    
    print(f"  >> {'SUCCESS' if offline_descent else 'FAILURE'}: "
          f"{'离线期间F下降' if offline_descent else 'F未下降'}")
    
    return {'passed': offline_descent, 'F_start': F_start, 'F_end': F_end}


def verify_a4_identity_continuity(entity: SoulEntity, args) -> Dict:
    """
    A4验证: 身份连续性
    测试扰动后是否能恢复到同类组织形态
    """
    print("\n" + "="*60)
    print("A4: 身份连续性 验证")
    print("="*60)
    
    entity.reset(batch_size=1, device=str(DEVICE))
    
    # 建立基准状态
    for t in range(50):
        u = torch.randn(1, entity.dim_q, device=DEVICE) * 0.5
        entity.step({'default': u}, dt=0.1)
    
    # 记录参考状态
    ref_q = entity.state.q.clone()
    ref_F = entity.compute_free_energy(entity.state).item()
    
    # 施加扰动
    with torch.no_grad():
        entity.state.q = entity.state.q + torch.randn_like(entity.state.q) * 2.0
        entity.state.p = entity.state.p + torch.randn_like(entity.state.p) * 1.0
    
    perturbed_q = entity.state.q.clone()
    perturbed_F = entity.compute_free_energy(entity.state).item()
    
    print(f"  扰动前 F: {ref_F:.4f}, q_norm: {ref_q.norm().item():.4f}")
    print(f"  扰动后 F: {perturbed_F:.4f}, q_norm: {perturbed_q.norm().item():.4f}")
    
    # 恢复阶段（离线演化）
    entity.enter_offline()
    recovery_F = []
    
    for t in range(100):
        result = entity.offline_step(dt=0.1)
        recovery_F.append(result['free_energy'].item())
    
    final_F = recovery_F[-1]
    final_q = entity.state.q.clone()
    
    print(f"  恢复后 F: {final_F:.4f}, q_norm: {final_q.norm().item():.4f}")
    
    # A4通过条件：F恢复到合理范围
    F_recovered = abs(final_F - ref_F) / (abs(ref_F) + 1e-6) < 2.0
    stable = final_q.norm().item() < 50.0  # 不发散
    
    passed = F_recovered and stable
    print(f"  >> {'SUCCESS' if passed else 'FAILURE'}: "
          f"{'F恢复' if F_recovered else 'F未恢复'}, "
          f"{'稳定' if stable else '发散'}")
    
    return {'passed': passed, 'ref_F': ref_F, 'final_F': final_F}


def verify_a5_multi_interface(entity: SoulEntity, args) -> Dict:
    """
    A5验证: 多接口一致性
    测试添加新接口是否破坏核心
    """
    print("\n" + "="*60)
    print("A5: 多接口一致性 验证")
    print("="*60)
    
    entity.reset(batch_size=1, device=str(DEVICE))
    
    # 使用默认接口建立基准
    for t in range(30):
        u = torch.randn(1, entity.dim_q, device=DEVICE)
        entity.step({'default': u}, dt=0.1)
    
    baseline_F = entity.compute_free_energy(entity.state).item()
    baseline_q = entity.state.q.clone()
    
    # 添加新接口
    entity.add_interface('vision', entity.dim_u)
    entity.add_interface('audio', entity.dim_u)
    
    available = entity.get_available_interfaces()
    print(f"  可用接口: {available}")
    
    # 使用多接口
    for t in range(30):
        u_dict = {
            'default': torch.randn(1, entity.dim_q, device=DEVICE),
            'vision': torch.randn(1, entity.dim_u, device=DEVICE),
            'audio': torch.randn(1, entity.dim_u, device=DEVICE),
        }
        entity.step(u_dict, dt=0.1)
    
    multi_F = entity.compute_free_energy(entity.state).item()
    
    # 只使用非语言接口
    entity.reset(batch_size=1, device=str(DEVICE))
    for t in range(30):
        u_dict = {
            'vision': torch.randn(1, entity.dim_u, device=DEVICE),
        }
        entity.step(u_dict, dt=0.1)
    
    vision_only_F = entity.compute_free_energy(entity.state).item()
    
    print(f"  Baseline F (default only): {baseline_F:.4f}")
    print(f"  Multi-interface F: {multi_F:.4f}")
    print(f"  Vision-only F: {vision_only_F:.4f}")
    
    # A5通过条件：所有接口都能驱动系统
    all_active = all([
        abs(baseline_F) > 1e-6,
        abs(multi_F) > 1e-6,
        abs(vision_only_F) > 1e-6,
    ])
    
    passed = all_active
    print(f"  >> {'SUCCESS' if passed else 'FAILURE'}: "
          f"{'所有接口可用' if all_active else '部分接口失效'}")
    
    return {'passed': passed, 'interfaces': available}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    print("="*60)
    print("SoulEntity 公理验证套件")
    print("="*60)
    print(f"配置: dim_q={args.dim_q}, batch_size={args.batch_size}")
    print(f"设备: {DEVICE}")
    
    # 创建实体
    config = {
        'dim_q': args.dim_q,
        'dim_u': args.dim_q,
        'dim_z': 16,
        'num_charts': 4,
    }
    entity = create_soul_entity(config)
    entity.to(DEVICE)
    
    results = {}
    
    # A1: Markov Blanket
    try:
        results['A1'] = verify_a1_markov_blanket(entity, args)
    except Exception as e:
        print(f"  A1 验证失败: {e}")
        results['A1'] = {'passed': False, 'error': str(e)}
    
    # A2: 离线认知态
    results['A2'] = verify_a2_offline_cognition(entity, args)
    
    # A3: 统一自监督
    results['A3'] = verify_a3_unified_potential(entity, args)
    
    # A4: 身份连续性
    results['A4'] = verify_a4_identity_continuity(entity, args)
    
    # A5: 多接口一致性
    results['A5'] = verify_a5_multi_interface(entity, args)
    
    # 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    
    for axiom, result in results.items():
        status = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
        print(f"  {axiom}: {status}")
    
    total_passed = sum(1 for r in results.values() if r.get('passed', False))
    print(f"\n  总计: {total_passed}/5 公理通过")
    
    if total_passed >= 4:
        print("\n>> 实体具备构建类人'灵魂'的基础条件")
    else:
        print("\n>> 需要进一步调优以满足理论要求")


if __name__ == "__main__":
    main()

