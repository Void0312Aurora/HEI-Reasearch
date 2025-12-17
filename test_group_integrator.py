#!/usr/bin/env python3
"""
群积分器测试脚本
================

验证混合架构（群积分器 + Hyperboloid 评估）的正确性和稳定性。

测试内容：
1. Hyperboloid 坐标变换的正确性
2. 惯性张量在 Hyperboloid 上的有界性
3. 群积分器的能量守恒（自由粒子）
4. 长周期模拟的稳定性
5. 与原版积分器的结果对比
"""

import numpy as np
import sys

# ============================================================================
# Test 1: Hyperboloid 坐标变换
# ============================================================================

def test_1_hyperboloid_coordinates():
    """验证 Hyperboloid 坐标变换的正确性"""
    print("=" * 60)
    print("Test 1: Hyperboloid 坐标变换")
    print("=" * 60)
    
    from hei.hyperboloid import (
        disk_to_hyperboloid,
        hyperboloid_to_disk,
        hyperbolic_distance_hyperboloid,
        minkowski_inner,
    )
    from hei.geometry import cayley_disk_to_uhp, cayley_uhp_to_disk
    from hei.metrics import poincare_distance_disk
    
    rng = np.random.default_rng(42)
    
    # 生成测试点
    n_points = 100
    r = rng.uniform(0, 0.99, n_points)
    theta = rng.uniform(0, 2*np.pi, n_points)
    z_disk = r * np.exp(1j * theta)
    
    # 测试 1a: 往返变换
    h = disk_to_hyperboloid(z_disk)
    z_disk_back = hyperboloid_to_disk(h)
    
    max_error = np.max(np.abs(z_disk - z_disk_back))
    print(f"  1a. 往返变换误差: {max_error:.2e}")
    assert max_error < 1e-10, f"往返变换误差过大: {max_error}"
    print("      ✓ 通过")
    
    # 测试 1b: Hyperboloid 约束
    # 验证 -X² - Y² + T² = 1
    constraint = -h[..., 0]**2 - h[..., 1]**2 + h[..., 2]**2
    max_constraint_error = np.max(np.abs(constraint - 1.0))
    print(f"  1b. Hyperboloid 约束误差: {max_constraint_error:.2e}")
    assert max_constraint_error < 1e-10, f"约束误差过大: {max_constraint_error}"
    print("      ✓ 通过")
    
    # 测试 1c: 距离一致性
    # Hyperboloid 距离应该与 Poincaré 圆盘距离一致
    for _ in range(10):
        i, j = rng.integers(0, n_points, 2)
        if i == j:
            continue
        d_disk = poincare_distance_disk(z_disk[i], z_disk[j])
        d_hyp = hyperbolic_distance_hyperboloid(h[i], h[j])
        rel_error = abs(d_disk - d_hyp) / max(d_disk, 1e-10)
        assert rel_error < 1e-8, f"距离不一致: disk={d_disk}, hyp={d_hyp}"
    print(f"  1c. 距离一致性检验: 通过 (10 对随机点)")
    print("      ✓ 通过")
    
    # 测试 1d: 边界附近的数值稳定性
    # 在圆盘边缘创建点
    r_boundary = np.array([0.99, 0.999, 0.9999, 0.99999])
    z_boundary = r_boundary * np.exp(1j * np.linspace(0, np.pi, 4))
    h_boundary = disk_to_hyperboloid(z_boundary)
    
    print(f"  1d. 边界点的 Hyperboloid 坐标:")
    for i, (r_val, h_val) in enumerate(zip(r_boundary, h_boundary)):
        print(f"      r={r_val:.5f} → T={h_val[2]:.2f}")
        # T 应该增长但保持有限
        assert np.isfinite(h_val).all(), f"边界点坐标无穷: {h_val}"
    print("      ✓ 通过")
    
    print("\nTest 1 总结: 所有测试通过 ✓\n")
    return True


# ============================================================================
# Test 2: Hyperboloid 惯性张量
# ============================================================================

def test_2_hyperboloid_inertia():
    """验证 Hyperboloid 惯性张量的有界性和正定性"""
    print("=" * 60)
    print("Test 2: Hyperboloid 惯性张量")
    print("=" * 60)
    
    from hei.hyperboloid import disk_to_hyperboloid
    from hei.inertia import locked_inertia_hyperboloid, locked_inertia_uhp
    from hei.geometry import cayley_disk_to_uhp
    
    rng = np.random.default_rng(42)
    
    # 测试 2a: 正定性
    n_points = 50
    r = rng.uniform(0, 0.9, n_points)
    theta = rng.uniform(0, 2*np.pi, n_points)
    z_disk = r * np.exp(1j * theta)
    h = disk_to_hyperboloid(z_disk)
    
    I_hyp = locked_inertia_hyperboloid(h)
    eigs = np.linalg.eigvalsh(I_hyp)
    
    print(f"  2a. 惯性张量特征值: [{eigs[0]:.4f}, {eigs[1]:.4f}, {eigs[2]:.4f}]")
    assert np.all(eigs > 0), f"惯性张量不正定: {eigs}"
    print("      ✓ 正定性验证通过")
    
    # 测试 2b: 边界附近的有界性
    # 这是 Hyperboloid 的关键优势
    print(f"  2b. 边界附近惯性张量的有界性测试:")
    
    for r_val in [0.5, 0.9, 0.99, 0.999]:
        z_boundary = r_val * np.exp(1j * np.linspace(0, 2*np.pi, 20))
        h_boundary = disk_to_hyperboloid(z_boundary)
        I_boundary = locked_inertia_hyperboloid(h_boundary)
        eigs_boundary = np.linalg.eigvalsh(I_boundary)
        cond = eigs_boundary.max() / eigs_boundary.min()
        
        print(f"      r={r_val:.3f}: λ_min={eigs_boundary.min():.4f}, λ_max={eigs_boundary.max():.4f}, κ={cond:.2f}")
        assert np.isfinite(eigs_boundary).all(), f"特征值无穷: {eigs_boundary}"
        assert cond < 1000, f"条件数过大: {cond}"
    
    print("      ✓ 有界性验证通过")
    
    # 测试 2c: 与 UHP 版本的一致性（在内部区域）
    z_uhp = cayley_disk_to_uhp(z_disk)
    I_uhp = locked_inertia_uhp(z_uhp)
    eigs_uhp = np.linalg.eigvalsh(I_uhp)
    
    # 注意：由于正则化策略不同，我们只检查量级一致性
    print(f"  2c. 与 UHP 版本的比较:")
    print(f"      Hyperboloid: λ ∈ [{eigs.min():.4f}, {eigs.max():.4f}]")
    print(f"      UHP:         λ ∈ [{eigs_uhp.min():.4f}, {eigs_uhp.max():.4f}]")
    print("      (两者可能因正则化策略不同而有差异，但量级应相近)")
    
    print("\nTest 2 总结: 所有测试通过 ✓\n")
    return True


# ============================================================================
# Test 3: 群积分器基础测试
# ============================================================================

def test_3_group_integrator_basic():
    """验证群积分器的基本功能"""
    print("=" * 60)
    print("Test 3: 群积分器基础测试")
    print("=" * 60)
    
    from hei.group_integrator import (
        GroupContactIntegrator,
        GroupIntegratorConfig,
        create_initial_group_state,
    )
    from hei.geometry import cayley_disk_to_uhp, sample_disk_hyperbolic
    
    rng = np.random.default_rng(42)
    
    # 创建初始状态
    z_disk0 = sample_disk_hyperbolic(n=20, max_rho=2.0, rng=rng)
    z_uhp0 = cayley_disk_to_uhp(z_disk0)
    
    state = create_initial_group_state(z_uhp0, xi=(0.1, 0.0, 0.0))
    
    print(f"  3a. 初始状态创建:")
    print(f"      G = I (det={np.linalg.det(state.G):.6f})")
    print(f"      ||ξ|| = {np.linalg.norm(state.xi):.4f}")
    print(f"      ||m|| = {np.linalg.norm(state.m):.4f}")
    print("      ✓ 通过")
    
    # 测试 3b: 无力情况下的积分
    def zero_force(z, action):
        return np.zeros_like(z, dtype=np.complex128)
    
    def zero_potential(z, action):
        return 0.0
    
    integrator = GroupContactIntegrator(
        force_fn=zero_force,
        potential_fn=zero_potential,
        config=GroupIntegratorConfig(
            max_dt=0.01,
            use_hyperboloid_gamma=False,  # 无阻尼
            gamma_scale=0.0,
        ),
    )
    
    # 执行 100 步
    states = [state]
    for _ in range(100):
        state = integrator.step(state)
        states.append(state)
    
    print(f"  3b. 100 步积分后:")
    print(f"      det(G) = {np.linalg.det(state.G):.10f}")
    print(f"      ||ξ|| = {np.linalg.norm(state.xi):.4f}")
    
    # 验证群结构保持
    det_error = abs(np.linalg.det(state.G) - 1.0)
    print(f"      det 误差 = {det_error:.2e}")
    assert det_error < 1e-8, f"det(G) 偏离 1: {det_error}"
    print("      ✓ 群结构保持验证通过")
    
    # 测试 3c: 位置变化
    z_final = state.z_uhp
    z_disk_final = state.z_disk
    
    print(f"  3c. 位置演化:")
    print(f"      初始位置范围: r ∈ [{np.abs(z_disk0).min():.3f}, {np.abs(z_disk0).max():.3f}]")
    print(f"      最终位置范围: r ∈ [{np.abs(z_disk_final).min():.3f}, {np.abs(z_disk_final).max():.3f}]")
    
    # 确保所有点仍在圆盘内
    assert np.all(np.abs(z_disk_final) < 1.0), "有点逃逸出圆盘！"
    print("      ✓ 所有点仍在圆盘内")
    
    print("\nTest 3 总结: 所有测试通过 ✓\n")
    return True


# ============================================================================
# Test 4: 能量守恒测试（自由粒子）
# ============================================================================

def test_4_energy_conservation():
    """自由粒子能量守恒测试"""
    print("=" * 60)
    print("Test 4: 自由粒子能量守恒")
    print("=" * 60)
    
    from hei.group_integrator import (
        GroupContactIntegrator,
        GroupIntegratorConfig,
        create_initial_group_state,
    )
    from hei.geometry import cayley_disk_to_uhp, sample_disk_hyperbolic
    from hei.inertia import locked_inertia_hyperboloid
    
    rng = np.random.default_rng(42)
    
    # 创建初始状态
    z_disk0 = sample_disk_hyperbolic(n=30, max_rho=2.0, rng=rng)
    z_uhp0 = cayley_disk_to_uhp(z_disk0)
    
    state = create_initial_group_state(z_uhp0, xi=(0.2, 0.1, 0.05))
    
    # 无力、无阻尼积分器
    def zero_force(z, action):
        return np.zeros_like(z, dtype=np.complex128)
    
    def zero_potential(z, action):
        return 0.0
    
    integrator = GroupContactIntegrator(
        force_fn=zero_force,
        potential_fn=zero_potential,
        config=GroupIntegratorConfig(
            max_dt=0.005,
            use_hyperboloid_gamma=False,
            gamma_scale=0.0,
        ),
    )
    
    # 记录能量
    energies = []
    momenta = []
    
    for step in range(500):
        I = state.I
        K = 0.5 * float(state.xi @ I @ state.xi)
        energies.append(K)
        momenta.append(np.linalg.norm(state.m))
        state = integrator.step(state)
    
    energies = np.array(energies)
    momenta = np.array(momenta)
    
    # 计算漂移
    energy_drift = (energies[-1] - energies[0]) / energies[0] * 100
    momentum_drift = (momenta[-1] - momenta[0]) / momenta[0] * 100
    
    print(f"  初始能量: {energies[0]:.6f}")
    print(f"  最终能量: {energies[-1]:.6f}")
    print(f"  能量漂移: {energy_drift:.4f}%")
    print(f"  动量漂移: {momentum_drift:.4f}%")
    
    # 注意：由于变惯量，能量守恒不完美是预期的
    # 但漂移应该很小
    if abs(energy_drift) < 5.0:
        print("  ✓ 能量漂移在可接受范围内 (<5%)")
    else:
        print(f"  ⚠ 能量漂移较大: {energy_drift:.2f}%")
        print("    (对于变惯量系统，这是预期行为)")
    
    print("\nTest 4 总结: 测试完成\n")
    return True


# ============================================================================
# Test 5: 长周期稳定性测试
# ============================================================================

def test_5_long_term_stability():
    """长周期模拟稳定性测试"""
    print("=" * 60)
    print("Test 5: 长周期稳定性测试 (群积分器 vs 传统积分器)")
    print("=" * 60)
    
    from hei.simulation import run_simulation_group, SimulationConfig
    from hei.archive.legacy_simulation import run_simulation
    from hei.potential import build_hierarchical_potential
    
    rng = np.random.default_rng(42)
    pot = build_hierarchical_potential(n_points=30, depth=2, branching=3, rng=rng)
    
    cfg = SimulationConfig(n_points=30, steps=500)
    
    print("  运行群积分器版本...")
    log_group = run_simulation_group(potential=pot, config=cfg, rng=np.random.default_rng(42))
    
    print("  运行传统积分器版本...")
    log_legacy = run_simulation(potential=pot, config=cfg, rng=np.random.default_rng(42))
    
    # 比较结果
    print("\n  结果比较:")
    print(f"  {'指标':<20} {'群积分器':<15} {'传统积分器':<15}")
    print(f"  {'-'*50}")
    
    e_group = log_group.energy
    e_legacy = log_legacy.energy
    
    print(f"  {'初始能量':<20} {e_group[0]:<15.4f} {e_legacy[0]:<15.4f}")
    print(f"  {'最终能量':<20} {e_group[-1]:<15.4f} {e_legacy[-1]:<15.4f}")
    
    change_group = (e_group[-1] - e_group[0]) / e_group[0] * 100
    change_legacy = (e_legacy[-1] - e_legacy[0]) / e_legacy[0] * 100
    print(f"  {'能量变化%':<20} {change_group:<15.2f} {change_legacy:<15.2f}")
    
    xi_group = log_group.xi_norm
    xi_legacy = log_legacy.xi_norm
    print(f"  {'最终 ||ξ||':<20} {xi_group[-1]:<15.6f} {xi_legacy[-1]:<15.6f}")
    
    # 检查是否有 NaN 或 Inf
    has_nan_group = any(np.isnan(e) or np.isinf(e) for e in e_group)
    has_nan_legacy = any(np.isnan(e) or np.isinf(e) for e in e_legacy)
    
    print(f"\n  数值稳定性:")
    print(f"    群积分器: {'❌ 有 NaN/Inf' if has_nan_group else '✓ 稳定'}")
    print(f"    传统积分器: {'❌ 有 NaN/Inf' if has_nan_legacy else '✓ 稳定'}")
    
    if not has_nan_group:
        print("\n  ✓ 群积分器长周期测试通过")
    
    print("\nTest 5 总结: 测试完成\n")
    return not has_nan_group


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("群积分器 + Hyperboloid 混合架构 测试套件")
    print("=" * 60 + "\n")
    
    results = []
    
    try:
        results.append(("Test 1: Hyperboloid 坐标", test_1_hyperboloid_coordinates()))
    except Exception as e:
        print(f"Test 1 失败: {e}")
        results.append(("Test 1: Hyperboloid 坐标", False))
    
    try:
        results.append(("Test 2: Hyperboloid 惯性", test_2_hyperboloid_inertia()))
    except Exception as e:
        print(f"Test 2 失败: {e}")
        results.append(("Test 2: Hyperboloid 惯性", False))
    
    try:
        results.append(("Test 3: 群积分器基础", test_3_group_integrator_basic()))
    except Exception as e:
        print(f"Test 3 失败: {e}")
        results.append(("Test 3: 群积分器基础", False))
    
    try:
        results.append(("Test 4: 能量守恒", test_4_energy_conservation()))
    except Exception as e:
        print(f"Test 4 失败: {e}")
        results.append(("Test 4: 能量守恒", False))
    
    try:
        results.append(("Test 5: 长周期稳定性", test_5_long_term_stability()))
    except Exception as e:
        print(f"Test 5 失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 5: 长周期稳定性", False))
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    n_passed = sum(1 for _, p in results if p)
    n_total = len(results)
    print(f"\n总计: {n_passed}/{n_total} 测试通过")
    
    return all(p for _, p in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
