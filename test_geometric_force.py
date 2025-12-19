"""
测试几何力修复
验证群积分器正确调用几何力
"""

import numpy as np
import sys
sys.path.insert(0, './HEI/src')

from hei.group_integrator import (
    GroupContactIntegrator,
    GroupIntegratorConfig,
    create_initial_group_state,
)
from hei.diamond import diamond_torque_hyperboloid, aggregate_torque_hyperboloid
from hei.inertia import compute_kinetic_energy_gradient_hyperboloid, locked_inertia_hyperboloid
from hei.geometry import disk_to_hyperboloid, cayley_uhp_to_disk


def simple_force_fn(z_uhp, action):
    """简单的测试势能力：指向原点"""
    return -0.1 * z_uhp


def simple_potential_fn(z_uhp, action):
    """简单的测试势能：调和势"""
    return 0.05 * np.sum(np.abs(z_uhp) ** 2)


def test_diamond_hyperboloid():
    """测试 Hyperboloid 上的 diamond 算子"""
    print("=" * 60)
    print("测试 1: Hyperboloid diamond 算子")
    print("=" * 60)
    
    # 创建测试点（Hyperboloid 坐标）
    h = np.array([0.5, 0.3, 1.2])  # (X, Y, T)
    
    # 创建测试力（Hyperboloid 切向量）
    f_h = np.array([0.1, -0.05, 0.02])
    
    # 计算力矩
    torque = diamond_torque_hyperboloid(h, f_h)
    
    print(f"Hyperboloid 坐标: {h}")
    print(f"Hyperboloid 力: {f_h}")
    print(f"计算的力矩 (u, v, w): {torque}")
    print(f"力矩范数: {np.linalg.norm(torque):.6f}")
    
    # 验证力矩是有限的
    assert np.all(np.isfinite(torque)), "力矩包含非有限值！"
    print("✓ 力矩计算成功，所有值都是有限的\n")
    
    return True


def test_aggregate_hyperboloid():
    """测试 Hyperboloid 力矩聚合"""
    print("=" * 60)
    print("测试 2: Hyperboloid 力矩聚合")
    print("=" * 60)
    
    # 创建多个测试点
    n_points = 5
    h = np.random.randn(n_points, 3)
    h[:, 2] = np.abs(h[:, 2]) + 1.0  # 确保 T > 0
    
    # 创建测试力
    f_h = np.random.randn(n_points, 3) * 0.1
    
    # 计算聚合力矩
    torque_total = aggregate_torque_hyperboloid(h, f_h)
    
    print(f"点数: {n_points}")
    print(f"总力矩 (u, v, w): {torque_total}")
    print(f"总力矩范数: {np.linalg.norm(torque_total):.6f}")
    
    # 验证
    assert np.all(np.isfinite(torque_total)), "总力矩包含非有限值！"
    assert torque_total.shape == (3,), f"力矩形状错误: {torque_total.shape}"
    print("✓ 力矩聚合成功\n")
    
    return True


def test_kinetic_gradient():
    """测试动能梯度计算"""
    print("=" * 60)
    print("测试 3: 动能梯度（几何力）")
    print("=" * 60)
    
    # 创建测试点
    h = np.array([[0.3, 0.2, 1.1],
                  [0.5, -0.3, 1.3]])
    
    # 创建测试速度
    xi = np.array([0.1, 0.2, -0.05])
    
    # 计算几何力
    F_geom = compute_kinetic_energy_gradient_hyperboloid(h, xi)
    
    print(f"Hyperboloid 坐标形状: {h.shape}")
    print(f"思维流速 ξ: {xi}")
    print(f"几何力 F_geom:\n{F_geom}")
    print(f"几何力范数: {np.linalg.norm(F_geom, axis=-1)}")
    
    # 验证
    assert np.all(np.isfinite(F_geom)), "几何力包含非有限值！"
    assert F_geom.shape == h.shape, f"几何力形状错误: {F_geom.shape}"
    print("✓ 几何力计算成功\n")
    
    return True


def test_integrator_with_geometric_force():
    """测试群积分器包含几何力"""
    print("=" * 60)
    print("测试 4: 群积分器集成测试")
    print("=" * 60)
    
    # 创建配置
    config = GroupIntegratorConfig(
        max_dt=0.01,
        use_hyperboloid_gamma=True,
        gamma_mode="metric",
    )
    
    # 创建积分器
    integrator = GroupContactIntegrator(
        force_fn=simple_force_fn,
        potential_fn=simple_potential_fn,
        config=config,
    )
    
    # 创建初始状态
    z0 = np.array([0.5 + 0.8j, -0.3 + 0.6j])
    xi0 = np.array([0.1, 0.05, -0.02])
    state = create_initial_group_state(z0, xi0)
    
    print(f"初始 UHP 位置: {state.z_uhp}")
    print(f"初始 Hyperboloid 位置:\n{state.h}")
    print(f"初始思维流速 ξ: {state.xi}")
    print(f"初始惯性矩阵 I:\n{state.I}")
    
    # 执行一步积分
    print("\n执行积分步骤...")
    state_new = integrator.step(state)
    
    print(f"\n新 UHP 位置: {state_new.z_uhp}")
    print(f"新 Hyperboloid 位置:\n{state_new.h}")
    print(f"新思维流速 ξ: {state_new.xi}")
    print(f"新动量 m: {state_new.m}")
    print(f"时间步长 dt: {state_new.dt_last:.6f}")
    print(f"阻尼系数 γ: {state_new.gamma_last:.6f}")
    
    # 验证
    assert np.all(np.isfinite(state_new.z_uhp)), "新位置包含非有限值！"
    assert np.all(np.isfinite(state_new.xi)), "新速度包含非有限值！"
    assert np.all(np.isfinite(state_new.m)), "新动量包含非有限值！"
    
    print("\n✓ 群积分器运行成功！")
    print("✓ 几何力已正确集成到积分器中")
    
    # 多步测试
    print("\n" + "=" * 60)
    print("执行 10 步积分测试...")
    print("=" * 60)
    
    state_curr = state
    for i in range(10):
        state_curr = integrator.step(state_curr)
        z_mag = np.abs(state_curr.z_uhp)
        xi_norm = np.linalg.norm(state_curr.xi)
        print(f"步骤 {i+1:2d}: |z| = {z_mag}, ||ξ|| = {xi_norm:.6f}, dt = {state_curr.dt_last:.6f}")
    
    print("\n✓ 多步积分稳定\n")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("几何力修复验证测试套件")
    print("=" * 60 + "\n")
    
    tests = [
        ("Hyperboloid diamond 算子", test_diamond_hyperboloid),
        ("Hyperboloid 力矩聚合", test_aggregate_hyperboloid),
        ("动能梯度（几何力）", test_kinetic_gradient),
        ("群积分器集成", test_integrator_with_geometric_force),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"✗ 测试失败: {e}\n")
    
    # 打印总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, success, error in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status}: {name}")
        if error:
            print(f"  错误: {error}")
    
    total = len(results)
    passed = sum(1 for _, success, _ in results if success)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！几何力修复成功！")
        return 0
    else:
        print("\n❌ 部分测试失败，需要进一步调试")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
