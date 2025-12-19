"""
诊断几何力修复后的数值异常问题
添加详细的调试输出来检查动能、几何力、坐标转换和力矩符号一致性。
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from hei.geometry import cayley_disk_to_uhp, cayley_uhp_to_disk, disk_to_hyperboloid
from hei.inertia import (
    locked_inertia_hyperboloid,
    compute_kinetic_energy_gradient_hyperboloid,
)
from hei.diamond import (
    diamond_torque_hyperboloid,
    aggregate_torque_hyperboloid,
)
from hei.group_integrator import GroupContactIntegrator, GroupIntegratorConfig, GroupIntegratorState, create_initial_group_state
from hei.potential import build_baseline_potential
from hei.simulation import SimulationConfig, run_simulation_group

# 猴子补丁：包装关键函数以打印调试信息
original_locked_inertia = locked_inertia_hyperboloid
original_gradient = compute_kinetic_energy_gradient_hyperboloid
original_diamond_torque = diamond_torque_hyperboloid
original_aggregate = aggregate_torque_hyperboloid

def debug_locked_inertia(h, weights=None):
    I = original_locked_inertia(h, weights)
    print(f"[DEBUG locked_inertia_hyperboloid] h shape: {h.shape}, I shape: {I.shape}")
    print(f"  I eigenvalues: {np.linalg.eigvalsh(I)}")
    print(f"  I determinant: {np.linalg.det(I)}")
    print(f"  I trace: {np.trace(I)}")
    # 检查是否有零特征值
    eigvals = np.linalg.eigvalsh(I)
    if np.any(eigvals < 1e-10):
        print(f"  WARNING: near-zero eigenvalue detected! min eig = {eigvals.min()}")
    return I

def debug_gradient(h, xi, weights=None):
    F = original_gradient(h, xi, weights)
    print(f"[DEBUG compute_kinetic_energy_gradient_hyperboloid] h shape: {h.shape}, xi = {xi}")
    print(f"  F shape: {F.shape}, F norm: {np.linalg.norm(F)}")
    print(f"  F max component: {np.max(np.abs(F))}")
    # 检查数值稳定性
    if np.any(np.isnan(F)) or np.any(np.isinf(F)):
        print(f"  ERROR: F contains NaN or Inf!")
    # 计算动能 K 用于验证
    I = original_locked_inertia(h, weights)
    K = 0.5 * xi @ I @ xi
    print(f"  Kinetic energy K = {K}")
    return F

def debug_diamond_torque(h, f_h):
    torque = original_diamond_torque(h, f_h)
    print(f"[DEBUG diamond_torque_hyperboloid] h shape: {h.shape}, f_h shape: {f_h.shape}")
    print(f"  torque = {torque}, norm = {np.linalg.norm(torque)}")
    # 检查坐标转换因子
    X, Y, T = h[..., 0], h[..., 1], h[..., 2]
    fX, fY, fT = f_h[..., 0], f_h[..., 1], f_h[..., 2]
    factor = 1.0 / (1.0 + T + 1e-15)
    print(f"  T = {T}, factor = {factor}")
    # 计算 UHP 力以验证
    z_uhp = (X + 1j * Y) * factor
    f_uhp = (fX + 1j * fY) * factor - z_uhp * fT * factor
    print(f"  z_uhp = {z_uhp}, f_uhp = {f_uhp}")
    return torque

def debug_aggregate_torque(h, f_h, weights=None):
    torque = original_aggregate(h, f_h, weights)
    print(f"[DEBUG aggregate_torque_hyperboloid] h shape: {h.shape}, f_h shape: {f_h.shape}")
    print(f"  aggregate torque = {torque}, norm = {np.linalg.norm(torque)}")
    return torque

# 应用猴子补丁
import hei.inertia
hei.inertia.locked_inertia_hyperboloid = debug_locked_inertia
hei.inertia.compute_kinetic_energy_gradient_hyperboloid = debug_gradient
import hei.diamond
hei.diamond.diamond_torque_hyperboloid = debug_diamond_torque
hei.diamond.aggregate_torque_hyperboloid = debug_aggregate_torque

# 同时包装 group_integrator 中的关键步骤
from hei.group_integrator import GroupContactIntegrator
original_step = GroupContactIntegrator.step

def debug_step(self, state):
    print("\n" + "="*60)
    print(f"Step {self._step_count}")
    print("="*60)
    # 计算当前位置
    z_current = state.z_uhp
    h_current = state.h
    xi = state.xi
    I = state.I
    print(f"  z shape: {z_current.shape}, h shape: {h_current.shape}")
    print(f"  xi = {xi}, norm = {np.linalg.norm(xi)}")
    print(f"  I eigenvalues: {np.linalg.eigvalsh(I)}")
    # 动能
    K = 0.5 * xi @ I @ xi
    print(f"  Kinetic energy K = {K}")
    # 势能
    # 需要访问 potential_fn，但暂时跳过
    # 调用原始步骤
    new_state = original_step(self, state)
    # 打印新状态的关键量
    print(f"  new xi = {new_state.xi}, norm = {np.linalg.norm(new_state.xi)}")
    print(f"  new K = {0.5 * new_state.xi @ new_state.I @ new_state.xi}")
    return new_state

GroupContactIntegrator.step = debug_step

def main():
    print("开始诊断几何力异常问题")
    # 使用简单的势能和少量点
    potential = build_baseline_potential(n_anchors=3, max_rho=2.0)
    config = SimulationConfig(
        n_points=5,
        steps=10,  # 仅运行10步
        max_rho=2.0,
        enable_diamond=True,
        initial_xi=(0.1, 0.0, 0.0),
        eps_dt=1e-1,
        max_dt=5e-2,
        min_dt=1e-5,
        disable_dissipation=False,
    )
    # 运行模拟（将使用我们的调试包装器）
    log = run_simulation_group(potential=potential, config=config)
    print("\n诊断完成。")
    # 输出日志摘要
    print(f"动能序列: {log.kinetic[:5]}")
    print(f"势能序列: {log.potential[:5]}")
    print(f"总能量序列: {log.energy[:5]}")
    print(f"xi 范数序列: {log.xi_norm[:5]}")
    print(f"惯性特征值最大值序列: {log.inertia_eig_max[:5]}")
    print(f"惯性特征值最小值序列: {log.inertia_eig_min[:5]}")
    print(f"力矩范数序列: {log.torque_norm[:5]}")

if __name__ == "__main__":
    main()