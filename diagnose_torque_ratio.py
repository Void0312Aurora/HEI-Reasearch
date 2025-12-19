"""
诊断力矩比例：几何力矩 vs 势能力矩
检查聚合方式是否一致。
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from hei.geometry import cayley_disk_to_uhp, disk_to_hyperboloid, sample_disk_hyperbolic
from hei.inertia import (
    locked_inertia_hyperboloid,
    compute_kinetic_energy_gradient_hyperboloid,
)
from hei.diamond import (
    aggregate_torque,
    aggregate_torque_hyperboloid,
)
from hei.potential import build_baseline_potential

# 生成随机点
rng = np.random.default_rng(42)
z_disk = sample_disk_hyperbolic(n=5, max_rho=2.0, rng=rng)
z_uhp = cayley_disk_to_uhp(z_disk)
h = disk_to_hyperboloid(z_disk)
print(f"有 {len(z_uhp)} 个点")
print(f"h shape: {h.shape}")

# 定义势能并计算势能力
potential = build_baseline_potential(n_anchors=3, max_rho=2.0, rng=rng)
grad_V = potential.dV_dz(z_uhp, z_action=None)
F_potential = -grad_V  # 势能力（UHP复数）
print(f"势能力 F_potential shape: {F_potential.shape}")
print(f"F_potential 范数: {np.linalg.norm(F_potential)}")

# 计算势能力矩（使用原有的 aggregate_torque）
torque_potential = aggregate_torque(z_uhp, F_potential)
print(f"势能力矩 torque_potential: {torque_potential}, 范数: {np.linalg.norm(torque_potential)}")

# 选择一个测试 xi
xi_test = np.array([0.1, 0.0, 0.0])
# 计算几何力（Hyperboloid）
F_geom_h = -compute_kinetic_energy_gradient_hyperboloid(h, xi_test, weights=None)
print(f"几何力 F_geom_h shape: {F_geom_h.shape}")
print(f"F_geom_h 范数: {np.linalg.norm(F_geom_h)}")

# 计算几何力矩（使用 aggregate_torque_hyperboloid）
torque_geom = aggregate_torque_hyperboloid(h, F_geom_h)
print(f"几何力矩 torque_geom: {torque_geom}, 范数: {np.linalg.norm(torque_geom)}")

# 检查聚合方式：比较每个点的 diamond 力矩
from hei.diamond import diamond_torque_hyperboloid
torques_per_point = diamond_torque_hyperboloid(h, F_geom_h)  # (N, 3)
print(f"每个点的力矩形状: {torques_per_point.shape}")
print("每个点的力矩:")
for i in range(torques_per_point.shape[0]):
    print(f"  点 {i}: {torques_per_point[i]}")
print(f"总和: {np.sum(torques_per_point, axis=0)}")
print(f"平均值: {np.mean(torques_per_point, axis=0)}")
print(f"aggregate_torque_hyperboloid 返回: {torque_geom}")

# 计算势能力矩的每个点贡献
from hei.diamond import diamond_torque_vec
torques_potential_per_point = np.array([diamond_torque_vec(z, f) for z, f in zip(z_uhp, F_potential)])
print(f"势能力矩每个点贡献: {torques_potential_per_point.shape}")
print(f"总和: {np.sum(torques_potential_per_point, axis=0)}")
print(f"aggregate_torque 返回: {torque_potential}")

# 比较比例
ratio_norm = np.linalg.norm(torque_geom) / np.linalg.norm(torque_potential)
print(f"几何力矩 / 势能力矩 范数比例: {ratio_norm}")
print(f"点数 N = {len(z_uhp)}")
print(f"如果几何力矩是平均值，则乘以 N 后的比例: {ratio_norm * len(z_uhp)}")

# 检查动能计算
I = locked_inertia_hyperboloid(h)
K = 0.5 * xi_test @ I @ xi_test
print(f"惯性矩阵 I 特征值: {np.linalg.eigvalsh(I)}")
print(f"动能 K = {K}")

# 检查几何力数值稳定性：计算有限差分误差
eps = 1e-5
K_plus = 0.5 * xi_test @ locked_inertia_hyperboloid(h + eps, None) @ xi_test
K_minus = 0.5 * xi_test @ locked_inertia_hyperboloid(h - eps, None) @ xi_test
grad_numeric = (K_plus - K_minus) / (2 * eps)
print(f"数值梯度（标量）: {grad_numeric}")
# 与几何力点积比较
# 几何力是向量，计算其与位移的点积
# 简化：仅检查一个点的 X 分量
point_idx = 0
coord = 0
h_plus = h.copy()
h_minus = h.copy()
h_plus[point_idx, coord] += eps
h_minus[point_idx, coord] -= eps
K_plus = 0.5 * xi_test @ locked_inertia_hyperboloid(h_plus, None) @ xi_test
K_minus = 0.5 * xi_test @ locked_inertia_hyperboloid(h_minus, None) @ xi_test
grad_numeric_single = (K_plus - K_minus) / (2 * eps)
print(f"点 {point_idx} 坐标 {coord} 数值梯度: {grad_numeric_single}")
print(f"对应的几何力分量 F_geom_h[{point_idx}, {coord}] = {F_geom_h[point_idx, coord]}")
print(f"比率 (数值/解析): {grad_numeric_single / F_geom_h[point_idx, coord]}")