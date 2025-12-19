"""
诊断边界附近的几何力和惯性特征值。
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from hei.geometry import cayley_disk_to_uhp, cayley_uhp_to_disk, disk_to_hyperboloid
from hei.inertia import locked_inertia_hyperboloid, compute_kinetic_energy_gradient_hyperboloid

# 生成靠近边界的点
r = 0.99
theta = np.linspace(0, 2*np.pi, 5, endpoint=False)
z_disk = r * np.exp(1j * theta)
print(f"圆盘坐标（靠近边界 r={r}）: {z_disk}")

z_uhp = cayley_disk_to_uhp(z_disk)
print(f"UHP 坐标: {z_uhp}")
print(f"虚部 y = {np.imag(z_uhp)}")

h = disk_to_hyperboloid(z_disk)
print(f"双曲面坐标 h shape: {h.shape}")
print(f"h:\n{h}")
print(f"T 分量: {h[:,2]}")

# 计算惯性矩阵
I = locked_inertia_hyperboloid(h)
eigvals = np.linalg.eigvalsh(I)
print(f"惯性矩阵特征值: {eigvals}")
print(f"lambda_max / lambda_min = {eigvals.max() / eigvals.min()}")

# 选择一个测试 xi
xi = np.array([0.1, 0.0, 0.0])
# 计算几何力
F_geom_h = -compute_kinetic_energy_gradient_hyperboloid(h, xi, weights=None)
print(f"几何力 F_geom_h shape: {F_geom_h.shape}")
print(f"F_geom_h 范数: {np.linalg.norm(F_geom_h)}")
print(f"F_geom_h 每点范数: {np.linalg.norm(F_geom_h, axis=1)}")

# 检查几何力方向：计算沿双曲面坐标的“径向”方向
# 对于每个点，计算从原点的位移（双曲面中的原点？）
# 双曲面原点在 (0,0,1)。计算向量 (X, Y, T-1)
origin = np.array([0.0, 0.0, 1.0])
radial = h - origin[None, :]
radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
radial_unit = radial / (radial_norm + 1e-12)
# 计算几何力在径向方向上的投影
proj = np.sum(F_geom_h * radial_unit, axis=1)
print(f"几何力沿径向的投影: {proj}")
print(f"正值表示向外推，负值表示向内拉")

# 计算动能
K = 0.5 * xi @ I @ xi
print(f"动能 K = {K}")

# 计算势能力（简单梯度）
from hei.potential import build_baseline_potential
potential = build_baseline_potential(n_anchors=3, max_rho=2.0, rng=np.random.default_rng(42))
grad_V = potential.dV_dz(z_uhp, z_action=None)
F_potential = -grad_V
print(f"势能力范数: {np.linalg.norm(F_potential)}")

# 计算力矩
from hei.diamond import aggregate_torque, aggregate_torque_hyperboloid
torque_potential = aggregate_torque(z_uhp, F_potential)
torque_geom = aggregate_torque_hyperboloid(h, F_geom_h)
print(f"势能力矩: {torque_potential}, 范数: {np.linalg.norm(torque_potential)}")
print(f"几何力矩: {torque_geom}, 范数: {np.linalg.norm(torque_geom)}")
print(f"比例: {np.linalg.norm(torque_geom) / np.linalg.norm(torque_potential)}")

# 检查坐标转换因子
X, Y, T = h[:,0], h[:,1], h[:,2]
factor = 1.0 / (1.0 + T)
print(f"因子 1/(1+T): {factor}")
print(f"T 的范围: {T.min()} ~ {T.max()}")