"""验证力矩聚合修复效果"""
import numpy as np
from hei.diamond import aggregate_torque, aggregate_torque_hyperboloid
from hei.geometry import hyperboloid_to_uhp

# 创建测试数据
N = 5
np.random.seed(42)

# Hyperboloid 坐标
h = np.random.randn(N, 3)
h[:, 2] = np.abs(h[:, 2]) + 2.0  # 确保 T > 0

# 力
f_h = np.random.randn(N, 3) * 0.1

# 方法1：通过 Hyperboloid 聚合
torque_h = aggregate_torque_hyperboloid(h, f_h)

# 方法2：先转 UHP 再聚合（参考实现）
z_uhp = hyperboloid_to_uhp(h)
X, Y, T = h[:, 0], h[:, 1], h[:, 2]
fX, fY, fT = f_h[:, 0], f_h[:, 1], f_h[:, 2]
factor = 1.0 / (1.0 + T)
f_uhp = (fX + 1j * fY) * factor - z_uhp * fT * factor
torque_ref = aggregate_torque(z_uhp, f_uhp)

print("=== 验证力矩聚合一致性 ===\n")
print(f"Hyperboloid 聚合: {torque_h}")
print(f"UHP 参考聚合:     {torque_ref}")
print(f"相对误差:         {np.linalg.norm(torque_h - torque_ref) / (np.linalg.norm(torque_ref) + 1e-12):.2e}")

# 检查是否使用求和而非平均
single_point_h = h[0:1]
single_point_f = f_h[0:1]
torque_single = aggregate_torque_hyperboloid(single_point_h, single_point_f)

multi_identical = np.tile(single_point_h, (N, 1))
multi_identical_f = np.tile(single_point_f, (N, 1))
torque_multi = aggregate_torque_hyperboloid(multi_identical, multi_identical_f)

print(f"\n=== 验证求和 vs 平均 ===\n")
print(f"单点力矩:          {torque_single}")
print(f"{N}个相同点力矩:    {torque_multi}")
print(f"比值:              {np.linalg.norm(torque_multi) / np.linalg.norm(torque_single):.2f}")
print(f"期望比值（求和）:  {N}")

if abs(np.linalg.norm(torque_multi) / np.linalg.norm(torque_single) - N) < 0.01:
    print("\n✓ 修复成功：使用求和聚合")
else:
    print("\n✗ 修复失败：仍在使用平均")
