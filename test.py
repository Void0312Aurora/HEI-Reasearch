"""
CCD Diagnostic Suite: Isolating the 'Right-Side Convergence' & Explosion bugs.
"""
import numpy as np
import matplotlib.pyplot as plt
from hei.geometry import cayley_disk_to_uhp, cayley_uhp_to_disk, uhp_distance_and_grad
from hei.inertia import locked_inertia_uhp
from hei.potential import GaussianWellsPotential
from hei.integrator import ContactSplittingIntegrator, IntegratorConfig, IntegratorState
from hei.lie import exp_sl2

def test_0_geometry_boundary():
    print("\n=== Test 0: Geometry Boundary Stability ===")
    # 测试点逼近 z=1 (右侧边界)
    z_disk_near_1 = np.array([0.9, 0.99, 0.999, 0.9999]) + 0j
    z_uhp = cayley_disk_to_uhp(z_disk_near_1)
    
    print(f"{'z_disk':<15} | {'z_uhp (imag)':<15} | {'Reconstruction Error'}")
    print("-" * 50)
    for zd, zu in zip(z_disk_near_1, z_uhp):
        z_back = cayley_uhp_to_disk(zu)
        err = np.abs(zd - z_back)
        print(f"{zd.real:<15.4f} | {zu.imag:<15.4e} | {err:<15.4e}")
        
    # 诊断：如果 z_uhp.imag 爆炸得太快导致 float64 溢出，或者回构误差大，说明几何层就崩了。
    # 预期：z_uhp.imag 应该随着 z -> 1 线性或双曲增长，但不应出现 NaN。

def test_1_force_direction():
    print("\n=== Test 1: Force Direction Sanity Check ===")
    # 场景：粒子在原点 (0,0)，目标在左边 (-0.5, 0)。
    # 物理直觉：力应该指向左边（实部为负）。
    center = -0.5 + 0j
    particle = 0.0 + 0j
    
    # 转换到 UHP
    center_uhp = cayley_disk_to_uhp(center)
    particle_uhp = cayley_disk_to_uhp(particle) # 应该是 1j
    
    # 定义简单的势能井
    pot = GaussianWellsPotential(centers=np.array([center_uhp]), width=0.5)
    
    # 计算梯度
    grad_uhp = pot.dV_dz(np.array([particle_uhp]))[0]
    
    # 关键：我们需要把 UHP 上的力映回 Disk 看看方向对不对
    # 这是一个定性检查。在 UHP 原点 (1j) 处，向左的力对应什么？
    # 简单的判断：势能应该减少。
    
    # 尝试沿梯度方向走一小步
    eps = 1e-3
    next_uhp = particle_uhp - eps * grad_uhp # 梯度下降方向
    next_disk = cayley_uhp_to_disk(next_uhp)
    
    print(f"Target (Disk): {center.real}")
    print(f"Start  (Disk): {particle.real}")
    print(f"Step   (Disk): {next_disk.real:.6f}")
    
    direction = np.sign(next_disk.real - particle.real)
    print(f"Force pushes: {'RIGHT (WRONG!)' if direction > 0 else 'LEFT (CORRECT)'}")
    
    # 诊断：如果这里显示 RIGHT，说明梯度算反了，或者坐标变换的导数推导有误。
    # 这将直接解释为什么点会聚在右边。

def test_2_inertia_at_infinity():
    print("\n=== Test 2: Inertia Singularity at z -> 1 ===")
    # 检查当点靠近右边界时，惯性矩阵是否奇异或过小
    z_edge = cayley_disk_to_uhp(0.99 + 0j) # y 很大
    I = locked_inertia_uhp([z_edge])
    
    eigvals = np.linalg.eigvals(I)
    print(f"z_disk = 0.99 => z_uhp = {z_edge:.2f}")
    print(f"Inertia Eigenvalues: {eigvals}")
    print(f"Determinant: {np.linalg.det(I):.4e}")
    
    # 诊断：如果特征值极小（接近我们加的 eps），说明物理惯性已经消失。
    # 如果这时受到非零的力，加速度 a = F/m 将趋于无穷。

def test_3_free_particle_drift():
    print("\n=== Test 3: Free Particle Integrator Drift ===")
    # 这是一个“真空”测试。没有势能，没有耗散。
    # 粒子应该沿测地线运动，能量守恒。
    
    z0 = cayley_disk_to_uhp(0.0 + 0j)
    xi0 = np.array([1.0, 0.0, 0.0]) # 纯旋转/缩放流
    
    # 构造无势能、无耗散的积分器
    dummy_force = lambda z, a: np.zeros_like(z, dtype=np.complex128)
    dummy_pot = lambda z, a: 0.0
    dummy_gamma = lambda z: 0.0
    
    integrator = ContactSplittingIntegrator(
        force_fn=dummy_force, 
        potential_fn=dummy_pot, 
        gamma_fn=dummy_gamma,
        config=IntegratorConfig(fixed_point_iters=2)
    )
    
    # 运行 100 步
    state = IntegratorState(
        z_uhp=np.array([z0]), 
        xi=xi0, 
        I=locked_inertia_uhp([z0])
    )
    
    energies = []
    for _ in range(100):
        # 显式更新惯性 (模拟主循环)
        state.I = locked_inertia_uhp(state.z_uhp)
        state = integrator.step(state)
        # E = 0.5 * xi^T I xi
        E = 0.5 * state.xi @ (state.I @ state.xi)
        energies.append(E)
        
    energies = np.array(energies)
    drift = (energies.max() - energies.min()) / energies[0]
    print(f"Initial Energy: {energies[0]:.4e}")
    print(f"Max Drift %: {drift*100:.4f}%")
    
    # 诊断：如果自由粒子的能量都在剧烈漂移，说明积分器本身的离散化公式有问题，
    # 或者 exp_sl2 的实现精度不够。

if __name__ == "__main__":
    try:
        test_0_geometry_boundary()
        test_1_force_direction()
        test_2_inertia_at_infinity()
        test_3_free_particle_drift()
    except Exception as e:
        print(f"\n!!! CRITICAL FAILURE !!!\n{e}")
        import traceback
        traceback.print_exc()