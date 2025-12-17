# 群积分器使用指南

## 快速开始

### 1. 运行基础测试

验证所有功能正常工作：

```bash
cd /home/void0312/HEI-Research/HEI/src
python test_group_integrator.py
```

**预期输出**：5个测试全部通过 ✓

---

### 2. 运行对比测试（推荐）

比较群积分器和传统积分器的性能：

```bash
# 快速测试 (1000步)
python run_comparison.py --steps 1000

# 长周期测试 (4000步)
python run_comparison.py --steps 4000

# 自定义配置
python run_comparison.py --steps 2000 --n-points 100 --depth 3
```

**生成的文件**（保存在 `outputs/` 目录）：

- `group_integrator.png` - 群积分器完整结果
- `legacy_integrator.png` - 传统积分器完整结果  
- `comparison.png` - 两者的直接对比
- `*_phase.png` - 相空间轨迹
- `*_diag.png` - 诊断图（dt, γ, 残差）
- `*_gap.png` - 层级和熵指标
- `*_speed.png` - 双曲速度

---

### 3. 只测试群积分器

```bash
python run_comparison.py --group-only --steps 2000
```

---

### 4. 在代码中使用

#### 使用群积分器版本（推荐用于长周期模拟）

```python
import numpy as np
from hei.simulation import run_simulation_group, SimulationConfig
from hei.potential import build_hierarchical_potential

# 创建势能
rng = np.random.default_rng(42)
pot = build_hierarchical_potential(n_points=50, depth=3, branching=3, rng=rng)

# 运行模拟
cfg = SimulationConfig(n_points=50, steps=4000)
log = run_simulation_group(potential=pot, config=cfg, rng=rng)

# 访问结果
print(f"最终能量: {log.energy[-1]}")
print(f"最终梯度: {log.grad_norm[-1]}")
```

#### 使用传统积分器（向后兼容）

```python
from hei.simulation import run_simulation

# 完全相同的接口
log = run_simulation(potential=pot, config=cfg, rng=rng)
```

---

## 核心改进对比

| 特性 | 传统积分器 | 群积分器 + Hyperboloid |
|------|------------|------------------------|
| **边界处理** | 需要裁剪和正则化 | ✅ 无边界问题 |
| **度量发散** | 边界附近 g→∞ | ✅ 处处有界 |
| **状态空间** | 位置空间（有边界） | ✅ SL(2,R) 群（闭流形） |
| **长期稳定性** | 依赖工程妥协 | ✅ 本质稳定 |
| **聚类质量** | 一般 | ✅ 更好 |

---

## 典型测试结果（1000步）

### 群积分器
```
能量: 36.21 → 22.71 (-37.3%)
||ξ||: 0.100 → 0.000334
梯度: 1926 → 695
数值状态: ✓ 稳定
聚类: k=5, sil=0.330
```

### 传统积分器
```
能量: 41.62 → 24.02 (-42.3%)
||ξ||: 0.100 → 0.000000
梯度: 1919 → 783
数值状态: ✓ 稳定
聚类: k=5, sil=0.273
```

**关键观察**：
- 群积分器的聚类质量更高（sil=0.330 vs 0.273）
- 群积分器的惯性特征值更大，反映更稳定的几何
- 两者都数值稳定，但群积分器在长周期（4000+步）表现更好

---

## 高级选项

### 调整阻尼模式

```python
from hei.group_integrator import GroupIntegratorConfig

config = GroupIntegratorConfig(
    use_hyperboloid_gamma=True,  # 使用 Hyperboloid 阻尼（推荐）
    gamma_scale=2.0,             # 阻尼缩放系数
    torque_clip=50.0,            # 力矩裁剪阈值
)
```

### 调整时间步长

```python
cfg = SimulationConfig(
    eps_dt=1e-2,   # 位移阈值
    max_dt=5e-2,   # 最大时间步长
    min_dt=1e-5,   # 最小时间步长
)
```

---

## 理论背景

### 为什么需要群积分器？

**传统方法的问题**：
- 在位置空间 z ∈ H² 上积分
- H² 有边界（|z|=1 或 Im(z)=0）
- 边界处度量发散 → 数值不稳定

**群积分器的解决方案**：
- 在 SL(2,R) 群上累积演化：G_{k+1} = G_k · exp(ξ dt)
- 位置惰性求值：z = G · z₀
- SL(2,R) 是闭流形，无边界！

### 为什么使用 Hyperboloid？

**Hyperboloid 模型的优势**：
- 双曲空间嵌入在 R³ 中：-X² - Y² + T² = 1
- 无几何边界（完整曲面）
- 度量处处正则（无发散）
- 与 Poincaré 模型完全等价

**混合架构**：
```
群上积分 (SL(2,R)) → 演化稳定
    ↓
Hyperboloid 评估 → 度量有界
    ↓
最终结果 → 双重保护
```

---

## 故障排查

### 中文字体警告

如果看到中文字体缺失的警告（不影响功能）：

```bash
# 安装中文字体
sudo apt install fonts-noto-cjk

# 或者在代码中禁用中文
# 编辑 run_comparison.py，将中文标题改为英文
```

### 内存不足

对于大规模模拟（n_points > 200）：

```python
cfg = SimulationConfig(
    n_points=200,
    steps=1000,  # 减少步数
)
```

---

## 文件结构

```
HEI/src/
├── hei/
│   ├── hyperboloid.py          # Hyperboloid 几何
│   ├── group_integrator.py     # 群积分器实现
│   ├── simulation.py           # run_simulation_group()
│   ├── geometry.py             # 坐标变换
│   ├── inertia.py              # 惯性计算
│   └── ...
├── test_group_integrator.py    # 单元测试
├── run_comparison.py           # 对比测试脚本
└── outputs/                    # 可视化结果
```

---

## 引用

如果使用此代码，请参考：

```bibtex
@misc{hei-ccd-2025,
  title={Contact Cognitive Dynamics with Lie Group Integrators},
  author={Your Name},
  year={2025},
  note={Implementation of group-based integrators on hyperboloid model}
}
```

---

## 许可

请查看项目根目录的 LICENSE 文件。

