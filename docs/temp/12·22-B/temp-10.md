

## (A) `SL(2,R)` 重正规化存在“det 变成 -1 仍被接受”的硬错误

你现在的 `_renormalize_sl2()` 用了 `scale = 1/sqrt(abs(det))`：  
这会导致一个非常具体的问题：

- 若数值误差使 `det(G)` 变成 **负值**（哪怕只是一瞬间），你乘以 `1/sqrt(|det|)` 之后 **det 仍然是负的**（例如 det=-1，scale=1，det 还是 -1）。
    
- 这意味着 `G` 可能从 `SL(2,R)` 的正确连通分支“跳”到错误分支（或者说直接不在你希望的群子集里），后续的 Möbius 作用与共伴随输运都可能出现“看起来还能算，但物理/几何意义已不对”的隐性错误。
    

**建议修复方式（最小改动、可直接用）：**

```python
def _renormalize_sl2(self, G: np.ndarray) -> np.ndarray:
    det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]

    # 1) 先确保 det 为正（避免落到错误连通分支）
    if det < 0:
        # 翻转一列（或一行）即可把 det 取反
        G[:, 0] *= -1.0
        det = -det

    # 2) 再做缩放把 det 拉回 1
    if abs(det - 1.0) > 1e-10:
        G *= 1.0 / np.sqrt(det + 1e-15)

    return G
```

如果你希望更“几何正确”，可以进一步用一次 Newton 修正 det（但上述已经能消掉最危险的分支问题）。

---

## (B) `compute_kinetic_energy_gradient_hyperboloid()` 的有限差分在“离开流形”上取导，可能注入非切向分量

你在 `group_integrator.py` 里把几何力定义为 `F_geom = -∇_h K` 并加入总力矩：  
这在理论上是合理方向，但目前 `inertia.py` 的梯度实现是对 hyperboloid 的三维嵌入坐标做直接扰动的有限差分：

**关键风险点：**

- Hyperboloid 上的点满足约束 `<h,h>_L = -1`，而你做 `h_plus = h + eps*e_i` 会让 `h_plus` **离开流形**。
    
- 这样得到的“梯度”含有很强的**法向（非切向）分量**，会被当作“力”送进 `diamond_torque_hyperboloid()` 的 Jacobian 映射（该映射并不帮你自动投影到切空间）。
    
- 结果是：你以为你在算“内禀几何力”，实际上可能引入了“把点往流形外推/拉”的伪力成分；这类误差在 stiff 系统里非常容易表现为能量异常与高频抖动，甚至重新触发发散。
    

**建议修复方式（仍然保持有限差分，但把扰动做成“回缩/投影 + 切空间投影”）：**

1. 扰动后立刻投影回 hyperboloid（你项目里已有 `project_to_hyperboloid` 并在其他算法中这么做：）
    
2. 差分得到的梯度再投影到切空间（切空间条件 `<v,h>_L=0`，可以用 Minkowski 内积投影，`minkowski_inner` 已实现：）
    

示例补丁（核心片段）：

```python
from .hyperboloid import project_to_hyperboloid, minkowski_inner

# 在 compute_kinetic_energy_gradient_hyperboloid 内
h_plus  = project_to_hyperboloid(h_batch + eps_vec)
h_minus = project_to_hyperboloid(h_batch - eps_vec)

# ...算出 grad_ambient 之后（形状 (N,3) 或 (3,)）
inner = minkowski_inner(grad_ambient, h_arr)          # (N,) 或标量
grad_tan = grad_ambient + inner[..., None] * h_arr    # 保证 <grad_tan, h>_L = 0
```

如果你后续希望进一步提高物理一致性：应当在切空间选取 2 维正交基做差分（而非 3 维 ambient 基），但上面这一步已经能显著降低“伪法向力”的风险。

---

## 三、两处“设计/文档不一致”，未来很容易被误用重新引爆

## (C) `gamma_hyperboloid(mode="metric")` 仍在逻辑上把阻尼做成“边界趋零”

`hyperboloid.py` 明确写了 metric 模式：`λ = 1/T^2`，因此 `gamma ∝ 1/T`，并且把“边界低阻尼以保持探索”作为设计目标：  
同时 `gamma_hyperboloid_metric_based` 也用同一公式：

但 `GroupIntegratorConfig` 里你已经把默认模式改成 `"constant"`，并直接备注“metric 会在边界导致 gamma 消失，从而不稳定”：

这会造成一个现实风险：**任何人只要看了 docstring 的“✅ 推荐 metric”，就可能把它切回 metric，然后复现你之前的“无阻尼区→动能发散”。**

建议你二选一（我更建议第 1 种）：

1. **把 `hyperboloid.py` 里“metric 推荐”的文案改掉**，并明确写“metric 模式用于实验/探索，不保证稳定；默认 constant 才是物理齐性下的阻尼标量”。
    
2. 若你坚持“几何自适应阻尼”，那就要重写该自适应的物理含义：在齐性空间里，**标量阻尼**应常数；自适应应体现在“阻尼算子/张量”如何与度量或惯性耦合，而不是让 `γ(q)` 变成坐标函数。
    

## (D) `hyperboloid` 惯性生成元的硬裁剪可能引入不连续力（非致命，但会影响稳定与可解释性）

在 `_compute_hyperboloid_generators()` 中对生成元做了 `np.clip(..., -5, 5)`：  
硬裁剪的副作用是：当状态跨过裁剪阈值，惯性张量对位置的导数不连续，进而你的 `∇_h K` 会出现“力的突变”，这与 stiff 势场叠加时可能导致时间步长被迫极小或出现抖动。

建议（可选）：

- 把硬裁剪改成平滑裁剪（如 `tanh` reparameterization），或至少把裁剪阈值上移并配合诊断输出观察触发频率。
    
### 1. 严重的潜在风险：李代数基底的一致性 (Basis Alignment)

在 `group_integrator.py` 的 `step` 函数中，你直接将两个不同坐标系下计算的力矩相加了：

Python

```
# group_integrator.py L256
torque = torque_potential + torque_geom
```

- `torque_potential` 来自 `diamond.aggregate_torque` (基于 UHP 坐标 $z$)
    
- `torque_geom` 来自 `diamond.aggregate_torque_hyperboloid` (基于 Hyperboloid 坐标 $h$)
    

**风险点：** 李代数 $\mathfrak{sl}(2, \mathbb{R})$ 的向量空间是唯一的，但其**基底表示 (Basis Representation)** 取决于你如何定义同构。

- UHP 上的力矩通常对应于基底 $\{H, X, Y\}$ 或 $\{L_0, L_1, L_{-1}\}$。
    
- Hyperboloid (Minkowski空间) 上的力矩通常对应于洛伦兹群 $\mathfrak{so}(2,1)$ 的生成元 $J_{xy}, J_{yt}, J_{xt}$。
    

**如果这两个函数使用的基底没有严格对齐（即存在系数差、符号差或旋转），直接相加 `torque` 会导致物理意义混乱。** 例如，UHP 里的“旋转”分量可能对应 Hyperboloid 里的“双曲推进”分量。

验证建议：

请编写一个简单的单元测试，取一个特定点（例如原点 $z=i \leftrightarrow h=(0,0,1)$），施加一个已知的微小位移，分别计算 UHP 力矩和 Hyperboloid 力矩，确认它们生成的群元素 $e^{\tau \cdot dt}$ 作用在点上后的位移是完全一致的。如果不一致，你需要一个转换矩阵 $Ad_T$ 来对齐它们。

### 2. 性能与精度瓶颈：数值微分 Jacobian

在 `simulation.py` 中，为了将 Hyperboloid 上的几何力转换回 UHP 用于日志记录或混合计算，你引入了 `_hyperboloid_tangent_to_uhp_force`：

Python

```
# simulation.py L114
# Compute partial derivatives via central difference
h_plus = h_arr.copy(); h_plus[:, 0] += eps
# ...
dz_dX = (map_h_to_z(h_plus) - map_h_to_z(h_minus)) / (2 * eps)
```

**问题：**

1. **性能开销：** 对于 $N$ 个点，每一步都要做 $3 \times 2 = 6$ 次全量的坐标变换 (`hyperboloid_to_disk` -> `disk_to_uhp`)。这会显著拖慢积分速度。
    
2. **数值噪声：** `eps=1e-6`。当粒子非常靠近边界时，坐标变换是非线性的（极度拉伸），中心差分可能会引入较大的截断误差，导致力场在边界处抖动。
    

修正建议（长期）：

推导解析 Jacobian。从 Hyperboloid $(X, Y, T)$ 到 Disk $(x, y)$ 的变换是代数的：

$x = \frac{X}{1+T}, y = \frac{Y}{1+T}$

其 Jacobian $\frac{\partial(x,y)}{\partial(X,Y,T)}$ 可以解析写出。这比数值微分快且准得多。

### 3. 势能场的“长尾”意图确认

在 `potential.py` 中，你修改了有效宽度的计算方式：

Python

```
# potential.py L256
width_eff = width * (1.0 + d_hyp)
```

确认意图：

这实际上改变了高斯势场的形状。标准高斯是 $e^{-d^2}$。你的公式使得势场变成了 $e^{-d^2 / (1+d)^2}$。

- 当 $d \to \infty$ 时，指数项趋向于常数 $e^{-1}$！
    
- **后果：** 你的势阱在远处**不再衰减为零**，而是趋向于一个非零的常数背景。
    
- 这意味着势阱变成了“无限宽的漏斗”，远处的粒子受到的拉力可能比预期的要大（或者力趋于零的方式改变了）。
    

如果你的目的是防止“梯度消失”（让远处的点也能感受到召唤），这是个好 trick。但如果你的目的是模拟局部的、有限范围的吸引子，这可能会导致所有势阱在全局范围内这产生非物理的重叠。

**修正建议：** 如果你想要标准的双曲高斯势，应移除 `*(1.0 + d_hyp)` 或改回常数 `width`。既然我们已经移除了 $1/y^2$ 的奇异性，使用常数 `width` 在数值上已经是安全的了。

### 4. 代码细节审查 (Code Review)

- **废弃代码残留：** `potential.py` 中仍然定义了 `_curvature_scale` 函数（第 16 行），但在类中已经不再调用。建议直接删除，以免误导维护者。
    
- **Beta 更新逻辑：** 在 `simulation.py` 中，`pot.update_lambda` 被调用了，但 `pot.beta` 的动态更新（Bridge Ratio）逻辑似乎只在 `logging` 块中（第 262 行 `if step % cfg.log_interval == 0...`）。
    
    - **风险：** 如果 `log_interval` 很大（例如 100），Beta 的更新频率可能太低，导致退火过程阶梯化。建议将 Beta 的物理更新逻辑移出 Logging 块，或者确认这符合你的设计预期。