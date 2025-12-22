

### 方案一：物理修正 — 移除人为的曲率缩放（立即实施）

**核心观点：** 既然你的模型建立在双曲空间上，势能井应当是**内禀的 (Intrinsic)**。一个在双曲空间中“固定宽度”的高斯井，其数学形式应仅依赖于双曲距离 ，而不应显式依赖于坐标 。

`hei/geometry.py` 中的 `uhp_distance_and_grad` 已经计算了正确的测地线距离 。你不需要再根据  调整 `width`。

**修改 `hei/potential.py`：**

删除或注释掉 `_curvature_scale` 的调用，并在计算势能时使用固定的 `self.width`（或 `base_width`）。

```python
# 修改前 (Current potential.py)
# curv_scale = _curvature_scale(z)
# width_eff = self.width / max(np.sqrt(curv_scale), 1e-6)
# wells = np.exp(-(dists**2) / (2 * width_eff**2))

# 修改后 (Proposed Fix)
# 完全移除 curv_scale，相信黎曼几何的内禀性
width_eff = self.width  # 使用常数宽度！
wells = np.exp(-(dists**2) / (2 * width_eff**2))

```

**物理效果：**

* 当粒子靠近边界 () 时，虽然  变小了，但在双曲度量下，它与势能中心的距离  会自然趋于无穷大（因为 ）。
* 势能  会自然衰减到 0（如果井在内部），或者平滑增加（如果井在边界）。
* **消除了  的刚度奇异性**，力的增长将变为对数级或线性级，而非指数级爆炸。

---

### 方案二：架构修正 — 切换至 Hyperboloid 坐标计算势能（推荐）

虽然移除了缩放，但在 UHP 坐标系下计算  附近的  仍然涉及 `log(small_number)`，存在浮点数精度风险。

既然你已经实现了 `group_integrator.py` 和 `hyperboloid.py`，最彻底的方法是在 **Hyperboloid (闵可夫斯基空间)** 中计算势能。

**优势：** Hyperboloid 模型  没有边界奇异性。边界对应 ，坐标数值大但平滑，不会出现除以零。

**实施步骤：**

1. **扩展 Potential 接口**：在 `PotentialOracle` 中增加支持 Hyperboloid 坐标的接口。
2. **重写势能计算**：使用 `hei.hyperboloid.hyperbolic_distance_hyperboloid` 直接计算距离。

**代码示例 (修改 `hei/potential.py`)：**

```python
from .hyperboloid import (
    disk_to_hyperboloid, 
    hyperbolic_distance_hyperboloid, 
    hyperbolic_distance_grad_hyperboloid
)

class HyperboloidGaussianPotential:
    def __init__(self, centers_disk, width=0.5):
        # 预先将中心转换为 Hyperboloid 坐标
        self.centers_h = disk_to_hyperboloid(centers_disk)
        self.width = width

    def potential(self, h_coords):
        # h_coords shape: (N, 3)
        V_sum = 0.0
        # 计算所有点到所有中心的双曲距离
        for c_h in self.centers_h:
            # d 是内禀双曲距离，无奇异性
            d = hyperbolic_distance_hyperboloid(h_coords, c_h)
            # 使用固定宽度，无需 rescaling
            V_sum += -np.sum(np.exp(-d**2 / (2 * self.width**2)))
        return V_sum
    
    def gradient(self, h_coords):
        # 计算梯度 (在切空间)
        # 需调用 hyperbolic_distance_grad_hyperboloid
        # ...
        pass

```

### 方案三：软化边界 (Soft Barrier / Regularization)

如果你必须保留 `_curvature_scale` 的逻辑（例如为了模拟某种特定的物理禁闭效应），你必须切断其与  的直接关联，设置一个安全截断。

**修改 `_curvature_scale`：**

```python
def _curvature_scale(z: np.ndarray) -> float:
    y = np.imag(np.asarray(z, dtype=np.complex128)).ravel()
    # 强制设置 y 的下限，例如 0.05
    # 这相当于在距离边界 0.05 的地方不再增加“刚度”
    y_clamped = np.maximum(y, 0.05) 
    return float(np.median(1.0 / (y_clamped * y_clamped)))

```

### 方案 A（最推荐的热修）：移除 `width_eff` 的曲率缩放，让“宽度”只在双曲距离尺度上定义

你势阱用的距离 `dists` 已经是**双曲距离**（`uhp_distance_and_grad(...)[0]`）。在这种定义下，“宽度”最自然的解释就是**双曲尺度上的 σ**，应当是常数：

* 把 `width_eff = self.width / sqrt(curv_scale)` 改为 `width_eff = self.width`
* 同样在 `dV_dz` 里保持一致 

这一步会直接消除你描述的“墙变无限硬”的主要来源（因为不会再人为把 σ→0）。

### 方案 B：对曲率缩放做“有界化 + 平滑化”，保留自适应但禁止发散

如果你确实需要曲率自适应（例如想让势阱在欧式投影上呈现某种尺度一致性），至少要保证 `curv_scale` 有界且时间上连续：

* **y 正则化**：把 `y = max(imag, 1e-6)`  提升到更物理的下界（如 `1e-3 ~ 1e-2`），或使用 `y_eff = sqrt(y^2 + y0^2)` 代替 y
* **硬上限**：`curv_scale = min(curv_scale, curv_max)`（例如 1e4 或更小，取决于 dt 与速度尺度）
* **EMA 平滑**：`curv_scale_t = (1-β) curv_scale_{t-1} + β curv_inst`，避免一帧内刚度跳变

这仍然是“补丁”，但能把爆炸从必然变成可控。

---

## 3) 结构性改造（推荐的“物理一致”版本）

你的原意看起来是“曲率越大，越应该强约束布局”。这可以实现，但不应通过让势阱宽度趋零来实现，而应通过**边界处有限斜率的约束项**，即让势能增长快但梯度不发散。

### 方案 C：把“防逃逸”交给一个边界势（Barrier），并保证梯度上界

你已经有“全局双曲谐振子先验” `V = 0.5*k*d(z,i)^2` ，而且它的 k 还被 `k_eff = prior_stiffness * _curvature_scale(z)` 放大 ；这同样可能造成硬化。建议改成两层：

1. **温和的全局先验**：k 常数或随 action 缓慢变化（不要随 1/y² 爆炸）
2. **显式边界 Barrier**：例如对 y 加一个“软障碍”

   * `V_bar(y) = α * softplus((y_min - y)/s)^2`（光滑、梯度有界）
   * 或 `V_bar(y)= α * log(1 + exp((y_min - y)/s))`

这样你得到的是“靠近边界能量迅速变大”，但不会出现“有限 dt 下无限冲量”。

### 方案 D：把曲率自适应从“宽度缩放”迁移到“势能权重缩放”，并做饱和

如果你想保留“曲率高→势能影响更强”，更安全的做法是：

* 保持 `width_eff = const`（双曲 σ 常数）
* 改成 `weight_eff = weight * f(curv_scale)`，其中 `f` 必须饱和，比如 `f(s)=sqrt(1+s/s0)` 或 `f(s)=tanh(s/s0)`
  这样改变“井的深度/影响力”，不会制造 σ→0 的无限刚度。

---

## 4) 配套数值策略（即便改了模型，也建议加，防止残余奇异）

1. **局部化而非全局化**：如果仍要用 `1/y^2` 类尺度，尽量改为 per-point 的局部尺度，不要用全局 median 去同步收缩所有点。当前 median 机制会把边界事故全局传播 、。
2. **基于力的 dt 限制**：你现在 dt 主要由速度/加速度/阻尼限制；当势场刚度突增时，应该额外用 `||F||` 或估计的 Lipschitz（近似 `1/width_eff^2`）给 dt 一个上界。否则即使势场已“有限”，也可能出现过冲。
3. **避免把 `y` 下界设到 1e-6**：你在 `_curvature_scale` 用了 1e-6 ，在 `clamp_uhp` 也默认 `min_im=1e-6` 。这在“含 1/y² 的模型”里几乎等同于允许 1e12 量级的系数出现。若保留任何 1/y² 项，应把这个下界提高到与你的 dt、速度、力的尺度相匹配的值。

---
