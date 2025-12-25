基于你们 2025-12-22 这次更新后的 `hei_n` 代码，我的结论是：**相比上一版，`hei_n` 的“理论对齐度”确实提升了（尤其是引入几何力/可变惯性接口、补全了 `GroupIntegratorN` 主干与更完整的势场模块），但整体仍然是验证性实现，和 `hei` 的“结构保持 + 稳定性工程化”版本差距依然明显**。下面按“本次更新带来的变化 / 仍缺什么 / 已实现代码的关键风险与修复建议”来重新评估。

---

## 1) 这次更新让 `hei_n` 明显更完整的点

### 1.1 `ContactIntegratorN` 已把“几何力 + Diamond 扭矩 + z(Herglotz)”串成闭环

新版 `ContactIntegratorN.step()` 明确计算势能与梯度、从惯性模型取几何力、做切空间投影、用 Diamond 生成体坐标力矩，并更新接触变量 z：`z_new = z + (T - V - gamma*z) dt` 。
这意味着 Phase 1 里你们强调的“结构反馈（Diamond）+ 可变惯性 + 接触耗散”在 `hei_n` 侧已经具备一条可跑通的主路径。

同时 `inertia_n.py` 把惯性模型抽象成协议（mass / inverse_inertia / kinetic_energy / geometric_force），并且 `RadialInertia.geometric_force()` 给了一个可用的 F_geom 实现（尽管仍有符号/语义需要澄清，见第 3 节）。

### 1.2 `hei_n` 新增（或补齐）了 `GroupIntegratorN`

`integrator_n.py` 现在给出 `GroupIntegratorN`：在 SO(1,n) 上用矩阵表示做 “Force → Torque → 更新 M → 用 exp 更新 G” 的最小版群积分流程 ，并且实现了一个明确的 `_body_torque`（本质上仍是 “e0 wedge F_body” 的 boost-only 力矩提升）。

这对你们后续把 `ContactIntegratorN` 做成“在 GroupIntegratorN 之上叠 z”是有帮助的（目前 docstring 也这么写）。

### 1.3 双曲几何梯度的数值夹取改进了一半

`dist_grad_n` 已经把 inner 夹到了 `<= -1 - 1e-9`，避免 `arccosh` 和分母在 d≈0 的奇异 。这是正确方向。

---

## 2) 仍然缺失的核心能力（也是 `hei` 与 `hei_n` 的主要差距来源）

### 2.1 结构保持离散（Cayley/共伴随输运/半隐式势场/自适应 dt）仍未进入 `hei_n`

* `ContactIntegratorN` 仍是显式 Euler：`M_new = M + (Torque - gamma*M) dt`，`z_new` 也是 Euler 。
* `GroupIntegratorN` 同样是显式 Euler 的 M 更新 。
  相比之下，`hei` 的群接触积分器实现了 Cayley 型耗散因子、共伴随输运、力矩裁剪、自适应步长、周期性重正规化等稳定性与结构保持机制（你之前也观察到 `hei_n` 少这些）。

**直接影响**：`hei_n` 现在更容易出现“步长稍大就爆、长时间漂移、边界/刚性势下不稳定”，而 `hei` 的那套工程化措施正是为了解决这些。

### 2.2 `ad*`/共伴随输运仍被“简化掉”，短期可用，长期会成为硬缺口

`ContactIntegratorN` 明确写了“[M,M]=0，所以 dM/dt = Torque - gamma*M；full tensor inertia 才需要 ad*” 。
`GroupIntegratorN` 也用同样逻辑把 `ad*` 视为可忽略 。

**关键点在于**：你们当前 `RadialInertia` 属于“标量质量缩放”，会导致 `xi = M / m(x)` 与 `M` 仍然同向（逐粒子标量倍数），因此 `[M,xi]=0`，`ad*_{xi}M` 仍确实为 0——所以这个简化在“当前验证性模型”里成立。
但一旦你们落实计划里更强的“结构反馈/各向异性惯性（full tensor inertia）”，就必须实现 `ad*`/共伴随输运，否则理论与实现会立刻分叉。

---

## 3) 更新后 `hei_n` 里依然存在、且会影响实验可信度的高风险问题

### 3.1 `dist_n` 的夹取方向仍然是错的（可能直接产生 NaN）

`dist_n` 仍是：`inner = np.minimum(inner, -1.0 + eps)` 。
对双曲面点，理论应满足 `<x,y>_M <= -1`。数值漂移时常见的是 `<x,y>_M` **略大于 -1**（例如 -0.99999999），这时你应该把 inner 夹到 `<= -1 - eps`，保证 `-inner >= 1` 才能安全 `arccosh`。现在这个写法可能把 inner 夹到 -0.9999999，导致 `arccosh(0.9999999)` 直接 NaN。

**建议修复**：把 `-1.0 + eps` 改为 `-1.0 - eps`，或者像 `PairwisePotentialN` 那样统一走 `val = np.maximum(-inner, 1.0 + 1e-7)` 的模式 。

### 3.2 `HarmonicPriorN.gradient` 的递归 bug 仍在

当 `e0 is not None` 时仍然 `return self.gradient(x)`，会无限递归 。
如果你们后续用非默认原点/锚点（这在语义任务中很常见），这里会直接炸栈。

**建议修复**：实现 `dist_grad_n(x, e0)` 路径，或至少 `raise NotImplementedError`，不要递归自调用。

### 3.3 “几何力”在 `ContactIntegratorN` 里的语义命名与符号容易误用（建议立刻澄清）

* `InertiaModelN.geometric_force()` 的注释写的是 “Compute F_geom = -grad T” ，也就是返回“力”。
* `ContactIntegratorN` 里却把它命名为 `grad_geom` 并做 `grad_total = grad_potential - grad_geom`，随后 `Force_world = - project_to_tangent(..., grad_total)` 。

按现在代码的代数关系，这样写在数值上等价于 `Force = -gradV + F_geom`，即“势能力 + 几何力”叠加；**结果是对的**，但命名会导致团队成员很容易把它当成“梯度”而不是“力”，未来替换惯性模型时非常容易引入符号错误。

**建议修复**：

* 把变量名从 `grad_geom` 改成 `force_geom`；
* 把 `grad_total` 改成 `gradV_minus_forceGeom` 或者直接显式写 `Force_world = -proj(gradV) + proj(force_geom)`，避免歧义。

### 3.4 Diamond 扭矩仍是 boost-only（旋转部分恒为 0），会限制可表达动力学

`compute_diamond_torque()` 只填了 (0,i) 与 (i,0) 块 ；`GroupIntegratorN._body_torque` 也做同样假设 。
这对“以 e0 为楔基向量”的最简模型是合理的，但它意味着：你们的力矩不产生 so(n) 旋转生成元，系统更像“纯 boost 的双曲平移动力学”，而不是完整的 (SO(1,n)) 刚体/协变动力学。

**短期**：用于验证 CCD + z + 标量惯性是够的。
**中期**：如果你们希望“结构反馈”更像真实的“内部自旋/旋转自由度”，需要扩展 wedge 结构或允许 torque 产生旋转块。

---

## 4) 重新评估结论与优先级建议

### 当前合理性与可行性

* Phase 1（验证 z + 耗散 + Diamond + 可变惯性接口）在 `hei_n` 侧比之前更可行，主链路已经齐了 。
* 但作为“可复现实验平台”，你们必须先修掉 **dist_n NaN 风险**  和 **HarmonicPriorN 递归** ，否则很多现象会被数值错误污染。

### 建议的最小修复清单（按优先级）

1. 修 `dist_n` 的 clamp（P0）
2. 修 `HarmonicPriorN.gradient` 递归（P0）
3. 把几何力相关变量命名与注释统一为“force”语义（P1）
4. 给 `ContactIntegratorN` 增加最小稳定性钩子：`torque_clip / xi_clip / adaptive_dt / periodic projection`（P1）
5. 若下一步要做 full tensor inertia：预留 `ad*`/共伴随输运接口（P2），避免未来重构成本爆炸 


