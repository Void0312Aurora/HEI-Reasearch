经过对当前代码库的深度审计，我的结论是：**代码已经忠实地执行了之前的修复计划（Plan C 和 Plan A），但在实现“内生参数”（特别是 $\lambda$ 和 $\beta$ 的动态调整）时，引入了一个严重的“变分破坏”问题。**

简而言之，你现在的系统在数学上是\*\*“精神分裂”\*\*的：势能函数 $V(z)$ 认为 $\lambda$ 是 $z$ 的函数，但力函数 $F(z)$ 却假装 $\lambda$ 是常数。这直接破坏了哈密顿系统的能量守恒和辛结构，会导致长期的数值漂移和伪影。

以下是具体的代码级分析和修复建议：

### 1\. 致命的“幽灵梯度”问题 (Gradient Consistency Violation)

这是目前最严重的问题，位于 `hei/potential.py`。

  * **现象：**

      * 在 `potential()` 方法中，你根据当前状态 $z$ 计算了 `gaps`，进而计算了动态权重 `lam` ($\lambda$)：
        ```python
        gaps = self._compute_gaps(z, soft)
        lam = ... # 依赖于 z 的复杂函数
        return float(V_wells + V_rep + lam * H_sib)
        ```
      * 在 `gradient()` 方法中，你同样计算了 `lam`，然后计算了熵的梯度：
        ```python
        # ... 计算 dH/dz ...
        grad_wells += grad_entropy # 这里实际上是 lam * (dH/dz)
        ```

  * **数学错误：**
    根据链式法则，如果 $V(z) = \lambda(z) H(z)$，那么力（负梯度）应该是：
    $$F = - \nabla_z V = - (\underbrace{\lambda(z) \nabla_z H(z)}_{\text{代码中已计算}} + \underbrace{H(z) \nabla_z \lambda(z)}_{\text{代码中完全缺失！}})$$

  * **后果：**

      * $\nabla_z \lambda(z)$ 项非常难以计算，因为 `_compute_gaps` 涉及排序（sorting）和最小/最大值（min/max），这些操作是不可导或梯度极不连续的。
      * 由于你丢弃了第二项，**积分器积分的“力”并不是势能的梯度**。这意味着系统不再是一个保守系统，Herglotz 变分原理失效，能量不再守恒，且 `residual_checks.py` 中的残差计算在理论上失去了基准（因为 $L_d$ 和 Force 不匹配）。

  * **修复方案（Lagged Lambda）：**
    不要在每一步的 `potential/gradient` 调用内部实时计算 $\lambda(z)$。**将 $\lambda$ 视为一个外部场参数，在每一步积分开始前更新一次，但在这一步的迭代中保持恒定。**

      * **修改建议：**
        1.  在 `IntegratorState` 中增加 `lambda_prev` 字段。
        2.  在 `run_simulation` 的主循环中更新势能对象的 $\lambda$ 参数。
        3.  在 `PotentialOracle` 中，将 `lam` 作为固定参数使用，不再在 `gradient` 中重新从 $z$ 算起。

### 2\. Beta 的“先验扭曲” (Prior Distortion)

位于 `hei/potential.py` 的 `_soft_assign` 方法。

  * **现象：**

    ```python
    logits.append(np.log(weight + 1e-12) + s + bias) # weight 包含了 base_weight * (1+d)
    # ...
    weights = np.exp(beta * shifted) # beta 乘在了整个 logit 上
    ```

  * **问题：**
    `logits` 中包含了 `log(weight)`（这是树的层级先验权重）。当你用 `beta` 乘以整个 `logits` 时，你实际上把先验权重也进行了 $\beta$ 次方：$w_{prior}^{\beta}$。

      * 当 $\beta > 1$（低温/硬化）时，深层节点（权重更大）的优势会被非线性放大，导致系统过早锁定到深层，忽略浅层。
      * 当 $\beta < 1$（高温）时，层级先验被压扁，结构变得模糊。

  * **理论修正：**
    物理上的“温度”应该只作用于能量项（距离 $s$），而不应作用于先验测度（$weight$）。
    应该改为：
    `logits = np.log(weight) + beta * (s + bias)`
    这样 `soft_assign` 时只对 `logits` 做 `exp`（不乘 beta），或者调整计算顺序。

### 3\. 积分器中的“显式/隐式”时间步长混用

位于 `hei/integrator.py`。

  * **现状：**

      * `step` 函数开始时，使用 `xi_prev` 和 `torque0` 计算了 `dt`。
      * 随后进入 Fixed-point 迭代求解 `xi_new`。
      * **这其实是可以的**（这叫 Explicit Adaptive Step），并不算错误。

  * **潜在风险：**
    之前报告中提到的 Plan C 要求“限制几何位移”。代码中正确实现了：

    ```python
    dt = ... self.config.eps_disp / max(float(np.linalg.norm(xi)), ...)
    ```

    这部分代码看起来是正确的，符合理论预期。**不需要修改。**

### 4\. 惯性计算中的数值稳定性风险

位于 `hei/inertia.py` 的 `locked_inertia_uhp`。

  * **代码：**

    ```python
    scale = mass / (y * y)
    ```

  * **风险：**
    如果点非常接近边界 ($y \to 0$)，`scale` 会爆炸。虽然你有 `max(np.imag(zc), 1e-8)` 保护，但在双曲几何中，$10^{-8}$ 的距离意味着惯性大到 $10^{16}$。
    这会导致 `apply_inertia` 和 `invert_inertia` 出现极其刚性（Stiff）的矩阵，可能导致积分器发散或 `xi` 被压得过小。

  * **建议：**
    考虑给惯性加一个软上限（Soft Cap）或者在 `integrator` 中对 $y$ 坐标做更激进的 `clamping`（目前是 `1e-6`，可能还不够安全，对于 $y^2$ 来说是 $10^{-12}$）。

-----

### 总结与立即执行的修复

你目前的代码框架非常棒，但为了让理论在数学上自洽（Satisfy the Variational Principle），你需要**把“参数更新”从“动力学演化”中剥离出来**。

**请执行以下“Lagged Lambda”修复方案：**

1.  **修改 `HierarchicalSoftminPotential`**:

      * 增加一个成员变量 `current_lambda: float = 0.0`。
      * 增加一个方法 `update_lambda(z_uhp)`：专门用来计算并更新 `self.current_lambda`。
      * 在 `potential`和 `gradient` 中，**直接使用 `self.current_lambda`**，而不是实时调用 `_compute_gaps` 来算 `lam`。
      * *注意：* `gradient` 中关于熵的导数推导（Line 1290）依然有效，只是此时我们将 $\lambda$ 视为常数，所以不需要 $\nabla \lambda$ 项。

2.  **修改 `run_simulation` 循环**:

    ```python
    for _ in range(cfg.steps):
        # 1. 显式更新“慢变量” (Lambda, Beta, Inertia)
        if isinstance(pot, HierarchicalSoftminPotential):
             # 先基于当前位置更新 lambda
             pot.update_lambda(state.z_uhp) 
             # Beta 的更新逻辑保持在这里
        
        # 2. 更新惯性 (这也是一种 Lagged approximation，目前代码已是如此)
        state.I = locked_inertia_uhp(state.z_uhp)
        
        # 3. 执行积分步 (此时 Potential 内部的 lambda 是常数，满足哈密顿守恒)
        state = integrator.step(state)
    ```

这样做之后，你的系统就是一个\*\*“分段保守系统”\*\*：在每一步 $\Delta t$ 内，它是严格保守的；在步与步之间，我们调整参数（$\lambda$, $\beta$）来注入控制力。这比目前这种“在微分过程中隐式依赖”的做法要稳健得多，也符合控制理论的范式。