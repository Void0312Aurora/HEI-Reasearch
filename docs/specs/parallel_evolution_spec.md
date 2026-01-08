# 技术规格书：基于并行分支演化的动力学选择机制 (Parallel Branch Evolution)

## 1. 背景与目标
当前训练受限于 Python 解释器的串行开销，GPU 利用率低（<20%）。
同时，动力学系统容易陷入局部极小值。
本方案旨在利用 GPU 的并行计算能力，通过在每个时间步并行演化多个“思维分支”（Mental Branches），并基于自由能（Free Energy）选择最佳分支，从而：
1.  **提升 GPU 利用率**：将串行的时间计算转化为并行的空间计算。
2.  **增强探索能力**：通过并行噪声扰动探索相空间。
3.  **加速收敛**：类似于粒子滤波（Particle Filter），筛选出高质量的轨迹进行梯度更新。

## 2. 核心算法：共振选择 (Resonant Selection)

在每个时间步 $t$：

1.  **扩展 (Expand)**:
    *   当前状态 $s_t$ 维度 $[B, D]$。
    *   复制 $K$ 份：$[B, K, D]$。
    *   注入噪声 $\xi \sim \mathcal{N}(0, \sigma)$ 到 $q$ 分量：$q_{t,k} = q_t + \xi_k$。
    *   这模拟了“思维发散”或“尝试不同微调”。

2.  **演化 (Evolve)**:
    *   将维度展平为 $[B \times K, D]$。
    *   执行动力学演化：$s_{t+1, k} = f(s_{t, k}, u_t)$。
    *   此步骤充分利用 GPU 算力。

3.  **评估 (Evaluate)**:
    *   计算每个分支的自由能 $F_k = V(q_k) + \beta KL + \gamma E_{pred}(q_k, y_t)$。
    *   注意：在训练阶段，我们可以利用 $y_t$ (Target) 来指导选择（Oracle Selection），帮助模型找到正确轨迹。

4.  **选择 (Select)**:
    *   对每个样本 $b$，选择 $F$ 最小的分支 $k^*$。
    *   $s_{t+1} = s_{t+1, k^*}$。
    *   维度恢复为 $[B, D]$。

## 3. 实现细节

### 3.1 配置变更 (Stage1Config)
*   `num_branches`: int = 16 (分支数量)
*   `branch_noise`: float = 0.02 (注入噪声标准差)
*   `batch_size`: 调整为 `total_capacity / num_branches`。例如若 GPU 能跑 8192，且 $K=16$，则 $B=512$。

### 3.2 代码变更 (`stage1_counting.py`)
*   重写 `train_loop` 或 `SequenceModel`。
*   由于 `torch.compile` 已禁用，直接在 Python 循环中实现分支逻辑。

## 4. 预期效果
*   **硬件**：GPU Compute 利用率显著上升。
*   **训练**：相同 Step 下，Loss 下降更快（因为实际上搜索了 K 倍的空间）。
*   **速度**：单步耗时可能略微增加（数据搬运），但“有效训练速度”（收敛速度）应大幅提升。
