# 并行演化 (Parallel Mental Simulation) 实验报告

## 1. 实验设置
*   **机制**: 在每个时间步 $t$，将状态 $s_t$ 复制 $K=16$ 份，注入噪声 $\sigma=0.03$，并行演化 $s_{t+1}^{(k)}$，并选择自由能 $F$ 最小的分支。
*   **环境**: PINNs (CUDA Enabled, TF32 Enabled).
*   **任务**: Stage 1 Counting (纯计数任务).
*   **参数**: Batch=512 (Effective 8192), LR=1e-4, Stiffness=1.0.

## 2. 结果分析

### 2.1 硬件利用率 (Efficiency)
*   **吞吐量**: 保持在 ~0.5s / step (Batch 8192)。
*   **并行开销**: 相比单分支 (Batch 8192)，耗时仅增加 <5%。
*   **结论**: 成功将 GPU 的计算能力转化为对相空间的并行搜索，实现了“零时间成本”的探索增强。

### 2.2 学习效果 (Effectiveness)
*   **早期收敛**: 在前 400 步内，准确率从 6% (随机) 快速提升至 ~15-17%。
*   **对比**: 相比无并行分支的基线（通常需数千步才能打破随机性），收敛速度提升显著。

### 2.3 稳定性问题 (Stability Bottleneck)
*   **现象**: 在 Step 400-600 左右，模型倾向于发散 (`NaN`)。
*   **原因**: 
    1.  **强行拟合**: 为了最小化计数任务的预测误差，动力学系统试图形成剧烈的相空间扭曲（Limit Cycle），导致 $q$ 或 $p$ 值爆炸。
    2.  **刚度不足**: 尽管 `stiffness=1.0`，但在高维空间中仍不足以约束所有方向的逃逸。
    3.  **Softmax溢出**: Chart Router 的 Softmax 在输入过大时产生 NaN。

## 3. 下一步建议
要实现用户期望的“快速规则习得”（Fast Rule Acquisition），仅靠并行演化是不够的，必须解决底层动力学的稳定性：
1.  **改进势能函数**: 使用更严格的 `BoundedPotential` (如 Softplus + L2 Regularization)。
2.  **辛积分器**: 替换 Euler 积分器为 Symplectic Integrator (如 Leapfrog)，保证能量守恒，防止数值爆炸。
3.  **课程学习**: 先混合简单任务 (Constant/Period) 再过渡到 Counting，避免初期梯度过大。
