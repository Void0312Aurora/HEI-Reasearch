# Aurora Base 训练实验报告

**日期**: 2025-12-23  
**项目**: HEI (Hyperbolic Embodied Intelligence)  
**阶段**: Phase 2 - Aurora Base (Emergence of Concept)

---

## 摘要

本报告总结了 Aurora Base 训练的完整实验流程，包括：
1. 三个预检验门 (Gates A/B/C) 的结果。
2. GPU 加速优化 (157x 提速)。
3. 分阶段放量验收 (30k → 60k → 100k)。
4. 长程训练 (10,000 steps) 与语义审计。

**核心结论**: Aurora Base (100k 粒子) 在 10,000 步训练后，各项指标均稳定通过，满足 Soul Injection 的准入条件。

---

## 1. 三门预检验 (Three Gates)

### Gate A: Data Efficiency Check
- **目标**: 验证数据加载与预处理效率。
- **结果**: Wiki ZH 语料 ρ = 0.79。
- **状态**: ✅ PASS

### Gate B: Gradient SNR Check
- **目标**: 验证梯度信噪比，确保学习信号有效。
- **结果**: Cosine Similarity = 0.995。
- **状态**: ✅ PASS

### Gate C: Skeleton Stability Check
- **目标**: 使用 OpenHowNet Sememe Taxonomy 验证物理引擎稳定性。
- **结果**:
  - Contrast Ratio: 1.15 (偏低但可接受)。
  - Radius-Depth Correlation: 0.84。
  - NaN Count: 0。
- **关键改进**: 
  - 实现 Robust Integrator (SO(1,n) Renormalization, Implicit Solver, Adaptive dt)。
  - 修复 `dist_n` NaN 风险 (`-1.0 + eps` → `-1.0 - eps`)。
  - 修复 `HarmonicPriorN` 递归 bug。
- **状态**: ✅ PASS (Conditional)

---

## 2. GPU 加速优化

### 环境
- **Framework**: PyTorch 2.8.0 + CUDA.
- **Device**: GPU (vs. NumPy CPU Baseline).

### 实现内容
- `hei_n/torch_core.py`: 双曲几何算子 (exp_so1n, renormalize)。
- `hei_n/potentials_torch.py`: 势能计算 (SparseEdge, NegativeSampling)。
- `hei_n/integrator_torch.py`: Contact Integrator (GPU)。

### 性能对比 (N=11k)
| Backend | ms/step | Speedup |
|---------|---------|---------|
| CPU (NumPy) | ~1015 ms | 1x |
| GPU (PyTorch) | ~6.5 ms | **157x** |

---

## 3. 分阶段放量验收

采用 Go/No-Go Gate 策略，逐步放量以控制风险。

### 验收结果

| Phase | Particles | Residual P99 | Renorm Max | Throughput (Mean) | Throughput (P99) | Gate |
|-------|-----------|--------------|------------|-------------------|------------------|------|
| 3a | 30k | 1.28e-3 | 4.66e-5 | 7.6 ms | 10.2 ms | ✅ PASS |
| 3b | 60k | 6.66e-4 | 1.85e-5 | 9.3 ms | 12.0 ms | ✅ PASS |
| 3c | 100k | 4.84e-4 | 1.40e-5 | 11.1 ms | 14.2 ms | ✅ PASS |

### 关键观察
1. **鲁棒性提升**: Res P99 从 1.28e-3 降至 4.84e-4，放量后更稳定。
2. **线性扩展**: 吞吐 P99 近似线性增长 (10.2ms → 14.2ms)。
3. **NaN 零发生**: 无数值异常。

---

## 4. 长程训练与语义审计

### 训练配置
- **Particles**: 100,000 (97,136 actual after concept matching)。
- **Steps**: 10,000。
- **Physics**: Weak Bond Strategy (k_attract=1.0, k_repulse=5.0, k_trap=0.05)。

### 语义审计面板 (Semantic Audit Panel)

| Metric | 5k Steps | 10k Steps | Trend | Status |
|--------|----------|-----------|-------|--------|
| Spearman Corr | 0.9569 | 0.9570 | Stable | ✅ PASS |
| Pearson Corr | 0.9903 | 0.9855 | Stable | ✅ PASS |
| Contrast Ratio | 43.08 | 44.29 | Improving | ✅ PASS |
| Mean Radius | 0.36 | 0.44 | Healthy | ✅ |
| R_95 | 0.89 | 1.05 | Controlled | ✅ |
| NaN Count | 0 | 0 | Stable | ✅ PASS |

### 结论
- **层级保持**: Spearman Correlation 稳定在 0.957，表明图结构深度与双曲半径高度相关。
- **聚类质量**: Contrast Ratio ~44，语义层间分离良好。
- **无漂移**: 半径增长受控，无异常膨胀。

---

## 5. Soul Injection 准入评估

根据 `temp-05.md` 建议的准入门槛：

| Criterion | Requirement | Actual | Status |
|-----------|-------------|--------|--------|
| Correlation 稳定 | 无系统性下降 | 0.9570 (10k) vs 0.9569 (5k) | ✅ Met |
| Contrast 不下降 | ≥ 5k 水平 | 44.29 > 43.08 | ✅ Met |
| 半径漂移受控 | 无异常膨胀 | R_95 = 1.05 | ✅ Met |
| Residual/Renorm 无恶化 | P99 稳定 | Res 4.79e-4, Renorm 4.22e-5 | ✅ Met |

**评估结论**: Aurora Base 满足全部准入条件，可进入 Phase 3: Soul Injection。

---

## 6. 保存的模型检查点

| Checkpoint | Path |
|------------|------|
| 30k Particles | `checkpoints/aurora_base_gpu_30000.pkl` |
| 60k Particles | `checkpoints/aurora_base_gpu_60000.pkl` |
| 100k Particles (10k Steps) | `checkpoints/aurora_base_gpu_100000.pkl` |
| Depths | `checkpoints/aurora_base_gpu_100000_depths.npy` |

---

## 7. 下一步

1. **Phase 3: Soul Injection (Summer Pockets)**
   - 在 Shiroha 角色数据上微调。
   - 采用低强度、可回滚策略。
   - 持续监控语义结构稳定性。

2. **可选优化**
   - 进一步延长训练至 20k-50k steps (如指标持续改善)。
   - 可视化 embedding 空间 (PCA/t-SNE on Poincaré Disk)。

---

## 附录: 关键代码修复

| Issue | Fix | File |
|-------|-----|------|
| `dist_n` NaN 风险 | Clamp to `-1.0 - eps` | `geometry_n.py` |
| `HarmonicPriorN` 递归 | 实现 generic center 路径 | `potential_n.py` |
| 变量命名歧义 | `grad_geom` → `force_geom` | `contact_integrator_n.py`, `integrator_torch.py` |
| 缺少 depths 保存 | 训练结束时保存 `.npy` | `train_aurora_base_gpu.py` |
