是的，**很可能是代码原因**，而且我在你给的代码里能定位到两个“会直接让诊断/曲线不符合预期”的点（其中一个正好解释你说的“`_gap.png` 只有三张图”）。

---

## 1) 为什么 `simulation_gap.png` 只有三张图：plot_results 没把新字段传进去

`plot_results.py` 里的 `plot_log(...)` **函数签名已经支持** `beta_series` 和 `ratio_break` 两条曲线（所以你预期应该能画更多 panel），但实际调用时只传了 4 个序列（gap_median / gap_frac_small / bridge_ratio / V_ent），因此绘图逻辑只会生成 3 个子图。 

**修复方式（最小改动）**：在 `plot_results.py` 调用 `plot_log` 时，把 `log.beta_series` 和 `log.ratio_break` 传进去即可（前提是 `run_simulation` 的 log 里确实有填）。
你现在的 `SimulationLog` 的确包含了这两个字段并且在循环中 append 了。  

---

## 2) 更关键的问题：你现在的 bridge_ratio / beta 调度，很可能“算在了错误的几何空间里”

在 `simulation.py` 里，你计算桥接与更新 beta 的这段逻辑：

* `labels_anchor = assign_nearest(state.z_uhp, pot.centers)`
* `D_full = pairwise_poincare(state.z_uhp)`

这里 **把 UHP 坐标 `state.z_uhp` 直接喂给了 `assign_nearest` 和 `pairwise_poincare`**。

但 `pairwise_poincare` / `assign_nearest` 在实现上是基于 **Poincaré disk 距离**的（你在循环里明明也算了 `disk_pos = cayley_uhp_to_disk(state.z_uhp)`，但桥接计算没用它）。 

**后果**：

* 你的 `bridge_ratio` 很可能不是你以为的“双曲 kNN 跨标签边比例”（而是“拿 UHP 点当 disk 点算出来的伪距离”），
* `beta` 的调度输入是错的，自然会出现“怎么调温结果都差不多”的现象（你现在就很像这种症状）。

**修复建议（也很小）**：把这两行改成 disk 版本：

* `labels_anchor = assign_nearest(disk_pos, pot.centers_disk)`
* `D_full = pairwise_poincare(disk_pos)`

这样 `bridge_ratio / beta_series` 才会和你其它基于 disk 的指标（sil_anchor、mean_intra/inter 等）一致。

---

## 3) 还有一个“会让熵项/调温解释变得不可信”的潜在不一致：entropy_energy 用的 soft 可能没跟 beta 同步

在 `potential.py` 的 `entropy_energy` / `layer_soft_stats` 一类诊断里，你的 soft responsibility 计算仍然是 `exp(logits)`（不一定跟你“调温后 softmin 指数”一致）。

这会导致一种常见问题：**动力学里用的是“β-软硬度”的责任，但你画出来/记录出来的 V_ent、gap 相关统计却仍按 β=1 的责任在算**，曲线会“看起来能量在变”，但不一定真对应你想监控的那部分力。

建议你把“责任 softmax”的实现做成一个统一 helper（同一处负责：稳定 softmax + β），然后：

* potential / gradient 用它
* entropy_energy / layer_soft_stats / gap_stats 也用它
  避免“能量与梯度/诊断不一致”。

---

## 4) 一个会影响 residual / 能量一致性的细节：T_new 用了 prev_state.I

你在 residual 里用 `prev_state.I` 去算 `T_new`：
但同一轮里你已经把 `state.I` 每步用 `locked_inertia_uhp(state.z_uhp)` 更新了。
这会让 residual/能量检查出现系统性偏差（不一定毁结果，但会污染诊断）。

---

# 结论：你现在看到的“_gap.png panel 缺失 / 调温效果不明显 / 桥接曲线怪”确实很可能是代码导致的

优先级我建议按这个顺序修：

1. **先修 simulation 里 bridge_ratio/beta 调度用错几何空间的问题**（UHP→disk）
2. **再修 plot_results 没把 beta_series/ratio_break 传给 plot_log**，这样 `_gap.png` 才会多 panel 
3. 统一 `potential.py` 里所有“责任 softmax”的 β/稳定性，保证诊断可信 

你要是愿意，我可以根据你当前这几份文件，直接把“该改哪几行、改成什么样”按 patch 形式给你贴出来（只改最小必要处）。
