
---

## A. 你现在“看起来矛盾”的根因：silhouette 不是 k-means silhouette，而是“按 anchor 最近邻标签”的 silhouette

* `compute_hierarchical_metrics()` 里的 `silhouette` 是通过 `labels = assign_nearest(z_disk, anchors_disk)` 得到的 **最近 anchor 标签**，再在点对点双曲距离上算 silhouette。 
* `cluster_summary_kmeans()` 的 `best_k` 则来自 `_kmeans_once()` 的无监督聚类标签。 

所以出现：

* k-means 扫描退化成单簇（best_k=1）
* 但按 anchor 最近邻标签仍能有 silhouette≈0.28（因为 anchor 标签天然会把空间切成多个“语义扇区”）

这说明：你当前输出的 `silhouette` 更接近“**对齐层级模板/锚点分区的可分性**”，不等价于“**涌现簇结构**”。

**建议（立刻做）**：把指标改名并拆开输出，避免误读：

* `sil_anchor`：nearest-anchor silhouette（你现在 `compute_hierarchical_metrics` 的 silhouette）
* `sil_kmeans`：k-means silhouette（你 `cluster_summary_kmeans.best_silhouette`）

---

## B. silhouette_score 的退化处理会“偏乐观”，尤其是单点簇（singleton clusters）

在 `silhouette_score()` 里，如果某个点所在簇只有它自己：你把 `a=0`，并继续算 `b`，这会使该点的 silhouette 可能被推高（接近 1），从而 **抬高整体均值**。 

这对 anchor 标签尤其危险：很多 anchor 可能只分到 1 个点，于是 `sil_anchor` 会被系统性抬高，导致你误以为“出现了清晰分簇”。

**建议修法（标准做法）**：

* 若该点所属簇 size==1，则该点 silhouette 直接记为 0（或在均值里剔除，但需要同时报告剔除比例）。
* 或者整体层面：若任意簇 size==1，占比超过阈值（例如 10%），直接报告 `silhouette_unreliable=true`。

---

## C. cluster_anchor_profiles() 基本是“逻辑错误”，会让你的簇画像（mean_depth / anchor_entropy）失真

`cluster_anchor_profiles()` 的注释写的是“对每个 cluster label 计算 anchor 分布一致性”，但实现里 **并没有**把“点 → 最近 anchor”映射进去，而是做了一个非常不可靠的假设：

* 如果 `len(labels)==len(anchors)`，就把 `np.where(mask)[0]` 当作 anchor_ids（这等于把“簇标签数组的索引”当成 anchor id，语义不成立）；
* 否则就用 `idx % n_anchors` 这种循环近似。 

这会直接污染你之前一直在看的三项画像指标：

* `mean_depth`
* `frac_deepest`
* `anchor_entropy`

**正确做法**应该是：函数需要额外输入 `point_to_anchor_ids`（每个点对应的最近 anchor id），或者在函数内部显式调用 `assign_nearest(points, anchors)` 得到 anchor_ids，再在簇内统计这些 anchor_ids 的 depth/entropy。

> 如果你不修这一条，你的“簇画像”基本不可解释，即使数值看起来很合理也可能是伪象。

---

## D. mean_tree_intra 的定义和实现不一致（当前实现不是“intra”）

`compute_hierarchical_metrics()` 返回的 `mean_tree_intra` 是：
`tree_dist[labels[:, None], labels].mean()` 

这其实是在算：对所有点对 (i,j)，取它们对应 anchor 的树距离再平均——**包含大量 inter-anchor 的项**，不等价于“簇内（intra）”。变量名会误导你把它当成“簇内紧致度”。

**建议**：明确你要哪个量：

* 如果你要“同一 anchor label 内的平均树距离”，那应该只在 `labels[i]==labels[j]` 的子集上求均值（更直观：对每个 label 的 anchor 子图求均值再平均）。
* 如果你要“整体点对的平均树距离”，那就把名字改成 `mean_tree_pair_distance`。

---

## E. tree_dist_corr 的相关性计算包含对角线与重复项，可能被系统性偏置

你现在 `pearson_corr(dist_points, tree_pairs)` 直接对 **完整 n×n 矩阵**做相关。 
这会带来两个偏置源：

1. 对角线全是 0（dist 和 tree 都是 0），会增加“共同结构”成分；
2. 上下三角重复计算（每对点计两次）。

**建议**：只取上三角 `i<j` 的向量再算相关（这会让 corr 更接近你想表达的“点对关系一致性”）。

---

## F. potential.py 里一个容易引发“全局塌缩/单簇”的设计点：repulsion 默认可为 0，而 soft-min + depth-weight 会加强深层吸引

在 `HierarchicalSoftminPotential` 里：

* wells 部分是 soft-min 聚合（log-sum-exp），并且 weight 随 depth 增大（`weight = base_weight * (1+d)`，`width = base_width/(1+d)`）。 
  这会让“深层 anchor”更强更窄，若缺少体积项/排斥项，就很容易把点集吸到有限区域，导致 k-means 退化成 1 簇。

你虽然在 `build_hierarchical_potential()` 里给了 `repulsion_weight=1e-3`，但在实验中如果把它设为 0（你之前确实做过），那么从理论上这就相当于移除了“防塌缩的测度项”。 

**更关键的是**：即便你理论上不想“手调 repulsion”，也需要一个由几何/信息约束导出的等价项；否则“单簇”是系统的自然吸引子之一。

---

# 你现在最该优先修的 3 个点（按影响排序）

1. **修 cluster_anchor_profiles 的 anchor 统计逻辑**（否则簇画像完全不可信）。 
2. **修 silhouette_score 对 singleton cluster 的处理**，并把 `sil_anchor` / `sil_kmeans` 分开命名。 
3. **tree_dist_corr 只用上三角点对**；同时把 `mean_tree_intra` 更名或重定义。 

---

# 你可以用来“快速验尸”的两个 sanity check

1. **构造一个明确两簇的数据**（例如两组点分别围绕两个相距很远的 anchor），检查：

   * `sil_kmeans` 是否 > 0 且 best_k=2
   * `sil_anchor` 是否也 > 0
     只要其中一个不符合，优先怀疑 metrics 实现。

2. **统计 singleton-anchor 的比例**（在 `assign_nearest` 得到的 labels 上做 bincount）：

   * 如果 singleton 占比高，而 `sil_anchor` 还在 0.2~0.4，这基本就是 silhouette_score 的 singleton 偏置在作祟。

