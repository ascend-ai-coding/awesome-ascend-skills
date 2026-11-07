# 第 1 层：数学语义建模（方法 + SFA 实例）

本篇是最佳实践库第 1 层**数学语义**：讲一种建模手段——在写任何流水/tiling/代码之前，先把"算子算什么、哪些数值语义必须保留"写成可检验的约束。后面四层（[流水建模](2-pipeline-modeling.md) → [容量方程](3-capacity-equations.md) → [积木模块](../assemble/index.md) → [胶水](../assemble/glue-idioms.md)）都从这里推导；语义不变量被破坏的"优化"一律无效。各步骤用 SparseFlashAttention（SFA，MLA-absorb 稀疏注意力）作贯穿实例。

## 1. 写计算链与张量清单（方法）

把算子拆成内部计算单元，每个 GEMM 独立记录 M/N/K、dtype、layout；融合算子按内部计算单元逐个拆解，不能按外层算子名套用。

> SFA 实例：`Gather(KV 稀疏索引) → GEMM(QK^T × scale) → 逐元素(online softmax) → GEMM(PV) → O 更新/分片合并`。输入 D=512（NoPE）、head_count=64、每 query 640 个稀疏 KV 索引、bf16（存储规格见[容量方程](3-capacity-equations.md)）；QK 的 B 矩阵来自 Gather 的 NZ 布局。

## 2. 钉死数值不变量（方法）

- online 重缩放与累加 dtype 是语义的一部分，不是实现细节：确定 running max/sum 的更新形式、重缩放因子、无效行/mask 的 exp(−inf) 路径与输出清零规则、分片合并的归一化形式。
- **结合顺序是语义的一部分**：seed 逐位一致要求每 tile 内累加顺序固定；任何重排（分块、融合、指令重排）都必须证明不改变浮点结合顺序，或显式声明改为容差验证并重跑精度。

> SFA 实例：running max `m_i`、running sum `l_i` 逐 tile 更新；旧 accumulator 按 `exp(m_old − m_new)` 重缩放后与新 PV 部分和相加；不同 sample 的分片归一化形式不互相照搬。

## 3. 冻结输入分布与寻址语义（方法）

- case 冻结实际有效长度、候选块分布、页表、重复/合法索引约束、mask 与 metadata；**同 shape 的均匀合成数据不能替代真实网络分布**——稀疏度决定 gather 粒度、每核工作量和长尾。
- 索引顺序即合法写序：重新聚合/排序搬运前必须核验输出等价、数值结合顺序与代价；不因"Gather 模样"强制 SIMT，也不把整个 KV 全载当默认答案。

> SFA 实例：每 query 640 个稀疏索引，契约是"有效前缀 + (−1) padding 后缀"；二分/快速路径都建立在该契约上（见[流水建模](2-pipeline-modeling.md) §4.1）。

## 4. 划定保留与验证边界（方法）

- 保留 online state 重缩放、累加 dtype、无效行和输出/LSE 语义；实际 case 未命中已有优化路径时，先评估合法派发，不能仅凭源码存在该分支宣称运行有效。
- 流水修改要同时验证单块、首/尾块、跨槽位和长序列；给出生成代码/trace 观察，再比较完整调用，不把局部 MM 或 VF 加速当作整体收益。
- 样例 README 的历史收益不代表当前候选通过；案例经验按 case 沉淀验证记录（含无收益/失败与原因）。
