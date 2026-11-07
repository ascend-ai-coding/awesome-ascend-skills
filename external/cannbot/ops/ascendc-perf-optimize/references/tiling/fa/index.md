# FA 族 Tiling 算法路由

> FA 族（FlashAttention、FusedInferAttentionScore、QSFA、SparseFlashAttention、SCFA）Tiling 算法入口。
>
> 本文件按代际路由：arch22 与 arch35 各自维护注册算法，不能跨代互相补位。

---

## 1. 算法注册表

| 优先级 | 代际 | 注册算法 | 选择条件 | 位置 |
|---|---|---|---|---|
| 100 | arch22（DAV_2201 / Atlas A2/A3） | FIA 目标模型（Tiling、buffer 与同步） | DAV_2201 / 220x；FIA nonquant/MLA，或已证明共享同一 arch22 数据流的变体 | [arch22/tiling_pipeline_design.md](arch22/tiling_pipeline_design.md) |
| 100 | arch35（DAV_3510 / A5） | FA AIC+AIV tiling建模 | DAV_3510；FA 变体与 Cube+Vector 融合算子，计算链 Gather→GEMM→softmax→GEMM 及同构形态 | [arch35/four-layer-derivation.md](arch35/four-layer-derivation.md) |

- 优先级同为 100：代际隔离，先按目标 SoC/DAV 锁定代际，再在代际内按选择条件匹配，不存在跨代排序。
- arch35 注册的是[公共四层推导方法](arch35/four-layer-derivation.md)（数学语义 → 评价函数 → 流水 → tiling）；[sfa.md](arch35/sfa.md)（SparseFlashAttention，CANNBot-DSL 或同构 AIC+AIV 混合实现）只是它在 arch35 上的应用实例，不作为独立算法注册。
- arch22 FIA 路线联合求解 `mBaseSize`、`sInnerSize`、核数、L1/L0/UB buffer、事件和 normal/FlashDecode workspace（M/S2 多核切分与负载均衡、Split-KV / FlashDecode workspace、L1/L0/UB 多缓冲与精确事件、DataCopy 对齐搬运）。
- A3 没有目标实测时保留 `A3_not_run`，不得迁移 A5 的 SCALAR/SCALARLDST 阈值、Reg/VF、SIMT 或 `__ssbuf__` 路线；变体未证明与已注册实例同构时返回变体级参考缺口。

## 2. 拓展新实例

拓展包含两部分，新算法与新实例的注册优先级都是 100，由选择条件在本代际内匹配：

1. **代际下拓展新算法**：在对应代际目录（`arch22/`、`arch35/`）下创建新目录/文档并在上表注册，选择条件必须包含目标架构和具体 FA 变体，文档必须自包含。
2. **代际新算法下拓展新算子的实例**：在已注册算法下为新算子变体补充应用实例（如 [sfa.md](arch35/sfa.md) 之于 arch35 的 FA AIC+AIV tiling建模）：在算法目录下创建实例文档并注明选择条件；推导过程按该注册算法组织，方法、台账结构与"贴顶即最优"论证可复用，数学语义分解、台账逐项、可行域、滞后系数必须按新算子重推导。
