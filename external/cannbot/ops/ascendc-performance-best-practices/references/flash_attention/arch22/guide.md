# FlashAttention 家族性能优化策略索引（arch22）

> 适用架构：arch22（DAV_2201 / 220x，Atlas A2/A3）。当前直接实现证据集中在 FIA nonquant/MLA；A2/A3 共用编译架构不等于所有型号性能相同。具体源码证据见主题分册。

## 策略索引

| 策略 | arch22 结论 | 详细文档 |
|---|---|---|
| arch22 QK/PV 分块与模板路由 | 源码类型名使用 `Arch::AtlasA2`、`MmadAtlasA2FAI*` 和 `EpilogueAtlasA2*`；它们属于 arch22 路由，不能因命名或目标为 A3 而替换成 A5 模板 | [Tiling、buffer 与同步](tiling_pipeline_design.md) |
| M/S2 多核切分与负载均衡 | arch22 host tiling 已按 B/N2/S1G、核数与尾块求解 | [Tiling、buffer 与同步](tiling_pipeline_design.md) |
| Split-KV / FlashDecode | arch22 已有 workspace、核数和归并路径 | [Tiling、buffer 与同步](tiling_pipeline_design.md) |
| L1/L0/UB 多缓冲与精确事件 | arch22 源码有 L1 4-buffer/3-buffer、L0 双缓冲和事件表 | [Tiling、buffer 与同步](tiling_pipeline_design.md) |
| `DataCopy` / `DataCopyPad` 对齐搬运 | arch22 有直接实现；NDDMA 结论不可套入 | [Tiling、buffer 与同步](tiling_pipeline_design.md) |
| ICache 预取 | arch22 FIA 已调用 `ICachePreLoad` | [Scalar 与启动开销](scalar_overhead_design.md) |
| 静态 LocalTensor / TilingData 直读 / 热变量局部化 | 机制有效，arch22 源码存在静态 buffer 与 GM tiling 指针读法 | [Scalar 与启动开销](scalar_overhead_design.md) |

## 静态检视检查点

先按 [arch22 检视适用性矩阵](../../common/arch22/perf_review_applicability.md) 筛选通用检查点；以下编号在对应主题分册中给出识别方法、误报边界和证据要求。

| ID | 检视重点 | 条目 |
|---|---|---|
| PRC-FA-001 | S2/M 切分与 workspace 联合求解 | [Tiling、buffer 与同步](tiling_pipeline_design.md#prc-fa-001) |
| PRC-FA-002 | 多级 buffer 深度与事件 ID 共同闭合 | [Tiling、buffer 与同步](tiling_pipeline_design.md#prc-fa-002) |
| PRC-FA-003 | 启动与热循环 Scalar 减载 | [Scalar 与启动开销](scalar_overhead_design.md#prc-fa-003) |

## 跨代策略适用性

| arch35 策略 | arch22 结论 | 处理方式 |
|---|---|---|
| `scalar_general_techniques` | 部分有效 | ICache 预取、静态 tensor、局部化、tiling 指针读可在 arch22独立验证；A5 收益表不迁移 |
| `scalar_precompute` | 算法机制有效 | 仅迁移循环不变量外提/预乘思想；不使用 A5 `SCALAR%` 阈值和字段名 |
| `scalarldst_subbound` | 方法论候选 | A5 的 `SCALARLDST%` 阈值、case cycle 和成员组合不迁移；arch22 先以 trace/反汇编确认 |
| `getvalue_cacheline_batching` | 未建立 arch22 性能证据 | 保留为待验证假设，不进入默认 arch22 方案 |
| `tilingdata_cv_ssbuf_forwarding` | 无效 | A5 mixed AIC/AIV `__ssbuf__` 路线，不复制到 arch22 |

## 强制边界

1. arch22 优先沿用目标工程的 A2 专用 template/ABI/tiling key；不从 arch35 复制 RegBase、SIMT、SSBUF 或 A5 sparse metadata 阈值。
2. 核数、L2/UB/L1 可用量从目标平台或现有 compile info 获取。源码中的 32 MB workspace 限制、512/128 等 tile 是具体实现策略，变更需同步核验所有 layout/feature 分支。
3. A2 源码证据可证明机制和实现存在；A3 需要目标型号的构建、精度和性能复验。
4. FA 改动必须同时覆盖 QK、online softmax、PV、rescale-O、FlashDecode/归并与可选 mask/paged-cache 路径，不能只测主分支。
5. 当前直接源码证据集中在 FIA nonquant/MLA；QSFA、SparseFlashAttention、SCFA 等变体只有在目标工程证明共享 arch22 ABI、模板和 feature 路径后才能复用本目录，否则记录变体级参考缺口。
