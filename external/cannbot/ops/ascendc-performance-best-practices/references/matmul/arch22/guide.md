# MatMul 类算子性能优化策略索引（arch22）

> 适用架构：arch22（DAV_2201 / 220x，Atlas A2/A3）。公开样例的 A2/A3 支持范围不等于性能数据已在全部型号验证；A3 的收益与阈值仍需在目标型号复测。具体实现证据见主题分册。

本目录不复制 arch35 的 Blaze、RegBase、MXFP4/MXFP8 专用实现或固定阈值，只保留在 arch22 有独立实现/API 证据的策略。

## 策略索引

| 策略 | arch22 结论 | 适用场景 | 详细文档 |
|---|---|---|---|
| L1/L0 ping-pong + 精确事件 | **可直接进入 arch22 方案候选** | K 方向有多轮，MTE2/MTE1/Cube/Fixpipe 可重叠 | [流水、预取与大包搬运](pipeline_design.md) |
| M/N 预加载 | **API/模板已有 A2/A3 支持证据** | K 可全载，M 或 N 很大且对应方向可双缓冲 | [流水、预取与大包搬运](pipeline_design.md) |
| 大包搬运与紧迫侧优先 | **机制和 arch22 实现均有证据** | `stepKa/stepKb > 1`，小 DMA 或 MTE2 发射间隙明显 | [流水、预取与大包搬运](pipeline_design.md) |
| LoadData3D 降低 MTE1 队列占用 | **arch22 专用优先项** | LoadData2D 需要循环发多条指令，MTE1 队列压力高 | [流水、预取与大包搬运](pipeline_design.md) |
| M/N 多核切分与负载均衡 | **可直接进入 arch22 方案候选** | M/N 有足够独立 tile | [复用、切分与缓存](reuse_and_split_design.md) |
| Split-K | **API/样例明确支持 A2/A3** | M/N 太小不能填满 AIC，K 足够长；允许原子累加或 workspace 归并 | [复用、切分与缓存](reuse_and_split_design.md) |
| L0/L1 驻留复用 | **API/模板已有 A2/A3 支持证据** | 一侧基本块可全载，且会与另一侧多个块复用 | [复用、切分与缓存](reuse_and_split_design.md) |
| L2 分片复用 | **API/样例明确支持 A2/A3** | 工作集超过目标 L2，且某一输入可跨核/跨轮复用 | [复用、切分与缓存](reuse_and_split_design.md) |
| 常量 Tiling / UnitFlag | **API/样例明确支持 A2/A3** | shape 集合固定或可分桶；MMAD 与 Fixpipe 有重叠空间 | [流水、预取与大包搬运](pipeline_design.md) |

## 静态检视检查点

先按 [arch22 检视适用性矩阵](../../common/arch22/perf_review_applicability.md) 筛选通用检查点；以下编号在对应主题分册中给出识别方法、误报边界和证据要求。

| ID | 检视重点 | 条目 |
|---|---|---|
| PRC-MM-001 | 多级 Ping-Pong 与事件闭环 | [流水、预取与大包搬运](pipeline_design.md#prc-mm-001) |
| PRC-MM-002 | 大包搬运、紧迫侧优先与预加载门禁 | [流水、预取与大包搬运](pipeline_design.md#prc-mm-002) |
| PRC-MM-003 | 先 M/N 填核、欠并行再 Split-K | [复用、切分与缓存](reuse_and_split_design.md#prc-mm-003) |
| PRC-MM-004 | L0/L1 驻留与 L2 分片复用 | [复用、切分与缓存](reuse_and_split_design.md#prc-mm-004) |

## 跨代策略适用性

| arch35 策略 | arch22 结论 | 处理方式 |
|---|---|---|
| `pingpong` | 有效 | 使用本目录的 arch22 事件数、L1/L0 预算和 LoadData 路线重新建模 |
| `mte2_preload` | 有效，但 API/前置条件不同 | 使用 Matmul API 的 `doMTE2Preload` 与 K 全载、M/N DB 门禁，不复制 A5 三段手排实现 |
| `fullload` | 原理有效 | 使用 arch22 的 L0 cache / NBuffer33 / 目标工程驻留实现；不复制 A5 MX scale 布局 |
| `streamk` | “切 K 填核”原理有效 | 使用 arch22 `EnableMultiCoreSplitK` 或目标工程 Split-K；不复制 A5 AIC/AIV workspace 同步拓扑和阈值 |
| `swat` | 负载均衡与遍历复用原理有效 | 重新按目标核数、tile 数和 L2 命中建模；不复制固定窗口、尾块常量 |
| `scale_coalescing` | 仅保留“小 DMA 合并”机制 | MX scale dtype、20 KB 阈值和 A5 带宽数据不可迁移 |
| `blaze` | 无效 | arch22 不进入 Blaze/tensor_api 路线 |

## 强制边界

1. `dav-2201` 是 A2/A3 的编译架构，不是“A2 实测即可证明 A3 性能相同”。A3 必须重新记录型号、SDK、核数、L1/L2 容量与逐 case 性能。
2. 核数和容量优先由平台接口/目标 tiling 取得；样例中的 24 核、512 KB L1、192 MB L2只可作为样例证据，不作为所有 A2/A3 型号的常量。
3. arch22 的事件 ID 预算、LoadData 能力和 dtype 支持必须按目标 SDK 核验；不得把 arch35 的 Reg/MicroAPI、SIMT 或 `__ssbuf__` 写法带入。
4. Split-K、原子累加、L2 分片与驻留复用都必须有对照 case；仅“API 支持”不等于对当前 shape 有收益。
