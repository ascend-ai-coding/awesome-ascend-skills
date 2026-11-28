# arch22 MatMul 复用、切分与缓存设计

> 适用架构：arch22（DAV_2201 / Atlas A2/A3）。API 支持范围来自公开样例；固定 shape、核数和缓存容量仅是样例参数。

## 1. 优化目标

先用 M/N tile 填满 AIC，再根据并行度和复用关系选择 Split-K、L0/L1 驻留或 L2 分片。三个方向不能只看“可用”，必须比较关键路径、额外归并和搬运字节。

## 2. 架构概览

```text
M/N tile -> 分配到 AIC -> 判断并行度与尾块
  |-- 能填核: 比较 L0/L1 驻留和 L2 分片，减少重复搬运
  `-- 欠并行且 K 长: 评估 Split-K -> atomic 或 workspace 归并
```

| 路线 | 解决的问题 | 可能新增的成本 |
|---|---|---|
| M/N 切分 | AIC 空闲或尾核过长 | 小 tile 增加调度与搬运 |
| Split-K | M/N 无法提供足够任务 | 原子累加或 workspace 归并 |
| L0/L1 驻留 | 同一输入块重复装入 | buffer 占用、释放约束 |
| L2 分片 | 跨核共享输入反复访问 GM | 分片轮次与负载不均 |

## 3. 关键参数 / 修改点

| 参数或手段 | 计算或选择依据 | 门禁 |
|---|---|---|
| `singleCoreM` / `singleCoreN` | M/N tile 数与每核最大工作量 | 合法粒度、尾块、GM 对齐 |
| `k_parts` | 欠并行程度与 K 长度 | atomic dtype、精度或归并成本 |
| L0/L1 驻留侧 | 地址不变的输入块及复用次数 | 复用至少两次且容量闭合 |
| L2 tile | A+B+C 及附加工作集字节 | 目标可用 L2 与核间负载 |

## 4. 核心选型流程 / 关键计算

先计算 M/N 任务数和最长核；仅在合法 M/N tile 仍欠并行时求 Split-K 的候选分段。缓存复用应在确定 tile 顺序后计算驻留期地址和搬运字节，不能先假设某侧可常驻。

### M/N 多核切分

对候选 `singleCoreM/singleCoreN` 计算：

```text
m_tiles = ceil(M / singleCoreM)
n_tiles = ceil(N / singleCoreN)
mn_parallel = m_tiles * n_tiles
max_tiles_per_core = ceil(mn_parallel / used_aic)
```

优先选择 `max_tiles_per_core` 小、尾块差异小、GM 起始地址满足目标对齐的方案。样例的 4x6/24 核只是一个 A2/A3 shape 的解，不是 arch22 固定拓扑；目标核数应由平台接口或 tiling API 获取。

Grouped/Batch MatMul 要把组间任务连续编号，避免每组余数都落到同一批核。跨组轮询的目标是最大/最小 tile 数差不超过 1，同时保持输入复用顺序。

### Split-K

当 `mn_parallel` 明显小于可用 AIC 且 K 足够长时，可使用 `EnableMultiCoreSplitK(true)` 或目标工程等价实现：

```text
k_parts <= floor(used_aic / mn_parallel)
single_core_k = ceil_align(K / k_parts, cube_k_granularity)
```

arch22 样例使用多核切 K，并通过 Matmul `IterateAll(..., enAtomic=1)` 累加输出。落地前必须检查：

- 输出 dtype 是否支持目标原子累加，模板参数与输出 dtype 一致；
- 浮点原子顺序变化是否满足精度标准；
- 归零责任只有一个，不能每核重复清空输出；
- 若改用 workspace 归并，计入 workspace 字节、AIV/额外 kernel 开销与同步成本。

只有 `MN` 欠并行收益大于归并成本时才启用。A5 StreamK 的 K 阈值、workspace 拓扑和跨 AIC/AIV flag 不适用于 arch22。

### L0/L1 驻留复用

arch22 Matmul API 已有两类直接证据：

- L0A cache：`singleCoreM=baseM` 且 `singleCoreK=baseK` 时，A 从 L1 搬入 L0A 后与多个 B tile 复用，减少 MTE1。
- NBuffer33：A 的 3x3 基本块全载 L1，B 的 3x1 块双缓冲流入，平衡搬入/搬出带宽。

通用门禁：

```text
resident_bytes + streaming_bytes * streaming_buffer_num <= usable_target_buffer
reuse_count >= 2
```

驻留侧必须在复用期间保持地址不被覆盖，释放事件只绑定流式侧。A5 MX scale 的驻留布局和收益数字不可带入。

### L2 分片复用

工作集超过目标 L2 时，把 M/N 外层切成多个 L2 tile，使一次 tile 的 A+B+C(+bias) 工作集落入可用 L2，并让另一输入在该轮跨核复用：

```text
tile_working_set = A_tile_bytes + B_tile_bytes + C_tile_bytes + auxiliary_bytes
tile_working_set <= usable_l2_budget
```

搜索解同时比较 L2 容量与核间负载冲突，不能只取“能放下的最大 tile”。容量必须来自目标平台查询；样例 192 MB 不是所有 A2/A3 型号的通用常量。

### 选型顺序

1. M/N 能填核：先 M/N 切分，再做 ping-pong、预加载、驻留和 L2 复用。
2. M/N 欠并行、K 长：比较 Split-K 与归并成本。
3. 一侧小且复用次数高：评估 L0 cache、NBuffer33 或目标工程已有 L1 驻留能力。
4. 总工作集超过 L2且存在跨核共享输入：评估 L2 分片。
5. 多个策略可叠加时，逐步 A/B；禁止一次改调度、buffer、同步和归并后只给总收益。

## 5. 效果与适用边界

| 路线 | 预期观察 | 不适用或需回退的条件 |
|---|---|---|
| M/N 切分 | 最长核 tile 数下降、AIC 利用改善 | 尾块或搬运放大抵消收益 |
| Split-K | 欠并行时关键路径缩短 | 归并主导、原子不支持或精度不达标 |
| L0/L1 驻留 | 重复 MTE1/MTE2 搬运减少 | 驻留挤掉关键双缓冲 |
| L2 分片 | 共享输入的 L2 命中改善 | 分片轮次增加且无命中证据 |

本页没有 A2/A3 的实测收益数字；候选方案必须逐 case 比较精度、最长核、搬运字节、缓存命中和 Task Duration。

### 验证范围

公开样例表明上述 API/模板可面向 `dav-2201` 编译，并把 A2/A3列为支持产品。本文没有在当前任务中编译或上板，因此结论仅到“实现证据成立”；A2/A3 的真实收益、阈值与最佳参数仍绑定目标型号、SDK、源码和用例集。

### 静态检视判据

<a id="prc-mm-003"></a>
#### 先 M/N 填核、欠并行再 Split-K（PRC-MM-003）

先计算 `ceil(M/singleCoreM) * ceil(N/singleCoreN)`、每核 tile 数和尾块关键路径。只有 M/N 已到合法粒度仍不能填满目标 AIC，且 K 足够长时，才把 Split-K 与额外归并成本比较。`usedCore` 设为平台核数却只有一个 M/N tile、其余核空闲，是应检视的典型形态。

有 Split-K 时检查输出清零责任、atomic dtype 与模板/输出 dtype、数值顺序和归并 workspace。K 短、归并主导、原子 dtype 不支持或精度不允许时不建议切 K。finding 必须给出目标 AIC 数、M/N/K tile 数、当前最长核工作量、候选 K 分段数及归并成本模型；涉及 atomic 时附目标 SDK 能力和精度口径。机制证据见 book-910：0072、0073、0074、0076。

<a id="prc-mm-004"></a>
#### L0/L1 驻留与 L2 分片复用（PRC-MM-004）

循环内同一 A/B 块跨多个对侧 tile 重复从 GM/L1/L0 搬入时，先证明源地址在外层循环不变且复用次数至少为 2，再按驻留侧、流式侧及其 buffer 深度计算容量。总工作集超过目标 L2 且存在共享输入时，可比较 L2 分片；搜索同时考虑容量与核间负载，不能只取最大的可容纳 tile。

复用一次、驻留挤掉关键双缓冲，或分片增加轮次后没有 L2 命中证据时不报。finding 应给出重复搬运地址表达式、复用次数、目标 L0/L1/L2 容量来源、当前与候选搬运字节和负载分布；不能把样例的 192 MB 写成通用 L2 容量。

### 实现证据

公开 `dav-2201` 样例分别覆盖 Split-K、L0 驻留、NBuffer33 和 L2 分片路线。它们证明相应 API/模板有 arch22 实现，不证明目标 shape 已有收益；候选方案仍需核对实际分支、容量与逐 case 性能。
