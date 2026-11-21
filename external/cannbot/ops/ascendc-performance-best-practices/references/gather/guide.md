# Gather 索引类算子高性能实现索引

> 适用架构：arch35（A5 / DAV_3510）；具体型号、SDK 与验证范围以正文为准。

Gather 族算子（GatherV2、GatherElementsV2、Gather、Embedding、torch.gather 等）在 Ascend Vector 核心上的高性能实现，沉淀为一组**可复用 Kernel 模板**。本指南是这些模板的入口，介绍每个模板解决的问题、关键技术与适用条件。

> 本指南是**优秀实践目录**，而非选型决策树：每个模板列出其适用条件与解决的问题，由实现者根据实际 shape、UB 容量与带宽情况判断采用。
>
> **重要**：Gather 族分为两种语义——`PER_ELEMENT`（torch.gather/GatherElementsV2，index 与 output 同形，每位置独立索引取 1 元素）和 `PER_SLICE`（GatherV2/Embedding，每索引取 innerSize 个连续元素）。两者模板选择不同，详见下文「torch.gather vs TF GatherV2 的关键差异」和 [tiling-flow](../../../ascendc-perf-optimize/references/tiling/gather/fallback/tiling-flow.md)。

## 文档结构

```
guide.md（本文件）
└── templates/
    ├── usage_guide.md                              模板集成指南
    ├── gather_tiling_data.template                 多套 TilingData 公共定义（→ .h）
    ├── gather_tiling.template                      Host tiling 门控 + 切分参考实现（→ .h）
    ├── gather_elements_v2_tiling.template          GatherElementsV2 Host tiling 主入口（→ .cpp）
    ├── gather_elements_v2_route.template           GatherElementsV2 上游路径门控（→ .h）
    ├── gather_elements_v2_last_dim_tiling.template GatherElementsV2 LastDim Host tiling（→ .h）
    ├── simt_gather.md                              SIMT 标量并行模板说明
    ├── simd_last_gather.md                         SIMD 末轴 Gather 模板说明
    ├── simd_row_copy.md                            SIMD 行拷贝模板说明 (PER_SLICE only)
    ├── ga_all_load.md                              GA 全载模板说明
    ├── full_load.md                                Full-Load（GA 全载 + SIMT 从 UB 取值）模板说明
    ├── simt_two_dim.md                             SIMT 两维（2D 特化）模板说明
    ├── gather_elements_v2_transpose.md             GatherElementsV2 Transpose + Gather 模板说明
    ├── gather_elements_small_gd_tiled.md            PER_ELEMENT small-GD tiled full-load 实验性实现模式
    ├── gather_elements_v2_scalar.md                GatherElementsV2 Scalar 兜底模板说明
    ├── gather_elements_v2_last_dim.md              GatherElementsV2 LastDim 模板说明
    └── dav310/
        ├── gather_simt.template                    SIMT Kernel（→ .h）
        ├── gather_simd_last_gather.template        SIMD 末轴 Gather Kernel（→ .h）
        ├── gather_simd_row_copy.template           SIMD 行拷贝 Kernel（→ .h）
        ├── gather_ga_all_load.template             GA 全载 Kernel（→ .h）
        ├── gather_full_load.template               Full-Load Kernel（→ .h）
        ├── gather_simt_two_dim.template            SIMT 两维 Kernel（→ .h）
        ├── gather_elements_v2_transpose.template   GatherElementsV2 Transpose Kernel（→ .h）
        ├── gather_elements_v2_scalar.template      GatherElementsV2 Scalar Kernel（→ .h）
        ├── gather_elements_v2_last_dim.template    GatherElementsV2 LastDim Kernel（→ .h）
        └── gather_elements_v2_common.template      GatherElementsV2 公共基类（→ .h）
```

> **GatherElementsV2 模板**（Transpose + Gather / Scalar / LastDim）现已沉淀到本目录，与 GatherV2 模板同级。原始参考实现位于 ops-nn 工程 `ops-nn/index/gather_elements_v2/`，集成时按工程实际适配 TilingData 结构体。

**阅读顺序**：先读本指南「Gather 计算的通用范式」「公共框架」理解骨架 → 读对应模板 `.md` 了解流程与关键技术 → 落代码时对照同名 `.template` → 集成时读 [usage_guide.md](templates/usage_guide.md)。

## Gather 计算的通用范式

Gather 族算子的核心操作是**按索引从输入张量中收集元素**，可统一分解为三维模型：

```
输入 x:    [outerSize, gatherDimSize, innerSize]
索引 index: [outerSize, gatherSize,   innerSize]  （torch.gather: index.shape[i] ≤ x.shape[i] for i ≠ dim）
输出 y:    [outerSize, gatherSize,   innerSize]

对每个输出元素 y[o, g, c]:
  idx = index[o, g, c]              // 从索引张量取值
  y[o, g, c] = x[o, idx, c]         // 从输入张量按索引取值
```

- **outerSize**：gather 轴之前的维度乘积（batch/outer 维）
- **gatherDimSize**：x 沿 gather 轴的长度
- **gatherSize**：index 沿 gather 轴的长度
- **innerSize**：gather 轴之后的维度乘积

各模板的差异在于：**如何从 x 取值**（标量读 vs DMA 搬运 vs 向量化 gather）和**如何利用 UB**（全载 vs 分块 vs 不载入）。

### torch.gather vs TF GatherV2 的关键差异

| 维度 | torch.gather (GatherElementsV2) | TF GatherV2 |
|------|-------------|-------------|
| 算子 | GatherElementsV2 | GatherV2 |
| index 秩 | == x.ndim | ≤ x.ndim（可低维） |
| 每个 index 值取 | **1 个元素** | **innerSize 个连续元素**（一个 slice） |
| 典型瓶颈 | 散点标量读（ScalarBound） | 逐 slice DMA（MTE2 碎读） |
| dim!=last 候选路径 | 物理 Transpose×3；kernel 内 Transpose；small-GD tiled full-load；不满足门控时 Scalar/SIMT | SIMD 行拷贝 / GA 全载 |
| dim==last 最优路径 | SIMD 末轴 Gather / LastDim | SIMD 末轴 Gather / LastDim |
| 兜底路径 | SIMT 并行 | SIMT 并行 |

> torch.gather 中 index 与输出同形，每个输出位置有独立索引 → 纯散点读；TF GatherV2 中 index 通常低维，一个索引取整个 slice → 可 DMA 批量搬。
>
> **关键**：torch.gather dim!=last 时，不能直接使用 GatherV2 的 SIMD 行拷贝/GA 全载模板（它们假设一个 index 取整行 slice）。但可以保留 PER_ELEMENT 语义，按 inner tile 连续搬入全部小 gather 轴平面，再在 UB 内逐元素 Gather。`dim!=last` 也不是任何 Transpose 路径的充分条件；所有候选都必须经过能力、UB 容量、搬运成本和 A/B 门控。

## Kernel 模板

### 1. Transpose + Gather（GatherElementsV2 Transpose 模板）

| 项 | 内容 |
|----|------|
| **适用** | torch.gather dim!=last，且 dtype ∈ {fp16/bf16/fp32/int32}、postDim/gatherDim/shape/UB 门控通过，并且成本模型判定内部 workspace 路径优于物理 Transpose |
| **核心策略** | kernel 内 3 阶段：stride DMA 转置 → Gather 指令向量化 → ReTranspose 写回，全程 1 个 kernel + workspace GM 中转 |
| **UB 使用** | transpose buffer (64KB in + 64KB out) + gather buffer (x 行 + index + output) |
| **关键 API** | `DataCopyPad(srcStride)` stride DMA + `TransDataTo5HD` UB 内转置 + `Gather` 指令向量化 |
| **参考实现** | [templates/dav310/gather_elements_v2_transpose.template](templates/dav310/gather_elements_v2_transpose.template) · [.md](templates/gather_elements_v2_transpose.md) |

> 该模板只实现 kernel 内 Transpose，当前仅支持 2B/4B 数据。int8 不可通过“提升为 int16”直接套用；若平台支持，应优先评估框架层物理 Transpose×3 + 末轴 Gather，否则回退 Scalar/SIMT。

### 1a. small-GD tiled full-load（PER_ELEMENT 实验性模式）

| 项 | 内容 |
|----|------|
| **适用** | torch.gather dim!=last、innerSize>1，gather 轴平面数较小，按 inner tile 搬入 x/index/y 后满足 UB 预算，且连续搬运成本低于 direct random GM |
| **核心策略** | 每个任务搬入全部 x gather 平面的同一 inner tile，在 UB 内将 PER_ELEMENT index 转为字节 offset 并执行向量 Gather |
| **UB 使用** | `2 × (Gx×tile×xBytes + Gi×tile×idxBytes + Gi×tile×xBytes) + reserved`（双缓冲模型） |
| **状态** | 已有实测收益的实现模式；当前未提供通用 `.template`，不得把固定 gatherDim specialization 当成通用模板 |
| **说明** | [templates/gather_elements_small_gd_tiled.md](templates/gather_elements_small_gd_tiled.md) |

### 2. SIMT 标量并行（通用兜底）

| 项 | 内容 |
|----|------|
| **适用** | 所有 Gather 场景（散点读，无连续性假设） |
| **核心策略** | 2048 SIMT 线程并行，每线程直接 GM 标量读 index + x，写 y；fast-division 替代 div/mod |
| **UB 使用** | 无（直接 GM→GM） |
| **代码** | [.template](templates/dav310/gather_simt.template) · [.md](templates/simt_gather.md) |
| **实测** | torch.gather P4(33.5M int32): 128ms→3.6ms (**35.7×**)；P6(67.1M int8): 231ms→5.3ms (**43.2×**) |

### 3. SIMD 末轴 Gather（Gather）

| 项 | 内容 |
|----|------|
| **适用** | innerSize==1（末轴 gather）+ gatherDimSize 可全载 UB + batchSize==1 |
| **核心策略** | x 一行（gatherDimSize 个元素）全载入 UB，用 Reg API `Gather` 向量化 gather 多个索引 |
| **UB 使用** | x 行全载 + index tile + output tile |
| **代码** | [.template](templates/dav310/gather_simd_last_gather.template) · [.md](templates/simd_last_gather.md) |

### 4. SIMD 行拷贝

| 项 | 内容 |
|----|------|
| **适用** | innerSize 较大（≥阈值）+ batchSize*outerSize*gatherSize ≥ coreNum/2 |
| **核心策略** | 对每个输出行，用 `DataCopyPad` 从 GM 搬 innerSize 个连续元素到 UB；index 缓存避免重复读 |
| **UB 使用** | x 行 tile + index 缓存 + output tile |
| **代码** | [.template](templates/dav310/gather_simd_row_copy.template) · [.md](templates/simd_row_copy.md) |

### 5. GA 全载（gatherDim × inner 全载 + Gather）

| 项 | 内容 |
|----|------|
| **适用** | innerSize 小（< 2048）+ gatherDimSize × innerSize 可全载 UB + batchSize==1 |
| **核心策略** | 将 x 的 [gatherDimSize, innerSize] 二维块全载入 UB，用 `Gather` 按索引向量 gather |
| **UB 使用** | x GA 块全载 + index tile + output tile |
| **代码** | [.template](templates/dav310/gather_ga_all_load.template) · [.md](templates/ga_all_load.md) |

### 6. Full-Load（GA 全载 + SIMT 从 UB 取值）

| 项 | 内容 |
|----|------|
| **适用** | gatherDimSize × innerSize × dtypeSize < UB - DCache + batchSize==1 + outerSize==1 + ySize×dtype > 2048 + gatherSize/gatherDimSize > 1 |
| **核心策略** | x 的 [gatherDimSize, innerSize] 全载入 UB，SIMT 线程从 UB（而非 GM）读 index + x 值，直接写 y 到 GM |
| **UB 使用** | x GA 块全载（单 buffer） |
| **代码** | [.template](templates/dav310/gather_full_load.template) · [.md](templates/full_load.md) |

### 7. SIMT 两维（2D 特化）

| 项 | 内容 |
|----|------|
| **适用** | batchSize==1 + outerSize==1（纯 2D：x[G,inner], index[G,inner], y[G,inner]） |
| **核心策略** | SIMT 线程直接 GM 读写，地址分解仅需 1 次 fast-div（yIndex → gatherI + innerI），比通用 SIMT 少 2 次 div |
| **UB 使用** | 无 |
| **代码** | [.template](templates/dav310/gather_simt_two_dim.template) · [.md](templates/simt_two_dim.md) |

## 模板选择速查

| 条件 | 倾向模板 |
|------|---------|
| **torch.gather dim==last** | **LastDim 路径**（4 级分层：batchProcess / indexSlice / xSlice+mask / scalarMode） |
| **torch.gather dim!=last，内部 Transpose 能力门控与成本门控均通过** | **kernel 内 Transpose + Gather** |
| **torch.gather dim!=last，内部路径不划算，且物理 Transpose 能力/成本门控通过** | **物理 Transpose(x/index) + 末轴 Gather + Transpose(y)** |
| **torch.gather dim!=last，small-GD tile 的 UB/成本门控通过** | **small-GD tiled full-load**；保持 PER_ELEMENT，参数化 Gx/Gi 后再推广 |
| **torch.gather dim!=last，其余连续化路径不可用或不划算** | **Scalar/SIMT**；两者必须 A/B 实测，Scalar 不是默认最优回退 |
| 散点读, GD 不可全载 UB / UB不够 Transpose | **SIMT 标量并行** |
| innerSize==1, gatherDimSize 可全载 UB | **SIMD 末轴 Gather** |
| innerSize 小, gatherDim×inner 可全载 UB | **GA 全载** |
| innerSize 大, 有连续 slice 可搬 (PER_SLICE) | **SIMD 行拷贝** |
| gatherDim×inner 可全载, outerSize==1 | **Full-Load** |
| 散点读, GD 不可全载 UB / UB不够 Transpose | **SIMT 标量并行** |
| 极小 innerSize + 2D | **SIMT 两维 / SIMD 两维(PER_SLICE)** |

> 以上是"特征—模板"对应关系，非决策树。实际选择需结合 gatherMode（PER_ELEMENT/PER_SLICE）、dim 位置、outerSize/gatherSize 规模、UB 预算、dtype 等综合判断。详见 [tiling-flow](../../../ascendc-perf-optimize/references/tiling/gather/fallback/tiling-flow.md)。
>
> **gatherMode 是模板选择的关键维度**：torch.gather（PER_ELEMENT，每位置独立索引取 1 元素）与 GatherV2/Embedding（PER_SLICE，每索引取 innerSize 个连续元素）的模板选择不同。SIMD 两维和 SIMD 行拷贝**仅适用于 PER_SLICE**。torch.gather dim!=last 只能把 Transpose 列为候选，不能跳过能力和成本门控。
>
> **Transpose 路径选择依据**：先做能力门控，再做搬运成本门控。成本模型必须同时计入 x/index/y、workspace 中转、核数和每核 workspace 行数；cache-line 理论模型只用于筛选候选，不能证明 Transpose 恒优。路由参考实现见 [gather_elements_v2_route.template](templates/gather_elements_v2_route.template)，诊断与 case4/case6 反例见 [gather_elements_v2_transpose.md](templates/gather_elements_v2_transpose.md)。
>
> **阈值规则**：禁止用 case ID、64 MiB/128 MiB 等固定输出大小或单个已知 shape 作为路由条件。优先从 SoC UB 容量、dtype/index 字节数、对齐、双缓冲布局、任务并行度和搬运成本推导候选；理论模型只负责筛选，最终用代表 case A/B 和完整用例集回归决定是否启用。

## 关键技术总览

| 技术 | 涉及模板 | 核心要点 |
|------|---------|---------|
| **stride DMA + TransDataTo5HD** | Transpose + Gather | `DataCopyPad(srcStride)` 跨步加载 + `TransDataTo5HD` UB 内转置，将 dim!=last 的散点读转为末轴顺序读 |
| **Gather 指令向量化** | Transpose + Gather | `Gather(yLocal, xLocal, offsetLocal, 0, count)` UB 内按索引向量 gather，替代逐元素标量读 |
| **workspace GM 中转** | Transpose + Gather | 每核独立 workspace 区域存转置中间结果，1 个 kernel 内完成 Transpose→Gather→ReTranspose |
| **inner tiled resident** | PER_ELEMENT small-GD | 连续搬入全部小 gather 轴平面的 inner tile，把 x 随机 GM 访问转换为连续 DMA + UB Gather |
| **SIMT 2048 线程并行** | SIMT | `asc_vf_call` + `__simt_vf__` + `LAUNCH_BOUND(2048)`，标量 GM 读并行化 |
| **Fast-Division** | SIMT | `GetUintDivMagicAndShift` 预计算 magic/shift，`Simt::UintDiv` 替代 div/mod |
| **Gather** | SIMD 末轴 / GA 全载 | Reg API `Gather(vd, xAddr, vregIndices, mask)` 一条指令 gather 多个索引 |
| **Dtype Improve** | 全部 | 小 dtype（int8/int16）的 innerSize×dtypeSize 若 32B 对齐，提升为更大 dtype 减少搬运次数 |
| **Index 缓存** | SIMD 行拷贝 | index tile 驻留 UB，滑动窗口式读取避免重复 GM 访问 |
| **负索引补偿** | SIMD 末轴 / GA 全载 | `CompareScalar<LT, 0>` → `Adds(dimSize)` → `Copy<MERGING>` 向量化补偿 |
| **多核切分** | 全部 | 按 outerSize × gatherSize 均分到 coreNum，threadNum 对齐保证线程利用率 |
| **Direct GM Access** | SIMT | 无 UB 中转，线程直接 `y[yIndex] = x[xIndex]`，消除 MTE2/MTE3 开销 |

## 适配同类算子

### torch.gather → LastDim / Transpose / small-GD / Scalar / SIMT

torch.gather 的 index 与输出同形，每个位置有独立索引 → 纯散点读。适配要点：

**路径选择（按 gatherMode = PER_ELEMENT）**：
1. **dim==last（末轴 gather）** → **LastDim 路径**（TILING_KEY=2）：内部 4 级分层
   - `batchProcess`：x 行 + index + y 放得下 UB/2 → 多行批量 Gather 向量化
   - `singleRow + indexSlice`：x 行放得下 UB/2 但 index 放不下 → 单行 x + index 切片
   - `singleRow + xSlice`：x 行放不下 UB/2 → x 切片 + mask 筛选 + Gather
   - `scalarMode`：xSliceNum > 5 或 ratio > 256 → 标量兜底（index DMA + x GM GetValue）
2. **dim!=last + 内部能力门控通过 + 内部成本低于物理 Transpose** → **kernel 内 Transpose + Gather**（TILING_KEY=1）
3. **dim!=last + 内部路径被拒绝 + 物理 Transpose 能力/成本门控通过** → **物理 Transpose×3 + 末轴 Gather**
4. **dim!=last + small-GD tiled full-load 的 UB 与成本门控通过** → **small-GD tiled full-load**；该模式可在 Transpose 前后参与候选比较，而不是依赖固定优先级
5. **连续化路径不可用或不划算** → **Scalar/SIMT**，用代表 case A/B 决定；禁止把 Scalar 当作 Transpose 失败后的无条件默认路径

> **关键**：大规模 dim!=last 的 SIMT 可能很慢，但不能据此直接选 kernel 内 Transpose。case4(int32) 的内部模板实测比 SIMT 略慢，官方快路径是物理 Transpose；case6(int8) 不受当前内部模板支持，官方同样使用物理 Transpose。
>
> **Transpose 路由策略**：参考 ops-nn API 外层 `IfUseGatherElementsV2`/`MemCheck`，检查 dtype、非 gather 轴 shape、合轴可连续性、`postDim>32`、两侧 gatherDim≥16、输入元素数上限和 workspace 搬运代理成本。只有这些条件都通过，才进入内部模板；否则继续检查物理 Transpose 的平台能力和实测/模型收益，再与 Scalar/SIMT 比较。可直接复用 [gather_elements_v2_route.template](templates/gather_elements_v2_route.template)。
>
> **Scalar 中间路径（候选，不是默认最优）**：Scalar 将 index/y 改为连续 DMA，但 x 仍通过 GM `GetValue` 逐元素随机读。当 x 随机读或标量循环主导时，它可能显著慢于 SIMT；必须与 SIMT 及 small-GD 候选 A/B。详见 [gather_elements_v2_scalar.md](templates/gather_elements_v2_scalar.md)。
>
> **LastDim 路径的 4 级分层（参考 ops-nn `GatherElementsV2LastDimTiling`）**：
> - 合轴（`MergeAxis`）：将相邻同尺寸轴合并，降低维度复杂度
> - `IfEnableBatch`：判断 x 行 + index + y 是否放得下 UB/2 → `batchProcess_` 标志
> - `DoUBSlice`：根据 batchProcess / xAlignSize / UB 容量选择 4 级策略
> - `DoScalarMode`：xSliceNum > 5 或 xAxisSize/indexAxisSize > 256 时降级为标量模式
> 详见 [gather_elements_v2_last_dim.md](templates/gather_elements_v2_last_dim.md)。
>
> **LastDim Host 适配**：torch.gather 允许非 gather 轴 `index.shape[i] <= x.shape[i]`。输出行数必须按 index shape 计算，x/index 必须保存各自的 shape 和 stride；不要沿用按 x shape 计算输出行数的模板假设。启用 LastDim 前还要比较整行 resident 流量与 direct random GM 成本，不能只判断 `dim==last`。

适配要点：
1. **地址模型**：`yIndex → (oOuter, g, c)`，`idx = index[yIndex]`，`xAddr = oOuter * G_x * innerX + idx * innerX + c`
2. **outerSame 优化**：当 `index.shape[i] == x.shape[i]` for all `i != dim` 时，`innerI == innerX`，地址公式简化
3. **位拷贝折叠**：T ∈ {uint8/16/32/64} 按 element_size 折叠，IDX_T ∈ {int8/32/64}，共 12 实例
4. **Transpose 策略**：对固定 (oOuter, c)，x[oOuter, :, c] 是 GD 个 stride=innerX 的元素，用 `DataCopyPad(srcStride=innerX)` 加载到 UB 后 `TransDataTo5HD` 转置为连续，再用 `Gather` 指令向量化取值

### Embedding → SIMD 行拷贝 / GA 全载

Embedding lookup 本质是 `gather(x=weight, index=token_ids, axis=0)`，innerSize=hidden_size：
- hidden 大 → SIMD 行拷贝（每 token 搬 hidden 个元素）
- hidden 小 → GA 全载（vocab×hidden 全载 UB，Gather）

### Segment Gather → SIMT + mask

带 segment_offsets 的 gather，每段独立 gather → SIMT 模板 + 段边界 mask 控制。
