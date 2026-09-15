# Cube MatMul → Vector Softmax/Reduce 场景设计指导

本文仅在 Step 3 的 Blaze 官方库方案已记录明确 `native_gaps`、场景索引唯一命中 `cube-matmul-then-vector-softmax-reduce` 后读取。不用于 Step 2、Step 4 或独立设计。

## 0. 输入与源码前提

### 0.1 必需输入

| 输入                            | 来源                      | 状态要求                                                                                                            |
| ------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `requirements_contract`       | Step 3 需求合同           | frozen                                                                                                              |
| `operator_interface_contract` | Step 3 接口合同           | frozen                                                                                                              |
| `matmul_base_analysis`        | Step 2 Investigation      | ready，含`abi_bindings[]`、`kernel_policy_block_scheduler_chain`、`tilingdata_params_and_scheduler_semantics` |
| `native_gaps`                 | Step 3 官方方案覆盖性分析 | 存在明确 gap：Blaze 库无 softmax epilogue                                                                           |
| `investigation_report_facts`  | Step 2                    | 具备 L0C2UB 输出能力的 BlockMmad 的 splitM CV sync 模式已记录                                                       |

### 0.2 Blaze 源码前提

| 前提                          | Investigation 须记录的事实                                                                                                                                                    |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| L0C2UB 输出路径               | BlockMmad 的 L0C2UB 输出路径使用`CopyL0C2UB` + splitM trait（如 `CopyL0C2UBTraitSplitM`）；`block_mmad_matmul_fixpipe_opti.h`、`block_mmad_qbmm_mx.h` 等均具备此能力 |
| splitM CV sync                | BlockMmad 构造 AIV`SetFlag(AIV_SYNC_AIC_FLAG, +1)`，析构 AIC `WaitFlag(flags with FLAG_ID_MAX)`，tile 循环内 Wait/Set per N-tile                                        |
| GemmUniversal 调用路径        | 对应 kernel（如`kernel_matmul_fixpipe_opti.h`）的 `operator()` → `epilogueOp.Init(params, problemShape)` → `blockMmad.Init()` → `MatmulProcess` tile loop        |
| BlockEpilogueFixpipe 接口模式 | `operator()(blockShape, dstOffset, splitM, baseM, baseN, ubDB)` 内部 N-tile 循环 + CV sync                                                                                  |
| 适用 kernel 结构特征          | 当前资产假设 5 个结构维度（见 [cv-sync §0](cv-sync-and-two-phase-entry.md)）：栈 epilogue + idle return 在 Init 之前 + MODE_4 CV sync + 6-arg 签名 + splitM。不匹配的维度需按 §0 适配方向调整 |

前提缺失时只允许一次无场景名的补充调查问题；仍缺则不生成可执行 DESIGN。

## 1. 精确匹配条件

### 1.1 设计主线

本场景支持两种行级归约变体，共享相同的两阶段执行结构和 CV sync 模式：

**变体 A — Online Softmax**（默认变体，资产 `online_softmax_per_tile_epilogue.h` + `online_softmax_cross_core_epilogue.h`）

MatMul（MM/BMM/GMM）产生最终 FP 输出 `[R,N]` 后，执行行级 online softmax：

```
logits = A @ B                    # MatMul，L0C2UB → UB
# Phase 1 (per-tile, V1):
for each N-tile:
    tileMax = reduce_max(logits[row, tile])           # 跨 N-tile 局部 max
    onlineMax = max(onlineMax, tileMax)               # 合并全局 max
    expVals = exp(logits[row, tile] - onlineMax)      # 减 + exp
    onlineSum = onlineSum * exp(oldMax - onlineMax)   # rescale 旧 sum
             + reduce_sum(expVals)                    # 累加新 sum
    write expVals → GM expWorkspace
    write onlineMax → GM onlineMax, write onlineSum → GM onlineSum
# Phase 2 (cross-core, V2):
    maxFinal = max over all cores of onlineMax         # 跨核归约
    sumFinal = Σ_c onlineSum_c * exp(onlineMax_c - maxFinal)
    output = expVals * exp(mHistory - maxFinal) / sumFinal  # 最终 rescale
```

**变体 B — Pure Reduce**（基于变体 A 资产改写，详见 [Reduce 适配指导](reduce-adaptation-guide.md)）

MatMul 产生最终 FP 输出 `[R,N]` 后，执行行级 reduce（如 reduce_max、reduce_sum、reduce_min 等），输出 `[R,1]`。相比 softmax 更简单：

```
logits = A @ B                    # MatMul，L0C2UB → UB
# Phase 1 (per-tile, V1):
for each N-tile:
    tileResult = reduce(logits[row, tile])             # 单 pass 归约（无 exp/rescale）
    onlineResult = merge(onlineResult, tileResult)     # max/min 取极值，sum 直接累加
    write onlineResult → GM partialResult
# Phase 2 (cross-core, V2):
    finalResult = merge over all cores of partialResult  # 跨核合并，直接输出
```

与 softmax 的关键差异：无 exp/rescale 逻辑、无 expWorkspace/mHistory、workspace 缩减为 1 个 buffer、V2 无 ping-pong rescale pass。详见 [Reduce 适配指导](reduce-adaptation-guide.md)。

### 1.2 有效性门禁

- 归约域为完整逻辑行 `[0, N)`，逐 N-tile 增量更新
- 两阶段执行：V1 逐 tile（AIC+AIV CV sync），V2 跨核归约（仅 AIV）
- MatMul 使用具备 L0C2UB 输出能力的 BlockMmad，splitM=1
- softmax 变体公式为标准 `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`
- reduce 变体公式为 `output[row] = reduce(logits[row, 0:N])`，reduce 为可按行合并的归约操作（如 max、sum、min 等）

### 1.3 常见误用

| 误用                               | 原因                                   | 正确场景                                   |
| ---------------------------------- | -------------------------------------- | ------------------------------------------ |
| matmul + elementwise（无 reduce）  | softmax/reduce 含跨元素归约            | `elementwise-broadcast-epilogue-fusion`  |
| matmul + GLU                       | GLU 是两路投影，不是行级归约           | `cube-matmul-then-vector-glu`            |
| matmul + per-token quant           | quant 输出 int8+scale，不是 FP 归约输出 | `cube-matmul-then-vector-pertoken-quant` |
| matmul + softmax/reduce + per-token quant | 两个独立 Vector 后处理阶段             | `unsupported`（多命中）                  |

## 2. 接口与数据流

### 2.1 Cube 输出

具备 L0C2UB 能力的 BlockMmad 通过 `CopyL0C2UB`（+ splitM trait）将 L0C 结果写入 UB。splitM=1 时每个 AIV sub-block 收到 `CeilDiv(baseM, TaskRation)` 行。

### 2.2 V1 PerTile 数据流

```
UB(matmulData) ──→ V: ReduceMax + Sub + Exp + ReduceSum ──→ UB(maxT, sumT, mmData)
GM(onlineMax, onlineSum) ──→ UB(preMax, preSum) ──→ V: merge + rescale
UB(maxT, sumT, mmData) ──→ MTE3 ──→ GM(onlineMax, onlineSum, expWorkspace, mHistory)
```

### 2.3 V2 CrossCore 数据流

```
GM(onlineMax, onlineSum) ──→ UB(allMaxCore, allSumCore) ──→ V: ComputeMaxSum ──→ UB(maxFinal, sumFinal)
GM(mHistory) ──→ UB(mHist)
GM(expWorkspace) ──→ UB(expTile[ping/pong]) ──→ V: rescale ──→ MTE3 ──→ GM(softmaxOut)
```

## 3. 三层增量合同

### 3.1 Block 层

- 复用 Blaze 库具备 L0C2UB 能力的 BlockMmad（如 `BlockMmad<MatmulMultiBlockFixpipeOpti>`、`BlockMmad<MatmulWithScaleMx>` 等），不修改
- splitM=1（PerTileEpilogue 按 sub-block 分配行）
- `ubDB=1`（首版不启用 UB ping-pong）

### 3.2 Kernel 层

- 从 `group_matmul_kernel_cv1_v2.h` 资产适配生成项目内 `matmul_softmax_kernel.h`，遵循 [cv1_v2 适配指导](cv1-v2-adaptation-for-softmax-reduce.md)
- 适配核心：保留 `tuple<V1, V2>` + `Cv1Kernel` 委托模式，重写 `RunV2` 为 `V2.Init(params) + V2.ReduceAll()`，删除现有实现特有的类型别名/Params 字段/Tensor 构造/行分配逻辑
- Phase 1 委托给 `GemmUniversal<BlockMmad, V1>`（Cv1Kernel），在作用域内闭合 CV sync 生命周期
- Phase 2 在特化内部：`SyncAll` → `V2.Init(params)` → `V2.ReduceAll()`（仅 AIV）
- kernel entry 只需 `GemmUniversal<PS, BM, tuple<V1, V2>, BS> kernel; kernel(params);`

### 3.3 Epilogue 层

**V1 PerTileEpilogue**（资产 `online_softmax_per_tile_epilogue.h`）：

- 适配标准 BlockEpilogue 接口
- `Init(params, problemShape)` 内含 `InitSyncFlag`（`if ASCEND_IS_AIV` 保护，`initialized_` guard）。workspace 初始化 + `SyncAll` 移到两阶段 kernel 的 `InitWorkspaceGlobal`（直接实现，不依赖 epilogue 实例）
- `operator()(blockShape, dstOffset, splitM, baseM, baseN, ubDB)` 内含 N-tile 循环 + CV sync
- 析构调用 `CleanUpSyncFlag`
- CV sync 常量硬编码，与 Blaze 库 `BlockEpilogueFixpipe` 一致（以 Blaze 库源码为准）

**V2 CrossCoreEpilogue**（资产 `online_softmax_cross_core_epilogue.h`）：

- 独立接口：`Init(params)` + `ReduceAll()`
- softmax 变体：内部 Phase 1 跨核归约 + Phase 2 N-tile ping-pong rescale
- reduce 变体：仅 Phase 1 跨核合并，无 Phase 2 rescale（详见 [Reduce 适配指导](reduce-adaptation-guide.md)）

## 4. Workspace 布局合同

### 4.1 Softmax 变体

```
GM workspace 布局:
  [0, cubeCoreNum*M*4)              : onlineMax  (float, [cubeCoreNum, M])
  [..., +cubeCoreNum*M*4)           : onlineSum  (float, [cubeCoreNum, M])
  [..., +M*ceil(N/baseN)*4)         : mHistory   (float, [M, numNTiles])
  [..., +M*nAlignExp*4)             : expWorkspace (float, [M, nAlignExp])

常量:
  FLOAT32_BYTES = sizeof(float) = 4
  DATA_BLOCK = 32
  ELM_PER_32B = DATA_BLOCK / FLOAT32_BYTES = 8
  nAlignExp = ceil(N / ELM_PER_32B) * ELM_PER_32B
```

### 4.2 Reduce 变体

```
GM workspace 布局:
  [0, cubeCoreNum*M*4)              : partialResult  (float, [cubeCoreNum, M])

常量:
  FLOAT32_BYTES = sizeof(float) = 4
```

reduce 变体仅需 1 个 buffer（每核每行的 partial 归约结果），无 mHistory / expWorkspace / onlineSum。详见 [Reduce 适配指导](reduce-adaptation-guide.md)。

## 5. 条件 SplitM

splitM=1 是本场景的必需条件：

- PerTileEpilogue 按 `CeilDiv(baseM, TaskRation)` 分配行，每个 AIV sub-block 独立处理
- BlockMmad 使用 `CopyL0C2UBTraitSplitM`（`DUAL_DST_SPLIT_M`）将 L0C 结果分别写入两个 sub-block 的 UB 区域
- 不启用 splitM 会导致 UB 布局不匹配（BlockMmad 写全 M 行，PerTileEpilogue 只读半行）

## 6. 验证增量

| 维度        | 覆盖                                         |
| ----------- | -------------------------------------------- |
| N-tile      | 单 N-tile（N=baseN）、多 N-tile（N>baseN）   |
| 多核        | 单核（cubeCoreNum=1）、多核（cubeCoreNum>1） |
| M 奇偶      | M 为偶数、M 为奇数（splitM 行分配 tail）     |
| 非对齐      | K/N 非 16 对齐、M 非 baseM 对齐              |
| tail        | N-tail、M-tail                               |
| 重复 launch | 相同输入连续运行多次                         |

精度门禁：`max_abs_error < 2e-3` 或 `max_rel_error < 0.01`。

## 7. 输出与门禁

DESIGN 必须冻结：

- `implementation_route: blaze_custom`
- `selected_scenario: cube-matmul-then-vector-softmax-reduce`
- `abi_crosswalk_delta`：softmax 变体追加 onlineMax/onlineSum/mHistory/expWorkspace/softmaxOut；reduce 变体追加 partialResult/outputReduce 的 operand、Params、offset 和 consumer
- `consumed_contracts`：matmul_base_analysis（BlockMmad + Scheduler + Tiling）
- `added_contracts`：V1/V2 epilogue 接口、workspace 布局、CV sync 生命周期
- `customization_scope`：仅 V1/V2 epilogue 资产和 kernel entry 编排
- `forbidden_change_scope`：Blaze 库 BlockMmad、Scheduler、GemmUniversal、Tiling 引擎
