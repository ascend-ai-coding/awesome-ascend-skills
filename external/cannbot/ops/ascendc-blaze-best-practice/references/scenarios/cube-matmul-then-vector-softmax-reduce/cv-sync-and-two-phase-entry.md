# CV Sync 与两阶段 Kernel 编排专题

本专题详细描述适配后的 `matmul_softmax_kernel.h`（从 `group_matmul_kernel_cv1_v2.h` 适配）的两阶段编排和 CV sync 生命周期。
是 [场景设计指导](cube-matmul-then-vector-softmax-reduce-design.md) Section 3.2 的展开，适配步骤见 [cv1_v2 适配指导](cv1-v2-adaptation-for-softmax-reduce.md)。

## 0. 适用范围

本专题的资产和编排方案基于以下 5 个 kernel 结构特征。适配时须逐一核对，不匹配的维度需按"适配方向"调整：

| 结构维度 | 当前资产假设 | 不匹配时的适配方向 |
|----------|-------------|-------------------|
| **epilogue 存储模型** | 栈对象（ctor/dtor 在所有核触发，含 idle） | 若为类成员：析构无死锁风险，`initialized_` guard 非必需 |
| **idle 核返回时机** | return 在 `Init` 之前 | 若 return 在 `Init` 之后或无 return：workspace 初始化 + `SyncAll` 可保留在 `Init` 中，无需 `InitWorkspaceGlobal` |
| **CV sync 机制** | ctor/dtor 中 CrossCore set/wait（MODE_4） | 若为 per-tile NotifyVector/WaitForCube：epilogue 需在 tile loop 内匹配 helper 调用，不依赖 ctor/dtor |
| **epilogue 调用签名** | `(blockShape, dstOffset, splitM, baseM, baseN, ubDB)` | 若不同：epilogue `operator()` 签名需匹配实际 kernel 传入的参数 |
| **splitM 支持** | 支持（splitM=1） | 若不支持：行分配逻辑需改为全 M 行，不用 `CeilDiv(baseM, TaskRation)` |

适配者只需检查自己的 kernel 在这 5 个维度上的实际行为，对照即可确定哪些部分可直接复用、哪些需要调整。

## 1. 两阶段 Kernel 编排

适配后的 kernel 是 `GemmUniversal` 的 `tuple<V1, V2>` 特化。kernel entry 只需：

```cpp
using EpiloguePipeline = AscendC::Std::tuple<PerTileEpilogue, CrossCoreEpilogue>;
using KernelImpl = GemmUniversal<ProblemShape, BlockMmad, EpiloguePipeline, BlockScheduler>;

typename KernelImpl::Params kParams{
    {problemShape, mmadParams, perTileParams, schParams},  // cv1Params
    crossCoreParams};                                        // epilogueV2Params

KernelImpl kernel;
kernel(kParams);
```

特化内部的 `operator()` 编排两阶段：

```cpp
// Phase 1: MatMul + PerTile online softmax (AIC BlockMmad + AIV PerTileEpilogue)
{
    Cv1Kernel cv1Kernel;
    cv1Kernel(params.cv1Params);
}

// Phase 2: Cross-core reduction + final rescale (AIV only)
AscendC::SyncAll();
if ASCEND_IS_AIV {
    RunV2(params);
}
```

`RunV2` 适配后只需 3 行：

```cpp
__aicore__ inline static void RunV2(Params const& params)
{
    BlockEpilogueV2 epilogueV2;
    epilogueV2.Init(params.epilogueV2Params);
    epilogueV2.ReduceAll();
}
```

### Phase 1 内部流程（Cv1Kernel = GemmUniversal<BlockMmad, V1>）

```
构造: AIV SetFlag(AIV_SYNC_AIC_FLAG, +1)              — 预置 ping+pong UB 空闲
epilogueOp.Init(params, problemShape):
    AIV: workspace 初始化 → SyncAll → InitSyncFlag()
    AIC: SyncAll()
blockMmad.Init(params.mmadParams)                       — AIC only, L1/L0 布局
MatmulProcess:                                          — tile loop
    for each block:
        AIC: blockMmad(gmBlockA, gmBlockB, gmBlockBias, ubLocal, validBlockShape)
            内部: WaitFlag(AIV_SYNC_AIC_FLAG) → CopyL0C2UB → SetFlag(AIC_SYNC_AIV_FLAG)
        AIV: epilogueOp(validBlockShape, offsetC, splitM, baseM, baseN, ubDB)
            内部: WaitFlag(AIC_SYNC_AIV_FLAG) → PerTileEpi → SetFlag(AIV_SYNC_AIC_FLAG)
析构: AIC WaitFlag(4 flags with FLAG_ID_MAX)            — drain
      AIV CleanUpSyncFlag()                             — 消耗 HardEvent
```

### Phase 2 内部流程

```
SyncAll()                                               — 所有核到达
AIV: CrossCoreEpi.Init(params)
     CrossCoreEpi.ReduceAll()
         Phase 1: ComputeMaxSum (跨核归约 maxFinal/sumFinal)
         Phase 2: N-tile ping-pong rescale → GM softmaxOut
```

## 2. CV Sync 常量

CV sync 常量硬编码在 PerTileEpilogue 中，必须与 Blaze 库 `BlockMmad` 和 `BlockEpilogueFixpipe` 的当前实现保持一致。Blaze 库演进时这些值可能变化，适配时以 Blaze 库源码为准：

| 常量                    | 方向     | 含义         |
| ----------------------- | -------- | ------------ |
| `AIC_SYNC_AIV_MODE_4` | —       | CV sync mode |
| `AIV_SYNC_AIC_FLAG`   | AIV→AIC | UB 槽位空闲  |
| `AIC_SYNC_AIV_FLAG`   | AIC→AIV | UB 数据就绪  |

不依赖 `cv_sync_constants.h` 资产文件，遵循 Blaze 库 `BlockEpilogueFixpipe` 的相同模式（在类内 `static constexpr` 硬编码）。

## 3. FLAG_ID_MAX 语义

`__mix__(1, 2)` 模式下，硬件自动将 AIV sub-block 1 的 flag N 映射到 AIC 的 flag N+FLAG_ID_MAX。FLAG_ID_MAX 的值由 `__mix__` 模式决定，取值以 Blaze 库当前实现为准。

- **AIC 侧**：在 splitM=1 时，每个 N-tile 的 `WaitFlag`/`SetFlag` 操作 4 个 flag（`AIV_SYNC_AIC_FLAG+slot`、`+slot+FLAG_ID_MAX`、`+slot+1`、`+slot+1+FLAG_ID_MAX`），对应 ping/pong × sub-block 0/1。
- **AIV 侧**：只操作 2 个 flag（`AIC_SYNC_AIV_FLAG+slot`、`+slot+1`），不显式操作 `+FLAG_ID_MAX`（硬件自动映射）。

## 4. InitWorkspaceGlobal

workspace 初始化 + `SyncAll` 已从 PerTileEpilogue 的 `Init` 中移除，移到两阶段 kernel 的 `InitWorkspaceGlobal`（在 `Cv1Kernel` 之前执行，直接实现初始化逻辑，不依赖 epilogue 实例）：

- 所有核（含 idle）在 `InitWorkspaceGlobal` 执行路径上（两阶段 kernel 的 `operator()` 无 idle return）
- AIV 执行 workspace 初始化（VF `Duplicate` 填充 UB → `V_MTE3` 同步 → MTE3 `CopyUB2GM` 分核并行写 GM），AIC 直接到达 `SyncAll` 等待
- 分核策略：`totalElems = cubeCoreNum * M`，按 `GetBlockNum() * GetTaskRation()` 均分到全部 AIV
- softmax 变体：onlineMax = -inf, onlineSum = 0；reduce 变体：partialResult 按归约类型初始化
- `SyncAll` 保证 workspace 初始化完成后才进入 Phase 1
- 完整实现见 [cv1_v2 适配指导](cv1-v2-adaptation-for-softmax-reduce.md) §5 代码骨架中的 `InitWorkspaceGlobal`

**原因**：`kernel_matmul_fixpipe_opti.h` 在 idle 核 return 之后才调用 `epilogueOp.Init()`，idle 核不到达 `Init` 中的 `SyncAll` → 死锁。移到两阶段 kernel 后，所有核都经过 `InitWorkspaceGlobal`。

PerTileEpilogue 的 `Init` 只保留参数设置 + `InitSyncFlag`（`if ASCEND_IS_AIV` 保护）。

## 5. Idle 核处理

`kernel_matmul_fixpipe_opti.h` 的 idle 核处理：

```
curBlockIdx >= realCoreNums → return (跳过 Init + tile loop)
```

- **Phase 0 `InitWorkspaceGlobal`**（两阶段 kernel）：所有核到达，包括 idle 核
- **Phase 1 CV sync 构造/析构**：idle 核也构造/析构 epilogue 栈对象。CrossCore sync（构造 set flag，析构 wait flag）自身配平。HardEvent sync 只在 active 核执行（idle 核未调 `Init` → `initialized_=false` → 析构跳过 `CleanUpSyncFlag`）
- **Phase 1 `SyncAll`**（两阶段 kernel Phase 2 之前）：所有核到达
- **Phase 2 `ReduceAll`**：`myRows_==0` 时内部直接 return

## 6. PerTileEpilogue 的 HardEvent 生命周期

PerTileEpilogue 内部使用 3 个 HardEvent flag（`ZERO_FLAG=0`）做 MTE2/V/MTE3 管线同步：

| 事件          | set 时机          | wait 时机            | 用途                        |
| ------------- | ----------------- | -------------------- | --------------------------- |
| `V_MTE2`    | V 计算完成        | 下一 tile MTE2 读 GM | 保证 V 写完 UB 后 MTE2 才读 |
| `MTE3_V`    | MTE3 写回完成     | V 读取前             | 保证 MTE3 写完 UB 后 V 才读 |
| `MTE3_MTE2` | MTE3 写回 GM 完成 | 下一 tile MTE2 读 GM | 保证 GM 数据可见            |

`InitSyncFlag` 在 tile loop 前预置这 3 个 flag（set 一次）。
`CleanUpSyncFlag` 在 tile loop 后消耗这 3 个 flag（wait 一次）。
配平关系：每个 flag 在整个生命周期中 set 次数 = wait 次数 + 1（InitSyncFlag 预置 1 个，CleanUpSyncFlag 消耗 1 个）。

## 7. 模板匹配优先级

适配后的特化与 `cv1_v2` 的 `Enable_` 都是 `void`（默认），模板参数列表相同。如果两个特化同时存在于编译单元中，编译器会报歧义错误。

**解决方法**：项目只需包含适配后的 `matmul_softmax_kernel.h`，不同时包含原始 `group_matmul_kernel_cv1_v2.h`。如果项目同时需要两个场景（per-token-quant + softmax），则需要在适配后的特化中增加 ScheduleType 约束以区分。
