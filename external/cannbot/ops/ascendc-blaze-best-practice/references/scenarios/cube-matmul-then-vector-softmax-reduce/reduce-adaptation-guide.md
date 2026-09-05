# Pure Reduce 适配指导

本指导说明如何在 softmax 资产基础上改写为纯行级 reduce（如 reduce_max、reduce_sum、reduce_min 等）的 V1/V2 Epilogue。不新增独立资产文件，在项目 `blaze_custom/epilogue/` 中基于 softmax 资产改写。

## 1. 何时需要本指导

- 场景为 `cube-matmul-then-vector-softmax-reduce`，归约变体为纯 reduce（非 softmax）
- 已读取 [场景设计指导](cube-matmul-then-vector-softmax-reduce-design.md)
- 已读取 [Online Softmax/Reduce Epilogue 设计专题](online-softmax-reduce-epilogue-design.md)
- V1/V2 softmax 资产已复制到项目

## 2. Softmax 与 Pure Reduce 的差异总览

| 层 | Softmax | Pure Reduce | 简化程度 |
|----|---------|-------------|---------|
| **V1 算法** | 两 pass：ReduceMax + Sub/Exp/ReduceSum + rescale | 单 pass：reduce 一次 | 删除 Pass 2 全部逻辑 |
| **V1 workspace 写回** | onlineMax + onlineSum + expWorkspace + mHistory | partialResult 仅 1 个 buffer | 4 → 1 |
| **V2 算法** | Phase 1 跨核归约 maxFinal/sumFinal + Phase 2 N-tile ping-pong rescale | 仅跨核合并，直接输出 | 删除 Phase 2 全部逻辑 |
| **V2 UB 布局** | allMaxCore + allSumCore + mHist + expTile[2] | partialResultCore 仅 1 个 buffer | 5 → 1 |
| **GM workspace** | onlineMax + onlineSum + mHistory + expWorkspace | partialResult 仅 1 个 | 4 → 1 |
| **输出** | `[R,N]`（rescale 后的 exp 值） | `[R,1]`（每行一个归约值） | — |

## 3. V1 PerTileEpilogue 改写

基于 `online_softmax_per_tile_epilogue.h` 改写为 `reduce_per_tile_epilogue.h`。

### 3.1 Params 简化

```cpp
// softmax Params（删除）:
//   softmaxOutAddr, mHistoryAddr, expWorkspaceAddr
//   onlineMaxAddr + onlineSumAddr → 合并为 partialResultAddr

// reduce Params:
struct Params {
    GM_ADDR outputReduceAddr{nullptr};       // [R,1] 最终输出（V2 写）
    GM_ADDR partialResultAddr{nullptr};      // [cubeCoreNum, M] per-core partial
    uint32_t cubeCoreNum{0};
    uint32_t m{0};
    uint32_t n{0};
};
```

### 3.2 UB 布局简化

```
// softmax UB（删除）:
//   preMax, preSum, maxT, sumT  — 4 个 buffer

// reduce UB:
//   [0, matmulAreaBytes)        : mmData  (BlockMmad L0C2UB 写入)
//   [..., +splitMRows*4)        : resultT (本 tile 归约结果，写回 GM)
```

### 3.3 InitWorkspaceGlobal 初始化值简化

softmax 的 `InitWorkspaceGlobal` 初始化 onlineMax=-inf + onlineSum=0。reduce 的 `InitWorkspaceGlobal` 只需初始化 partialResult，以三种常见 reduce 为例：

- reduce_max：初始化为 `-inf`
- reduce_min：初始化为 `+inf`
- reduce_sum：初始化为 `0`

### 3.4 归约算法替换

将 `RegbaseSoftmax` + `ProcessRowPass1` + `ProcessRowPass2` 三个函数替换为单个 `RegbaseReduce` 函数：

```cpp
__aicore__ inline void RegbaseReduce(int64_t localRows, int64_t curN)
{
    uint16_t rows = static_cast<uint16_t>(localRows);
    int64_t nAlignCur = CeilDiv(curN, ALIGN_ELEM_F32) * ALIGN_ELEM_F32;
    uint16_t vfN = static_cast<uint16_t>(CeilDiv(static_cast<uint64_t>(curN), VL_));
    uint16_t mainLoop = vfN - 1;
    uint32_t tailActive = static_cast<uint32_t>(curN) % VL_;
    if (tailActive == 0) { tailActive = VL_; }

    __VEC_SCOPE__
    {
        Reg::MaskReg allMask = Reg::CreateMask<ComputeType, Reg::MaskPattern::ALL>();
        Reg::MaskReg tailMask = Reg::UpdateMask<ComputeType>(tailActive);

        for (uint16_t r = 0; r < rows; ++r)
        {
            __ubuf__ ComputeType* rowSrc = mmDataAddr_ + r * nAlignCur;

            // 单 pass：load + reduce
            Reg::RegTensor<ComputeType> vregAcc;
            Reg::RegTensor<ComputeType> vregSrc;
            // 初始化 acc（max→-inf, sum→0, min→+inf）
            Reg::Duplicate(vregAcc, initValue_, allMask);

            for (uint16_t i = 0; i < mainLoop; ++i) {
                Reg::LoadAlign(vregSrc, rowSrc + i * VL_);
                ReduceOp(vregAcc, vregSrc, vregAcc, allMask);  // Max/Add/Min
            }
            Reg::LoadAlign(vregSrc, rowSrc + mainLoop * VL_);
            ReduceOp<ComputeType, Reg::MaskMergeMode::MERGING>(vregAcc, vregSrc, vregAcc, tailMask);

            // 标量归约
            Reg::RegTensor<ComputeType> vregReduced;
            Reg::Reduce<ReduceType>(vregReduced, vregAcc, allMask);

            // 与 GM partial 合并
            Reg::RegTensor<ComputeType> vregPre;
            Reg::LoadAlign<ComputeType, Reg::LoadDist::DIST_BRC_B32>(vregPre, preResultAddr_ + r);
            MergeOp(vregReduced, vregPre, vregReduced, allMask);  // Max/Add/Min

            Reg::StoreAlign<ComputeType, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                resultTAddr_ + r, vregReduced, allMask);
        }
    }
}
```

其中 `ReduceOp`/`MergeOp`/`ReduceType`/`initValue_` 按变体选择，以三种常见 reduce 为例：

| Reduce 变体 | `ReduceOp` | `MergeOp` | `ReduceType` | `initValue_` |
|-------------|-----------|-----------|-------------|-------------|
| reduce_max | `Reg::Max` | `Reg::Max` | `MAX` | `-inf` |
| reduce_sum | `Reg::Add` | `Reg::Add` | `SUM` | `0` |
| reduce_min | `Reg::Min` | `Reg::Min` | `MIN` | `+inf` |

其他 reduce 类型按其合并语义选择对应算子和初始值。

### 3.5 ProcessNTile 简化

删除 softmax 的 4 个 GM 写回（onlineMax/onlineSum/expWorkspace/mHistory），替换为 1 个：

```cpp
// softmax 写回（删除）:
//   CopyUbToGm1D(gmMaxT, ...)
//   CopyUbToGm1D(gmSumT, ...)
//   CopyUbToGm2D(gmExpWsT, ...)
//   CopyUbToGm2D(gmMHistoryT, ...)

// reduce 写回:
CopyUbToGm1D(gmPartialResult, wsOff, ubOffResultT_, localRows);
```

## 4. V2 CrossCoreEpilogue 改写

基于 `online_softmax_cross_core_epilogue.h` 改写为 `reduce_cross_core_epilogue.h`。

### 4.1 删除 Phase 2

softmax V2 含两个 Phase：
- Phase 1: ComputeMaxSum（跨核归约 maxFinal/sumFinal）
- Phase 2: N-tile ping-pong rescale（逐 tile 重新缩放 expWorkspace → softmaxOut）

reduce V2 只需 Phase 1 的简化版（跨核合并 partialResult → outputReduce），**删除整个 Phase 2**。

### 4.2 UB 布局简化

```
// softmax V2 UB（删除）:
//   allMaxCore, allSumCore, mHist, expTile[0], expTile[1] — 5 个 buffer

// reduce V2 UB:
//   partialCore : maxLoopNum * cubeCoreNum  — 跨核 partial 数据
```

### 4.3 ReduceAll 简化

```cpp
__aicore__ inline void ReduceAll()
{
    // Phase 1: 跨核合并（仅此阶段）
    for each batch of myRows_:
        MTE2: CopyGM2UB(partialResult → partialCore)
        V: for each row r:
            merged = merge over cores of partialCore[core][r]
            StoreAlign(outputReduceAddr + r, merged)
        MTE3: CopyUB2GM → GM outputReduce
    // Phase 2: 已删除（无 rescale）
}
```

## 5. 可直接复用的部分

以下部分与 softmax 变体完全一致，无需改写：

| 部分 | 说明 |
|------|------|
| CV sync 模式 | 构造 SetFlag / 析构 WaitFlag / tile loop 内 Wait+Set，常量以 Blaze 库为准 |
| Kernel 编排 | `GemmUniversal<BlockMmad, tuple<V1, V2>, Scheduler>` 特化，同 [cv1_v2 适配指导](cv1-v2-adaptation-for-softmax-reduce.md) |
| BlockMmad 选型 | 具备 L0C2UB + splitM 能力的 BlockMmad，不修改 |
| splitM=1 | 行分配逻辑不变 |
| ubDB=1 | 单缓冲限制不变 |
| workspace 初始化 + SyncAll + InitSyncFlag | workspace 初始化移到两阶段 kernel `InitWorkspaceGlobal`（直接实现）；`Init` 只保留 `InitSyncFlag`；析构需 `initialized_` guard；reduce 初始化值为 0（sum）/ -inf（max）/ +inf（min） |
| CleanUpSyncFlag | HardEvent 生命周期配平不变（`initialized_` guard 保护 idle 核） |
| idle 核处理 | idle 核通过 `InitWorkspaceGlobal`（两阶段 kernel）参与 workspace 初始化；析构需 `initialized_` guard |

## 6. 验证检查清单

| 检查项 | 预期 |
|--------|------|
| V1 归约算法 | 单 pass，无 exp/rescale 逻辑 |
| V1 GM 写回 | 仅 partialResult 1 个 buffer |
| V2 Phase 2 | 已删除 |
| V2 输出 | `[R,1]`，每行一个归约值 |
| GM workspace | 仅 partialResult（`[cubeCoreNum, M]`） |
| CV sync 常量 | 与 Blaze 库 BlockMmad 一致 |
| ReduceOp/MergeOp | 按所选 reduce 类型的合并语义正确选择 |
| initValue_ | 按所选 reduce 类型的合并语义正确选择（如 max→-inf, sum→0, min→+inf） |
