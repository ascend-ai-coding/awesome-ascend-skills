# Online Softmax/Reduce Epilogue 设计专题

本专题详细描述 V1 PerTileEpilogue 和 V2 CrossCoreEpilogue 的内部设计（以 softmax 变体为基准，reduce 变体的差异见 [Reduce 适配指导](reduce-adaptation-guide.md)）。
是 [场景设计指导](cube-matmul-then-vector-softmax-reduce-design.md) Section 3.3 的展开。

## 1. V1 PerTileEpilogue

### 1.1 标准接口适配

PerTileEpilogue 适配 Blaze 库标准 BlockEpilogue 接口，由 `GemmUniversal<BlockMmad>`（具备 L0C2UB 能力的 BlockMmad）调用：

| 标准调用                                                          | PerTileEpilogue 实现                                                    | 调用时机                  |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------- |
| `Init(params, problemShape)`                                    | GM 指针 + UB layout + `InitSyncFlag`                              | `blockMmad.Init()` 之前 |
| `operator()(blockShape, dstOffset, splitM, baseM, baseN, ubDB)` | `Run` → N-tile 内循环 + CV sync + `RegbaseSoftmax`                 | tile loop 中每个 block    |
| 析构                                                              | `CleanUpSyncFlag` (`initialized_` guard)                          | `MatmulProcess` 返回后  |

`Init` 全部在 `if ASCEND_IS_AIV` 内，AIC 不参与。workspace 初始化 + `SyncAll` 移到两阶段 kernel 的 `InitWorkspaceGlobal`（在 `Cv1Kernel` 之前，所有核到达，直接实现初始化逻辑）。

### 1.2 InitWorkspaceGlobal（workspace 初始化）

两阶段 kernel 的 `InitWorkspaceGlobal` 全量初始化 GM workspace：

- `onlineMax[cubeCoreNum * M] = -inf`
- `onlineSum[cubeCoreNum * M] = 0`

流程：VF `Duplicate` 填充 UB（negInf + zero）→ V_MTE3 同步 → MTE3 `CopyUB2GM` 分核并行写 GM。

分核策略：`totalElems = cubeCoreNum * M`，按 `GetBlockNum() * GetTaskRation()` 均分到全部 AIV。workspace 初始化在 `InitWorkspaceGlobal` 中直接实现，`SyncAll` 保证全部完成后才进入 Phase 1。

### 1.3 UB 布局

```
UB offset:
  [0, matmulAreaBytes)         : mmData  (BlockMmad L0C2UB 写入，splitMRows * nAlignL0C)
  [..., +splitMRows*4)         : preMax  (GM onlineMax 加载到 UB)
  [..., +splitMRows*4)         : preSum  (GM onlineSum 加载到 UB)
  [..., +splitMRows*4)         : maxT    (本 tile 新 max，写回 GM)
  [..., +splitMRows*4)         : sumT    (本 tile 新 sum，写回 GM)

其中:
  splitMRows = CeilDiv(baseM, TaskRation)
  nAlignL0C = CeilDiv(baseN, ALIGN_ELEM_F32) * ALIGN_ELEM_F32
  matmulAreaBytes = splitMRows * nAlignL0C * sizeof(float)
```

### 1.4 splitM 行分配

```
halfM = CeilDiv(curM, TaskRation)       // 每个 sub-block 的行数
localRows = (curM odd) ? (halfM - GetSubBlockIdx()) : halfM
subM0 = tileM0 + GetSubBlockIdx() * halfM
wsOff = coreIdx * M + subM0             // GM workspace 偏移
```

### 1.5 RegbaseSoftmax 算法

逐行处理，每行两 pass：

**Pass 1 — ReduceMax + merge**:

```
vregAcc = -inf
for each VL chunk: LoadAlign → Max(vregAcc, vregSrc)
vregMaxReduced = Reduce<MAX>(vregAcc)
vregNewMax = Max(vregMaxReduced, vregPreMax)     // 合并历史 max
vregMaxBrc = Duplicate(vregNewMax)                // 广播到全 VL
StoreAlign(maxTAddr + r, vregNewMax)              // 写 maxT
```

**Pass 2 — Sub + Exp + accumulate sum + rescale**:

```
vregSumAcc = 0
for each VL chunk:
    LoadAlign → Sub(vregSrc, vregMaxBrc) → Exp(vregSrc)
    StoreAlign(rowSrc, vregSrc)                   // 覆写 mmData 为 exp 值
    Add(vregSumAcc, vregSrc)
vregSumReduced = Reduce<SUM>(vregSumAcc)
vregDiff = Sub(vregPreMax, vregMaxBrc)
vregScale = Exp(vregDiff)                         // rescale 旧 sum
vregPreSum = Mul(vregPreSum, vregScale)
vregSumReduced = Add(vregPreSum, vregSumReduced)
StoreAlign(sumTAddr + r, vregSumReduced)          // 写 sumT
```

### 1.6 N-tile 内循环 + CV sync

```
for nIdx in 0..nL1Iter:
    slot = 0  (ubDB=1, 恒为 0)
    CrossCoreWaitFlag<AIC_SYNC_AIV_FLAG + slot, PIPE_V>   // 等 AIC 写完 UB
    ProcessNTile(...)                                       // softmax 计算 + GM 写回
    CrossCoreSetFlag<AIV_SYNC_AIC_FLAG + slot, PIPE_MTE3>  // 释放 UB 给 AIC
```

## 2. V2 CrossCoreEpilogue

### 2.1 接口

| 方法             | 功能                                                        |
| ---------------- | ----------------------------------------------------------- |
| `Init(params)` | 计算 UB 布局、GM 指针、行分配（按`vecCoreNum` 均分 M 行） |
| `ReduceAll()`  | 分批处理`myRows_` 行，每批调用 `ProcessBatch`           |

### 2.2 行分配

```
perCore = M / vecCoreNum
tail = M % vecCoreNum
myRows = perCore + (blk < tail ? 1 : 0)
mStart = (blk < tail) ? blk * (perCore+1) : blk * perCore + tail
```

### 2.3 UB 布局

```
UB offset (5 buffers, each 32B aligned):
  allMaxCore : maxLoopNum * cubeCoreNum     // 跨核 max 数据
  allSumCore : maxLoopNum * cubeCoreNum     // 跨核 sum 数据
  mHist      : maxLoopNum * numTiles        // mHistory 切片
  expTile[0] : maxLoopNum * nAlignTile      // ping
  expTile[1] : maxLoopNum * nAlignTile      // pong

maxLoopNum = (UB_SIZE - DATA_BLOCK*5) / (sizeof(float) * (2*cubeCoreNum + numTiles + 2*nAlignTile))
```

### 2.4 ProcessBatch — 两阶段

**Phase 1: ComputeMaxSum**:

```
MTE2: CopyGM2UB(onlineMax → allMaxCore, onlineSum → allSumCore, mHistory → mHist)
V: for each row r:
    maxFinal = max over cores of allMax[core][r]
    sumFinal = Σ_c allSum[core][r] * exp(allMax[core][r] - maxFinal)
    StoreAlign(maxFinal, sumFinal)  // 覆写 allMaxCore/allSumCore 前 cur 行
```

**Phase 2: N-tile ping-pong rescale**:

```
for t in 0..numTiles:
    bufId = t & 1
    if t >= 2: WaitFlag<MTE3_MTE2>(bufId)        // 等上一轮同 buffer 的 MTE3
    MTE2: CopyGM2UB(expWorkspace[t] → expTile[bufId])
    V: rescale: expTile *= exp(mHist[t] - maxFinal) / sumFinal
    MTE3: CopyUB2GM(expTile[bufId] → softmaxOut)
    SetFlag<MTE3_MTE2>(bufId)                     // 释放 buffer
Drain: WaitFlag<MTE3_MTE2>(0), WaitFlag<MTE3_MTE2>(1)
```

### 2.5 RegbaseRescaleV

```
vregMaxF = LoadAlign(maxFinalAddr + r)     // 从 UB 读 maxFinal
vregSumF = LoadAlign(sumFinalAddr + r)     // 从 UB 读 sumFinal
vregMHist = LoadAlign(mHistAddr + r * numTiles + t)  // 本 tile 的 max
vregScale = Exp(Sub(vregMHist, vregMaxF))
vregTotalScale = Div(vregScale, vregSumF)
vregScaleBrc = Duplicate(vregTotalScale)
for each VL chunk: expTile *= vregScaleBrc   // 原地覆写
```

## 3. Workspace 布局常量

| 常量               | 值                                          | 用途                              |
| ------------------ | ------------------------------------------- | --------------------------------- |
| `FLOAT32_BYTES`  | `sizeof(float)` = 4                       | 字节计算                          |
| `DATA_BLOCK`     | 32                                          | 32B 对齐                          |
| `ELM_PER_32B`    | `DATA_BLOCK / FLOAT32_BYTES` = 8          | 32B 对齐的 float 元素数           |
| `ALIGN_ELEM_F32` | `32 / sizeof(float)` = 8                  | L0C/UB N 向对齐（同 ELM_PER_32B） |
| `nAlignExp`      | `ceil(N / ELM_PER_32B) * ELM_PER_32B`     | expWorkspace GM 行间距            |
| `nAlignTile`     | `ceil(baseN / ELM_PER_32B) * ELM_PER_32B` | expTile UB 行间距                 |
