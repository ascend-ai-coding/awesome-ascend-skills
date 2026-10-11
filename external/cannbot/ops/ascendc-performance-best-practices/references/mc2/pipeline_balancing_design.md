# MC² 流水配平优化设计

> 本文档为**实现层**设计指南，提供 TilingData 结构、Host 侧搜索代码、Kernel 核心循环和修改点等实施细节。对应的**分析层**框架（Bound 判定、配平公式、搜索算法、膨胀分析、性能指标）详见 `/ascendc-perf-optimize` 的 [comm-compute/pipeline_balancing.md](../../../ascendc-perf-optimize/references/comm-compute/pipeline_balancing.md) 和 [comm-compute/index.md](../../../ascendc-perf-optimize/references/comm-compute/index.md)。实施前须先通过分析层完成 TilingData 采集和隔离测试。

## 1. 优化目标

将 MC² 通算融合算子从**均匀切分、通信与计算串行执行**改为**长短块配平、AIC/AIV 分离流水掩盖**，通过长短块排布最大化通信与计算的重叠度，预期通算掩盖率从 50%–70% 提升至 85%+。

> Local Matmul（本 rank 数据独立计算）是与流水配平并列的独立策略，详见 [local_matmul_design.md](local_matmul_design.md)。

## 2. 架构概览

### 2.1 AIC/AIV 分离架构

MC² 通算融合算子采用 AIC（计算核）+ AIV（向量核）分离架构，通过 CrossCore 事件同步实现通信与计算的流水掩盖：

```
┌─────────────────────────────────────────────────┐
│                    AIV（向量核）                   │
│  职责：通信（AllToAll via UDMA / shmem）          │
│  同步：CrossCoreSetFlag → 通知 AIC 数据就绪        │
│        WaitComputeComplete → 等 AIC 计算完成       │
└──────────────────────┬──────────────────────────┘
                       │ CrossCore 事件同步
┌──────────────────────┴──────────────────────────┐
│                    AIC（计算核）                   │
│  职责：matmul 计算（QBMM + Scale）                │
│  同步：CrossCoreWaitFlag → 等 AIV 数据就绪         │
│        NotifyComputeComplete → 通知 AIV 计算完成   │
└─────────────────────────────────────────────────┘
```

### 2.2 通算流水模型

按 matmul 结果矩阵的行（M）切分为多个块，每块先做通信再做计算（或反之），相邻块的通信与计算重叠执行。

**通信后计算（如 alltoall + matmul）**：

```
AIV: PutToAllRanks(块0) → SetFlag(0) → PutToAllRanks(块1) → SetFlag(1) → ...
AIC:                         WaitFlag(0) → DoMatmul(块0) → WaitFlag(1) → DoMatmul(块1) → ...
                                         ↕ 计算与通信重叠
```

**计算后通信（如 matmul + alltoall）**：

```
AIC: DoMatmul(块0) → NotifyCompute(0) → DoMatmul(块1) → NotifyCompute(1) → ...
AIV:                  WaitCompute(0) → BarrierAll → AlltoAll(块0) → WaitCompute(1) → ...
                                       ↕ 通信与计算重叠
```

### 2.3 长短块排布

均匀切分时，首块或尾块的通信/计算无法被掩盖，造成固定开销。通过引入**长短块**，将无法被掩盖的块设为短块（耗时最小化），长块保证高 Mac 利用率。短块数量最多一块；若长块长度与短块长度一致，则只需要保留长块，短块可以不做相关配置。

短块位置由执行顺序 × Bound 类型决定，通过 `shortBlockPos` flag 控制：

```
短块在前 (shortBlockPos=0): shortBlockCnt 个短块（shortMSize） + longBlockCnt 个长块（longMSize）
短块在后 (shortBlockPos=1):  longBlockCnt 个长块（longMSize） + shortBlockCnt 个短块（shortMSize）

示例（M=2304, longMSize=512, 短块在后）：
  longBlockCnt=4, longMSize=512, shortBlockCnt=1, shortMSize=256
  执行序列：长块1(512) → 长块2(512) → 长块3(512) → 长块4(512) → 短块(256)
```

> 长块和短块使用**各自独立的完整 matmul tiling data**，因为不同 M 大小导致不同的 L1/L0 切分策略。

## 3. 关键参数

### 3.0 TilingData 打印（实施前置条件）

> ⚠️ 流水配平的搜索算法依赖 baseM、baseN、usedCoreNum 等 matmul tiling 参数。实施前**必须**确保 host 代码打印这些参数，否则无法进行核利用率计算和搜索范围推导。

host 代码的 tiling 引擎通常包含 `PrintTilingData` 方法但被注释掉。参考 `quant_matmul_tiling_base.h`：

```cpp
void GetTilingData(uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB,
                   QuantMatmulTilingData& tilingData)
{
    ...
    DoOpTiling(tilingData);
    PrintTilingData(tilingData);  // ← 取消注释，启用打印
}

void PrintTilingData(const QuantMatmulTilingData& tilingData) const
{
    printf("[QuantMatmul Strategy]\n");
    printf("  strategy           : %s\n", TilingName());
    printf("[QuantMatmul Tiling Data]\n");
    printf("  usedCoreNum        : %u\n", tilingData.usedCoreNum);
    printf("  m                  : %u\n", tilingData.m);
    printf("  n                  : %u\n", tilingData.n);
    printf("  k                  : %u\n", tilingData.k);
    printf("  baseM              : %u\n", tilingData.baseM);
    printf("  baseN              : %u\n", tilingData.baseN);
    printf("  baseK              : %u\n", tilingData.baseK);
    printf("  stepK              : %u\n", tilingData.stepK);
    printf("  nBufferNum         : %u\n", tilingData.nBufferNum);
    printf("  dbL0c              : %u\n", tilingData.dbL0c);
}
```

**若算子无 PrintTilingData 方法**，须在 host tiling 计算完成后添加 printf 打印以下必需字段：

| 字段 | 用途 |
|------|------|
| `baseM` | 计算 mBlockCnt = ceil(longMSize / baseM) → 核利用率 |
| `baseN` | 计算 nBlockCnt = ceil(N / baseN) → 核利用率 |
| `usedCoreNum` | 确认可用核数 N_core |
| `baseK` / `stepK` | 辅助分析 L1/L0 切分合理性 |

> 长块和短块各有独立的 matmul tiling data（不同 M → 不同 baseM/baseN）。须对长块和短块分别打印 tiling data。

### 3.1 TilingData 结构

#### 通信后计算（以 alltoall + matmul 为例）

```cpp
struct AllToAllCommTilingData {
    uint32_t tileCnt;        // 总块数 = longBlockCnt + shortBlockCnt
    uint32_t longBlockCnt;   // 长块数量 = M / longMSize
    uint32_t longMSize;      // 长块 M 维大小
    uint32_t shortBlockCnt;  // 短块数量（0 或 1）
    uint32_t shortMSize;     // 短块 M 维大小 = M % longMSize
    uint8_t  shortBlockPos;  // 短块位置: 0=前(front), 1=后(back)
};

struct allToAllMatmulTilingData {
    AllToAllCommTilingData commTilingData;
    QuantMatmulTilingData longQbmmTilingData;    // 长块 matmul tiling
    QuantMatmulTilingData shortQbmmTilingData;   // 短块 matmul tiling（shortBlockCnt > 0 时有效）
    QuantMatmulTilingData localQbmmTilingData;   // local matmul tiling（详见 local_matmul_design.md）
};
```

#### 计算后通信（以 matmul + alltoall 为例）

```cpp
struct MatmulAllToAllTilingData {
    QuantBatchMatmulTilingData mmLong;    // 长块 matmul tiling
    QuantBatchMatmulTilingData mmShort;   // 短块 matmul tiling
    QuantBatchMatmulTilingData mmLocal;   // local matmul tiling（详见 local_matmul_design.md）
    uint32_t tileCnt;        // 总块数 = longBlockCnt + shortBlockCnt
    uint32_t longBlockCnt;   // 长块数量
    uint32_t longMSize;      // 长块 M 维大小
    uint32_t shortBlockCnt;  // 短块数量（0 或 1）
    uint32_t shortMSize;     // 短块 M 维大小
    uint8_t  shortBlockPos;  // 短块位置: 0=前(front), 1=后(back)
};
```

### 3.2 Host 侧参数计算

#### 切分方案搜索（TilingData 驱动）

longMSize 的选择不应凭经验取固定值（如 512），而应基于 TilingData 打印的 baseM/baseN/usedCoreNum 和隔离测试的 α/β/R 值，执行系统搜索：

```cpp
// ============ 搜索输入（从 TilingData 打印和隔离测试获取）============
// M, N: 算子 shape
// baseM, baseN, N_core: 从 TilingData 打印获取
// α (comm μs/row), β (compute μs/row), R = β/α: 从隔离测试获取

// ============ 搜索范围推导 ============
uint32_t nBlockCnt  = (N + baseN - 1) / baseN;
float    utilThresh = (R >= 1.0f) ? 0.80f : 0.50f;  // 计算 bound 80%，通信 bound 50%
uint32_t minTotalBlk = (uint32_t)ceil(utilThresh * N_core);
uint32_t minMBlkCnt  = max(1u, (minTotalBlk + nBlockCnt - 1) / nBlockCnt);
uint32_t hs_min = max(baseM, (minMBlkCnt - 1) * baseM + 16);
hs_min = (hs_min + 15) & ~15u;                        // 16 对齐
uint32_t hs_max = (M / 2) & ~15u;                      // 至少 2 块
uint32_t tcFloor = 2;
uint32_t tcCap   = (R >= 1.0f) ? min((uint32_t)ceil(R) + 2, 8u)
                                : min((uint32_t)ceil(1.0f/R) + 2, 16u);

// ============ 扫描所有候选 hs ============
struct Candidate { uint32_t hs, hc, tail, tc; float estT; float util; };
vector<Candidate> candidates;

for (uint32_t hs = hs_min; hs <= hs_max; hs += 16) {
    uint32_t hc   = M / hs;
    uint32_t tail = M % hs;

    // P1: 尾块下限（128 行以下排除，Mac 利用率极低）
    if (tail > 0 && tail < 128) continue;

    // P3: tileCnt 范围
    uint32_t tc = hc + (tail > 0 ? 1 : 0);
    if (tc < tcFloor || tc > tcCap) continue;

    // P2: 核利用率（主轮 = 长块）
    uint32_t mBlk = (hs + baseM - 1) / baseM;
    uint32_t totalBlk = mBlk * nBlockCnt;
    float util = min(totalBlk, N_core) / (float)N_core;
    if (util < utilThresh) continue;

    // P4: 配平条件
    float rho = max(R, 1.0f / R);
    if (tail > 0 && tail < rho * hs) continue;

    // 理论时间估算（计算 bound 示例）
    float T = alpha * hs                              // T_fill
            + (hc - 1) * beta * hs                    // T_steady
            + max(alpha * tail, beta * hs)            // T_transition
            + beta * tail;                            // T_drain

    candidates.push_back({hs, hc, tail, tc, T, util});
}

// 按理论时间排序，取 Top-15
sort(candidates.begin(), candidates.end(),
     [](auto& a, auto& b) { return a.estT < b.estT; });
int TOP_N = min(15, (int)candidates.size());
```

> ⚠️ 理论时间仅为排序代理，**Top-N 候选须全部 msprof 实测**才能确定最优。host 代码须支持命令行指定 `longMSize` 以便逐一测试。

#### 通信后计算

```cpp
// 1. 长短块切分（支持命令行指定 longMSize 以便 Top-N 候选实测）
if (longMSizeArg > 0) {
    longMSize = longMSizeArg;                           // 命令行指定（搜索算法输出）
    longBlockCnt = m / longMSize;
    shortMSize = m % longMSize;
    shortBlockCnt = (shortMSize > 0) ? 1 : 0;
    tileCnt = longBlockCnt + shortBlockCnt;
} else {
    longMSize = 512;                                    // 默认值（应替换为搜索算法结果）
    longBlockCnt = m / longMSize;
    shortMSize = m % longMSize;
    shortBlockCnt = (shortMSize > 0) ? 1 : 0;
    tileCnt = longBlockCnt + shortBlockCnt;
}
// shortBlockPos 由执行顺序 × Bound 类型决定（详见分析层策略矩阵）

// 2. 各块独立计算 matmul tiling
tilingEngine.GetTilingData(longMSize, n, ka, false, true, longQbmmTilingData);    // 长块
if (shortBlockCnt > 0) {
    tilingEngine.GetTilingData(shortMSize, n, ka, false, true, shortQbmmTilingData); // 短块
}

// 3. launchCoreNum 取长块和短块的最大值
launchCoreNum = max(longQbmmTilingData.usedCoreNum, shortQbmmTilingData.usedCoreNum);
// local tiling 生成详见 local_matmul_design.md
```

#### 计算后通信

```cpp
// 1. 长短块切分（支持命令行指定 longMSize 或均匀切分）
if (longMSizeArg > 0) {
    longMSize = longMSizeArg;
    longBlockCnt = m / longMSize;
    shortMSize = m % longMSize;
    shortBlockCnt = (shortMSize > 0) ? 1 : 0;
    tileCnt = longBlockCnt + shortBlockCnt;
} else {
    longMSize = m / tileCnt;                          // 均匀切分（无短块）
    longBlockCnt = tileCnt;
    shortMSize = 0;
    shortBlockCnt = 0;
}
// shortBlockPos 由执行顺序 × Bound 类型决定（详见分析层策略矩阵）

// 2. 各块独立计算 matmul tiling
tilingEngine.GetTilingData(longMSize, np, k, remoteBatch, mmLong);     // 长块
if (shortBlockCnt > 0) {
    tilingEngine.GetTilingData(shortMSize, np, k, remoteBatch, mmShort); // 短块
}
// local tiling 生成详见 local_matmul_design.md
```

## 4. 核心计算循环

### 4.1 通信后计算 — AIC 远程计算主循环

```cpp
// AllToAllMatmul: AIC 端 MatmulProcess()
// LocalMatmulProcess() 已在前置执行（详见 local_matmul_design.md）
for (int32_t mLoopIdx = 0; mLoopIdx < tileCnt_; ++mLoopIdx) {
    // 判断当前块是长块还是短块（由 shortBlockPos 决定顺序）
    bool isLong = shortBlockPos_  // 0=front, 1=back
        ? (mLoopIdx < longBlockCnt_)        // 短块在后: 前 longBlockCnt 个是长块
        : (mLoopIdx >= shortBlockCnt_);     // 短块在前: 后 longBlockCnt 个是长块

    const QuantMatmulTilingData* curTile = isLong
        ? &tilingData_->longQbmmTilingData    // 长块
        : &tilingData_->shortQbmmTilingData;  // 短块

    SetupParams(curTile, mLoopIdx, params, mode);

    uint64_t mOffset = CalcMOffset(mLoopIdx);  // 按 shortBlockPos 计算实际偏移
    uint32_t curMSize = GetCurMSize(mLoopIdx);  // 长块或短块的 M 大小

    // 更新地址偏移
    params.mmadParams.aGmAddr = allToAllComm_.GetDataAddrGm(mLoopIdx);
    params.mmadParams.cGmAddr = cGm_ + mOffset * axisN_ * sizeof(CType);

    CrossCoreWaitFlag<0x2, PIPE_MTE2>(mLoopIdx);  // 等待 AIV 数据就绪
    quantMatmulKernelImpl_(params);                    // 执行 matmul

    // 长短块切换时，确保 FIX 流水完成后再进入下一个 tile
    if (shortBlockCnt_ > 0 && mLoopIdx < tileCnt_ - 1) {
        SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
        WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
        SetFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(0);
    }
}
```

### 4.2 通信后计算 — AIV 通信主循环

```cpp
// AllToAllMatmul: AIV 端 AllToAllProcess()
allToAllComm_.PutScaleToAllRanks(0, axisM_);  // 先行发送全局 Scale

for (uint32_t mLoopIdx = 0; mLoopIdx < commTurn_; ++mLoopIdx) {
    uint64_t curMSize = GetCurMSize(mLoopIdx);  // 按 shortBlockPos 获取长块或短块 M 大小
    uint64_t mOffset = CalcMOffset(mLoopIdx);   // 按 shortBlockPos 计算偏移

    allToAllComm_.PutToAllRanks(mOffset, curMSize, mLoopIdx);  // 分发当前块
    CrossCoreSetFlag<0x2, PIPE_MTE3>(mLoopIdx);  // 通知 AIC 数据就绪
}
```

### 4.3 计算后通信 — AIC 计算主循环

```cpp
// MatmulAllToAll: AIC 端 MatmulProcess()
for (uint32_t tid = 0; tid < tileCnt_; ++tid) {
    // 判断当前块是长块还是短块（由 shortBlockPos 决定顺序）
    bool isLong = shortBlockPos_  // 0=front, 1=back
        ? (tid < longBlockCnt_)        // 短块在后: 前 longBlockCnt 个是长块
        : (tid >= shortBlockCnt_);     // 短块在前: 后 longBlockCnt 个是长块

    const auto& curMmTile = isLong ? tilingData_->mmLong : tilingData_->mmShort;
    uint32_t curTileM = GetCurTileM(tid);  // 按 shortBlockPos 获取长块或短块 M 大小

    SetupParams(curMmTile, remoteBatchSize, ..., param);
    param.skipMmadInit = (tid > 0);  // 首块后跳过初始化

    gemmKernel_(param);                          // 执行 QBMM 计算
    allToAllComm_.NotifyComputeComplete(tid);    // 通知 AIV 计算完成

    // 指针推进（按实际 tileM）
    aCur += curTileM * aByteStrideK_;
    cSelfCur += curTileM * cByteStrideN_;
}
```

### 4.4 计算后通信 — AIV 通信主循环

```cpp
// MatmulAllToAll: AIV 端 AllToAllProcess()
for (uint32_t tid = 0; tid < tileCnt_; ++tid) {
    allToAllComm_.WaitComputeComplete(tid);       // 等待 AIC 完成当前 tile
    aclshmemx_barrier_all_vec();                   // 所有 rank AIV 对齐
    uint32_t curTileM = GetCurTileM(tid);
    uint64_t mElemOffset = CalcMOffset(tid);
    allToAllComm_.AlltoAll(tid, curTileM, mElemOffset, rankSize_);
}
allToAllComm_.QuietAll();
```

## 5. 优化的关键修改点

| # | 修改点 | 改造前 | 改造后 | 说明 |
|---|--------|--------|--------|------|
| 1 | TilingData | 均匀 blockM + blockCount | longBlockCnt/longMSize + shortBlockCnt/shortMSize + shortBlockPos | 长短块 + 短块位置 |
| 2 | Matmul Tiling | 单一 tiling data | 长块/短块各有独立 tiling data | 不同 M 大小需不同 L1/L0 切分 |
| 3 | Host 侧 Tiling 计算 | 均匀切分 M | longMSize 切分 + 余数 short + 两组 tiling | 需先做 Bound 判定确定 longMSize |
| 4 | Kernel 主循环 | 单循环统一 tiling | 按 shortBlockPos 判断长/短块选择 tiling data | `isLong = shortBlockPos ? (tid<longBlockCnt) : (tid>=shortBlockCnt)` |
| 5 | AIC/AIV 分离 | 通信计算串行 | AIV 通信 + AIC 计算 + CrossCore 事件同步 | 通信与计算流水掩盖 |
| 6 | 长短块切换同步 | 无 | MTE1_MTE2 + M_MTE1 事件同步 | 确保 FIX 流水完成后再切换 tiling |

## 6. 注意事项 / 约束

### 6.1 TilingData 打印前置

实施前**必须**确保 host 代码打印 baseM/baseN/usedCoreNum 等 tiling 参数（见 §3.0）。这些参数是搜索算法计算核利用率和推导搜索范围的输入。缺少 TilingData 数据时不得进行 longMSize 选择——不得凭经验取固定值（如 512）替代系统搜索。

### 6.2 Bound 判定前置

长短块 `longMSize` 的选择依赖 Bound 类型判定。实施前**必须**先通过隔离测试法（注释通信或计算 process，tileCnt=1 下测试）确定 Bound 类型和 R 值。

### 6.3 系统搜索（禁止仅取经验值）

longMSize 必须通过搜索算法（§3.2）在配平可行域内系统遍历候选解，四条剪枝规则（尾块下限 128、核利用率 ≥ 80%/50%、tileCnt 范围、配平条件）过滤后取 Top-N 候选 msprof 实测。**禁止**只取 1-2 个经验值（如 longMSize=512）不做搜索就交付。

### 6.4 Mac 利用率下限

短块 M 维大小不能无限小——必须保证一定的 Mac 利用率。否则计算膨胀会抵消通算掩盖的收益。

- 计算 Bound：短块通信耗时需最小化（shortMSize 可适当小）
- 通信 Bound：短块计算耗时需最小化但保 Mac 利用率下限

### 6.5 长短块切换的流水同步

当 `shortBlockCnt > 0` 时，长块和短块使用不同的 tiling data，需要在块切换时确保 FIX 流水完成输出写入后再进入下一个 tile，避免 mmadOp 重新 Init 时冲突：

```cpp
if (shortBlockCnt_ > 0 && mLoopIdx < tileCnt_ - 1) {
    SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
    WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
    SetFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(0);
}
```

### 6.6 膨胀最小化

切分会导致计算膨胀（流水中断、用核不满）和通信膨胀（头开销、带宽下降）。应优先最小化**瓶颈方**的膨胀：

| Bound 类型 | 最小化优先级 |
|-----------|------------|
| 计算 Bound | 最小化 matmul 膨胀 |
| 通信 Bound | 最小化通信膨胀 |
| 相近 | 综合平衡，以端到端整体最优为目标 |

### 6.7 launchCoreNum 取最大值

长块和短块的 usedCoreNum 可能不同，launchCoreNum 需取两者的最大值，确保所有 tile 都有足够的核可用。

### 6.8 偏移与 M 大小自适应计算

kernel 侧的 `GetCurMSize` 和 `CalcMOffset` 需根据 `shortBlockPos` 判断当前块是长块还是短块：

```cpp
uint32_t GetCurMSize(uint32_t tid) {
    bool isLong = shortBlockPos_ ? (tid < longBlockCnt_) : (tid >= shortBlockCnt_);
    return isLong ? longMSize_ : shortMSize_;
}

uint64_t CalcMOffset(uint32_t tid) {
    // 短块在前: 前 shortBlockCnt 个是短块，之后是长块
    if (shortBlockPos_ == 0) {
        if (tid < shortBlockCnt_) return tid * shortMSize_;
        return shortBlockCnt_ * shortMSize_ + (tid - shortBlockCnt_) * longMSize_;
    }
    // 短块在后: 前 longBlockCnt 个是长块，之后是短块
    if (tid < longBlockCnt_) return tid * longMSize_;
    return longBlockCnt_ * longMSize_ + (tid - longBlockCnt_) * shortMSize_;
}
```

### 6.9 host 代码须支持命令行参数指定 longMSize（实施约束）

MC² 流水配平方案实施时，host 代码**必须**支持通过命令行参数指定 `longMSize`，以便主 agent 对 Top-N 候选逐一 msprof 实测。首选候选（Rank 1）为默认值，其余候选通过命令行参数切换。**禁止将 longMSize 硬编码为单一值。**

参考 §3.2 的 host 侧参数计算代码示例。

## 7. 实施常见问题与解决方案

| 问题 | 根因 | 解决方案 |
|------|------|---------|
| 短块计算耗时过长，掩盖率下降 | shortMSize 过大 | 调整 longMSize 使 shortMSize 更小，或增加 longBlockCnt |
| 短块 Mac 利用率过低 | shortMSize 过小 | 保证 shortMSize ≥ baseM 的下限，或调整 longMSize 使余数更大 |
| 长短块切换时 FIX 流水冲突 | 未在切换点做事件同步 | 添加 MTE1_MTE2 + M_MTE1 事件同步 |
| 通信膨胀过大 | 切分次数过多、单次通信数据量小 | 减少 tileCnt，增大 longMSize |

## 8. 选型决策

| 条件 | 推荐策略 |
|------|---------|
| 纯通信算子（无计算融合） | 不适用，走常规通信优化 |
| MC² 通算融合 + 计算 Bound | longMSize 取较大值，短块放 drain 端 |
| MC² 通算融合 + 通信 Bound | longMSize 适中，短块放 fill 端，尽量小但保 Mac |
| MC² 通算融合 + 相近 | 充分切分，最小粒度保 Mac 利用率 |
| M 不能被 longMSize 整除 | shortBlockCnt=1，shortMSize=余数，短块独立 tiling |
| 需优化本 rank 数据计算 | 叠加 [local_matmul 策略](local_matmul_design.md) |

## 9. Top-N 候选实测验证

搜索算法输出 Top-N（默认 15）个候选切分方案，须全部通过 msprof 实测验证后选最优。

### 9.1 实施流程

```
1. 搜索算法输出 Top-N 候选（含 hs, hc, tail, tc, estT, util）
2. host 代码支持命令行参数指定 longMSize
3. 对每个候选 hs 值运行 demo + msprof 采集 aiv_time
4. 按实测 aiv_time 排序，选最优作为最终方案
5. 在《性能调优报告》中列出 Top-N 候选的对比表
```

### 9.2 候选对比表格式

```markdown
| Rank | longMSize | longBlockCnt | shortMSize | tileCnt | 核利用率 | 理论T(μs) | 实测aiv(μs) | 备注 |
|------|-----------|-------------|------------|---------|---------|----------|------------|------|
| 1 | 512 | 4 | 256 | 5 | 100% | 45.2 | 48.3 | 最优 |
| 2 | 576 | 3 | 272 | 4 | 100% | 46.1 | 47.5 | 实测更优 |
| 3 | 448 | 5 | 128 | 6 | 94% | 47.8 | 52.1 | |
| 4 | 384 | 6 | 0 | 6 | 88% | 48.5 | 55.3 | 无短块 |
| 5 | 640 | 3 | 0 | 3 | 100% | 49.2 | 51.0 | 无短块 |
```

> 注意：理论排序（Rank）与实测排序可能不一致——理论模型未考虑计算膨胀和通信膨胀的精确量。实测 aiv_time 为最终择优依据。
