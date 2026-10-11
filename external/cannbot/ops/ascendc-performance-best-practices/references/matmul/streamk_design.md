# matmul StreamK 调度模式技术文档

## 1. 背景

在昇腾 AI 处理器上执行矩阵乘（MatMul）时，传统的 Data-Parallel（DP）分核只在 M、N 两个维度切块，把 `mCnt × nCnt` 个基块分给 `aicNum` 个核。这种切分在以下两类场景会出现明显的核间负载不均：

1. **MN 欠并行（小 M、小 N、长 K）**：当 `mCnt × nCnt` 明显小于 `aicNum` 时，相当多的核根本没有原 tile 可分，整个 kernel 被少数几个核拖到 K 向串行耗时。例如 `M=N=512, K=16384, baseM=baseN=256`：`mCnt·nCnt = 4`，而 24 个核只有 4 个在工作。
2. **末轮空闲核（MN 不整除 aicNum）**：`totalCnt = mCnt·nCnt` 不是 `aicNum` 的整数倍时，稳态区各核均匀跑，但最后一轮只有前 `tailMNCnt = totalCnt % aicNum` 个核有活。SWAT 的「末轮二维再切分」在末轮 tile 数量少、或 `baseM/baseN` 已经接近 CUBE 单元粒度时会失去收益（子块太小、CUBE 启动开销反超），此时需要另一条思路。

**StreamK** 就是为这两类场景设计的调度策略：**把空闲的核拉去参与 K 维累加**，让原本不参与计算的核也分担 `k` 方向的一部分 MMA，然后通过一块 workspace 做跨核 K 累加归约。结果是：**原本 4 个核的活被拆成 `4 × (K 切分数)` 份，接近满核利用率**。

StreamK 有两种子模式，由 host tiling 根据 shape 自动选择：

| 子模式 | 触发条件（简化） | 切分方式 |
|---|---|---|
| **Stream-K (SK)** | `mCnt·nCnt ≤ aicNum/2` 且 `K ≥ 4096 (FP16 列)` | 所有 MN 块都做 K 切分（`kCnt = aicNum / (mCnt·nCnt)`） |
| **DP + Stream-K (DP+SK)** | `mCnt·nCnt ≥ aicNum` 且 `mCnt·nCnt % aicNum ≠ 0` 且余数 ≤ `aicNum/2` 且 `K ≥ 8192/FP16` | 稳态轮按 DP 走（K 不切）；仅**最后一整轮**的 tail tile 做 K 切分 |

两种模式共用一套 kernel 实现（`matmul_a16w16_kernel_streamk.h`）和 scheduler（`matmul_a16w16_block_scheduler_streamk.h`），只靠 tiling 参数与 `CheckIsSkScene(tileIdx)` 区分。

## 2. 架构概览

### 2.1 存储层级与数据流

```
GM (A, B)
  │
  │ MTE2 (AIC 核，按本核分到的 {mIdx, nIdx, kIdx} 子片)
  ▼
L1 Buffer [A_ping | A_pong | B_ping | B_pong]         ← L1_BUFFER_NUM = 2
  │
  │ MTE1 (L1 → L0A/L0B，按 L0 pingpong 切半)
  ▼
L0A / L0B Buffer [half_0 | half_1]
  │
  │ CUBE (MMA 累加 L0A × L0B → L0C，unitFlag 控制是否 FINAL_ACCUMULATION)
  ▼
L0C Buffer
  │
  ├── SK 场景（本核只是 K 的一段）：CopyL0C2GM → Workspace[tileIdx-based offset]
  │                                      (FP32，避免中间累加精度损失)
  └── DP 场景（本核拿到完整 K）：CopyL0C2GM → GM[C]

               ↓ CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4>
               ↓ (SK 的 tile 写完 workspace 后通知 AIV)
               ↓
AIV 核
  │
  │ MTE2：把 workspace 上同一 tile 的 K 各段 (共 kCnt 段) 搬到 UB
  │ Vector：逐段 Add 累加到第 0 段
  │ Cast：FP32 → FP16/BF16
  │ MTE3：写回 GM[C]
  ▼
GM (C)
```

### 2.2 SK 与 DP+SK 的区别

```
SK  (mCnt·nCnt ≤ aicNum/2)                 DP+SK (mCnt·nCnt ≥ aicNum, 非整除)
                                              
每个 MN tile 都被 kCnt 个核切 K 累加         稳态轮：MN tile 完整落到单核（DP 路径）
                                            末轮  ：剩余 tailMNCnt 个 tile 各自被 kCnt 切 K
 ┌──── tile0 (K 被切 kCnt 段) ────┐          ┌ 稳态 ┐┌ 稳态 ┐ ... ┌ SK末轮 ┐
 │ core0 core1 ... core_{kCnt-1} │          │core0 ││core1 │ ... │ 切K   │
 └───────────────────────────────┘          └──────┘└──────┘     └───────┘
```

### 2.3 事件同步模型（共用）

| 事件类型 | 含义 | 用途 |
|---------|------|------|
| `MTE1_MTE2`(id 0/1) | MTE1 完成 → 允许 MTE2 覆写 L1[A_buf] | A 侧 L1 pingpong 释放 |
| `MTE1_MTE2`(id 2/3) | 同上 | B 侧 L1 pingpong 释放（与 A 分离 event id 避免紧耦合） |
| `MTE2_MTE1` | MTE2 完成 → 允许 MTE1 读 L1 | A、B 数据就绪 |
| `M_MTE1` | CUBE 完成 → 允许 MTE1 覆写 L0 | L0 pingpong 释放 |
| `MTE1_M` | MTE1 完成 → 允许 CUBE | L0 数据就绪 |
| `AIC_SYNC_AIV_MODE_4`（PIPE_FIX）| AIC 写完 workspace → 通知 AIV 累加 | **StreamK 专用**：跨 AIC/AIV 同步，让 AIV 等所有 K 分段都写完再开始归约 |
| `MTE3_MTE2`（ZERO_FLAG） | DP 场景 AIV 复用 UB 前等 MTE3 完成 | 避免 UB 数据竞争 |

**关键差异**：StreamK 的同步比 pingpong/SWAT 多一层 **跨核 AIC→AIV 的 CrossCoreSetFlag**，因为 SK 的最终输出（FP16/BF16 的 C 矩阵）必须等所有 K 分段的 FP32 部分和全部写入 workspace 后由 AIV 完成归约+cast。

## 3. 关键参数配置

```cpp
// 核心常量（matmul_a16w16_block_mmad_streamk.h & scheduler）
constexpr uint64_t L1_BUFFER_NUM = 2;                         // L1 双缓冲
constexpr uint64_t HALF_L0_SIZE  = L0A_SIZE / DOUBLE_BUFFER_COUNT;
constexpr uint16_t L1_EVENT_ID_OFFSET = 2;                    // A、B 用不同 event id 区间
constexpr uint16_t BLOCK_BASE_M = 256;                        // L0/workspace 对齐粒度
constexpr uint16_t BLOCK_BASE_N = 256;
constexpr uint64_t WINDOW_LEN   = 4;                          // serpentine 行窗口（与 SWAT 共享）

// Host tiling 侧（matmul_a16w16_tiling_streamk.h）决定：
struct RunInfo {
    uint64_t usedCoreNum;   // 实际使用的 AIC 核数 = aicNum
    uint64_t baseM, baseN;  // 通常 = 256
    uint64_t baseK;         // = 128 / sizeof(FP16) = 64，受 L0A 容量约束
    uint64_t kL1;           // = baseK * stepKa，被 STEPKA_THRESHOLD=4 截断
    uint64_t skSingleCoreK; // ★ StreamK 核心：单核 K 维负责的长度
    uint64_t tailInfo.kCnt; // ★ K 切分数 = aicNum / (mCnt·nCnt)  或  = aicNum / tailMNCnt
    uint64_t depthA1, depthB1; // L1 深度，depthA1 = stepKa * DB_SIZE
    uint64_t stepKa, stepKb;   // 互相倍数对齐，上限 4
};
```

### 3.1 L1 内存布局

```
L1 地址空间（FP16 路径）:
[A_ping | A_pong | B_ping | B_pong]

偏移计算（见 block_mmad_streamk.h::operator()）:
aL1OneBuffer_ = mL1 * kL1
bL1Init_      = aL1OneBuffer_ * L1_BUFFER_NUM        // B 区起点
bL1OneBuffer_ = nL1 * kL1

offsetAL1 = aL1OneBuffer_ * l1BufId * sizeof(TypeA)
offsetBL1 = (bL1Init_ + bL1OneBuffer_ * l1BufId) * sizeof(TypeB)
```

注意：StreamK 与 pingpong 的 L1 布局**一致**（都是 A_ping/A_pong/B_ping/B_pong），差别仅在 kL1 取值与 K 的切片长度。

### 3.2 Workspace 布局

```
GetWorkSpace() = aicNum * BLOCK_BASE_M * BLOCK_BASE_N * sizeof(FP32)   // 部分和累加空间
               + RPC_WORKSIZE * MB_SIZE                                // 跨核通信
```

每个核在 SK 场景下写入的 workspace 偏移：

```cpp
offsetWorkspace = (((tileIdx % usedCoreNum) / skKTileNum) * skKTileNum
                  + Get<MNK_K>(singleCoreCoord))
                  * BLOCK_BASE_M * BLOCK_BASE_N;
```

- `(tileIdx % usedCoreNum) / skKTileNum` → 本 tile 在末轮里的 MN 序号（用作归约基址）
- `Get<MNK_K>(singleCoreCoord)` → 本核负责的 K 段序号（0..kCnt-1）

AIV 归约时按「同一 MN 基址下 kCnt 段连续 FP32 存储」读取，Add 累加后 cast 回 FP16/BF16 写入 C。

### 3.3 SK Preload（DP+SK 专属优化）

DP+SK 模式下，最后一轮的 SK 子任务依赖 workspace 写入 + AIV 归约，延迟较长。Preload 机制把**末轮某些 tile 的 AIC 工作提前到倒数第二轮**执行（交换 `tileIdx` 与 `tileIdx + usedCoreNum`），使得 AIC 在末轮时同时写 workspace（for 本轮 tile）和前一轮已写好 workspace 的 tile 正好在归约完成，掩盖归约延迟：

```cpp
// see matmul_a16w16_kernel_streamk.h::operator()
if (!bs.CheckIsSkScene(0)) {  // 仅 DP+SK 启用（纯 SK 不需要）
    int64_t tailSKTotalTileNum = ((mTileNum * nTileNum) % usedCoreNum) * skKTileNum;
    // 条件：末轮前一轮的 tailSK 数量内
    if (前一轮后段属于 tail) swap(tileIdx, tileIdx ± usedCoreNum);
}
```

## 4. 核心计算循环

### 4.1 外层调度（kernel 顶层）

```cpp
// matmul_a16w16_kernel_streamk.h
for (tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += usedCoreNum) {
    // [DP+SK 才执行] SK Preload 交换
    tmpTileIdx = SkPreloadSwap(tileIdx);

    singleCoreShape = bs.GetSingleCoreShape(tmpTileIdx);   // serpentine + 尾修正
    singleCoreCoord = bs.GetSingleCoreCoord(tmpTileIdx);
    kSingleCore     = bs.GetCurKSingleCore(tmpTileIdx);    // SK 场景返回 skSingleCoreK，DP 返回 k

    gmBlockA = gmA[mPos * mL1 : +mL1, kPos * kSingleCore : +kSingleCore];
    gmBlockB = gmB[kPos * kSingleCore : +kSingleCore, nPos * nL1 : +nL1];
    gmBlockC = gmC[mPos * mL1 : +mL1, nPos * nL1 : +nL1];
    gmWorkSpace = workspace[offsetWorkspace ...];

    blockMmadOp(gmBlockC, gmBlockA, gmBlockB, gmWorkSpace,
                singleCoreShape, kIdx, bs.CheckIsSkScene(tmpTileIdx));
    //                                 ↑ 决定落 workspace (SK) 还是落 C (DP)

    if (tmpTileIdx + usedCoreNum >= tileNum) {
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreSetFlag<..>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
        // ↑ 最后一轮结束，通知 AIV 可以开始归约
    }
}
```

### 4.2 内层 K 流水（block_mmad）

```cpp
// block_mmad_streamk.h::operator()（伪代码）
for (iter0 = 0; iter0 < curKL1Iter; ++iter0) {    // K 在 L1 级的迭代
    l1BufId = abL1LoopCnt_ & 1;

    // A 侧（event id 0/1）
    WaitFlag<MTE1_MTE2>(l1BufId);
    CopyGM2L1(A → L1[offsetAL1]);
    SetFlag<MTE2_MTE1>(l1BufId);
    WaitFlag<MTE2_MTE1>(l1BufId);

    // B 侧（event id 2/3，独立于 A 避免紧耦合）
    WaitFlag<MTE1_MTE2>(l1BufId + L1_EVENT_ID_OFFSET);
    CopyGM2L1(B → L1[offsetBL1]);
    SetFlag<MTE2_MTE1>(l1BufId + L1_EVENT_ID_OFFSET);

    // 内层 L0 pingpong
    for (iter1 = 0; iter1 < kL0Iter; ++iter1) {
        l0Offset = HALF_L0_SIZE * (l0PingPong_ & 1);
        WaitFlag<M_MTE1>(l0PingPong_ & 1);
        CopyL12L0(A → L0A[l0Offset]);
        if (iter1 == 0) WaitFlag<MTE2_MTE1>(l1BufId + L1_EVENT_ID_OFFSET);  // B 只等一次
        CopyL12L0(B → L0B[l0Offset]);
        SetFlag<MTE1_M>(l0PingPong_ & 1);
        WaitFlag<MTE1_M>(l0PingPong_ & 1);

        unitFlag = (iter0+1==curKL1Iter && iter1+1==kL0Iter)
                   ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
        cmatrixInitVal = (iter0==0 && iter1==0);       // 首次累加初始化 L0C
        Mmad(tensorL0C, tensorAL0, tensorBL0, unitFlag, cmatrixInitVal);

        SetFlag<M_MTE1>(l0PingPong_ & 1);
        l0PingPong_++;
    }

    if (iter0 + 1 == curKL1Iter) {
        // 根据 checkIsSkScene 决定输出目的地
        if (checkIsSkScene) Copy L0C → workspace (FP32, FINAL_ACCUMULATION);
        else                Copy L0C → gmC       (FP16/BF16, FINAL_ACCUMULATION);
    }

    SetFlag<MTE1_MTE2>(l1BufId);
    SetFlag<MTE1_MTE2>(l1BufId + L1_EVENT_ID_OFFSET);
    abL1LoopCnt_++;
}
```

### 4.3 AIV 归约路径

```cpp
if ASCEND_IS_AIV {
    // 仅最后一轮的 tail tile 需要 AIV（kCnt 次累加）
    if (curBlockIdxInAiv >= lastLoopTotalCnt * GetTaskRation()) {
        WaitFlag<CrossCore>(AIC_SYNC_AIV_FLAG);
        SyncAll();
        return;
    }
    WaitFlag<CrossCore>(AIC_SYNC_AIV_FLAG);
    SyncAll();
    BlockEpilogueStreamK epilogue;
    epilogue.Init(...);   // 计算本 AIV 负责的 mBurstBase、offsetWorkspace、offsetCGm
    epilogue();           // for i in kCnt: Add(ub, ub[i*burstLen]); Cast FP32→FP16; DataCopyPad → C
}
```

## 5. 从 pingpong / SWAT 升级到 StreamK 的关键修改点

| 修改项 | pingpong / SWAT (优化前) | StreamK (优化后) |
|--------|------------------------|-----------------|
| 分核维度 | 仅 M、N 切分（DP） | M、N + **K 切分** |
| 并行度 | `min(mCnt·nCnt, aicNum)` | `min(mCnt·nCnt·kCnt, aicNum)` |
| Workspace | 不需要（或仅用于 L1 tiling） | **必需**：`aicNum × 256² × sizeof(FP32) + RPC` |
| 输出路径 | AIC 直接 CopyL0C2GM → C（FP16/BF16） | SK tile：AIC → workspace(FP32)；AIV 归约 + cast → C |
| AIV 启用 | 仅用于 ND2NZ / 量化类特殊处理 | **必需**：末轮 kCnt 段累加归约 |
| 跨核同步 | AIC/AIV 可独立调度 | **必需** `CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4>` 确保 workspace 就绪 |
| `unitFlag` 控制 | 始终 `FINAL_ACCUMULATION` | 按 `iter0+iter1` 判断：非末次为 `NON_FINAL_ACCUMULATION` |
| `cmatrixInitVal` | 仅首个 tile 初始化 | 每个 SK 段首次 MMA 都要初始化 L0C（因段间不累加） |
| `baseK` | 64 或 128 | 固定 `128 / sizeof(FP16) = 64`（保 L0 双缓冲） |
| `kL1` 上限 | 按 L1_BUFFER_NUM 推导 | **额外加 `STEPKA_THRESHOLD=4` 上限**，避免 SK 段过长挤掉 pingpong |
| 调度器 | `SwatScheduler` / `PingpongScheduler` | **新增** `BlockSchedulerA16W16StreamK`，带 `skKTileNum`、`totalMNTileNumInDP`、`CheckIsSkScene` |
| Epilogue | 无 / 简单 cast | **新增** `BlockEpilogueStreamK`：支持从 workspace 逐段 Add + Cast + DataCopyPad |

## 6. 注意事项

1. **只对特定 shape 有收益**：host tiling `IsCapable()` 会拒绝非目标 shape（SK 需 K≥4096 FP16 列 & mn≤核数/2；DP+SK 需 M/N 256 对齐 & K≥8192/FP16 & 余数≤核数/2）。不在目标区间的 shape 走 StreamK 反而会因 workspace 归约开销劣化，应退回 pingpong / SWAT。
2. **Workspace 容量开销**：约 `aicNum × 128KB = 3MB+`（FP32），需确认 HBM 预算。
3. **精度影响**：SK 路径的部分和在 workspace 用 FP32 保存，AIV 归约完成后再 cast 回 FP16/BF16，**精度等价或优于**纯 FP16 归约（减少了中间 round）。
4. **cmatrixInitVal 陷阱**：每个 SK 分段（每次 `blockMmadOp` 调用）都是独立的 L0C 累加序列，必须 `cmatrixInitVal = (iter0==0 && iter1==0)`；若误写为「全局首次」，则第二个段开始 L0C 会错误累加上个段的结果。
5. **SK Preload 仅适用 DP+SK**：纯 SK 模式下 `CheckIsSkScene(0) == true`，preload 分支跳过；否则会打乱稳态轮的 tile 分布，导致 workspace 偏移错位。
6. **末轮 unitFlag 处理**：`iter0 + 1 == curKL1Iter && iter1 + 1 == kL0Iter` 必须 `FINAL_ACCUMULATION`，否则 CUBE 不发出「L0C 就绪」的 block 级同步，FIXP 无法开始搬出。
7. **serpentine 行窗口继承**：scheduler 里仍用 `WINDOW_LEN=4` 做行窗口 + 奇数行反向（见 `UpdateMNTileIdx`），保证 B 流在片上的复用，不要因引入 K 切分而把这一层打掉。

## 7. 实施常见问题与解决方案

### 问题 1：Workspace 偏移计算错误，AIV 累加出来是错值

**现象**：精度校验失败，输出的 C 矩阵某些 tile 全 0 或重复叠加。

**原因**：offsetWorkspace 必须按「**末轮中 MN tile 的 index**」而非 `tileIdx` 直接算：

```cpp
// 错误写法（使用全局 tileIdx）
offsetWorkspace = tileIdx * BLOCK_BASE_M * BLOCK_BASE_N;

// 正确写法（先取末轮 MN 序号）
offsetWorkspace = (((tileIdx % usedCoreNum) / skKTileNum) * skKTileNum
                   + Get<MNK_K>(singleCoreCoord))
                  * BLOCK_BASE_M * BLOCK_BASE_N;
```

### 问题 2：AIV 等待超时 / 跨核同步 flag 不匹配

**现象**：运行 hang，或 AIV 在 SyncAll 处长时间停顿。

**原因**：AIC 侧提前 return 的核（`curBlockIdx >= bs.GetBlockNum()`）没有发送 `AIC_SYNC_AIV_FLAG`，AIV 侧 Wait 永远不会满足。

**解决方案**：空闲 AIC 核 **必须** 在 return 前补发 flag：

```cpp
if (curBlockIdx >= bs.GetBlockNum(usedCoreNum_)) {
    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
    return;
}
```

### 问题 3：SK 场景下 L0C 累加异常（段之间互相污染）

**现象**：多段 K 切分的 tile 算出来的 C 值是累加结果（而非该段的部分和）。

**原因**：`cmatrixInitVal` 判定没有考虑 StreamK 每次 `operator()` 都是独立段：

```cpp
// 错误：以为是 DP，固定 false
bool cmatrixInitVal = false;

// 正确：每次 blockMmadOp 调用都是一个独立段
bool cmatrixInitVal = (iter0 == 0 && iter1 == 0);
```

### 问题 4：MN 能被 aicNum 整除，误开 StreamK

**现象**：打印 `[ERROR] StreamK requires: K >= 4096 and MN split count <= half of core number`；或即使 shape 满足，性能反而劣化。

**原因**：
- DP+SK 的必要条件是 `mCnt·nCnt % aicNum ≠ 0`，整除时退回纯 DP + pingpong 即可。
- `mCnt·nCnt > aicNum/2` 但 `≤ aicNum` 时（未过 DP+SK 门槛），不要硬开 StreamK。

**解决方案**：在 tiling 入口处加门禁（已由 `CheckStreamKSKTiling` / `CheckStreamKDPSKTiling` 实现），否则调度器会跑错的 tileIdx 模式。

### 问题 5：baseK 取太大，stepK 被截到 1 失去 L1 pingpong

**现象**：优化后 busy 占比没提升，MTE2 和 CUBE 仍然串行。

**原因**：`baseK` 受 `l0aSize / DB_SIZE / FP16 / max(baseM,baseN)` 约束。如果误取 `baseK = 128`（FP16 路径），会让 `stepKa = depthA1/2 = 1`，L1 不再双缓冲。

**解决方案**：遵循 `CalBaseK` 公式：

```cpp
baseK = min(singleCoreK, FloorAlign(l0aSize/2/sizeof(FP16)/max(baseM,baseN), alignValue));
//                      ↑ 典型取 64 （128 bytes / 2 bytes），使 stepKa ≥ 2
```

### 问题总结

| # | 问题 | 根因 | 解决方案 | 影响 |
|---|------|------|---------|------|
| 1 | workspace offset 错位 | 直接用 tileIdx 而非末轮 MN 序号 | 按 `(tileIdx % usedCoreNum) / skKTileNum` 计算 | 精度错误 |
| 2 | AIV hang / 同步失败 | 空闲 AIC 未发 flag | 提前 return 前补发 CrossCoreSetFlag | 运行超时 |
| 3 | L0C 跨段污染 | `cmatrixInitVal` 固定 false | 每段首次 `(iter0==0 && iter1==0)` 置 true | 精度错误 |
| 4 | MN 整除仍开 StreamK | 未走 IsCapable 门禁 | 严格校验 SK/DPSK 判定条件 | 性能劣化 |
| 5 | stepKa=1 失去 L1 pingpong | baseK 过大挤掉 depth | 按 CalBaseK 公式取 `baseK=64` | 性能未达预期 |

## 8. 与 pingpong / SWAT 的取舍

| 场景特征 | 推荐策略 |
|---------|---------|
| MN 块数 ≥ aicNum 且整除 | **pingpong** 足够，无需 SWAT/StreamK |
| MN 块数 ≥ aicNum 非整除，末轮有明显尾块且 `baseM/baseN` 宽松 | **SWAT 机制 C**（末轮二维再切分）优先；`baseM·baseN` 已接近 CUBE 最小块时退化为 StreamK (DP+SK) |
| MN 块数 < aicNum 且 `K` 大（≥ 4096 列） | **StreamK (SK)** 把空闲核用 K 切分拉起来 |
| MN 块数 < aicNum 但 `K` 也小 | 无法通过 K 切分并行化，只能从 tiling / baseM/baseN 入手重做分块 |
| 有 M/N 边缘薄片且 `mCnt·nCnt ≥ aicNum` | **SWAT 机制 B**（尾块合并） |
| B 侧流式带宽严重 | 任何模式都应打开 **SWAT 机制 A**（serpentine 行窗口）；StreamK 已内置 |

**StreamK 与 SWAT 不是互斥**：StreamK 的 scheduler 内部用了 SWAT 机制 A 的 serpentine 遍历；机制 B（尾块合并）仍可在 tiling 层叠加使用（但在 StreamK 的 FP16 当前版本中暂默认 `mBaseTailSplitCnt=1`）。机制 C（末轮二维再切分）与 StreamK 的 DP+SK 是**同一痛点的两种解法**——根据 `baseM/baseN` 相对 CUBE 单元粒度的松紧，择一使用。
