# reduction 范式 binary-group 模板

> 本范式主模板的完整 kernel 实现，含骨架、偏移计算、CopyIn、Compute、CopyOut。

依赖输入：[reduction-template-overview.md](../reduction-template-overview.md)、[reduction-binary-group-dag-buffers.md](reduction-binary-group-dag-buffers.md)、[reduction-tiling-preprocess.md](../reduction-tiling-preprocess.md)、[reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md)、[common/vf-programming-rules.md](../../../common/vf-programming-rules.md)、[common/cast-rules.md](../../../common/cast-rules.md)、[common/sync-and-consistency.md](../../../common/sync-and-consistency.md)、[common/datacopypad-rules.md](../../../common/datacopypad-rules.md)

> ⚠ **本文档是差分文档**：只描述 Group 模板与 Base 模板的差异及 Phase 2 新增逻辑。Base 模板的完整设计见 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md)。


**范式 kernel 约束**:
- 统一使用 TBuf，不使用 TQue，TQue 隐藏的同步操作增加模型理解复杂度
- 同步必须配套使用，一个 Lock 就必须有一个 Unlock（pipe 与 MutexID 完全一致）-- AscendC::Mutex
- VF 统一使用 `asc_vf_call` 调用, 不使用 `__VEC_SCOPE__`

**Kernel 代码规则**（所有 reduce 算子遵守）：
1. **连续 vector 计算 ≥ 2 个必须合入一个 VF**：减少 UB↔register 来回
2. **VF 函数硬约束**：详见 [common/vf-programming-rules.md](../../../common/vf-programming-rules.md) §9
3. **C++ 标准库走 `AscendC::` 命名空间**，不用 `std::`
4. **`UpdateMask` 引用传递**：详见 [common/vf-programming-rules.md](../../../common/vf-programming-rules.md) §3
5. **b16↔fp32 必须配对使用 dist 模式**：详见 [common/cast-rules.md](../../../common/cast-rules.md) §4
6. **dtype 走 `DTYPE_X` 编译期实例化**：不需要手动 `if constexpr (dtype == ...)` 分支

## 1 kernel 概述

### 1.1 数据流图

Group 模板拆成两阶段。Phase 1 使用二分缓存树（局部 `rCount`）；Phase 2 R 全载单 chunk，无需二分。

```
Phase 1（局部 rCount 个 R chunk）：
  GM_x ──→ CopyIn → PreElewise ──┐
                                   ├─ Phase A(i < bisectionTail)：二分主块 + 二分尾块 merge → Reduce → cacheBuf[层i]
  GM_x ──→ CopyIn → PreElewise ──┘  Phase B(i ≥ bisectionTail)：仅二分主块 → Reduce → cacheBuf[层i]
                                   DoCaching：cacheBuf[层i] 就地吸收低层

  cacheBuf 树根 ──→ CopyOut ──→ workspace (fp32)

Phase 2（全载 rGroupCnt，无二分）：
  workspace ──→ CopyIn → preReduceResult ──→ Reduce(isReuseSource=true) ──→ cacheBuf[0]
                                                                              ↓
  GM_x2 (可选) → postReducePhase buffer ─┐
  cacheBuf[0] ───────────────────────────┴─ PostElewise → CopyOut ──→ GM_y
```

**关键分工**：Phase 1 的 Reduce 只消解"分配给本核的那一段 R"，Phase 2 的 Reduce 消解"所有核分组的 rGroupCnt 维"——两次 Reduce 串联合并等价于 Base 的一次全 R reduce。

**Phase 2 输入输出**：

| 项目 | Phase 1 产出 | Phase 2 输入 |
|------|-------------|-------------|
| 数据来源 | 每核沿局部 R 段 reduce 后写 workspace | workspace |
| 形状 | `[rGroupCnt, aTotal]` | `[rGroupCnt, aTotal]` |
| dtype | fp32 | fp32 |
| 语义 | 对每段 R 的部分 reduce 结果 | 同左 |

输出 = GM y `[aTotal]`，dtype = D_T。

**Phase 2 处理 RA pattern**：workspace 排布 `[rGroupCnt, aTotal]`，RA pattern，外 R 内 A，axisNum=2。数据已是"PreElewise + partial reduce"后的 fp32，phase2 不做 preReduce，CopyIn 直接进 Reduce。

### 1.2 kernel IR 描述

本节以IR的形式描述 Group 模板的两阶段算法。以 ReduceSum `A0 R0 A1 R1`（tail-R，aSplitIdx=A1，rSplitIdx=R0）为例。

**注意**：以下只是IR简述，未考虑`UB内对齐约束/DataCopyPad实现转置/buffer复用关系/pad清零`等等，细节会在后续展开讨论。

ProcessGroup 采用三段式封装：

```cpp
void ProcessGroup()
{
    Phase1Process();
    SyncAll();   // 全核同步
    Phase2Process();
}
```

# A1 → A1.o × A1.i, R0 → R0.o × R0.i
# tail-R: UB = [A_bundle, R_bundle]

(1) Phase 1：复用 Base 的二分缓存树，差异是 R 段局部化、CopyOut 改写 workspace

```pseudocode
# blockIdx → (aChunkIdx, rChunkIdx), aPerCore=1
aOuter = A0 × A1.o
# R 方向大小核式均匀分配（rGroupCnt ≤ R0.o 恒成立，每组 ≥1 chunk，无空组，详见 §3.1）
rSmallGroupLoopCnt = R0.o / rGroupCnt
rBigGroupCnt       = R0.o % rGroupCnt
rBigGroupLoopCnt   = rSmallGroupLoopCnt + (rBigGroupCnt > 0 ? 1 : 0)
if (rChunkIdx < rBigGroupCnt):
    rStart = rChunkIdx * rBigGroupLoopCnt
    rCount = rBigGroupLoopCnt
else:
    rStart = rBigGroupCnt * rBigGroupLoopCnt + (rChunkIdx - rBigGroupCnt) * rSmallGroupLoopCnt
    rCount = rSmallGroupLoopCnt
rEnd = rStart + rCount

for aLoopIdx in [aChunkIdx, aChunkIdx+1):   # aPerCore=1
    A0_idx = aLoopIdx / A1.o
    A1_o_idx = aLoopIdx % A1.o
    a1Len = min(A1.i, A1 - A1_o_idx × A1.i)

    cacheBuf = [0] # 16 KB
    bisectionPos = nearestPower2(rCount)
    bisectionTail = rCount - bisectionPos

    for rIdx in [0, bisectionPos):
        {
            # 二分主块（局部下标 rStart + rIdx）
            R0_o_idx = rStart + rIdx
            r0Len = min(R0.i, R0 - R0_o_idx × R0.i)
            srcGM = GM[A0_idx:1, R0_o_idx * R0.i:r0Len, A1_o_idx * A1.i:a1Len, 0:R1] # 分别表示GM中每一维的`偏移:长度`
            buffer0 = CopyIn(srcGM)
            preReduceResult = PreElewise(buffer0)
        }
        if (rIdx < bisectionTail) {
            # 二分尾块（局部下标 rStart + rIdx + bisectionPos）
            R0_o_idx_tail = rStart + rIdx + bisectionPos
            r0Len_tail = min(R0.i, R0 - R0_o_idx_tail × R0.i)
            srcGMTail = GM[A0_idx:1, R0_o_idx_tail * R0.i:r0Len_tail, A1_o_idx * A1.i:a1Len, 0:R1]
            buffer1 = CopyIn(srcGMTail)
            preReduceResultTail = PreElewise(buffer1)
            preReduceResult = preReduceResult + preReduceResultTail
        }

        cacheBuf[GetCacheID(rIdx)] = Reduce(preReduceResult)
        DoCaching(GetCacheID(rIdx))

    # CopyOut: cacheBuf 树根 → workspace[rChunkIdx, aTotal]
    # localRoot = CalLog2(bisectionPos) * levelStride（bisectionPos 为局部值，非全局 rLoopCntTotal）
    chunkOutOff = A0_idx * A1 + A1_o_idx * A1.i    # A chunk 列偏移（线性 A 空间起始位置；A1 = 轴实际总大小，非 .o × .i）
    wsOff = rChunkIdx * aTotal + chunkOutOff
    CopyOut(cacheBuf[localRoot], workspace[wsOff])   # fp32
```

> Phase 1 的二分缓存树算法与 BinaryBase 完全一致（详见 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §1.2 (2)），差异仅：R 段局部化（`rStart`~`rEnd`）、cacheBuf 树根用本地 `localRoot` 定位、CopyOut 目标改为 workspace fp32、跳过 PostElewise。

(2) Phase 2：RA mini-kernel，R 全载 rGroupCnt，无二分

```pseudocode
# kernel 侧现算 aUbFactorP2（优先全载 rGroupCnt，剩余切 A 轴）
aUbFactorP2 = preBufSize / sizeof(fp32) / rGroupCnt
if (aUbFactorP2 >= BS_FP32): aUbFactorP2 = FloorAlign(aUbFactorP2, BS_FP32)   # 小值不归零
aUbFactorP2 = min(aUbFactorP2, postBufSize / sizeof(D_T))
aUbFactorP2 = min(aUbFactorP2, aTotal)
aSplitChunkCnt = CeilDiv(aTotal, aUbFactorP2)

# 大小核均衡，核数 = usedCoreNump2（可能 < usedCoreNum）
for aLoopIdx in [aLoopStart, aLoopEnd):
    a_off = aLoopIdx * aUbFactorP2
    a_len = min(aUbFactorP2, aTotal - a_off)

    # CopyIn workspace → preReduceResult（复用 Phase 1 槽，R 全载）
    preReduceResult = CopyIn(workspace[a_off, 0:rGroupCnt])   # fp32，RA pattern

    # Reduce: 单 chunk，isReuseSource=true，dst 写 cacheBuf[0]
    cacheBuf[0] = Reduce(preReduceResult)   # 无二分树，直接写 [0]

    # PostElewise + CopyOut
    if (has_post_reduce_input): postInBuf = CopyIn(GM_x2[a_off])
    outBuf = PostElewise(cacheBuf[0], postInBuf)   # 算子 elewise + cast_down
    CopyOut(outBuf) → GM_y[a_off]
```

## 2 Kernel 骨架代码

### 2.1 kernel骨架代码

**Phase1Process()**：复用 BinaryBase 的 Process() 主骨架，差异如下：
- blockIdx → 2D 坐标（aChunkIdx, rChunkIdx），aPerCore=1
- R 方向只处理 [rStart, rEnd)（局部 rCount 个 R chunk）
- 二分树参数（bisectionPos / bisectionTail / cacheCount）按 rCount 局部重算
- CopyOut 改为写 workspace fp32（cacheBuf 树根 → workspace[rChunkIdx, aTotal]）
- PostElewise 跳过

**Phase2Process()**：RA mini-kernel，kernel 侧反推 aUbFactorP2：

```cpp
void Phase2Process()
{
    int64_t blockIdx = GetBlockIdx();
    // ── 1) kernel 侧算切分参数 ──
    // ⚠ aUbFactorP2 ≠ TilingData 的 aUbFactor（Phase 1），详见 §3.1
    const int64_t preInElems = preBufSize / sizeof(fp32);
    constexpr int64_t BS_FP32 = UB_BLOCK_BYTES / sizeof(fp32);     // = 8（UB_BLOCK_BYTES=GetUbBlockSize() 派生，继承 base 模板命名空间常量），禁止写死
    int64_t aUbFactorP2 = preInElems / rGroupCnt_;                 // floor
    if (aUbFactorP2 >= BS_FP32) {
        // ★ burst 尾轴对齐，小值不归零
        aUbFactorP2 = FloorAlign(aUbFactorP2, BS_FP32);
    }
    aUbFactorP2 = min(aUbFactorP2, postBufSize_ / sizeof(D_T)); // 上界：postReducePhase buffer 容量
    aUbFactorP2 = min(aUbFactorP2, aTotal_);                       // 上界：aTotal
    const int64_t aSplitChunkCnt = CeilDiv(aTotal_, aUbFactorP2);
    const int64_t aLoopCntTotal  = aSplitChunkCnt;                 // outerAProd = 1

    // 大小核均衡（实际参与核数可能 < usedCoreNum，见下）
    const int64_t aSmallCoreLoopCnt_p2 = aLoopCntTotal / usedCoreNum;
    const int64_t aBigCoreCnt_p2       = aLoopCntTotal % usedCoreNum;
    const int64_t aBigCoreLoopCnt_p2   = aSmallCoreLoopCnt_p2 + (aBigCoreCnt_p2 > 0 ? 1 : 0);
    const int64_t usedCoreNump2       = (aSmallCoreLoopCnt_p2 > 0) ? usedCoreNum : aBigCoreCnt_p2;
    if (blockIdx >= usedCoreNump2) {
        return;
    }

    // blockIdx → [aLoopStart, aLoopEnd)
    int64_t aLoopStart = 0;
    int64_t aLoopEnd = 0;
    if (blockIdx < aBigCoreCnt_p2) {
        aLoopStart = blockIdx * aBigCoreLoopCnt_p2;
        aLoopEnd   = aLoopStart + aBigCoreLoopCnt_p2;
    } else {
        aLoopStart = aBigCoreCnt_p2 * aBigCoreLoopCnt_p2 +
                     (blockIdx - aBigCoreCnt_p2) * aSmallCoreLoopCnt_p2;
        aLoopEnd   = aLoopStart + aSmallCoreLoopCnt_p2;
    }

    // ── 2) aLoop 主循环（无同步版，同步代码由 §7.1 持有法则 trace 推导后插入）──
    // ⚠ 已知跨迭代 WAR（preReduceResult 的 V→MTE2、outBuf 的 MTE3→V）详见 §7.1
    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        int64_t aSplitChunkIdx = aLoopIdx;                          // outerAProd = 1
        int64_t a_off          = aSplitChunkIdx * aUbFactorP2;
        int64_t a_len          = min(aUbFactorP2, aTotal_ - a_off);
        int64_t a_len_ub       = CeilAlign(a_len, 32 / sizeof(fp32));

        // 2a) CopyIn from workspace[rGroupCnt, aTotal] → preReduceResult（复用 Phase 1 槽）
        DataCopyExtParams ext;
        ext.blockLen   = a_len * sizeof(fp32);
        ext.blockCount = rGroupCnt_;
        ext.srcStride  = aTotal_ * sizeof(fp32) - ext.blockLen;   // gap 语义：R 维 stride - blockLen
        ext.dstStride  = 0;
        DataCopyPadExtParams<fp32> padParams {
            /*isPad=*/false, /*leftPad=*/0, /*rightPad=*/0, /*paddingValue=*/0.0f
        };
        DataCopyPad(preReduceResult, wsGm[a_off], ext, padParams);

        // 2b) Reduce + PostElewise（V 读写 preReduceResult / cacheBuf / postReduceBuf，同 V 流水线无 sync）
        // Reduce: Pattern::Reduce::RA, isReuseSource=true, dst 写 cacheBuf[0]
        // uint32_t srcShape[2] = {rGroupCnt_, a_len_ub}
        AscendC::ReduceXxx<float, AscendC::Pattern::Reduce::RA,
                            /*isReuseSource=*/true>(
            cacheBuf, preReduceResult, preReduceResultTail,
            /*srcShape=*/srcShape, /*srcInnerPad=*/true);
        // PostElewise（内含 算子专属 elewise + 缩位 Cast，同一 VF 完成）
        // CopyIn_post（有 post_reduce_input 时，MTE2 写 postReducePhase buffer）

        // 2c) CopyOut（MTE3 读 postReduceBuf）
        DataCopyPad(yGm[a_off], postReduceBuf,
                    { /*blockLen=*/a_len * sizeof(D_T), /*blockCount=*/1,
                      /*srcStride=*/0, /*dstStride=*/0 });
    }
}
```

## 3 地址偏移计算

### 3.1 核间偏移计算

**Phase 1（2D 坐标计算）**：

每个核根据 `blockIdx` 计算自己的 2D 坐标和负责的 (A, R) 段。全部 kernel 侧现算：

```cpp
// kernel 侧读取
usedCoreNum = tilingData.usedCoreNum       // 已有字段
aOuter      = tilingData.aLoopCntTotal     // 已有字段
rOuter      = tilingData.rLoopCntTotal     // 已有字段
rGroupCnt   = tilingData.rGroupCnt         // 唯一新增字段

// 2D 网格坐标
aChunkIdx   = blockIdx / rGroupCnt         // 0 .. aOuter-1
rChunkIdx   = blockIdx % rGroupCnt         // 0 .. rGroupCnt-1

// R 方向偏移（大小核式均匀分配；rGroupCnt ≤ rOuter 恒成立，每组 ≥1 chunk，无空组）
rSmallGroupLoopCnt = rOuter / rGroupCnt
rBigGroupCnt       = rOuter % rGroupCnt
rBigGroupLoopCnt   = rSmallGroupLoopCnt + (rBigGroupCnt > 0 ? 1 : 0)
if (rChunkIdx < rBigGroupCnt) {
    rStart = rChunkIdx × rBigGroupLoopCnt
    rCount = rBigGroupLoopCnt
} else {
    rStart = rBigGroupCnt × rBigGroupLoopCnt + (rChunkIdx − rBigGroupCnt) × rSmallGroupLoopCnt
    rCount = rSmallGroupLoopCnt
}
rEnd        = rStart + rCount
// 防御性早退（rGroupCnt ≤ rOuter 恒成立，理论上不可达；保留防 tiling 公式被改坏）：if (rStart >= rOuter) return;

// A 方向偏移（恒为 aChunkIdx → aChunkIdx+1，末 chunk 由 base unravel 自然兜尾）
aStart      = aChunkIdx
aEnd        = aChunkIdx + 1
```

> Phase 1 核内完全复用 base 的 aLoop 调度逻辑。`aChunkIdx` 即 base 的 aLoopIdx，A 方向不需要额外切分或大小核。

> ⚠ **R 方向必须大小核式均匀分配**：`rGroupCnt ≤ rOuter` 恒成立（tiling 侧 `perCoreNum ≥ 1` → `numBlocks ≤ totalOuter + aOuter − 1`），均匀分配下每组 ≥1 chunk、无空组、所有核有工作。禁止 `rPerCore = CeilDiv(rOuter, rGroupCnt)` 截断式分配——CeilDiv 量化会造出空组（rOuter=37、rGroupCnt=18 → 5 组空），空组 workspace 行无人写，Phase 2 全行 Reduce 读入脏数据（静默数值错误）。

**Phase 2（1D aLoop）**：大核/小核分流公式结构同 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §3.1（blockIdx → [aLoopStart, aLoopEnd)），但所有参数均为 **kernel 侧现算**，不读 TilingData（参见 [reduction-template-overview.md](../reduction-template-overview.md) §4.2 "Phase 2 大小核字段由 kernel 侧用 usedCoreNum 现算"）：

```cpp
// ── kernel 侧现算 Phase 2 专用 tiling 参数 ──
// 1) aUbFactorP2：基于 preBufSize / postBufSize 反推，与 Phase 1 的 aUbFactor 不同
preInElems   = preBufSize / sizeof(fp32)
BS_FP32      = kBlockBytes / sizeof(fp32)            // 32B / 4B = 8
aUbFactorP2  = preInElems / rGroupCnt_               // 按 R 分组均分 preBuf
if (aUbFactorP2 >= BS_FP32) {
    aUbFactorP2 = (aUbFactorP2 / BS_FP32) * BS_FP32  // 32B 对齐
}
aUbFactorP2  = min(aUbFactorP2, postBufSize / sizeof(D_T))  // postBuf 容量约束
aUbFactorP2  = min(aUbFactorP2, aTotal_)             // 不超过 A 总长

// 2) 切分与核数（全部现算，不进 TilingData）
aSplitChunkCntP2 = CeilDiv(aTotal_, aUbFactorP2)
aLoopCntTotalP2  = aSplitChunkCntP2                  // outerAProd=1 退化
aSmallCoreLoopCntP2 = aLoopCntTotalP2 / usedCoreNum  // floor
aBigCoreCntP2       = aLoopCntTotalP2 % usedCoreNum
aBigCoreLoopCntP2   = aSmallCoreLoopCntP2 + (aBigCoreCntP2 > 0 ? 1 : 0)
usedCoreNumP2       = (aSmallCoreLoopCntP2 > 0) ? usedCoreNum : aBigCoreCntP2

// 3) blockIdx → [aLoopStart, aLoopEnd)（同 base §3.1 公式结构，但参数全部用 P2 版本）
if (blockIdx >= usedCoreNumP2) {
    return;               // 早退
}
if (blockIdx < aBigCoreCntP2) {
    aLoopStart = blockIdx * aBigCoreLoopCntP2
    aLoopEnd   = aLoopStart + aBigCoreLoopCntP2
} else {
    aLoopStart = aBigCoreCntP2 * aBigCoreLoopCntP2 +
                 (blockIdx - aBigCoreCntP2) * aSmallCoreLoopCntP2
    aLoopEnd   = aLoopStart + aSmallCoreLoopCntP2
}
```

> ⚠ **Phase 2 的 `aUbFactorP2` ≠ TilingData 的 `aUbFactor`**：Phase 1 UB 布局是 `[A_bundle, R_bundle]`（R 在内），Phase 2 UB 布局是 `[rGroupCnt, a_len_ub]`（RA，A 在内，R=rGroupCnt 全载），两者 UB 容量切分方向不同，因此单 chunk A 元素数需重新计算。
>
> ⚠ **`usedCoreNumP2` 可能 < `usedCoreNum`**：当 `aSmallCoreLoopCntP2 == 0`（即 `aLoopCntTotalP2 < usedCoreNum`），只有前 `aBigCoreCntP2` 个核工作，多余核早退。

### 3.2 核内循环间偏移计算

**Phase 1**：
- `UnravelALoop` / `UnravelRLoop`：同 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §3.2
- **局部二分树参数**：Phase 1 每核只处理 `rCount` 个 R chunk（非全局 `rLoopCntTotal`）。`bisectionPos`/`bisectionTail`/`cacheCount` 均按 `rCount` 局部重算，`GetCacheID(i)` 以局部下标 `i` 调用

**Phase 2**：
- `aLoopIdx → aSplitChunkIdx`：`outerAProd = 1` 退化为单维除法（`aSplitChunkIdx = aLoopIdx`）
- `a_off = aSplitChunkIdx × aUbFactorP2`，`a_len = min(aUbFactorP2, aTotal_ - a_off)`，`a_len_ub = CeilAlign(a_len, BS_FP32)`（UB 行宽，srcShape 使用；行宽由 HW 自动 32B 补齐）

## 4 CopyIn MTE2 逻辑

### 4.1 MTE2 指令选型

**Phase 1**：同 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §4.1~§4.2（DataCopyPad + Loop 模式完成"转置"装入，BuildUBAxes / UBAxisDesc / DoCopyInTile 完整骨架见 §4.2.4）

**Phase 2**：workspace fp32 → UB，R 全载（rGroupCnt）。只有两维（R, A），K=2，DataCopyPad 只需 blockLen + blockCount，不需要 Loop 模式：

```cpp
DataCopyExtParams ext;
ext.blockLen   = a_len * sizeof(fp32);
ext.blockCount = rGroupCnt_;                                     // 全部 R 分组，不是 1
ext.srcStride  = aTotal_ * sizeof(fp32) - ext.blockLen;         // workspace 行间 gap，不是 0
ext.dstStride  = 0;                                              // 块间 gap=0，HW 自动按 CeilAlign(blockLen, 32B) 放置下一个块
DataCopyPadExtParams<fp32> padParams {
    /*isPad=*/false, /*leftPad=*/0, /*rightPad=*/0, /*paddingValue=*/0.0f
};
DataCopyPad(preReduceResult, wsGm[a_off], ext, padParams);
```

> ⚠ **srcStride 必须用 aTotal_ 而非 a_len_ub**：workspace 布局是 `[rGroupCnt, aTotal]`，每行有 `aTotal` 个 float，不是 `a_len_ub` 个。

### 4.2 UB内数据排布格式

**Phase 1**：同 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §4.2

**Phase 2**：UB 布局 `[R_bundle, A_bundle] = [rGroupCnt, a_len_ub]`，R 在外、A 在内（tail-A，Pattern::Reduce::RA）。

## 5 Compute 计算逻辑

### 5.1 算法描述

#### 5.1.1 二分缓存树算法

**Phase 1**：复用 BinaryBase 的二分缓存树算法（Phase A 主-尾配对 + Phase B 主块独立）。差异仅：`rLoopCntTotal` 换成局部 `rCount`，其余公式不变。详见 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.1.1。

**kernel 侧变量与 reduction-binary-sum.md 的映射**（Phase 1 用局部 `rCount` 替换 `rLoopCntTotal`）：

| kernel-template 变量 | reduction-binary-sum.md 概念 | 计算方式 | 说明 |
|---|---|---|---|
| `rCount`（局部） | `M`（总块数） | `rEnd - rStart` | 本核 R chunk 数 |
| `bisectionPos` | `P`（主段长度，2 幂） | `FindNearestPower2(rCount)` | 严格小于 M 的最大 2 幂 |
| `bisectionTail` | `T`（尾段长度） | `rCount - bisectionPos` | Phase A 配对数 |
| `cacheCount` | `L`（缓存树层数） | `CalLog2(bisectionPos) + 1` | cacheBuf 层数 |
| `GetCacheID(rIdx)` | `GetCacheID(i)` | `count_ones(rIdx ^ (rIdx+1)) - 1` | 写入层级，局部下标 |
| `levelStride` | —（sum.md 无此概念） | `CeilAlign(laneA, 8)` | cacheBuf 每层物理 stride |
| `localRoot` | `GetCacheRoot(P)` = `cache[L-1]` | `CalLog2(bisectionPos) × levelStride` | 树根位置（本地参数） |

**Phase 1 局部二分树参数**（关键约束）：
- Phase 1 每核只处理 `rCount` 个 R chunk（非全局 `rLoopCntTotal`）
- `bisectionPos`/`bisectionTail`/`cacheCount` 均按 `rCount` 局部重算
- `GetCacheID(i)` 以局部下标 `i` 调用

**Phase 1 OutputToWorkspace cache 树根定位**（⚠ 关键约束）：

必须使用**本地参数**而非全局参数：

```cpp
// ❌ 错误：使用全局 cacheCount（基于 rLoopCntTotal）
const int32_t rootOff = (cacheCount - 1) * levelStride;

// ✅ 正确：使用本地参数（基于本 core 实际处理的 rCount）
const uint64_t localBisPos = FindNearestPower2(static_cast<uint64_t>(rCount));
const int32_t localRootLevel = static_cast<int32_t>(CalLog2(localBisPos));
const int32_t rootOff = localRootLevel * levelStride;
```

**Phase 2**：无二分树（R 全载单 chunk，cacheBuf 写[0]）。

#### 5.1.2 R_bundle 清零

**Phase 1**：同 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.1.2。

**Phase 2**：不需要显式清零。Phase 2 是 tail-A，行 pad 在 A 方向、不进入 reduce 结果（Reduce 沿 R 累加、A 各 lane 独立），所有 reducer 走同一条内核路径。

#### 5.1.3 A 方向 BurstPad/ExtensionPad 隔离

**Phase 1**：同 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.1.3。

**Phase 2**：A 方向的 BurstPad 和 ExtensionPad **不需要主动清零**。

#### 5.1.4 ReduceSum 高阶 API

**Phase 1**：同 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.1.4（局部 rCount）。

**Phase 2**：RA mini-kernel，单 chunk reduce（R 全载，无二分树），`isReuseSource=true`（src 调用后失效），dst 写 cacheBuf[0]：

```cpp
// Reduce: Pattern::Reduce::RA
// dst = cacheBuf[0], src = preReduceResult, sharedTmpBuffer = preReduceResultTail, isReuseSource = true
srcShape = {rGroupCnt, a_len_ub}
AscendC::ReduceXxx<float, AscendC::Pattern::Reduce::RA,
                    /*isReuseSource=*/true>(
    cacheBuf, preReduceResult, preReduceResultTail, srcShape, /*srcInnerPad=*/true);
```

**cacheBuf 在 Phase 2 兼作 reduceResult**：Phase 2 的 R 维 = workspace 行数 `rGroupCnt`，一次全载单 chunk（无 R 切分 → 无二分树层级）：
- Reduce dst 直接写入 cacheBuf[0]，不需要二分树层级
- → Phase 2 **复用 cacheBuf**，Reduce dst 写入 cacheBuf[0] 作为 reduceResult，postReducePhase 从 cacheBuf[0] 读取

**Phase 2 ReduceXxx 调用要点**：
- Pattern = `Pattern::Reduce::RA`（tail-A）
- srcShape = `{rGroupCnt, a_len_ub}`
- `srcInnerPad=true`
- `isReuseSource=true`（src 调用后失效，不后续读）
- dst = cacheBuf[0]，作为 reduceResult 供 postReducePhase 读取

#### 5.1.5 PostElewise 计算

**Phase 1**：跳过（无 PostElewise）。

**Phase 2**：PostElewise 读 cacheBuf[0]（Reduce 输出），UB 内有效数据是 dense, 按 `a_len`（valid）处理。**Phase 2 的 post_reduce_input 在 GM 是 dense 排布, 直接按 A 轴偏移从 GM 搬入，UB 内排布与 GM 一致，尾轴不单独补 pad**。

```cpp
constexpr uint32_t kRepF32 = VL_BYTES / sizeof(float);  // = 64，寄存器 VL（VL_BYTES=GetVRegSize() 派生，继承 base 模板命名空间常量），禁止写死

// cacheBuf[0] = Reduce 输出（fp32, [a_len] dense）
// outBuf = PostElewise 输出（D_T）
__ubuf__ float* srcPtr = cacheBuf.GetPhyAddr();       // cacheBuf[0]
__ubuf__ D_T* dstPtr = outBuf.Get<D_T>().GetPhyAddr();
uint16_t repeatTime = CeilDiv(a_len, kRepF32);
asc_vf_call<PostElewiseVfImpl<D_T>>(srcPtr, dstPtr, a_len, repeatTime);
```

### 5.2 Cast知识

**Phase 1**：同 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.2（PreElewise 内的扩位 Cast）。

**Phase 2**：PostElewise 内的缩位 Cast（fp32 → D_T），详见 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.1.5。

### 5.3 VF 编程规则

同 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.3。

## 6 CopyOut MTE3 逻辑

### 6.1 UB Buffer数据排布格式

**Phase 1**：cache 树根（VECCALC, fp32），形状 `aLen × innerAProd`（`innerAProd` kernel 端从 axisShape 现算）。同 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §6.1。

**Phase 2**：postReducePhase buffer（D_T），形状 `[a_len]`，dense。

### 6.2 CopyOut 指令

**Phase 1 CopyOut**：cache 树根 → workspace fp32。

**MTE3 参数模式**：与 base `CopyOut` 三条路径完全一致，仅 `sizeof(D_T)` → `sizeof(float)`，且 srcStride 需按 fp32 域重算：

| 场景 | blockLen | blockCount | srcStride |
|------|----------|------------|-----------|
| tail-R | `aLen × innerAProd × sizeof(float)` | 1 | 0 |
| tail-A + aSplitIdx==lastA | `aLen × sizeof(float)` | 1 | 0 |
| tail-A + aSplitIdx!=lastA | `lastASize × sizeof(float)` | `aLen × innerAProd / lastASize` | 见下文 |

> `innerAProd` kernel 端从 `axisShape` 现算，不依赖 TilingData 传递。

`dstStride = 0`（GM dense）。

**⚠ srcStride：padding gap 的 32B 量化**：cache 树 padding 在 Phase 1 CopyIn 阶段按 **DType 的 bsElem**（`= 32 / sizeof(D_T)`）计算，但元素已变为 fp32。相邻 block 间 gap = `(lastASizeAlign - lastASize) × sizeof(float)` 字节：

```
lastASizeAlign = CeilAlign(lastASize, 32 / sizeof(D_T))
srcStride = (lastASizeAlign - lastASize) × sizeof(float) / 32
```

gap 最大 `(bsElem_DType - 1) × 4B`，对 fp32/fp16 均 ≤ 60B，`srcStride` 只能为 0 或 1。

**⚠ Workspace 写入偏移**：每个 core 必须写入 workspace 的不同位置，避免互相覆盖：

```cpp
DataCopyPad(wsGm_[rChunkIdx * aTotal_ + chunkOutOff], cacheLocal[rootOff], ext);
```

| 变量 | 含义 |
|------|------|
| `rChunkIdx` | 本 core 处理的 R chunk 索引（0 ~ rGroupCnt-1） |
| `aTotal_` | A 轴总元素数（workspace 行宽） |
| `chunkOutOff` | A chunk 在输出空间中的列偏移（与 base 模板完全一致） |

workspace 布局为 `[rGroupCnt, aTotal]`，`rChunkIdx * aTotal_` 定位到对应行，`chunkOutOff` 定位到行内列偏移。

**特殊情况**：
- **All Reduce**（`aTotal=1`）：`chunkOutOff` 恒为 0，可简化为 `wsOff = rChunkIdx * aTotal_`
- **一般 Reduction**（`aTotal>1`）：必须计算 `chunkOutOff`，否则多个 core 会写入 workspace 的同一列

**Phase 1 CopyOut 必须直接从 cacheBuf_ DataCopyPad，不要中转**：
- ✅ 正确：`DataCopyPad(wsGm_[wsOff], cacheLocal[rootOff], ext);`
- ❌ 错误：中转（fp16 时越界风险：中转 buf 的大小可能比 `cacheBuf_` 小）
- V→MTE3 同步通过 Mutex Lock/Unlock 段（V 段 → MTE3 段，同 id 链式串行）实现

**Phase 2 CopyOut**：postReducePhase buffer → GM y。

```cpp
DataCopyPad(yGm[a_off], postReduceBuf,
    { /*blockLen=*/a_len * sizeof(D_T), /*blockCount=*/1, /*srcStride=*/0, /*dstStride=*/0 });
```

## 7 流水同步

### 7.1 Sync 知识

引用公共知识 [common/sync-and-consistency.md](../../../common/sync-and-consistency.md) §2.2（持有法则；本范式将推导出的同步点以 Mutex Lock/Unlock 段实现）。

**TBuf 模型**：Mutex Lock/Unlock 段式流水同步（`AscendC::Mutex::Lock/Unlock`）。
三段流水——MTE2 搬入段 → V 计算段 → MTE3 搬出段——以同一 MutexID 的 Lock/Unlock 段链式串行，
跨迭代 WAR 由段链顺序天然覆盖；MutexID 经 `AllocMutexID/ReleaseMutexID` 申请释放，同 id 不得嵌套。
单流水内：不同的 VF 间硬件保证串行，不需要 PipeBarrier。

**同步点推导方法**：同 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §7.1 的持有法则驱动方法。Phase 1 和 Phase 2 各自独立做持有法则 trace：
- **Phase 1**：取 Phase1Process 无同步版代码，逐行标注三态 → 推导同步点。buffer 生命周期与 BinaryBase 一致，差异仅 CopyOut 目标改为 workspace fp32
- **Phase 2**：取 Phase2Process 无同步版代码（§2.1），逐行标注三态 → 推导同步点

> ⚠ **禁止凭直觉预设同步点**。Phase 1/Phase 2 各段如何划分、是否存在跨流水依赖，完全取决于各自持有法则 trace 的结果。

**Buffer 非复用约束**（持有法则 trace 的设计输入）：
- **Phase 1**：与 BinaryBase 相同——CopyOut 直接从 cacheBuf 读（无 PostElewise），与 `preInBuf` 不复用，两段间无需额外依赖
- **Phase 2**：`outBuf`（postReducePhase buffer）与 `preReduceResult`（复用 Phase 1 槽，workspace 搬入）是独立物理 buffer，跨 aLoopIdx 迭代不复用，两段间无需额外依赖
- **Phase 2 跨迭代 WAR**：`preReduceResult` 每轮被 MTE2 CopyIn 覆写、上一轮被 ReduceSum（V）读——跨 aLoopIdx 迭代的 V→MTE2 WAR，下轮 MTE2 段排在上轮 V 段之后即被段链覆盖。Phase 1→Phase 2 边界无需额外同步（SyncAll 含 fence 语义）
- **Phase 2 outBuf 跨迭代 WAR**：`outBuf` 每轮被 Phase2PostElewise（V）写、上一轮被 CopyOut（MTE3）读——跨 aLoopIdx 迭代的 MTE3→V WAR，下轮 V 段排在上轮 MTE3 段之后即被段链覆盖

**持有法则 trace 必查项**：
- Phase 1 R chunk 循环内：同 BinaryBase §7.1 的必查项（循环内 WAR + 跨迭代 WAR）
- Phase 2 aLoopIdx 循环内：逐行标注三态，检查是否存在跨迭代 WAR（preReduceResult 的 V→MTE2 WAR、outBuf 的 MTE3→V WAR，其他根据持有法则 trace）

**MutexID 获取**（ProcessGroup 入口统一申请、末尾释放，Phase 1/Phase 2 共用）：

```cpp
// ProcessGroup 入口申请（TPipe 范式禁止硬编码/自行管理 MutexID），Phase 1/Phase 2 共用
uint8_t mutexId = AscendC::AllocMutexID();
// ……Phase 1 / Phase 2 各自的三段流水：Lock<PIPE_MTE2|PIPE_V|PIPE_MTE3>(mutexId)→Unlock ……
AscendC::ReleaseMutexID(mutexId);   // 末尾释放
```

> Phase 1 与 Phase 2 共用同一 MutexID：SyncAll 含 fence 语义，Phase 1 末段 Unlock 后 Phase 2 首段 Lock 自然衔接，无残留问题。

### 7.2 范式特殊的同步

**Phase 1 → Phase 2 跨核同步**：Phase 1 CopyOut 后调用 `SyncAll()`，确保所有核的 workspace 写完成后再进入 Phase 2。AscendC `SyncAll()` 自身包含 fence 语义，无需额外的跨核同步原语。

**Phase 2 outBuf 跨迭代 MTE3→V WAR**：Phase2PostElewise（V 写 outBuf）划入 V 段、CopyOut（MTE3 读 outBuf）划入 MTE3 段——同 id 段链保证上轮 MTE3 段完成后才进入下轮 V 段。

## 8 kernel 产出校验

不涉及。
