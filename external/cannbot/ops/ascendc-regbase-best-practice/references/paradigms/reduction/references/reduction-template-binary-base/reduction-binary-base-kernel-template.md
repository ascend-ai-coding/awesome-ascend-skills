# reduction 范式 binary-base 模板

> 本范式主模板的完整 kernel 实现，含骨架、偏移计算、CopyIn、Compute、CopyOut。

依赖输入：
- [reduction-binary-base-dag-buffers.md](reduction-binary-base-dag-buffers.md)（核心：VF 融合后的数据流及存活节点直接决定 kernel 的搬运、计算流程）
- [reduction-template-overview.md](../reduction-template-overview.md)（§1.1 术语表）
- [reduction-binary-base-tiling.md](reduction-binary-base-tiling.md)
- [reduction-tiling-preprocess.md](../reduction-tiling-preprocess.md)
- [common/vf-programming-rules.md](../../../common/vf-programming-rules.md)
- [common/cast-rules.md](../../../common/cast-rules.md)
- [common/sync-and-consistency.md](../../../common/sync-and-consistency.md)
- [common/datacopypad-rules.md](../../../common/datacopypad-rules.md)

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

BinaryBase 模板使用二分缓存树，适用于所有 reduce 算子（sum/mean/max/min/prod/any/all）。对 sum/mean 有精度收益，对 max/min/any/all 结果不变（满足结合律），prod 因 fp 舍入不满足结合律、结果可能有差异。每个 aLoopIdx 一棵独立二分缓存树，单 for 循环内通过 `i < bisectionTail` 判定是否配对二分尾块（Phase A），否则二分主块独立处理（Phase B）。

```
每个 aLoopIdx 内：
  GM_x ──→ CopyIn → PreElewise ──┐
                                   ├─ Phase A(i < bisectionTail)：二分主块 + 二分尾块 merge → Reduce → cacheBuf[层i]
  GM_x ──→ CopyIn → PreElewise ──┘  Phase B(i ≥ bisectionTail)：仅二分主块 → Reduce → cacheBuf[层i]
                                   DoCaching：cacheBuf[层i] 就地吸收低层

  cacheBuf 树根 ──→ PostElewise ──→ CopyOut ──→ GM_y
```

### 1.2 kernel IR 描述

本节以IR的形式描述如何将二分缓存树算法引入实际 kernel 代码中。以 ReduceSum `A0 R0 A1 R1`（tail-R，aSplitIdx=A1，rSplitIdx=R0）为例。

**注意**：以下只是IR简述，未考虑`UB内对齐约束/DataCopyPad实现转置/buffer复用关系/pad清零`等等，细节会在后续展开讨论。

# A1 → A1.o × A1.i, R0 → R0.o × R0.i
# tail-R: UB = [A_bundle, R_bundle]

(1) 原始态，考虑切分，但不考虑二分累加

```pseudocode
aOuter = A0 × A1.o
for aLoopIdx in [0, aOuter):
    A0_idx = aLoopIdx / A1.o
    A1_o_idx = aLoopIdx % A1.o
    a1Len = min(A1.i, A1 - A1_o_idx × A1.i) # 可能存在 `A 切分尾块`（a1Len < A1.i）

    # 顺序累加进 reduceResult
    reduceResult = [0] * A1.i
    rOuter = R0.o # R切分轴无更外层R了，fuse后就是 R0.o
    for rIdx in [0, rOuter):
        R0_o_idx = rIdx
        r0Len = min(R0.i, R0 - R0_o_idx × R0.i) # 可能存在 `R 切分尾块`（r0Len < R0.i，即 partial chunk）

        # 源地址
        srcGM = GM[A0_idx:1, R0_o_idx * R0.i:r0Len, A1_o_idx * A1.i:a1Len, 0:R1] # 分别表示GM中每一维的`偏移:长度`
        buffer0 = CopyIn(srcGM) # buffer0 内只看到2根轴 [A_bundle, R_bundle]；此处未考虑 MTE2 通过 DataCopyPad 实现转置装入的细节，后续 §4.2 展开
        buffer1 = PreElewise(buffer0)
        buffer2 = Reduce(buffer1)    # reduce后为[A_bundle]大小
        reduceResult += buffer2
    buffer3 = PostElewise(reduceResult)
    CopyOut(buffer3) → GM[A0_idx:1, A1_o_idx * A1.i:a1Len] # 有效长度 a1Len
```

(2) 为提升累加精度，引入二分缓存树算法，(1) 中原始态的算法有如下改变

```pseudocode
aOuter = A0 × A1.o
for aLoopIdx in [0, aOuter):
    A0_idx = aLoopIdx / A1.o
    A1_o_idx = aLoopIdx % A1.o
    a1Len = min(A1.i, A1 - A1_o_idx × A1.i) # 可能存在 `A 切分尾块`（a1Len < A1.i）

    # 此处引入二分缓存树，使用cacheBuf缓存中间reduce的结果。大小固定 16KB，见 [reduction-binary-base-tiling.md](reduction-binary-base-tiling.md) §5.1
    cacheBuf = [0] # 16 KB
    rOuter = R0.o # R切分轴无更外层R了，fuse后就是 R0.o
    bisectionPos = nearestPower2(rOuter) # 小于 rOuter 的最大的 2^n  二分主块数
    bisectionTail = rOuter - bisectionPos # 二分尾块数

    for rIdx in [0, bisectionPos):
        {
            # 二分主块
            R0_o_idx = rIdx
            r0Len = min(R0.i, R0 - R0_o_idx × R0.i) # 可能存在 `R 切分尾块`（r0Len < R0.i，即 partial chunk）
            # 源地址
            srcGM = GM[A0_idx:1, R0_o_idx * R0.i:r0Len, A1_o_idx * A1.i:a1Len, 0:R1] # 分别表示GM中每一维的`偏移:长度`
            buffer0 = CopyIn(srcGM) # buffer0 内只看到2根轴 AR: [A_bundle, R_bundle]
            preReduceResult = PreElewise(buffer0)
        }
        if (rIdx < bisectionTail) {
            # 二分尾块
            R0_o_idx_tail = rIdx + bisectionPos
            r0Len_tail = min(R0.i, R0 - R0_o_idx_tail × R0.i) # 可能存在 `R 切分尾块`（r0Len_tail < R0.i，即 partial chunk）
            # 源地址
            srcGMTail = GM[A0_idx:1, R0_o_idx_tail * R0.i:r0Len_tail, A1_o_idx * A1.i:a1Len, 0:R1] # 分别表示GM中每一维的`偏移:长度`
            buffer1 = CopyIn(srcGMTail) # buffer1 内只看到2根轴 AR: [A_bundle, R_bundle]
            preReduceResultTail = PreElewise(buffer1)
            preReduceResult = preReduceResult + preReduceResultTail # 合并二分主块和二分尾块
        }

        cacheBuf[GetCacheID(rIdx) * a1Len : a1Len] = Reduce(preReduceResult)    # reduce后为[A_bundle]大小；此处未考虑 cacheBuf 层 stride 的 32B 对齐，后续 §5.1 展开为 levelStride
        DoCaching(GetCacheID(rIdx)) # 合并 cacheBuf[GetCacheID(rIdx)] 及前面的数据并写回到 cacheBuf[GetCacheID(rIdx)]
    buffer3 = PostElewise(GetCacheRoot()) # GetCacheRoot() = cacheBuf + CalLog2(bisectionPos) * levelStride，是最终的reduce结果
    CopyOut(buffer3) → GM[A0_idx:1, A1_o_idx * A1.i:a1Len] # 有效长度 a1Len
```


## 2 Kernel 骨架代码

### 2.1 kernel骨架代码

```cpp
template <typename D_T>
class KernelXxx {     // ← 占位名，每算子换成自己的类名
    // ─── 成员变量 ───
    // GM:
    GlobalTensor<D_T> xGm_, yGm_;
    // UB:
    TBuf preInBuf_, preReduceResult_, preReduceResultTail_, cacheBuf_, ...;
    TPipe *pipe_ = nullptr;
    const TilingData *td_ = nullptr;
    bool isTailR_ = false;   // tail 类型：Init 由 axisNum 奇偶现算

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingData* td, TPipe* pipe)
    {
        // Init GM
        // Init TBuf
        pipe_ = pipe;
        td_ = td;
        isTailR_ = (td_->axisNum % 2 == 0);   // 偶数轴→tail-R，奇数轴→tail-A
    }
    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx >= td_->usedCoreNum) {
            return;
        }

        // 核间偏移计算: blockIdx → [aLoopStart, aLoopEnd)
        int64_t aLoopStart = 0;
        int64_t aLoopEnd = 0;
        UnravelBlockLoop();

        for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
            int64_t aIdx[MAX_PATTERN_RANK];
            int64_t aSplitChunkIdx;
            UnravelALoop(aLoopIdx, aIdx, aSplitChunkIdx);
            int64_t a_off = aSplitChunkIdx * aUbFactor;
            int64_t a_len = min(aUbFactor, axisShape[aSplitIdx] - a_off);
            OuterIdx aIdxBundle { aIdx, a_off, a_len };
            // ─── Base 路径（二分缓存树）───
            int64_t rLoopCntTotal = td_->rLoopCntTotal;
            int64_t bisectionPos  = FindNearestPower2(rLoopCntTotal);
            int64_t bisectionTail = rLoopCntTotal - bisectionPos;

            for (int64_t rIdx = 0; rIdx < bisectionPos; ++rIdx) {
                // ── 二分主块 → preReduceResult ──
                int64_t rOuterIdx_main[MAX_PATTERN_RANK];
                int64_t rChunkIdx_main;
                int64_t rLen_main;
                UnravelRLoop(rIdx, rOuterIdx_main, rChunkIdx_main, rLen_main);
                ProcessOneRChunk(aIdxBundle, rOuterIdx_main, rChunkIdx_main, rLen_main,
                                 preReduceResult_, /*paddedSize=*/rUbFactorAlign);
                if (rLen_main < rUbFactor) {
                    // 简化示意；完整参数见 §5.1.2.2
                    ClearChunkExtensionVf(preReduceResult_, rLen_main);
                }

                // ── 二分主块 BurstPad 清零（tail-R 恒清）──
                if (isTailR_) {
                    burstTail = (rSplitIdx == LastR) ? rLen_main : axisShape[LastR];
                    if (burstTail % bsElem != 0) {
                        // 简化示意；完整参数见 §5.1.2.2
                        ClearInnerBurstTailPadVf(preReduceResult_, burstTail);
                    }
                }

                // ── Phase A（rIdx < bisectionTail）：配对二分尾块 → preReduceResultTail ──
                if (rIdx < bisectionTail) {
                    int64_t rOuterIdx_tail[MAX_PATTERN_RANK];
                    int64_t rChunkIdx_tail;
                    int64_t rLen_tail;
                    UnravelRLoop(rIdx + bisectionPos, rOuterIdx_tail, rChunkIdx_tail, rLen_tail);
                    ProcessOneRChunk(aIdxBundle, rOuterIdx_tail, rChunkIdx_tail, rLen_tail,
                                     preReduceResultTail_, /*paddedSize=*/rUbFactorAlign);
                    if (rLen_tail < rUbFactor) {
                        // 简化示意；完整参数见 §5.1.2.2
                        ClearChunkExtensionVf(preReduceResultTail_, rLen_tail);
                    }

                    // ── 二分尾块 BurstPad 清零（tail-R 恒清）──
                    if (isTailR_) {
                        burstTail = (rSplitIdx == LastR) ? rLen_tail : axisShape[LastR];
                        if (burstTail % bsElem != 0) {
                            // 简化示意；完整参数见 §5.1.2.2
                            ClearInnerBurstTailPadVf(preReduceResultTail_, burstTail);
                        }
                    }
                    MergeTmpBufVf(preReduceResult_, preReduceResultTail_);
                }

                // ── Reduce: preReduceResult(src) → cacheBuf[GetCacheID(rIdx)](dst) ──
                // levelStride = CeilAlign(laneA, 8)，laneA = aUbFactor × innerAProdAlign，详见 §5.1
                int64_t cacheID = GetCacheID(rIdx);
                ReduceXxx(cacheBuf_ + cacheID * levelStride, preReduceResult_,
                                /*srcShape={..., rUbFactorAlign × innerRProdAlign}*/);
                DoCaching(cacheID);
            }

            auto result = GetCacheRoot();
            PostElewise(result);  // 含 算子专属 elewise + cast_down（同一 VF 完成）
            CopyOut(aIdxBundle, result);
        }
    }
private:
    // CopyIn(MTE2 转置装入)→preInBuf + PreElewise→preReduceResult/preReduceResultTail
    __aicore__ inline void ProcessOneRChunk(...);
    __aicore__ inline void CopyOut(...);
};
```

> 具体算子实现时按 §5 替换为具体值/API。

## 3 地址偏移计算

### 3.1 核间偏移计算

当前模板只使用A轴开多核，需要计算当前核处理的A轴的起/止位置，即 blockIdx → [aLoopStart, aLoopEnd)，对应 `UnravelBlockLoop()`：

```cpp
// blockIdx → [aLoopStart, aLoopEnd)
if (blockIdx < aBigCoreCnt) {
    aLoopStart = blockIdx * aBigCoreLoopCnt;
    aLoopEnd   = aLoopStart + aBigCoreLoopCnt;
} else {
    aLoopStart = aBigCoreCnt * aBigCoreLoopCnt +
                 (blockIdx - aBigCoreCnt) * aSmallCoreLoopCnt;
    aLoopEnd   = aLoopStart + aSmallCoreLoopCnt;
}
```

### 3.2 核内循环间偏移计算

单核内既有A轴的循环，又有R轴的循环（即UB间的循环），分别计算 loop 循环 → 多维索引解码：

#### UnravelALoop()

reduction的切分策略是将A轴切分轴的 aSplitChunkCnt（即 A.o）与左侧所有A轴fuse在一起的，一维的循环索引 `aLoopIdx` 需解码为多维 A 轴索引: `aSplitChunkIdx` 为 aSplitIdx 轴上的 chunk 序号，`aIdx[]` 为 aSplitIdx 左侧所有外层 A 轴的索引（用于算 GM 偏移）。

```cpp
// aLoopIdx → (aIdx[outer A], aSplitChunkIdx)
void UnravelALoop(int64_t aLoopIdx, int64_t aIdx[], int64_t& aSplitChunkIdx)
{
    int64_t rem = aLoopIdx;
    aSplitChunkIdx = rem % aSplitChunkCnt;
    rem /= aSplitChunkCnt;
    // 从 aSplitIdx 的左邻 A 向外侧依次拆（k 取 A 轴位置=偶下标，步长 2 越过 R 轴）
    for (int32_t k = aSplitIdx - 2; k >= 0; k -= 2) {
        aIdx[k]  = rem % axisShape[k];
        rem     /= axisShape[k];
    }
}
```

#### UnravelRLoop()

R轴切分轴的 rSplitChunkCnt（即 R.o）与左侧所有 R 轴是fuse在一起的, 一维 `rIdx` 索引需解码为多维 R 轴索引：`rChunkIdx` 为 rSplitIdx 轴上的 chunk 序号，`rLen` 为当前 chunk 的实际长度（可能 partial），`rOuterIdx[]` 为 rSplitIdx 左侧所有外层 R 轴的索引（用于算 GM 偏移）。其中 `rSplitChunkCnt = CeilDiv(axisShape[rSplitIdx], rUbFactor)`（R 切分轴的 chunk 数）。

```cpp
// rIdx → (rOuterIdx[outer R], rChunkIdx, rLen)
void UnravelRLoop(int64_t rIdx, int64_t rOuterIdx[], int64_t& rChunkIdx, int64_t& rLen)
{
    rChunkIdx = rIdx % rSplitChunkCnt;
    int64_t rem = rIdx / rSplitChunkCnt;
    // 从 rSplitIdx 的左邻 R 向外侧依次拆（k 取 R 轴位置=奇下标，步长 2 越过 A 轴）
    // 下标 0 必为 A（leading A 规则），R 轴最低下标为 1
    for (int32_t k = rSplitIdx - 2; k >= 1; k -= 2) {
        rOuterIdx[k] = rem % axisShape[k];
        rem /= axisShape[k];
    }
    rLen = min(rUbFactor, axisShape[rSplitIdx] - rChunkIdx * rUbFactor);
}
```

#### GM 偏移计算

**输入起始偏移**：

```
offset_in = Σ_{k_A < aSplitIdx, k_A 是 A 轴} aIdx[k_A] × axisStride[k_A]    // 外层 A 轴（来自 UnravelALoop）
          + aSplitChunkIdx × aUbFactor × axisStride[aSplitIdx]              // A 切分轴
          + Σ_{k_R < rSplitIdx, k_R 是 R 轴} rOuterIdx[k_R] × axisStride[k_R]  // 外层 R 轴（来自 UnravelRLoop）
          + rChunkIdx × rUbFactor × axisStride[rSplitIdx]                   // R 切分轴
```

**输出偏移**（输出 shape = 各 A 轴 size 顺序拼接，连续、紧凑，无 R 维）：

```
output_strides[k_A] = ∏(更内 A 轴的 size)
offset_out = Σ_{k_A} a_idx_{k_A} × output_strides[k_A]
           + aSplitChunkIdx × aUbFactor × output_strides[aSplitIdx 对应的 A 轴序号]
```

## 4 CopyIn MTE2 逻辑

### 4.1 MTE2 指令选型

必读知识 [common/datacopypad-rules.md](../../../common/datacopypad-rules.md)。reduce 场景使用 DataCopyPad + Loop 模式完成"转置"装入。

**UB 内 aSplitIdx 的切分因子 aUbFactor 及其右侧所有 A 轴聚成 A_bundle、rSplitIdx 的切分因子 rUbFactor 及其右侧所有 R 轴聚成 R_bundle**，由 DataCopyPad 的 stride/blockCount 实现"转置"装入（尾轴不转置）。bundle 内外顺序由 tail 类型决定：
- **tail R**：UB = `[A_bundle 在外, R_bundle 在内]`，最内 R 是 burst 尾轴
- **tail A**：UB = `[R_bundle 在外, A_bundle 在内]`，最内 A 是 burst 尾轴

**字段映射规律**（从 UB 内到外）：

| UB 轴位置 | 对应字段 |
|-----------|---------|
| 第 1 根（burst 尾轴） | `blockLen`（byte） |
| 第 2 根 | `blockCount` |
| 第 3 根 | `loop1Size` |
| 第 4 根 | `loop2Size` |
| 第 5 根及更外 | for 循环 |

**stride 双语义**（不可混用）：

> 本章是描述 MTE2，从 gm->ub 语义

| 字段类型 | 语义 | 公式 |
|---------|------|------|
| `DataCopyExtParams.srcStride` | **gap**（尾→头），减一次 `blockLen` | `axisShape[burst尾轴] × sizeof(D_T) − blockLen` |
| `LoopModeParams.loop*SrcStride` | **advance**（头→头），直接用 | `axisStride[对应外层轴] × sizeof(D_T)` |

**K 分级**：
- K=2：只用 blockLen + blockCount
- K∈[3,4]：必须 Loop 模式，且 `SetLoopModePara` + `ResetLoopModePara`
- K≥5：第 5 根及更外用 for 拆


**isPad 与 paddingValue**：

burst 尾轴 bundle 必须 block 对齐（非对齐时访存性能很差），UB 行宽由 `paddedNum` 保证，BurstPad 区域是行宽对齐的必然产物。

```cpp
DataCopyPadExtParams<D_T> padParams{/*isPad=*/false, /*leftPadding=*/0, /*rightPadding=*/0, /*paddingValue=*/0};
```

- 行宽对齐由 HW 自动保证：UB 侧每行实际占用恒为 `CeilAlign(blockLen, 32B)`，与 `rightPadding` 无关
- `isPad=false`、`rightPadding=0`：补区填首元素值（脏数据），不承载任何语义，Reduce 前由 kernel 统一清零（§5.1.2）

### 4.2 UB内数据排布格式

**原则**：
由于二分累加存在merge的操作，即会将 `切分尾块 UB` + `切分整块 UB` 做合并，我们默认要求两块UB占用大小一样（即切分尾块使用切分整块的 layout， laneA = aUbFactor × innerAProdAlign; laneR = rUbFactorAlign × innerRProdAlign），非有效数据位置必须清为 `pad_value`（kernel 清零保证，§5.1.2）

MTE2 搬运 UB 后，UB 内是 bundled layout。装入 UB 后只看尾轴的类型：尾 R → 视为 `AR`；尾 A → 视为 `RA`。

UB 内存在两类 pad：

- **BurstPad**：burst 尾轴方向尾部的 pad（行宽对齐的必然产物），值是脏数据（rightPadding=0 → HW 填首元素值）
- **ExtensionPad**：partial chunk（rLen < rUbFactor）时，R 切分轴方向 `[rLen, rUbFactor)` 范围的切片，未搬入的 stale，值不确定。需要代码显示清理 stale，设置为 `pad_value`

> A 方向 BurstPad/ExtensionPad stale数据不影响累加结果; R 方向 BurstPad/ExtensionPad stale数据会影响累加结果。

### 4.2.1 非切尾轴场景

burst 尾轴是整根轴（未被 UB 切分），BurstPad 取决于该轴 size 是否对齐。

**tail-R 非切尾轴**：`A0 R0 A1 R1`（aSplit=A1，rSplit=R0），fp32，R1=5（非对齐），rUbFactor=4，rLen=3（partial），aUbFactor=2。UB = `[A_bundle, R_bundle]`，从内到外：R1 → R0.i → A1.i。每个 R0.i 切片含 R1 个元素（padded 到 8）：

```
R切分主块(rUbFactor) R_bundle 的排布:
R0.i=0: [R1=5 valid │ BurstPad=3]     ← DataCopyPad 搬入，BurstPad=脏数据
R0.i=1: [R1=5 valid │ BurstPad=3]
R0.i=2: [R1=5 valid │ BurstPad=3]
R0.i=3: [R1=5 valid │ BurstPad=3]

R切分尾块(rLen) R_bundle 的排布:
R0.i=0: [R1=5 valid │ BurstPad=3]     ← DataCopyPad 搬入，BurstPad=脏数据
R0.i=1: [R1=5 valid │ BurstPad=3]
R0.i=2: [R1=5 valid │ BurstPad=3]
R0.i=3: [───── ExtensionPad ────]     ← partial 未搬，stale
```

BurstPad 在 R 方向（R1 是 R 轴）→ 需置0 → Reduce 累加。ExtensionPad 在 R 方向（R0.i 是 R 轴）→ 需置0 → Reduce 累加。

**tail-A 非切尾轴**：`A0 R0 A1`（aSplit=A0，rSplit=R0），fp32，A1=5（非对齐），rUbFactor=4，rLen=3（partial），aUbFactor=2。UB = `[R_bundle, A_bundle]`，从内到外：A1 → A0.i → R0.i。每个 R0.i 切片含整个 A_bundle（A0.i × A1，A1 padded 到 8）：

```
R切分主块(rUbFactor) R_bundle 的排布:
R0.i=0, A0.i=0: [A1=5 valid │ BurstPad=3]     ← A 方向 pad
R0.i=0, A0.i=1: [A1=5 valid │ BurstPad=3]
...
R0.i=3, A0.i=0: [A1=5 valid │ BurstPad=3]
R0.i=3, A0.i=1: [A1=5 valid │ BurstPad=3]

R切分尾块(rLen) R_bundle 的排布:
R0.i=0, A0.i=0: [A1=5 valid │ BurstPad=3]     ← A 方向 pad
R0.i=0, A0.i=1: [A1=5 valid │ BurstPad=3]
...
R0.i=3, A0.i=0: [────── stale ×8 ──────]       ← ExtensionPad
R0.i=3, A0.i=1: [────── stale ×8 ──────]
```

BurstPad 在 A 方向（A1 是 A 轴）→ Reduce 不累加 → **不需清**。ExtensionPad 在 R 方向（R0.i 是 R 轴）→ Reduce 累加 → 需置0。

### 4.2.2 切尾轴场景

burst 尾轴是 split 切分内层（rSplit==LastR 或 aSplit==LastA），BurstPad 取决于 rLen/aLen 是否对齐，且同一行内 BurstPad 后面紧跟该方向的 ExtensionPad。

**tail-R 切尾轴**（rSplit==LastR）：`A0 R0 A1 R1`（aSplit=A0，rSplit=R1=LastR），fp32，rUbFactor=16，rUbFactorAlign=16，rLen=5（partial），aUbFactor=2，A1=3。UB = `[A_bundle, R_bundle]`，从内到外：R1.i → A1 → A0.i。每个 A entry 一行 rUbFactorAlign=16 个 fp32：

```
R切分主块(rUbFactor) R_bundle 的排布:
[------------- valid=16 --------------]
  ↑
rLen=rUbFactor=16（对齐，无 BurstPad；full chunk，无 ExtensionPad）

R切分尾块(rLen) R_bundle 的排布:
[valid=5 │ BurstPad=3 │ ExtensionPad=8]
  ↑         ↑               ↑
 rLen   HW自动补到32B  [CeilAlign(rLen,8), rUbFactorAlign)
```

BurstPad 在 R 方向 → 需置0。ExtensionPad 也在 R 方向 → 需置0。与非切尾轴的区别：同一行内 BurstPad 后面紧跟 ExtensionPad（rUbFactorAlign > CeilAlign(rLen, bsElem)）。

**tail-A 切尾轴**（aSplit==LastA）：`A0 R0 A1`（aSplit=A1=LastA，rSplit=R0），fp32，aUbFactor=16（对齐），aLen=5（partial），rUbFactor=4，rLen=3（partial）。UB = `[R_bundle, A_bundle]`，从内到外：A1.i → R0.i。每个 R0.i 切片 = aUbFactor=16 个 fp32：

```
R切分主块(rUbFactor) R_bundle 的排布:
R0.i=0: [valid=5 │ BurstPad=3 │ ExtensionPad=8]    ← A 方向 pad
R0.i=1: [valid=5 │ BurstPad=3 │ ExtensionPad=8]
R0.i=2: [valid=5 │ BurstPad=3 │ ExtensionPad=8]
R0.i=3: [valid=5 │ BurstPad=3 │ ExtensionPad=8]

R切分尾块(rLen) R_bundle 的排布:
R0.i=0: [valid=5 │ BurstPad=3 │ ExtensionPad=8]    ← A 方向 pad
R0.i=1: [valid=5 │ BurstPad=3 │ ExtensionPad=8]
R0.i=2: [valid=5 │ BurstPad=3 │ ExtensionPad=8]
R0.i=3: [──────── stale =16 ────────]              ← R 方向 ExtensionPad
```

A1.i 是 A 轴 → BurstPad 和 A 方向 ExtensionPad 都在 A 方向 → **不需置0**。R0.i=3 是 R 方向 ExtensionPad → 需置0。与非切尾轴的区别：同一行内 BurstPad 后面有 A 方向 ExtensionPad（aUbFactor > CeilAlign(aLen, bsElem)），两者都不需置0。

### 4.2.3 总结

| | BurstPad（burst 尾轴方向） | ExtensionPad（尾块方向） |
|---|---|---|
| **tail-R** | R 方向，Reduce 累加，需置0 | R 方向，Reduce 累加，需置0 |
| **tail-A** | A 方向，Reduce 不累加，不需置0 | R 方向，Reduce 累加，需置0 |

- **BurstPad**：搬运不指定 paddingValue（isPad=false），是脏数据。tail-R 恒需置0（§5.1.2）；tail-A 在 A 方向，不需置0
- **ExtensionPad**：DataCopyPad 无法覆盖 Extension 部分，必须显式置0

### 4.2.4 DoCopyInTile 完整骨架

**目的**：统计需要搬入UB内的每个 `A/R` 轴信息（如有效长度、stride等），供 DataCopyPad 逐字段映射搬运参数。

`BuildUBAxes` 根据 aSplitIdx / rSplitIdx / isTailR，从 **GM 最内层向外**构造 UB 轴列表。下标 `[0]` 为 burst 尾轴（UB 最内层），下标 `[K-1]` 为 UB 最外侧轴，`K` 是 UB 内总轴数（2 ≤ K ≤ axisNum）。

```cpp
struct UBAxisDesc {
    int32_t gmIdx;         // 原 pattern 轴序号（0..axisNum-1），GM 偏移还原用
    int64_t actualNum;     // actual 元素数 → blockLen / blockCount / loopSize
    int64_t paddedNum;     // UB 行步距元素数 → dstStride / ubStride
                           // 切分轴（rSplitIdx/aSplitIdx）：rUbFactorAlign / aUbFactor（对齐后的切分因子，整块 layout）
                           // 非切分的 burst 尾轴：CeilAlign(axisSize, bsElem)；其余 = actualNum
    int64_t gmStride;      // GM stride（元素，来自 axisStride → srcStride / loop*SrcStride）
};
```

```cpp
void DoCopyInTile(int64_t baseGmOff, int64_t aLen, int64_t rLen, LocalTensor<D_T>& preIn)
{
    UBAxisDesc ubAxes[MAX_PATTERN_RANK];
    const int32_t K = BuildUBAxes(aLen, rLen, ubAxes);
    const int64_t dt = sizeof(D_T);

    // ── 第 1 根 → blockLen；行内 padding 由 paddedNum 决定 dstStride ──
    DataCopyExtParams ext;
    ext.blockLen  = ubAxes[0].actualNum * dt;                              // 有效字节

    DataCopyPadExtParams<D_T> padParams{/*isPad=*/false, /*leftPadding=*/0, /*rightPadding=*/0, /*paddingValue=*/0};

    int64_t copyPadBytes = CeilAlign(ext.blockLen, 32);                 // HW 在 UB 上占的字节（32B 粒度）
    int64_t target0Bytes = ubAxes[0].paddedNum * dt;                   // 固定 UB 行步距
    ext.dstStride = (target0Bytes - copyPadBytes) / 32;                 // UB 侧 gap（partial chunk 时 > 0），单位 datablock

    // ── 第 2 根 → blockCount；srcStride = gap(尾→头)，减一次 blockLen ──
    if (K >= 2) {
        ext.blockCount = ubAxes[1].actualNum;
        ext.srcStride  = ubAxes[1].gmStride * dt - ext.blockLen;        // GM 侧 gap 语义
    } else {
        ext.blockCount = 1;
        ext.srcStride  = 0;
    }
    ext.rsv = 0;    // 必须显式填 0（datacopypad-rules.md）

    // ── UB 每层字节步长：ubStride[0]=dt；ubStride[k]=ubStride[k-1]×paddedNum[k-1] ──
    int64_t ubStride[MAX_PATTERN_RANK];
    ubStride[0] = dt;
    for (int32_t i = 1; i < K; ++i) {
        ubStride[i] = ubStride[i-1] * ubAxes[i-1].paddedNum;
    }

    // ── 第 3/4 根 → loop1/loop2；loop*Stride = advance(头→头)，直接 axisStride×dt ──
    LoopModeParams lp{};        // 全 0
    if (K >= 3) {
        lp.loop1Size      = ubAxes[2].actualNum;
        lp.loop1SrcStride = ubAxes[2].gmStride * dt;                    // advance 语义
        lp.loop1DstStride = ubStride[2];
        lp.loop2Size      = 1;     // ★ 哪怕外层不用也必须 ≥ 1，否则整批搬运 0 次
    }
    if (K >= 4) {
        lp.loop2Size      = ubAxes[3].actualNum;
        lp.loop2SrcStride = ubAxes[3].gmStride * dt;
        lp.loop2DstStride = ubStride[3];
    }

    const bool useLoopMode = (K >= 3);
    if (useLoopMode) {
        SetLoopModePara(lp, DataCopyMVType::OUT_TO_UB);    // ★ 调用前 set
    }

    // ── 第 5 根及更外 → for 拆；K≤4 时 outerProd=1，循环一次 ──
    int64_t outerProd = 1;
    for (int32_t k = 4; k < K; ++k) {
        outerProd *= ubAxes[k].actualNum;
    }
    for (int64_t flat = 0; flat < outerProd; ++flat) {
        int64_t addGm = 0;
        int64_t addUbBytes = 0;
        int64_t cur = flat;
        for (int32_t k = 4; k < K; ++k) {
            int64_t ix = cur % ubAxes[k].actualNum; cur /= ubAxes[k].actualNum;
            addGm      += ix * ubAxes[k].gmStride;
            addUbBytes += ix * ubStride[k];
        }
        DataCopyPad(preIn[addUbBytes / dt], xGm[baseGmOff + addGm], ext, padParams);
    }

    if (useLoopMode) {
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);      // ★ 调用后 reset，成对出现
    }
}
```

## 5 Compute 计算逻辑

### 5.1 算法描述

#### 5.1.1 二分缓存树算法

二分缓存树的完整算法（GetCacheID 公式与演化表、DoCaching 吸收逻辑、FindNearestPower2/CalLog2、Phase A/B 主尾配对）详见 [reduction-binary-sum.md](../reduction-binary-sum.md)。

**kernel 侧变量与 reduction-binary-sum.md 的映射**：

| kernel-template 变量 | reduction-binary-sum.md 概念 | 计算方式 | 说明 |
|---|---|---|---|
| `rLoopCntTotal` | `M`（总块数） | tiling 算出 | R 切分后的总 chunk 数 |
| `bisectionPos` | `P`（主段长度，2 幂） | `FindNearestPower2(rLoopCntTotal)` | 严格小于 M 的最大 2 幂 |
| `bisectionTail` | `T`（尾段长度） | `rLoopCntTotal - bisectionPos` | Phase A 配对数 |
| `cacheCount` | `L`（缓存树层数） | `CalLog2(bisectionPos) + 1` | cacheBuf 层数 |
| `GetCacheID(rIdx)` | `GetCacheID(i)` | `count_ones(rIdx ^ (rIdx+1)) - 1` | 写入层级 |
| `levelStride` | —（sum.md 无此概念） | `CeilAlign(laneA, 8)` | cacheBuf 每层物理 stride，kernel 侧新增 |
| `GetCacheRoot()` | `GetCacheRoot(P)` = `cache[L-1]` | `cacheBuf + (cacheCount-1) × levelStride` | 树根位置 |

**levelStride 对齐约束**：

```cpp
// 命名空间作用域（VF 函数外）：平台参数经 API 运行时获取，禁止写死 32/256/512
constexpr uint32_t VL_BYTES = Ops::Base::GetVRegSize();            // 如 arch35 系列为 256B，寄存器宽度
constexpr uint32_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();   // 32B，UB block 粒度
constexpr uint32_t BLOCK_F32 = UB_BLOCK_BYTES / sizeof(float);     // = 8
const uint32_t levelStride = (laneA + BLOCK_F32 - 1) / BLOCK_F32 * BLOCK_F32;
```

**约束**：cacheBuf 每层 stride 必须 `CeilAlign(laneA, 8)`（32B 对齐）。VF `LoadAlign`/`StoreAlign` 强制 32B 对齐起点。

- `laneA` = `aUbFactor × innerAProdAlign`（padded A 维 lane 数，与 Reduce dst 写入形状一致）
- `laneA` 可能已经对齐，也可能未对齐，所以统一计算对齐后的结果给 `levelStride=CeilAlign(laneA, 8)`
- `GetCacheRoot()` = `cacheBuf + (cacheCount-1) × levelStride`，fp32，形状 `aUbFactor × innerAProdAlign`（padded）。PostElewise 按 padded 整行处理，CopyOut 按 `aLen × innerAProd`（有效）拷出。`innerAProd` = ∏ axisShape[aSplitIdx+1..LastA]（真实乘积），kernel 端从 axisShape 计算
- 每个 aLoopIdx 进入前**不需要**清零 cacheBuf：DoCaching 覆盖写 + GetCacheID 序列保证低层先写，上轮残留不会进入结果

#### 5.1.2 R_bundle 清零

**以下讨论均是基于UB内 R 方向的 pad/stale，A 方向不需要清零**。

**清零口径**：
- **按 fp32 元素个数清零**：BurstPad 按 InputT 的 bsElem 对齐（fp16 补到 16 个元素），Cast 后元素数守恒变 fp32。mask 按 fp32 个数精确覆盖 [validR, padEndInRow)，不按字节或 block 粒度
- **双类型必需**：InputT 决定 BurstPad 补到多少个元素（fp16=16, fp32=8），TmpT 固定 fp32 决定 mask 和 store 的粒度

##### 5.1.2.1 pad 区域脏数据与清零规则

搬运不指定 paddingValue（isPad=false），BurstPad / ExtensionPad 均为脏数据，Reduce 前必须将 **R 方向的 pad** 清零为 `pad_value`；A 方向的 pad 不参与 Reduce、不被拷出，不需清零。

**BurstPad**：（使用 `ClearInnerBurstTailPadVf` 清零）
只在 tail-R 出现（burst 尾轴是 R），尾轴非对齐时恒清零。

> **行序注意（仅 rSplit<LastR）**：这里的"行"指 burst 尾轴方向的一次遍历（宽 rowStride）。partial 与 full chunk 用同一 UB layout、大小相同，行按全量布局排位；UB 从外到内依次是 A entry、R 切分轴、中间 R 轴（如有）、burst 尾轴，所以行按 A entry 分段排列。partial 时每段只有 R 切分轴前 rLen 个位置有数据，要清的行是每段开头的一小段，在内存里不连续——与其跳着清，不如按全量行清：多清的 stale 行本就是脏数据，清掉无害。

**ExtensionPad**：（使用 `ClearChunkExtensionVf` 清零）
partial chunk 的 stale 区域，无论 tail 类型都必须显式清。

**pad_value 取值表（fp32 计算域）**：

| reducer | pad_value |
|---------|-----------|
| sum / mean | `0` |
| max | `dtype::lowest()` |
| min | `dtype::highest()` |
| prod | `1` |
| any | `false` (0) |
| all | `true` (1) |

以下 4 个场景以 exp+sum（`pad_value=0`）为例，展示脏数据清零流程。

**场景 A：tail-R 切尾轴 + partial**（rSplit==LastR）

`A0 R0 A1 R1`（aSplit=A0，rSplit=R1=LastR），fp32，rUbFactor=16，rLen=5（partial），aUbFactor=2，A1=3。每个 A entry 一行 16 个 fp32：

```
MTE2 后:     [valid=5 │ BurstPad=3 │ ExtensionPad=8]     ← 全部脏数据
PreElewise: [exp(×)=5 │ 脏×3        │ 脏×8]

ClearInnerBurstTailPadVf: validR=rLen=5, rowStride=rUbFactorAlign×innerRProdAlign=16×1=16, rowCnt=aUbFactor×innerAProdAlign=2×3=6
  → 每行清 [5, CeilAlign(5,8)=8)，共 6 行
ClearChunkExtensionVf: extStart=CeilAlign(rLen×innerRProdAlign, 8)=8, extLanes=rUbFactorAlign×innerRProdAlign−extStart=16−8=8, aBundleEntries=6
  → 每行清 [8, 16)，共 6 行

清完后:       [exp(×)=5 │ 0×3        │ 0×8]
→ Reduce 沿 R 累加，pad_value 零贡献，结果正确
```

BurstPad 与 ExtensionPad 相邻，extStart = CeilAlign(rLen, 8) = 8，两者各自由不同 VF 清。

**场景 B：tail-R 非切尾轴 + partial**（rSplit<LastR）

`A0 R0 A1 R1`（aSplit=A1，rSplit=R0），fp32，R1=5（非对齐），rUbFactor=4，rLen=3（partial），aUbFactor=2。每个 R0.i 对应一行（R1 padded 到 8）：

```
MTE2 后（单个 A entry）:
R0.i=0: [R1=5 valid │ BurstPad=3]     ← 脏数据
R0.i=1: [R1=5 valid │ BurstPad=3]
R0.i=2: [R1=5 valid │ BurstPad=3]
R0.i=3: [──────── ExtensionPad ──────] ← 未搬，stale

ClearChunkExtensionVf: extStart=CeilAlign(rLen×innerRProdAlign, 8)=CeilAlign(3×8,8)=24, aStride=rUbFactorAlign×innerRProdAlign=4×8=32
  → extLanes=32−24=8，每个 A entry 清 [24,32)（R0.i=3 那一行），共 2 行
ClearInnerBurstTailPadVf: validR=axisShape[LastR]=5, rowStride=CeilAlign(5,8)=8,
  rowCnt = UB 总元素数 ÷ rowStride = (A entry 数 2 × R0 切片数 4 × R1 宽 8) ÷ 8 = 8（全量行）
  → 每行清 [5,8)，共 8 行

→ Reduce 沿 R 累加，pad_value 零贡献，结果正确
```

BurstPad 在 LastR 方向（每行尾部），ExtensionPad 在 R0.i 方向（整行），两者方向不同。

**场景 C：tail-A + partial**

`A0 R0 A1`（aSplit=A0，rSplit=R0），fp32，A1=5（非对齐），rUbFactor=4，rLen=3（partial），aUbFactor=2。每个 R0.i 切片含整个 A_bundle（A0.i × A1，A1 padded 到 8）：

```
MTE2 后:
R0.i=0, A0.i=0: [A1=5 valid │ BurstPad=3]     ← A 方向 pad（脏数据）
R0.i=0, A0.i=1: [A1=5 valid │ BurstPad=3]
R0.i=1, A0.i=0: [A1=5 valid │ BurstPad=3]
R0.i=1, A0.i=1: [A1=5 valid │ BurstPad=3]
R0.i=2, A0.i=0: [A1=5 valid │ BurstPad=3]
R0.i=2, A0.i=1: [A1=5 valid │ BurstPad=3]
R0.i=3, A0.i=0: [────── stale ×8 ──────]       ← ExtensionPad
R0.i=3, A0.i=1: [────── stale ×8 ──────]

ClearChunkExtensionVf: cellElems=aUbFactor×innerAProdAlign×innerRProdAlign=2×8×1=16
  → startElem=rLen×cellElems=3×16=48, totalClear=(rUbFactor−rLen)×cellElems=(4−3)×16=16
  → 清 R0.i=3 整段 [48, 64)
不需 ClearInnerBurstTailPadVf：BurstPad 在 A 方向，Reduce Pattern=RA 沿 R 累加、A 各 lane 独立，BurstPad 不串到 valid lane

→ Reduce 沿 R 累加，ExtensionPad 已清为 pad_value 零贡献，BurstPad 不参与 Reduce，结果正确
```

**场景 D：tail-R 非切尾轴 + 中间 R 轴 + partial**（rSplit<LastR 且中间隔 R 轴）

`A0 R0 A1 R1 A2 R2`（aSplit=A0，rSplit=R0，LastR=R2，中间隔 R1），fp16，R2=2（非对齐），rUbFactorAlign=20，rLen=14（partial），A entry 数=8。UB 从内到外：`[R2(2, padded 到 16), R1(2), R0(20), A(8)]`。每个 A entry 对应一段 40 行，partial 时每段开头 28 行（rLen 14 × R1 2）有数据：

```
需要清的行（r0∈[0,14)）：A entry 0 的段 [0,28)、A entry 1 的段 [40,68)、…，各段之间不连续
若按 "前 rLen×R1×A entry 数 = 14×2×8 = 224 行" 连续清 [0,224)：数量上恰好等于需要的总行数，位置却全错——a=6、7 段一行没清到

ClearInnerBurstTailPadVf: validR=axisShape[LastR]=2, rowStride=CeilAlign(2,16)=16,
  rowCnt = (A entry 数 8 × R0 切片数 20 × innerRProdAlign 32) ÷ rowStride 16 = 320（全量行）
    innerRProdAlign = R1 宽 2 × R2 宽 16（中间 R 轴并入，行数自然包含它）
  → 每行清 [2,16)，共 320 行
```

##### 5.1.2.2 清零总览

**决策汇总表**（BinaryBase 路径）：

> rLen：切分轴本次搬入ub的大小, 切分整块`=rUbFactor`，切分尾块 `< rUbFactor`

| tail | rChunk | rSplitIdx | 清零部分 |
|------|-------|--------|---------------------|
| tail-R | partial | ==LastR | ExtensionPad + BurstPad(尾轴非对齐) |
| tail-R | partial | <LastR | ExtensionPad + BurstPad(尾轴非对齐) |
| tail-R | full | 任意 | BurstPad(尾轴非对齐) |
| tail-A | partial | 任意 | ExtensionPad |
| tail-A | full | 任意 | 不需要显式清零 |

**清零准入条件及参数汇总**：  

```cpp
// rLen：切分轴本次搬入ub的大小, 切分整块`=rUbFactor`，切分尾块 `< rUbFactor`
if (rLen < rUbFactor) {
    // R 切分尾块，必清 ExtensionPad
    ClearChunkExtensionVf();

    // 清零参数描述
    // tail-R：BurstPad 按 A entry 逐行清
    extStart       = CeilAlign(rLen × innerRProdAlign, 8)   // 清零起点（8 = 32B / sizeof(fp32)），跳过 BurstPad 区间
    aStride        = rUbFactorAlign × innerRProdAlign         // 每个 A entry 的行宽（fp32 元素数）
    extLanes       = aStride − extStart                        // 每个 A entry 内清零长度
    aBundleEntries = aUbFactor × innerAProdAlign               // A entry 数（外层循环次数）

    // tail-A：ExtensionPad 连续整段清
    cellElems  = aUbFactor × innerAProdAlign × innerRProdAlign  // 每个 R 切分轴切片的元素数
    startElem  = rLen × cellElems                                // 清零起点
    totalClear = (rUbFactor − rLen) × cellElems                  // 清零总长度
}
```

```cpp
bsElem = blockSize / sizeof(D_T) // BurstPad 是按 D_T 计算的个数
if (isTailR_) {
    burstTail = (rSplitIdx == LastR) ? rLen : axisShape[LastR];
    if (burstTail % bsElem != 0) {
        ClearInnerBurstTailPadVf();

        // 清零参数描述
        // D_T 的 block 对齐元素数（fp32=8, fp16=16）
        bsElem    = blockSize / sizeof(D_T)
        // burst 尾轴有效元素数
        validR    = (rSplitIdx == LastR) ? rLen : axisShape[LastR]
        // rSplit==LastR：行宽 = R_bundle 宽；rSplit!=LastR：行宽 = burst 尾轴 padded 宽
        rowStride = (rSplitIdx == LastR) ? rUbFactorAlign × innerRProdAlign
                                        : CeilAlign(axisShape[LastR], bsElem)
        // rSplit==LastR：每 A entry 一行
        // rSplit!=LastR：UB 总元素数 ÷ 行宽 = 全量行（理由见 5.1.2.1 行序注意）
        rowCnt    = (rSplitIdx == LastR) ? aUbFactor × innerAProdAlign
                                         : aUbFactor × innerAProdAlign × rUbFactorAlign × innerRProdAlign / rowStride
    }
}
```

**实现要点**：
- **保留 valid 只清 pad**：从最后一个含 valid 数据的 fp32 block 起点开始写，mask 跳过 valid 部分，只写 padding 部分，不污染下一行
- **起点必须 block 对齐**：StoreAlign/LoadAlign 起点须block对齐，非对齐起点会运行时错
- **UpdateMask 引用语义**：`UpdateMask(N)` 生成"前 N lane 为 1"的 mask；参数是引用且被消耗（N>64 减 64、≤64 置 0），cntEnd/cntStart 必须是独立的非 const 局部变量


**ClearChunkExtensionVf 伪码**：

```cpp
// __simd_vf__ 函数：tail-R 分支
template <typename T = float>
__simd_vf__ inline void ClearChunkExtTailRVfImpl(
    __ubuf__ T* slotBase,
    uint32_t extStart, uint32_t aStride,
    uint32_t extLanes, uint16_t aU16, uint16_t repPerA,
    T padVal)  // kPadClearValue 作为参数传入
{
    constexpr uint32_t VL = VL_BYTES / sizeof(float);  // = 64，寄存器 VL（fp32 元素数），禁止写死
    AscendC::Reg::RegTensor<T> idReg;
    AscendC::Reg::Duplicate(idReg, padVal);
    for (uint16_t a = 0; a < aU16; ++a) {
        int32_t aOff = static_cast<int32_t>(a) * static_cast<int32_t>(aStride);
        uint32_t remaining = extLanes;
        for (uint16_t r = 0; r < repPerA; ++r) {
            int32_t off = aOff + static_cast<int32_t>(extStart) +
                          static_cast<int32_t>(r) * static_cast<int32_t>(VL);
            auto mask = AscendC::Reg::UpdateMask<T>(remaining);
            AscendC::Reg::StoreAlign(slotBase + off, idReg, mask);
        }
    }
}

// __simd_vf__ 函数：tail-A 分支
template <typename T = float>
__simd_vf__ inline void ClearChunkExtTailAVfImpl(
    __ubuf__ T* slotBase,
    uint32_t startElem, uint32_t totalClear, uint16_t repCount,
    T padVal)  // kPadClearValue 作为参数传入
{
    constexpr uint32_t VL = VL_BYTES / sizeof(float);  // = 64，寄存器 VL（fp32 元素数），禁止写死
    AscendC::Reg::RegTensor<T> idReg;
    AscendC::Reg::Duplicate(idReg, padVal);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalClear;
    for (uint16_t i = 0; i < repCount; ++i) {
        int32_t off = static_cast<int32_t>(startElem) +
                      static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::StoreAlign(slotBase + off, idReg, mask);
    }
}

// 调用侧
template <typename T = float>
__aicore__ inline void ClearChunkExtensionVf(
    __ubuf__ T* slotBase,             // preReduceResult 或 preReduceResultTail
    bool isTailR,                     // tail 类型
    uint32_t aBundleEntries,          // tail-R: aUbFactor × innerAProdAlign
    uint32_t innerRPA,                // tail-R: innerRProdAlign（tail-A 视图传 1）
    uint32_t cellElems,               // tail-A: aUbFactor × innerAProdAlign × innerRProdAlign（tail-R 视图传 0）
    uint32_t rUbFactor,
    uint32_t rUbFactorAlign,
    int64_t rLen)                     // 调用前已保证 rLen < rUbFactor
{
    constexpr uint32_t VL = VL_BYTES / sizeof(float);          // = 64，寄存器 VL，禁止写死
    constexpr uint32_t BS_TMP = UB_BLOCK_BYTES / sizeof(float); // 32B block（fp32 元素数）= 8，禁止写死

    if (isTailR) {
        const uint32_t rLenInner = static_cast<uint32_t>(rLen) * innerRPA;
        const uint32_t extStart  = (rLenInner + BS_TMP - 1) / BS_TMP * BS_TMP;  // 32B 对齐起点
        const uint32_t aStride   = rUbFactorAlign * innerRPA;
        if (extStart >= aStride) {
            return;
        }
        const uint32_t extLanes  = aStride - extStart;
        const uint32_t repPerA   = (extLanes + VL - 1) / VL;
        const uint16_t aU16      = static_cast<uint16_t>(aBundleEntries);

        asc_vf_call<ClearChunkExtTailRVfImpl<T>>(
            slotBase, extStart, aStride, extLanes, aU16, static_cast<uint16_t>(repPerA),
            static_cast<T>(kPadClearValue));
    } else {  // tail-A：rSplit 在 UB 最外层，单段连续清零
        const uint32_t startElem  = static_cast<uint32_t>(rLen) * cellElems;
        const uint32_t totalClear = (rUbFactor - static_cast<uint32_t>(rLen)) * cellElems;
        const uint32_t repCount   = (totalClear + VL - 1) / VL;

        asc_vf_call<ClearChunkExtTailAVfImpl<T>>(
            slotBase, startElem, totalClear, static_cast<uint16_t>(repCount),
            static_cast<T>(kPadClearValue));
    }
}
```

**MergeTmpBufVf 伪码**：

```cpp
// __simd_vf__ 函数
template <typename T = float>
__simd_vf__ inline void MergeTmpBufVfImpl(
    __ubuf__ T* p0, __ubuf__ T* p1,
    uint32_t totalElems, uint16_t repU16)
{
    AscendC::Reg::RegTensor<T> aReg, bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repU16; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(aReg, p0 + off);
        AscendC::Reg::LoadAlign(bReg, p1 + off);
        AscendC::Reg::Add(aReg, aReg, bReg, mask);     // 占位：reducer 的 merge（sum 系 = Add，max/min = Max/Min）
        AscendC::Reg::StoreAlign(p0 + off, aReg, mask);
    }
}

// 调用侧
template <typename T = float>
__aicore__ inline void MergeTmpBufVf(
    __ubuf__ T* p0,                   // preReduceResult（二分主块，merge 结果写回此处）
    __ubuf__ T* p1,                   // preReduceResultTail（二分尾块，merge 后释放）
    uint32_t totalElems)              // aUbFactor × innerAProdAlign × rUbFactorAlign × innerRProdAlign
{
    constexpr uint32_t VL = VL_BYTES / sizeof(float);  // = 64，寄存器 VL，禁止写死
    uint16_t repU16 = static_cast<uint16_t>((totalElems + VL - 1) / VL);
    asc_vf_call<MergeTmpBufVfImpl<T>>(p0, p1, totalElems, repU16);
}
```

**ClearInnerBurstTailPadVf 伪码**：

```cpp
// __simd_vf__ 函数
template <typename TmpT = float>
__simd_vf__ inline void ClearInnerBurstTailPadVfImpl(
    __ubuf__ TmpT* base,
    uint16_t rowCntU16, int32_t rowStrideI, int32_t windowOff,
    uint32_t padEnd, uint32_t partialStartInBlock,
    TmpT padVal)  // kPadClearValue 作为参数传入
{
    AscendC::Reg::RegTensor<TmpT> idReg;
    AscendC::Reg::Duplicate(idReg, padVal);   // 不是 0：max→-inf, prod→1, ...

    uint32_t cntEnd   = padEnd;
    uint32_t cntStart = partialStartInBlock;
    auto maskEnd   = AscendC::Reg::UpdateMask<TmpT>(cntEnd);
    auto maskStart = AscendC::Reg::UpdateMask<TmpT>(cntStart);
    auto allMask   = AscendC::Reg::CreateMask<TmpT, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg notStart, padMask;
    AscendC::Reg::Not(notStart, maskStart, allMask);
    AscendC::Reg::And(padMask, maskEnd, notStart, allMask);

    for (uint16_t row = 0; row < rowCntU16; ++row) {
        int32_t rowOff = static_cast<int32_t>(row) * rowStrideI;
        AscendC::Reg::StoreAlign(base + rowOff + windowOff, idReg, padMask);
    }
}

// 调用侧
template <typename InputT, typename TmpT = float>   // preReduceResult/preReduceResultTail 固定 fp32
__aicore__ inline void ClearInnerBurstTailPadVf(__ubuf__ TmpT* base,
                                                uint32_t rowCnt,
                                                uint32_t rowStride,      // UB 行步距（元素数，调用方算）
                                                uint32_t validR)         // R_inner（业务有效列数）
{
    constexpr uint32_t BS_INPUT = UB_BLOCK_BYTES / sizeof(InputT);  // CopyIn 对齐用：b16=16，fp32=8，禁止写死
    constexpr uint32_t BS_TMP   = UB_BLOCK_BYTES / sizeof(TmpT);    // VL 视窗下块边界用：fp32=8，禁止写死
    if (validR >= rowStride) {
        return;                        // 无 padding，直接返回
    }

    uint32_t padEndInRow         = ((validR + BS_INPUT - 1) / BS_INPUT) * BS_INPUT;   // CeilAlign(validR, BS_INPUT)
    uint32_t partialBlockIdx     = validR / BS_TMP;
    uint32_t partialStartInBlock = validR % BS_TMP;
    uint32_t padEnd              = padEndInRow - partialBlockIdx * BS_TMP;

    uint16_t rowCntU16  = static_cast<uint16_t>(rowCnt);
    int32_t  rowStrideI = static_cast<int32_t>(rowStride);
    int32_t  windowOff  = static_cast<int32_t>(partialBlockIdx * BS_TMP);

    asc_vf_call<ClearInnerBurstTailPadVfImpl<TmpT>>(
        base, rowCntU16, rowStrideI, windowOff, padEnd, partialStartInBlock,
        static_cast<TmpT>(kPadClearValue));
}
```

#### 5.1.3 A 方向 BurstPad/ExtensionPad 隔离

A 方向的 BurstPad 和 ExtensionPad **不需要主动清零**，因为 Reduce 沿 R 维度合并，A 各位置独立

**与 R 方向的关键区别**：R 方向 BurstPad/ExtensionPad 会被 Reduce 沿 R 累加进结果 → 必须显式清。A 方向 Reduce 沿 R 走、A 各 lane 独立 → 天然隔离。**禁止额外 A 方向清零操作**。

#### 5.1.4 ReduceSum 高阶 API

API 原型、硬约束详见 [reduction-highlevel-api.md](../reduction-highlevel-api.md)。

> srcShape 即 UB 内数据排布的维度（由 §4.2 CopyIn 时 DataCopyPad stride/blockCount 决定）：
> - tail-R：`[A_bundle, R_bundle]` = `{aUbFactor × innerAProdAlign, rUbFactorAlign × innerRProdAlign}`
> - tail-A：`[R_bundle, A_bundle]` = `{rUbFactorAlign × innerRProdAlign, aUbFactor × innerAProdAlign}`

```cpp
// 运行时选择，各分支内 Pattern 仍是编译期常量
// isTailR → Pattern::Reduce::AR（沿内层 R reduce，保留外层 A）
// !isTailR → Pattern::Reduce::RA（沿外层 R reduce，保留内层 A）

const uint32_t laneA = aUbFactor * innerAProdAlign;  // padded A 维 lane 数（与 Reduce dst 写入形状一致）
constexpr uint32_t BLOCK_F32 = UB_BLOCK_BYTES / sizeof(float);   // = 8，禁止写死
const uint32_t levelStride = (laneA + BLOCK_F32 - 1) / BLOCK_F32 * BLOCK_F32;
int32_t levelOff = cacheID × levelStride; // cacheID 由 GetCacheID 算

// srcShape 按 padded 值计算
if (isTailR_) {
    uint32_t srcShape[2] = {static_cast<uint32_t>(aUbFactor * innerAProdAlign),
                            static_cast<uint32_t>(rUbFactorAlign * innerRProdAlign)};
    ReduceSum<float, AscendC::Pattern::Reduce::AR, /*isReuseSource=*/true>(
        cacheBuf + levelOff, preReduceResult, preReduceResultTail, srcShape, /*srcInnerPad=*/true);
} else {
    uint32_t srcShape[2] = {static_cast<uint32_t>(rUbFactorAlign * innerRProdAlign),
                            static_cast<uint32_t>(aUbFactor * innerAProdAlign)};
    ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
        cacheBuf + levelOff, preReduceResult, preReduceResultTail, srcShape, /*srcInnerPad=*/true);
}
//   dst=cacheBuf[levelOff]  src=preReduceResult  sharedTmpBuffer=preReduceResultTail
```

本范式统一使用带 sharedTmpBuffer 的重载，`preReduceResultTail` 兼作 sharedTmpBuffer。BinaryBase 路径下 buffer 与 ReduceSum API 参数的映射关系：

| ReduceSum API 参数 | 本范式 buffer | 说明 |
|---|---|---|
| `dst` | `cacheBuf[levelOff]` | 二分缓存树当前层，Reduce 结果直接写入 |
| `src` | `preReduceResult` | 二分主块 或 merge 后结果 |
| sharedTmpBuffer | `preReduceResultTail` | Phase A 配对后空闲，兼作 sharedTmpBuffer |

共需 **3 份** buffer：`preReduceResult` + `preReduceResultTail`（sharedTmpBuffer 复用）+ `cacheBuf`（dst）。

#### 5.1.5 PostElewise 计算

> **post_reduce_input 搬运约束**：搬入 UB 后排布必须与 Reduce 输出的 padded 排布（aUbFactor × innerAProdAlign）一致，按 A 轴切分搬入。

从 cacheBuf 树根取 reduce 结果：`rootOff = (cacheCount - 1) × levelStride`，`rootPtr = cacheBuf + rootOff`。PostElewise 读 `rootPtr`，按 `laneA = aUbFactor × innerAProdAlign`（padded）整行处理（算子专属 elewise + 缩位 Cast），结果写入 outBuf，供 CopyOut 搬出。

```cpp
constexpr uint32_t kRepF32 = VL_BYTES / sizeof(float);  // = 64，寄存器 VL（VL_BYTES=GetVRegSize() 派生），禁止写死

// __simd_vf__ 函数
template <typename DType>
__simd_vf__ inline void PostElewiseVfImpl(
    __ubuf__ float* src, __ubuf__ DType* outBuf,
    uint32_t laneA, uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneA;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(f32Reg, src + off);

        // 1) 算子专属 PostElewise（如 mean 的 ×invRTotal；每算子替换）
        PostElewiseOp(f32Reg, mask);

        // 2) 缩位 Cast + 写出
        if constexpr (NeedCast<DType>) {
            AscendC::Reg::RegTensor<DType> bReg;
            AscendC::Reg::Cast<DType, float, kCastTraitFromFp32>(bReg, f32Reg, mask);
            AscendC::Reg::StoreAlign<DType, AscendC::Reg::StoreDist::DIST_PACK_B32>(outBuf + off, bReg, mask);
        } else {
            AscendC::Reg::StoreAlign(outBuf + off, f32Reg, mask);
        }
    }
}

// 调用侧
template <typename DType>
__aicore__ inline void PostElewise(__ubuf__ float* src, __ubuf__ DType* outBuf, uint32_t laneA)
{
    const uint16_t repeatTime = static_cast<uint16_t>((laneA + kRepF32 - 1) / kRepF32);
    asc_vf_call<PostElewiseVfImpl<DType>>(src, outBuf, laneA, repeatTime);
}
```


### 5.2 Cast知识

引用公共知识 [common/cast-rules.md](../../../common/cast-rules.md)。

**使用位置**：
- **CopyIn 后、Reduce 前**：b16 输入 `DIST_UNPACK_B16` load → `Cast<b16→fp32>` → fp32 计算
- **Reduce 后、CopyOut 前**：fp32 结果 `Cast<fp32→b16>` → `DIST_PACK_B32` store → outBuf

**必须的 CastTrait**：

```cpp
// 扩位 b16 → fp32
CastTrait{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::CAST_NONE}

// 缩位 fp32 → b16
CastTrait{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT}
```

### 5.3 VF 编程规则

引用公共知识 [common/vf-programming-rules.md](../../../common/vf-programming-rules.md)。

**DoCaching VF 骨架**（BinaryBase 专用，Reduce 结果已在 `cacheBuf[levelOff]`，DoCaching 就地吸收低层）：

```cpp
// __simd_vf__ 函数
__simd_vf__ inline void DoCachingVfImpl(
    __ubuf__ float* cacheBuf,           // Reduce 结果已写入 cacheBuf[levelOff]
    uint32_t laneA, uint32_t levelStride, int32_t levelOff,
    uint16_t repeatTime, uint16_t cacheLvlU16)
{
    AscendC::Reg::RegTensor<float> aReg, bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneA;

    for (uint16_t i = 0; i < repeatTime; ++i) {             // A 维分段（kRepF32 = 64 lane）
        int32_t off = i * kRepF32;
        mask = AscendC::Reg::UpdateMask<float>(remaining);  // 自动递减 kRepF32

        // 读取本层（Reduce 已写入的结果）
        AscendC::Reg::LoadAlign(aReg, cacheBuf + levelOff + off);

        for (uint16_t j = 0; j < cacheLvlU16; ++j) {
            int32_t jOff = j * levelStride + off;
            AscendC::Reg::LoadAlign(bReg, cacheBuf + jOff);
            AscendC::Reg::Add(aReg, aReg, bReg, mask);      // 占位：reducer 的 merge（sum 系 = Add，max/min = Max/Min）
        }

        AscendC::Reg::StoreAlign(cacheBuf + levelOff + off, aReg, mask);
    }
}

// 调用侧
const uint32_t laneA = aUbFactor * innerAProdAlign;  // padded A 维 lane 数（与 Reduce dst 写入形状一致）
constexpr uint32_t BLOCK_F32 = UB_BLOCK_BYTES / sizeof(float);   // = 8，禁止写死
const uint32_t levelStride = (laneA + BLOCK_F32 - 1) / BLOCK_F32 * BLOCK_F32;
int32_t levelOff = cacheID × levelStride;
uint16_t repeatTime = (laneA + kRepF32 - 1) / kRepF32;
uint16_t cacheLvlU16 = static_cast<uint16_t>(cacheID);

asc_vf_call<DoCachingVfImpl>(
    cacheBuf, laneA, levelStride, levelOff, repeatTime, cacheLvlU16);
```

**要点**：
- **双层 uint16_t 循环**：外层按 A 维 kRepF32 分段，内层按 cacheID 吸收
- **`levelStride` 与 `laneA` 的区别**：`laneA`=padded A 维 lane 数（= `aUbFactor × innerAProdAlign`，与 Reduce dst 写入形状一致），`levelStride`=cacheBuf 每层在 fp32 lane 上的物理 stride（CeilAlign 到 32B），AR + A bundle=1 等极小 A 场景两者背离

## 6 CopyOut MTE3 逻辑

### 6.1 UB Buffer数据排布格式

Reduce 把 R 合掉之后，UB 上保留的 A_bundle 与 CopyIn 时的结构一致。**关键不同**：输出 GM 上没有 R 轴穿插，纯 A 轴 dense 排列。所以**无论 A_bundle 有几根 A 轴，CopyOut 永远塌成 2D，一次 DataCopyPad 搞定**——不需要 Loop 模式、不需要 for 拆外层。

| pattern tail | UB 输出布局 | CopyOut 策略 |
|---|---|---|
| **tail-R** | `[A_bundle 紧密]`（dense，最内 A 不 padded） | **单 burst**：`blockCount=1, blockLen=aLen × innerAProd × sizeof` |
| **tail-A + aSplitIdx==LastA** | 单根 A 轴 `[aUbFactor]` | 单 burst：`blockCount=1, blockLen=aLen × sizeof` |
| **tail-A + aSplitIdx!=LastA** | `[A_bundle, 最内 A 按 block 对齐]` | 多 burst：`blockCount=aLen×innerAProd/LastA, blockLen=LastA×sizeof`，HW 按 32B 跨 padding |

> `innerAProd` = ∏ axisShape[aSplitIdx+1..LastA]（aSplitIdx 右侧所有 A 轴的真实元素乘积），kernel 端从 `axisShape` 现算，不依赖 TilingData 传递。tail-R 下 `innerAProd == innerAProdAlign`（Climb 不调 CeilAlign）。

### 6.2 CopyOut 指令

CopyOut 字段映射依 tail 类型（`isTailR_` 运行时值）与 `aSplitIdx` 位置分**三条独立路径**，**不能合并为单一通用公式**。

```cpp
template <typename DType>
__aicore__ inline void CopyOut(int64_t outerOutOff, int64_t aLen, AscendC::LocalTensor<D_T>& outDeq)
{
    // innerAProd = ∏ axisShape[aSplitIdx+1..LastA]，kernel 端现算
    int64_t innerAProd = 1;
    for (int32_t k = aSplitIdx + 2; k <= LastAAxis(); k += 2) {
        innerAProd *= axisShape[k];
    }

    DataCopyExtParams outParams;
    if (isTailR_) {
        // 路径 1：tail-R，单 burst
        outParams.blockLen = static_cast<uint32_t>(aLen * innerAProd * static_cast<int64_t>(sizeof(D_T)));
        outParams.blockCount = 1;
    } else {
        const int32_t lastA = LastAAxis();
        const int64_t lastASize = axisShape[lastA];
        if (aSplitIdx == lastA) {
            // 路径 2：tail-A + aSplitIdx == LastA，单 burst
            outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(D_T)));
            outParams.blockCount = 1;
        } else {
            // 路径 3：tail-A + aSplitIdx != LastA，多 burst
            outParams.blockLen = static_cast<uint32_t>(lastASize * static_cast<int64_t>(sizeof(D_T)));
            outParams.blockCount = static_cast<uint16_t>(aLen * innerAProd / lastASize);
        }
    }
    outParams.srcStride = 0;   // UB 侧 gap=0，HW 自动按 CeilAlign(blockLen, 32B) 读取下一个块
    outParams.dstStride = 0;   // GM 侧 byte 对齐，dense 写出
    outParams.rsv = 0;         // 必须显式填 0（datacopypad-rules.md）
    DataCopyPad(yGm_[outerOutOff], outDeq, outParams);
}
```

**约束**：
- **不要把路径 3 的多 burst 公式套到 tail-R 上**：tail-R 下 A_bundle 是 dense 无 padding，多 burst 的 CeilAlign(blockLen, 32B) 跨度会跳过有效数据
- CopyOut 全部用 `srcStride = 0, dstStride = 0`
- CopyOut 不需要 `DataCopyPadExtParams`（UB→GM 方向不支持也不需 padParams）
- CopyOut 不区分 partial chunk（aLen 自然覆盖 A 切分尾块）

**4 个典型例**：

**例 1（路径 1，tail-R）**：pattern `A0R0A1R1`，aSplitIdx=A0，A_bundle = `[aUbFactor(A0), A1]`：
```cpp
blockLen   = aLen * A1 * sizeof(D_T);
blockCount = 1;  srcStride = 0;  dstStride = 0;
```

**例 2（路径 2，tail-A 单 A 轴）**：pattern `A0R0A1`，aSplitIdx=A1=LastA：
```cpp
blockLen   = aLen * sizeof(D_T);
blockCount = 1;  srcStride = 0;  dstStride = 0;
```

**例 3（路径 3，tail-A 2 根 A）**：pattern `A0R0A1R1A2`，aSplitIdx=A0：
```cpp
blockLen   = A2 * sizeof(D_T);
blockCount = aLen * A1;
srcStride  = 0;  dstStride = 0;
```

**例 4（路径 3，tail-A 深 pattern 3 根 A）**：pattern `A0R0A1R1A2R2A3R3A4`，aSplitIdx=A0：
```cpp
blockLen   = A4 * sizeof(D_T);
blockCount = aLen * A1 * A2 * A3;
srcStride  = 0;  dstStride = 0;
```

## 7 流水同步

### 7.1 Sync 知识

引用公共知识 [common/sync-and-consistency.md](../../../common/sync-and-consistency.md) §2.2（持有法则；本范式将推导出的同步点以 Mutex Lock/Unlock 段实现）。

**TBuf 模型**：Mutex Lock/Unlock 段式流水同步（`AscendC::Mutex::Lock/Unlock`）。
三段流水——MTE2 搬入段 → V 计算段 → MTE3 搬出段——以同一 MutexID 的 Lock/Unlock 段链式串行：
后一段等前一段 Unlock 后才进入，跨迭代 WAR 也由段链顺序天然覆盖。MutexID 必须经
`AscendC::AllocMutexID()` 申请、`AscendC::ReleaseMutexID()` 释放（TPipe 范式强制，禁止硬编码/自行管理）；
同 id 的 Lock/Unlock 不得嵌套。单流水内：不同的 VF 间硬件保证串行，不需要 PipeBarrier<PIPE_V>()。

**同步点推导方法**（持有法则驱动，禁止跳过）：

1. 取 §2.1 Process 无同步版代码，逐行标注三态持有（`// 执行前/中/后: 持有=[...]`）
2. 遍历每个 buffer 的生命周期，标记跨流水线的 RAW（生产者写→消费者读）和 WAR（消费者读→生产者覆写）
3. 每个 crossing 的生产者/消费者分别划入对应流水的 Lock/Unlock 段（DataCopyPad → MTE2/MTE3 段，VF/Reduce → V 段），同 id 段链即建立依赖
4. VF 链中间无 sync：CastSquare → ClearPad → Merge → Reduce → DoCaching → PostElewise 全在 V 流水线内，硬件保证串行
5. 循环间反向 WAR（如上轮 MTE3 读→下轮 V 覆写、上轮 V 读→下轮 MTE2 覆写）：无需首末轮跳过技巧，同 id 段链跨迭代自动按序覆盖——**仅在持有法则 trace 确认存在跨迭代 WAR 时才需要保证段序**

> ⚠ **禁止凭直觉预设同步点**。是否存在跨流水依赖、各段如何划分，完全取决于持有法则 trace 的结果。不同算子的 buffer 复用方式不同，同步点可能不同。

**Buffer 非复用约束**（持有法则 trace 的设计输入）：
- `outBuf`（postReducePhase buffer）与 `preInBuf`（preReducePhase buffer）是独立分配的物理 buffer，跨 aLoopIdx 迭代不复用
- 因此 CopyOut（MTE3 读 outBuf）与下轮 CopyIn（MTE2 写 preInBuf）**不构成 WAR**，两段间无需额外依赖
- `outBuf` 跨 aLoopIdx 迭代复用：本轮 CopyOut（MTE3 读 outBuf）→ 下轮 PostElewise（V 写 outBuf）构成跨迭代 MTE3→V WAR——下轮 V 段排在本轮 MTE3 段之后即被同 id 段链覆盖
- `preInBuf` 在 R chunk 循环内可能复用（Phase A 主块→尾块）：循环内 V→MTE2 WAR——尾块 MTE2 段排在主块 V 段之后
- `preInBuf` 跨 rIdx 迭代复用：本轮 V 链末尾 → 下一轮 rIdx 的 CopyIn 构成跨迭代 V→MTE2 WAR——下轮 MTE2 段排在本轮 V 段之后
- `preInBuf` 跨 aLoopIdx 迭代复用：本轮 aLoop 最后一个 R chunk 的 V → 下一轮 aLoop 第一个 R chunk 的 CopyIn（MTE2 覆写 preInBuf）构成跨 aLoopIdx 的 V→MTE2 WAR——段链跨迭代连续即覆盖

**持有法则 trace 必查项**：
- R chunk 循环内 WAR：同一 rIdx 迭代内，是否存在 V 读 buffer → MTE2 覆写同一 buffer 的 Phase A 配对
- R chunk 跨迭代 WAR：本轮 V 链末尾读的 buffer，是否在下一轮 rIdx 被 MTE2 覆写
- aLoopIdx 跨迭代 WAR：本轮最后一个 R chunk 的 V 链 → 下一轮第一个 R chunk 的 CopyIn（MTE2 覆写 preInBuf），preInBuf 跨 aLoopIdx 复用
- outBuf 跨迭代 WAR：本轮 CopyOut（MTE3 读 outBuf）→ 下轮 PostElewise（V 写 outBuf），下轮 V 段须排在本轮 MTE3 段之后

**MutexID 获取**（Process 入口统一申请、末尾释放，与 Lock/Unlock 严格配对）：

```cpp
// Process/ProcessGroup 入口申请（TPipe 范式禁止硬编码/自行管理 MutexID）
uint8_t mutexId = AscendC::AllocMutexID();
// ……三段流水：Lock<PIPE_MTE2>(mutexId)→Unlock → Lock<PIPE_V>(mutexId)→Unlock →
//    Lock<PIPE_MTE3>(mutexId)→Unlock，同 id 段链式串行 ……
AscendC::ReleaseMutexID(mutexId);   // 末尾释放
```

### 7.2 范式特殊的同步

**无跨核同步**（Base 模板不调用 SyncAll）。

**outBuf 跨迭代 MTE3→V WAR**：PostElewise（V 写 outBuf）划入 V 段、CopyOut（MTE3 读 outBuf）划入 MTE3 段——同 id 段链保证上轮 MTE3 段完成后才进入下轮 V 段。

## 8 kernel 产出校验

暂不涉及。
