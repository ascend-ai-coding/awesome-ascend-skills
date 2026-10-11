# broadcast 范式 Kernel 辅助函数

> Kernel 侧泛型辅助函数的完整实现：坐标解码、偏移计算、搬运量计算、CopyInBrc。

依赖输入：broadcast-template-overview.md、broadcast-standard-kernel-template.md

## 1 四个泛型辅助函数

> 对应 patterns.md §3.4.3 Kernel 调度框架。四个函数直接抄模板，只改类型名；搬运量计算函数 `CalcTransferCount` 见 §2（与 §1 共同构成 patterns.md 要求的 5 个必抄函数）。

### 1.1 GetCoreRange

核间 tile 区间解码。`blockIdx` → 本核负责的 `[start, end)` flat tile 区间。

```cpp
__aicore__ inline void GetCoreRange(int64_t coreId, int64_t mainTiles, int64_t mainCoreNum,
    int64_t& start, int64_t& end)
{
    if (coreId < mainCoreNum) {
        start = coreId * mainTiles;
        end = start + mainTiles;
    } else {
        start = mainCoreNum * mainTiles + (coreId - mainCoreNum) * (mainTiles - 1);
        end = start + mainTiles - 1;
    }
}
```

前 `mainCoreNum` 个核各处理 `mainTiles` 块，其余核各处理 `mainTiles-1` 块（母模板核间均衡切分公式）。

### 1.2 GetUBSplitRange

核内 UB 切分段大小。`ubBlockIdx`（split 轴外层偏移）→ 当前段实际元素数 `ubBlockLength`。

```cpp
__aicore__ inline int64_t GetUBSplitRange(
    int64_t ubBlockIdx, int64_t ubOuter, int64_t ubFactor, int64_t ubTail)
{
    return (ubBlockIdx == ubOuter - 1) ? ubTail : ubFactor;
}
```

末段（`ubBlockIdx == ubOuter - 1`）返回 `ubTail`（尾块大小），其余段返回 `ubFactor`（标准段大小）。

### 1.3 FlatToEffectiveCoord

flat 索引 → effective_shape 坐标。将核间分配的 flat tile 索引解码为多维坐标。

```cpp
__aicore__ inline bool FlatToEffectiveCoord(int64_t flat, const int64_t *maxBroShape,
    int64_t rank, int64_t ubSplitIdx, int64_t ubFactor, int64_t ubOuter, int64_t *effCoord)
{
    for (int64_t d = 0; d < rank; d++) {
        effCoord[d] = 0;
    }
    int64_t ubBlockIdx = flat % ubOuter;
    int64_t outer = flat / ubOuter;
    for (int64_t d = ubSplitIdx - 1; d >= 0; d--) {
        effCoord[d] = outer % maxBroShape[d];
        outer /= maxBroShape[d];
    }
    effCoord[ubSplitIdx] = ubBlockIdx * ubFactor;
    return true;
}
```

解码逻辑：
- `ubBlockIdx = flat % ubOuter` → split 轴的外层偏移
- `outer = flat / ubOuter` → split 轴以外各维的 flat 索引
- 从 ubSplitIdx-1 向外逐维取模解码
- `effCoord[ubSplitIdx] = ubBlockIdx * ubFactor` → split 轴坐标 = 外层偏移 × 内轴段大小

### 1.4 CalcOffset

effective_shape 坐标 → GM 元素偏移。broadcast 轴 stride=0，实现随路广播。

> **命名约束**：输入与输出的偏移计算规则完全相同，**共用一份 `CalcOffset` 实现**，不拆分为 CalcInputOffset / CalcOutputOffset。

```cpp
__aicore__ inline int64_t CalcOffset(
    const int64_t *effCoord, const int64_t *strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++) {
        offset += effCoord[d] * strides[d];
    }
    return offset;  // 元素个数，gmIn_[]/gmOut_[] 的 index
}
```

> ⛔ **入参单位约束**：返回值是**元素个数**（非字节偏移）。`GlobalTensor<T>::operator[]` 取元素索引。若函数内部做了 `* sizeof(T)` 返回字节，则 `gmIn_[bytes]` 会被误当元素索引，地址错位 ×4。

## 2 搬运量计算函数

> `CopyOutOne` 内部调用（CopyInBrc 的搬运量由 NDDMA params + ubBlockLength 决定，不走本函数），计算本次搬运的实际元素数。

### 2.1 CalcTransferCount

> **命名约束**：输入与输出的搬运量计算规则完全相同，**共用一份 `CalcTransferCount` 实现**，不拆分为 CalcInputTransferCount / CalcOutputTransferCount。

```cpp
__aicore__ inline int64_t CalcTransferCount(
    const int64_t *normalShape, int64_t rank, int64_t ubSplitIdx, int64_t ubBlockLength)
{
    int64_t splitElems = (normalShape[ubSplitIdx] == 1) ? 1 : ubBlockLength;
    int64_t innerElems = 1;
    for (int64_t d = ubSplitIdx + 1; d < rank; d++) {
        innerElems *= normalShape[d];
    }
    return splitElems * innerElems;
}
```

- 输出侧：Host 侧已保证输出稠密（normalOutputShape = maximumBroShape，广播输出被 `CheckBroadcastShape` 拒绝），`normalShape[ubSplitIdx] == 1` 分支仅作防御性保留
- split 轴为 broadcast 轴（`normalShape[ubSplitIdx] == 1`）时，split 段只搬 1 个元素（broadcast 由 NDDMA 展开）
- 否则搬 `ubBlockLength` 个元素
- innerElems = split 轴内侧所有维的连乘

## 3 CopyInBrc

> 对应 patterns.md §3.4.7。NDDMA broadcast 搬入的核心函数，含 `if constexpr (RANK <= MAX_NDDMA_DIMS)` 分叉。

**前置知识**：NDDMA API 结构体（`NdDmaLoopInfo` / `NdDmaParams` / `NdDmaConfig`）及三原则见 [broadcast-standard-kernel-template.md](broadcast-standard-kernel-template.md) §4。本节只讲 CopyInBrc 函数的完整实现。

```cpp
__aicore__ inline void CopyInBrc(
    const int64_t *coord, int inputIdx, int slot, int64_t ubBlockLength)
{
    int64_t k = td_->split.ubSplitIdx;
    int64_t off = CalcOffset(coord, td_->inputStrides[inputIdx], RANK);
    const int64_t *dstShape = td_->maxBroShape;

    auto params = nddmaParams_[inputIdx];
    int64_t kNd = RANK - 1 - k;
    int64_t inner = 1;
    for (int64_t nd = 0; nd < NDDMA_DIMS; nd++) {
        if (nd == kNd) {
            params.loopInfo.loopSize[nd] = ubBlockLength;
        }
        params.loopInfo.loopDstStride[nd] = inner;
        inner *= params.loopInfo.loopSize[nd];
    }

    static constexpr AscendC::NdDmaConfig cfg = { false, AscendC::NdDmaConfig::unsetPad,
                                                   AscendC::NdDmaConfig::unsetPad, false };

    if constexpr (RANK <= MAX_NDDMA_DIMS) {
        AscendC::DataCopy<T, NDDMA_DIMS, cfg>(
            buf_[slot].Get<T>(), gmIn_[inputIdx][off], params);
    } else {
        AscendC::LocalTensor<T> buf = buf_[slot].Get<T>();
        int64_t elemBase = off;
        for (int64_t oi = 0; oi < nddmaOuterIters_[inputIdx]; oi++) {
            int64_t elemAdj = 0;
            int64_t tmp = oi;
            for (int64_t d = RANK - nddmaDims_ - 1; d >= k; d--) {
                int64_t sz = (d == k) ? ubBlockLength : dstShape[d];
                elemAdj += (tmp % sz) * td_->inputStrides[inputIdx][d];
                tmp /= sz;
            }
            AscendC::DataCopy<T, NDDMA_DIMS, cfg>(
                buf[oi * inner], gmIn_[inputIdx][elemBase + elemAdj], params);
        }
    }
}
```

**分叉逻辑**：

| 分支 | 条件 | 行为 |
|------|------|------|
| 单次搬运 | `RANK <= MAX_NDDMA_DIMS`（5） | 单次 `DataCopy<T, NDDMA_DIMS, cfg>` 覆盖所有维 |
| 逐段搬运 | `RANK > MAX_NDDMA_DIMS` | `nddmaOuterIters_` 次 `DataCopy`，每次覆盖最内 5 维，外维通过 `elemAdj` 偏移 |

**运行时参数更新**：`Init()` 中预计算 NDDMA 静态参数（loopSrcStride / loopLpSize / loopRpSize），`CopyInBrc()` 中只更新 split 轴的 `loopSize`（替换为当前段 `ubBlockLength`）和所有维的 `loopDstStride`（基于累积 inner 重新计算）。

**`nddmaOuterIters_` 的计算**（Init 中）：

```cpp
nddmaOuterIters_[inp] = 1;
for (int64_t d = k; d < RANK - nddmaDims_; d++) {
    nddmaOuterIters_[inp] *= (d == k) ? td_->split.ubFactor : dstShape[d];
}
```

覆盖 flat 层和 NDDMA 之间的 gap 维度（`d = k .. RANK-nddmaDims-1`）。

**三层不重叠**：
- Flat loop（`d < k`）：由 `FlatToEffectiveCoord` 解码
- Outer loop（`k ≤ d < RANK-nddmaDims`）：由 `nddmaOuterIters_` 逐段搬运
- NDDMA（`d ≥ RANK-nddmaDims`）：由单次 `DataCopy` 覆盖

## 4 函数调用关系

```
Process():
  GetCoreRange(blockIdx, ...) → [start, end)
  for flat in [start, end):
    GetUBSplitRange(flat % ubOuter, ...) → ubBlockLength
    FlatToEffectiveCoord(flat, ...) → coord
    CopyInBrc(coord, inputIdx, slot, ubBlockLength):
      CalcOffset(coord, strides, RANK) → off
      [if constexpr 分叉] DataCopy or 逐段 DataCopy
    Compute(VF)
    CopyOutOne(coord, outputIdx, slot, ubBlockLength):
      CalcOffset(coord, strides, RANK) → off
      CalcTransferCount(shape, RANK, ubSplitIdx, ubBlockLength) → cnt
      DataCopyPad(gmOut[off], buf[slot], cnt)
```
