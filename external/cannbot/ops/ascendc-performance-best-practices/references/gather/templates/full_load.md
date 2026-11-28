# Full-Load 模板（GA 全载 + SIMT 从 UB 取值）

## 解决的问题

当 gatherDimSize × innerSize 可全载入 UB，但 innerSize 不足以走 SIMD 行拷贝（< 2048）时，将 x 的 GA 块一次性搬入 UB，再用 SIMT 线程从 UB 读值。相比直接 GM 读的 SIMT 模板，UB 读取延迟远低于 GM，适合**索引高度复用**（gatherSize/gatherDimSize > 1）的场景。

## 适用条件

| 条件 | 要求 |
|------|------|
| gatherDimSize × innerSize × improveDtypeSize | < UB - DCache(128KB) |
| batchSize | == 1 |
| outerSize | == 1（2D 场景）或 > 1（走 LoopColProcess 分块循环） |
| inputBatchDims | == 0 |
| ySize × improveDtypeSize | > 2048（最小输出阈值，太小不值得全载） |
| gatherSize / gatherDimSize | > 1（索引复用，否则全载无收益） |
| innerSize × improveDtypeSize | < 2048（否则走 SIMD 行拷贝） |

> 当 outerSize > 1 时，UB 无法一次载入所有 GA 块，按 `maxGdimA = ubAviable / (gatherDimSize × innerSize)` 分块循环载入，每块独立 SIMT gather。

## 执行流程

```
Init:
  xGm / indicesGm / yGm SetGlobalBuffer
  inQueue_ (TQue<VECIN, depth=1>, 大小 = maxElement × dtypeSize)

Process:
  if batchSize==1 && outerSize>1:
    LoopColProcess()   // 分块循环
  else:
    NoLoopProcess()    // 单块全载

NoLoopProcess:
  1. CopyIn: x[0 : gatherDimSize×innerSize] GM → UB (DataCopyPad)
  2. GetUintDivMagicAndShift(m0, shift0, innerSize)  // fast-div
  3. asc_vf_call<GatherSimt2D<NIS>>(threadNum, ...):
     每线程: yIndex → gatherI = UintDiv(yIndex, m0, shift0)
                       innerI = yIndex - gatherI × innerSize
              idx = indices[gatherI]  (GM 读)
              y[yIndex] = srcTensor[idx × innerSize + innerI]  (UB 读)

LoopColProcess:
  for loop in [0, loopNum):
    1. 计算本块 pOffset / copySize / currentCoreElements
    2. CopyIn: x[pOffset×GD×inner : ...] GM → UB
    3. GetUintDivMagicAndShift × 2 (innerSize, gatherSize)
    4. asc_vf_call<GatherSimt3D<NIS>>(threadNum, ...):
       每线程: yIndex → tmp = UintDiv(yIndex, m0, shift0)
                         gatherI = tmp - tmp1 × gatherSize
                         innerI = yIndex - tmp × innerSize
              idx = indices[gatherI]
              y[yIndex] = srcTensor[((tmp1-pOffset)×GD + idx)×inner + innerI]
```

## 关键技术

| 技术 | 说明 |
|------|------|
| **UB 全载 x** | `DataCopyPad` 一次性搬 GA 块到 UB，SIMT 线程从 UB 读（延迟 ~10× 低于 GM） |
| **SIMT 从 UB 读** | `__ubuf__ T* srcTensor` 传给 SIMT kernel，线程读 UB 而非 GM |
| **2D/3D 地址分解** | 2D：1 次 fast-div（innerSize）；3D：2 次 fast-div（innerSize + gatherSize） |
| **LoopCol 分块** | outerSize > 1 时按 maxGdimA 分块，每块独立全载 + SIMT gather |
| **负索引编译期分支** | `NIS` 模板 bool 参数，`if constexpr` 消除分支发散 |

## 关键代码索引

- 模板源码：[dav310/gather_full_load.template](dav310/gather_full_load.template)
- TilingData：[gather_tiling_data.template](gather_tiling_data.template) → `GatherTilingData`（复用通用结构，maxElement 字段表示 UB 可用空间）
- Host tiling：[gather_tiling.template](gather_tiling.template) → `CalFullLoadTiling()`
- 集成：[usage_guide.md](usage_guide.md) 第 5 节

## 与其他模板的关系

| 场景 | 模板 |
|------|------|
| gatherDim×inner 超 UB | [SIMT](simt_gather.md)（无 UB 装载）或 [SIMD 行拷贝](simd_row_copy.md) |
| innerSize ≥ 2048 | [SIMD 行拷贝](simd_row_copy.md)（按 slice 连续搬） |
| innerSize == 1 | [SIMD 末轴 Gather](simd_last_gather.md)（Gather 向量化） |
| innerSize 小 + GA 可全载 + Gather 可用 | [GA 全载](ga_all_load.md)（向量化 gather 更快） |
