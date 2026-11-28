# SIMT 两维模板（2D 特化）

## 解决的问题

当输入为纯 2D（batchSize==1, outerSize==1），即 `x[gatherDimSize, innerSize]`、`index[gatherSize, innerSize]`、`y[gatherSize, innerSize]` 时，地址分解无需 outer 维计算。专用 2D 模板比通用 SIMT 少 2 次 fast-div（仅需 1 次），SIMT 线程计算更快。

## 适用条件

| 条件 | 要求 |
|------|------|
| batchSize | == 1 |
| outerSize | == 1 |
| 其他 | 无额外限制（2D 兜底路径） |

> 当 innerSize × improveDtypeSize ≥ SIMD 阈值且有足够多核工作量时，优先走 [SIMD 两维](simd_row_copy.md)（向量化搬运更快）；否则走本模板。

## 执行流程

```
Init:
  xGm / indicesGm / yGm SetGlobalBuffer

Process:
  1. GetUintDivMagicAndShift(m0, shift0, innerSize)  // fast-div
  2. asc_vf_call<GatherSimt<NIS>>(dim3(threadNum), ...):
     每线程:
       yIndex = yIndexBase + threadIdx + i × blockDim
       gatherI = UintDiv(yIndex, m0, shift0)         // 1 次 fast-div
       innerI = yIndex - gatherI × innerSize          // 1 次 mul + sub
       idx = indices[gatherI]                          // GM 读
       xIndex = idx × innerSize + innerI
       y[yIndex] = x[xIndex]                           // GM 读 + GM 写
```

## 关键技术

| 技术 | 说明 |
|------|------|
| **1 次 fast-div** | 仅需 `innerSize` 的 magic/shift，比通用 SIMT（3 次 div）少 2 次 |
| **无 UB** | 直接 GM→GM，无 MTE2/MTE3 开销 |
| **SIMT 2048 线程** | `LAUNCH_BOUND(2048)` + `asc_vf_call`，自适应降到 128 |
| **负索引编译期** | `NIS` 模板 bool，`if constexpr` 消除分支 |

## 关键代码索引

- 模板源码：[dav310/gather_simt_two_dim.template](dav310/gather_simt_two_dim.template)
- TilingData：[gather_tiling_data.template](gather_tiling_data.template) → `GatherSimtTilingData`
- Host tiling：[gather_tiling.template](gather_tiling.template) → `SimtTwoDimTiling()`
- 集成：[usage_guide.md](usage_guide.md) 第 5 节

## 与通用 SIMT 的区别

| 维度 | 通用 SIMT | SIMT 两维 |
|------|----------|----------|
| 适用 | 所有维度 | batchSize==1 && outerSize==1 |
| 地址分解 | 3 次 fast-div（innerI, gatherI, outerI） | 1 次 fast-div（gatherI） |
| index 读取 | `indices[batchI × gatherSize + gatherI]` | `indices[gatherI]` |
| x 地址 | `(oOuter × G_x + idx) × inner + innerI` | `idx × innerSize + innerI` |
| 线程效率 | 较低（3 次 div） | 较高（1 次 div） |

> 通用 SIMT 在 outerSame==1 时也有快路径（innerI==innerX），但仍有 2 次 div。2D 特化模板进一步消除 outer 维计算，适合纯 2D 场景。
