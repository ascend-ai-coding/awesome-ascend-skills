/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

# SIMD 行拷贝 Gather 模板

## 1. 模板用途

针对 **innerSize 较大**（gather 轴之后有连续维度）场景的高效实现。每个索引对应 x 中一段 `innerSize` 个连续元素（一个 slice），用 `DataCopyPad` 跨 stride DMA 批量搬运，替代逐元素标量读。适合 TF GatherV2（一个索引取整个 slice）和 Embedding lookup（hidden_size 大）。

## 2. 解决的问题

| 痛点 | 标量/SIMT 实现 | SIMD 行拷贝 |
|------|---------------|------------|
| 取数方式 | 逐元素散点读 | `DataCopyPad` 按 slice 连续搬 innerSize 个元素 |
| burst 效率 | 单元素 32B burst 浪费 | 整 slice 连续，burst 满载 |
| 索引重复读 | 每元素读一次 index | index 滑动窗口缓存，复用 |
| innerSize 利用 | innerSize 越大标量越慢 | innerSize 越大 DMA 效率越高 |

## 3. 适用条件

| 条件 | 要求 |
|------|------|
| innerSize | >= SIMD 阈值（建议 >= 8 元素或 32B，使 DataCopyPad burst 满载） |
| 工作量 | `batchSize * outerSize * gatherSize >= coreNum / 2`（保证多核并行度） |
| dtype | FP32/FP16/BF16/int8/int16/int32（int8 可提升 int16） |
| 索引重复 | 相邻索引指向相近 x 行时，缓存收益最大 |

> innerSize==1 用 [SIMD 末轴 Gather](simd_last_gather.md)；innerSize 小且 gatherDim×inner 可全载 UB 用 [GA 全载](ga_all_load.md)；散点读无连续性用 [SIMT](simt_gather.md)。

## 4. 执行流程

```
Host:
  TilingArith: needCoreNum = min(coreNumAiv, CeilDiv(gatherSize, blockFactor))
               onceMaxRows = tokenUB / colsAlign   (UB 内一次容纳的输出行数)
               onceIndices = index tile 大小（索引缓存窗口）
  <<<needCoreNum, 1>>>

Device (每核):
  Init:
    1. SetGlobalBuffer(x, indices, y)
    2. InitBuffer: inQueue (TQueBind, 双 buffer), indexBuf (索引缓存窗口)
    3. CalculateCoreRange: [yStart, yEnd) 按 blockFactor 切分
  Process:
    colsAlign = CeilAlign(innerSize * dtypeSize, ubBlockSize)
    if colsAlign <= tokenUB:    NoSplitColProcess (一行整 slice 放得下 UB)
    else:                       SplitColProcess   (一行 slice 需分多次搬)
    ── NoSplitColProcess ──
    for i in loopNum (按 onceMaxRows 分块):
      rows = min(onceMaxRows, currentCoreElements - i*onceMaxRows)
      for j in rows:
        yIdx = yStart + i*onceMaxRows + j
        index = GetIndex(yIdx, indiceEndIdx)        ← 滑动窗口缓存
        xIndex = index * innerSize
        if xIndex in {preIdx0, preIdx1, preIdx2}:   ← 3-deep LRU 命中
          Copy(xLocal[j] ← xLocal[backStep])         UB 内拷贝，免 GM 读
        else:
          CopyIn(xLocal[j*colsAlign] ← xGm[xIndex*dtypeSize])   DataCopyPad 连续搬 innerSize
          更新 preIdx0/1/2 + backOffset0/1/2
      CopyOut(yGm[yOffset] ← yLocal)
    ── SplitColProcess ──
    for i in [yStart, yEnd):
      index = GetIndex(i, indiceEndIdx)
      for j in loopSize (inner 方向分块到 tokenUB):
        cols = min(tokenUB, innerSize - j*tokenUB)
        CopyIn(xLocal[0] ← xGm[(xIndex + j*tokenUB)*dtypeSize])
        CopyOut(yGm[(yIndex + j*tokenUB)*dtypeSize] ← yLocal)
```

## 5. 关键技术

| 技术 | 要点 |
|------|------|
| **DataCopyPad 跨 stride DMA** | `DataCopyExtParams{blockCount, blockLen, srcStride, dstStride}` 一次搬多个 slice，`blockLen = innerSize * dtypeSize` 连续段，burst 满载 32B。`DataCopyPadExtParams` 处理非对齐尾块填充。 |
| **索引滑动窗口缓存** | `GetIndex` 把 index tile 驻留 UB（`onceIndices` 大小窗口），滑动刷新；命中窗口内索引免 GM 读。配合 3-deep LRU（`preIdx0/1/2` + `backOffset0/1/2`）缓存最近 3 个 x 行，相邻索引指向同/近行时 UB 内 `Copy` 免 GM 读。 |
| **NoSplit / Split 双路** | `colsAlign`（一行 slice 对齐后字节数）≤ `tokenUB` 时整行一次搬（NoSplitColProcess，DMA 次数最少）；超 `tokenUB` 时按 `tokenUB` 分块搬（SplitColProcess，inner 方向切）。 |
| **TQueBind 双 buffer** | `TQueBind<VECIN, VECOUT, 2>` 输入输出共用队列双缓冲，MTE2 搬入与 MTE3 搬出流水重叠。 |
| **Dtype Improve** | int8 数据按 `oneTokenBtypeSize`（提升后字节）搬运，innerSize×dtypeSize 32B 对齐时提升 dtype 减少搬运次数。 |

## 6. TilingData 字段

行拷贝模板的 TilingData 字段（参考 `IndexCopyComputeRMTilingData` 模式，集成时按工程实际定义）：

| 字段 | 含义 |
|------|------|
| `needCoreNum` / `tailCoreNum` | 实际核数 / 尾核数 |
| `coreCalcNum` | 主核处理行数 |
| `tokenUB` | UB 内单 token 块字节预算 |
| `oneTokenBtypeSize` | 单元素提升后字节（dtype improve） |
| `onceUbTokenNums` | innerSize（一行元素数） |
| `onceIndices` | 索引缓存窗口大小 |
| `onceIndicesTokenNums` | 每核 SIMT 线程数（混合路径） |

## 7. 关键代码索引

- 参考实现：`ops-transformer/.../arch35/gather_v2_simd_two_dim.h`（`Gatherv2SimdTwoDim` 类）
  - `GetIndex`：索引滑动窗口缓存
  - `NoSplitColProcess` / `SplitColProcess`：双路 DMA
  - `CopyIn` / `CopyOut`：DataCopyPad 跨 stride 搬运
- 模板源码：[dav310/gather_simd_row_copy.template](dav310/gather_simd_row_copy.template)
- 集成：[usage_guide.md](usage_guide.md)

## 8. 不适用场景与替代

| 场景 | 推荐替代 |
|------|---------|
| innerSize==1 | [SIMD 末轴 Gather](simd_last_gather.md) |
| innerSize 小 + gatherDim×inner 可全载 UB | [GA 全载](ga_all_load.md) |
| 纯散点读（torch.gather dim=0，无连续 slice） | [SIMT](simt_gather.md) |
