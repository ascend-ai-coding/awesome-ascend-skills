/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

# GA 全载 Gather 模板

## 1. 模板用途

针对 **innerSize 小**且 **gatherDimSize × innerSize 可全载 UB** 场景的向量化实现。将 x 的 `[gatherDimSize, innerSize]` 二维块（GA block）整块载入 UB，用 `Gather` 按索引向量 gather，配合 `GenIndexBuf` 生成列偏移实现二维 gather。适合 Embedding lookup（vocab×hidden 小）和 TF GatherV2 末两轴 gather。

## 2. 解决的问题

| 痛点 | 标量/逐 slice 实现 | GA 全载 |
|------|-------------------|---------|
| 取数方式 | 每索引一次 GM DMA（MTE2 碎读） | x 块一次全载 UB，索引全在 UB 内 gather |
| GM 访问 | O(gatherSize) 次随机读 | O(1) 次连续读（全载）+ UB 内散点 |
| 二维 gather | 行偏移 + 列偏移逐元素算 | `GenIndexBuf` 向量化生成列偏移 + `Adds` 行偏移 |
| innerSize 小 | slice 太短 burst 浪费 | 全载后 UB 内向量化，burst 不再是瓶颈 |

## 3. 适用条件

| 条件 | 要求 |
|------|------|
| innerSize | 小（< 2048，否则用 [SIMD 行拷贝](simd_row_copy.md)） |
| gatherDimSize × innerSize | 可全载 UB（`gatherDimSize * innerSize * dtypeSize ≤ UB - indexBuf - outBuf`） |
| batchSize | == 1（单 GA 块全载；多 batch 需外层循环重载） |
| dtype | FP32/FP16/BF16/int8/int16/int32 |

> innerSize==1 用 [SIMD 末轴 Gather](simd_last_gather.md)（其一维特例，更轻量）；gatherDim×inner 超 UB 用 [SIMT](simt_gather.md) 或 [SIMD 行拷贝](simd_row_copy.md)。

## 4. 执行流程

```
Host:
  TilingArith: 判定 gatherDimSize * innerSize * dtypeSize <= UB 预算
               needCoreNum = min(coreNumAiv, CeilDiv(gatherSize, blockFactor))
               tileA0Len = VL 对齐的列 tile 长度
  <<<needCoreNum, 1>>>

Device (每核):
  Init:
    1. SetGlobalBuffer(x, indices, y)
    2. InitBuffer: xInQue (GA 块全载), indexBuf (GenIndexBuf 列偏移), yOutQue (输出 tile)
  Process:
    1. CopyIn GA 块：DataCopyPad(xGm[0] → xInQue)
       一次载入 gatherDimSize * innerSize 个元素，整块常驻 UB
    2. GenIndexBuf：向量化生成列偏移
       - Arange 生成 [0..VL) 序列
       - Muls(dst, src, innerSize)  生成行步长（每索引跨 innerSize 个元素）
       - 或 Div + Mul + Sub 分解二维坐标 (gatherI, innerCoord)
    3. for each index tile:
         a. LoadIndices: 加载 + 校验索引（负索引补偿 + 越界 clamp）
         b. Adds(idxReg, idxOriReg, rowOffset, mask)   ← 索引叠加行偏移
         c. DataCopyGatherImpl(xReg, xInAddr, idxReg, mask)  ← UB 内向量化 gather
         d. Reg::StoreAlign(yOutAddr, xReg, mask)              ← 写输出 tile
    4. CopyOut: DataCopyPad(yGm ← yOutQue)
```

## 5. 关键技术

| 技术 | 要点 |
|------|------|
| **GA block 全载** | `gatherDimSize × innerSize` 二维块一次 `DataCopyPad` 载入 UB 常驻，所有索引 gather 都在 UB 内完成，消除 O(gatherSize) 次 GM 随机读，转为 O(1) 次连续 GM 读 + UB 内散点。 |
| **GenIndexBuf 列偏移** | 用 `Reg::Arange` + `Muls`/`Div`+`Mul`+`Sub` 向量化生成二维坐标偏移（行步长 = innerSize），避免逐元素标量算地址。`DataCopy` 写回 `indexBuf` 供后续 `Adds` 叠加。 |
| **Gather + Add offset** | `Adds(idxReg, idxOriReg, rowOffset, mask)` 把行偏移叠加到索引向量，再用 `DataCopyGatherImpl(xReg, xInAddr, idxReg, mask)` 一次 gather 一个 VL，实现二维 `[gatherI, innerCoord]` gather。 |
| **多相位 index buffer** | 不同 UB 切块相位（主块/尾块）预生成多套 index buffer（`GenGatherIndex4AllPhase`），运行时按 `gIndexId_` 选取，避免尾块非对齐时重新生成。 |
| **Dtype Improve** | int8 数据 `CastType` 提升为 int16，`DIST_PACK_B16` 写出，32B burst 对齐。 |

## 6. TilingData 字段

GA 全载模板的 TilingData 字段（参考 `GatherTransposeTilingData` 模式，集成时按工程实际定义）：

| 字段 | 含义 |
|------|------|
| `usedCoreCnt` | 实际使用核数 |
| `dataTensorSize` | GA 块在 UB 占用字节（= gatherDimSize × innerSize × dtypeSize） |
| `indexTensorSize` | index buffer（GenIndexBuf）UB 占用字节 |
| `ubAxesCnt` | 参与 gather 的 UB 轴数 |
| `blkFactor` / `blkTailFactor` | 主核/尾核处理块数 |
| `inUbCutAxisFactor` / `outUbCutAxisFactor` | UB 切块因子 |

## 7. 关键代码索引

- 参考实现：`ascendc-performance-best-practices/.../conversion/templates/dav3510/transpose_with_gather.template`（`TransposeWithGather` 类）
  - `GenGatherIndex4AllPhase` / `GenIndex4OneDim/TwoDim/ThreeDim`：向量化生成列偏移
  - `GatherData`：`DataCopyGatherImpl` + `Adds` 行偏移叠加
  - `CopyDataIn` / `CopyDataOut`：DataCopyPad 跨 stride 全载/搬出
- 模板源码：[dav310/gather_ga_all_load.template](dav310/gather_ga_all_load.template)
- 集成：[usage_guide.md](usage_guide.md)

## 8. 不适用场景与替代

| 场景 | 推荐替代 |
|------|---------|
| innerSize==1 | [SIMD 末轴 Gather](simd_last_gather.md)（一维特例，更轻量） |
| innerSize 大（>= 2048） | [SIMD 行拷贝](simd_row_copy.md)（按 slice 连续搬） |
| gatherDim×inner 超 UB | [SIMT](simt_gather.md)（无 UB 装载）或 [SIMD 行拷贝](simd_row_copy.md) |
| batchSize > 1 | 外层循环重载 GA 块，或 [SIMT](simt_gather.md) |
