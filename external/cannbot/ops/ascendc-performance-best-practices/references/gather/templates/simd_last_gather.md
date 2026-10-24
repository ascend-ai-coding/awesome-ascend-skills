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

# SIMD 末轴 Gather 模板

## 适用条件

`innerSize == 1`、`batchSize == 1`，且 x 的一行及工作 buffer 可装入平台实际 UB。
数据支持 FP32/FP16/BF16/int8/int16/int32；索引支持 int8/int16/int32/int64。
超出条件时由 Host 路由到 [SIMT 模板](simt_gather.md)。

## Host 与 Device 的共同约定

使用 [gather_tiling.template](gather_tiling.template) 中的 `LastGatherTiling`，
以及 [gather_tiling_data.template](gather_tiling_data.template) 中的 `GatherLastTilingData`。
UB 预算和 AIV 核数由调用方查询平台后传入。

| 字段 | 含义 |
|------|------|
| `inputUbSize` / `outUbSize` / `indiceUbSize` / `indiceCastUbSize` | 单个 buffer 的字节预算，直接传给 `InitBuffer` |
| `inputNum` / `indicesNum` | Host 预留的输入/索引 buffer 数量 |
| `ubCols` / `ubRows` | Host 输出 tile 的列数/行数 |
| `blockFactor` / `tailBlockFactor` | 常规/尾部行分组负责的 outer 行数 |
| `gFactor` | 每个列分组负责的索引数 |
| `coreInCols` | 每个行分组内的列分组数量 |
| `needCoreNum` | 行分组数乘以 `coreInCols` |

Device 按 `blockIdx / coreInCols` 和 `blockIdx % coreInCols` 分别定位行、列分组。
每行从 `x[row * gatherDimSize]` 搬入，结果写到 `y[row * gatherSize + col]`。
当前模板逐行驻留 x，并以一个 B32 索引寄存器的容量细分 Host 列 tile；分组尾部单独限制有效元素数。

## 执行流程

1. 按字节预算初始化输入队列、双 buffer 输出队列、原始索引与转换 scratch。
2. 将当前 outer 行全载入 UB，用队列同步 MTE2 与 V。
3. 滑动加载当前列分组的索引。I8/I16 先转换为 I32；I64 使用双寄存器载入，校验完整值后取低 32 位。
4. 在 VF 内补偿合法负索引；补偿后仍越界的索引置为 0，读取当前 x 行的首元素。
5. B8/B16 数据将索引 `Reg::Pack` 为 U16；B32 数据使用 U32 索引。
   `Reg::Gather` 按元素索引取数，`Reg::StoreAlign` 写输出 UB。
6. `DataCopyPad` 仅写有效输出字节；索引 scratch 下次复用前等待 V 完成读取。

I64 的 `RegTraitNumTwo` 以 B32 predicate 表示各 lane。必须在缩窄前检查整个 I64 值，
避免 `2^32 + k` 等越界索引被误认为合法的 `k`。
int8 gather 结果位于 B16 寄存器中，输出使用 `DIST_PACK_B16` 恢复字节布局。

## 集成示例

```cpp
#include "gather_tiling_data.h"
#include "dav310/gather_simd_last_gather.h"

extern "C" __global__ __aicore__ __vector__ void
gather_simd_last_fp32_i32(GM_ADDR x, GM_ADDR indices, GM_ADDR y, GatherLastTilingData tiling)
{
    AscendC::TPipe pipe;
    gatherv2::GatherSimdLastGather<float, int32_t, true> op(&pipe);
    op.Init(x, indices, y, &tiling);
    op.Process();
}
```

源码：[dav310/gather_simd_last_gather.template](dav310/gather_simd_last_gather.template)。
非末轴可使用 [GA 全载](ga_all_load.md) 或 [SIMD 行拷贝](simd_row_copy.md)。
