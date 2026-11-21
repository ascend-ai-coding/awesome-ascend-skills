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

# SIMT 标量并行 Gather 模板

## 1. 模板用途

解决 Gather 算子在 Vector 核上的**标量瓶颈**：单线程 `GetValue`/`SetValue` 逐元素搬运，被 ScalarBound（标量发射速率上限）卡死，输出越大越慢。SIMT 模板用 2048 线程并行做散点 GM 读写，把标量瓶颈转化为带宽利用，是**所有 Gather 场景的最通用高性能路径**。

## 2. 解决的问题

| 痛点 | 单线程标量实现 | SIMT 模板 |
|------|--------------|----------|
| 取数方式 | `GetValue(idx)` 逐元素标量读 | 2048 线程并行直接 GM 读 |
| 发射速率 | 受 ScalarBound 限制（~每周期 1 元素） | 线程级并行隐藏延迟 |
| 地址计算 | `div`/`mod` 指令 | fast-division（乘法 + 移位） |
| UB 占用 | 需 UB 中转 | 无 UB 中转，直接 GM→GM |
| 适用范围 | 仅小张量可用 | 全场景（含散点读 torch.gather dim=0） |

## 3. 适用条件

- **全场景适用**，无 shape / 连续性假设。
- **尤其适合**散点读：torch.gather（index 与输出同形，每位置独立索引）、torch.gather dim=0、TF GatherV2 低维 index。
- 无 UB 预算要求（不装载 x 到 UB）。
- dtype：FP32/FP16/BF16/int8/int16/int32/int64 均可（位拷贝）。
- INDEX_SIZE_T：`uint32_t`（元素总数 < 4G）或 `uint64_t`（超大张量）。

> 当 innerSize==1 且 gatherDimSize 可全载 UB 时，[SIMD 末轴 Gather](simd_last_gather.md) 通常更快（向量化 gather 比标量并行带宽利用率更高）；不命中其门控时回退到本模板。

## 4. 执行流程

```
Host:
  TilingArith: 计算 needCoreNum = min(coreNumAiv, CeilDiv(outputTotal, perCoreElements))
               threadNum = 2048 (uint32) / 1024 (uint64)
               perCoreElements / lastCoreElements 切分
  <<<needCoreNum, threadNum>>>

Device (每核):
  Init:
    1. SetGlobalBuffer(x, indices, y)
    2. GetBlockIdx() 判定主核/尾核，取 currentCoreElements
    3. GetUintDivMagicAndShift(m0Outer, shift0Outer, gatherSize*innerSize)  ← fast-div 预计算
       GetUintDivMagicAndShift(m0Inner, shift0Inner, innerSize)
  Process:
    asc_vf_call<GatherSimtVF<NIS, OUTER_SAME>>(dim3(threadNum), ...)
      └─ 每线程循环（stride = blockDim.x）:
           yIndex = yIndexBase + threadIdx + k*blockDim
           oOuter   = Simt::UintDiv(yIndex, m0Outer, shift0Outer)   ← 替代 yIndex/(gatherSize*innerSize)
           rem      = yIndex - oOuter * (gatherSize*innerSize)
           gatherI  = Simt::UintDiv(rem, m0Inner, shift0Inner)       ← 替代 rem/innerSize
           innerCoord = rem - gatherI * innerSize
           idx = (OUTER_SAME) ? indices[oOuter*gatherSize + gatherI] : indices[yIndex]
           [NIS] if (idx < 0) idx += gatherDimSize
           xIndex = oOuter*gatherDimSize*innerSize + idx*innerSize + innerCoord
           y[yIndex] = x[xIndex]    ← 直接 GM 写
```

## 5. 关键技术

| 技术 | 要点 |
|------|------|
| **SIMT 2048 线程并行** | `__simt_vf__` + `LAUNCH_BOUND(2048)` + `asc_vf_call`；`threadIdx.x`/`blockDim.x` 控制线程索引与步长。搬运类算子带宽受限，开满线程隐藏 GM 延迟。 |
| **Fast-Division** | `GetUintDivMagicAndShift(m0, shift, divisor)` 预计算 magic/shift，`Simt::UintDiv(a, m0, shift)` 用乘法+移位替代 `div`，余数用 `a - quot*div` 计算。两级 fast-div 分解三维坐标。 |
| **Direct GM Access** | 线程直接 `y[yIndex] = x[xIndex]`，无 UB 中转、无 MTE2/MTE3 搬运开销。UB 压力最小。 |
| **编译期分支消除** | `NIS`（负索引）与 `OUTER_SAME`（地址路径）均为模板 bool，`if constexpr` 提升到编译期，消除 SIMT VF 内 if-else 分支发散。 |
| **Dtype Improve** | 小 dtype（int8/int16）位拷贝直通，无精度损失；INDEX_SIZE_T 按元素总数选 32/64 位。 |

## 6. 线程数选择

| INDEX_SIZE_T | 推荐 threadNum | 每线程寄存器 | 说明 |
|-------------|---------------|------------|------|
| `uint32_t` | 2048 | 16 | 两级 fast-div + 有限坐标变量，寄存器压力中低 |
| `uint64_t` | 1024 | 32 | 64 位运算寄存器压力翻倍，2048 会 spill |

> 寄存器与线程数关系：1025~2048 线程→16 寄存器/线程；513~1024→32。调优步骤：uint32 起步 2048，uint64 起步 1024，逐步上调验证无 spill。

## 7. 性能

Ascend950（dav-3510）实测，相对单线程标量 GetValue/SetValue 基线：

| Case | 数据量 | 基线 | SIMT | 加速比 |
|------|--------|------|------|--------|
| torch.gather P4 | 33.5M int32 | 128 ms | 3.6 ms | **35.7×** |
| torch.gather P6 | 67.1M int8 | 231 ms | 5.3 ms | **43.2×** |

加速比随输出元素数增长而提升（线程并行度被填满后趋稳）。

## 8. 地址模型与两条路径

三维统一模型 `x[outerSize, gatherDimSize, innerSize]`，`yIndex → (oOuter, gatherI, innerCoord)`：

- **OUTER_SAME=true 快路径**：index 沿 gather 轴低维（`index.shape[i]==x.shape[i] for i!=dim`，即 TF GatherV2 常见形态），`innerI==innerX`，`idx = index[oOuter*gatherSize + gatherI]`，一个索引取整个 slice。地址公式简化，少一次索引读取。
- **OUTER_SAME=false 通用路径**：index 与输出同形（torch.gather），`idx = index[yIndex]`，每个输出位置有独立索引。

路径由 Host 侧据 `batchSize`/index rank 判定，传入模板编译期分发。

## 9. 关键代码索引

- 模板源码：[dav310/gather_simt.template](dav310/gather_simt.template)
  - `GatherSimtVF`（`__simt_vf__` 核心）：地址分解 + fast-div + GM 读写
  - `Init`：`GetUintDivMagicAndShift` 预计算
  - `Process`：`asc_vf_call` 编译期分发（NIS × OUTER_SAME 四实例）
- TilingData：[gather_tiling_data.template](gather_tiling_data.template) → `GatherTilingData`
- 集成：[usage_guide.md](usage_guide.md) 第 5 节

## 10. 独立 Kernel 入口示例

```cpp
#include "gather_tiling_data.h"
#include "dav310/gather_simt.h"

using namespace gatherv2;

extern "C" __global__ __aicore__ __vector__ void
gather_simt_fp32_i32(GM_ADDR x, GM_ADDR indices, GM_ADDR y, GatherTilingData tiling)
{
    GatherSimt<float, int32_t, uint32_t> op;
    op.Init(x, indices, y, &tiling);
    op.Process();
}
```

> 按值传 `GatherTilingData` 需入口标注 `__vector__`（见 [usage_guide.md](usage_guide.md) 问题 2）；亦可改传指针 + `CopyTiling`。

## 11. 不适用场景与替代

| 场景 | 推荐替代 |
|------|---------|
| innerSize==1 + gatherDimSize 可全载 UB | [SIMD 末轴 Gather](simd_last_gather.md)（向量化 gather 带宽更高） |
| innerSize 大 + 有连续 slice | [SIMD 行拷贝](simd_row_copy.md)（DataCopyPad 批量搬） |
| innerSize 小 + gatherDim×inner 可全载 UB | [GA 全载](ga_all_load.md) |
