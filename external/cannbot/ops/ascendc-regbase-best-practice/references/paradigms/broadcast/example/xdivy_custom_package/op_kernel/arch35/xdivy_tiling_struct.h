/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Xdivy TilingData — 按 rank 模板化，host tiling 与 device kernel 共享。
// 结构必须可平凡拷贝（纯 int64_t 定长字段），经 GM 字节流传给 kernel
#ifndef XDIVY_TILING_STRUCT_H_
#define XDIVY_TILING_STRUCT_H_

#include <cstdint>

// === 算子特定常量 — 新算子必须修改 ===
constexpr int64_t MAX_INPUT_SLOTS = 2;  // Xdivy 输入: x1, x2
constexpr int64_t MAX_OUTPUT_SLOTS = 1; // Xdivy 输出: y

// 防御性初值（NSDMI）：计数/块数类字段默认 1（单块、单核退化配置），索引默认 0，
// 避免 Host 侧异常路径（如空 shape 导致 FindSplitAxis 未写入）时 Kernel 读到未定义值
// FindSplitAxis 输出: UB 切分轴 ubSplitIdx / 主块元素数 ubFactor / 块数 ubOuter / 尾块 ubTail
struct SplitResult {
    int64_t ubSplitIdx = 0; // UB 切分轴
    int64_t ubFactor = 1;   // UB 主块元素数
    int64_t ubOuter = 1;    // UB 块数（外层次数）
    int64_t ubTail = 1;     // UB 尾块元素数
};

// MultiCoreSplit 输出: 实际核数 usedCoreNum / 总 tile 数 / 每核主块数 mainTiles / 主块核数 mainCoreNum
struct MultiCoreResult {
    int64_t usedCoreNum = 1; // 实际使用的核数
    int64_t totalTiles = 1;  // 总 tile 数
    int64_t mainTiles = 1;   // 每核主块数 = CeilDiv(totalTiles, usedCoreNum)
    int64_t mainCoreNum = 1; // 处理 mainTiles 块的核数，其余核处理 mainTiles - 1 块
};

// TilingData 主体: RANK=4 (rank 1..4) 或 8 (rank 5..8)
// shape/stride 数组按 R 维右移存放，前补 shape=1 / stride=0
template <int64_t RANK>
struct XdivyTilingData {
    SplitResult split;         // 初值见 SplitResult
    MultiCoreResult multicore; // 初值见 MultiCoreResult
    int64_t rank = 1;          // 实际 rank (1~8)
    int64_t perBufBytes = 0;   // UB/P 向下对齐 32B，kernel 用此初始化 TBuf
    int64_t maxBroShape[RANK] = {0};
    int64_t numInputs = 0;
    int64_t numOutputs = 0;
    int64_t inputShapes[MAX_INPUT_SLOTS][RANK] = {0};
    int64_t inputStrides[MAX_INPUT_SLOTS][RANK] = {0};
    int64_t outputShapes[MAX_OUTPUT_SLOTS][RANK] = {0};
    int64_t outputStrides[MAX_OUTPUT_SLOTS][RANK] = {0};
};

#endif // XDIVY_TILING_STRUCT_H_
