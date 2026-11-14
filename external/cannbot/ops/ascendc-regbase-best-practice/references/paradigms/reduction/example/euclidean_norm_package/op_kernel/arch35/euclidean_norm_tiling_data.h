/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file euclidean_norm_tiling_data.h
 * \brief EuclideanNorm 算子 TilingData 定义
 *
 * 套用 reduction 范式 ReduceGenericTilingData 通用布局，
 * 无 EuclideanNorm 专属字段（sqrt 是 in-register、不需要传系数；与 mean 需要 invRTotal 不同）。
 */
#ifndef OPS_NORM_EUCLIDEAN_NORM_TILING_DATA_H_
#define OPS_NORM_EUCLIDEAN_NORM_TILING_DATA_H_

#include <cstdint>

constexpr int32_t MAX_PATTERN_RANK = 9;

struct EuclideanNormTilingData {
    // ── A/R 规整 pattern 与定长数组（未用槽 axisShape=1 / axisStride=0）──
    int32_t axisNum = 0;
    int64_t axisShape[MAX_PATTERN_RANK] = {0};
    int64_t axisStride[MAX_PATTERN_RANK] = {0};

    // ── A 方向多核参数（大小核均分）──
    int64_t aLoopCntTotal = 0;
    int64_t aSplitChunkCnt = 0;
    int64_t aBigCoreLoopCnt = 0;
    int64_t aSmallCoreLoopCnt = 0;
    int32_t aBigCoreCnt = 0;
    int32_t usedCoreNum = 0;

    // ── UB 切分参数（valid rUbFactor / padded rUbFactorAlign 双字段）──
    int32_t aSplitIdx = 0;
    int32_t rSplitIdx = 0;
    int64_t aUbFactor = 0;
    int64_t rUbFactor = 0;
    int64_t rUbFactorAlign = 0;
    int64_t innerAProdAlign = 0;
    int64_t innerRProdAlign = 0;

    // ── R 方向总迭代数（kernel 二分树参数来源）──
    int64_t rLoopCntTotal = 0;

    // ── UB buffer 尺寸（pre 三份同尺寸 / out 独立 / cache 固定 16KB）──
    int64_t preBufSize = 0;
    int64_t postBufSize = 0;
    int64_t cacheBufUbSize = 0;

    // ── Group 2D 分核参数（R 分组数 = workspace 行数，仅 Group 非零）──
    int64_t rGroupCnt = 0;
};

// Empty 模板专用 TilingData（EMPTY_A 全零 / EMPTY_R 7 字段输出填充切分）。
struct EuclideanNormEmptyTilingData {
    // ── 输出区间切分（按 a 元素数均分）──
    int32_t usedCoreNum = 0;
    int64_t aTotal = 0;
    int64_t aUbFactor = 0;
    int32_t aBigCoreCnt = 0;
    int64_t aBigCoreLoopCnt = 0;
    int64_t aSmallCoreLoopCnt = 0;

    // ── 唯一 UB buffer（固化值输出）──
    int64_t postBufSize = 0;
};

#endif // OPS_NORM_EUCLIDEAN_NORM_TILING_DATA_H_
