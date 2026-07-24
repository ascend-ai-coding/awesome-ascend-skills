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
 * \file common_utils.h
 * \brief Device-side constants and integer helpers for matmul recipe kernels.
 */

#ifndef UTILS_COMMON_UTILS_H
#define UTILS_COMMON_UTILS_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "std/algorithm.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#include "integral_constant.h"

// On-chip buffer capacities.
// L0A/L0B are architecture-independent (defined unconditionally by the SDK).
// L0C/L1/UB are architecture-dependent -- use AscendC::TOTAL_L0C_SIZE,
// AscendC::TOTAL_L1_SIZE, AscendC::TOTAL_UB_SIZE directly at point of use
// (inside __aicore__ class bodies or functions where they are available).
static constexpr int64_t L0A_SIZE = AscendC::TOTAL_L0A_SIZE;
static constexpr int64_t L0B_SIZE = AscendC::TOTAL_L0B_SIZE;

// Execution mode shared by SWAT schedulers, dispatch tags, and launchers.
// Blaze skill keeps only the streaming SWAT path: both A and B are staged by K
// tiles each round. Full-load variants are intentionally not provided.
constexpr uint64_t NO_FULL_LOAD_MODE = 0UL;

constexpr int MNK_M = 0;
constexpr int MNK_N = 1;
constexpr int MNK_K = 2;
constexpr int MNK_B = 3;
constexpr int MNK_M0 = 4;
constexpr int MNK_N0 = 5;

struct MatmulShape {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t b;
};

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__host_aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__host_aicore__ inline int64_t CeilAlign(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}


#endif
