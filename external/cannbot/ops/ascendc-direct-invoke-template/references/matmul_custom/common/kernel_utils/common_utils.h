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

// On-chip buffer capacities used by the kernel helper code.
static constexpr int64_t L0A_SIZE = 64 * 1024;
static constexpr int64_t L0B_SIZE = 64 * 1024;
static constexpr int64_t L0C_SIZE = 256 * 1024;
static constexpr int64_t L1_SIZE = 512 * 1024;
static constexpr int32_t BT_SIZE = 4096;

// Execution modes shared by SWAT schedulers, dispatch tags, and sample
// launchers. `NO_FULL_LOAD_MODE` streams both A and B, while
// `A_FULL_LOAD_MODE` keeps the A tile resident and streams B.
constexpr uint64_t NO_FULL_LOAD_MODE = 0UL;
constexpr uint64_t A_FULL_LOAD_MODE = 1UL;

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
