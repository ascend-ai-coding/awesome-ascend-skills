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
// AIV Unified Buffer 容量（DAV_3510）。融合算子 Epilogue 在 AIV 侧使用 UB 暂存
// matmul L0C 结果（int32）+ scale/pertoken + bf16 输出。
static constexpr int64_t UB_SIZE = 248 * 1024;

// Execution mode tag shared by SWAT scheduler / dispatch policy.
// 本脚手架只实现 NO_FULL_LOAD_MODE（A、B 均从 GM 流式搬入 L1）。
// 如需 A/B 常驻 L1 的 full-load 变体，请参考 blaze-best-practice 自行扩展。
constexpr uint64_t NO_FULL_LOAD_MODE = 0UL;

constexpr int MNK_M = 0;
constexpr int MNK_N = 1;
constexpr int MNK_K = 2;
constexpr int MNK_B = 3;

struct MatmulShape {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t b;
};

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


#endif
