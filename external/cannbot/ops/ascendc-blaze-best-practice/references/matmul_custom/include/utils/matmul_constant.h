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
 * \file matmul_constant.h
 * \brief Shared constants and tuple indices for matmul kernels.
 */
#ifndef UTILS_MATMUL_CONSTANT_H
#define UTILS_MATMUL_CONSTANT_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

// Tuple indices for block shape packing:
//  (M tile extent, N tile extent, optional M split offset, optional N split offset)
constexpr uint64_t IDX_M_TILEIDX = 0UL;
constexpr uint64_t IDX_N_TILEIDX = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDX = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDX = 3UL;

// Generic (M, N, K) tuple indices shared by host and device helpers.
constexpr uint64_t IDX_M_IDX = 0UL;
constexpr uint64_t IDX_N_IDX = 1UL;
constexpr uint64_t IDX_K_IDX = 2UL;

// MMAD accumulation-mode selectors passed via FixpipeParams / MmadParams.unitFlag.
// Use FINAL_ACCUMULATION on the last MMAD of the reduction chain and when flushing L0C.
constexpr uint32_t FINAL_ACCUMULATION = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;

// Event-ID slots reserved for the M<->MTE1 barrier (A/B L0 ping-pong).
constexpr uint16_t ZERO_FLAG = 0;
constexpr uint16_t FIRST_FLAG = 1;

// L0C and L0A/B share a 2-stage physical buffering budget in this template.
constexpr int64_t DOUBLE_BUFFER_COUNT = 2LL;

#endif // UTILS_MATMUL_CONSTANT_H
