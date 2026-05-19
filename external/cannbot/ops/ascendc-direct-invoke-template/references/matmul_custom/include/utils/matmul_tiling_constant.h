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
 * \file matmul_tiling_constant.h
 * \brief Host-side constants used by the matmul tiling engine.
 */

#ifndef UTILS_MATMUL_TILING_CONSTANT_H
#define UTILS_MATMUL_TILING_CONSTANT_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

// [MODIFY] Matmul tiling 常量。切换数据类型时，调整 DATA_SIZE_FP16 为实际输入的字节数：
//   bf16/fp16 = 2，fp8 = 1，fp4x2 packed = 1 (2 个元素共享 1 字节)。
constexpr uint64_t BASIC_BLOCK_SIZE_16  = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_64  = 64UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_SIZE_512 = 512UL;
constexpr uint64_t BLOCK_BYTE_SIZE      = 32UL;
constexpr uint64_t DATA_SIZE_FP16       = 2UL;
constexpr uint64_t DATA_SIZE_FP32       = 4UL;
constexpr uint64_t DB_SIZE              = 2UL;
constexpr uint64_t NUM_TWO              = 2UL;
constexpr uint64_t BIAS_TABLE_NUM       = 0UL;
constexpr uint64_t WINDOW_LEN           = 4UL;

#endif // UTILS_MATMUL_TILING_CONSTANT_H
