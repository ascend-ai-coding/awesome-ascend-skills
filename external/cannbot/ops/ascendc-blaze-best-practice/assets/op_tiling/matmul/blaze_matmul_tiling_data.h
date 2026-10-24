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
 * \file matmul_tiling_data.h
 * \brief Serialized tiling data passed from host launcher to kernel.
 */

#ifndef BLAZE_MATMUL_TILING_DATA_H
#define BLAZE_MATMUL_TILING_DATA_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

// host 端填写、device 端解包的 POD。普通 Grouped MatMul 复用该结构，group 元信息
// 由 grouped kernel 独立接收，不进入 tiling data。
#pragma pack(push, 8)
struct alignas(8) MatmulTilingData {
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    uint32_t mL1{0};
    uint32_t nL1{0};
    uint32_t kL1{0};
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t baseK{0};
    // `mTailCnt/nTailCnt` 的真实语义是尾块二次切分份数（tail split factor）。
    // 传给 blaze_custom scheduler 时分别对应 `mTailTile/nTailTile`；禁用该优化时固定为 1。
    uint32_t mTailCnt{1};
    uint32_t nTailCnt{1};
    uint32_t mBaseTailSplitCnt{1};
    uint32_t nBaseTailSplitCnt{1};
    uint32_t mTailMain{0};
    uint32_t nTailMain{0};
    uint32_t usedCoreNum{0};
    uint8_t l1BufferNum{0};
    uint8_t l0cDB{1};
};
#pragma pack(pop)

#endif // BLAZE_MATMUL_TILING_DATA_H
