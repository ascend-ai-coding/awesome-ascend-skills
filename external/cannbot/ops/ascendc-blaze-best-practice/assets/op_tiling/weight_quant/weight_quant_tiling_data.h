/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file weight_quant_tiling_data.h
 * \brief Serialized tiling data for weight-quant matmul (host → kernel).
 *
 * Extends MatmulTilingData with AIV-specific UB tiling fields (nUbSize,
 * kUbSize) and weight-quant semantic flags (transB, hasOffset, hasBias).
 * The SWAT load-balancing fields (tail split, edge merge) are inherited
 * from the base MatmulTilingData layout for scheduler compatibility.
 */

#ifndef WEIGHT_QUANT_TILING_DATA_H
#define WEIGHT_QUANT_TILING_DATA_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#pragma pack(push, 8)
struct alignas(8) WeightQuantMatmulTilingData {
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    uint32_t mL1{0};
    uint32_t nL1{0};
    uint32_t kL1{0};
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t baseK{0};
    uint32_t mTailCnt{1};
    uint32_t nTailCnt{1};
    uint32_t mBaseTailSplitCnt{1};
    uint32_t nBaseTailSplitCnt{1};
    uint32_t mTailMain{0};
    uint32_t nTailMain{0};
    uint32_t usedCoreNum{0};
    uint8_t l1BufferNum{2};
    uint8_t l0cDB{1};

    uint32_t nUbSize{0};
    uint32_t kUbSize{0};
    uint8_t transB{0};
    uint8_t hasOffset{0};
    uint8_t hasBias{0};
};
#pragma pack(pop)

#endif // WEIGHT_QUANT_TILING_DATA_H
