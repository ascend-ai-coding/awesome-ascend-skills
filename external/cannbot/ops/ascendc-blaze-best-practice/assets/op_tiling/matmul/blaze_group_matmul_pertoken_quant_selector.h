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

#ifndef BLAZE_GROUP_MATMUL_PERTOKEN_QUANT_SELECTOR_H
#define BLAZE_GROUP_MATMUL_PERTOKEN_QUANT_SELECTOR_H

#include <cstdint>
#if !defined(__CCE_AICORE__) || defined(__ASC_NPU_HOST__)
#include <cmath>
#include <limits>
#include <stdexcept>
#endif

namespace BlazeGroupMatmulPertokenQuant {

constexpr uint64_t UB_ALIGN_BYTES = 256;
constexpr uint64_t SCALE_STAGING_BYTES = UB_ALIGN_BYTES;
constexpr uint64_t CHUNK_COL_ALIGNMENT = 32;

enum class ScaleClampMode : uint8_t
{
    BEFORE_DIV,
    AFTER_DIV,
};

struct UbLayout {
    uint64_t inputBytes;
    uint64_t outputBytes;
    uint64_t scaleOffset;
    uint64_t requiredBytes;
};

#if defined(__CCE_AICORE__) && !defined(__ASC_NPU_HOST__)
#define BLAZE_PQ_SELECTOR_SHARED_FN __aicore__ inline
#else
#define BLAZE_PQ_SELECTOR_SHARED_FN inline constexpr
#endif

BLAZE_PQ_SELECTOR_SHARED_FN uint64_t AlignBytes(uint64_t bytes)
{
    return (bytes + UB_ALIGN_BYTES - 1U) / UB_ALIGN_BYTES * UB_ALIGN_BYTES;
}

BLAZE_PQ_SELECTOR_SHARED_FN UbLayout MakeUbLayout(uint64_t cols)
{
    const uint64_t inputBytes = AlignBytes(cols * sizeof(float));
    const uint64_t outputBytes = AlignBytes(cols * sizeof(int8_t));
    return {inputBytes, outputBytes, inputBytes + outputBytes, inputBytes + outputBytes + SCALE_STAGING_BYTES};
}

BLAZE_PQ_SELECTOR_SHARED_FN bool FitsUb(uint64_t cols, uint64_t availableUbBytes)
{
    return cols > 0 && cols <= (UINT64_MAX / sizeof(float)) && MakeUbLayout(cols).requiredBytes <= availableUbBytes;
}

#if !defined(__CCE_AICORE__) || defined(__ASC_NPU_HOST__)
enum class KernelVariant : uint8_t
{
    SINGLE_PASS,
    TWO_PASS_CHUNKED,
};

struct Selection {
    KernelVariant variant;
    uint32_t chunkCols;
};

inline uint64_t MaxFittingCols(uint64_t logicalWidth, uint64_t availableUbBytes)
{
    uint64_t low = 1;
    uint64_t high = logicalWidth;
    uint64_t best = 0;
    while (low <= high) {
        const uint64_t mid = low + (high - low) / 2;
        if (FitsUb(mid, availableUbBytes)) {
            best = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return best;
}

inline Selection Select(uint64_t logicalWidth, uint64_t availableUbBytes, float scaleMin)
{
    if (logicalWidth == 0 || logicalWidth > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("per-token quant logical width must fit uint32_t and be non-zero");
    }
    if (!std::isfinite(scaleMin) || scaleMin < 0.0f) {
        throw std::invalid_argument("per-token quant scale minimum must be finite and non-negative");
    }
    if (FitsUb(logicalWidth, availableUbBytes)) {
        return {KernelVariant::SINGLE_PASS, static_cast<uint32_t>(logicalWidth)};
    }

    uint64_t chunkCols = MaxFittingCols(logicalWidth, availableUbBytes);
    chunkCols = chunkCols / CHUNK_COL_ALIGNMENT * CHUNK_COL_ALIGNMENT;
    if (chunkCols == 0 || !FitsUb(chunkCols, availableUbBytes)) {
        throw std::invalid_argument("UB capacity cannot hold an aligned two-pass quant chunk");
    }
    return {KernelVariant::TWO_PASS_CHUNKED, static_cast<uint32_t>(chunkCols)};
}
#endif

#undef BLAZE_PQ_SELECTOR_SHARED_FN

} // namespace BlazeGroupMatmulPertokenQuant

#endif // BLAZE_GROUP_MATMUL_PERTOKEN_QUANT_SELECTOR_H
