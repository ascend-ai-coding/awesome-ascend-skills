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
 * \file dispatch_policy.h
 * \brief Dispatch policy tags used by the quantized matmul recipe kernels.
 */
#pragma once

#include <cstdint>

// Tag for kernels that split work along K while still carrying dedicated
// scaleA and scaleB tensors through the pipeline.
struct KernelMultiBlockOnKAxisWithScale {};

/**
 * @brief Dispatch tag for MX quantized matmul kernels that use the SWAT
 *        scheduling family.
 * @tparam FULL_LOAD_MODE_ Selects the execution mode: streaming or A-full-load.
 * @tparam STAGES_ Configures how many L1 pipeline stages the block MMAD
 *         implementation should provision for this dispatch path.
 */
template <uint64_t FULL_LOAD_MODE_, uint64_t STAGES_>
struct QuantMatmulMxMultiBlockWithSwat {
    using ScheduleType = KernelMultiBlockOnKAxisWithScale;
    static constexpr uint64_t fullLoadMode = FULL_LOAD_MODE_;
    static constexpr uint64_t stages = STAGES_;
};

/**
 * @brief Dispatch tag for MatmulA16W16 matmul kernels that use the SWAT
 *        scheduling family.
 * @tparam SingleCoreShape Placeholder for the per-core tile shape recorded
 *         in the dispatch traits.
 * @tparam FULL_LOAD_MODE_ Selects the SWAT variant: streaming or A-full-load or B-full-load.
 */
template <uint64_t FULL_LOAD_MODE_>
struct MatmulA16W16MultiBlockWithSwat {
    static constexpr uint64_t fullLoadMode = FULL_LOAD_MODE_;
};
