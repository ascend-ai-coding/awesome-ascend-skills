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
 * \brief Dispatch policy tags for matmul kernels.
 */
#ifndef DISPATCH_POLICY_H
#define DISPATCH_POLICY_H

#include <cstdint>

// ============================================================================
// [MODIFY] 新算子时，可自定义 dispatch policy 标签。该标签同时出现在：
//   - BlockMmad 的 SFINAE 约束（matmul_block_mmad.h）
//   - launcher 模板参数组装（matmul_custom.cpp）
// 默认只提供 non-full-load 一条流水路径；需要 A-full-load 等变种时可再新增标签。
// ============================================================================

/**
 * @brief Matmul dispatch tag for the SWAT streaming (non-full-load) path.
 * @tparam FULL_LOAD_MODE_ Selects the streaming variant (0 = stream both A and B).
 */
template <uint64_t FULL_LOAD_MODE_>
struct MatmulMultiBlockPolicy {
    static constexpr uint64_t fullLoadMode = FULL_LOAD_MODE_;
};

#endif
