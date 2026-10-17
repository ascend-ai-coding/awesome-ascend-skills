/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_x_all_gather_matmul.cpp
 * \brief Public aclnn wrapper for XAllGatherMatmul (non-quant, A2/Ascend910B).
 *
 * Migrated from ops-transformer/mc2/x_all_gather_matmul/op_api/aclnn_x_all_gather_matmul.cpp, adapted to the
 * x_ops MC2 aclnn convention (see aclnn_x_mega_moe.cpp): a lean wrapper that declares the build-auto-
 * generated `aclnnInnerXAllGatherMatmul*` as extern and delegates to it.
 *
 * Per the non-quant A2 scope:
 *  - the source's IsAscend910A5() branch (which delegated to aclnnXAllGatherMatmulV2) is removed, which
 *    also drops the dependency on the mc2/x_all_gather_matmul_v2 sibling operator;
 *  - the source's heavy op_api param-check helpers (which pulled mc2/3rd matmul_util / aclnn_kernels
 *    op_error_check / IsTransposeLastTwoDims) are not carried over — x_ops MC2 wrappers omit them
 *    (correctness is enforced by the op def / tiling / kernel). The transpose flags are read from the
 *    op attributes by the tiling (is_trans_a / is_trans_b); x1 must not be transposed.
 */

#include "aclnn_x_all_gather_matmul.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build-auto-generated inner API (from x_all_gather_matmul_def.cpp via the aclnnInner pipeline).
extern aclnnStatus aclnnInnerXAllGatherMatmulGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
    const aclTensor* bias, const char* group, bool transposeX1, bool transposeX2, int64_t gatherIndex,
    int64_t commTurn, int64_t rankSize, bool isGatherOut, const aclTensor* output, const aclTensor* gatherOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerXAllGatherMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

aclnnStatus aclnnXAllGatherMatmulGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                                   const char* group, int64_t gatherIndex, int64_t commTurn,
                                                   int64_t streamMode,
                                                   const aclTensor* output, const aclTensor* gatherOut,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor) {
    (void)streamMode;  // x_ops lean wrapper does not re-validate params; tiling/kernel enforce correctness.
    // x1 does not support transpose (is_trans_a must be false); x2 transpose is resolved by tiling
    // from the is_trans_b attribute. rankSize is resolved from the group by the tiling side.
    const bool transposeX1 = false;
    const bool transposeX2 = false;
    const int64_t rankSize = 0;
    const bool isGatherOut = (gatherOut != nullptr);
    return aclnnInnerXAllGatherMatmulGetWorkspaceSize(x1, x2, bias, group, transposeX1, transposeX2, gatherIndex,
        commTurn, rankSize, isGatherOut, output, gatherOut, workspaceSize, executor);
}

aclnnStatus aclnnXAllGatherMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  aclrtStream stream) {
    return aclnnInnerXAllGatherMatmul(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
