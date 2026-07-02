/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aclnn_quant_matmul_gelu_example.cpp
 * @brief ACLNN L2 API 实现 - AIC+AIV 融合算子脚手架
 *
 *
 * L2 API 职责: 参数检查、Contiguous 处理、bias bf16→float cast。
 * L0 API 职责: 形状推导、Kernel 调度。
 */

#include "aclnn_quant_matmul_gelu_example.h"
#include "quant_matmul_gelu_example.h"

extern "C" aclnnStatus aclnnQuantMatmulGeluExampleGetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    const aclTensor* scale,
    const aclTensor* pertokenScale,
    const aclTensor* bias,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    
}

extern "C" aclnnStatus aclnnQuantMatmulGeluExample(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantMatmulGeluExample);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}