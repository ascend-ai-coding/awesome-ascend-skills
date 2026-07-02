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
 * @file aclnn_quant_matmul_gelu_example.h
 * @brief ACLNN L2 API 接口声明 - AIC+AIV 融合算子脚手架
 *
 * [MODIFY] 本文件是 L2 API 声明的骨架。开发新融合算子时：
 *   1. 修改函数名 aclnnQuantMatmulGeluExample → aclnn{您的算子名}
 *   2. 修改参数列表（输入/输出张量数量和类型）
 *   3. 确保 GetWorkspaceSize + Execute 两段式设计不变
 *
 * ACLNN 接口采用两段式设计：
 * - GetWorkspaceSize: 计算 workspace 大小，创建执行器
 * - aclnn{Op}: 执行计算
 *
 * 当前示例：out = gelu( (x1 @ x2^T) * scale[n] * pertoken_scale[m] + bias[n] )
 *   - x1[m, k]            int8
 *   - x2[n, k]            int8     (转置语义)
 *   - scale[n]            float32  (perchannel)
 *   - pertoken_scale[m]   float    (pertoken)
 *   - bias[n]             bfloat16 (perchannel, L2 层 cast 为 float)
 *   - out[m, n]           bfloat16
 */

#ifndef ACLNN_QUANT_MATMUL_GELU_EXAMPLE_H_
#define ACLNN_QUANT_MATMUL_GELU_EXAMPLE_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnQuantMatmulGeluExampleGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *scale,
    const aclTensor *pertokenScale,
    const aclTensor *bias,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnQuantMatmulGeluExample(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_QUANT_MATMUL_GELU_EXAMPLE_H_