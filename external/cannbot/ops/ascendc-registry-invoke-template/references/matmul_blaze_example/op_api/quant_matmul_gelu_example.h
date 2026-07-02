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
 * @file quant_matmul_gelu_example.h
 * @brief ACLNN L0 API 接口声明 - AIC+AIV 融合算子脚手架
 *
 * [MODIFY] 修改函数名和参数列表以匹配您的算子
 */

#ifndef OP_API_INC_LEVEL0_QUANT_MATMUL_GELU_EXAMPLE_H_
#define OP_API_INC_LEVEL0_QUANT_MATMUL_GELU_EXAMPLE_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* QuantMatmulGeluExample(const aclTensor* x1, const aclTensor* x2,
    const aclTensor* scale, const aclTensor* pertokenScale, const aclTensor* bias,
    aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_QUANT_MATMUL_GELU_EXAMPLE_H_