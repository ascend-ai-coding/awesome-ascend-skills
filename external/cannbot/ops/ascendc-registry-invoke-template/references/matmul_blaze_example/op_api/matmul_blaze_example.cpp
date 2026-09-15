/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_blaze_example.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(MatmulBlazeExample);

const aclTensor* MatmulBlazeExample(const aclTensor* a, const aclTensor* b, aclOpExecutor* executor)
{
    OP_CHECK_WRONG_DIMENSION(a, 2, return nullptr);
    OP_CHECK_WRONG_DIMENSION(b, 2, return nullptr);
    int64_t M = a->GetViewShape().GetDim(0), K = a->GetViewShape().GetDim(1);
    int64_t bK = b->GetViewShape().GetDim(0), N = b->GetViewShape().GetDim(1);
    OP_CHECK(K == bK, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MatmulBlazeExample K mismatch"), return nullptr);

    Shape outShape({M, N});
    auto out = executor->AllocTensor(outShape, DataType::DT_BF16, Format::FORMAT_ND);
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor failed"), return nullptr);

    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_3510,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MatmulBlazeExample only supports DAV_3510"), return nullptr);

    auto ret = INFER_SHAPE(MatmulBlazeExample, OP_INPUT(a, b), OP_OUTPUT(out));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "InferShape failed ret=%d", ret);
    ret = ADD_TO_LAUNCHER_LIST_AICORE(MatmulBlazeExample, OP_INPUT(a, b), OP_OUTPUT(out));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list failed ret=%d", ret);
    return out;
}
} // namespace l0op
