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
 * @file add_example.cpp
 * @brief ACLNN L0 API 实现 - 二元算子示例
 *
 * L0 API 职责：形状推导、Kernel 调度
 * L2 API 职责：参数检查、Contiguous/ViewCopy 处理
 */

#include "add_example.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AddExample);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32
};

static bool IsAiCoreSupport(const aclTensor* x1, const aclTensor* x2)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "AddExample not supported on this platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);
    OP_CHECK(CheckType(x1->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
             CheckType(x2->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "AddExample not supported: dtype x1=%d, x2=%d. Supported dtypes: FLOAT, INT32.",
                     static_cast<int>(x1->GetDataType()), static_cast<int>(x2->GetDataType())),
             return false);
    return true;
}

static bool AddExampleInferShape(const op::Shape& x1Shape, const op::Shape& x2Shape, op::Shape& outShape)
{
    OP_CHECK(BroadcastInferShape(x1Shape, x2Shape, outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape broadcast failed."), return false);
    return true;
}

static const aclTensor* AddExampleAiCore(const aclTensor* x1, const aclTensor* x2,
                                          const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AddExampleAiCore, x1, x2, out);

    // ADD_TO_LAUNCHER_LIST_AICORE 宏定义在 opdev/make_op_executor.h，支持多种调用形式：
    //   带属性：   ADD_TO_LAUNCHER_LIST_AICORE(TopKV2, OP_INPUT(self, k), OP_OUTPUT(values, indices), OP_ATTR(sorted, dim, largest));
    //   显式引擎： ADD_TO_LAUNCHER_LIST_AICORE(ReduceAll, op::AI_CORE, OP_INPUT(self, dims), OP_ATTR(keepdim), OP_OUTPUT(out));
    //   动态形状： ADD_TO_LAUNCHER_LIST_AICORE(MaskedSelectV3, OP_INPUT(self, mask), OP_OUTPUT(out), OP_OUTSHAPE({outShapeTensor, 0}));
    // 更多用法参见 opdev/make_op_executor.h 中 OP_INPUT / OP_OUTPUT / OP_ATTR / OP_OUTSHAPE 等宏定义
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AddExample,
        OP_INPUT(x1, x2), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AddExampleAiCore failed."),
        return nullptr);
    return out;
}

/**
 * @brief L0 API 入口
 *
 * 流程：
 * 1. InferShape      - 形状推导（广播规则）
 * 2. IsAiCoreSupport - 判断执行路径
 * 3. AllocTensor     - 分配输出 Tensor
 * 4. {Op}AiCore      - 调用 Kernel
 */
const aclTensor* AddExample(const aclTensor* x1, const aclTensor* x2, aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    OP_CHECK(AddExampleInferShape(x1->GetViewShape(), x2->GetViewShape(), outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."), return nullptr);

    out = executor->AllocTensor(outShape, x1->GetDataType());

    OP_CHECK(IsAiCoreSupport(x1, x2),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    return AddExampleAiCore(x1, x2, out, executor);
}

} // namespace l0op
