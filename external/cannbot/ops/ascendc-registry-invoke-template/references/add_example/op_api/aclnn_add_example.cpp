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
 * @file aclnn_add_example.cpp
 * @brief ACLNN L2 API 实现 - 二元算子示例
 *
 * ACLNN 接口采用两段式设计：
 * 1. aclnn{Op}GetWorkspaceSize - 计算 workspace 大小，创建执行器
 * 2. aclnn{Op} - 执行计算
 *
 * 文件组织：
 * - aclnn_{op}.h/cpp  -> L2 API（本文件）：参数检查、Contiguous/ViewCopy 处理
 * - {op}.h/cpp        -> L0 API（底层实现）：形状推导、Kernel 调度
 */

#include "aclnn_add_example.h"
#include "add_example.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static bool HasEmptyTensor(const aclTensor* x1, const aclTensor* x2)
{
    return x1->IsEmpty() || x2->IsEmpty();
}

static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(x1, x2->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, x1->GetDataType(), return false);
    
    OP_CHECK(IsDtypeSupported(x1->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Dtype not supported: dtype=%d. Supported: FLOAT, INT32.",
                     static_cast<int>(x1->GetDataType())),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    auto format1 = x1->GetStorageFormat();
    auto format2 = x2->GetStorageFormat();
    auto formatOut = out->GetStorageFormat();

    OP_CHECK(!(IsPrivateFormat(format1) || IsPrivateFormat(format2) || IsPrivateFormat(formatOut)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Private format not supported: x1=%d, x2=%d, out=%d",
                     static_cast<int>(format1), static_cast<int>(format2), static_cast<int>(formatOut)),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(x1, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(x2, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(x1, x2, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(x1, x2, out), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: x1_dtype=%d, x2_dtype=%d, out_dtype=%d",
               static_cast<int>(x1->GetDataType()), static_cast<int>(x2->GetDataType()),
               static_cast<int>(out->GetDataType()));
    CHECK_COND(CheckFormat(x1, x2, out), ACLNN_ERR_PARAM_INVALID,
               "CheckFormat failed: x1_format=%d, x2_format=%d, out_format=%d",
               static_cast<int>(x1->GetStorageFormat()), static_cast<int>(x2->GetStorageFormat()),
               static_cast<int>(out->GetStorageFormat()));
    CHECK_COND(CheckShape(x1, x2, out), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed: x1_dim=%zu, x2_dim=%zu, out_dim=%zu",
               x1->GetViewShape().GetDimNum(), x2->GetViewShape().GetDimNum(),
               out->GetViewShape().GetDimNum());
    return ACLNN_SUCCESS;
}

/**
 * @brief 第一段接口：计算 workspace 大小
 *
 * 标准流程：
 * 1. CREATE_EXECUTOR()         - 创建执行器
 * 2. CheckParams()             - 参数检查
 * 3. HasEmptyTensor()          - 空 Tensor 快速返回
 * 4. Contiguous()              - 非连续 Tensor 转换（保证 Kernel 输入连续）
 * 5. l0op::{Op}()              - 调用 L0 算子
 * 6. ViewCopy()                - 输出非连续处理（支持用户 out tensor 任意 stride）
 * 7. GetWorkspaceSize()        - 获取 workspace 大小
 */
extern "C" aclnnStatus aclnnAddExampleGetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAddExample, DFX_IN(x1, x2), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x1, x2, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(x1, x2)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto x1Contiguous = l0op::Contiguous(x1, uniqueExecutor.get());
    CHECK_RET(x1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto x2Contiguous = l0op::Contiguous(x2, uniqueExecutor.get());
    CHECK_RET(x2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::AddExample(x1Contiguous, x2Contiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief 第二段接口：执行计算
 */
extern "C" aclnnStatus aclnnAddExample(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddExample);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
