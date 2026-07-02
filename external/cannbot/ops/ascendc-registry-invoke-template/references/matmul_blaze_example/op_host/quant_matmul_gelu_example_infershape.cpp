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
 * \file quant_matmul_gelu_example_infershape.cpp
 * \brief quant_matmul_gelu_example 算子形状推导与数据类型推导
 *   如需从输入推导输出 dtype，从 context->GetInputDataType(0) 获取 x1 dtype。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4QuantMatmulGeluExample(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    int64_t x1DimNum = x1Shape->GetDimNum();
    int64_t x2DimNum = x2Shape->GetDimNum();

    // [MODIFY] 维数约束。当前仅支持 2D GEMM，如需 3D BMM 放宽为 >= 2。
    OP_CHECK_IF(x1DimNum != 2,
        OP_LOGE(context, "quant_matmul_gelu_example: x1 must be 2D (M,K), got rank %ld", x1DimNum),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(x2DimNum != 2,
        OP_LOGE(context, "quant_matmul_gelu_example: x2 must be 2D (N,K), got rank %ld", x2DimNum),
        return ge::GRAPH_FAILED);

    // x2 为 (N, K) 转置语义：K = x2[-1]，N = x2[-2]
    int64_t x1K = x1Shape->GetDim(x1DimNum - 1);
    int64_t x2K = x2Shape->GetDim(x2DimNum - 1);
    OP_CHECK_IF(x1K != x2K,
        OP_LOGE(context, "quant_matmul_gelu_example: K dimension mismatch. x1.K=%ld, x2.K=%ld", x1K, x2K),
        return ge::GRAPH_FAILED);

    int64_t M = x1Shape->GetDim(x1DimNum - 2);
    int64_t N = x2Shape->GetDim(x2DimNum - 2);
    // [MODIFY] 最小维数约束。BLOCK_CUBE 对齐要求 M/N >= 16，如需放宽调整此阈值。
    OP_CHECK_IF(M < 16 || N < 16,
        OP_LOGE(context, "quant_matmul_gelu_example: M and N must >= 16 (BLOCK_CUBE alignment). M=%ld, N=%ld", M, N),
        return ge::GRAPH_FAILED);

    yShape->SetDimNum(2);
    yShape->SetDim(0, M);
    yShape->SetDim(1, N);

    return ge::GRAPH_SUCCESS;
}

// [MODIFY] 输出 dtype 推导。当前固定为 BF16，如需动态推导修改此函数。
static ge::graphStatus InferDataType4QuantMatmulGeluExample(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantMatmulGeluExample)
    .InferShape(InferShape4QuantMatmulGeluExample)
    .InferDataType(InferDataType4QuantMatmulGeluExample);

} // namespace ops