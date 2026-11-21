/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
using namespace ge;
namespace ops {
static ge::graphStatus InferShape4MatmulBlazeExample(gert::InferShapeContext* context)
{
    const gert::Shape* aShape = context->GetInputShape(0); OP_CHECK_NULL_WITH_CONTEXT(context, aShape);
    const gert::Shape* bShape = context->GetInputShape(1); OP_CHECK_NULL_WITH_CONTEXT(context, bShape);
    gert::Shape* cShape = context->GetOutputShape(0); OP_CHECK_NULL_WITH_CONTEXT(context, cShape);
    OP_CHECK_IF(aShape->GetDimNum() != 2, OP_LOGE(context, "MatmulBlazeExample: a must be 2D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(bShape->GetDimNum() != 2, OP_LOGE(context, "MatmulBlazeExample: b must be 2D"), return ge::GRAPH_FAILED);
    int64_t M = aShape->GetDim(0); int64_t K = aShape->GetDim(1); int64_t bK = bShape->GetDim(0); int64_t N = bShape->GetDim(1);
    OP_CHECK_IF(M <= 0 || K <= 0 || N <= 0, OP_LOGE(context, "MatmulBlazeExample: invalid M/K/N"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(K != bK, OP_LOGE(context, "MatmulBlazeExample: K mismatch. a.K=%ld, b.K=%ld", K, bK), return ge::GRAPH_FAILED);
    cShape->SetDimNum(2); cShape->SetDim(0, M); cShape->SetDim(1, N); return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType4MatmulBlazeExample(gert::InferDataTypeContext* context)
{ context->SetOutputDataType(0, ge::DT_BF16); return ge::GRAPH_SUCCESS; }
IMPL_OP_INFERSHAPE(MatmulBlazeExample).InferShape(InferShape4MatmulBlazeExample).InferDataType(InferDataType4MatmulBlazeExample);
} // namespace ops
