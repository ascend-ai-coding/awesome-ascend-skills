/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Xdivy InferShape — 输出 shape = broadcastMax(x1, x2)，右对齐逐维取 max
#include "register/op_impl_registry.h"
#include <cstdint>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

// InferShapeForXdivy: 广播 shape 推导 — 右对齐补 1，逐维取 max；
// 两维既不相等又都非 1 时报错
static ge::graphStatus InferShapeForXdivy(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const int64_t rank1 = static_cast<int64_t>(x1Shape->GetDimNum());
    const int64_t rank2 = static_cast<int64_t>(x2Shape->GetDimNum());
    const int64_t outRank = (rank1 > rank2) ? rank1 : rank2;

    for (int64_t d = 0; d < outRank; ++d) {
        const int64_t off1 = d - (outRank - rank1);
        const int64_t off2 = d - (outRank - rank2);

        const int64_t dim1 = (off1 >= 0) ? x1Shape->GetDim(off1) : 1;
        const int64_t dim2 = (off2 >= 0) ? x2Shape->GetDim(off2) : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1,x2", "incompatible",
                                                   "broadcast incompatible: dim sizes must be equal or 1");
            return GRAPH_FAILED;
        }

        const int64_t outDim = (dim1 > dim2) ? dim1 : dim2;
        yShape->AppendDim(outDim);
    }

    return GRAPH_SUCCESS;
}

// 注册 shape 推导回调
IMPL_OP_INFERSHAPE(Xdivy).InferShape(InferShapeForXdivy);

} // namespace ops
