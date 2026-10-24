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
 * \file x_all_gather_matmul_proto.cpp
 * \brief InferShape / InferDataType for XAllGatherMatmul.
 *        Migrated from ops-transformer/mc2/x_all_gather_matmul/op_host/x_all_gather_matmul_infershape.cpp.
 *        x_ops convention uses <op>_proto.cpp as the infershape file (no separate REG_OP proto header;
 *        x_ops registers the op via OpDef/OP_ADD in <op>_def.cpp). The infer body delegates to the
 *        vendored mc2_common_infershape helper (AllGatherMatmulCommonInferShape, GATHER_OUT_V1).
 */
#include "register/op_impl_registry.h"
#include "mc2_hcom_topo_info.h"
#include "op_host/mc2_common_infershape.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeXAllGatherMatmul(gert::InferShapeContext* context)
{
    if (AllGatherMatmulCommonInferShape(context, GATHER_OUT_V1) != GRAPH_SUCCESS) { return GRAPH_FAILED; }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeXAllGatherMatmul(gert::InferDataTypeContext* context) {
    auto d_type = context->GetInputDataType(0);
    context->SetOutputDataType(0, d_type);
    context->SetOutputDataType(1, d_type);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(XAllGatherMatmul).InferShape(InferShapeXAllGatherMatmul).InferDataType(InferDataTypeXAllGatherMatmul);
} // namespace ops
