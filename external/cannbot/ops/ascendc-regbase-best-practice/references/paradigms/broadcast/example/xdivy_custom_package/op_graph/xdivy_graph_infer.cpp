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
 * \file xdivy_graph_infer.cpp
 * \brief Xdivy 算子的 graph 侧推理注册（InferDataType）。
 *
 *   - 与 op_host/xdivy_infershape.cpp 分工：shape 推理（broadcast 输出 shape =
 *     broadcastMax(x1, x2)）留在 op_host 侧 IMPL_OP_INFERSHAPE；输出 dtype 推理
 *     （y 跟随 x1 直通）在本文件 IMPL_OP(Xdivy).InferDataType(...) 注册。
 */
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"

namespace ops {
using namespace ge;

static constexpr int64_t IDX_0 = 0;

// Xdivy: 输出 y dtype 跟随输入 x1 dtype（fp16 / bf16 / fp32 直通）
static ge::graphStatus InferDataTypeXdivy(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeXdivy");
    const ge::DataType x1Dtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, x1Dtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeXdivy");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Xdivy).InferDataType(InferDataTypeXdivy);
} // namespace ops
