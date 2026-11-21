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
 * \file euclidean_norm_graph_infer.cpp
 * \brief EuclideanNorm 算子的 graph 侧推理注册（InferDataType）。
 *
 *   - 与 op_host/euclidean_norm_infershape.cpp 分工：shape 推理（含 axes 值依赖读取）留在
 *     op_host 侧 IMPL_OP_INFERSHAPE；输出 dtype 推理（y 跟随 x 直通）在本文件
 *     IMPL_OP(EuclideanNorm).InferDataType(...) 注册（结构对齐 ops-cv/image/crop 的
 *     op_graph/crop_graph_infer.cpp）。
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
using namespace ge;

static constexpr int64_t IDX_0 = 0;

// EuclideanNorm: 输出 y dtype 跟随输入 x dtype（fp16 / bf16 / fp32 / int32 直通）
static ge::graphStatus InferDataTypeEuclideanNorm(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeEuclideanNorm");
    const ge::DataType xDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, xDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeEuclideanNorm");
    return GRAPH_SUCCESS;
}

IMPL_OP(EuclideanNorm).InferDataType(InferDataTypeEuclideanNorm);
} // namespace ops
