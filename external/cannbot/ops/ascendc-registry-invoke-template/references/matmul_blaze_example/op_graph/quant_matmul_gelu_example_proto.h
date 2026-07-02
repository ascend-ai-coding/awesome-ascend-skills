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
 * \file quant_matmul_gelu_example_proto.h
 * \brief quant_matmul_gelu_example IR 算子原型定义
 *
 * [MODIFY] 修改 REG_OP 名称、INPUT/OUTPUT 名称和 TensorType 以匹配您的算子
 * 注：bias 的 aclnn 外部接口为 bfloat16，L2 层 cast 为 float 后下发，
 *     proto 中 bias 注册为 FLOAT（与 _def.cpp 一致）。
 */

#ifndef OPS_OP_PROTO_INC_QUANT_MATMUL_GELU_EXAMPLE_H_
#define OPS_OP_PROTO_INC_QUANT_MATMUL_GELU_EXAMPLE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(QuantMatmulGeluExample)
    .INPUT(x1, TensorType({DT_INT8}))
    .INPUT(x2, TensorType({DT_INT8}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(pertoken_scale, TensorType({DT_FLOAT}))
    .INPUT(bias, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16}))
    .OP_END_FACTORY_REG(QuantMatmulGeluExample)

} // namespace ge

#endif // OPS_OP_PROTO_INC_QUANT_MATMUL_GELU_EXAMPLE_H_