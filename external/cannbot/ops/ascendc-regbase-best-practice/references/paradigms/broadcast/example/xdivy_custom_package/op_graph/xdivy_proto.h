/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Xdivy GE IR 原型注册 — 声明算子在图引擎中的节点签名 (2入1出)
#ifndef XDIVY_PROTO_H_
#define XDIVY_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * REG_OP(Xdivy): registers a GE IR node type named "Xdivy".
 *
 *   .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT})): dividend, named
 *     "x1", accepts bf16 / float16 / float32.
 *   .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT})): divisor, named
 *     "x2", accepts the same types.
 *   .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT})): result tensor,
 *     named "y", same type as inputs.
 *   .OP_END_FACTORY_REG(Xdivy): completes the factory registration so the
 *     node can be instantiated via op::Xdivy("name").
 *
 * Formula:  y = (x1 == 0) ? 0 : (x1 / x2)
 */
REG_OP(Xdivy)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Xdivy)

} // namespace ge

#endif // XDIVY_PROTO_H_
