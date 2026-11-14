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
 * \file euclidean_norm_proto.h
 * \brief EuclideanNorm 算子原型声明
 */
#ifndef OPS_OP_PROTO_INC_EUCLIDEAN_NORM_H_
#define OPS_OP_PROTO_INC_EUCLIDEAN_NORM_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Computes the Euclidean norm along the given axes:
 *        y = sqrt( sum( x^2 ) along axes )
 *
 * @par Inputs:
 * Two inputs, including:
 * @li x: A ND tensor of type NumberType.
 *Must be one of the following types: float32, float64, int32, uint8, int16,
 *int8, complex64, int64, qint8, quint8, qint32, bfloat16, uint16, complex128, float16, uint32, uint64.
 * @li axes: An IndexNumberType tensor (int32/int64), 1-D, listing the axes to reduce.
 *
 * @par Attributes:
 * keep_dims: Optional bool, default false. If true, the reduced axes are
 *            retained as size-1 dimensions in the output.
 *
 * @par Outputs:
 * y: A tensor with the same dtype as x. Output shape is derived from x's shape,
 *    axes, and keep_dims.
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator tf.math.reduce_euclidean_norm.
 */
REG_OP(EuclideanNorm)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(EuclideanNorm)

} // namespace ge

#endif // OPS_OP_PROTO_INC_EUCLIDEAN_NORM_H_
