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
 * \file matmul_tiling.h
 * \brief Host-side tiling engine for matmul kernels.
 *
 */

#ifndef MATMUL_TILING_H
#define MATMUL_TILING_H

// MatmulTilingSwat的实现参考 ascendc-blaze-best-practice/references/matmul_custom/include/tiling
// 差异点：quant_matmul_gelu_example算子输入矩阵的dtype为int8，tiling计算中应按照DATA_SIZE_INT8=1来计算，不能按照默认DATA_SIZE_FP16=2计算

class MatmulTilingSwat  {

};

#endif // MATMUL_TILING_H
