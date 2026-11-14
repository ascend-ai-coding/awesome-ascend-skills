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
 * \file euclidean_norm_tiling_key.h
 * \brief EuclideanNorm TilingKey 模板参数声明
 *
 * 两个 bool —— isGroup / isEmptyTensor（reduction 范式 TPL 双 bool 轴）。
 * dtype 走 DTYPE_X 编译期实例化（fp16 / bf16 / fp32 / int32 共 4 份），
 * 与 tilingkey 的 3 个组合（base×1 + empty×1 + group×1）相乘 ⇒ 共 12 份 binary。
 */
#ifndef OPS_NORM_EUCLIDEAN_NORM_TILING_KEY_H_
#define OPS_NORM_EUCLIDEAN_NORM_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// TPL 模板参数声明：isGroup / isEmptyTensor 双 bool 轴（dtype 走 DTYPE_X 不进 key）。
ASCENDC_TPL_ARGS_DECL(EuclideanNorm, ASCENDC_TPL_BOOL_DECL(isGroup, 0, 1), ASCENDC_TPL_BOOL_DECL(isEmptyTensor, 0, 1));

// TPL 组合选择：枚举 (isGroup, isEmptyTensor) 的 3 个合法组合（base / empty / group）。
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isGroup, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isGroup, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isGroup, 1), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)));

#endif // OPS_NORM_EUCLIDEAN_NORM_TILING_KEY_H_
