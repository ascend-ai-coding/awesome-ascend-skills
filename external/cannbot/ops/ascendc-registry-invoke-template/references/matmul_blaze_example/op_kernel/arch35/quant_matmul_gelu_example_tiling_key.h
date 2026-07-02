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
 * \file quant_matmul_gelu_example_tiling_key.h
 * \brief quant_matmul_gelu_example 算子 TilingKey 定义（单一 TilingKey 版本）
 *
 * 单一 TilingKey（默认 0）：当前 MIX 融合算子使用固定数据类型组合和 layout，
 * 无需多 TilingKey 分发。
 *
 * [MODIFY] 如需支持多 dtype 组合（如 int8/bf16 切换），
 *   可改用 ASCENDC_TPL_ARGS_DECL/ASCENDC_TPL_SEL 模板编程方式，
 *   参考 add_example_tiling_key.h 的写法。
 *   注意：DAV_3510 的 Bisheng 编译器对 dual SEL 支持有限，
 *   单一 TilingKey 是更稳妥的选择。
 */

#ifndef __QUANT_MATMUL_GELU_EXAMPLE_TILING_KEY_H__
#define __QUANT_MATMUL_GELU_EXAMPLE_TILING_KEY_H__

// 单一 TilingKey (无 SEL)

#endif // __QUANT_MATMUL_GELU_EXAMPLE_TILING_KEY_H__