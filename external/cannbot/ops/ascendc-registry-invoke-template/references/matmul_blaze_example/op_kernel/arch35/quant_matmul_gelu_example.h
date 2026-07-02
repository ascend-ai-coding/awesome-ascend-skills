/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_gelu_example.h
 * \brief quant_matmul_gelu_example 算子 Kernel 头文件占位（arch35 架构）
 *
 * 基于 Blaze/tensor_api 的 MIX 融合算子通过 MatmulKernelFused 模板驱动
 * BlockScheduler + BlockMmad (AIC) + Epilogue (AIV) 流水。
 * 本文件为空定义，仅满足 registry-invoke 构建系统对 *.h 的存在性要求。
 * [MODIFY] 实际计算逻辑在 arch35.cpp 和 Epilogue 类中实现，无需修改本文件。
 */

#ifndef QUANT_MATMUL_GELU_EXAMPLE_H
#define QUANT_MATMUL_GELU_EXAMPLE_H

#endif // QUANT_MATMUL_GELU_EXAMPLE_H