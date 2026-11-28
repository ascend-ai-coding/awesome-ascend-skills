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
 * \file add_example_arch22.cpp
 * \brief AddExample 算子 kernel 入口（arch22 架构）
 * 
 * 模板参数说明（与 add_example_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义对应）：
 *   - D_T_X: 数据类型，由 ASCENDC_TPL_DATATYPE_DECL 定义
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲），由 ASCENDC_TPL_UINT_DECL 定义
 */

#include "arch22/add_example.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void add_example(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddExampleTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddExampleTilingData, tilingData, tiling);
    NsAddExample::AddExample<D_T_X, BUFFER_MODE> op;
    op.Init(x, y, z, &tilingData);
    op.Process();

    // if constexpr 典型用法示例（按需在 Process 内部使用）：
    // if constexpr (BUFFER_MODE == 1) { /* 双缓冲逻辑 */ }
    // if constexpr (std::is_same_v<D_T_X, float>) { /* float 专属逻辑 */ }
}
