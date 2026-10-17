/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Xdivy Kernel 入口 — arch35 (Ascend 950)
// RANK 来自 TilingKey, DTYPE 来自 CANN 框架 (def 注册的 Input("x1") 类型)
// 注：kernel 入口函数名须与 opFile 名一致，受框架约束使用 snake_case

#include "kernel_operator.h"
#include "arch35/xdivy_kernel.h"
#include "arch35/xdivy_tiling_struct.h"

using TilingData4 = XdivyTilingData<XDIVY_RANK_4>;
using TilingData8 = XdivyTilingData<XDIVY_RANK_8>;

template <int32_t RANK>
__global__ __aicore__ void xdivy(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR ins[MAX_INPUT_SLOTS] = {x1, x2};
    GM_ADDR outs[MAX_OUTPUT_SLOTS] = {y};

    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (RANK == XDIVY_RANK_4) {
        GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
        XdivyKernel<DTYPE_X1, XDIVY_RANK_4> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
        XdivyKernel<DTYPE_X1, XDIVY_RANK_8> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    }
}
