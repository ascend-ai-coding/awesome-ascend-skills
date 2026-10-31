/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * AdamApplyOneAssign Kernel 入口
 * RANK 来自 TilingKey, DTYPE 来自 CANN 框架 (def 注册的 Input("input0") 类型)
 * 注：kernel 入口函数名须与 opFile 名一致，受框架约束使用 snake_case
 */
#include "kernel_operator.h"
#include "arch35/adam_apply_one_assign_kernel.h"
#include "arch35/adam_apply_one_assign_tiling_struct.h"

using TilingData4 = AdamTilingData<ADAM_RANK_4>;
using TilingData8 = AdamTilingData<ADAM_RANK_8>;

template <int32_t RANK>
__global__ __aicore__ void adam_apply_one_assign(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2, GM_ADDR input3,
                                                 GM_ADDR input4, GM_ADDR mul0_x, GM_ADDR mul1_x, GM_ADDR mul2_x,
                                                 GM_ADDR mul3_x, GM_ADDR add2_y, GM_ADDR out0, GM_ADDR out1,
                                                 GM_ADDR out2, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR ins[MAX_INPUT_SLOTS] = {input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, add2_y};
    GM_ADDR outs[MAX_OUTPUT_SLOTS] = {out0, out1, out2};
    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (RANK == ADAM_RANK_4) {
        GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
        AdamKernel<DTYPE_INPUT0, ADAM_RANK_4> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
        AdamKernel<DTYPE_INPUT0, ADAM_RANK_8> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    }
}
