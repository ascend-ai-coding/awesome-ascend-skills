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
 * \file moe_dispatch.cpp
 * \brief
 */
#include "kernel/moe_dispatch.h"

using namespace AscendC;
using namespace MoeDispatchImpl;

/**
 * @brief FP16 dispatch kernel 的内部入口
 *
 * 这里负责：
 * - 固定当前 sample 的数据类型为 FP16
 * - 创建本次 launch 对应的 TPipe
 * - 初始化并执行最小 dispatch kernel
 */
__attribute__((always_inline)) __aicore__ __inline__ void moe_dispatch_fp16(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expandX, GM_ADDR expandIdx,
    GM_ADDR expertTokenNums, GM_ADDR epRecvCounts, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    MoeDispatchOp<half> op;
    op.Init(mc2Context, x, expertIds, expandX, expandIdx, expertTokenNums, epRecvCounts, tilingGM, &pipe);
    op.Process();
}

/**
 * @brief AscendC 生成 host stub 时会绑定的全局 kernel 符号
 *
 * 这个函数是真正被 `<<<>>>` 调起的设备侧入口。
 */
extern "C" __global__ __aicore__ void MoeDispatch_Generic(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expandX, GM_ADDR expandIdx,
    GM_ADDR expertTokenNums, GM_ADDR epRecvCounts, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    moe_dispatch_fp16(mc2Context, x, expertIds, expandX, expandIdx, expertTokenNums, epRecvCounts, workspaceGM, tilingGM);
}

/**
 * @brief Host 侧的最小 launch wrapper
 *
 * 作用：
 * - 对外暴露稳定的 C 接口，便于 host 测试调用
 * - 将 blockDim、stream 和原始 GM 指针传给 AscendC kernel
 */
extern "C" void moe_dispatch_demo(uint32_t blockDim, void* stream,
    uint8_t* mc2Context, uint8_t* x, uint8_t* expertIds, uint8_t* expandX, uint8_t* expandIdx,
    uint8_t* expertTokenNums, uint8_t* epRecvCounts, uint8_t* workspaceGM,
    uint8_t* tilingGM)
{
    MoeDispatch_Generic<<<blockDim, nullptr, stream>>>(
        mc2Context, x, expertIds, expandX, expandIdx, expertTokenNums, epRecvCounts, workspaceGM, tilingGM);
}