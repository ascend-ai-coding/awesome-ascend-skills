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
 * Heavy kernels：参考标准 Python perf 模板（mc2_alltoall_quant_matmul.py:408）
 *   boost = torch.randn(10000, 10000, dtype=torch.bfloat16)  # 200MB
 *   for _ in range(100): torch.exp(boost)                    # 去 host bound
 *   self.cache_flush = torch.zeros((128*1024*1024,), dtype=torch.bfloat16)  # 256MB
 *   for _ in range(10): self.cache_flush.add_(1)             # 刷 L2 cache
 *   self.execute_custom_operator()                           # 测目标算子
 *
 * heavy_exp_kernel: boost 预热用（y = exp(x)），bfloat16
 * heavy_add_kernel: cache_flush 用（x += 1，inplace），bfloat16
 */

#pragma once

#include "kernel_basic_intf.h"

using HeavyT = half;

__global__ __aicore__ __vector__ void heavy_exp_kernel(
    GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t blockLength)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr int64_t PIPELINE_DEPTH = 2;
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<HeavyT> xGm, yGm;
    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueY;

    constexpr int64_t tileSize = 32 * 1024;
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueY, PIPELINE_DEPTH, tileSize);

    int64_t blockOffset = blockLength * AscendC::GetBlockIdx();
    xGm.SetGlobalBuffer((__gm__ HeavyT *)x + blockOffset);
    yGm.SetGlobalBuffer((__gm__ HeavyT *)y + blockOffset);

    int64_t currentBlockLength = totalLength - blockOffset;
    if (currentBlockLength > blockLength) {
        currentBlockLength = blockLength;
    }
    if (currentBlockLength <= 0) {
        return;
    }
    int64_t elementNumPerTile = tileSize / sizeof(HeavyT);
    int64_t tileNum = currentBlockLength / elementNumPerTile;

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPadExtParams<HeavyT> padParams{false, 0, 0, 0};

    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t offset = i * elementNumPerTile;
        copyParams.blockLen = elementNumPerTile * sizeof(HeavyT);
        AscendC::LocalTensor<HeavyT> xLocal = inQueueX.AllocTensor<HeavyT>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<HeavyT>();
        AscendC::LocalTensor<HeavyT> yLocal = outQueueY.AllocTensor<HeavyT>();
        AscendC::Exp(yLocal, xLocal, elementNumPerTile);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        yLocal = outQueueY.DeQue<HeavyT>();
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }
}

__global__ __aicore__ __vector__ void heavy_add_kernel(
    GM_ADDR x, int64_t totalLength, int64_t blockLength)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr int64_t PIPELINE_DEPTH = 2;
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<HeavyT> xGm;
    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueY;

    constexpr int64_t tileSize = 32 * 1024;
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueY, PIPELINE_DEPTH, tileSize);

    int64_t blockOffset = blockLength * AscendC::GetBlockIdx();
    xGm.SetGlobalBuffer((__gm__ HeavyT *)x + blockOffset);

    int64_t currentBlockLength = totalLength - blockOffset;
    if (currentBlockLength > blockLength) {
        currentBlockLength = blockLength;
    }
    if (currentBlockLength <= 0) {
        return;
    }
    int64_t elementNumPerTile = tileSize / sizeof(HeavyT);
    int64_t tileNum = currentBlockLength / elementNumPerTile;

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPadExtParams<HeavyT> padParams{false, 0, 0, 0};

    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t offset = i * elementNumPerTile;
        copyParams.blockLen = elementNumPerTile * sizeof(HeavyT);
        AscendC::LocalTensor<HeavyT> xLocal = inQueueX.AllocTensor<HeavyT>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<HeavyT>();
        AscendC::LocalTensor<HeavyT> yLocal = outQueueY.AllocTensor<HeavyT>();
        AscendC::Adds(yLocal, xLocal, static_cast<HeavyT>(1.0f), elementNumPerTile);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        yLocal = outQueueY.DeQue<HeavyT>();
        AscendC::DataCopyPad(xGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }
}
