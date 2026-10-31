// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// ----------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <tuple>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "compiled_warmup_launch.h"

namespace {
constexpr int64_t kWarmupElements = 128LL * 1024 * 1024;
constexpr int32_t kPipelineDepth = 2;
constexpr int32_t kBufferCount = 2;
}

extern "C" __global__ __aicore__ void OpsProfilingWarmup(
    GM_ADDR x,
    GM_ADDR out,
    int64_t totalLength,
    int64_t blockLength,
    uint32_t tileBytes)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> outGm;
    AscendC::TQue<AscendC::TPosition::VECIN, kPipelineDepth> inputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, kPipelineDepth> outputQueue;

    pipe.InitBuffer(inputQueue, kPipelineDepth, tileBytes);
    pipe.InitBuffer(outputQueue, kPipelineDepth, tileBytes);

    const int64_t blockOffset = blockLength * AscendC::GetBlockIdx();
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x) + blockOffset);
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(out));

    int64_t currentBlockLength = totalLength - blockOffset;
    currentBlockLength = currentBlockLength > blockLength ? blockLength : currentBlockLength;
    if (currentBlockLength <= 0) {
        return;
    }

    const int64_t elementsPerTile = tileBytes / sizeof(half);
    for (int64_t offset = 0; offset < currentBlockLength; offset += elementsPerTile) {
        const int64_t remaining = currentBlockLength - offset;
        const int64_t tileElements = remaining > elementsPerTile ? elementsPerTile : remaining;
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(tileElements * sizeof(half));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};

        auto inputLocal = inputQueue.AllocTensor<half>();
        AscendC::DataCopyPad(inputLocal, xGm[offset], copyParams, padParams);
        inputQueue.EnQue(inputLocal);
        inputLocal = inputQueue.DeQue<half>();

        auto outputLocal = outputQueue.AllocTensor<half>();
        AscendC::Adds(outputLocal, inputLocal, static_cast<half>(1.0f), tileElements);
        outputQueue.EnQue(outputLocal);
        inputQueue.FreeTensor(inputLocal);

        outputLocal = outputQueue.DeQue<half>();
        AscendC::DataCopyPad(xGm[offset], outputLocal, copyParams);
        outputQueue.FreeTensor(outputLocal);
    }

    if (AscendC::GetBlockIdx() == 0) {
        auto dummy = outputQueue.AllocTensor<half>();
        AscendC::Duplicate(dummy, static_cast<half>(0.0f), 1);
        AscendC::DataCopyExtParams copyParams{1, sizeof(half), 0, 0, 0};
        AscendC::DataCopyPad(outGm, dummy, copyParams);
        outputQueue.FreeTensor(dummy);
    }
}

std::tuple<int64_t, int64_t, uint32_t> CalcCompiledWarmupTiling()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    int64_t blockDim = platform->GetCoreNumAiv();
    blockDim = blockDim > 0 ? blockDim : 1;
    const int64_t blockLength = (kWarmupElements + blockDim - 1) / blockDim;
    uint32_t tileBytes = static_cast<uint32_t>(ubBytes / kPipelineDepth / kBufferCount);
    tileBytes = (tileBytes / 32) * 32;
    return std::make_tuple(blockDim, blockLength, tileBytes);
}

extern "C" void LaunchCompiledWarmupHalf(
    GM_ADDR x,
    GM_ADDR out,
    int64_t totalLength,
    int64_t blockLength,
    uint32_t tileBytes,
    uint32_t blockDim,
    void *stream)
{
    OpsProfilingWarmup<<<blockDim, nullptr, stream>>>(x, out, totalLength, blockLength, tileBytes);
}
