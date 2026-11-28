// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include "scalar_vector_ops.h"

// Launch one AIV block on A2. Inputs: 64 float scores, 64 uint32 indices.
// Outputs: 8 float scores and 8 uint32 indices. Initialize input indices on host.
// Host queries GetSortTmpSize/GetConcatTmpSize for 64 floats, rounds sizes up
// to 32B and checks the sum of all allocations against available UB.
extern "C" __global__ __aicore__ void topk64(GM_ADDR scores, GM_ADDR indices, GM_ADDR outScores, GM_ADDR outIndices,
                                             uint32_t sortTmpBytes, uint32_t concatTmpBytes)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> valueBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortedBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> concatTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outIndexBuf;
    pipe.InitBuffer(valueBuf, 64 * sizeof(float));
    pipe.InitBuffer(indexBuf, 64 * sizeof(uint32_t));
    pipe.InitBuffer(sortedBuf, 128 * sizeof(float));
    pipe.InitBuffer(sortTmpBuf, sortTmpBytes);
    pipe.InitBuffer(concatTmpBuf, concatTmpBytes == 0 ? 32 : concatTmpBytes);
    pipe.InitBuffer(outBuf, 64 * sizeof(float));
    pipe.InitBuffer(outIndexBuf, 64 * sizeof(uint32_t));
    auto values = valueBuf.Get<float>();
    auto inputIndices = indexBuf.Get<uint32_t>();
    auto sorted = sortedBuf.Get<float>();
    auto sortTmp = sortTmpBuf.Get<float>();
    auto concatTmp = concatTmpBuf.Get<float>();
    auto outputValues = outBuf.Get<float>();
    auto outputIndices = outIndexBuf.Get<uint32_t>();
    AscendC::GlobalTensor<float> srcGm;
    AscendC::GlobalTensor<float> dstGm;
    AscendC::GlobalTensor<uint32_t> indexGm;
    AscendC::GlobalTensor<uint32_t> dstIndexGm;
    srcGm.SetGlobalBuffer((__gm__ float*)scores, 64);
    indexGm.SetGlobalBuffer((__gm__ uint32_t*)indices, 64);
    dstGm.SetGlobalBuffer((__gm__ float*)outScores, 8);
    dstIndexGm.SetGlobalBuffer((__gm__ uint32_t*)outIndices, 8);
    AscendC::DataCopy(values, srcGm, 64);
    AscendC::DataCopy(inputIndices, indexGm, 64);
    ScalarVectorExamples::Sync<AscendC::HardEvent::MTE2_V>(pipe);
    ScalarVectorExamples::Sort64(values, inputIndices, sorted, sortTmp, concatTmp, outputValues, outputIndices);
    ScalarVectorExamples::Sync<AscendC::HardEvent::V_MTE3>(pipe);
    AscendC::DataCopy(dstGm, outputValues, 8);
    AscendC::DataCopy(dstIndexGm, outputIndices, 8);
    ScalarVectorExamples::Sync<AscendC::HardEvent::MTE3_S>(pipe);
}
