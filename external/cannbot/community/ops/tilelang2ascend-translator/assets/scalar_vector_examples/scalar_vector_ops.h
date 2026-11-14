// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#ifndef SCALAR_VECTOR_EXAMPLES_OPS_H
#define SCALAR_VECTOR_EXAMPLES_OPS_H

#include "kernel_operator.h"

namespace ScalarVectorExamples {
template <AscendC::HardEvent Event>
__aicore__ inline void Sync(AscendC::TPipe& pipe)
{
    const int32_t id = pipe.FetchEventID(Event);
    AscendC::SetFlag<Event>(id);
    AscendC::WaitFlag<Event>(id);
}

// A2 float32: 64 scores, 64 input indices, 128-float sorted records.
// Buffers are disjoint except concat, which Concat aliases to values on A2.
__aicore__ inline void Sort64(const AscendC::LocalTensor<float>& values, const AscendC::LocalTensor<uint32_t>& indices,
                              const AscendC::LocalTensor<float>& sorted, AscendC::LocalTensor<float>& sortTmp,
                              const AscendC::LocalTensor<float>& concatTmp,
                              const AscendC::LocalTensor<float>& outValues,
                              const AscendC::LocalTensor<uint32_t>& outIndices)
{
    AscendC::LocalTensor<float> concat;
    AscendC::Concat(concat, values, concatTmp, 4);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sort<float, true>(sorted, concat, indices, sortTmp, 2);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Extract(outValues, outIndices, sorted, 2);
}

// count in [64, 255*64], divisible by 64; base divisible by 8.
// Source has base+count floats, destination has count floats; buffers disjoint.
__aicore__ inline void SplitEvenOdd(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
                                    uint32_t base, uint32_t count)
{
    const uint32_t half = count / 2;
    auto input = src[base];
    uint64_t evenCount = 0;
    uint64_t oddCount = 0;
    const AscendC::GatherMaskParams params{1, static_cast<uint16_t>(count / 64), 8, 0};
    AscendC::GatherMask(dst, input, static_cast<uint8_t>(1), false, 0, params, evenCount);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::GatherMask(dst[half], input, static_cast<uint8_t>(2), false, 0, params, oddCount);
    AscendC::PipeBarrier<PIPE_V>();
}
} // namespace ScalarVectorExamples
#endif
