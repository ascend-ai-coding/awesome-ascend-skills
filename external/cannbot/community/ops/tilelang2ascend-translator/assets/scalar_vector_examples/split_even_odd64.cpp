// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include "scalar_vector_ops.h"

// Launch one AIV block on A2. Input and output: 64 floats each, disjoint.
extern "C" __global__ __aicore__ void split_even_odd64(GM_ADDR input, GM_ADDR output)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> srcBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dstBuf;
    pipe.InitBuffer(srcBuf, 64 * sizeof(float));
    pipe.InitBuffer(dstBuf, 64 * sizeof(float));
    auto src = srcBuf.Get<float>();
    auto dst = dstBuf.Get<float>();
    AscendC::GlobalTensor<float> srcGm;
    AscendC::GlobalTensor<float> dstGm;
    srcGm.SetGlobalBuffer((__gm__ float*)input, 64);
    dstGm.SetGlobalBuffer((__gm__ float*)output, 64);
    AscendC::DataCopy(src, srcGm, 64);
    ScalarVectorExamples::Sync<AscendC::HardEvent::MTE2_V>(pipe);
    ScalarVectorExamples::SplitEvenOdd(dst, src, 0, 64);
    ScalarVectorExamples::Sync<AscendC::HardEvent::V_MTE3>(pipe);
    AscendC::DataCopy(dstGm, dst, 64);
    ScalarVectorExamples::Sync<AscendC::HardEvent::MTE3_S>(pipe);
}
