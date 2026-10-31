/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Compile-only probe for register_resident_vector_chain_vf.h. This file is
// not a runnable kernel: it instantiates both template instances (with and
// without the in-register round-trip) for FP16 and BF16 under the
// device-only contract, mirroring how the caller derives __ubuf__ pointers
// from LocalTensor.GetPhyAddr().
#include "../code/register_resident_vector_chain_vf.h"

using namespace AscendC;

template <typename T>
__aicore__ void InstantiateVectorChainForType(LocalTensor<T>& inA, LocalTensor<T>& inB, LocalTensor<T>& out,
                                              uint32_t elements, float a, float b, float c, float d, float e, float f)
{
    __ubuf__ T* aPtr = (__ubuf__ T*)inA.GetPhyAddr();
    __ubuf__ T* bPtr = (__ubuf__ T*)inB.GetPhyAddr();
    __ubuf__ T* outPtr = (__ubuf__ T*)out.GetPhyAddr();
    VectorChainVF<T, true>(aPtr, bPtr, outPtr, elements, a, b, c, d, e, f);
    VectorChainVF<T, false>(aPtr, bPtr, outPtr, elements, a, b, c, d, e, f);
}

__aicore__ void InstantiateVectorChainFp16(LocalTensor<half>& inA, LocalTensor<half>& inB, LocalTensor<half>& out,
                                           uint32_t elements, float a, float b, float c, float d, float e, float f)
{
    InstantiateVectorChainForType<half>(inA, inB, out, elements, a, b, c, d, e, f);
}

__aicore__ void InstantiateVectorChainBf16(LocalTensor<bfloat16_t>& inA, LocalTensor<bfloat16_t>& inB,
                                           LocalTensor<bfloat16_t>& out, uint32_t elements, float a, float b, float c,
                                           float d, float e, float f)
{
    InstantiateVectorChainForType<bfloat16_t>(inA, inB, out, elements, a, b, c, d, e, f);
}
