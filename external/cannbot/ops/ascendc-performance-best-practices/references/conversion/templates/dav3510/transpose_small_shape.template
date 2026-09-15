/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_small_shape.h
 * \brief transpose_small_shape
 */
#ifndef TRANSPOSE_SMALL_SHAPE_H
#define TRANSPOSE_SMALL_SHAPE_H

#include "transpose_base.h"
#include "simt_api/asc_simt.h"

#ifdef __DAV_FPGA__
constexpr int32_t THREAD_DIM = 512;
constexpr int32_t HALF_THREAD_DIM = 512;
constexpr int32_t QUARTER_THREAD_DIM = 512;
constexpr int32_t AN_EIGHTH_THREAD_DIM = 256;
#else
constexpr int32_t THREAD_DIM = 2048;
constexpr int32_t HALF_THREAD_DIM = 1024;
constexpr int32_t QUARTER_THREAD_DIM = 512;
constexpr int32_t AN_EIGHTH_THREAD_DIM = 256;
#endif

namespace Transpose {
using namespace AscendC;

template <typename T>
class TransposeSmallShape : public TransposeBase<T> {
public:
    __aicore__ inline TransposeSmallShape(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
    int32_t blockIdx_;
    const TransposeOpTilingData* tilingData_;
};

template <typename T>
__aicore__ inline void TransposeSmallShape<T>::Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimTwo(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                           uint32_t outputShape0, uint32_t outputShape1, uint32_t m0, uint32_t m1, uint32_t s0,
                           uint32_t s1)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        uint32_t inputIndex0 = yIdx - Simt::UintDiv(yIdx, m0, s0) * outputShape0;
        yIdx = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * outputShape1;
        uint32_t xIdx = inputIndex0 * outputShape1 + inputIndex1;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimThree(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                             uint32_t outputShape0, uint32_t outputShape1, uint32_t outputShape2, uint32_t dstStride0,
                             uint32_t dstStride1, uint32_t dstStride2, uint32_t m0, uint32_t m1, uint32_t m2,
                             uint32_t s0, uint32_t s1, uint32_t s2)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        uint32_t inputIndex0 = yIdx - Simt::UintDiv(yIdx, m0, s0) * outputShape0;
        yIdx = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * outputShape1;
        yIdx = Simt::UintDiv(yIdx, m1, s1);
        uint32_t inputIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * outputShape2;
        uint32_t xIdx = inputIndex0 * dstStride0 + inputIndex1 * dstStride1 + inputIndex2 * dstStride2;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimFour(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                            uint32_t outputShape0, uint32_t outputShape1, uint32_t outputShape2, uint32_t outputShape3,
                            uint32_t dstStride0, uint32_t dstStride1, uint32_t dstStride2, uint32_t dstStride3,
                            uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t s0, uint32_t s1, uint32_t s2,
                            uint32_t s3)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        uint32_t inputIndex0 = yIdx - Simt::UintDiv(yIdx, m0, s0) * outputShape0;
        yIdx = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * outputShape1;
        yIdx = Simt::UintDiv(yIdx, m1, s1);
        uint32_t inputIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * outputShape2;
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint32_t inputIndex3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * outputShape3;
        uint32_t xIdx = inputIndex0 * dstStride0 + inputIndex1 * dstStride1 + inputIndex2 * dstStride2 +
                        inputIndex3 * dstStride3;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimFive(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                            uint32_t outputShape0, uint32_t outputShape1, uint32_t outputShape2, uint32_t outputShape3,
                            uint32_t outputShape4, uint32_t dstStride0, uint32_t dstStride1, uint32_t dstStride2,
                            uint32_t dstStride3, uint32_t dstStride4, uint32_t m0, uint32_t m1, uint32_t m2,
                            uint32_t m3, uint32_t m4, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        uint32_t inputIndex0 = yIdx - Simt::UintDiv(yIdx, m0, s0) * outputShape0;
        yIdx = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * outputShape1;
        yIdx = Simt::UintDiv(yIdx, m1, s1);
        uint32_t inputIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * outputShape2;
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint32_t inputIndex3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * outputShape3;
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint32_t inputIndex4 = yIdx - Simt::UintDiv(yIdx, m4, s4) * outputShape4;
        uint32_t xIdx = inputIndex0 * dstStride0 + inputIndex1 * dstStride1 + inputIndex2 * dstStride2 +
                        inputIndex3 * dstStride3 + inputIndex4 * dstStride4;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimSix(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                           uint32_t outputShape0, uint32_t outputShape1, uint32_t outputShape2, uint32_t outputShape3,
                           uint32_t outputShape4, uint32_t outputShape5, uint32_t dstStride0, uint32_t dstStride1,
                           uint32_t dstStride2, uint32_t dstStride3, uint32_t dstStride4, uint32_t dstStride5,
                           uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t m4, uint32_t m5, uint32_t s0,
                           uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        uint32_t inputIndex0 = yIdx - Simt::UintDiv(yIdx, m0, s0) * outputShape0;
        yIdx = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * outputShape1;
        yIdx = Simt::UintDiv(yIdx, m1, s1);
        uint32_t inputIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * outputShape2;
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint32_t inputIndex3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * outputShape3;
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint32_t inputIndex4 = yIdx - Simt::UintDiv(yIdx, m4, s4) * outputShape4;
        yIdx = Simt::UintDiv(yIdx, m4, s4);
        uint32_t inputIndex5 = yIdx - Simt::UintDiv(yIdx, m5, s5) * outputShape5;
        uint32_t xIdx = inputIndex0 * dstStride0 + inputIndex1 * dstStride1 + inputIndex2 * dstStride2 +
                        inputIndex3 * dstStride3 + inputIndex4 * dstStride4 + inputIndex5 * dstStride5;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimSeven(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                             uint32_t outputShape0, uint32_t outputShape1, uint32_t outputShape2, uint32_t outputShape3,
                             uint32_t outputShape4, uint32_t outputShape5, uint32_t outputShape6, uint32_t dstStride0,
                             uint32_t dstStride1, uint32_t dstStride2, uint32_t dstStride3, uint32_t dstStride4,
                             uint32_t dstStride5, uint32_t dstStride6, uint32_t m0, uint32_t m1, uint32_t m2,
                             uint32_t m3, uint32_t m4, uint32_t m5, uint32_t m6, uint32_t s0, uint32_t s1, uint32_t s2,
                             uint32_t s3, uint32_t s4, uint32_t s5, uint32_t s6)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        uint32_t inputIndex0 = yIdx - Simt::UintDiv(yIdx, m0, s0) * outputShape0;
        yIdx = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * outputShape1;
        yIdx = Simt::UintDiv(yIdx, m1, s1);
        uint32_t inputIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * outputShape2;
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint32_t inputIndex3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * outputShape3;
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint32_t inputIndex4 = yIdx - Simt::UintDiv(yIdx, m4, s4) * outputShape4;
        yIdx = Simt::UintDiv(yIdx, m4, s4);
        uint32_t inputIndex5 = yIdx - Simt::UintDiv(yIdx, m5, s5) * outputShape5;
        yIdx = Simt::UintDiv(yIdx, m5, s5);
        uint32_t inputIndex6 = yIdx - Simt::UintDiv(yIdx, m6, s6) * outputShape6;
        uint32_t xIdx = inputIndex0 * dstStride0 + inputIndex1 * dstStride1 + inputIndex2 * dstStride2 +
                        inputIndex3 * dstStride3 + inputIndex4 * dstStride4 + inputIndex5 * dstStride5 +
                        inputIndex6 * dstStride6;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimEight(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                             uint32_t outputShape0, uint32_t outputShape1, uint32_t outputShape2, uint32_t outputShape3,
                             uint32_t outputShape4, uint32_t outputShape5, uint32_t outputShape6, uint32_t outputShape7,
                             uint32_t dstStride0, uint32_t dstStride1, uint32_t dstStride2, uint32_t dstStride3,
                             uint32_t dstStride4, uint32_t dstStride5, uint32_t dstStride6, uint32_t dstStride7,
                             uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t m4, uint32_t m5, uint32_t m6,
                             uint32_t m7, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5,
                             uint32_t s6, uint32_t s7)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        uint32_t inputIndex0 = yIdx - Simt::UintDiv(yIdx, m0, s0) * outputShape0;
        yIdx = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * outputShape1;
        yIdx = Simt::UintDiv(yIdx, m1, s1);
        uint32_t inputIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * outputShape2;
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint32_t inputIndex3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * outputShape3;
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint32_t inputIndex4 = yIdx - Simt::UintDiv(yIdx, m4, s4) * outputShape4;
        yIdx = Simt::UintDiv(yIdx, m4, s4);
        uint32_t inputIndex5 = yIdx - Simt::UintDiv(yIdx, m5, s5) * outputShape5;
        yIdx = Simt::UintDiv(yIdx, m5, s5);
        uint32_t inputIndex6 = yIdx - Simt::UintDiv(yIdx, m6, s6) * outputShape6;
        yIdx = Simt::UintDiv(yIdx, m6, s6);
        uint32_t inputIndex7 = yIdx - Simt::UintDiv(yIdx, m7, s7) * outputShape7;
        uint32_t xIdx = inputIndex0 * dstStride0 + inputIndex1 * dstStride1 + inputIndex2 * dstStride2 +
                        inputIndex3 * dstStride3 + inputIndex4 * dstStride4 + inputIndex5 * dstStride5 +
                        inputIndex6 * dstStride6 + inputIndex7 * dstStride7;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__aicore__ inline void TransposeSmallShape<T>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }
    uint32_t blkProcessNum = tilingData_->blkFactor;
    uint32_t blkStartOffset = blockIdx_ * tilingData_->blkFactor;
    if (blockIdx_ == tilingData_->realCoreNum - 1 && tilingData_->blkTailFactor != 0) {
        blkProcessNum = tilingData_->blkTailFactor;
    }

    uint32_t permSize = tilingData_->permSize;
    uint32_t outputShape[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t dstStride[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t dstStrideTmp[8] = {1, 1, 1, 1, 1, 1, 1, 1};

    for (uint32_t i = 1; i < permSize; i++) {
        dstStrideTmp[i] = tilingData_->inputShape[permSize - i] * dstStrideTmp[i - 1];
    }
    for (uint32_t i = 0; i < permSize; i++) {
        outputShape[i] = tilingData_->outputShape[permSize - 1 - i];
        dstStride[i] = dstStrideTmp[permSize - 1 - tilingData_->perm[permSize - 1 - i]];
    }

    uint32_t shift[8];
    uint32_t m[8];
    for (uint32_t i = 0; i < permSize; i++) {
        GetUintDivMagicAndShift(m[i], shift[i], outputShape[i]);
    }

    if (permSize == DIM_TWO) {
        asc_vf_call<SimtComputeDimTwo<T>>(dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
                                          (__gm__ volatile T*)(outputGM_.GetPhyAddr()), blkProcessNum, blkStartOffset,
                                          outputShape[0], outputShape[1], m[0], m[1], shift[0], shift[1]);
    } else if (permSize == DIM_THREE) {
        asc_vf_call<SimtComputeDimThree<T>>(dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
                                            (__gm__ volatile T*)(outputGM_.GetPhyAddr()), blkProcessNum, blkStartOffset,
                                            outputShape[0], outputShape[1], outputShape[2], dstStride[0], dstStride[1],
                                            dstStride[2], m[0], m[1], m[2], shift[0], shift[1], shift[2]);
    } else if (permSize == DIM_FOUR) {
        asc_vf_call<SimtComputeDimFour<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            blkProcessNum, blkStartOffset, outputShape[0], outputShape[1], outputShape[2], outputShape[3], dstStride[0],
            dstStride[1], dstStride[2], dstStride[3], m[0], m[1], m[2], m[3], shift[0], shift[1], shift[2], shift[3]);
    } else if (permSize == DIM_FIVE) {
        asc_vf_call<SimtComputeDimFive<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            blkProcessNum, blkStartOffset, outputShape[0], outputShape[1], outputShape[2], outputShape[3],
            outputShape[4], dstStride[0], dstStride[1], dstStride[2], dstStride[3], dstStride[4], m[0], m[1], m[2],
            m[3], m[4], shift[0], shift[1], shift[2], shift[3], shift[4]);
    } else if (permSize == DIM_SIX) {
        asc_vf_call<SimtComputeDimSix<T>>(dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
                                          (__gm__ volatile T*)(outputGM_.GetPhyAddr()), blkProcessNum, blkStartOffset,
                                          outputShape[0], outputShape[1], outputShape[2], outputShape[3],
                                          outputShape[4], outputShape[5], dstStride[0], dstStride[1], dstStride[2],
                                          dstStride[3], dstStride[4], dstStride[5], m[0], m[1], m[2], m[3], m[4], m[5],
                                          shift[0], shift[1], shift[2], shift[3], shift[4], shift[5]);
    } else if (permSize == DIM_SEVEN) {
        asc_vf_call<SimtComputeDimSeven<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            blkProcessNum, blkStartOffset, outputShape[0], outputShape[1], outputShape[2], outputShape[3],
            outputShape[4], outputShape[5], outputShape[6], dstStride[0], dstStride[1], dstStride[2], dstStride[3],
            dstStride[4], dstStride[5], dstStride[6], m[0], m[1], m[2], m[3], m[4], m[5], m[6], shift[0], shift[1],
            shift[2], shift[3], shift[4], shift[5], shift[6]);
    } else if (permSize == DIM_EIGHT) {
        asc_vf_call<SimtComputeDimEight<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            blkProcessNum, blkStartOffset, outputShape[0], outputShape[1], outputShape[2], outputShape[3],
            outputShape[4], outputShape[5], outputShape[6], outputShape[7], dstStride[0], dstStride[1], dstStride[2],
            dstStride[3], dstStride[4], dstStride[5], dstStride[6], dstStride[7], m[0], m[1], m[2], m[3], m[4], m[5],
            m[6], m[7], shift[0], shift[1], shift[2], shift[3], shift[4], shift[5], shift[6], shift[7]);
    }
}

} // namespace Transpose

#endif // TRANSPOSE_SMALL_SHAPE_H
