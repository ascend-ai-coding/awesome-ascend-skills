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
 * \file transpose_big_dim.h
 * \brief transpose_big_dim
 */
#ifndef TRANSPOSE_BIG_DIM_H
#define TRANSPOSE_BIG_DIM_H

#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;

template <typename T>
class TransposeBigDim : public TransposeBase<T> {
public:
    __aicore__ inline TransposeBigDim(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DecimalToMixed(int64_t num, int64_t bases[], int64_t mixedBase[]);
    __aicore__ inline void SetLoopInfo(MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM>& loopInfo);
    __aicore__ inline void CopyIn(int64_t loopIdx, MultiCopyParams<T, NDDMA_MAX_DIM_NUM>& params);
    __aicore__ inline void CopyOut(int64_t loopIdx, DataCopyExtParams& copyOutParams);
    __aicore__ inline void ProcessPerCore(MultiCopyParams<T, NDDMA_MAX_DIM_NUM>& paramsMain,
                                          MultiCopyParams<T, NDDMA_MAX_DIM_NUM>& paramsTail,
                                          DataCopyExtParams& copyOutParamsMain, DataCopyExtParams& copyOutParamsTail);

private:
    int64_t blockIdx_;

    // buffer
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> vecQue_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    // tiling params
    const TransposeOpTilingData* tiling_;

    // core params
    int64_t blkProcessNum_ = 0;
    int64_t blkProcessIdxStart_ = 0;
    int64_t blkProcessIdxEnd_ = 0;

    // addressOffset params
    int64_t dstAddressOffsetMixedBase_[TRANSPOSE_MAX_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t dstAddressOffsetMixedBaseRes_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t dstLoopStride_[TRANSPOSE_MAX_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t srcLoopStride_[TRANSPOSE_MAX_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};

    int64_t cutIdx_ = 0;
};

template <typename T>
__aicore__ inline void TransposeBigDim<T>::Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData,
                                                TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
    pipe->InitBuffer(vecQue_, 1, tiling_->ubSize);
}

template <typename T>
__aicore__ inline void TransposeBigDim<T>::DecimalToMixed(int64_t num, int64_t bases[], int64_t mixedBase[])
{
    if (num == 0) {
        return;
    }
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        mixedBase[i] = num % bases[i];
        num /= bases[i];
        if (num == 0) {
            break;
        }
    }
}

template <typename T>
__aicore__ inline void TransposeBigDim<T>::SetLoopInfo(MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM>& loopInfo)
{
    int64_t nddmaFlag[NDDMA_MAX_DIM_NUM] = {0};
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        if (tiling_->nddmaIdx[i] == tiling_->perm[tiling_->outCutIndex]) {
            nddmaFlag[i] = -1;
        }
        for (int64_t j = tiling_->outCutIndex + 1; j < tiling_->permSize; j++) {
            if (tiling_->perm[j] == tiling_->nddmaIdx[i]) {
                nddmaFlag[i] = 1;
            }
        }
    }

    for (int64_t i = NDDMA_MAX_DIM_NUM - 1; i >= 0; i--) {
        if (nddmaFlag[NDDMA_MAX_DIM_NUM - 1 - i] == 1) {
            loopInfo.loopSize[i] = tiling_->inputShape[tiling_->nddmaIdx[NDDMA_MAX_DIM_NUM - 1 - i]];
            loopInfo.loopSrcStride[i] = tiling_->baseInShape[tiling_->nddmaIdx[NDDMA_MAX_DIM_NUM - 1 - i]];
            loopInfo.loopDstStride[i] = tiling_->baseNddmaShape[NDDMA_MAX_DIM_NUM - 1 - i];
        } else if (nddmaFlag[NDDMA_MAX_DIM_NUM - 1 - i] == -1) {
            cutIdx_ = i;
            loopInfo.loopSize[i] = tiling_->outUbFactor; // not tail
            loopInfo.loopSrcStride[i] = tiling_->baseInShape[tiling_->nddmaIdx[NDDMA_MAX_DIM_NUM - 1 - i]];
            loopInfo.loopDstStride[i] = tiling_->baseNddmaShape[NDDMA_MAX_DIM_NUM - 1 - i];
        } else {
            loopInfo.loopSize[i] = 1;
            loopInfo.loopSrcStride[i] = 1;
            loopInfo.loopDstStride[i] = tiling_->baseNddmaShape[NDDMA_MAX_DIM_NUM - 1 - i];
        }

        loopInfo.loopLpSize[i] = 0;
        loopInfo.loopRpSize[i] = 0;
    }
}

template <typename T>
__aicore__ inline void TransposeBigDim<T>::Process()
{
    if (!ParseMultiCoreRange(blockIdx_, tiling_->realCoreNum, tiling_->blkFactor, tiling_->blkTailFactor,
                             blkProcessNum_, blkProcessIdxStart_, blkProcessIdxEnd_)) {
        return;
    }

    MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM> loopInfo;
    SetLoopInfo(loopInfo);

    // calculate blockCount
    DataCopyExtParams copyOutParamsMain{1, 0, 0, 0, 0};
    copyOutParamsMain.blockLen = tiling_->totalNddmaNum * sizeof(T);

    DataCopyExtParams copyOutParamsTail{1, 0, 0, 0, 0};
    copyOutParamsTail.blockLen = tiling_->totalNddmaNum / tiling_->outUbFactor * tiling_->outTailFactor * sizeof(T);

    T constValue = 0;
    MultiCopyParams<T, NDDMA_MAX_DIM_NUM> paramsMain = {loopInfo, constValue};
    if (tiling_->outTailFactor != 0) {
        loopInfo.loopSize[cutIdx_] = tiling_->outTailFactor;
        loopInfo.loopDstStride[cutIdx_] = tiling_->baseNddmaShape[NDDMA_MAX_DIM_NUM - 1 - cutIdx_];
    }
    MultiCopyParams<T, NDDMA_MAX_DIM_NUM> paramsTail = {loopInfo, constValue};
    ProcessPerCore(paramsMain, paramsTail, copyOutParamsMain, copyOutParamsTail);
}

template <typename T>
__aicore__ inline void TransposeBigDim<T>::CopyIn(int64_t loopIdx, MultiCopyParams<T, NDDMA_MAX_DIM_NUM>& params)
{
    int64_t srcAddressOffset = 0;
    DecimalToMixed(loopIdx, dstAddressOffsetMixedBase_, dstAddressOffsetMixedBaseRes_);
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        srcAddressOffset += dstAddressOffsetMixedBaseRes_[i] *
                            srcLoopStride_[tiling_->permSize - 1 - tiling_->perm[tiling_->permSize - 1 - i]];
    }
    LocalTensor<T> bindLocalIn = vecQue_.AllocTensor<T>();
    DataCopy<T, NDDMA_MAX_DIM_NUM, config>(bindLocalIn, inputGM_[srcAddressOffset], params);
    vecQue_.EnQue(bindLocalIn);
}

template <typename T>
__aicore__ inline void TransposeBigDim<T>::CopyOut(int64_t loopIdx, DataCopyExtParams& copyOutParams)
{
    int64_t dstAddressOffset = 0;
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        dstAddressOffset += dstAddressOffsetMixedBaseRes_[i] * dstLoopStride_[i];
    }
    LocalTensor<T> bindLocalOut = vecQue_.DeQue<T>();
    DataCopyPad(outputGM_[dstAddressOffset], bindLocalOut, copyOutParams);
    vecQue_.FreeTensor(bindLocalOut);
}

template <typename T>
__aicore__ inline void TransposeBigDim<T>::ProcessPerCore(MultiCopyParams<T, NDDMA_MAX_DIM_NUM>& paramsMain,
                                                          MultiCopyParams<T, NDDMA_MAX_DIM_NUM>& paramsTail,
                                                          DataCopyExtParams& copyOutParamsMain,
                                                          DataCopyExtParams& copyOutParamsTail)
{
    int64_t outCutLoopSize = Ops::Base::CeilDiv(tiling_->outputShape[tiling_->outCutIndex], tiling_->outUbFactor);
    const int64_t permSize = tiling_->permSize;
    for (int64_t i = 0; i < permSize; i++) {
        if (permSize - 1 - i == tiling_->outCutIndex) {
            dstAddressOffsetMixedBase_[i] = outCutLoopSize;
        }
        if (permSize - 1 - i < tiling_->outCutIndex) {
            dstAddressOffsetMixedBase_[i] = tiling_->outputShape[permSize - 1 - i];
        }

        if (i >= 1) {
            dstLoopStride_[i] = dstLoopStride_[i - 1] * tiling_->outputShape[permSize - i];
            srcLoopStride_[i] = srcLoopStride_[i - 1] * tiling_->inputShape[permSize - i];
        }
    }
    dstLoopStride_[permSize - 1 - tiling_->outCutIndex] *= tiling_->outUbFactor;
    srcLoopStride_[permSize - 1 - tiling_->perm[tiling_->outCutIndex]] *= tiling_->outUbFactor;

    for (int64_t loopIdx = blkProcessIdxStart_; loopIdx < blkProcessIdxEnd_; loopIdx++) {
        if (tiling_->outTailFactor != 0 && (loopIdx + 1) % outCutLoopSize == 0) { // tail
            CopyIn(loopIdx, paramsTail);
            CopyOut(loopIdx, copyOutParamsTail);
        } else { // main
            CopyIn(loopIdx, paramsMain);
            CopyOut(loopIdx, copyOutParamsMain);
        }
    }
}

} // namespace Transpose

#endif // TRANSPOSE_BIG_DIM_H
