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
 * \file transpose_cut_one_axis.h
 * \brief transpose_cut_one_axis
 */
#ifndef TRANSPOSE_CUT_ONE_AXIS_H
#define TRANSPOSE_CUT_ONE_AXIS_H

#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;

template <typename T>
class TransposeCutOneAxis : public TransposeBase<T> {
public:
    __aicore__ inline TransposeCutOneAxis(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData();
    __aicore__ inline MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM> SetupLoopInfo(const int64_t inUbSrcShape[],
                                                                         const int64_t inUbDstShape[]);
    __aicore__ inline void ProcessMain(int64_t loopidxEnd);
    __aicore__ inline void ProcessTail();
    __aicore__ inline void GetLoopParams(int64_t n);
    __aicore__ inline void DecimalToMixed(int64_t num, int64_t bases[], int64_t mixedBase[]);
    __aicore__ inline void GetLoopAndStride();
    __aicore__ inline void CopyIn(int64_t loopIdx, MultiCopyParams<T, NDDMA_MAX_DIM_NUM>& params);
    __aicore__ inline void CopyOut(int64_t loopIdx, int64_t loopSize[], int64_t loopSrcStride[],
                                   int64_t loopDstStride[]);
    __aicore__ inline void ProcessPerCore();

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

    // blockCount and blockLen of copyOut
    int64_t mainBlockLen_ = 1;
    int64_t tailBlockLen_ = 1;
    int64_t blockCount_ = 1;
    int64_t dstStrideTotalLen_ = 1;

    // expand output init
    int64_t expandedOutputCutIndex_;
    int64_t expandedInputCutIndex_;

    // address offset
    int64_t srcLoopSize_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};
    int64_t dstLoopSize_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};
    int64_t srcLoopStride_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};
    int64_t dstLoopStride_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};
    int64_t srcAddressOffsetMixedBase_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};
    int64_t dstAddressOffsetMixedBase_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};

    // dataCopy params
    int64_t loopSizeMain_[NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1};
    int64_t loopSrcStrideMain_[NDDMA_MAX_DIM_NUM] = {1, 0, 0, 0, 0};
    int64_t loopDstStrideMain_[NDDMA_MAX_DIM_NUM] = {1, 0, 0, 0, 0};
    int64_t loopSizeTail_[NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1};
    int64_t loopSrcStrideTail_[NDDMA_MAX_DIM_NUM] = {1, 0, 0, 0, 0};
    int64_t loopDstStrideTail_[NDDMA_MAX_DIM_NUM] = {1, 0, 0, 0, 0};
};

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData,
                                                    TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;
    ParseTilingData();
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
    pipe->InitBuffer(vecQue_, 1, tiling_->ubSize);
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::Process()
{
    if (!ParseMultiCoreRange(blockIdx_, tiling_->realCoreNum, tiling_->blkFactor, tiling_->blkTailFactor,
                             blkProcessNum_, blkProcessIdxStart_, blkProcessIdxEnd_)) {
        return;
    }
    ProcessPerCore();
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::ParseTilingData()
{
    expandedOutputCutIndex_ = tiling_->outCutIndex + NDDMA_MAX_DIM_NUM - tiling_->permSize;
}

template <typename T>
__aicore__ inline MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM> TransposeCutOneAxis<T>::SetupLoopInfo(
    const int64_t inUbSrcShape[], const int64_t inUbDstShape[])
{
    MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM> loopInfo;
    int64_t loopDstStrideTmp[NDDMA_MAX_DIM_NUM] = {0};
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        int64_t recumMultiSrc = 1;
        int64_t recumMultiDst = 1;
        loopInfo.loopSize[i] = inUbSrcShape[i];
        for (int64_t j = i + 1; j < NDDMA_MAX_DIM_NUM; j++) {
            recumMultiSrc *= tiling_->expandedInputShape[j];
            recumMultiDst *= inUbDstShape[j];
        }
        loopInfo.loopSrcStride[i] = recumMultiSrc;
        loopDstStrideTmp[i] = recumMultiDst;
        loopInfo.loopLpSize[i] = 0;
        loopInfo.loopRpSize[i] = 0;
    }
    loopInfo.loopSrcStride[NDDMA_MAX_DIM_NUM - 1] = 1;
    loopDstStrideTmp[NDDMA_MAX_DIM_NUM - 1] = 1;

    int64_t alignDstStride = 1;
    if (expandedOutputCutIndex_ - 1 >= 0) {
        alignDstStride = (loopDstStrideTmp[expandedOutputCutIndex_ - 1] * sizeof(T) + BLOCK_SIZE_BYTE - 1) /
                         BLOCK_SIZE_BYTE;
        alignDstStride = alignDstStride * BLOCK_SIZE_BYTE / sizeof(T);
        loopDstStrideTmp[expandedOutputCutIndex_ - 1] = alignDstStride;
    }

    for (int64_t i = expandedOutputCutIndex_ - 2; i >= 0; i--) {
        loopDstStrideTmp[i] = inUbDstShape[i + 1] * loopDstStrideTmp[i + 1];
    }

    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        loopInfo.loopDstStride[tiling_->expandedPerm[i]] = loopDstStrideTmp[i];
    }
    this->reverseArray(loopInfo.loopSize);
    this->reverseArray(loopInfo.loopSrcStride);
    this->reverseArray(loopInfo.loopDstStride);

    return loopInfo;
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::DecimalToMixed(int64_t num, int64_t bases[], int64_t mixedBase[])
{
    if (num == 0) {
        return;
    }
    for (int64_t i = NDDMA_MAX_DIM_NUM - 1; i >= 0; i--) {
        mixedBase[i] = num % bases[i];
        num /= bases[i];
        if (num == 0) {
            break;
        }
    }
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::GetLoopParams(int64_t n)
{
    const int64_t reverseIdx = NDDMA_MAX_DIM_NUM - 1 - n;
    // main and tail loopSize
    loopSizeMain_[n] = tiling_->inUbMainDstShape[reverseIdx];
    loopSizeTail_[n] = tiling_->inUbTailDstShape[reverseIdx];
    // main loopSrcStride and loopDstStride
    if (n > 0) {
        if (n == NDDMA_MAX_DIM_NUM - expandedOutputCutIndex_) {
            loopSrcStrideMain_[n] = Ops::Base::CeilAlign(
                                        static_cast<int64_t>(tiling_->inUbMainDstShape[NDDMA_MAX_DIM_NUM - n] *
                                                             loopSrcStrideMain_[n - 1] * sizeof(T)),
                                        BLOCK_SIZE_BYTE) /
                                    sizeof(T);
        } else {
            loopSrcStrideMain_[n] = loopSrcStrideMain_[n - 1] * tiling_->inUbMainDstShape[NDDMA_MAX_DIM_NUM - n];
        }
        loopDstStrideMain_[n] = loopDstStrideMain_[n - 1] * tiling_->expandedOutputShape[NDDMA_MAX_DIM_NUM - n];
    }
    // tail loopSrcStride and loopDstStride
    if (tiling_->outTailFactor != 0) {
        if (n > 0) {
            if (n == NDDMA_MAX_DIM_NUM - expandedOutputCutIndex_) {
                loopSrcStrideTail_[n] = Ops::Base::CeilAlign(
                                            static_cast<int64_t>(tiling_->inUbTailDstShape[NDDMA_MAX_DIM_NUM - n] *
                                                                 loopSrcStrideTail_[n - 1] * sizeof(T)),
                                            BLOCK_SIZE_BYTE) /
                                        sizeof(T);
            } else {
                loopSrcStrideTail_[n] = loopSrcStrideTail_[n - 1] * tiling_->inUbTailDstShape[NDDMA_MAX_DIM_NUM - n];
            }
            loopDstStrideTail_[n] = loopDstStrideTail_[n - 1] * tiling_->expandedOutputShape[NDDMA_MAX_DIM_NUM - n];
        }
    }
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::GetLoopAndStride()
{
    // calculate the loopSize and stride of src and dst
    expandedInputCutIndex_ = tiling_->expandedPerm[expandedOutputCutIndex_];
    srcLoopStride_[NDDMA_MAX_DIM_NUM - 1] = 1;
    dstLoopStride_[NDDMA_MAX_DIM_NUM - 1] = 1;
    srcLoopStride_[NDDMA_MAX_DIM_NUM - 2] = tiling_->expandedInputShape[NDDMA_MAX_DIM_NUM - 1];
    dstLoopStride_[NDDMA_MAX_DIM_NUM - 2] = tiling_->expandedOutputShape[NDDMA_MAX_DIM_NUM - 1];
    for (int64_t i = NDDMA_MAX_DIM_NUM - 3; i >= 0; i--) {
        srcLoopStride_[i] = srcLoopStride_[i + 1] * tiling_->expandedInputShape[i + 1];
        dstLoopStride_[i] = dstLoopStride_[i + 1] * tiling_->expandedOutputShape[i + 1];
    }
    srcLoopStride_[expandedInputCutIndex_] *= tiling_->outUbFactor;
    dstLoopStride_[expandedOutputCutIndex_] *= tiling_->outUbFactor;

    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        srcLoopSize_[i] = Ops::Base::CeilDiv(tiling_->expandedInputShape[i], tiling_->inUbMainSrcShape[i]);
        dstLoopSize_[i] = Ops::Base::CeilDiv(tiling_->expandedOutputShape[i], tiling_->inUbMainDstShape[i]);
    }
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::CopyIn(int64_t loopIdx, MultiCopyParams<T, NDDMA_MAX_DIM_NUM>& params)
{
    int64_t srcAddressOffset = 0;
    DecimalToMixed(loopIdx, dstLoopSize_, dstAddressOffsetMixedBase_);
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        srcAddressOffsetMixedBase_[tiling_->expandedPerm[i]] = dstAddressOffsetMixedBase_[i];
        srcAddressOffset += srcAddressOffsetMixedBase_[tiling_->expandedPerm[i]] *
                            srcLoopStride_[tiling_->expandedPerm[i]];
    }
    LocalTensor<T> bindLocalIn = vecQue_.AllocTensor<T>();
    DataCopy<T, NDDMA_MAX_DIM_NUM, config>(bindLocalIn, inputGM_[srcAddressOffset], params);
    vecQue_.EnQue(bindLocalIn);
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::CopyOut(int64_t loopIdx, int64_t loopSize[], int64_t loopSrcStride[],
                                                       int64_t loopDstStride[])
{
    int64_t dstAddressOffset = 0;
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        dstAddressOffset += dstAddressOffsetMixedBase_[i] * dstLoopStride_[i];
    }
    // MTE3 parms
    DataCopyExtParams copyOutParams{1, 0, 0, 0, 0};
    LoopModeParams loopParams;

    copyOutParams.blockLen = sizeof(T);
    int64_t endIndex = 0;
    int64_t dstStride = 1;
    for (int64_t i = NDDMA_MAX_DIM_NUM - 1; i >= 0; i--) {
        copyOutParams.blockLen *= loopSize[NDDMA_MAX_DIM_NUM - 1 - i];
        if (tiling_->expandedOutputShape[i] != tiling_->inUbMainDstShape[i]) {
            endIndex = NDDMA_MAX_DIM_NUM - 1 - i;
            break;
        }
    }
    if (endIndex + 1 < NDDMA_MAX_DIM_NUM) {
        copyOutParams.blockCount = loopSize[endIndex + 1];
        copyOutParams.dstStride = copyOutParams.blockCount == 1 ?
                                      0 :
                                      (loopDstStride[endIndex + 1] - copyOutParams.blockLen / sizeof(T)) * sizeof(T);
    }
    loopParams.loop1Size = 1;
    if (endIndex + 2 < NDDMA_MAX_DIM_NUM) {
        loopParams.loop1Size = loopSize[endIndex + 2];
        loopParams.loop1SrcStride = loopSrcStride[endIndex + 2] * sizeof(T);
        loopParams.loop1DstStride = loopDstStride[endIndex + 2] * sizeof(T);
    }
    loopParams.loop2Size = 1;
    if (endIndex + 3 < NDDMA_MAX_DIM_NUM) {
        loopParams.loop2Size = loopSize[endIndex + 3];
        loopParams.loop2SrcStride = loopSrcStride[endIndex + 3] * sizeof(T);
        loopParams.loop2DstStride = loopDstStride[endIndex + 3] * sizeof(T);
    }
    LocalTensor<T> bindLocalOut = vecQue_.DeQue<T>();
    if (endIndex + 4 < NDDMA_MAX_DIM_NUM) {
        for (int64_t loop4Idx = 0; loop4Idx < loopSize[4]; loop4Idx++) {
            SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
            DataCopyPad(outputGM_[dstAddressOffset + loop4Idx * loopDstStride[4]],
                        bindLocalOut[loop4Idx * loopSrcStride[4]], copyOutParams);
            ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        }
    } else {
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyPad(outputGM_[dstAddressOffset], bindLocalOut, copyOutParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }
    vecQue_.FreeTensor(bindLocalOut);
}

template <typename T>
__aicore__ inline void TransposeCutOneAxis<T>::ProcessPerCore()
{
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        GetLoopParams(i);
    }
    GetLoopAndStride();

    // MTE2 params main
    T constValue = 0;
    MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM> loopInfoMain = SetupLoopInfo(tiling_->inUbMainSrcShape,
                                                                      tiling_->inUbMainDstShape);
    MultiCopyParams<T, NDDMA_MAX_DIM_NUM> paramsMain = {loopInfoMain, constValue};

    int64_t outCutLoopSize = Ops::Base::CeilDiv(tiling_->expandedOutputShape[expandedOutputCutIndex_],
                                                tiling_->outUbFactor);
    for (int64_t loopIdx = blkProcessIdxStart_; loopIdx < blkProcessIdxEnd_; loopIdx++) {
        if (tiling_->outTailFactor != 0 && (loopIdx + 1) % outCutLoopSize == 0) { // tail
            // MTE2 params tail
            MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM> loopInfoTail = SetupLoopInfo(tiling_->inUbTailSrcShape,
                                                                              tiling_->inUbTailDstShape);
            MultiCopyParams<T, NDDMA_MAX_DIM_NUM> paramsTail = {loopInfoTail, constValue};
            CopyIn(loopIdx, paramsTail);
            CopyOut(loopIdx, loopSizeTail_, loopSrcStrideTail_, loopDstStrideTail_);
        } else {
            CopyIn(loopIdx, paramsMain);
            CopyOut(loopIdx, loopSizeMain_, loopSrcStrideMain_, loopDstStrideMain_);
        }
    }
}
} // namespace Transpose

#endif // TRANSPOSE_CUT_ONE_AXIS
