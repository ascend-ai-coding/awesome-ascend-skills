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
 * \file transpose_cut_two_axis.h
 * \brief transpose_cut_two_axis
 */
#ifndef TRANSPOSE_CUT_TWO_AXIS_H
#define TRANSPOSE_CUT_TWO_AXIS_H

#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;

template <typename T>
class TransposeCutTwoAxis : public TransposeBase<T> {
public:
    __aicore__ inline TransposeCutTwoAxis(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    int64_t blockIdx_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> vecQue_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    // tiling params
    const TransposeOpTilingData* tiling_;

    // expand input and output init
    int64_t expandedInputCutIndex_;
    int64_t expandedOutputCutIndex_;

    // core params
    int64_t blkProcessNum_ = 0;
    int64_t srcOffset_ = 0;

    // copyOut params
    int64_t mainLoopSize_[NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1};
    int64_t mainLoopSrcStride_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};
    int64_t inputTailLoopSize_[NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1};
    int64_t inputTailLoopSrcStride_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};
    int64_t outputTailLoopSize_[NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1};
    int64_t outputTailLoopSrcStride_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};
    int64_t tailLoopSize_[NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1};
    int64_t tailLoopSrcStride_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};
    int64_t loopDstStride_[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};

    int64_t inputTailSrcAddressOffsetBase_ = 1;
    int64_t inputTailDstAddressOffsetBase_ = 1;
    int64_t outputTailSrcAddressOffsetBase_ = 1;
    int64_t outputTailDstAddressOffsetBase_ = 1;

    int64_t inputOutputCutIndex_;
    int64_t outputInputCutIndex_;
    int64_t maxCutIndex_;
    int64_t minCutIndex_;
    int64_t outUbLoop_ = 1;

    struct Interval_ {
        int64_t start = 0;
        int64_t end = 0;
    };

    __aicore__ inline void ParseTilingData();
    __aicore__ inline int64_t FindPermIndex(int64_t j);
    __aicore__ inline void GetLoopAddressOffsetImpl(bool isSrc, int64_t loopNumArray[], int64_t loopShapeSizeArray[],
                                                    int64_t cutIndex1, int64_t cutIndex2, const int64_t expandedShape[],
                                                    const int64_t inUbShape[]);
    __aicore__ inline void GetMainLoopAddressOffset(bool isSrc, int64_t loopNumArray[], int64_t loopShapeSizeArray[],
                                                    const int64_t expandedShape[], const int64_t inUbShape[]);
    __aicore__ inline void GetTailLoopAddressOffset(bool isSrc, int64_t loopNumArray[], int64_t loopShapeSizeArray[],
                                                    int64_t cutIndex, const int64_t expandedShape[],
                                                    const int64_t inUbShape[]);
    __aicore__ inline void GetTailTailLoopAddressOffset(bool isSrc, int64_t loopNumArray[],
                                                        int64_t loopShapeSizeArray[], int64_t inputCutIndex,
                                                        int64_t outputCutIndex, const int64_t expandedShape[],
                                                        const int64_t inUbShape[]);
    __aicore__ inline MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM> SetupLoopInfo(const int64_t inUbSrcShape[],
                                                                         const int64_t inUbDstShape[]);
    __aicore__ inline void ProcessMain(int64_t loopidxStart, int64_t loopidxEnd);
    __aicore__ inline void ProcessOutputTail(int64_t loopidxStart, int64_t loopidxEnd);
    __aicore__ inline void ProcessInputTail(int64_t loopidxStart, int64_t loopidxEnd);
    __aicore__ inline void ProcessTail(int64_t loopidxStart, int64_t loopidxEnd);
    __aicore__ inline void PorcessOffsetRange(int64_t start, int64_t end, Interval_& offsetRangeRes);
    __aicore__ inline void GetCopyOutParams();
    __aicore__ inline void GetTailAddressOffsetBase();
    __aicore__ inline void ProcessPerCore();
    __aicore__ inline void ProcessBlock();
};

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData,
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
__aicore__ inline void TransposeCutTwoAxis<T>::Process()
{
    if (blockIdx_ >= tiling_->realCoreNum) {
        return;
    }

    blkProcessNum_ = tiling_->blkFactor;
    srcOffset_ = blockIdx_ * tiling_->blkFactor;
    if (blockIdx_ < tiling_->blkTailFactor) {
        blkProcessNum_ += 1;
        srcOffset_ += blockIdx_;
    } else {
        srcOffset_ += tiling_->blkTailFactor;
    }

    ProcessPerCore();
}

__aicore__ inline void DecimalToMixedBase(int64_t num, int64_t bases[], int64_t mixedBase[])
{
    if (num == 0) {
        return;
    }
    for (int64_t i = 0; i < TRANSPOSE_MAX_AXIS_NUM; ++i) {
        mixedBase[i] = num % bases[i];
        num /= bases[i];
        if (num == 0) {
            break;
        }
    }
}

__aicore__ inline int64_t AlignToMultiple(int64_t a, int64_t b)
{
    if (b <= 0) {
        return a;
    }
    return CeilDivision(a, b) * b;
}

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::PorcessOffsetRange(int64_t start, int64_t end, Interval_& offsetRangeRes)
{
    if ((srcOffset_ < start) && (srcOffset_ + blkProcessNum_ - 1 > end)) {
        return;
    }
    if ((srcOffset_ >= start) && (srcOffset_ <= end)) {
        offsetRangeRes.start = srcOffset_;
        if ((srcOffset_ + blkProcessNum_ - 1 >= start) && (srcOffset_ + blkProcessNum_ - 1 <= end)) {
            offsetRangeRes.end = srcOffset_ + blkProcessNum_ - 1;
        }
    } else {
        if ((srcOffset_ + blkProcessNum_ - 1 >= start) && (srcOffset_ + blkProcessNum_ - 1 <= end)) {
            offsetRangeRes.end = srcOffset_ + blkProcessNum_ - 1;
        } else {
            offsetRangeRes.start = 0;
            offsetRangeRes.end = 0;
        }
    }
}

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::ParseTilingData()
{
    expandedOutputCutIndex_ = tiling_->outCutIndex + NDDMA_MAX_DIM_NUM - tiling_->permSize;
    expandedInputCutIndex_ = tiling_->inCutIndex + NDDMA_MAX_DIM_NUM - tiling_->permSize;
}

template <typename T>
__aicore__ inline int64_t TransposeCutTwoAxis<T>::FindPermIndex(int64_t j)
{
    for (int64_t permIndex = 0; permIndex < NDDMA_MAX_DIM_NUM; permIndex++) {
        if (tiling_->expandedPerm[permIndex] == j) {
            return permIndex;
        }
    }
    return j;
}

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::GetLoopAddressOffsetImpl(bool isSrc, int64_t loopNumArray[],
                                                                        int64_t loopShapeSizeArray[], int64_t cutIndex1,
                                                                        int64_t cutIndex2,
                                                                        const int64_t expandedShape[],
                                                                        const int64_t inUbShape[])
{
    int64_t startIndex = NDDMA_MAX_DIM_NUM - 1;
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        for (int64_t j = startIndex; j >= 0; j--) {
            int64_t idx = j;
            if (!isSrc) {
                idx = FindPermIndex(j);
            }
            if (idx == cutIndex1 || idx == cutIndex2) {
                continue;
            }
            int64_t loopNum = expandedShape[idx] / inUbShape[idx];
            int64_t loopSize = inUbShape[idx];
            for (int64_t k = idx + 1; k < NDDMA_MAX_DIM_NUM; k++) {
                loopSize *= expandedShape[k];
            }
            if (loopNum > 1) {
                startIndex = j - 1;
                loopNumArray[i] = loopNum;
                loopShapeSizeArray[i] = loopSize;
                break;
            }
        }
    }
}

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::GetMainLoopAddressOffset(bool isSrc, int64_t loopNumArray[],
                                                                        int64_t loopShapeSizeArray[],
                                                                        const int64_t expandedShape[],
                                                                        const int64_t inUbShape[])
{
    GetLoopAddressOffsetImpl(isSrc, loopNumArray, loopShapeSizeArray, -1, -1, expandedShape, inUbShape);
}

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::GetTailLoopAddressOffset(bool isSrc, int64_t loopNumArray[],
                                                                        int64_t loopShapeSizeArray[], int64_t cutIndex,
                                                                        const int64_t expandedShape[],
                                                                        const int64_t inUbShape[])
{
    GetLoopAddressOffsetImpl(isSrc, loopNumArray, loopShapeSizeArray, cutIndex, -1, expandedShape, inUbShape);
}

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::GetTailTailLoopAddressOffset(
    bool isSrc, int64_t loopNumArray[], int64_t loopShapeSizeArray[], int64_t inputCutIndex, int64_t outputCutIndex,
    const int64_t expandedShape[], const int64_t inUbShape[])
{
    GetLoopAddressOffsetImpl(isSrc, loopNumArray, loopShapeSizeArray, inputCutIndex, outputCutIndex, expandedShape,
                             inUbShape);
}

template <typename T>
__aicore__ inline MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM> TransposeCutTwoAxis<T>::SetupLoopInfo(
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
    if (maxCutIndex_ - 1 >= 0) {
        alignDstStride = (loopDstStrideTmp[maxCutIndex_ - 1] * sizeof(T) + BLOCK_SIZE_BYTE - 1) / BLOCK_SIZE_BYTE;
        alignDstStride = alignDstStride * BLOCK_SIZE_BYTE / sizeof(T);
        loopDstStrideTmp[maxCutIndex_ - 1] = alignDstStride;
    } else {
        alignDstStride = (loopDstStrideTmp[0] * sizeof(T) + BLOCK_SIZE_BYTE - 1) / BLOCK_SIZE_BYTE;
        alignDstStride = alignDstStride * BLOCK_SIZE_BYTE / sizeof(T);
        loopDstStrideTmp[0] = alignDstStride;
    }

    for (int64_t i = maxCutIndex_ - 2; i >= 0; i--) {
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
__aicore__ inline void TransposeCutTwoAxis<T>::ProcessMain(int64_t loopidxStart, int64_t loopidxEnd)
{
    // main
    T constValue = 0;
    MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM> loopInfoMain = SetupLoopInfo(tiling_->inUbMainSrcShape,
                                                                      tiling_->inUbMainDstShape);
    MultiCopyParams<T, NDDMA_MAX_DIM_NUM> paramsMain = {loopInfoMain, constValue};

    DataCopyExtParams copyOutParamsMain;
    copyOutParamsMain.blockLen = mainLoopSize_[0] * sizeof(T);
    copyOutParamsMain.blockCount = mainLoopSize_[1];
    copyOutParamsMain.srcStride = 0;
    copyOutParamsMain.dstStride = (loopDstStride_[1] - mainLoopSize_[0]) * sizeof(T);

    LoopModeParams loopParams;
    loopParams.loop1Size = mainLoopSize_[2];
    loopParams.loop1SrcStride = mainLoopSrcStride_[2] * sizeof(T);
    loopParams.loop1DstStride = loopDstStride_[2] * sizeof(T);
    loopParams.loop2Size = mainLoopSize_[3];
    loopParams.loop2SrcStride = mainLoopSrcStride_[3] * sizeof(T);
    loopParams.loop2DstStride = loopDstStride_[3] * sizeof(T);

    int64_t srcLoopNumArray[NDDMA_MAX_DIM_NUM] = {0};
    int64_t srcLoopShapeSizeArray[NDDMA_MAX_DIM_NUM] = {0};
    GetMainLoopAddressOffset(true, srcLoopNumArray, srcLoopShapeSizeArray, tiling_->expandedInputShape,
                             tiling_->inUbMainSrcShape);

    int64_t dstLoopNumArray[NDDMA_MAX_DIM_NUM] = {0};
    int64_t dstLoopShapeSizeArray[NDDMA_MAX_DIM_NUM] = {0};
    GetMainLoopAddressOffset(false, dstLoopNumArray, dstLoopShapeSizeArray, tiling_->expandedOutputShape,
                             tiling_->inUbMainDstShape);

    // multicore cycle align main blocks
    for (int64_t loopidx = loopidxStart; loopidx <= loopidxEnd; loopidx++) {
        int64_t srcAddressOffset = 0;
        int64_t dstAddressOffset = 0;
        int64_t ubAddressOffset = 0;

        int64_t srcAddressOffsetMixedBase[NDDMA_MAX_DIM_NUM] = {0};
        DecimalToMixedBase(loopidx, srcLoopNumArray, srcAddressOffsetMixedBase);

        int64_t dstAddressOffsetMixedBase[NDDMA_MAX_DIM_NUM] = {0};
        DecimalToMixedBase(loopidx, dstLoopNumArray, dstAddressOffsetMixedBase);
        for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
            if (srcLoopShapeSizeArray[i] == 0) {
                break;
            }
            srcAddressOffset += srcAddressOffsetMixedBase[i] * srcLoopShapeSizeArray[i];
            dstAddressOffset += dstAddressOffsetMixedBase[i] * dstLoopShapeSizeArray[i];
        }
        LocalTensor<T> bindLocalIn = vecQue_.AllocTensor<T>();
        DataCopy<T, NDDMA_MAX_DIM_NUM, config>(bindLocalIn, inputGM_[srcAddressOffset], paramsMain);
        vecQue_.EnQue(bindLocalIn);
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        LocalTensor<T> bindLocalOut = vecQue_.DeQue<T>();
        for (int64_t i = 0; i < mainLoopSize_[4]; i++) {
            DataCopyPad(outputGM_[dstAddressOffset + i * loopDstStride_[4]], bindLocalOut[i * mainLoopSrcStride_[4]],
                        copyOutParamsMain);
        }
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        vecQue_.FreeTensor(bindLocalOut);
    }
}

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::ProcessInputTail(int64_t loopidxStart, int64_t loopidxEnd)
{
    // NDDMA loopInfo init
    T constValue = 0;
    MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM> loopInfoInputTail = SetupLoopInfo(tiling_->inUbInputTailSrcShape,
                                                                           tiling_->inUbInputTailDstShape);
    MultiCopyParams<T, NDDMA_MAX_DIM_NUM> paramsInputTail = {loopInfoInputTail, constValue};

    DataCopyExtParams copyOutParamsInputTail;
    copyOutParamsInputTail.blockLen = inputTailLoopSize_[0] * sizeof(T);
    copyOutParamsInputTail.blockCount = inputTailLoopSize_[1];
    copyOutParamsInputTail.srcStride = 0;
    copyOutParamsInputTail.dstStride = (loopDstStride_[1] - inputTailLoopSize_[0]) * sizeof(T);

    LoopModeParams loopParams;
    loopParams.loop1Size = inputTailLoopSize_[2];
    loopParams.loop1SrcStride = inputTailLoopSrcStride_[2] * sizeof(T);
    loopParams.loop1DstStride = loopDstStride_[2] * sizeof(T);
    loopParams.loop2Size = inputTailLoopSize_[3];
    loopParams.loop2SrcStride = inputTailLoopSrcStride_[3] * sizeof(T);
    loopParams.loop2DstStride = loopDstStride_[3] * sizeof(T);

    // inputTail
    int64_t inputTailSrcLoopNumArray[NDDMA_MAX_DIM_NUM] = {0};
    int64_t inputTailSrcLoopShapeSizeArray[NDDMA_MAX_DIM_NUM] = {0};
    GetTailLoopAddressOffset(true, inputTailSrcLoopNumArray, inputTailSrcLoopShapeSizeArray, expandedInputCutIndex_,
                             tiling_->expandedInputShape, tiling_->inUbInputTailSrcShape);

    int64_t inputTailDstLoopNumArray[NDDMA_MAX_DIM_NUM] = {0};
    int64_t inputTailDstLoopShapeSizeArray[NDDMA_MAX_DIM_NUM] = {0};
    GetTailLoopAddressOffset(false, inputTailDstLoopNumArray, inputTailDstLoopShapeSizeArray, outputInputCutIndex_,
                             tiling_->expandedOutputShape, tiling_->inUbInputTailDstShape);

    for (int64_t loopidx = loopidxStart; loopidx <= loopidxEnd; loopidx++) {
        int64_t srcAddressOffset = inputTailSrcAddressOffsetBase_;
        int64_t dstAddressOffset = inputTailDstAddressOffsetBase_;
        int64_t ubAddressOffset = 0;

        int64_t srcAddressOffsetMixedBase[NDDMA_MAX_DIM_NUM] = {0};
        DecimalToMixedBase(loopidx - ((tiling_->expandedInputShape[expandedInputCutIndex_] /
                                       tiling_->inUbMainSrcShape[expandedInputCutIndex_]) *
                                      (tiling_->expandedInputShape[inputOutputCutIndex_] /
                                       tiling_->inUbMainSrcShape[inputOutputCutIndex_]) *
                                      outUbLoop_),
                           inputTailSrcLoopNumArray, srcAddressOffsetMixedBase);

        int64_t dstAddressOffsetMixedBase[NDDMA_MAX_DIM_NUM] = {0};
        DecimalToMixedBase(loopidx - ((tiling_->expandedInputShape[expandedInputCutIndex_] /
                                       tiling_->inUbMainSrcShape[expandedInputCutIndex_]) *
                                      (tiling_->expandedInputShape[inputOutputCutIndex_] /
                                       tiling_->inUbMainSrcShape[inputOutputCutIndex_]) *
                                      outUbLoop_),
                           inputTailDstLoopNumArray, dstAddressOffsetMixedBase);
        for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
            if (inputTailSrcLoopShapeSizeArray[i] == 0) {
                break;
            }
            srcAddressOffset += srcAddressOffsetMixedBase[i] * inputTailSrcLoopShapeSizeArray[i];
            dstAddressOffset += dstAddressOffsetMixedBase[i] * inputTailDstLoopShapeSizeArray[i];
        }
        LocalTensor<T> bindLocalIn = vecQue_.AllocTensor<T>();
        DataCopy<T, NDDMA_MAX_DIM_NUM, config>(bindLocalIn, inputGM_[srcAddressOffset], paramsInputTail);
        vecQue_.EnQue(bindLocalIn);
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        LocalTensor<T> bindLocalOut = vecQue_.DeQue<T>();
        for (int64_t i = 0; i < inputTailLoopSize_[4]; i++) {
            DataCopyPad(outputGM_[dstAddressOffset + i * loopDstStride_[4]],
                        bindLocalOut[i * inputTailLoopSrcStride_[4]], copyOutParamsInputTail);
        }
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        vecQue_.FreeTensor(bindLocalOut);
    }
}

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::ProcessOutputTail(int64_t loopidxStart, int64_t loopidxEnd)
{
    // NDDMA loopInfo init
    T constValue = 0;
    MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM> loopInfoOutputTail = SetupLoopInfo(tiling_->inUbOutputTailSrcShape,
                                                                            tiling_->inUbOutputTailDstShape);
    MultiCopyParams<T, NDDMA_MAX_DIM_NUM> paramsOutputTail = {loopInfoOutputTail, constValue};

    DataCopyExtParams copyOutParamsOutputTail;
    copyOutParamsOutputTail.blockLen = outputTailLoopSize_[0] * sizeof(T);
    copyOutParamsOutputTail.blockCount = outputTailLoopSize_[1];
    copyOutParamsOutputTail.srcStride = 0;
    copyOutParamsOutputTail.dstStride = (loopDstStride_[1] - outputTailLoopSize_[0]) * sizeof(T);

    LoopModeParams loopParams;
    loopParams.loop1Size = outputTailLoopSize_[2];
    loopParams.loop1SrcStride = outputTailLoopSrcStride_[2] * sizeof(T);
    loopParams.loop1DstStride = loopDstStride_[2] * sizeof(T);
    loopParams.loop2Size = outputTailLoopSize_[3];
    loopParams.loop2SrcStride = outputTailLoopSrcStride_[3] * sizeof(T);
    loopParams.loop2DstStride = loopDstStride_[3] * sizeof(T);

    // outputTail
    int64_t outputTailSrcLoopNumArray[NDDMA_MAX_DIM_NUM] = {0};
    int64_t outputTailSrcLoopShapeSizeArray[NDDMA_MAX_DIM_NUM] = {0};
    GetTailLoopAddressOffset(true, outputTailSrcLoopNumArray, outputTailSrcLoopShapeSizeArray, inputOutputCutIndex_,
                             tiling_->expandedInputShape, tiling_->inUbOutputTailSrcShape);

    int64_t outputTailDstLoopNumArray[NDDMA_MAX_DIM_NUM] = {0};
    int64_t outputTailDstLoopShapeSizeArray[NDDMA_MAX_DIM_NUM] = {0};
    GetTailLoopAddressOffset(false, outputTailDstLoopNumArray, outputTailDstLoopShapeSizeArray, expandedOutputCutIndex_,
                             tiling_->expandedOutputShape, tiling_->inUbOutputTailDstShape);

    int64_t idxOffset = 0;
    if (tiling_->inTailFactor == 0) {
        idxOffset = (tiling_->expandedInputShape[expandedInputCutIndex_] /
                     tiling_->inUbMainSrcShape[expandedInputCutIndex_]) *
                    (tiling_->expandedInputShape[inputOutputCutIndex_] /
                     tiling_->inUbMainSrcShape[inputOutputCutIndex_]) *
                    outUbLoop_;
    } else {
        idxOffset = (tiling_->expandedInputShape[expandedInputCutIndex_] /
                     tiling_->inUbMainSrcShape[expandedInputCutIndex_]) *
                        (tiling_->expandedInputShape[inputOutputCutIndex_] /
                         tiling_->inUbMainSrcShape[inputOutputCutIndex_]) *
                        outUbLoop_ +
                    (tiling_->expandedInputShape[inputOutputCutIndex_] /
                     tiling_->inUbMainSrcShape[inputOutputCutIndex_]) *
                        outUbLoop_;
    }

    for (int64_t loopidx = loopidxStart; loopidx <= loopidxEnd; loopidx++) {
        int64_t srcAddressOffset = outputTailSrcAddressOffsetBase_;
        int64_t dstAddressOffset = outputTailDstAddressOffsetBase_;
        int64_t ubAddressOffset = 0;

        int64_t srcAddressOffsetMixedBase[NDDMA_MAX_DIM_NUM] = {0};
        DecimalToMixedBase(loopidx - idxOffset, outputTailSrcLoopNumArray, srcAddressOffsetMixedBase);
        int64_t dstAddressOffsetMixedBase[NDDMA_MAX_DIM_NUM] = {0};
        DecimalToMixedBase(loopidx - idxOffset, outputTailDstLoopNumArray, dstAddressOffsetMixedBase);

        for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
            if (outputTailSrcLoopShapeSizeArray[i] == 0) {
                break;
            }
            srcAddressOffset += srcAddressOffsetMixedBase[i] * outputTailSrcLoopShapeSizeArray[i];
            dstAddressOffset += dstAddressOffsetMixedBase[i] * outputTailDstLoopShapeSizeArray[i];
        }
        LocalTensor<T> bindLocalIn = vecQue_.AllocTensor<T>();
        DataCopy<T, NDDMA_MAX_DIM_NUM, config>(bindLocalIn, inputGM_[srcAddressOffset], paramsOutputTail);
        vecQue_.EnQue(bindLocalIn);
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        LocalTensor<T> bindLocalOut = vecQue_.DeQue<T>();
        for (int64_t i = 0; i < outputTailLoopSize_[4]; i++) {
            DataCopyPad(outputGM_[dstAddressOffset + i * loopDstStride_[4]],
                        bindLocalOut[i * outputTailLoopSrcStride_[4]], copyOutParamsOutputTail);
        }
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        vecQue_.FreeTensor(bindLocalOut);
    }
}

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::ProcessTail(int64_t loopidxStart, int64_t loopidxEnd)
{
    // NDDMA loopInfo init
    T constValue = 0;
    MultiCopyLoopInfo<NDDMA_MAX_DIM_NUM> loopInfoTail = SetupLoopInfo(tiling_->inUbTailSrcShape,
                                                                      tiling_->inUbTailDstShape);
    MultiCopyParams<T, NDDMA_MAX_DIM_NUM> paramsTail = {loopInfoTail, constValue};

    DataCopyExtParams copyOutParamsTail;
    copyOutParamsTail.blockLen = tailLoopSize_[0] * sizeof(T);
    copyOutParamsTail.blockCount = tailLoopSize_[1];
    copyOutParamsTail.srcStride = 0;
    copyOutParamsTail.dstStride = (loopDstStride_[1] - tailLoopSize_[0]) * sizeof(T);

    LoopModeParams loopParams;
    loopParams.loop1Size = tailLoopSize_[2];
    loopParams.loop1SrcStride = tailLoopSrcStride_[2] * sizeof(T);
    loopParams.loop1DstStride = loopDstStride_[2] * sizeof(T);
    loopParams.loop2Size = tailLoopSize_[3];
    loopParams.loop2SrcStride = tailLoopSrcStride_[3] * sizeof(T);
    loopParams.loop2DstStride = loopDstStride_[3] * sizeof(T);

    int64_t tailSrcAddressOffsetBase = inputTailSrcAddressOffsetBase_ + outputTailSrcAddressOffsetBase_;
    int64_t tailDstAddressOffsetBase = inputTailDstAddressOffsetBase_ + outputTailDstAddressOffsetBase_;
    int64_t ubAddressOffset = 0;

    int64_t tailSrcLoopNumArray[NDDMA_MAX_DIM_NUM] = {0};
    int64_t tailSrcLoopShapeSizeArray[NDDMA_MAX_DIM_NUM] = {0};
    GetTailTailLoopAddressOffset(true, tailSrcLoopNumArray, tailSrcLoopShapeSizeArray, expandedInputCutIndex_,
                                 inputOutputCutIndex_, tiling_->expandedInputShape, tiling_->inUbTailSrcShape);

    int64_t tailDstLoopNumArray[NDDMA_MAX_DIM_NUM] = {0};
    int64_t tailDstLoopShapeSizeArray[NDDMA_MAX_DIM_NUM] = {0};
    GetTailTailLoopAddressOffset(false, tailDstLoopNumArray, tailDstLoopShapeSizeArray, expandedOutputCutIndex_,
                                 outputInputCutIndex_, tiling_->expandedOutputShape, tiling_->inUbTailDstShape);

    for (int64_t loopidx = loopidxStart; loopidx <= loopidxEnd; loopidx++) {
        int64_t srcAddressOffset = tailSrcAddressOffsetBase;
        int64_t dstAddressOffset = tailDstAddressOffsetBase;

        int64_t tailLoopLocalIdx = loopidx -
                                   ((tiling_->expandedInputShape[expandedInputCutIndex_] /
                                     tiling_->inUbMainSrcShape[expandedInputCutIndex_]) *
                                    (tiling_->expandedInputShape[inputOutputCutIndex_] /
                                     tiling_->inUbMainSrcShape[inputOutputCutIndex_]) *
                                    outUbLoop_) -
                                   ((tiling_->expandedInputShape[expandedInputCutIndex_] /
                                     tiling_->inUbMainSrcShape[expandedInputCutIndex_]) *
                                    outUbLoop_) -
                                   ((tiling_->expandedInputShape[inputOutputCutIndex_] /
                                     tiling_->inUbMainSrcShape[inputOutputCutIndex_]) *
                                    outUbLoop_);

        int64_t srcAddressOffsetMixedBase[NDDMA_MAX_DIM_NUM] = {0};
        DecimalToMixedBase(tailLoopLocalIdx, tailSrcLoopNumArray, srcAddressOffsetMixedBase);

        int64_t dstAddressOffsetMixedBase[NDDMA_MAX_DIM_NUM] = {0};
        DecimalToMixedBase(tailLoopLocalIdx, tailDstLoopNumArray, dstAddressOffsetMixedBase);
        for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
            if (tailSrcLoopShapeSizeArray[i] == 0) {
                break;
            }
            srcAddressOffset += srcAddressOffsetMixedBase[i] * tailSrcLoopShapeSizeArray[i];
            dstAddressOffset += dstAddressOffsetMixedBase[i] * tailDstLoopShapeSizeArray[i];
        }
        LocalTensor<T> bindLocalIn = vecQue_.AllocTensor<T>();
        DataCopy<T, NDDMA_MAX_DIM_NUM, config>(bindLocalIn, inputGM_[srcAddressOffset], paramsTail);
        vecQue_.EnQue(bindLocalIn);
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        LocalTensor<T> bindLocalOut = vecQue_.DeQue<T>();
        for (int64_t i = 0; i < tailLoopSize_[4]; i++) {
            DataCopyPad(outputGM_[dstAddressOffset + i * loopDstStride_[4]], bindLocalOut[i * tailLoopSrcStride_[4]],
                        copyOutParamsTail);
        }
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        vecQue_.FreeTensor(bindLocalOut);
    }
}

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::GetCopyOutParams()
{
    // calculate the main and tail loopSize and loopSrc/DstStirde of copyOut
    int64_t idx = 0;
    int64_t loopSize = 1;
    int64_t cutIdx[NDDMA_MAX_DIM_NUM] = {-1, -1, -1, -1, -1};
    for (int64_t i = NDDMA_MAX_DIM_NUM - 1; i >= 0; i--) {
        if (tiling_->inUbMainDstShape[i] != tiling_->expandedOutputShape[i]) {
            mainLoopSize_[idx] = loopSize * tiling_->inUbMainDstShape[i];
            inputTailLoopSize_[idx] = loopSize * tiling_->inUbInputTailDstShape[i];
            outputTailLoopSize_[idx] = loopSize * tiling_->inUbOutputTailDstShape[i];
            tailLoopSize_[idx] = loopSize * tiling_->inUbTailDstShape[i];
            cutIdx[idx] = NDDMA_MAX_DIM_NUM - 1 - i;
            idx++;
            loopSize = 1;
        } else {
            loopSize *= tiling_->inUbMainDstShape[i];
        }
        if (i == 0) {
            mainLoopSize_[idx] = loopSize;
            inputTailLoopSize_[idx] = loopSize;
            outputTailLoopSize_[idx] = loopSize;
            tailLoopSize_[idx] = loopSize;
        }
    }
    mainLoopSrcStride_[1] = Ops::Base::CeilDiv(static_cast<int64_t>(mainLoopSize_[0] * sizeof(T)), BLOCK_SIZE_BYTE) *
                            BLOCK_SIZE_BYTE / sizeof(T);
    inputTailLoopSrcStride_[1] = Ops::Base::CeilDiv(static_cast<int64_t>(inputTailLoopSize_[0] * sizeof(T)),
                                                    BLOCK_SIZE_BYTE) *
                                 BLOCK_SIZE_BYTE / sizeof(T);
    outputTailLoopSrcStride_[1] = Ops::Base::CeilDiv(static_cast<int64_t>(outputTailLoopSize_[0] * sizeof(T)),
                                                     BLOCK_SIZE_BYTE) *
                                  BLOCK_SIZE_BYTE / sizeof(T);
    tailLoopSrcStride_[1] = Ops::Base::CeilDiv(static_cast<int64_t>(tailLoopSize_[0] * sizeof(T)), BLOCK_SIZE_BYTE) *
                            BLOCK_SIZE_BYTE / sizeof(T);
    for (int64_t i = 2; i < NDDMA_MAX_DIM_NUM; i++) {
        mainLoopSrcStride_[i] = mainLoopSize_[i - 1] * mainLoopSrcStride_[i - 1];
        inputTailLoopSrcStride_[i] = inputTailLoopSize_[i - 1] * inputTailLoopSrcStride_[i - 1];
        outputTailLoopSrcStride_[i] = outputTailLoopSize_[i - 1] * outputTailLoopSrcStride_[i - 1];
        tailLoopSrcStride_[i] = tailLoopSize_[i - 1] * tailLoopSrcStride_[i - 1];
    }

    int64_t loopDstStride[NDDMA_MAX_DIM_NUM] = {0, 0, 0, 0, 0};
    loopDstStride[1] = tiling_->expandedOutputShape[NDDMA_MAX_DIM_NUM - 1];
    for (int64_t i = 2; i < NDDMA_MAX_DIM_NUM; i++) {
        loopDstStride[i] = tiling_->expandedOutputShape[NDDMA_MAX_DIM_NUM - i] * loopDstStride[i - 1];
    }
    idx = 1;
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        if (cutIdx[i] < 0) {
            break;
        }
        if (cutIdx[i] + 1 >= NDDMA_MAX_DIM_NUM) {
            break;
        }
        loopDstStride_[idx] = loopDstStride[cutIdx[i] + 1];
        idx++;
    }
}

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::GetTailAddressOffsetBase()
{
    // inputTail base address offset
    inputTailSrcAddressOffsetBase_ = 1;
    for (int64_t i = NDDMA_MAX_DIM_NUM - 1; i >= expandedInputCutIndex_; i--) {
        if (i == expandedInputCutIndex_) {
            inputTailSrcAddressOffsetBase_ *= (tiling_->expandedInputShape[i] - tiling_->inTailFactor);
        } else {
            inputTailSrcAddressOffsetBase_ *= tiling_->expandedInputShape[i];
        }
    }
    inputTailDstAddressOffsetBase_ = 1;
    for (int64_t i = NDDMA_MAX_DIM_NUM - 1; i >= outputInputCutIndex_; i--) {
        if (i == outputInputCutIndex_) {
            inputTailDstAddressOffsetBase_ *= (tiling_->expandedOutputShape[i] - tiling_->inTailFactor);
        } else {
            inputTailDstAddressOffsetBase_ *= tiling_->expandedOutputShape[i];
        }
    }

    // outputTail base address offset
    outputTailSrcAddressOffsetBase_ = 1;
    for (int64_t i = NDDMA_MAX_DIM_NUM - 1; i >= inputOutputCutIndex_; i--) {
        if (i == inputOutputCutIndex_) {
            outputTailSrcAddressOffsetBase_ *= (tiling_->expandedInputShape[i] - tiling_->outTailFactor);
        } else {
            outputTailSrcAddressOffsetBase_ *= tiling_->expandedInputShape[i];
        }
    }
    outputTailDstAddressOffsetBase_ = 1;
    for (int64_t i = NDDMA_MAX_DIM_NUM - 1; i >= expandedOutputCutIndex_; i--) {
        if (i == expandedOutputCutIndex_) {
            outputTailDstAddressOffsetBase_ *= (tiling_->expandedOutputShape[i] - tiling_->outTailFactor);
        } else {
            outputTailDstAddressOffsetBase_ *= tiling_->expandedOutputShape[i];
        }
    }
}

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::ProcessBlock()
{
    // 初始化四个结果数据区间
    Interval_ offsetRangeMainRes;
    offsetRangeMainRes.start = 0;
    offsetRangeMainRes.end = tiling_->rangeMainEnd;

    Interval_ offsetRangeInputTailRes;
    offsetRangeInputTailRes.start = tiling_->rangeInputTailStart;
    offsetRangeInputTailRes.end = tiling_->rangeInputTailEnd;

    Interval_ offsetRangeOutputTailRes;
    offsetRangeOutputTailRes.start = tiling_->rangeOutputTailStart;
    offsetRangeOutputTailRes.end = tiling_->rangeOutputTailEnd;

    Interval_ offsetRangeTailRes;
    offsetRangeTailRes.start = tiling_->rangeTailStart;
    offsetRangeTailRes.end = tiling_->rangeTailEnd;

    // 计算本核上的循环区间
    PorcessOffsetRange(0, tiling_->rangeMainEnd, offsetRangeMainRes);
    PorcessOffsetRange(tiling_->rangeInputTailStart, tiling_->rangeInputTailEnd, offsetRangeInputTailRes);
    PorcessOffsetRange(tiling_->rangeOutputTailStart, tiling_->rangeOutputTailEnd, offsetRangeOutputTailRes);
    PorcessOffsetRange(tiling_->rangeTailStart, tiling_->rangeTailEnd, offsetRangeTailRes);

    if (offsetRangeMainRes.end != 0 || blockIdx_ == 0) {
        ProcessMain(offsetRangeMainRes.start, offsetRangeMainRes.end);
    }

    if (offsetRangeInputTailRes.end != 0) {
        ProcessInputTail(offsetRangeInputTailRes.start, offsetRangeInputTailRes.end);
    }

    if (offsetRangeOutputTailRes.end != 0) {
        ProcessOutputTail(offsetRangeOutputTailRes.start, offsetRangeOutputTailRes.end);
    }

    if (offsetRangeTailRes.end != 0) {
        ProcessTail(offsetRangeTailRes.start, offsetRangeTailRes.end);
    }
}

template <typename T>
__aicore__ inline void TransposeCutTwoAxis<T>::ProcessPerCore()
{
    // cut two axis
    // 对于输入，输出切分轴的index和对于输出，输入切分轴的index
    inputOutputCutIndex_ = tiling_->expandedPerm[expandedOutputCutIndex_];
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        if (tiling_->expandedPerm[i] == expandedInputCutIndex_) {
            outputInputCutIndex_ = i;
            break;
        }
    }
    maxCutIndex_ = expandedOutputCutIndex_ > outputInputCutIndex_ ? expandedOutputCutIndex_ : outputInputCutIndex_;
    minCutIndex_ = expandedOutputCutIndex_ > outputInputCutIndex_ ? outputInputCutIndex_ : expandedOutputCutIndex_;

    GetCopyOutParams();
    GetTailAddressOffsetBase();

    // 该核总循环数：blkProcessNum_
    // 该核循环的起始数：srcOffset_
    // 该核的终止数：srcOffset_ + blkProcessNum_ - 1
    // 四个数据区间，初始化
    outUbLoop_ = 1;
    for (int64_t i = NDDMA_MAX_DIM_NUM - 1; i >= 0; i--) {
        if (i != expandedInputCutIndex_ && i != inputOutputCutIndex_) {
            outUbLoop_ = outUbLoop_ * (tiling_->expandedInputShape[i] / tiling_->inUbMainSrcShape[i]);
        }
    }
    ProcessBlock();
}

} // namespace Transpose

#endif // TRANSPOSE_CUT_TWO_AXIS
