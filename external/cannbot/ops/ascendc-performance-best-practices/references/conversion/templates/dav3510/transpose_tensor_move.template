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
 * \file transpose_tensor_move.h
 * \brief transpose_tensor_move
 */
#ifndef TRANSPOSE_TENSOR_MOVE_H
#define TRANSPOSE_TENSOR_MOVE_H

#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;

template <typename T>
class TransposeTensorMove : public TransposeBase<T> {
public:
    __aicore__ inline TransposeTensorMove(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerCore();

private:
    int64_t blockIdx_;

    // buffer
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> vecQue_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    // tiling params
    const TransposeOpTilingData* tiling_;
    int64_t inputTailFactor_ = 0;
    int64_t blockLoopNum_ = 0;
    int64_t inLoopNum_ = 0;
    int64_t srcOffset_ = 0;

    // Datacopy params
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
    DataCopyExtParams copyOutParamsMain_{1, 0, 0, 0, 0};
};

template <typename T>
__aicore__ inline void TransposeTensorMove<T>::Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData,
                                                    TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
    pipe->InitBuffer(vecQue_, BUFFER_NUM, tiling_->ubSize / BUFFER_NUM);
}

template <typename T>
__aicore__ inline void TransposeTensorMove<T>::Process()
{
    if (blockIdx_ >= tiling_->realCoreNum) {
        return;
    }
    blockLoopNum_ = tiling_->blkFactor;
    srcOffset_ = blockIdx_ * tiling_->blkFactor;
    if (blockIdx_ < tiling_->blkTailFactor) {
        blockLoopNum_ += 1;
        srcOffset_ += blockIdx_;
    } else {
        srcOffset_ += tiling_->blkTailFactor;
    }
    inLoopNum_ = Ops::Base::CeilDiv(blockLoopNum_, tiling_->inUbFactor);
    inputTailFactor_ = blockLoopNum_ % tiling_->inUbFactor;
    ProcessPerCore();
}

template <typename T>
__aicore__ inline void TransposeTensorMove<T>::ProcessPerCore()
{
    int64_t copyNum = tiling_->inUbFactor;
    for (int64_t loopIdx = 0; loopIdx < inLoopNum_; loopIdx++) {
        if (loopIdx == inLoopNum_ - 1 && inputTailFactor_ != 0) {
            copyNum = inputTailFactor_;
        }
        copyOutParamsMain_.blockLen = copyNum * sizeof(T);
        LocalTensor<T> bindLocalIn = vecQue_.AllocTensor<T>();
        DataCopyPad(bindLocalIn, inputGM_[srcOffset_ + loopIdx * tiling_->inUbFactor], copyOutParamsMain_, padParams_);
        vecQue_.EnQue(bindLocalIn);
        LocalTensor<T> bindLocalOut = vecQue_.DeQue<T>();
        DataCopyPad(outputGM_[srcOffset_ + loopIdx * tiling_->inUbFactor], bindLocalOut, copyOutParamsMain_);
        vecQue_.FreeTensor(bindLocalOut);
    }
}
} // namespace Transpose

#endif // TRANSPOSE_TENSOR_MOVE_H
