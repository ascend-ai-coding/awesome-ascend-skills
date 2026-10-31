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
 * \file gather_elements_v2_scalar.h
 * \brief
 */

#ifndef GATHER_ELEMENTS_V2_SCALAR_H
#define GATHER_ELEMENTS_V2_SCALAR_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "gather_elements_v2_common.h"

namespace AscendC {
template <typename T_X, typename T_IDX>
class GatherElementsV2ScalarKernel : public GatherElementsV2KernelBase<T_X, T_IDX> {
public:
    __aicore__ inline GatherElementsV2ScalarKernel() = delete;
    __aicore__ inline GatherElementsV2ScalarKernel(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace,
                                                   const GatherElementsV2TilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(x, index, y);
    }

    __aicore__ inline void Process()
    {
        uint64_t xDimPerPre = this->xGatherDim_ * this->xPostDim_;
        uint64_t idxDimPerPre = this->idxGatherDim_ * this->idxPostDim_;

        uint64_t groupId;
        uint64_t curGroupPreDim = 1;
        uint64_t curGroupData = 1;
        uint64_t xGroupOffset = 0;
        uint64_t idxGroupOffset = 0;
        uint64_t groupCoreOffset = 0;
        ComputeProcessParams(groupId, xGroupOffset, idxGroupOffset, groupCoreOffset, curGroupPreDim, curGroupData);

        uint64_t dataPartNum = curGroupData / maxIdxDataAlign_;
        uint64_t idxDataTail = curGroupData - dataPartNum * maxIdxDataAlign_;

        for (size_t preDimId = 0; preDimId < curGroupPreDim; preDimId++) {
            uint64_t xPreDimOffset = preDimId * xDimPerPre;
            uint64_t idxPreDimOffset = preDimId * idxDimPerPre;
            uint64_t xBaseOffset = xGroupOffset + xPreDimOffset;
            // former
            for (size_t dataPartId = 0; dataPartId < dataPartNum; dataPartId++) {
                uint64_t curIdxPos = groupCoreOffset + dataPartId * maxIdxDataAlign_;
                uint64_t idxDataOffset = idxGroupOffset + idxPreDimOffset + curIdxPos;
                CopyInIdx(idxDataOffset, maxIdxDataAlign_);
                Compute(xBaseOffset, curIdxPos, maxIdxDataAlign_);
                CopyOutY(idxDataOffset, maxIdxDataAlign_);
            }
            // tail
            if (idxDataTail) {
                uint64_t xBaseOffset = xGroupOffset + xPreDimOffset;
                uint64_t curIdxPos = groupCoreOffset + dataPartNum * maxIdxDataAlign_;
                uint64_t idxDataOffset = idxGroupOffset + idxPreDimOffset + curIdxPos;
                CopyInIdx(idxDataOffset, idxDataTail);
                Compute(xBaseOffset, curIdxPos, idxDataTail);
                CopyOutY(idxDataOffset, idxDataTail);
            }
        }
    }

private:
    __aicore__ inline void InitParams(const GatherElementsV2TilingData& tiling)
    {
        this->InitBasicParams(tiling);
        this->InitGroupParams(tiling);
        formerGroupFormerData_ = tiling.scalarTiling.formerGroupFormerData;
        formerGroupTailData_ = tiling.scalarTiling.formerGroupTailData;
        tailGroupFormerData_ = tiling.scalarTiling.tailGroupFormerData;
        tailGroupTailData_ = tiling.scalarTiling.tailGroupTailData;
        maxIdxDataAlign_ = tiling.scalarTiling.maxIdxDataAlign;
    }

    __aicore__ inline void ComputeProcessParams(uint64_t& groupId, uint64_t& xGroupOffset, uint64_t& idxGroupOffset,
                                                uint64_t& groupCoreOffset, uint64_t& curGroupPreDim,
                                                uint64_t& curGroupData)
    {
        uint64_t groupCoreId;
        uint64_t xDimPerPre = this->xGatherDim_ * this->xPostDim_;
        uint64_t idxDimPerPre = this->idxGatherDim_ * this->idxPostDim_;
        this->ComputeGroupInfo(groupId, groupCoreId);
        uint64_t curGroupCoreNum = 1;
        uint64_t curGroupFormerNum = 1;
        uint64_t curGroupTailNum = 1;
        uint64_t curGroupFormerData = 1;
        uint64_t curGroupTailData = 1;
        if (groupId < this->formerGroupNum_) {
            curGroupPreDim = this->formerGroupPreDim_;
            curGroupCoreNum = this->formerGroupCoreNum_;
            curGroupFormerNum = this->formerGroupFormerNum_;
            curGroupTailNum = this->formerGroupTailNum_;
            curGroupFormerData = formerGroupFormerData_;
            curGroupTailData = formerGroupTailData_;
            xGroupOffset = groupId * this->formerGroupPreDim_ * xDimPerPre;
            idxGroupOffset = groupId * this->formerGroupPreDim_ * idxDimPerPre;
        } else {
            curGroupPreDim = this->tailGroupPreDim_;
            curGroupCoreNum = this->tailGroupCoreNum_;
            curGroupFormerNum = this->tailGroupFormerNum_;
            curGroupTailNum = this->tailGroupTailNum_;
            curGroupFormerData = tailGroupFormerData_;
            curGroupTailData = tailGroupTailData_;
            xGroupOffset = (this->formerGroupNum_ * this->formerGroupPreDim_ +
                            (groupId - this->formerGroupNum_) * this->tailGroupPreDim_) *
                           xDimPerPre;
            idxGroupOffset = (this->formerGroupNum_ * this->formerGroupPreDim_ +
                              (groupId - this->formerGroupNum_) * this->tailGroupPreDim_) *
                             idxDimPerPre;
        }
        if (groupCoreId < curGroupFormerNum) {
            curGroupData = curGroupFormerData;
            groupCoreOffset = groupCoreId * curGroupFormerData;
        } else {
            curGroupData = curGroupTailData;
            groupCoreOffset = curGroupFormerNum * curGroupFormerData +
                              (groupCoreId - curGroupFormerNum) * curGroupTailData;
        }
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        pipe.InitBuffer(idxInQue_, BUFFER_NUM, maxIdxDataAlign_ * sizeof(T_IDX));
        pipe.InitBuffer(yOutQue_, BUFFER_NUM, maxIdxDataAlign_ * sizeof(T_X));
    }

    __aicore__ inline void SetGmAddr(GM_ADDR x, GM_ADDR index, GM_ADDR y)
    {
        this->xGm_.SetGlobalBuffer((__gm__ T_X*)x);
        this->yGm_.SetGlobalBuffer((__gm__ T_X*)y);
        this->idxGm_.SetGlobalBuffer((__gm__ T_IDX*)index);
    }

    __aicore__ inline void CopyInIdx(const uint64_t& idxDataOffset, const uint64_t& curDataNum)
    {
        LocalTensor<T_IDX> idxLocal = idxInQue_.AllocTensor<T_IDX>();
        DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(sizeof(T_IDX) * curDataNum), 0, 0, 0};
        DataCopyPadExtParams<T_IDX> idxPadParams{true, 0, 0, 0};
        DataCopyPad(idxLocal, this->idxGm_[idxDataOffset], idxCopyParams, idxPadParams);
        idxInQue_.EnQue<T_IDX>(idxLocal);
    }

    __aicore__ inline void Compute(const uint64_t& xBaseOffset, const uint64_t& groupCoreOffset,
                                   const uint64_t& curDataNum)
    {
        LocalTensor<T_IDX> idxLocal = idxInQue_.DeQue<T_IDX>();
        LocalTensor<T_X> yLocal = yOutQue_.AllocTensor<T_X>();
        for (size_t idxPos = 0; idxPos < curDataNum; idxPos++) {
            T_IDX xGatherPos = idxLocal.GetValue(idxPos);
            xGatherPos = (xGatherPos + this->xGatherDim_) % this->xGatherDim_;
            uint64_t realIdxPos = groupCoreOffset + idxPos;
            uint64_t idxPostPos = realIdxPos % this->idxPostDim_;
            uint64_t xDataOffset = xBaseOffset + xGatherPos * this->xPostDim_ + idxPostPos;
            T_X xValue = this->xGm_[xDataOffset].GetValue(0);
            yLocal.SetValue(idxPos, xValue);
        }
        idxInQue_.FreeTensor<T_IDX>(idxLocal);
        yOutQue_.EnQue<T_X>(yLocal);
    }

    __aicore__ inline void CopyOutY(const uint64_t& yDataOffset, const uint64_t& curDataNum)
    {
        LocalTensor<T_X> yLocal = yOutQue_.DeQue<T_X>();
        DataCopyExtParams yCopyParams{1, static_cast<uint32_t>(sizeof(T_X) * curDataNum), 0, 0, 0};
        DataCopyPad(this->yGm_[yDataOffset], yLocal, yCopyParams);
        yOutQue_.FreeTensor<T_X>(yLocal);
    }

private:
    TQue<TPosition::VECIN, BUFFER_NUM> idxInQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> yOutQue_;

    uint64_t formerGroupFormerData_;
    uint64_t formerGroupTailData_;
    uint64_t tailGroupFormerData_;
    uint64_t tailGroupTailData_;
    uint64_t maxIdxDataAlign_;
};
} // namespace AscendC

#endif // GATHER_ELEMENTS_V2_SCALAR_H