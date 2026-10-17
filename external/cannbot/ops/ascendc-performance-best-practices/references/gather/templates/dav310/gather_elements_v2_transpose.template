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
 * \file gather_elements_v2_transpose.h
 * \brief
 */

#ifndef GATHER_ELEMENTS_V2_TRANSPOSE_H
#define GATHER_ELEMENTS_V2_TRANSPOSE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "gather_elements_v2_common.h"

constexpr int32_t RIGHT_SHIFT_LEN = 31;

namespace AscendC {
template <typename T_X, typename T_IDX>
class GatherElementsV2TransposeKernel : public GatherElementsV2KernelBase<T_X, T_IDX> {
public:
    __aicore__ inline GatherElementsV2TransposeKernel() = delete;
    __aicore__ inline GatherElementsV2TransposeKernel(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace,
                                                      const GatherElementsV2TilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(x, index, y, workspace);
    }
    __aicore__ inline void Process()
    {
        uint64_t xDimPerPre = this->xGatherDim_ * this->xPostDim_;
        uint64_t idxDimPerPre = this->idxGatherDim_ * this->idxPostDim_;

        uint64_t groupId;
        uint64_t curGroupPreDim = 1;
        uint64_t curGroupPostDim = 1;
        uint64_t xGroupOffset = 0;
        uint64_t idxGroupOffset = 0;
        uint64_t groupCoreOffset = 0;
        ComputeProcessParams(groupId, xGroupOffset, idxGroupOffset, groupCoreOffset, curGroupPreDim, curGroupPostDim);

        uint64_t xBlockOffset = xGroupOffset + groupCoreOffset;
        uint64_t idxBlockOffset = idxGroupOffset + groupCoreOffset;
        uint64_t postDimPartNum = curGroupPostDim / carryNumAlign_;
        carryTailNum_ = curGroupPostDim - postDimPartNum * carryNumAlign_;

        for (size_t preDimId = 0; preDimId < curGroupPreDim; preDimId++) {
            uint64_t xPreDimOffset = preDimId * xDimPerPre;
            uint64_t idxPreDimOffset = preDimId * idxDimPerPre;
            // former
            for (size_t postDimPartId = 0; postDimPartId < postDimPartNum; postDimPartId++) {
                uint64_t xBaseOffset = xBlockOffset + xPreDimOffset + postDimPartId * carryNumAlign_;
                uint64_t idxBaseOffset = idxBlockOffset + idxPreDimOffset + postDimPartId * carryNumAlign_;
                TransposeProcess(xBaseOffset, idxBaseOffset, carryNumAlign_, xCarryNumAlign_, idxCarryNumAlign_);
                this->MTE3ToMTE2Sync();
                GatherProcess(carryNumAlign_);
                this->MTE3ToMTE2Sync();
                ReTransposeProcess(idxBaseOffset, carryNumAlign_);
            }
            // tail
            if (carryTailNum_) {
                uint64_t xBaseOffset = xBlockOffset + xPreDimOffset + postDimPartNum * carryNumAlign_;
                uint64_t idxBaseOffset = idxBlockOffset + idxPreDimOffset + postDimPartNum * carryNumAlign_;
                TransposeProcess(xBaseOffset, idxBaseOffset, carryTailNum_, this->Min(carryTailNum_, xCarryNumAlign_),
                                 this->Min(carryTailNum_, idxCarryNumAlign_));
                this->MTE3ToMTE2Sync();
                GatherProcess(carryTailNum_);
                this->MTE3ToMTE2Sync();
                ReTransposeProcess(idxBaseOffset, carryTailNum_);
            }
        }
    }

private:
    __aicore__ inline void InitParams(const GatherElementsV2TilingData& tiling)
    {
        this->InitBasicParams(tiling);
        this->InitGroupParams(tiling);
        InitTransParams(tiling);
    }

    __aicore__ inline void InitTransParams(const GatherElementsV2TilingData& tiling)
    {
        carryNumAlign_ = tiling.transTiling.carryNumAlign;
        xCarryNumAlign_ = tiling.transTiling.xCarryNumAlign;
        idxCarryNumAlign_ = tiling.transTiling.idxCarryNumAlign;
        inBufferSize_ = tiling.transTiling.inBufferSize;
        outBufferSize_ = tiling.transTiling.outBufferSize;

        uint64_t transGatherDimSlice = tiling.transTiling.transGatherDimSlice;
        transParam_.xGatherDimSlice = transGatherDimSlice;
        transParam_.xSliceNum = this->xGatherDim_ / transGatherDimSlice;
        transParam_.xGatherDimTail = this->xGatherDim_ % transGatherDimSlice;
        transParam_.idxGatherDimSlice = transGatherDimSlice;
        transParam_.idxSliceNum = this->idxGatherDim_ / transGatherDimSlice;
        transParam_.idxGatherDimTail = this->idxGatherDim_ % transGatherDimSlice;
        transParam_.yGatherDimSlice = transGatherDimSlice;
        transParam_.ySliceNum = this->idxGatherDim_ / transGatherDimSlice;
        transParam_.yGatherDimTail = this->idxGatherDim_ % transGatherDimSlice;
        gatherParam_.idxGatherDimSlice = this->Min(this->idxGatherDim_, tiling.transTiling.idxGatherDimSlice);
        gatherParam_.idxSliceNum = this->idxGatherDim_ / gatherParam_.idxGatherDimSlice;
        gatherParam_.idxGatherDimTail = this->idxGatherDim_ % gatherParam_.idxGatherDimSlice;

        uint64_t workspacePerBlock = tiling.transTiling.workspacePerBlock;
        workspaceBlockOffset_ = this->blockIdx_ * workspacePerBlock / sizeof(T_X);
    }

    __aicore__ inline void ComputeProcessParams(uint64_t& groupId, uint64_t& xGroupOffset, uint64_t& idxGroupOffset,
                                                uint64_t& groupCoreOffset, uint64_t& curGroupPreDim,
                                                uint64_t& curGroupPostDim)
    {
        uint64_t groupCoreId;
        uint64_t xDimPerPre = this->xGatherDim_ * this->xPostDim_;
        uint64_t idxDimPerPre = this->idxGatherDim_ * this->idxPostDim_;
        this->ComputeGroupInfo(groupId, groupCoreId);
        uint64_t curGroupCoreNum = 1;
        uint64_t curGroupFormerNum = 1;
        uint64_t curGroupTailNum = 1;
        uint64_t curGroupFormerPostDim = 1;
        uint64_t curGroupTailPostDim = 1;
        if (groupId < this->formerGroupNum_) {
            curGroupPreDim = this->formerGroupPreDim_;
            curGroupCoreNum = this->formerGroupCoreNum_;
            curGroupFormerNum = this->formerGroupFormerNum_;
            curGroupTailNum = this->formerGroupTailNum_;
            curGroupFormerPostDim = this->formerGroupFormerPostDim_;
            curGroupTailPostDim = this->formerGroupTailPostDim_;
            xGroupOffset = groupId * this->formerGroupPreDim_ * xDimPerPre;
            idxGroupOffset = groupId * this->formerGroupPreDim_ * idxDimPerPre;
        } else {
            curGroupPreDim = this->tailGroupPreDim_;
            curGroupCoreNum = this->tailGroupCoreNum_;
            curGroupFormerNum = this->tailGroupFormerNum_;
            curGroupTailNum = this->tailGroupTailNum_;
            curGroupFormerPostDim = this->tailGroupFormerPostDim_;
            curGroupTailPostDim = this->tailGroupTailPostDim_;
            xGroupOffset = (this->formerGroupNum_ * this->formerGroupPreDim_ +
                            (groupId - this->formerGroupNum_) * this->tailGroupPreDim_) *
                           xDimPerPre;
            idxGroupOffset = (this->formerGroupNum_ * this->formerGroupPreDim_ +
                              (groupId - this->formerGroupNum_) * this->tailGroupPreDim_) *
                             idxDimPerPre;
        }
        if (groupCoreId < curGroupFormerNum) {
            curGroupPostDim = curGroupFormerPostDim;
            groupCoreOffset = groupCoreId * curGroupFormerPostDim;
        } else {
            curGroupPostDim = curGroupTailPostDim;
            groupCoreOffset = curGroupFormerNum * curGroupFormerPostDim +
                              (groupCoreId - curGroupFormerNum) * curGroupTailPostDim;
        }
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        pipe.InitBuffer(dataInQue_, BUFFER_NUM, inBufferSize_);
        pipe.InitBuffer(dataOutQue_, BUFFER_NUM, outBufferSize_);
    }

    __aicore__ inline void SetGmAddr(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace)
    {
        uint64_t usedWorkspaceLen = this->Min(
            carryNumAlign_, this->Max(this->formerGroupFormerPostDim_, this->tailGroupFormerPostDim_));
        uint64_t yWorkspaceOffset = workspaceBlockOffset_ +
                                    usedWorkspaceLen * CeilAlign(this->xGatherDim_ * sizeof(T_X), sizeof(T_IDX)) /
                                        sizeof(T_X);
        uint64_t idxWorkspaceOffset = yWorkspaceOffset * sizeof(T_X) / sizeof(T_IDX);
        this->xGm_.SetGlobalBuffer((__gm__ T_X*)x);
        this->yGm_.SetGlobalBuffer((__gm__ T_X*)y);
        this->idxGm_.SetGlobalBuffer((__gm__ T_IDX*)index);
        xWorkspaceGm_.SetGlobalBuffer((__gm__ T_X*)workspace + workspaceBlockOffset_);
        yWorkspaceGm_.SetGlobalBuffer((__gm__ T_X*)workspace + yWorkspaceOffset);
        idxWorkspaceGm_.SetGlobalBuffer((__gm__ T_IDX*)workspace + idxWorkspaceOffset);
    }

    __aicore__ inline void TransposeProcess(const uint64_t& xBaseOffset, const uint64_t& idxBaseOffset,
                                            const uint64_t& curCarryNum, const uint64_t& curXCarryNum,
                                            const uint64_t& curIdxCarryNum)
    {
        uint64_t xCarryOffset = 0;
        uint64_t idxCarryOffset = 0;
        // x
        if (curXCarryNum) {
            for (size_t carryId = 0; carryId < curCarryNum / curXCarryNum; carryId++) {
                xCarryOffset = carryId * curXCarryNum;
                SubTransposeProcess<T_X, X_PROCESS>(xBaseOffset, xCarryOffset, curXCarryNum, this->xGatherDim_,
                                                    this->xPostDim_, transParam_.xSliceNum, transParam_.xGatherDimSlice,
                                                    transParam_.xGatherDimTail);
            }
        }

        // idx
        if (curIdxCarryNum) {
            for (size_t carryId = 0; carryId < curCarryNum / curIdxCarryNum; carryId++) {
                idxCarryOffset = carryId * curIdxCarryNum;
                SubTransposeProcess<T_IDX, IDX_PROCESS>(idxBaseOffset, idxCarryOffset, curIdxCarryNum,
                                                        this->idxGatherDim_, this->idxPostDim_, transParam_.idxSliceNum,
                                                        transParam_.idxGatherDimSlice, transParam_.idxGatherDimTail);
            }
            uint64_t idxTailCarryNum = curCarryNum % curIdxCarryNum;
            if (idxTailCarryNum) {
                idxCarryOffset = curCarryNum - idxTailCarryNum;
                SubTransposeProcess<T_IDX, IDX_PROCESS>(idxBaseOffset, idxCarryOffset, idxTailCarryNum,
                                                        this->idxGatherDim_, this->idxPostDim_, transParam_.idxSliceNum,
                                                        transParam_.idxGatherDimSlice, transParam_.idxGatherDimTail);
            }
        }
    }

    template <typename T, uint8_t PROCESS>
    __aicore__ inline void SubTransposeProcess(const uint64_t& baseOffset, const uint64_t& carryOffset,
                                               const uint64_t& curCarryNum, const uint64_t& gatherDim,
                                               const uint64_t& postDim, const uint64_t& sliceNum,
                                               const uint64_t& gatherDimSlice, const uint64_t& gatherDimTail)
    {
        uint64_t dataOffset = 0;
        uint64_t workspaceOffset = 0;
        // former
        for (size_t xSliceId = 0; xSliceId < sliceNum; xSliceId++) {
            dataOffset = baseOffset + carryOffset + xSliceId * gatherDimSlice * postDim;
            CopyIn4Transpose<T, PROCESS>(dataOffset, gatherDimSlice, curCarryNum);
            TransposeInUb<T>(gatherDimSlice, curCarryNum);
            workspaceOffset = carryOffset * gatherDim + xSliceId * gatherDimSlice;
            CopyOut2Workspace<T, PROCESS>(workspaceOffset, curCarryNum, gatherDimSlice);
        }
        // tail
        if (gatherDimTail) {
            dataOffset = baseOffset + carryOffset + sliceNum * gatherDimSlice * postDim;
            CopyIn4Transpose<T, PROCESS>(dataOffset, gatherDimTail, curCarryNum);
            TransposeInUb<T>(gatherDimTail, curCarryNum);
            workspaceOffset = carryOffset * gatherDim + sliceNum * gatherDimSlice;
            CopyOut2Workspace<T, PROCESS>(workspaceOffset, curCarryNum, gatherDimTail);
        }
    }

    __aicore__ inline void GatherProcess(const uint64_t& curCarryNum)
    {
        uint64_t idxWorkspaceOffset = 0;
        for (size_t gatherId = 0; gatherId < curCarryNum; gatherId++) {
            CopyInX4Gather(gatherId * this->xGatherDim_, this->xGatherDim_);
            // former
            for (size_t idxSliceId = 0; idxSliceId < gatherParam_.idxSliceNum; idxSliceId++) {
                idxWorkspaceOffset = gatherId * this->idxGatherDim_ + idxSliceId * gatherParam_.idxGatherDimSlice;
                CopyInIdx4Gather(idxWorkspaceOffset, this->xGatherDim_, gatherParam_.idxGatherDimSlice);
                DoNegativeIndices(this->xGatherDim_, gatherParam_.idxGatherDimSlice);
                GatherInUb(this->xGatherDim_, gatherParam_.idxGatherDimSlice);
                this->VToMTE2Sync();
                CopyOutY2Workspace(idxWorkspaceOffset, gatherParam_.idxGatherDimSlice);
                this->MTE3ToVSync();
            }
            // tail
            if (gatherParam_.idxGatherDimTail) {
                idxWorkspaceOffset = gatherId * this->idxGatherDim_ +
                                     gatherParam_.idxSliceNum * gatherParam_.idxGatherDimSlice;
                CopyInIdx4Gather(idxWorkspaceOffset, this->xGatherDim_, gatherParam_.idxGatherDimTail);
                DoNegativeIndices(this->xGatherDim_, gatherParam_.idxGatherDimSlice);
                GatherInUb(this->xGatherDim_, gatherParam_.idxGatherDimTail);
                this->VToMTE2Sync();
                CopyOutY2Workspace(idxWorkspaceOffset, gatherParam_.idxGatherDimTail);
            }
            LocalTensor<T_X> xLocal = dataInQue_.DeQue<T_X>();
            dataInQue_.FreeTensor<T_X>(xLocal);
        }
    }

    __aicore__ inline void ReTransposeProcess(const uint64_t& yBaseOffset, const uint64_t& curCarryNum)
    {
        uint64_t yWorkspaceOffset = 0;
        uint64_t yDataOffset = 0;
        for (size_t sliceId = 0; sliceId < transParam_.ySliceNum; sliceId++) {
            yWorkspaceOffset = sliceId * transParam_.yGatherDimSlice;
            yDataOffset = yBaseOffset + sliceId * transParam_.yGatherDimSlice * this->idxPostDim_;
            CopyIn4ReTranspose(yWorkspaceOffset, curCarryNum, transParam_.yGatherDimSlice);
            TransposeInUb<T_X>(curCarryNum, transParam_.yGatherDimSlice);
            CopyOutY2Gm(yDataOffset, curCarryNum, transParam_.yGatherDimSlice);
        }
        // tail
        if (transParam_.yGatherDimTail) {
            yWorkspaceOffset = transParam_.ySliceNum * transParam_.yGatherDimSlice;
            yDataOffset = yBaseOffset + transParam_.ySliceNum * transParam_.yGatherDimSlice * this->idxPostDim_;
            CopyIn4ReTranspose(yWorkspaceOffset, curCarryNum, transParam_.yGatherDimTail);
            TransposeInUb<T_X>(curCarryNum, transParam_.yGatherDimTail);
            CopyOutY2Gm(yDataOffset, curCarryNum, transParam_.yGatherDimTail);
        }
    }

    template <typename T, uint8_t PROCESS>
    __aicore__ inline void CopyIn4Transpose(const uint64_t& dataOffset, const uint64_t& gatherDimSlice,
                                            const uint64_t& curCarryNum)
    {
        if constexpr (PROCESS == X_PROCESS) {
            LocalTensor<T> xLocal = dataInQue_.AllocTensor<T>();
            uint8_t padNum = (this->xAlign_ - curCarryNum % this->xAlign_) % this->xAlign_;
            DataCopyExtParams copyParams{static_cast<uint16_t>(gatherDimSlice),
                                         static_cast<uint32_t>(sizeof(T) * curCarryNum),
                                         static_cast<uint32_t>((this->xPostDim_ - curCarryNum) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, padNum, 0};
            DataCopyPad(xLocal, this->xGm_[dataOffset], copyParams, padParams);
            dataInQue_.EnQue<T>(xLocal);
        } else if constexpr (PROCESS == IDX_PROCESS) {
            LocalTensor<T> idxLocal = dataInQue_.AllocTensor<T>();
            uint8_t padNum = (this->idxAlign_ - curCarryNum % this->idxAlign_) % this->idxAlign_;
            DataCopyExtParams copyParams{static_cast<uint16_t>(gatherDimSlice),
                                         static_cast<uint32_t>(sizeof(T) * curCarryNum),
                                         static_cast<uint32_t>((this->idxPostDim_ - curCarryNum) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, padNum, 0};
            DataCopyPad(idxLocal, this->idxGm_[dataOffset], copyParams, padParams);
            dataInQue_.EnQue<T>(idxLocal);
        }
    }

    template <typename T, uint8_t PROCESS>
    __aicore__ inline void CopyOut2Workspace(const uint64_t& workspaceOffset, const uint64_t& curCarryNum,
                                             const uint64_t& gatherDimSlice)
    {
        if constexpr (PROCESS == X_PROCESS) {
            LocalTensor<T> xTransLocal = dataOutQue_.DeQue<T>();
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(curCarryNum), static_cast<uint32_t>(sizeof(T) * gatherDimSlice),
                static_cast<uint32_t>((CeilAlign(gatherDimSlice, TRANS_LEN) - gatherDimSlice) / this->xAlign_),
                static_cast<uint32_t>((this->xGatherDim_ - gatherDimSlice) * sizeof(T)), 0};
            DataCopyPad(xWorkspaceGm_[workspaceOffset], xTransLocal, copyParams);
            dataOutQue_.FreeTensor<T>(xTransLocal);
        } else if constexpr (PROCESS == IDX_PROCESS) {
            LocalTensor<T> idxTransLocal = dataOutQue_.DeQue<T>();
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(curCarryNum), static_cast<uint32_t>(sizeof(T) * gatherDimSlice),
                static_cast<uint32_t>((CeilAlign(gatherDimSlice, TRANS_LEN) - gatherDimSlice) / this->idxAlign_),
                static_cast<uint32_t>((this->idxGatherDim_ - gatherDimSlice) * sizeof(T)), 0};
            DataCopyPad(idxWorkspaceGm_[workspaceOffset], idxTransLocal, copyParams);
            dataOutQue_.FreeTensor<T>(idxTransLocal);
        }
    }

    __aicore__ inline void CopyInX4Gather(const uint64_t& xWorkspaceOffset, const uint64_t& xGatherDimSlice)
    {
        LocalTensor<T_X> xLocal = dataInQue_.AllocTensor<T_X>();
        DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(sizeof(T_X) * xGatherDimSlice), 0, 0, 0};
        DataCopyPadExtParams<T_X> xPadParams{true, 0, 0, 0};
        DataCopyPad(xLocal, xWorkspaceGm_[xWorkspaceOffset], xCopyParams, xPadParams);
        dataInQue_.EnQue<T_X>(xLocal);
    }

    __aicore__ inline void CopyInIdx4Gather(const uint64_t& idxWorkspaceOffset, const uint64_t& xGatherDimSlice,
                                            const uint64_t& idxGatherDimSlice)
    {
        LocalTensor<T_X> xLocal = dataInQue_.DeQue<T_X>();
        LocalTensor<T_IDX> idxLocal = xLocal[CeilAlign(xGatherDimSlice, this->xAlign_)]
                                          .template ReinterpretCast<T_IDX>();
        DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(sizeof(T_IDX) * idxGatherDimSlice), 0, 0, 0};
        DataCopyPadExtParams<T_IDX> idxPadParams{true, 0, 0, 0};
        DataCopyPad(idxLocal, idxWorkspaceGm_[idxWorkspaceOffset], idxCopyParams, idxPadParams);
        dataInQue_.EnQue<T_X>(xLocal);
    }

    __aicore__ inline void GatherInUb(const uint64_t& xGatherDimSlice, const uint64_t& computeNum)
    {
        LocalTensor<T_X> xLocal = dataInQue_.DeQue<T_X>();
        LocalTensor<T_IDX> idxLocal = xLocal[CeilAlign(xGatherDimSlice, this->xAlign_)]
                                          .template ReinterpretCast<T_IDX>();
        LocalTensor<T_X> yLocal = dataOutQue_.AllocTensor<T_X>();

        Muls(idxLocal, idxLocal, static_cast<T_IDX>(sizeof(T_X)), computeNum);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> gatherOffsetLocal = idxLocal.template ReinterpretCast<uint32_t>();
        Gather(yLocal, xLocal, gatherOffsetLocal, (uint32_t)0, computeNum);

        dataInQue_.EnQue<T_X>(xLocal);
        dataOutQue_.EnQue<T_X>(yLocal);
    }

    __aicore__ inline void CopyOutY2Workspace(const uint64_t& yWorkspaceOffset, const uint64_t& idxGatherDimSlice)
    {
        LocalTensor<T_X> yLocal = dataOutQue_.DeQue<T_X>();
        DataCopyExtParams yCopyParams{1, static_cast<uint32_t>(sizeof(T_X) * idxGatherDimSlice), 0, 0, 0};
        DataCopyPad(yWorkspaceGm_[yWorkspaceOffset], yLocal, yCopyParams);
        dataOutQue_.FreeTensor<T_X>(yLocal);
    }

    __aicore__ inline void CopyIn4ReTranspose(const uint64_t& yWorkspaceOffset, const uint64_t& curCarryNum,
                                              const uint64_t& gatherDimSlice)
    {
        LocalTensor<T_X> yLocal = dataInQue_.AllocTensor<T_X>();
        uint8_t yPadNum = (this->xAlign_ - gatherDimSlice % this->xAlign_) % this->xAlign_;
        DataCopyExtParams yCopyParams{
            static_cast<uint16_t>(curCarryNum), static_cast<uint32_t>(sizeof(T_X) * gatherDimSlice),
            static_cast<uint32_t>((this->idxGatherDim_ - gatherDimSlice) * sizeof(T_X)), 0, 0};
        DataCopyPadExtParams<T_X> yPadParams{true, 0, yPadNum, 0};
        DataCopyPad(yLocal, yWorkspaceGm_[yWorkspaceOffset], yCopyParams, yPadParams);
        dataInQue_.EnQue<T_X>(yLocal);
    }

    __aicore__ inline void CopyOutY2Gm(const uint64_t& yDataOffset, const uint64_t& curCarryNum,
                                       const uint64_t& gatherDimSlice)
    {
        LocalTensor<T_X> yTransLocal = dataOutQue_.DeQue<T_X>();
        DataCopyExtParams yCopyParams{
            static_cast<uint16_t>(gatherDimSlice), static_cast<uint32_t>(sizeof(T_X) * curCarryNum),
            static_cast<uint32_t>((CeilAlign(curCarryNum, TRANS_LEN) - curCarryNum) / this->xAlign_),
            static_cast<uint32_t>((this->idxPostDim_ - curCarryNum) * sizeof(T_X)), 0};
        DataCopyPad(this->yGm_[yDataOffset], yTransLocal, yCopyParams);
        dataOutQue_.FreeTensor<T_X>(yTransLocal);
    }

    template <typename T>
    __aicore__ inline void TransposeInUb(const uint64_t& h, const uint64_t& w)
    {
        LocalTensor<T> dataLocal = dataInQue_.DeQue<T>();
        LocalTensor<T> dataTransLocal = dataOutQue_.AllocTensor<T>();
        if constexpr (sizeof(T) == BYTE4) {
            TransposeByte4<T>(dataTransLocal, dataLocal, h, w);
        } else if constexpr (sizeof(T) == BYTE2) {
            TransposeByte2<T>(dataTransLocal, dataLocal, h, w);
        }
        dataInQue_.FreeTensor<T>(dataLocal);
        dataOutQue_.EnQue<T>(dataTransLocal);
    }

    template <typename T>
    __aicore__ inline void TransposeByte4(LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal, const uint64_t& h,
                                          const uint64_t& w)
    {
        uint64_t srcList[TRANS_LEN];
        uint64_t dstList[TRANS_LEN];
        uint64_t hAlign = CeilAlign(h, TRANS_LEN);
        uint64_t blockNum = BLOCK_SIZE / BYTE4;
        uint64_t wAlign = CeilAlign(w, blockNum);
        uint64_t blockPerTransLen = TRANS_LEN / blockNum;

        for (size_t j = 0; j < hAlign / TRANS_LEN; j++) {
            for (size_t i = 0; i < TRANS_LEN; i++) {
                srcList[i] = (uint64_t)(srcLocal[j * TRANS_LEN * wAlign + i * wAlign].GetPhyAddr());
                dstList[i] = (uint64_t)(dstLocal[j * TRANS_LEN + i / blockPerTransLen * hAlign +
                                                 i % blockPerTransLen * blockNum]
                                            .GetPhyAddr());
            }
            TransDataTo5HDParams transDataParams;
            transDataParams.repeatTimes = wAlign / blockNum;
            if (transDataParams.repeatTimes == 1) {
                transDataParams.srcRepStride = 0;
                transDataParams.dstRepStride = 0;
            } else {
                transDataParams.srcRepStride = 1;
                transDataParams.dstRepStride = hAlign;
            }
            TransDataTo5HD<T>(dstList, srcList, transDataParams);
        }
    }

    template <typename T>
    __aicore__ inline void TransposeByte2(LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal, const uint64_t& h,
                                          const uint64_t& w)
    {
        uint64_t srcList[TRANS_LEN];
        uint64_t dstList[TRANS_LEN];
        uint64_t hAlign = CeilAlign(h, TRANS_LEN);
        uint64_t blockNum = BLOCK_SIZE / BYTE2;
        uint64_t wAlign = CeilAlign(w, blockNum);

        for (size_t j = 0; j < hAlign / TRANS_LEN; j++) {
            for (size_t i = 0; i < TRANS_LEN; i++) {
                srcList[i] = (uint64_t)(srcLocal[j * TRANS_LEN * wAlign + i * wAlign].GetPhyAddr());
                dstList[i] = (uint64_t)(dstLocal[j * TRANS_LEN + i * hAlign].GetPhyAddr());
            }
            TransDataTo5HDParams transDataParams;
            transDataParams.repeatTimes = wAlign / blockNum;
            if (transDataParams.repeatTimes == 1) {
                transDataParams.srcRepStride = 0;
                transDataParams.dstRepStride = 0;
            } else {
                transDataParams.srcRepStride = 1;
                transDataParams.dstRepStride = hAlign;
            }
            TransDataTo5HD<T>(dstList, srcList, transDataParams);
        }
    }

    __aicore__ inline void DoNegativeIndices(const uint64_t& xGatherDimSlice, const uint64_t& idxGatherDimSlice)
    {
        LocalTensor<T_X> xLocal = dataInQue_.DeQue<T_X>();
        LocalTensor<T_IDX> idxLocal = xLocal[CeilAlign(xGatherDimSlice, this->xAlign_)]
                                          .template ReinterpretCast<T_IDX>();
        uint64_t idxCountAlign = CeilAlign(idxGatherDimSlice, this->idxAlign_);

        ShiftRight(idxLocal[idxCountAlign], idxLocal, RIGHT_SHIFT_LEN, idxGatherDimSlice);
        Muls(idxLocal[idxCountAlign], idxLocal[idxCountAlign], -static_cast<int32_t>(this->xGatherDim_),
             idxGatherDimSlice);
        Add(idxLocal, idxLocal, idxLocal[idxCountAlign], idxGatherDimSlice);

        dataInQue_.EnQue<T_X>(xLocal);
    }

private:
    GlobalTensor<T_X> xWorkspaceGm_;
    GlobalTensor<T_X> yWorkspaceGm_;
    GlobalTensor<T_IDX> idxWorkspaceGm_;

    TQue<TPosition::VECIN, BUFFER_NUM> dataInQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> dataOutQue_;

    TransParam transParam_;
    GatherParam gatherParam_;

    uint64_t carryNumAlign_;
    uint64_t carryTailNum_;
    uint64_t xCarryNumAlign_;
    uint64_t idxCarryNumAlign_;
    uint64_t inBufferSize_;
    uint64_t outBufferSize_;
    uint64_t workspaceBlockOffset_;
};
} // namespace AscendC

#endif // GATHER_ELEMENTS_V2_TRANSPOSE_H