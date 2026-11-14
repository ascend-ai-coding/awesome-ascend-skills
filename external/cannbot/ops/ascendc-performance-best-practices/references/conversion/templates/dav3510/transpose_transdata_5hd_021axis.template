/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_transdata_5hd_021.h
 * \brief 3D 021 transpose using TransDataTo5HD: [N, H, W] -> [N, W, H]
 * Supports 8-bit (int8/uint8), 16-bit (half/bfloat16), and 32-bit (float/int32) data types.
 */

#ifndef KERNEL_TRANSPOSE_TRANSDATA_5HD_021_H_
#define KERNEL_TRANSPOSE_TRANSDATA_5HD_021_H_

#include <type_traits>
#include "op_kernel/platform_util.h"
#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;
static constexpr int64_t TRANSELEM_021 = 16;
static constexpr int64_t BLOCKELEM_8BIT = 32;
static constexpr int64_t BLOCKELEM_16BIT = 16;
static constexpr int64_t BLOCKELEM_32BIT = 8;
static constexpr int64_t DST_STRIDE_FACTOR_16BIT = 1;
static constexpr int64_t DST_STRIDE_FACTOR_32BIT = 2;

template <typename T>
class KernelTransDataTo5HD021 {
public:
    __aicore__ inline KernelTransDataTo5HD021(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const Transpose021VCONVTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneBatch(int64_t nIdx);
    __aicore__ inline void ProcessOneBatchHSplit(int64_t nIdx);
    __aicore__ inline void CopyIn(int64_t nIdx, uint32_t ubLoop, uint32_t r, uint32_t c);
    __aicore__ inline void CopyInWAligned(uint32_t actualValidRows, uint32_t r, uint32_t c);
    __aicore__ inline void CopyInWUnaligned(uint32_t actualValidRows, uint32_t r, uint32_t c, uint32_t ubLoop);
    __aicore__ inline void CopyInHSplit(int64_t nIdx, uint32_t ubLoop, uint32_t r, uint32_t c);
    __aicore__ inline void Compute(uint32_t r, uint32_t c, uint32_t rAlign, uint32_t cAlign);
    __aicore__ inline void ComputeRConv(uint32_t r, uint32_t c, uint32_t rAlign, uint32_t cAlign);
    __aicore__ inline void Compute8BitCore(uint32_t r, uint32_t c, uint32_t rAlign);
    __aicore__ inline void ComputeRConvGeneric(uint32_t r, uint32_t c, uint32_t rAlign, uint32_t cAlign);
    __aicore__ inline void ComputeCConv(uint32_t r, uint32_t c, uint32_t rAlign);
    __aicore__ inline void ComputeCConvGeneric(uint32_t r, uint32_t c, uint32_t rAlign);
    __aicore__ inline void CopyOut(int64_t nIdx, uint32_t ubLoop, uint32_t r, uint32_t c);
    __aicore__ inline void CopyOutHSplit(int64_t nIdx, uint32_t ubLoop, uint32_t r, uint32_t c);
    __aicore__ inline void CopyOutAligned(uint32_t r, uint32_t c);
    __aicore__ inline void CopyOutUnaligned(uint32_t r, uint32_t c, uint32_t ubLoop);

    static constexpr int64_t blockElem = (sizeof(T) == 1) ? BLOCKELEM_8BIT :
                                         (sizeof(T) == 4) ? BLOCKELEM_32BIT :
                                                            BLOCKELEM_16BIT;
    static constexpr int64_t dstStrideFactor = (sizeof(T) == 4) ? DST_STRIDE_FACTOR_32BIT : DST_STRIDE_FACTOR_16BIT;

    const Transpose021VCONVTilingData* tiling_ = nullptr;
    TQue<TPosition::VECIN, 1> inQueueSrc;
    TQue<TPosition::VECOUT, 1> outQueueDst;
    GlobalTensor<T> srcGlobal, dstGlobal;

    int64_t blockIdx_ = 0;
    int64_t fullCoreNum_ = 0;
    int64_t nStart_ = 0;
    int64_t nCount_ = 0;
    int64_t hStart_ = 0;
    int64_t hCount_ = 0;

    DataCopyExtParams copyInParams_;
    DataCopyExtParams copyOutParams_;
};

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::Init(GM_ADDR x, GM_ADDR y,
                                                        const Transpose021VCONVTilingData* tilingData, TPipe* pipe)
{
    tiling_ = tilingData;
    blockIdx_ = GetBlockIdx();
    srcGlobal.SetGlobalBuffer((__gm__ T*)x);
    dstGlobal.SetGlobalBuffer((__gm__ T*)y);

    if (tiling_->UseHSplit) {
        fullCoreNum_ = tiling_->UsedCoreNum - 1;
        if (blockIdx_ < fullCoreNum_) {
            hStart_ = blockIdx_ * tiling_->HPerCore;
            hCount_ = tiling_->HPerCore;
        } else if (blockIdx_ < tiling_->UsedCoreNum) {
            hStart_ = fullCoreNum_ * tiling_->HPerCore;
            hCount_ = tiling_->HTailCore;
        }
        nStart_ = 0;
        nCount_ = tiling_->NLen;
    } else {
        fullCoreNum_ = tiling_->UsedCoreNum - 1;
        if (blockIdx_ < fullCoreNum_) {
            nStart_ = blockIdx_ * tiling_->NPerCore;
            nCount_ = tiling_->NPerCore;
        } else if (blockIdx_ < tiling_->UsedCoreNum) {
            nStart_ = fullCoreNum_ * tiling_->NPerCore;
            nCount_ = tiling_->NTailCore;
        }
    }

    pipe->InitBuffer(inQueueSrc, BUFFER_NUM, tiling_->AvailableUbSize);
    pipe->InitBuffer(outQueueDst, BUFFER_NUM, tiling_->AvailableUbSize);
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::ComputeRConv(uint32_t r, uint32_t c, uint32_t rAlign,
                                                                uint32_t cAlign)
{
    if constexpr (sizeof(T) == 1) {
        Compute8BitCore(r, c, rAlign);
    } else {
        ComputeRConvGeneric(r, c, rAlign, cAlign);
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::Compute8BitCore(uint32_t r, uint32_t c, uint32_t rAlign)
{
    LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
    LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
    uint16_t cAlignBlocks = c / blockElem;
    uint8_t evenUbCount = (rAlign + 1) / 2;
    uint8_t oddUbCount = rAlign / 2;
    TransDataTo5HDParams evenParams{false, false, evenUbCount,
                                    static_cast<uint16_t>((evenUbCount <= 1) ? 0 : dstStrideFactor),
                                    static_cast<uint16_t>((evenUbCount <= 1) ? 0 : c)};
    TransDataTo5HDParams oddParams{true, false, oddUbCount,
                                   static_cast<uint16_t>((oddUbCount <= 1) ? 0 : dstStrideFactor),
                                   static_cast<uint16_t>((oddUbCount <= 1) ? 0 : c)};
    for (uint16_t j = 0; j < cAlignBlocks; j++) {
        uint64_t srcList[TRANSELEM_021];
        for (uint16_t i = 0; i < TRANSELEM_021; i++) {
            srcList[i] = srcLocal[i * c + j * blockElem].GetPhyAddr();
        }
        uint64_t srcListOdd[TRANSELEM_021];
        if (oddUbCount > 0) {
            for (uint16_t i = 0; i < TRANSELEM_021; i++) {
                srcListOdd[i] = srcLocal[(i + TRANSELEM_021) * c + j * blockElem].GetPhyAddr();
            }
        }
        for (uint16_t srcHalfIdx = 0; srcHalfIdx < 2; srcHalfIdx++) {
            bool srcHighHalf = (srcHalfIdx == 1);
            uint16_t dstRowBase = j * blockElem + srcHalfIdx * TRANSELEM_021;

            uint64_t dstList[TRANSELEM_021];
            for (uint16_t k = 0; k < TRANSELEM_021; k++) {
                dstList[k] = dstLocal[(dstRowBase + k) * r].GetPhyAddr();
            }
            evenParams.srcHighHalf = srcHighHalf;
            if (evenUbCount > 0) {
                TransDataTo5HD<T>(dstList, srcList, evenParams);
            }
            if (oddUbCount > 0) {
                PipeBarrier<PIPE_V>();
                oddParams.srcHighHalf = srcHighHalf;
                TransDataTo5HD<T>(dstList, srcListOdd, oddParams);
            }
            if (srcHalfIdx == 0) {
                PipeBarrier<PIPE_V>();
            }
        }
    }
    outQueueDst.EnQue<T>(dstLocal);
    inQueueSrc.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::ComputeRConvGeneric(uint32_t r, uint32_t c, uint32_t rAlign,
                                                                       uint32_t cAlign)
{
    LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
    LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = r / TRANSELEM_021;
    transDataParams.dstRepStride = transDataParams.repeatTimes == 1 ? 0 : dstStrideFactor;
    transDataParams.srcRepStride = transDataParams.repeatTimes == 1 ? 0 : cAlign * TRANSELEM_021;

    for (uint16_t j = 0; j < cAlign; j++) {
        uint64_t srcLocalList[TRANSELEM_021];
        for (uint16_t i = 0; i < TRANSELEM_021; i++) {
            srcLocalList[i] = srcLocal[i * c + j * blockElem].GetPhyAddr();
        }

        uint64_t dstLocalList[TRANSELEM_021];
        if constexpr (sizeof(T) == 4) {
            constexpr int64_t dstRows = TRANSELEM_021 / dstStrideFactor;
            for (uint16_t k = 0; k < dstRows; k++) {
                dstLocalList[dstStrideFactor * k] = dstLocal[(k + j * blockElem) * r].GetPhyAddr();
                dstLocalList[dstStrideFactor * k + 1] = dstLocal[(k + j * blockElem) * r + blockElem].GetPhyAddr();
            }
        } else {
            for (uint16_t i = 0; i < TRANSELEM_021; i++) {
                dstLocalList[i] = dstLocal[i * r + j * TRANSELEM_021 * r].GetPhyAddr();
            }
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
    }
    outQueueDst.EnQue<T>(dstLocal);
    inQueueSrc.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::ComputeCConv(uint32_t r, uint32_t c, uint32_t rAlign)
{
    if constexpr (sizeof(T) == 1) {
        Compute8BitCore(r, c, rAlign);
    } else {
        ComputeCConvGeneric(r, c, rAlign);
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::ComputeCConvGeneric(uint32_t r, uint32_t c, uint32_t rAlign)
{
    LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
    LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = c / blockElem;
    transDataParams.dstRepStride = transDataParams.repeatTimes == 1 ? 0 : rAlign * TRANSELEM_021;
    transDataParams.srcRepStride = transDataParams.repeatTimes == 1 ? 0 : 1;

    for (uint16_t j = 0; j < rAlign; j++) {
        uint64_t srcLocalList[TRANSELEM_021];
        for (uint16_t i = 0; i < TRANSELEM_021; i++) {
            srcLocalList[i] = srcLocal[i * c + j * TRANSELEM_021 * c].GetPhyAddr();
        }

        uint64_t dstLocalList[TRANSELEM_021];
        if constexpr (sizeof(T) == 4) {
            constexpr int64_t dstRows = TRANSELEM_021 / dstStrideFactor;
            for (uint16_t k = 0; k < dstRows; k++) {
                dstLocalList[dstStrideFactor * k] = dstLocal[k * r + j * TRANSELEM_021].GetPhyAddr();
                dstLocalList[dstStrideFactor * k + 1] = dstLocal[k * r + j * TRANSELEM_021 + blockElem].GetPhyAddr();
            }
        } else {
            for (uint16_t i = 0; i < TRANSELEM_021; i++) {
                dstLocalList[i] = dstLocal[i * r + j * TRANSELEM_021].GetPhyAddr();
            }
        }
        TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
    }
    outQueueDst.EnQue<T>(dstLocal);
    inQueueSrc.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::Compute(uint32_t r, uint32_t c, uint32_t rAlign, uint32_t cAlign)
{
    if (tiling_->UseRConv) {
        ComputeRConv(r, c, rAlign, cAlign);
    } else {
        ComputeCConv(r, c, rAlign);
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::CopyIn(int64_t nIdx, uint32_t ubLoop, uint32_t r, uint32_t c)
{
    LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();

    int64_t nSrcOffset = nIdx * tiling_->HLen * tiling_->WLen;
    int64_t ubSrcOffset = 0;
    if (tiling_->UseRConv) {
        ubSrcOffset = ubLoop * tiling_->rUbSplitPara.UbAlignFactor * TRANSELEM_021 * tiling_->WLen;
    } else {
        ubSrcOffset = ubLoop * tiling_->cUbSplitPara.UbAlignFactor * TRANSELEM_021;
    }

    uint32_t actualValidRows;
    if (tiling_->UseRConv) {
        int64_t hOffsetSrc = ubLoop * tiling_->rUbSplitPara.UbAlignFactor * TRANSELEM_021;
        int64_t remainingRows = tiling_->HLen - hOffsetSrc;
        actualValidRows = (remainingRows < r) ? remainingRows : r;
    } else {
        actualValidRows = tiling_->HLen;
    }

    DataCopyPadExtParams<T> copyInPadParams{false, 0, 0, 0};
    int64_t wAlignCheck = (sizeof(T) == 1) ? blockElem : TRANSELEM_021;
    if (tiling_->WLen % wAlignCheck == 0) {
        CopyInWAligned(actualValidRows, r, c);
    } else {
        CopyInWUnaligned(actualValidRows, r, c, ubLoop);
    }
    DataCopyPad(srcLocal, srcGlobal[nSrcOffset + ubSrcOffset], copyInParams_, copyInPadParams);
    inQueueSrc.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::CopyInWAligned(uint32_t actualValidRows, uint32_t r, uint32_t c)
{
    if (tiling_->UseRConv) {
        copyInParams_.blockCount = 1;
        copyInParams_.blockLen = actualValidRows * c * sizeof(T);
        copyInParams_.srcStride = 0;
        copyInParams_.dstStride = 0;
    } else if (c == tiling_->WLen) {
        copyInParams_.blockCount = 1;
        copyInParams_.blockLen = actualValidRows * c * sizeof(T);
        copyInParams_.srcStride = 0;
        copyInParams_.dstStride = 0;
    } else {
        copyInParams_.blockCount = actualValidRows;
        copyInParams_.blockLen = c * sizeof(T);
        copyInParams_.srcStride = (tiling_->WLen - c) * sizeof(T);
        copyInParams_.dstStride = 0;
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::CopyInWUnaligned(uint32_t actualValidRows, uint32_t r, uint32_t c,
                                                                    uint32_t ubLoop)
{
    if (tiling_->UseRConv) {
        copyInParams_.blockCount = actualValidRows;
        copyInParams_.blockLen = tiling_->WLen * sizeof(T);
        copyInParams_.srcStride = 0;
        copyInParams_.dstStride = (tiling_->WAlignBlockElem - tiling_->WLen) * sizeof(T) / BLOCK_SIZE_BYTE;
    } else {
        int32_t wOffset = ubLoop * tiling_->cUbSplitPara.UbAlignFactor * TRANSELEM_021;
        int32_t validW = (tiling_->WLen - wOffset < c) ? tiling_->WLen - wOffset : c;
        copyInParams_.blockCount = actualValidRows;
        copyInParams_.blockLen = validW * sizeof(T);
        copyInParams_.srcStride = (tiling_->WLen - validW) * sizeof(T);
        copyInParams_.dstStride = (c - validW) * sizeof(T) / BLOCK_SIZE_BYTE;
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::CopyOut(int64_t nIdx, uint32_t ubLoop, uint32_t r, uint32_t c)
{
    LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();

    int64_t nDstOffset = nIdx * tiling_->WLen * tiling_->HLen;
    int64_t ubDstOffset = 0;
    if (tiling_->UseRConv) {
        ubDstOffset = ubLoop * tiling_->rUbSplitPara.UbAlignFactor * TRANSELEM_021;
    } else {
        ubDstOffset = ubLoop * tiling_->cUbSplitPara.UbAlignFactor * TRANSELEM_021 * tiling_->HLen;
    }

    int64_t hAlignCheck = (sizeof(T) == 1) ? blockElem : TRANSELEM_021;
    int64_t wAlignCheck = (sizeof(T) == 1) ? blockElem : TRANSELEM_021;
    if (tiling_->HLen % hAlignCheck == 0 && tiling_->WLen % wAlignCheck == 0) {
        CopyOutAligned(r, c);
    } else {
        CopyOutUnaligned(r, c, ubLoop);
    }
    DataCopyPad(dstGlobal[nDstOffset + ubDstOffset], dstLocal, copyOutParams_);
    outQueueDst.FreeTensor(dstLocal);
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::CopyOutAligned(uint32_t r, uint32_t c)
{
    if (tiling_->UseRConv) {
        if (r == tiling_->HLen) {
            copyOutParams_.blockCount = 1;
            copyOutParams_.blockLen = c * r * sizeof(T);
            copyOutParams_.srcStride = 0;
            copyOutParams_.dstStride = 0;
        } else {
            copyOutParams_.blockCount = c;
            copyOutParams_.blockLen = r * sizeof(T);
            copyOutParams_.srcStride = 0;
            copyOutParams_.dstStride = (tiling_->HLen - r) * sizeof(T);
        }
    } else {
        copyOutParams_.blockCount = 1;
        copyOutParams_.blockLen = c * r * sizeof(T);
        copyOutParams_.srcStride = 0;
        copyOutParams_.dstStride = 0;
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::CopyOutUnaligned(uint32_t r, uint32_t c, uint32_t ubLoop)
{
    if (tiling_->UseRConv) {
        int32_t hOffset = ubLoop * tiling_->rUbSplitPara.UbAlignFactor * TRANSELEM_021;
        uint32_t validLen = (tiling_->HLen - hOffset < r) ? tiling_->HLen - hOffset : r;
        uint32_t validCount = (tiling_->WLen < c) ? tiling_->WLen : c;
        copyOutParams_.blockCount = validCount;
        copyOutParams_.blockLen = validLen * sizeof(T);
        copyOutParams_.srcStride = (r - validLen) * sizeof(T) / BLOCK_SIZE_BYTE;
        copyOutParams_.dstStride = (tiling_->HLen - validLen) * sizeof(T);
    } else {
        uint32_t validLen = tiling_->HLen;
        int32_t wOffset = ubLoop * tiling_->cUbSplitPara.UbAlignFactor * TRANSELEM_021;
        uint32_t validCount = (tiling_->WLen - wOffset < c) ? tiling_->WLen - wOffset : c;
        copyOutParams_.blockCount = validCount;
        copyOutParams_.blockLen = validLen * sizeof(T);
        copyOutParams_.srcStride = (r - validLen) * sizeof(T) / BLOCK_SIZE_BYTE;
        copyOutParams_.dstStride = 0;
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::ProcessOneBatch(int64_t nIdx)
{
    for (uint32_t i = 0; i < tiling_->UbLoopCount; i++) {
        uint32_t r, c, rAlign, cAlign;
        if (i < tiling_->UbLoopCount - 1) {
            rAlign = tiling_->rUbSplitPara.UbAlignFactor;
            r = tiling_->rUbSplitPara.UbFactor;
            cAlign = tiling_->cUbSplitPara.UbAlignFactor;
            c = tiling_->cUbSplitPara.UbFactor;
        } else {
            rAlign = tiling_->rUbSplitPara.UbTailAlignFactor;
            r = tiling_->rUbSplitPara.UbTailFactor;
            cAlign = tiling_->cUbSplitPara.UbTailAlignFactor;
            c = tiling_->cUbSplitPara.UbTailFactor;
        }
        CopyIn(nIdx, i, r, c);
        Compute(r, c, rAlign, cAlign);
        CopyOut(nIdx, i, r, c);
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::CopyInHSplit(int64_t nIdx, uint32_t ubLoop, uint32_t r, uint32_t c)
{
    LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();

    int64_t nSrcOffset = nIdx * tiling_->HLen * tiling_->WLen;
    int64_t hSrcOffset = hStart_ * tiling_->WLen;
    int64_t ubSrcOffset = ubLoop * tiling_->cUbSplitPara.UbFactor;

    int64_t totalSrcOffset = nSrcOffset + hSrcOffset + ubSrcOffset;

    DataCopyPadExtParams<T> copyInPadParams{false, 0, 0, 0};

    int64_t wAlignCheck = blockElem;
    if (tiling_->WLen % wAlignCheck == 0) {
        if (c == tiling_->WAlignBlockElem) {
            copyInParams_.blockCount = 1;
            copyInParams_.blockLen = hCount_ * c * sizeof(T);
            copyInParams_.srcStride = 0;
            copyInParams_.dstStride = 0;
        } else {
            copyInParams_.blockCount = hCount_;
            copyInParams_.blockLen = c * sizeof(T);
            copyInParams_.srcStride = (tiling_->WLen - c) * sizeof(T);
            copyInParams_.dstStride = 0;
        }
        DataCopyPad(srcLocal, srcGlobal[totalSrcOffset], copyInParams_, copyInPadParams);
    } else {
        int32_t wOffset = ubLoop * tiling_->cUbSplitPara.UbFactor;
        int32_t validW = (tiling_->WLen - wOffset < c) ? tiling_->WLen - wOffset : c;
        copyInParams_.blockCount = hCount_;
        copyInParams_.blockLen = validW * sizeof(T);
        copyInParams_.srcStride = (tiling_->WLen - validW) * sizeof(T);
        copyInParams_.dstStride = (c - validW) * sizeof(T) / BLOCK_SIZE_BYTE;
        DataCopyPad(srcLocal, srcGlobal[totalSrcOffset], copyInParams_, copyInPadParams);
    }
    inQueueSrc.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::CopyOutHSplit(int64_t nIdx, uint32_t ubLoop, uint32_t r, uint32_t c)
{
    LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();

    int64_t nDstOffset = nIdx * tiling_->WLen * tiling_->HLen;
    int64_t ubDstOffset = ubLoop * tiling_->cUbSplitPara.UbFactor * tiling_->HLen;
    int32_t validH = (tiling_->HLen - hStart_ < hCount_) ? tiling_->HLen - hStart_ : hCount_;

    int32_t validW = (tiling_->WLen - (ubLoop * tiling_->cUbSplitPara.UbFactor) < c) ?
                         tiling_->WLen - (ubLoop * tiling_->cUbSplitPara.UbFactor) :
                         c;
    copyOutParams_.blockCount = validW;
    copyOutParams_.blockLen = validH * sizeof(T);
    copyOutParams_.srcStride = (r - validH) * sizeof(T) / BLOCK_SIZE_BYTE;
    copyOutParams_.dstStride = (tiling_->HLen - validH) * sizeof(T);
    DataCopyPad(dstGlobal[nDstOffset + hStart_ + ubDstOffset], dstLocal, copyOutParams_);
    outQueueDst.FreeTensor(dstLocal);
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::ProcessOneBatchHSplit(int64_t nIdx)
{
    int64_t hAlignUnit = (sizeof(T) == 1) ? blockElem : TRANSELEM_021;
    uint32_t rAlign = (hCount_ + hAlignUnit - 1) / hAlignUnit;
    uint32_t r = hCount_;
    for (uint32_t i = 0; i < tiling_->UbLoopCount; i++) {
        uint32_t c, cAlign;
        if (i < tiling_->UbLoopCount - 1) {
            cAlign = tiling_->cUbSplitPara.UbAlignFactor;
            c = tiling_->cUbSplitPara.UbFactor;
        } else {
            cAlign = tiling_->cUbSplitPara.UbTailAlignFactor;
            c = tiling_->cUbSplitPara.UbTailFactor;
        }
        CopyInHSplit(nIdx, i, r, c);
        Compute(r, c, rAlign, cAlign);
        CopyOutHSplit(nIdx, i, r, c);
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD021<T>::Process()
{
    if (blockIdx_ >= tiling_->UsedCoreNum) {
        return;
    }
    if (tiling_->UseHSplit) {
        for (int64_t n = 0; n < nCount_; n++) {
            ProcessOneBatchHSplit(n);
        }
    } else {
        for (int64_t n = 0; n < nCount_; n++) {
            ProcessOneBatch(nStart_ + n);
        }
    }
}

} // namespace Transpose
#endif
