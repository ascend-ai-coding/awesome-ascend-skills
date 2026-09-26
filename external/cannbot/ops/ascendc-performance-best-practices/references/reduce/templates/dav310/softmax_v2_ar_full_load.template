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
 * \file softmax_v2_ar_full_load.h
 * \brief
 */

#ifndef SOFTMAX_V2_AR_FULL_LOAD_H
#define SOFTMAX_V2_AR_FULL_LOAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

#include "softmax_v2_base.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace SoftmaxV2Ops
{
using namespace AscendC;

constexpr static uint32_t DOUBLE_BUFFER = 2;
constexpr static uint32_t BLOCK_SIZE = 32;  // 32B
constexpr static uint32_t AR_FULL_LOAD_BINARY_TMP_BYTES = 512;

template <typename T_in, typename T_out>
class SoftmaxV2AR : public SoftmaxV2OpsBase
{
public:
    __aicore__ inline SoftmaxV2AR(TPipe* pipe)
    {
        pipe_ = pipe;
    };

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SoftmaxV2ARTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessUB(int64_t ubA, int64_t aOffset);

    __aicore__ inline void FirstNormCompute(int64_t ubA, __local_mem__ T_in* xInAddr,
                                            __local_mem__ float* xTmpLocalAddr);
    __aicore__ inline void SecondNormCompute(const LocalTensor<T_out>& dstTensor, const LocalTensor<float>& srcTensor,
                                             const LocalTensor<float>& reduceSumTempTensor, const int64_t aSize,
                                             const int64_t rSize, const int64_t stride);
    __aicore__ inline void SecondNormComputePost(const LocalTensor<T_out>& dstTensor,
                                                 const LocalTensor<float>& srcTensor,
                                                 const LocalTensor<float>& oriSrcTensor, const int64_t aSize,
                                                 const int64_t rSize, const int64_t stride);

    __aicore__ inline void CopyInX(const LocalTensor<T_in>& xInUb, int64_t ubA, int64_t offset);
    __aicore__ inline void CopyOutY(const LocalTensor<T_out>& yOutUb, int64_t ubA, int64_t offset);
    __aicore__ inline void LoadTensorForDtypeTIn(__local_mem__ T_in* src, AscendC::MicroAPI::RegTensor<float>& dst,
                                                 AscendC::MicroAPI::MaskReg& preg, uint32_t offset);
    __aicore__ inline void StoreTensorForDtypeTOut(__local_mem__ T_out* dst, AscendC::MicroAPI::RegTensor<float>& src,
                                                   AscendC::MicroAPI::MaskReg& preg, uint32_t offset);

private:
    /* global memory address */
    GlobalTensor<T_in> xGm_;
    GlobalTensor<T_out> yGm_;

    /* ascendc variable */
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;

    TBuf<> tmpLocalBuffer_;

    int64_t blockA_ = 0;  // 获取分块操作中的单个块的大小
    const SoftmaxV2ARTilingData* tl_ = nullptr;
};

template <typename T_in, typename T_out>
__aicore__ inline void SoftmaxV2AR<T_in, T_out>::Init(GM_ADDR x, GM_ADDR y, const SoftmaxV2ARTilingData* tilingData)
{
    this->tl_ = tilingData;
    // GM not need align.
    // 获取分块操作中的单个块的大小。判断是否是最后一块，是最后一块，则等于剩余元素的数量，否则等于固定的单核处理的行数
    this->blockA_ = (AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1)
                        ? (tl_->a - tl_->aBlockFactor * (AscendC::GetBlockNum() - 1))
                        : tl_->aBlockFactor;
    int64_t aGmOffset = tl_->aBlockFactor * AscendC::GetBlockIdx() * tl_->r;
    // 初始化GM Tensor
    xGm_.SetGlobalBuffer((__gm__ T_in*)x + aGmOffset);
    yGm_.SetGlobalBuffer((__gm__ T_out*)y + aGmOffset);

    // 初始化Pipe
    int64_t ubBufferSize = tl_->ubFactor * tl_->rAligned;
    pipe_->InitBuffer(this->xQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T_in));
    pipe_->InitBuffer(this->yQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T_out));
    pipe_->InitBuffer(this->tmpLocalBuffer_, ubBufferSize * sizeof(float) + AR_FULL_LOAD_BINARY_TMP_BYTES);
}

template <typename T_in, typename T_out>
__aicore__ inline void SoftmaxV2AR<T_in, T_out>::Process()
{
    // ubLoop: 表示需要多少个子块来覆盖singleA大小的数据
    int64_t ubLoop = ops::CeilDiv(this->blockA_, tl_->ubFactor);
    int64_t lastUbFactor = this->blockA_ - tl_->ubFactor * (ubLoop - 1);
    // 循环处理每个子块
    for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoop; ubLoopIdx++) {
        // aOffset：计算当前子块的偏移量
        int64_t aOffset = ubLoopIdx * tl_->ubFactor * tl_->r;
        int64_t ubA = (ubLoopIdx == (ubLoop - 1)) ? lastUbFactor : tl_->ubFactor;
        ProcessUB(ubA, aOffset);
    }
}

template <typename T_in, typename T_out>
__aicore__ inline void SoftmaxV2AR<T_in, T_out>::ProcessUB(int64_t ubA, int64_t aOffset)
{
    LocalTensor<T_in> xInUb = xQueue_.AllocTensor<T_in>();
    CopyInX(xInUb, ubA, aOffset);
    xQueue_.EnQue(xInUb);
    xInUb = xQueue_.DeQue<T_in>();

    LocalTensor<float> tmpLocalTensor = tmpLocalBuffer_.AllocTensor<float>();
    LocalTensor<float> binaryTmpLocalTensor = tmpLocalTensor[tl_->ubFactor * tl_->rAligned];
    LocalTensor<float> xTmpLocalTensor = tmpLocalTensor[0];

    __local_mem__ T_in* xInUbAddr = (__local_mem__ T_in*)xInUb.GetPhyAddr();
    __local_mem__ float* xTmpLocalAddr = (__local_mem__ float*)xTmpLocalTensor.GetPhyAddr();
    __local_mem__ float* binaryTmpLocalAddr = (__local_mem__ float*)binaryTmpLocalTensor.GetPhyAddr();

    FirstNormCompute(ubA, xInUbAddr, xTmpLocalAddr);
    xQueue_.FreeTensor<T_in>(xInUb);
    LocalTensor<T_out> yInUb = yQueue_.AllocTensor<T_out>();
    SecondNormCompute(yInUb, xTmpLocalTensor, binaryTmpLocalTensor, ubA, tl_->r, tl_->rAligned);
    yQueue_.EnQue(yInUb);

    LocalTensor<T_out> yOutUb = yQueue_.DeQue<T_out>();
    CopyOutY(yOutUb, ubA, aOffset);
    yQueue_.FreeTensor(yOutUb);
}

template <typename T_in, typename T_out>
__aicore__ inline void SoftmaxV2AR<T_in, T_out>::FirstNormCompute(int64_t ubA, __local_mem__ T_in* xInAddr,
                                                                  __local_mem__ float* xTmpLocalAddr)
{
    int64_t rAligned = tl_->rAligned;
    int64_t ubActualR = tl_->r;
    uint16_t ubActualA = static_cast<uint16_t>(ubA);
    int64_t tailUbBlockSize = tl_->r - VL_FP32 * (tl_->rLoopCount - 1);
    uint16_t rLoopCount = static_cast<uint16_t>(tl_->rLoopCount);
    uint16_t rLoopCountTmp = rLoopCount - 1;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vreg0;
        AscendC::MicroAPI::RegTensor<float> vreg1;
        AscendC::MicroAPI::RegTensor<float> vreg2;
        AscendC::MicroAPI::RegTensor<float> vreg3;
        AscendC::MicroAPI::RegTensor<float> vreg4;
        AscendC::MicroAPI::RegTensor<float> vreg5;

        AscendC::MicroAPI::MaskReg maskTail;
        AscendC::MicroAPI::MaskReg maskSub;
        AscendC::MicroAPI::MaskReg maskMax;
        AscendC::MicroAPI::MaskReg maskAll;

        uint32_t sreg0 = tailUbBlockSize;
        uint32_t sreg1 = VL_FP32;
        maskTail = AscendC::MicroAPI::UpdateMask<uint32_t>(sreg0);
        maskAll = AscendC::MicroAPI::UpdateMask<uint32_t>(sreg1);

        for (uint16_t k = 0; k < ubActualA; k++) {
            uint32_t tailAddrPtr = k * rAligned + VL_FP32 * (rLoopCount - 1);
            AscendC::MicroAPI::Duplicate(vreg0, static_cast<float>(-INFINITY), maskAll);
            LoadTensorForDtypeTIn(xInAddr, vreg1, maskTail, tailAddrPtr);
            AscendC::MicroAPI::Max(vreg1, vreg0, vreg1, maskTail);
            AscendC::MicroAPI::Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vreg0, vreg1, maskTail);

            uint32_t sreg2 = static_cast<uint32_t>(ubActualR - tailUbBlockSize);
            for (uint16_t i = 0; i < rLoopCountTmp; i++) {
                maskMax = AscendC::MicroAPI::UpdateMask<float>(sreg2);
                uint32_t addrPtr = i * VL_FP32 + k * rAligned;
                LoadTensorForDtypeTIn(xInAddr, vreg1, maskMax, addrPtr);
                AscendC::MicroAPI::Max(vreg0, vreg0, vreg1, maskMax);
            }
            AscendC::MicroAPI::ReduceMax(vreg2, vreg0, maskAll);
            AscendC::MicroAPI::Duplicate(vreg3, vreg2, maskAll);

            // 求sub和exp
            uint32_t sreg3 = ubActualR;
            for (uint16_t i = 0; i < rLoopCount; i++) {
                maskSub = AscendC::MicroAPI::UpdateMask<float>(sreg3);
                uint32_t addrPtr = i * VL_FP32 + k * rAligned;
                LoadTensorForDtypeTIn(xInAddr, vreg1, maskSub, addrPtr);
                AscendC::MicroAPI::Sub(vreg4, vreg1, vreg3, maskSub);
                AscendC::MicroAPI::Exp(vreg5, vreg4, maskSub);
                AscendC::MicroAPI::DataCopy(((__local_mem__ float*)xTmpLocalAddr + addrPtr), vreg5, maskSub);
            }
        }
    }
}

template <typename T_in, typename T_out>
__aicore__ inline void SoftmaxV2AR<T_in, T_out>::SecondNormComputePost(const LocalTensor<T_out>& dstTensor,
                                                                       const LocalTensor<float>& srcTensor,
                                                                       const LocalTensor<float>& oriSrcTensor,
                                                                       const int64_t aSize, const int64_t rSize,
                                                                       const int64_t stride)
{
    if (aSize <= 0) {
        return;
    }
    if (rSize <= 0) {
        return;
    }
    if (rSize > CONST_TWO * VL_FP32) {
        return;
    }

    uint16_t loopTimes = aSize;
    uint16_t rLoopCount = tl_->rLoopCount;
    uint16_t oriR = tl_->r;
    uint16_t oriRAligned = tl_->rAligned;

    if (rSize <= VL_FP32) {
        __local_mem__ T_out* dst = (__local_mem__ T_out*)dstTensor.GetPhyAddr();
        __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();
        __local_mem__ float* oriSrc = (__local_mem__ float*)oriSrcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg, dReg;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                DataCopy(aReg, (__local_mem__ float*)src + i * static_cast<uint32_t>(stride));
                ReduceSum(bReg, aReg, pMask);
                Duplicate(cReg, bReg, pFull);
                uint32_t sreg0 = static_cast<uint32_t>(oriR);
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    DataCopy(dReg, ((__local_mem__ float*)oriSrc + addrPtr));
                    Div(dReg, dReg, cReg, maskOri);
                    StoreTensorForDtypeTOut(dst, dReg, maskOri, addrPtr);
                }
            }
        }
    } else {
        __local_mem__ T_out* dst = (__local_mem__ T_out*)dstTensor.GetPhyAddr();
        __local_mem__ float* src0 = (__local_mem__ float*)srcTensor.GetPhyAddr();
        __local_mem__ float* src1 = (__local_mem__ float*)srcTensor.GetPhyAddr() + VL_FP32;
        __local_mem__ float* oriSrc = (__local_mem__ float*)oriSrcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg, dReg, eReg;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                DataCopy(aReg, (__local_mem__ float*)src0 + i * static_cast<uint32_t>(stride));
                DataCopy(bReg, (__local_mem__ float*)src1 + i * static_cast<uint32_t>(stride));
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(aReg, cReg, pMask);
                ReduceSum(bReg, aReg, pFull);
                Duplicate(dReg, bReg, pFull);
                uint32_t sreg0 = static_cast<uint32_t>(oriR);
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    DataCopy(eReg, ((__local_mem__ float*)oriSrc + addrPtr));
                    Div(eReg, eReg, dReg, maskOri);
                    StoreTensorForDtypeTOut(dst, eReg, maskOri, addrPtr);
                }
            }
        }
    }
}

template <typename T_in, typename T_out>
__aicore__ inline void SoftmaxV2AR<T_in, T_out>::SecondNormCompute(const LocalTensor<T_out>& dstTensor,
                                                                   const LocalTensor<float>& srcTensor,
                                                                   const LocalTensor<float>& reduceSumTempTensor,
                                                                   const int64_t aSize, const int64_t rSize,
                                                                   const int64_t stride)
{
    if (aSize <= 0) {
        return;
    }
    if (rSize <= 0) {
        return;
    }
    if (rSize <= CONST_TWO * VL_FP32) {
        SecondNormComputePost(dstTensor, srcTensor, srcTensor, aSize, rSize, stride);
        return;
    }

    int64_t ceilVLCount =
        ops::CeilDiv(static_cast<int64_t>(rSize * sizeof(float)), static_cast<int64_t>(GetVRegSize()));
    int64_t floorVLCount =
        ops::FloorDiv(static_cast<int64_t>(rSize * sizeof(float)), static_cast<int64_t>(GetVRegSize()));
    int64_t foldPoint = FindNearestPower2(ceilVLCount);

    uint16_t outerLoopTimes = aSize;
    uint16_t tailFoldLoopTimes = ceilVLCount - floorVLCount;
    uint32_t tailFoldElemCount = static_cast<uint32_t>(rSize - floorVLCount * VL_FP32);
    uint16_t mainFoldLoopTimes = floorVLCount - foldPoint;
    uint16_t unFoldLoopTimes = foldPoint + foldPoint - ceilVLCount;
    uint32_t outerLoopStride = stride;
    uint32_t innerLoopStride = VL_FP32;
    uint32_t outerLoopDstStride =
        ops::Aligned(static_cast<int64_t>(foldPoint), static_cast<int64_t>(GetUbBlockSize() / sizeof(float)));

    int64_t foldSrcBOffset = foldPoint * VL_FP32;
    int64_t tailSrcAOffset = mainFoldLoopTimes * VL_FP32;
    int64_t tailSrcBOffset = floorVLCount * VL_FP32;
    int64_t unFoldSrcOffset = (mainFoldLoopTimes + tailFoldLoopTimes) * VL_FP32;

    __local_mem__ float* dst = (__local_mem__ float*)reduceSumTempTensor.GetPhyAddr();
    __local_mem__ float* foldSrcA = (__local_mem__ float*)srcTensor.GetPhyAddr();
    __local_mem__ float* foldSrcB = (__local_mem__ float*)srcTensor.GetPhyAddr() + foldSrcBOffset;
    __local_mem__ float* tailSrcA = (__local_mem__ float*)srcTensor.GetPhyAddr() + tailSrcAOffset;
    __local_mem__ float* tailSrcB = (__local_mem__ float*)srcTensor.GetPhyAddr() + tailSrcBOffset;
    __local_mem__ float* unFoldSrc = (__local_mem__ float*)srcTensor.GetPhyAddr() + unFoldSrcOffset;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg pFull = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignReg UReg;

        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            dst = (__local_mem__ float*)reduceSumTempTensor.GetPhyAddr() + i * outerLoopDstStride;
            for (uint16_t j = 0; j < mainFoldLoopTimes; ++j) {
                AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg, dReg;
                DataCopy(aReg, (__local_mem__ float*)foldSrcA + i * outerLoopStride + j * innerLoopStride);
                DataCopy(bReg, (__local_mem__ float*)foldSrcB + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pFull);
                ReduceSum(dReg, cReg, pFull);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, dReg, UReg, 1);
            }
            for (uint16_t j = 0; j < tailFoldLoopTimes; ++j) {
                uint32_t count = static_cast<uint32_t>(tailFoldElemCount);
                AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
                AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                DataCopy(aReg, (__local_mem__ float*)tailSrcA + i * outerLoopStride + j * innerLoopStride);
                DataCopy(bReg, (__local_mem__ float*)tailSrcB + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(aReg, cReg, pMask);
                ReduceSum(bReg, aReg, pFull);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, bReg, UReg, 1);
            }
            for (uint16_t j = 0; j < unFoldLoopTimes; ++j) {
                AscendC::MicroAPI::RegTensor<float> aReg, bReg;
                DataCopy(aReg, (__local_mem__ float*)unFoldSrc + i * outerLoopStride + j * innerLoopStride);
                ReduceSum(bReg, aReg, pFull);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, bReg, UReg, 1);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ float*&)dst, UReg, 0);
        }
    }
    SecondNormComputePost(dstTensor, reduceSumTempTensor, srcTensor, aSize, foldPoint, outerLoopDstStride);
}

template <typename T_in, typename T_out>
__aicore__ inline void SoftmaxV2AR<T_in, T_out>::LoadTensorForDtypeTIn(__local_mem__ T_in* src,
                                                                       AscendC::MicroAPI::RegTensor<float>& dst,
                                                                       AscendC::MicroAPI::MaskReg& preg,
                                                                       uint32_t offset)
{
    if constexpr (IsSameType<T_in, float>::value) {
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        AscendC::MicroAPI::RegTensor<T_in> xFp16;
        DataCopy<T_in, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, src + offset);
        Cast<float, T_in, castTraitFp16ToFp32>(dst, xFp16, preg);
    }
}

template <typename T_in, typename T_out>
__aicore__ inline void SoftmaxV2AR<T_in, T_out>::StoreTensorForDtypeTOut(__local_mem__ T_out* dst,
                                                                         AscendC::MicroAPI::RegTensor<float>& src,
                                                                         AscendC::MicroAPI::MaskReg& preg,
                                                                         uint32_t offset)
{
    if constexpr (IsSameType<T_out, float>::value) {
        DataCopy<T_out, AscendC::MicroAPI::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        AscendC::MicroAPI::RegTensor<T_out> xFp16;
        Cast<T_out, float, castTraitFp32ToFp16>(xFp16, src, preg);
        DataCopy<T_out, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}

template <typename T_in, typename T_out>
__aicore__ inline void SoftmaxV2AR<T_in, T_out>::CopyInX(const LocalTensor<T_in>& xInUb, int64_t ubA, int64_t offset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams copyInParams;
    copyInParams.blockCount = ubA;
    copyInParams.blockLen = tl_->r * sizeof(T_in);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (tl_->rAligned - tl_->r) * sizeof(T_in) / BLOCK_SIZE;
    DataCopyPad(xInUb, xGm_[offset], copyInParams, padParams);
}

template <typename T_in, typename T_out>
__aicore__ inline void SoftmaxV2AR<T_in, T_out>::CopyOutY(const LocalTensor<T_out>& yOutUb, int64_t ubA, int64_t offset)
{
    DataCopyParams copyOutParams;
    copyOutParams.blockCount = ubA;
    copyOutParams.blockLen = tl_->r * sizeof(T_out);
    copyOutParams.srcStride = (tl_->rAligned - tl_->r) * sizeof(T_out) / BLOCK_SIZE;
    copyOutParams.dstStride = 0;
    DataCopyPad(yGm_[offset], yOutUb, copyOutParams);
}

}  // namespace SoftmaxV2Ops
#endif  // SOFTMAX_V2_AR_FULL_LOAD_H
