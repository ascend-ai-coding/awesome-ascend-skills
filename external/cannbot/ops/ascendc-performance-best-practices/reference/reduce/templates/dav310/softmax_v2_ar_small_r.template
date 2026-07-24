/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_v2_ar_small_r.h
 * \brief
 */

#ifndef SOFTMAX_V2_AR_SMALL_R_H
#define SOFTMAX_V2_AR_SMALL_R_H

#include "kernel_tiling/kernel_tiling.h"

#include "kernel_operator.h"
#include "softmax_v2_base.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace SoftmaxV2Ops
{
using namespace AscendC;

using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

template <typename Tx, typename Ty>
class SoftmaxV2ArSmallR
{
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;
    static constexpr int64_t DATA_BLOCK_COUNT = 16;
    static constexpr int64_t DATA_BLOCK_COUNT_HALF = 8;
    static constexpr uint32_t VL_FP32 = GetVRegSize() / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = GetUbBlockSize();

public:
    __aicore__ inline SoftmaxV2ArSmallR(TPipe* pipe)
    {
        pipe_ = pipe;
    };

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SoftmaxV2ArSmallRTilingData* tilingDataIn)
    {
        tl_ = tilingDataIn;

        xGm_.SetGlobalBuffer((__gm__ Tx*)x);
        yGm_.SetGlobalBuffer((__gm__ Ty*)y);

        int64_t xShapeLen = tl_->tileA0Len * tl_->rAligned;
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, xShapeLen * sizeof(Tx));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, xShapeLen * sizeof(Ty));

        pipe_->InitBuffer(tmpBuf_, BUFFER_NUM * xShapeLen * sizeof(float) + tl_->tileA0Len * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tl_->tilesPerCore;
        int64_t endIdx = beginIdx + tl_->tilesPerCore;
        endIdx = endIdx > tl_->totalTiles ? tl_->totalTiles : endIdx;
        LocalTensor<float> totalTmpLocal = tmpBuf_.Get<float>();
        tmpLocal_ = totalTmpLocal[0];
        sumLocal_ = totalTmpLocal[BUFFER_NUM * tl_->tileA0Len * tl_->rAligned];

        // preload
        int64_t xOffset = beginIdx * tl_->tileA0Len * tl_->totalRLen;
        CopyInAndTransPose(xOffset, beginIdx == (tl_->totalTiles - 1) ?  tl_->tileA0Tail : tl_->tileA0Len, tl_->totalRLen);

        int curIdx;
        uint32_t curTileA0Len = tl_->tileA0Len;
        uint32_t nextTileA0Len = tl_->tileA0Len;
        for (curIdx = beginIdx; curIdx < endIdx - 1; curIdx++) {
            if (curIdx + 1 == (tl_->totalTiles - 1)) {
                nextTileA0Len = tl_->tileA0Tail;
            }
            int64_t xOffsetPreLoad = (curIdx + 1) * tl_->tileA0Len * tl_->totalRLen;
            xOffset = curIdx * tl_->tileA0Len * tl_->totalRLen;

            CalcMaxSubExp(curTileA0Len, tl_->totalRLen);
            CopyInAndTransPose(xOffsetPreLoad, nextTileA0Len, tl_->totalRLen);      // (curA0, R) -> (R, curA0)
            CalcReduceSum(curTileA0Len, tl_->totalRLen);
            CalcOutput(curTileA0Len, tl_->totalRLen);
            CalcTranspose(curTileA0Len, tl_->rAligned);
            CopyOutY(xOffset, curTileA0Len);
        }

        if (curIdx == (tl_->totalTiles - 1)) {
            curTileA0Len = tl_->tileA0Tail;
        }
        xOffset = curIdx * tl_->tileA0Len * tl_->totalRLen;
        CalcMaxSubExp(curTileA0Len, tl_->totalRLen);
        CalcReduceSum(curTileA0Len, tl_->totalRLen);
        CalcOutput(curTileA0Len, tl_->totalRLen);
        CalcTranspose(curTileA0Len, tl_->rAligned);
        CopyOutY(xOffset, curTileA0Len);
    }

private:
    __aicore__ inline void CalcMaxSubExp(uint32_t curTileA0Len, uint32_t totalRLen)
    {
        LocalTensor<Tx> xLocal_ = xQueue_.DeQue<Tx>();
        __local_mem__ Tx* xAddr = (__local_mem__ Tx*)xLocal_.GetPhyAddr();
        __local_mem__ float* tmpAddr = (__local_mem__ float*)tmpLocal_.GetPhyAddr();
        __local_mem__ float* tmpAddr2 = (__local_mem__ float*)tmpLocal_[tl_->tileA0Len * tl_->rAligned].GetPhyAddr();

        uint16_t aLoopTimes = ops::CeilDiv(curTileA0Len, VL_FP32);
        uint16_t rLoopTimes = static_cast<uint16_t>(totalRLen);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<Tx> reg0;
            MicroAPI::RegTensor<float> reg1, reg2, reg3, maxReg;
            MicroAPI::MaskReg mask;
            uint32_t width = curTileA0Len;
            uint32_t tileA0LenLocal = tl_->tileA0Len;

            for (uint16_t j = 0; j < aLoopTimes; j++) {
                mask = MicroAPI::UpdateMask<uint32_t>(width);
                MicroAPI::Duplicate<float>(maxReg, static_cast<float>(-INFINITY));

                for (uint16_t i = 0; i < rLoopTimes; i++) {
                    uint32_t offset = j * VL_FP32 + i * tileA0LenLocal;
                    LoadTensorForDtypeT(xAddr, reg1, mask, offset);
                    MicroAPI::Max(maxReg, maxReg, reg1, mask);
                }

                for (uint16_t i = 0; i < rLoopTimes; i++) {
                    uint32_t offset = j * VL_FP32 + i * tileA0LenLocal;
                    LoadTensorForDtypeT(xAddr, reg2, mask, offset);
                    MicroAPI::Sub(reg2, reg2, maxReg, mask);
                    MicroAPI::Exp(reg2, reg2, mask);
                    MicroAPI::DataCopy(tmpAddr + offset, reg2, mask);
                    MicroAPI::DataCopy(tmpAddr2 + offset, reg2, mask);
                }
            }
        }

        xQueue_.FreeTensor(xLocal_);
    }

    __aicore__ inline void CalcReduceSum(uint32_t curTileA0Len, uint32_t totalRLen)
    {
        uint32_t srcShape[2] = {uint32_t(totalRLen), uint32_t(tl_->tileA0Len)};
        AscendC::ReduceSum<float, Pattern::Reduce::RA, true>(sumLocal_, tmpLocal_, srcShape, false);
    }

    __aicore__ inline void CalcOutput(uint32_t curTileA0Len, uint32_t totalRLen)
    {
        __local_mem__ float* sumAddr = (__local_mem__ float*)sumLocal_.GetPhyAddr();
        __local_mem__ float* tmpAddr2 = (__local_mem__ float*)tmpLocal_[tl_->tileA0Len * tl_->rAligned].GetPhyAddr();
        tmpLocalTy_ = tmpLocal_.template ReinterpretCast<Ty>();
        __local_mem__ Ty* tmpAddrTy = (__local_mem__ Ty*)tmpLocalTy_.GetPhyAddr();

        uint16_t aLoopTimes = static_cast<uint16_t>(ops::CeilDiv(curTileA0Len, VL_FP32));
        uint16_t rLoopTimes = static_cast<uint16_t>(tl_->totalRLen);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> reg1;
            MicroAPI::RegTensor<float> sumReg;
            MicroAPI::MaskReg mask;

            uint32_t sreg = curTileA0Len;
            uint32_t tileA0LenLocal = tl_->tileA0Len;

            for (uint16_t j = 0; j < aLoopTimes; j++) { // 列
                mask = MicroAPI::UpdateMask<float>(sreg);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(sumReg, (__local_mem__ float*)sumAddr + j * VL_FP32);

                for (uint16_t i = 0; i < rLoopTimes; i++) { // 行
                    uint32_t offset = j * VL_FP32 + i * tileA0LenLocal;

                    MicroAPI::DataCopy(reg1, tmpAddr2 + offset);
                    MicroAPI::Div(reg1, reg1, sumReg, mask);

                    if constexpr (yToFp32_) {
                        MicroAPI::DataCopy(tmpAddrTy + offset, reg1, mask);
                    } else {  // fp16、bf16
                        MicroAPI::RegTensor<Ty> xFp16;
                        MicroAPI::Cast<Ty, float, castTraitFp32ToFp16>(xFp16, reg1, mask);
                        MicroAPI::DataCopy<Ty, MicroAPI::StoreDist::DIST_PACK_B32>(tmpAddrTy + offset, xFp16, mask);
                    }
                }
            }
        }
    }

    __aicore__ inline void CalcTransposeFp32(const LocalTensor<Ty>& yLocal, uint32_t curTileA0Len, uint32_t rAligned)
    {
        int rRepeartTimes = ops::CeilDiv(static_cast<int64_t>(rAligned), DATA_BLOCK_COUNT_HALF);
        for (int i = 0; i < rRepeartTimes; i++) {
            TransDataTo5HDParams params;
            LocalTensor<Ty> srcLocalList[DATA_BLOCK_COUNT];
            LocalTensor<Ty> dstLocalList[DATA_BLOCK_COUNT];

            uint32_t aRepeartTimes = ops::CeilDiv(curTileA0Len, uint32_t(DATA_BLOCK_COUNT));
            params.repeatTimes = aRepeartTimes;
            params.srcRepStride = aRepeartTimes == 1 ? 0 : CONST_TWO;
            params.dstRepStride = aRepeartTimes == 1 ? 0 : DATA_BLOCK_COUNT * rRepeartTimes;
            for (int j = 0; j < DATA_BLOCK_COUNT_HALF; j++) {
                uint32_t offset = DATA_BLOCK_COUNT_HALF * tl_->tileA0Len * i + tl_->tileA0Len * j;
                srcLocalList[j] = tmpLocalTy_[offset];
                srcLocalList[j + DATA_BLOCK_COUNT_HALF] = tmpLocalTy_[offset + DATA_BLOCK_COUNT_HALF];
            }
            for (int j = 0; j < DATA_BLOCK_COUNT_HALF; j++) {
                uint32_t offset = DATA_BLOCK_COUNT_HALF * i + DATA_BLOCK_COUNT_HALF * rRepeartTimes * j;
                dstLocalList[j * CONST_TWO] = yLocal[offset];
                dstLocalList[j * CONST_TWO + 1] = yLocal[offset + DATA_BLOCK_COUNT_HALF * DATA_BLOCK_COUNT_HALF * rRepeartTimes];
            }

            AscendC::TransDataTo5HD(dstLocalList, srcLocalList, params);
        }
    }

    __aicore__ inline void CalcTransposeFp16(const LocalTensor<Ty>& yLocal, uint32_t curTileA0Len, uint32_t rAligned)
    {
        int rRepeartTimes = ops::CeilDiv(static_cast<int64_t>(rAligned), tl_->rTileBase);
        for (int i = 0; i < rRepeartTimes; i++) {
            TransDataTo5HDParams params;
            int64_t aRepeartTimes = ops::CeilDiv(static_cast<int64_t>(curTileA0Len), tl_->rTileBase);
            params.repeatTimes = aRepeartTimes;
            params.srcRepStride = aRepeartTimes == 1 ? 0 : 1;
            params.dstRepStride = aRepeartTimes == 1 ? 0 : (tl_->rTileBase * rRepeartTimes);
            LocalTensor<Ty> srcLocalList[DATA_BLOCK_COUNT];
            LocalTensor<Ty> dstLocalList[DATA_BLOCK_COUNT];
            for (int j = 0; j < DATA_BLOCK_COUNT; j++) {
                uint32_t offset = tl_->rTileBase * tl_->tileA0Len * i + tl_->tileA0Len * j;
                srcLocalList[j] = tmpLocalTy_[offset];
            }
            for (int j = 0; j < DATA_BLOCK_COUNT; j++) {
                uint32_t offset = tl_->rTileBase * i + rAligned * j;
                dstLocalList[j] = yLocal[offset];
            }

            AscendC::TransDataTo5HD<Ty>(dstLocalList, srcLocalList, params);
        }
    }

    __aicore__ inline void CalcTranspose(uint32_t curTileA0Len, uint32_t rAligned)
    {
        LocalTensor<Ty> yLocal = yQueue_.AllocTensor<Ty>();

        if constexpr (yToFp32_) {
            CalcTransposeFp32(yLocal, curTileA0Len, rAligned);
        } else {
            CalcTransposeFp16(yLocal, curTileA0Len, rAligned);
        }

        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void LoadTensorForDtypeT(const __local_mem__ Tx* src, RegTensor<float>& dst, MaskReg& preg,
                                               uint32_t offset)
    {  
        if constexpr (xToFp32_) {
            MicroAPI::RegTensor<Tx> xFp16;
            MicroAPI::DataCopy<Tx, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ Tx*)src + offset));
            MicroAPI::Cast<float, Tx, castTraitFp16ToFp32>(dst, xFp16, preg);
        } else {
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
        }
    }

    __aicore__ inline void CopyInAndTransPose(int64_t xGmOffset, uint32_t curTileA0Len, uint32_t totalRLen)
    {
        static constexpr MultiCopyConfig config = {false};
        MultiCopyLoopInfo<CONST_TWO> copyLoopInfo;
        copyLoopInfo.loopSrcStride[0] = 1;
        copyLoopInfo.loopSrcStride[1] = totalRLen;
        copyLoopInfo.loopDstStride[0] = tl_->tileA0Len;
        copyLoopInfo.loopDstStride[1] = 1;
        copyLoopInfo.loopSize[0] = totalRLen;
        copyLoopInfo.loopSize[1] = curTileA0Len;
        MultiCopyParams<Tx, CONST_TWO> params = {copyLoopInfo, 0};

        LocalTensor<Tx> xLocal_ = xQueue_.AllocTensor<Tx>();
        DataCopy<Tx, CONST_TWO, config>(xLocal_, xGm_[xGmOffset], params);
        xQueue_.EnQue(xLocal_);
    }

    __aicore__ inline void CopyOutFp32(const LocalTensor<Ty>& yLocal, int64_t yGmOffset, uint32_t curTileA0Len)
    {
        DataCopyParams copyOutParams;
        copyOutParams.blockCount = curTileA0Len;
        copyOutParams.blockLen = tl_->totalRLen * sizeof(Ty);
        copyOutParams.dstStride = 0;
        copyOutParams.srcStride = 0;
        DataCopyPad(yGm_[yGmOffset], yLocal[0], copyOutParams);
    }

    __aicore__ inline void CopyOutFp16(const LocalTensor<Ty>& yLocal, int64_t yGmOffset, uint32_t curTileA0Len)
    {
        DataCopyParams copyOutParams;
        copyOutParams.blockCount = curTileA0Len;
        copyOutParams.blockLen =  tl_->totalRLen * sizeof(Ty);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad(yGm_[yGmOffset], yLocal[0], copyOutParams);
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, uint32_t curTileA0Len)
    {
        LocalTensor<Ty> yLocal = yQueue_.DeQue<Ty>();

        if constexpr (yToFp32_) {
            CopyOutFp32(yLocal, yGmOffset, curTileA0Len);
        } else {
            CopyOutFp16(yLocal, yGmOffset, curTileA0Len);
        }

        yQueue_.FreeTensor(yLocal);
    }

private:
    const SoftmaxV2ArSmallRTilingData* tl_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> xQueue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> yQueue_;

    GlobalTensor<Tx> xGm_;
    GlobalTensor<Ty> yGm_;

    TBuf<TPosition::VECCALC> tmpBuf_;

    LocalTensor<float> sumLocal_;
    LocalTensor<float> tmpLocal_;
    LocalTensor<Ty> tmpLocalTy_;

    static constexpr bool xToFp32_ = !IsSameType<Tx, float>::value;
    static constexpr bool yToFp32_ = IsSameType<Ty, float>::value;
};


}  // namespace SoftmaxV2Ops

#endif