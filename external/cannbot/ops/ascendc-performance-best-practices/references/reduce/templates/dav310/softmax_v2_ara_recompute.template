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
 * \file softmax_v2_ara_recompute.h
 * \brief
 */

#ifndef SOFTMAX_V2_ARA_RECOMPUTE_H
#define SOFTMAX_V2_ARA_RECOMPUTE_H

#include "kernel_tiling/kernel_tiling.h"

#include "kernel_operator.h"
#include "softmax_v2_base.h"

namespace SoftmaxV2Ops
{
using namespace AscendC;
using namespace AscendC::MicroAPI;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;

template <typename T1, typename T2>
class SoftmaxV2ARARecompute : public SoftmaxV2OpsBase
{
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = GetVRegSize();
    static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = GetUbBlockSize();

public:
    __aicore__ inline SoftmaxV2ARARecompute(){};

    __aicore__ inline SoftmaxV2ARARecompute(const SoftmaxV2ARARecomputeTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TPipe* pipeIn)
    {
        pipe_ = pipeIn;

        xGm_.SetGlobalBuffer((__gm__ T1*)x);
        yGm_.SetGlobalBuffer((__gm__ T2*)y);

        int64_t xShapeLen = tilingData_->tileA0Len * tilingData_->binAddRFactor;
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, xShapeLen * sizeof(T1));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, xShapeLen * sizeof(float));

        pipe_->InitBuffer(xMaxBuf_, tilingData_->tileA0Len * sizeof(float));
        pipe_->InitBuffer(xSumBuf_, tilingData_->tileA0Len * sizeof(float));

        pipe_->InitBuffer(cacheBuffer_, tilingData_->binAddCacheBufferCount * tilingData_->tileA0Len * sizeof(float));
        pipe_->InitBuffer(tempBuffer_, tilingData_->tileA0Len * sizeof(float));
        pipe_->InitBuffer(reduceSumTempBuffer_, CONST_EIGHT * tilingData_->tileA0Len * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        // pattern is [A1, R, A0]
        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curA0Idx = curIdx % tilingData_->tileA0Outer;
            int64_t curA1Idx = curIdx / tilingData_->tileA0Outer;

            uint32_t curTileA0Len =
                curA0Idx == (tilingData_->tileA0Outer - 1) ? tilingData_->tileA0Tail : tilingData_->tileA0Len;

            int64_t xOffset =
                // a1 offset
                curA1Idx * tilingData_->totalRLen * tilingData_->totalA0Len +
                // a0 offset
                curA0Idx * tilingData_->tileA0Len;

            uint16_t loopA0Num = ops::CeilDiv(curTileA0Len, VL_FP32);

            CalcReduceMax(curTileA0Len, xOffset, loopA0Num);
            CalcReduceSum(curTileA0Len, xOffset, loopA0Num);

            for (int64_t idx = 0; idx < tilingData_->binAddRTotalLoop; idx++) {
                int64_t curTileRLen =
                    idx == (tilingData_->binAddRTotalLoop - 1) ? tilingData_->binAddRTail : tilingData_->binAddRFactor;

                int64_t offset = xOffset + idx * tilingData_->binAddRFactor * tilingData_->totalA0Len;
                CopyInX(offset, curTileRLen, curTileA0Len);
                CalcOutput(curTileRLen, curTileA0Len, loopA0Num);
                CopyOutY(offset, curTileRLen, curTileA0Len);
            }
        }
    }

private:
    __aicore__ inline void CalcReduceMax(uint32_t curTileA0Len, int64_t xOffsetStart, uint16_t loopA0Num)
    {
        // max 初始化
        LocalTensor<float> xMaxTensor = xMaxBuf_.Get<float>();
        __local_mem__ float* xMaxLocal = (__local_mem__ float*)xMaxTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            RegTensor<float> maxReg;
            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;
            for (uint16_t k = 0; k < loopA0Num; k++) {
                pregMask = UpdateMask<float>(sreg);
                Duplicate(maxReg, static_cast<float>(-INFINITY), pregMask);
                DataCopy(((__local_mem__ float*)xMaxLocal) + k * VL_FP32, maxReg, pregMask);
            }
        }

        uint32_t tileA0Len = tilingData_->tileA0Len;

        for (int64_t idx = 0; idx < tilingData_->binAddRTotalLoop; idx++) {
            int64_t curTileRLen =
                idx == (tilingData_->binAddRTotalLoop - 1) ? tilingData_->binAddRTail : tilingData_->binAddRFactor;
            uint16_t curTileRLenVl = static_cast<uint16_t>(curTileRLen);
            int64_t xOffset = xOffsetStart + idx * tilingData_->binAddRFactor * tilingData_->totalA0Len;

            CopyInX(xOffset, curTileRLen, curTileA0Len);
            LocalTensor<T1> x = xQueue_.DeQue<T1>();
            __local_mem__ T1* xLocal = (__local_mem__ T1*)x.GetPhyAddr();

            __VEC_SCOPE__
            {
                RegTensor<float> x;
                RegTensor<float> maxReg;

                MaskReg pregMask;
                uint32_t sreg = curTileA0Len;

                for (uint16_t k = 0; k < loopA0Num; k++) {
                    pregMask = UpdateMask<float>(sreg);
                    // load max
                    DataCopy<float, LoadDist::DIST_NORM>(maxReg, (__local_mem__ float*)xMaxLocal + k * VL_FP32);

                    for (uint16_t i = 0; i < curTileRLenVl; i++) {
                        uint32_t offset = i * tileA0Len + k * VL_FP32;
                        // load x
                        LoadTensorForDtypeT1(xLocal, x, pregMask, offset);
                        Max(maxReg, maxReg, x, pregMask);
                    }
                    DataCopy(((__local_mem__ float*)xMaxLocal) + k * VL_FP32, maxReg, pregMask);
                }
            }

            xQueue_.FreeTensor<T1>(x);
        }
    }

    __aicore__ inline void CalcReduceSum(uint32_t curTileA0Len, int64_t xOffsetStart, uint16_t loopA0Num)
    {
        int64_t xSrcStride = tilingData_->totalA0Len;

        tempTensor_ = tempBuffer_.Get<float>();
        cacheTensor_ = cacheBuffer_.Get<float>();
        reduceSumTempTensor_ = reduceSumTempBuffer_.Get<float>();

        for (int64_t basicBlockIdx = 0; basicBlockIdx < tilingData_->binAddBasicBlockLoop; basicBlockIdx++) {
            ProcessMainBlock(tilingData_->binAddRFactor, curTileA0Len,
                             xOffsetStart + basicBlockIdx * xSrcStride * tilingData_->binAddRFactor, loopA0Num);
            if (basicBlockIdx < tilingData_->binAddMainFoldCount) {
                ProcessFoldBlock(tilingData_->binAddRFactor, curTileA0Len,
                                 xOffsetStart + (basicBlockIdx + tilingData_->binAddBasicBlockLoop) * xSrcStride *
                                                    tilingData_->binAddRFactor,
                                 loopA0Num);
            } else if (basicBlockIdx == tilingData_->binAddMainFoldCount && tilingData_->binAddRTail > 0 &&
                       tilingData_->binAddRTail != tilingData_->binAddRFactor) {
                ProcessFoldBlock(tilingData_->binAddRTail, curTileA0Len,
                                 xOffsetStart + (basicBlockIdx + tilingData_->binAddBasicBlockLoop) * xSrcStride *
                                                    tilingData_->binAddRFactor,
                                 loopA0Num);
            }
            ProcessSummation(basicBlockIdx, tilingData_->binAddRFactor);
        }

        if (tilingData_->binAddBasicBlockLoop == 0) {
            ProcessMainBlock(tilingData_->binAddRTail, curTileA0Len, xOffsetStart, loopA0Num);
            ProcessSummation(0, tilingData_->binAddRTail);
        }
        LocalTensor<float> xSumTensor = xSumBuf_.Get<float>();
        DataCopy(xSumTensor, cacheTensor_[tilingData_->binAddResultCacheID * tilingData_->tileA0Len],
                 tilingData_->tileA0Len);
    }

    // copy to main
    __aicore__ inline void ProcessMainBlock(const int64_t curTileRLen, const int64_t curTileA0Len,
                                            const int64_t gmOffset, uint16_t loopA0Num)
    {
        LocalTensor<T1> xMain_ = xQueue_.AllocTensor<T1>();
        CopyIn(xMain_, xGm_[gmOffset], curTileRLen, curTileA0Len);
        xQueue_.EnQue(xMain_);
        xMain_ = xQueue_.DeQue<T1>();
        yMain_ = yQueue_.AllocTensor<float>();
        LocalTensor<float> xMaxTensor = xMaxBuf_.Get<float>();

        uint16_t outerLoopTimes = static_cast<uint16_t>(curTileRLen);
        uint32_t outerLoopSrcStride = tilingData_->tileA0Len;

        __local_mem__ float* dst = (__local_mem__ float*)yMain_.GetPhyAddr();
        __local_mem__ T1* src = (__local_mem__ T1*)xMain_.GetPhyAddr();
        __local_mem__ float* xMaxLocal = (__local_mem__ float*)xMaxTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;

            RegTensor<float> srcReg;
            RegTensor<float> maxReg;
            RegTensor<float> dstReg;

            for (uint16_t j = 0; j < loopA0Num; ++j) {
                pregMask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(maxReg, (__local_mem__ float*)xMaxLocal + j * VL_FP32);
                for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                    uint32_t xOffset = i * outerLoopSrcStride + j * VL_FP32;
                    LoadTensorForDtypeT1(src, srcReg, pregMask, xOffset);
                    Sub(dstReg, srcReg, maxReg, pregMask);
                    Exp(dstReg, dstReg, pregMask);
                    DataCopy((__local_mem__ float*)dst + xOffset, dstReg, pregMask);
                }
            }
        }

        xQueue_.FreeTensor(xMain_);
    }

    // fold cast fp32 加到main
    __aicore__ inline void ProcessFoldBlock(const int64_t curTileRLen, const int64_t curTileA0Len,
                                            const int64_t gmOffset, uint16_t loopA0Num)
    {
        LocalTensor<T1> xFold = xQueue_.AllocTensor<T1>();
        LocalTensor<float> xMaxTensor = xMaxBuf_.Get<float>();
        CopyIn(xFold, xGm_[gmOffset], curTileRLen, curTileA0Len);
        xQueue_.EnQue(xFold);
        xFold = xQueue_.DeQue<T1>();

        uint16_t outerLoopTimes = static_cast<uint16_t>(curTileRLen);
        uint32_t outerLoopSrcStride = tilingData_->tileA0Len;

        __local_mem__ float* dst = (__local_mem__ float*)yMain_.GetPhyAddr();
        __local_mem__ T1* src = (__local_mem__ T1*)xFold.GetPhyAddr();
        __local_mem__ float* xMaxLocal = (__local_mem__ float*)xMaxTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;

            RegTensor<float> srcReg;
            RegTensor<float> maxReg;
            RegTensor<float> dstReg;

            for (uint16_t j = 0; j < loopA0Num; ++j) {
                pregMask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(maxReg, (__local_mem__ float*)xMaxLocal + j * VL_FP32);
                for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                    uint32_t xOffset = i * outerLoopSrcStride + j * VL_FP32;
                    LoadTensorForDtypeT1(src, srcReg, pregMask, i * outerLoopSrcStride + j * VL_FP32);
                    Sub(dstReg, srcReg, maxReg, pregMask);
                    Exp(dstReg, dstReg, pregMask);
                    DataCopy(srcReg, (__local_mem__ float*)dst + xOffset);
                    Add(dstReg, dstReg, srcReg, pregMask);
                    DataCopy((__local_mem__ float*)dst + xOffset, dstReg, pregMask);
                }
            }
        }

        xQueue_.FreeTensor(xFold);
    }

    __aicore__ inline void CopyIn(const LocalTensor<T1>& dstTensor, const GlobalTensor<T1>& srcTensor,
                                  const int64_t curTileRLen, const int64_t curTileA0Len)
    {
        DataCopyExtParams params;
        params.blockCount = curTileRLen;
        params.blockLen = curTileA0Len * sizeof(T1);
        params.srcStride = (tilingData_->totalA0Len - curTileA0Len) * sizeof(T1);
        params.dstStride = (tilingData_->tileA0Len * sizeof(T1) - params.blockLen) / BLOCK_SIZE;
        DataCopyPadExtParams<T1> padParams;
        padParams.isPad = false;
        DataCopyPad(dstTensor, srcTensor, params, padParams);
    }

    __aicore__ inline void ProcessSummation(const int64_t basicBlockIdx, int64_t binAddRLen)
    {
        int64_t cacheID = GetCacheID(basicBlockIdx);
        NlastReduceSum(tempTensor_, yMain_, reduceSumTempTensor_, binAddRLen, tilingData_->tileA0Len,
                       tilingData_->tileA0Len);
        yQueue_.FreeTensor(yMain_);
        UpdateCache(cacheTensor_, tempTensor_, cacheID, tilingData_->tileA0Len, tilingData_->tileA0Len);
    }

    __aicore__ inline void CopyInX(int64_t xGmOffset, int64_t curTileRLen, uint32_t curTileA0Len)
    {
        LocalTensor<T1> xLocal = xQueue_.AllocTensor<T1>();
        DataCopyPadExtParams<T1> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileRLen;
        copyInParams.blockLen = curTileA0Len * sizeof(T1);
        copyInParams.srcStride = (tilingData_->totalA0Len - curTileA0Len) * sizeof(T1);
        copyInParams.dstStride = (tilingData_->tileA0Len - curTileA0Len) * sizeof(T1) / BLOCK_SIZE;
        DataCopyPad<T1, PaddingMode::Normal>(xLocal, xGm_[xGmOffset], copyInParams, dataCopyPadExtParams);

        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CalcOutput(int64_t curTileRLen, uint32_t curTileA0Len, uint16_t loopA0Num)
    {
        LocalTensor<T1> x = xQueue_.DeQue<T1>();
        __local_mem__ T1* xLocal = (__local_mem__ T1*)x.GetPhyAddr();

        LocalTensor<T2> y = yQueue_.template AllocTensor<T2>();
        __local_mem__ T2* yLocal = (__local_mem__ T2*)y.GetPhyAddr();

        LocalTensor<float> xMaxTensor = xMaxBuf_.Get<float>();
        __local_mem__ float* xMaxLocal = (__local_mem__ float*)xMaxTensor.GetPhyAddr();

        LocalTensor<float> xSumTensor = xSumBuf_.Get<float>();
        __local_mem__ float* xSumLocal = (__local_mem__ float*)xSumTensor.GetPhyAddr();

        uint32_t tileA0Len = tilingData_->tileA0Len;
        uint16_t curTileRLenVl = static_cast<uint16_t>(curTileRLen);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> yReg;
            RegTensor<float> sumReg;
            RegTensor<float> maxReg;

            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;

            for (uint16_t k = 0; k < loopA0Num; k++) {
                pregMask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(sumReg, (__local_mem__ float*)xSumLocal + k * VL_FP32);
                DataCopy<float, LoadDist::DIST_NORM>(maxReg, (__local_mem__ float*)xMaxLocal + k * VL_FP32);
                for (uint16_t i = 0; i < curTileRLenVl; i++) {
                    uint32_t xOffset = i * tileA0Len + k * VL_FP32;
                    LoadTensorForDtypeT1(xLocal, xReg, pregMask, xOffset);

                    Sub(xReg, xReg, maxReg, pregMask);
                    Exp(xReg, xReg, pregMask);
                    Div(yReg, xReg, sumReg, pregMask);

                    // copy out
                    if constexpr (IsSameType<T2, float>::value) {
                        DataCopy(((__local_mem__ float*)yLocal) + xOffset, yReg, pregMask);
                    } else {  // fp16、bf16
                        RegTensor<T2> xFp16;
                        Cast<T2, float, castTraitFp32ToFp16>(xFp16, yReg, pregMask);
                        DataCopy<T2, StoreDist::DIST_PACK_B32>(((__local_mem__ T2*)yLocal) + xOffset, xFp16, pregMask);
                    }
                }
            }
        }

        yQueue_.EnQue(y);

        xQueue_.FreeTensor<T1>(x);
    }

    __aicore__ inline void LoadTensorForDtypeT1(__local_mem__ T1* src, RegTensor<float>& dst, MaskReg& preg,
                                                uint32_t offset)
    {
        if constexpr (IsSameType<T1, float>::value) {
            DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
        } else {  // fp16、bf16
            RegTensor<T1> xFp16;
            DataCopy<T1, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T1*)src + offset));
            Cast<float, T1, castTraitFp16ToFp32>(dst, xFp16, preg);
        }
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, int64_t curTileRLen, uint32_t curTileA0Len)
    {
        LocalTensor<T2> y = yQueue_.DeQue<T2>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileRLen;
        copyInParams.blockLen = curTileA0Len * sizeof(T2);
        copyInParams.srcStride = (tilingData_->tileA0Len - curTileA0Len) * sizeof(T2) / BLOCK_SIZE;
        copyInParams.dstStride = (tilingData_->totalA0Len - curTileA0Len) * sizeof(T2);
        DataCopyPad<T2, PaddingMode::Normal>(yGm_[yGmOffset], y, copyInParams);
        yQueue_.FreeTensor(y);
    }

private:
    const SoftmaxV2ARARecomputeTilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> xQueue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> yQueue_;

    TBuf<TPosition::VECCALC> xMaxBuf_;
    TBuf<TPosition::VECCALC> xSumBuf_;

    GlobalTensor<T2> yGm_;
    GlobalTensor<T1> xGm_;

    // bin add
    TBuf<TPosition::VECCALC> cacheBuffer_;
    TBuf<TPosition::VECCALC> tempBuffer_;
    TBuf<TPosition::VECCALC> reduceSumTempBuffer_;

    LocalTensor<float> tempTensor_;
    LocalTensor<float> cacheTensor_;
    LocalTensor<float> reduceSumTempTensor_;

    LocalTensor<float> yMain_;
};
}  // namespace SoftmaxV2Ops

#endif
