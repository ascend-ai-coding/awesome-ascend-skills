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
 * \file softmax_v2_ara_full_load.h
 * \brief
 */

#ifndef SOFTMAX_V2_ARA_FULL_LOAD_H
#define SOFTMAX_V2_ARA_FULL_LOAD_H

#include "kernel_tiling/kernel_tiling.h"

#include "kernel_operator.h"
#include "softmax_v2_base.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace SoftmaxV2Ops
{
using namespace AscendC;
using namespace AscendC::MicroAPI;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;

constexpr int64_t SCALE_COEF_TWO = 2;
constexpr int64_t SCALE_COEF_FOUR = 4;
constexpr int64_t SCALE_COEF_EIGHT = 8;

constexpr int64_t BIN_ADD_THRES = 8;
constexpr int64_t BIN_ADD_PADDING = 1;

constexpr uint16_t ROW_ZERO = 0;
constexpr uint16_t ROW_ONE = 1;
constexpr uint16_t ROW_TWO = 2;
constexpr uint16_t ROW_THREE = 3;
constexpr uint16_t ROW_FOUR = 4;
constexpr uint16_t ROW_FIVE = 5;
constexpr uint16_t ROW_SIX = 6;
constexpr uint16_t ROW_SEVEN = 7;

constexpr uint32_t ROW_TWO_OFFSET = 2;
constexpr uint32_t ROW_THREE_OFFSET = 3;
constexpr uint32_t ROW_FOUR_OFFSET = 4;
constexpr uint32_t ROW_FIVE_OFFSET = 5;
constexpr uint32_t ROW_SIX_OFFSET = 6;
constexpr uint32_t ROW_SEVEN_OFFSET = 7;

template <typename T1, typename T2>
class SoftmaxV2ARA
{
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = GetVRegSize();
    static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = GetUbBlockSize();

public:
    __aicore__ inline SoftmaxV2ARA(){};

    __aicore__ inline SoftmaxV2ARA(const SoftmaxV2ARATilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TPipe* pipeIn)
    {
        pipe_ = pipeIn;

        xGm_.SetGlobalBuffer((__gm__ T1*)x);
        yGm_.SetGlobalBuffer((__gm__ T2*)y);

        int64_t xShapeLen = tilingData_->tileA0Len * tilingData_->totalRLen;
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, xShapeLen * sizeof(T1));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, xShapeLen * sizeof(float));

        pipe_->InitBuffer(xTmpBuf_,
                          tilingData_->tileA0Len * (tilingData_->totalRLen + BIN_ADD_PADDING) * sizeof(float));
        pipe_->InitBuffer(xReduceBuf_, tilingData_->tileA0Len * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curA0Idx = curIdx % tilingData_->a0Outer;
            int64_t curA1Idx = curIdx / tilingData_->a0Outer;

            uint32_t curTileA0Len =
                curA0Idx == (tilingData_->a0Outer - 1) ? tilingData_->tileA0Tail : tilingData_->tileA0Len;

            int64_t xOffset =
                // a1 offset
                curA1Idx * tilingData_->totalRLen * tilingData_->totalA0Len +
                // a0 offset
                curA0Idx * tilingData_->tileA0Len;

            CopyInX(xOffset, tilingData_->totalRLen, curTileA0Len);
            Compute(tilingData_->totalRLen, curTileA0Len);
            CopyOutY(xOffset, tilingData_->totalRLen, curTileA0Len);
        }
    }

private:
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

    __aicore__ inline void Compute(int64_t curTileRLen, uint32_t curTileA0Len)
    {
        LocalTensor<T1> x = xQueue_.DeQue<T1>();
        __local_mem__ T1* xLocal = (__local_mem__ T1*)x.GetPhyAddr();

        LocalTensor<float> xTmpTensor = xTmpBuf_.Get<float>();
        __local_mem__ float* xTmpLocal = (__local_mem__ float*)xTmpTensor.GetPhyAddr();

        uint16_t loopA0Num = ops::CeilDiv(curTileA0Len, VL_FP32);

        VFShiftVector(xTmpLocal, xLocal, curTileRLen, curTileA0Len, loopA0Num);
        xQueue_.FreeTensor<T1>(x);

        LocalTensor<float> y = yQueue_.AllocTensor<float>();
        __local_mem__ float* yLocal = (__local_mem__ float*)y.GetPhyAddr();

        LocalTensor<float> xReduceTensor = xReduceBuf_.Get<float>();
        __local_mem__ float* xReduceLocal = (__local_mem__ float*)xReduceTensor.GetPhyAddr();

        VFReduceSum(xReduceLocal, xTmpLocal, yLocal, curTileRLen, curTileA0Len);

        VFCalculateOutput(yLocal, xTmpLocal, xReduceLocal, curTileRLen, curTileA0Len, loopA0Num);

        yQueue_.EnQue(y);
    }

    __aicore__ inline void VFShiftVector(__local_mem__ float* xTmpLocal, __local_mem__ T1* xLocal, uint16_t curTileRLen,
                                         uint16_t curTileA0Len, uint16_t loopA0Num)
    {
        uint32_t tileA0Len = tilingData_->tileA0Len;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> maxReg;

            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;

            for (uint16_t k = 0; k < loopA0Num; k++) {
                pregMask = UpdateMask<float>(sreg);
                // get max
                Duplicate(maxReg, static_cast<float>(-INFINITY), pregMask);
                for (uint16_t i = 0; i < curTileRLen; i++) {
                    uint32_t xOffset = i * tileA0Len + k * VL_FP32;
                    // load x
                    LoadTensorForDtypeT1(xLocal, x, pregMask, xOffset);
                    Max(maxReg, maxReg, x, pregMask);
                }
                // shift vector
                for (uint16_t i = 0; i < curTileRLen; i++) {
                    uint32_t xOffset1 = i * tileA0Len + k * VL_FP32;
                    LoadTensorForDtypeT1(xLocal, x, pregMask, xOffset1);
                    Sub(x, x, maxReg, pregMask);
                    Exp(x, x, pregMask);
                    DataCopy(((__local_mem__ float*)xTmpLocal) + xOffset1, x, pregMask);
                }
            }
        }
    }

    __aicore__ inline void VFReduceSum(__local_mem__ float* xReduceLocal, __local_mem__ float* xTmpLocal,
                                       __local_mem__ float* yInUb, uint16_t curTileRLen, uint16_t curTileA0Len)
    {
        if (tilingData_->totalRLen <= SCALE_COEF_TWO) {
            SumRLessThan2(xTmpLocal, xReduceLocal, curTileA0Len);
        } else if (tilingData_->totalRLen <= SCALE_COEF_FOUR) {
            SumRLessThan4(xTmpLocal, xReduceLocal, curTileA0Len);
        } else if (tilingData_->totalRLen <= SCALE_COEF_EIGHT) {
            SumRLessThan8(xTmpLocal, xReduceLocal, curTileA0Len);
        } else {
            SumRMoreThan8(xTmpLocal, yInUb, xReduceLocal, curTileA0Len);
        }
    }

    __aicore__ inline void SumRLessThan2(__local_mem__ float* xTmpLocal, __local_mem__ float* xReduceLocal,
                                         uint32_t curTileA0Len)
    {
        uint32_t rStride = tilingData_->tileA0Len;
        uint16_t rLoopCount = tilingData_->totalRLen;

        uint16_t aLoopCount = ops::CeilDiv(curTileA0Len, VL_FP32);

        __VEC_SCOPE__
        {
            RegTensor<float> xld;
            RegTensor<float> sum;
            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                Duplicate(sum, 0.0, pregLoop);
                for (uint16_t i = 0; i < rLoopCount; i++) {
                    DataCopy(xld, ((__local_mem__ float*)xTmpLocal + i * rStride + k * VL_FP32));
                    Add(sum, sum, xld, pregLoop);
                }
                DataCopy(((__local_mem__ float*)xReduceLocal + k * VL_FP32), sum, pregLoop);
            }
        }
    }

    __aicore__ inline void SumRLessThan4(__local_mem__ float* xTmpLocal, __local_mem__ float* xReduceLocal,
                                         uint32_t curTileA0Len)
    {
        uint32_t remainderOffset = SCALE_COEF_TWO * tilingData_->tileA0Len;
        uint32_t aLength = tilingData_->tileA0Len;
        uint32_t validNumInXUb = tilingData_->totalRLen * tilingData_->tileA0Len;

        uint16_t remainderTailCount = tilingData_->totalRLen - SCALE_COEF_TWO;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;

        uint16_t aLoopCount = ops::CeilDiv(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> zero;
            Duplicate(zero, 0.0, pregMain);

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(((__local_mem__ float*)xTmpLocal + validNumInXUb + aLoopOffset), zero, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                TwoRowAddWithTail(x1, xTmpLocal, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset,
                                  aLength + aLoopOffset, remainderTailOffset1 + aLoopOffset, rem, nextRow, remNextRow);
                DataCopy(((__local_mem__ float*)xReduceLocal + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void TwoRowAddWithTail(RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg,
                                             uint32_t offset1, uint32_t offset2, uint32_t offset3, uint32_t offset4,
                                             RegTensor<float>& rem, RegTensor<float>& nextRow,
                                             RegTensor<float>& remNextRow)
    {
        DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
        DataCopy(rem, ((__local_mem__ float*)(input) + (offset2)));
        Add(dst, dst, rem, preg);
        DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset3)));
        DataCopy(remNextRow, ((__local_mem__ float*)(input) + (offset4)));
        Add(nextRow, nextRow, remNextRow, preg);
        Add(dst, dst, nextRow, preg);
    }

    __aicore__ inline void SumRLessThan8(__local_mem__ float* xTmpLocal, __local_mem__ float* xReduceLocal,
                                         uint32_t curTileA0Len)
    {
        uint32_t remainderOffset = SCALE_COEF_FOUR * tilingData_->tileA0Len;
        uint32_t aLength = tilingData_->tileA0Len;
        uint32_t validNumInXUb = tilingData_->totalRLen * tilingData_->tileA0Len;

        uint16_t remainderTailCount = tilingData_->totalRLen - SCALE_COEF_FOUR;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;
        uint32_t remainderTailOffset2 =
            (ROW_TWO > remainderTailCount) ? validNumInXUb : remainderOffset + ROW_TWO_OFFSET * aLength;
        uint32_t remainderTailOffset3 =
            (ROW_THREE > remainderTailCount) ? validNumInXUb : remainderOffset + ROW_THREE_OFFSET * aLength;

        uint16_t aLoopCount = ops::CeilDiv(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> zero;
            Duplicate(zero, 0.0, pregMain);

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(((__local_mem__ float*)xTmpLocal + validNumInXUb + aLoopOffset), zero, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                TwoRowAddWithTail(x1, xTmpLocal, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset,
                                  aLength + aLoopOffset, remainderTailOffset1 + aLoopOffset, rem, nextRow, remNextRow);
                TwoRowAddWithTail(x2, xTmpLocal, pregLoop, ROW_TWO_OFFSET * aLength + aLoopOffset,
                                  remainderTailOffset2 + aLoopOffset, ROW_THREE_OFFSET * aLength + aLoopOffset,
                                  remainderTailOffset3 + aLoopOffset, rem, nextRow, remNextRow);
                Add(x1, x1, x2, pregLoop);
                DataCopy(((__local_mem__ float*)xReduceLocal + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void SumRMoreThan8(__local_mem__ float* xInUb, __local_mem__ float* yInUb,
                                         __local_mem__ float* xReduceLocal, uint32_t curTileA0Len)
    {
        uint16_t remainderLoopCount = tilingData_->remainderLoopCount;
        uint16_t remainderLoopCountTmp = remainderLoopCount - 1;
        uint16_t quotientLoopCount = tilingData_->quotientLoopCount;

        uint32_t remainderOffset = tilingData_->remainderOffset;
        uint32_t baseLineOffset = tilingData_->baseLineOffset;
        uint32_t aLength = tilingData_->tileA0Len;

        uint16_t binaryAddKLoop = tilingData_->binaryAddK;
        uint16_t binaryAddInnerLoop = tilingData_->binaryAddInnerLoop;
        uint16_t binaryAddLastLoop = tilingData_->binaryAddLast;

        uint32_t validNumInXUb = tilingData_->validNumInXUb;

        uint16_t remainderTailCount = tilingData_->remainderTailCount;
        uint32_t quotientTailOffset = tilingData_->quotientTailOffset;
        uint32_t remainderTailOffset = tilingData_->remainderTailOffset;
        uint32_t remainderTailOffset0 = tilingData_->remainderTailOffset0;
        uint32_t remainderTailOffset1 = tilingData_->remainderTailOffset1;
        uint32_t remainderTailOffset2 = tilingData_->remainderTailOffset2;
        uint32_t remainderTailOffset3 = tilingData_->remainderTailOffset3;
        uint32_t remainderTailOffset4 = tilingData_->remainderTailOffset4;
        uint32_t remainderTailOffset5 = tilingData_->remainderTailOffset5;
        uint32_t remainderTailOffset6 = tilingData_->remainderTailOffset6;
        uint32_t remainderTailOffset7 = tilingData_->remainderTailOffset7;

        uint16_t aLoopCount = ops::CeilDiv(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;

            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> zero;
            Duplicate(zero, 0.0, pregMain);

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(((__local_mem__ float*)xInUb + validNumInXUb + aLoopOffset), zero, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 前半部分与后半部分中，都为8行的部分
                for (uint16_t i = 0; i < remainderLoopCountTmp; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    TwoRowAddWithTail(x1, xInUb, pregLoop, quotOffset, remOffset, quotOffset + aLength,
                                      remOffset + aLength, rem, nextRow, remNextRow);
                    TwoRowAddWithTail(x2, xInUb, pregLoop, quotOffset + ROW_TWO_OFFSET * aLength,
                                      remOffset + ROW_TWO_OFFSET * aLength, quotOffset + ROW_THREE_OFFSET * aLength,
                                      remOffset + ROW_THREE_OFFSET * aLength, rem, nextRow, remNextRow);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddWithTail(x3, xInUb, pregLoop, quotOffset + ROW_FOUR_OFFSET * aLength,
                                      remOffset + ROW_FOUR_OFFSET * aLength, quotOffset + ROW_FIVE_OFFSET * aLength,
                                      remOffset + ROW_FIVE_OFFSET * aLength, rem, nextRow, remNextRow);
                    TwoRowAddWithTail(x4, xInUb, pregLoop, quotOffset + ROW_SIX_OFFSET * aLength,
                                      remOffset + ROW_SIX_OFFSET * aLength, quotOffset + ROW_SEVEN_OFFSET * aLength,
                                      remOffset + ROW_SEVEN_OFFSET * aLength, rem, nextRow, remNextRow);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(((__local_mem__ float*)yInUb + i * aLength + aLoopOffset), x1, pregLoop);
                }
                // 前半部分为8行，后半部分可能不足8行
                {
                    TwoRowAddWithTail(x1, xInUb, pregLoop, quotientTailOffset + aLoopOffset,
                                      remainderTailOffset0 + aLoopOffset, quotientTailOffset + aLength + aLoopOffset,
                                      remainderTailOffset1 + aLoopOffset, rem, nextRow, remNextRow);
                    TwoRowAddWithTail(x2, xInUb, pregLoop, quotientTailOffset + ROW_TWO_OFFSET * aLength + aLoopOffset,
                                      remainderTailOffset2 + aLoopOffset,
                                      quotientTailOffset + ROW_THREE_OFFSET * aLength + aLoopOffset,
                                      remainderTailOffset3 + aLoopOffset, rem, nextRow, remNextRow);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddWithTail(x3, xInUb, pregLoop, quotientTailOffset + ROW_FOUR_OFFSET * aLength + aLoopOffset,
                                      remainderTailOffset4 + aLoopOffset,
                                      quotientTailOffset + ROW_FIVE_OFFSET * aLength + aLoopOffset,
                                      remainderTailOffset5 + aLoopOffset, rem, nextRow, remNextRow);
                    TwoRowAddWithTail(x4, xInUb, pregLoop, quotientTailOffset + ROW_SIX_OFFSET * aLength + aLoopOffset,
                                      remainderTailOffset6 + aLoopOffset,
                                      quotientTailOffset + ROW_SEVEN_OFFSET * aLength + aLoopOffset,
                                      remainderTailOffset7 + aLoopOffset, rem, nextRow, remNextRow);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(((__local_mem__ float*)yInUb + (remainderLoopCount - 1) * aLength + aLoopOffset), x1,
                             pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理8行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    TwoRowAdd(x1, xInUb, pregLoop, baseOffset, baseOffset + aLength, nextRow);
                    TwoRowAdd(x2, xInUb, pregLoop, baseOffset + ROW_TWO_OFFSET * aLength,
                              baseOffset + ROW_THREE_OFFSET * aLength, nextRow);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAdd(x3, xInUb, pregLoop, baseOffset + ROW_FOUR_OFFSET * aLength,
                              baseOffset + ROW_FIVE_OFFSET * aLength, nextRow);
                    TwoRowAdd(x4, xInUb, pregLoop, baseOffset + ROW_SIX_OFFSET * aLength,
                              baseOffset + ROW_SEVEN_OFFSET * aLength, nextRow);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(((__local_mem__ float*)yInUb + (remainderLoopCount + i) * aLength + aLoopOffset), x1,
                             pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                BinaryAddVF((__local_mem__ float*)yInUb, aLength, aLoopOffset, binaryAddKLoop, binaryAddInnerLoop,
                            binaryAddLastLoop, pregLoop, x1, x2, x3, x4);
                DataCopy(x1, ((__local_mem__ float*)yInUb + aLoopOffset));
                DataCopy(((__local_mem__ float*)xReduceLocal + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void TwoRowAdd(RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg, uint32_t offset1,
                                     uint32_t offset2, RegTensor<float>& nextRow)
    {
        DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
        DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset2)));
        Add(dst, dst, nextRow, preg);
    }

    __aicore__ inline void BinaryAddVF(__local_mem__ float* binaryAddTmpAddr, uint32_t rLoopStride, uint32_t offset,
                                       uint16_t binaryAddKLoop, uint16_t binaryAddInnerLoop, uint16_t binaryAddLastLoop,
                                       MaskReg& pregLoop, RegTensor<float>& x1, RegTensor<float>& x2,
                                       RegTensor<float>& x3, RegTensor<float>& x4)
    {
        uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
        for (uint16_t i = 0; i < binaryAddKLoop; i++) {
            curBinaryAddInnerLoop = curBinaryAddInnerLoop / ROW_FOUR_OFFSET;
            for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
                DataCopy(x1, ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET) * rLoopStride + offset));
                DataCopy(x2,
                         ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + 1) * rLoopStride + offset));
                Add(x1, x1, x2, pregLoop);
                DataCopy(x3, ((__local_mem__ float*)binaryAddTmpAddr +
                              (j * ROW_FOUR_OFFSET + ROW_TWO_OFFSET) * rLoopStride + offset));
                DataCopy(x4, ((__local_mem__ float*)binaryAddTmpAddr +
                              (j * ROW_FOUR_OFFSET + ROW_THREE_OFFSET) * rLoopStride + offset));
                Add(x3, x3, x4, pregLoop);
                Add(x1, x1, x3, pregLoop);
                DataCopy(((__local_mem__ float*)binaryAddTmpAddr + j * rLoopStride + offset), x1, pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        for (uint16_t i = 0; i < binaryAddLastLoop; i++) {
            DataCopy(x1, ((__local_mem__ float*)binaryAddTmpAddr + offset));
            DataCopy(x2, ((__local_mem__ float*)binaryAddTmpAddr + rLoopStride + offset));
            Add(x1, x1, x2, pregLoop);
            DataCopy(((__local_mem__ float*)binaryAddTmpAddr + offset), x1, pregLoop);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }

    __aicore__ inline void VFCalculateOutput(__local_mem__ float* yLocal, __local_mem__ float* xTmpLocal,
                                             __local_mem__ float* xReduceLocal, uint16_t curTileRLen,
                                             uint16_t curTileA0Len, uint16_t loopA0Num)
    {
        uint32_t tileA0Len = tilingData_->tileA0Len;
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> yReg;
            RegTensor<float> sumReg;

            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;

            for (uint16_t k = 0; k < loopA0Num; k++) {
                pregMask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(sumReg, (__local_mem__ float*)xReduceLocal + k * VL_FP32);

                for (uint16_t i = 0; i < curTileRLen; i++) {
                    uint32_t xOffset = i * tileA0Len + k * VL_FP32;

                    DataCopy<float, LoadDist::DIST_NORM>(xReg, (__local_mem__ float*)xTmpLocal + xOffset);
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
    const SoftmaxV2ARATilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> xQueue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> yQueue_;

    TBuf<TPosition::VECCALC> xTmpBuf_;
    TBuf<TPosition::VECCALC> xReduceBuf_;

    GlobalTensor<T2> yGm_;
    GlobalTensor<T1> xGm_;
};
}  // namespace SoftmaxV2Ops

#endif
