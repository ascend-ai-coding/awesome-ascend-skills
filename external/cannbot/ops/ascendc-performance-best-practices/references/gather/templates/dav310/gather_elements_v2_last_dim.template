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
 * \file gather_elements_v2_last_dim.h
 * \brief
 */

#ifndef GATHER_ELEMENTS_V2_LAST_DIM_H
#define GATHER_ELEMENTS_V2_LAST_DIM_H

#include "kernel_operator.h"

constexpr int64_t NUM_TWO = 2;
constexpr int64_t BUFFER_NUM_UB = 1;
constexpr int64_t MAX_DIM_LEN = 8;
constexpr int64_t INT_BLOCL_SIZE = 8;
constexpr int64_t HALF_BLOCL_SIZE = 16;
constexpr int64_t BLOCK_SIZE_UB = 32;
constexpr int64_t INT64_DSIZE = 8;
constexpr int64_t EIGHT_BLOCK_IDX_LEN = 64;
constexpr int64_t INT_MAX = 2147483647;
constexpr int32_t RIGHT_SHIFT_LENGTH = 31;

namespace AscendC {

template <typename T_DATA, typename T_IDX>
class GatherElementsV2LastDim {
public:
    __aicore__ inline GatherElementsV2LastDim(GM_ADDR x, GM_ADDR index, GM_ADDR y,
                                              const GatherElementsV2TilingData& tiling, TPipe* pipe)
    {
        InitParams(x, index, y, tiling, pipe);
    }

    __aicore__ inline void Process()
    {
        if (scalarMode_ == 1) {
            ProcessScalarMode();
        } else if (eachCalculationLines_ > 1) {
            ProcessMultRow();
        } else {
            ProcessSingleRow();
        }
    }

    __aicore__ inline float intToFloatBits(int i)
    {
        // Use a union to reinterpret the bits of an int as a float
        union {
            int i;
            float f;
        } u;
        u.i = i;
        return u.f;
    }
    template <typename T>
    __aicore__ inline T min(T a, T b)
    {
        return a < b ? a : b;
    };

    __aicore__ inline void CopyArray(const int64_t* src, int64_t* dst)
    {
        for (int i = 0; i < MAX_DIM_LEN; i++) {
            dst[i] = src[i];
        }
    }

    __aicore__ inline void SyncM2toV()
    {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    };

    __aicore__ inline void SyncVtoM2()
    {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    };

    __aicore__ inline void SyncM3toV()
    {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    };

    __aicore__ inline void SyncVtoM3()
    {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    };

    __aicore__ inline void InitParams(GM_ADDR x, GM_ADDR index, GM_ADDR y, const GatherElementsV2TilingData tiling,
                                      TPipe* pipe)
    {
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();

        dimNum_ = tiling.lastDimTiling.dimNum;
        xSliceNum_ = tiling.lastDimTiling.xSliceNum;
        indexSliceNum_ = tiling.lastDimTiling.indexSliceNum;
        reservedXSize_ = tiling.lastDimTiling.reservedXSize;
        reservedIndexSize_ = tiling.lastDimTiling.reservedIndexSize;
        indexAxisSizeEqualOne_ = tiling.lastDimTiling.indexAxisSizeEqualOne;
        eachCalculationLines_ = tiling.lastDimTiling.eachCalculationLines;
        xBufferSize_ = tiling.lastDimTiling.xBufferSize;
        indexBufferSize_ = tiling.lastDimTiling.indexBufferSize;
        ubStorageIdxCount_ = indexBufferSize_ / INT64_DSIZE;
        yBufferSize_ = tiling.lastDimTiling.yBufferSize;
        maskBufferSize_ = tiling.lastDimTiling.maskBufferSize;
        scalarModeLength_ = tiling.lastDimTiling.scalarModeLength;

        formerCoreNum_ = tiling.lastDimTiling.formerCoreNum;
        eachCoreCalculationLines_ = tiling.lastDimTiling.formerCoreRowNum;
        scalarMode_ = tiling.lastDimTiling.scalarMode;
        dataMoveUBStride_ = tiling.lastDimTiling.dataMoveUBStride;

        if (blockIdx_ < formerCoreNum_) {
            ++eachCoreCalculationLines_;
        }
        specialDataMove_ = tiling.lastDimTiling.specialDataMove;
        CopyArray(tiling.lastDimTiling.xShape, xShapeArray_);
        CopyArray(tiling.lastDimTiling.indexShape, indexShapeArray_);
        CopyArray(tiling.lastDimTiling.xStrideArray, xStrideArray_);
        CopyArray(tiling.lastDimTiling.indexStrideArray, indexStrideArray_);
        xAxisSize_ = static_cast<int32_t>(xShapeArray_[dimNum_ - 1]);
        indexAxisSize_ = static_cast<int32_t>(indexShapeArray_[dimNum_ - 1]);
        xAxisAlignSize_ = CeilAlign(xAxisSize_, xPerBlockLen_);
        indexAxisAlignSize_ = CeilAlign(indexAxisSize_, INT_BLOCL_SIZE);
        yAxisAlignSize_ = CeilAlign(indexAxisSize_, xPerBlockLen_);
        penultimateDimensionIndex_ = dimNum_ - NUM_TWO;
        if (blockIdx_ < formerCoreNum_) {
            indexBaseOffset_ = blockIdx_ * eachCoreCalculationLines_ * indexAxisSize_;
        } else {
            indexBaseOffset_ = blockIdx_ * eachCoreCalculationLines_ * indexAxisSize_ + formerCoreNum_ * indexAxisSize_;
        }
        xOffset_ = (sizeof(T_DATA) == sizeof(uint8_t) && scalarMode_ == 0) ? xBufferSize_ / sizeof(half) : 0;

        xGm_.SetGlobalBuffer((__gm__ T_DATA*)x);
        indexGm_.SetGlobalBuffer((__gm__ int32_t*)index);
        yGm_.SetGlobalBuffer((__gm__ T_DATA*)y);

        if (scalarMode_ == 0) {
            pipe_->InitBuffer(xQue_, BUFFER_NUM_UB, xBufferSize_);
            xTensor_ = xQue_.AllocTensor<T_DATA>();
        }

        pipe_->InitBuffer(yQue_, BUFFER_NUM_UB, yBufferSize_);
        yTensor_ = yQue_.AllocTensor<T_DATA>();

        pipe_->InitBuffer(indexQue_, BUFFER_NUM_UB, indexBufferSize_);
        indexTensor_ = indexQue_.AllocTensor<int32_t>();
        gatherOffsetTensor_ = indexTensor_.template ReinterpretCast<uint32_t>();
    }

    __aicore__ inline void ProcessScalarMode()
    {
        int64_t computeTime = Ceil(eachCoreCalculationLines_, eachCalculationLines_);
        if (!scalarModeLength_) {
            int64_t realCalculationLines = eachCalculationLines_;
            for (int i = 0; i < computeTime; i++) {
                indexOffset_ = indexBaseOffset_ + i * eachCalculationLines_ * indexAxisSize_;
                if (i == computeTime - 1) {
                    realCalculationLines = eachCoreCalculationLines_ - i * eachCalculationLines_;
                }
                CopyInIndex(realCalculationLines, indexAxisSize_);
                ComputeScalarMode(realCalculationLines, indexAxisSize_);
                CopyOut<T_DATA>(realCalculationLines, indexAxisSize_);
            }
        } else {
            int64_t coreNum = GetBlockNum();
            int64_t laterLength = indexAxisSize_ / indexSliceNum_;
            int64_t fromerLength = laterLength + 1;
            int64_t fromerCore = indexAxisSize_ % indexSliceNum_;
            int64_t realCountIdxCount = 0;
            while (blockIdx_ < scalarModeLength_) {
                indexBaseOffset_ = blockIdx_ / indexSliceNum_ * indexAxisSize_;
                int64_t coreId = blockIdx_ % indexSliceNum_;
                if (coreId < fromerCore) {
                    realCountIdxCount = fromerLength;
                    indexOffset_ = indexBaseOffset_ + coreId * realCountIdxCount;
                } else {
                    realCountIdxCount = laterLength;
                    indexOffset_ = indexBaseOffset_ + coreId * realCountIdxCount + fromerCore;
                }
                CopyInIndex(1, realCountIdxCount);
                ComputeScalarMode(1, realCountIdxCount);
                CopyOut<T_DATA>(1, realCountIdxCount);
                blockIdx_ += coreNum;
            }
        }
    }

    __aicore__ inline void ProcessMultRow()
    {
        int64_t computeTime = Ceil(eachCoreCalculationLines_, eachCalculationLines_);
        int64_t realCalculationLines = eachCalculationLines_;
        for (int i = 0; i < computeTime; i++) {
            indexOffset_ = indexBaseOffset_ + i * eachCalculationLines_ * indexAxisSize_;
            GetGmOffset(indexOffset_, xGmOffset_);
            if (i == computeTime - 1) {
                realCalculationLines = eachCoreCalculationLines_ - i * eachCalculationLines_;
            }
            if (indexAxisSizeEqualOne_) {
                CopyInIndex(1, realCalculationLines);
                CopyInData(1, realCalculationLines * xAxisSize_);
                ComputeIndexAxisSizeEqualOne(1, realCalculationLines);
                CopyOut<T_DATA>(1, realCalculationLines);
            } else {
                CopyInIndex(realCalculationLines, indexAxisSize_);
                CopyInData(realCalculationLines, xAxisSize_);
                ComputePerCoreMultCompute(realCalculationLines, indexAxisSize_);
                CopyOut<T_DATA>(realCalculationLines, indexAxisSize_);
            }
        }
    }

    __aicore__ inline void ProcessSingleRow()
    {
        int64_t realCountIdxCount = 0;
        if (xSliceNum_ == 1) {
            for (int i = 0; i < eachCoreCalculationLines_; i++) {
                realCountIdxCount = ubStorageIdxCount_;
                indexOffset_ = indexBaseOffset_ + i * indexAxisSize_;
                GetGmOffset(indexOffset_, xGmOffset_);
                CopyInData(1, xAxisSize_);
                for (int j = 0; j < indexSliceNum_; j++) {
                    if (j == indexSliceNum_ - 1) {
                        realCountIdxCount = reservedIndexSize_;
                    }
                    CopyInIndex(1, realCountIdxCount);
                    ComputePerCoreSingleCompute(1, realCountIdxCount);
                    CopyOut<T_DATA>(1, realCountIdxCount);
                    indexOffset_ += ubStorageIdxCount_;
                }
            }
        } else {
            pipe_->InitBuffer(upMaskQue_, BUFFER_NUM_UB, maskBufferSize_);
            pipe_->InitBuffer(downMaskQue_, BUFFER_NUM_UB, maskBufferSize_);
            pipe_->InitBuffer(yPartQue_, BUFFER_NUM_UB, yBufferSize_);
            upMaskTensor_ = upMaskQue_.AllocTensor<uint8_t>();
            downMaskTensor_ = downMaskQue_.AllocTensor<uint8_t>();
            yPartTensor_ = yPartQue_.AllocTensor<T_DATA>();

            for (int i = 0; i < eachCoreCalculationLines_; i++) {
                realCountIdxCount = ubStorageIdxCount_;
                for (int j = 0; j < indexSliceNum_; j++) {
                    indexOffset_ = indexBaseOffset_ + i * indexAxisSize_ + j * ubStorageIdxCount_;
                    if (j == indexSliceNum_ - 1) {
                        realCountIdxCount = reservedIndexSize_;
                    }
                    CopyInIndex(1, realCountIdxCount);
                    ComputePerCoreSingleComputeForX(1, realCountIdxCount);
                    CopyOut<T_DATA>(1, realCountIdxCount);
                }
            }
        }
    }

    __aicore__ inline void CopyInIndex(uint16_t nbrust, int64_t length)
    {
        SyncVtoM2();
        DataCopyExtParams copyParams = {nbrust, static_cast<uint32_t>(length * sizeof(T_IDX)), 0, dataMoveUBStride_, 0};
        DataCopyPadExtParams<int32_t> padParams = {true, 0, 0, 0};
        DataCopyPad(indexTensor_, indexGm_[indexOffset_ * dtypeSizeRatio_], copyParams, padParams);

        if constexpr (sizeof(T_IDX) == INT64_DSIZE) {
            SyncM2toV();
            LocalTensor<int64_t> tmpTensor = indexTensor_.template ReinterpretCast<int64_t>();
            Cast(indexTensor_, tmpTensor, RoundMode::CAST_NONE, ubStorageIdxCount_);
        }
        SyncM2toV();
    }

    __aicore__ inline void CopyInData(uint16_t nbrust, int64_t length)
    {
        SyncVtoM2();
        if (specialDataMove_) {
            SpecialDataMove(nbrust, length);
        } else {
            DataCopyExtParams copyParams = {nbrust, static_cast<uint32_t>(length * sizeof(T_DATA)), 0, 0, 0};
            DataCopyPadExtParams<T_DATA> padParams = {true, 0, 0, 0};
            DataCopyPad(xTensor_[xOffset_], xGm_[xGmOffset_], copyParams, padParams);
        }
        SyncM2toV();
    }

    __aicore__ inline void GetGmOffset(int64_t tmpOffset, int64_t& xGmBaseOffset)
    {
        xGmBaseOffset = 0;
        for (int i = 0; i <= penultimateDimensionIndex_; i++) {
            indexOffsetArray_[i] = tmpOffset / indexStrideArray_[i];
            tmpOffset %= indexStrideArray_[i];
            xGmBaseOffset = xGmBaseOffset + indexOffsetArray_[i] * xStrideArray_[i];
        }
    }

    __aicore__ inline void SpecialDataMove(uint16_t nbrust, int64_t length)
    {
        int64_t tmpOffset = indexOffset_;
        int64_t tensorOffset = 0;
        int64_t tmpXGmOffset = 0;
        while (nbrust > 0) {
            GetGmOffset(tmpOffset, tmpXGmOffset);
            uint16_t moveBrust = min(nbrust, static_cast<uint16_t>(indexShapeArray_[penultimateDimensionIndex_] -
                                                                   indexOffsetArray_[penultimateDimensionIndex_]));
            DataCopyExtParams copyParams = {moveBrust, static_cast<uint32_t>(length * sizeof(T_DATA)), 0, 0, 0};
            DataCopyPadExtParams<T_DATA> padParams = {true, 0, 0, 0};
            DataCopyPad(xTensor_[xOffset_ + tensorOffset], xGm_[tmpXGmOffset], copyParams, padParams);
            tensorOffset = tensorOffset + moveBrust * xAxisAlignSize_;
            tmpOffset = tmpOffset + moveBrust * indexAxisSize_;
            nbrust -= moveBrust;
        }
    }

    __aicore__ inline void DoNegativeIndices(int64_t length)
    {
        ShiftRight(indexTensor_[ubStorageIdxCount_], indexTensor_, RIGHT_SHIFT_LENGTH, length);

        Muls(indexTensor_[ubStorageIdxCount_], indexTensor_[ubStorageIdxCount_], -static_cast<int32_t>(xAxisSize_),
             length);

        Add(indexTensor_, indexTensor_, indexTensor_[ubStorageIdxCount_], length);
    }

    __aicore__ inline void ComputeIndexAxisSizeEqualOne(uint16_t nbrust, int64_t length)
    {
        DoNegativeIndices(length);
        CreateVecIndex(indexTensor_[ubStorageIdxCount_], 0, length);
        Muls(indexTensor_[ubStorageIdxCount_], indexTensor_[ubStorageIdxCount_], static_cast<int32_t>(xAxisSize_),
             length);
        Add(indexTensor_, indexTensor_, indexTensor_[ubStorageIdxCount_], length);
        GatherData<T_DATA>(0, length);
    }

    __aicore__ inline void ComputePerCoreMultCompute(uint16_t nbrust, int64_t length)
    {
        DoNegativeIndices(nbrust * indexAxisAlignSize_);
        PipeBarrier<PIPE_V>();
        for (uint16_t i = 0; i < nbrust; i++) {
            GatherData<T_DATA>(i, length);
        }
    }

    __aicore__ inline void ComputePerCoreSingleCompute(uint16_t nbrust, int64_t length)
    {
        DoNegativeIndices(length);
        GatherData<T_DATA>(0, length);
    }

    __aicore__ inline void ComputePerCoreSingleComputeForX(uint16_t nbrust, int64_t length)
    {
        int64_t xCountLength = xBufferSize_ / xDataSize_;
        DoNegativeIndices(length);
        GetGmOffset(indexOffset_, xGmOffset_);
        LocalTensor<float> castFloatTargetTensor = indexTensor_.template ReinterpretCast<float>();
        for (int k = 0; k < xSliceNum_; k++) {
            int64_t xRealCountLength = xCountLength;
            float upLimit = intToFloatBits(static_cast<int32_t>((k + 1) * xCountLength));
            float downLimit = intToFloatBits(static_cast<int32_t>(k * xCountLength));
            int32_t negDownLimit = -static_cast<int32_t>(k * xCountLength);
            if (k == xSliceNum_ - 1) {
                xRealCountLength = reservedXSize_;
                upLimit = intToFloatBits(static_cast<int32_t>(xAxisSize_));
            }
            int64_t alignLength = CeilAlign(length, EIGHT_BLOCK_IDX_LEN);
            CopyInData(1, xRealCountLength);
            CompareScalar(downMaskTensor_, castFloatTargetTensor, downLimit, CMPMODE::GE, alignLength);
            CompareScalar(upMaskTensor_, castFloatTargetTensor, upLimit, CMPMODE::LT, alignLength);
            PipeBarrier<PIPE_V>();
            Adds(indexTensor_[ubStorageIdxCount_], indexTensor_, negDownLimit, length);
            Select(castFloatTargetTensor[ubStorageIdxCount_], downMaskTensor_,
                   castFloatTargetTensor[ubStorageIdxCount_], 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            Select(castFloatTargetTensor[ubStorageIdxCount_], upMaskTensor_, castFloatTargetTensor[ubStorageIdxCount_],
                   0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            GatherData<T_DATA>(yPartTensor_, 0, ubStorageIdxCount_, length);
            xGmOffset_ = xGmOffset_ + xCountLength;
        }
    }

    __aicore__ inline void ComputeScalarMode(uint16_t nbrust, int64_t length)
    {
        int64_t indexAlignLength = CeilAlign(length, INT_BLOCL_SIZE);
        int64_t yAlignLength = CeilAlign(length, xPerBlockLen_);
        int64_t tmpOffset = indexOffset_;
        int64_t xGmBaseOffset = 0;
        int64_t indexTensorOffset = 0;
        int64_t yTensorOffset = 0;
        int64_t idx = 0;
        DoNegativeIndices(indexAlignLength * nbrust);

        for (int64_t i = 0; i < nbrust; i++) {
            indexTensorOffset = i * indexAlignLength;
            yTensorOffset = i * yAlignLength;
            GetGmOffset(tmpOffset, xGmBaseOffset);
            if (xGmBaseOffset < INT_MAX - NUM_TWO * xAxisSize_) {
                Adds(indexTensor_[indexTensorOffset], indexTensor_[indexTensorOffset],
                     static_cast<int32_t>(xGmBaseOffset), length);
                for (int64_t j = 0; j < length; j++) {
                    idx = static_cast<int64_t>(indexTensor_.GetValue(j + indexTensorOffset));
                    T_DATA data = xGm_.GetValue(idx);
                    yTensor_.SetValue(j + yTensorOffset, data);
                }
            } else {
                for (int64_t j = 0; j < length; j++) {
                    idx = indexTensor_.GetValue(j + indexTensorOffset);
                    T_DATA data = xGm_.GetValue(idx + xGmBaseOffset);
                    yTensor_.SetValue(j + yTensorOffset, data);
                }
            }
            tmpOffset += indexAxisSize_;
        }
    }

    template <typename T>
    __aicore__ inline void GatherData(LocalTensor<T> targetTensor, int64_t dataOffset, int64_t indexOffset,
                                      int64_t length)
    {
        Muls(indexTensor_[indexOffset], indexTensor_[indexOffset], static_cast<int32_t>(sizeof(T_DATA)), length);
        Gather(targetTensor, xTensor_, gatherOffsetTensor_[indexOffset], static_cast<uint32_t>(0), length);
        Select(targetTensor, downMaskTensor_, targetTensor, yTensor_, SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
        Select(yTensor_, upMaskTensor_, targetTensor, yTensor_, SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
    }

    template <typename T>
    __aicore__ inline void GatherData(int64_t idx, int64_t length)
    {
        int64_t xTensorOffset = idx * xAxisAlignSize_;
        int64_t indexTensorOffset = idx * indexAxisAlignSize_;
        int64_t yTensorOffset = idx * yAxisAlignSize_;
        Muls(indexTensor_[indexTensorOffset], indexTensor_[indexTensorOffset], static_cast<int32_t>(sizeof(T_DATA)),
             length);
        Gather(yTensor_[yTensorOffset], xTensor_[xTensorOffset], gatherOffsetTensor_[indexTensorOffset],
               static_cast<uint32_t>(0), length);
    }

    template <typename T>
    __aicore__ inline void CopyOut(uint16_t nbrust, int64_t length)
    {
        SyncVtoM3();
        DataCopyExtParams copyParams(nbrust, static_cast<uint32_t>(length * sizeof(T_DATA)), 0, 0, 0);
        DataCopyPad(yGm_[indexOffset_], yTensor_, copyParams);
        SyncM3toV();
    }

private:
    TPipe* pipe_;
    GlobalTensor<T_DATA> xGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<T_DATA> yGm_;

    TQue<TPosition::VECIN, BUFFER_NUM_UB> xQue_;
    TQue<TPosition::VECIN, BUFFER_NUM_UB> indexQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM_UB> yQue_;
    TQue<TPosition::VECIN, BUFFER_NUM_UB> yPartQue_;
    TQue<TPosition::VECIN, BUFFER_NUM_UB> downMaskQue_;
    TQue<TPosition::VECIN, BUFFER_NUM_UB> upMaskQue_;

    LocalTensor<T_DATA> xTensor_;
    LocalTensor<int32_t> indexTensor_;
    LocalTensor<uint32_t> gatherOffsetTensor_;
    LocalTensor<T_DATA> yTensor_;
    LocalTensor<T_DATA> yPartTensor_;
    LocalTensor<uint8_t> downMaskTensor_;
    LocalTensor<uint8_t> upMaskTensor_;

    int64_t xDataSize_ = sizeof(T_DATA) == 1 ? 2 : sizeof(T_DATA);
    int64_t blockIdx_{0};
    int64_t dimNum_{0};
    int64_t xSliceNum_{0};
    int64_t indexSliceNum_{0};
    int64_t reservedXSize_{0};
    int64_t reservedIndexSize_{0};
    int64_t indexAxisSizeEqualOne_{0};
    int64_t eachCoreCalculationLines_{0};
    int64_t eachCalculationLines_{0};
    int64_t xBufferSize_{0};
    int64_t indexBufferSize_{0};
    int64_t yBufferSize_{0};
    int64_t maskBufferSize_{0};
    int64_t xAxisSize_{0};
    int64_t indexAxisSize_{0};
    int64_t xAxisAlignSize_{0};
    int64_t indexAxisAlignSize_{0};
    int64_t yAxisAlignSize_{0};
    int64_t ubStorageIdxCount_{0};
    int64_t formerCoreNum_{0};
    int64_t specialDataMove_{0};
    int64_t indexBaseOffset_{0};
    int64_t indexOffset_{0};
    int64_t xOffset_{0};
    int64_t xGmOffset_{0};
    int64_t penultimateDimensionIndex_{0};
    int64_t scalarMode_{0};
    int64_t scalarModeLength_{0};
    uint32_t dataMoveUBStride_{0};

    int64_t xPerBlockLen_ = BLOCK_SIZE_UB / sizeof(T_DATA);
    int64_t indexPerBlockLen_ = BLOCK_SIZE_UB / sizeof(T_IDX);
    int64_t dtypeSizeRatio_ = sizeof(T_IDX) / sizeof(int32_t);

    int64_t xShapeArray_[MAX_DIM_LEN] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t indexShapeArray_[MAX_DIM_LEN] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t indexOffsetArray_[MAX_DIM_LEN] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t xStrideArray_[MAX_DIM_LEN] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t indexStrideArray_[MAX_DIM_LEN] = {0, 0, 0, 0, 0, 0, 0, 0};
};

} // namespace AscendC

#endif // GATHER_ELEMENTS_V2_LAST_DIM_H