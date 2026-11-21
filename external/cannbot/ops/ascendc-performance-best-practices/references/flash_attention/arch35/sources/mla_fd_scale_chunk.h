/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MLA_FD_SCALE_CHUNK_H
#define MLA_FD_SCALE_CHUNK_H
namespace FaVectorApi {
template <typename T, typename SINK_T, SinkInjectStage SINK_INJECT_STAGE = SinkInjectStage::AT_FD_EPILOGUE>
__simd_vf__ void MlaScaleValueChunkVF(__ubuf__ T* lseSink, __ubuf__ T* lseMax, __ubuf__ T* lseMaxTmp,
                                      __ubuf__ T* lseSum, __ubuf__ T* lseSumTmp, __ubuf__ T* lseOutUb,
                                      __ubuf__ T* lseUb, __ubuf__ T* lseMaxReduce, uint32_t dealCount,
                                      uint32_t activeCount, uint16_t i, uint32_t dealRowCount,
                                      uint32_t actualCombineLoopSize, uint16_t softmaxLseFlag,
                                      uint16_t learnableSinkFlag)
{
    Reg::RegTensor<T> vregLseMax;
    Reg::RegTensor<T> vregLseMaxTmp;
    Reg::RegTensor<T> vregRes;
    Reg::RegTensor<T> vregLseSum;
    Reg::RegTensor<T> vregLseSumTmp;
    Reg::RegTensor<T> vregLseSinkCast;
    uint16_t blockStride = 0x1;
    uint16_t repeatStride = dealRowCount;
    uint32_t n = activeCount;
    Reg::MaskReg pregTailN = Reg::UpdateMask<T>(n);
    Reg::MaskReg pregSinkTailN = Reg::UpdateMask<SINK_T>(n);

    Reg::Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vregLseSum, FLT_ZERO, pregTailN);
    Reg::Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vregLseMax, -FLT_MAX_NEW, pregTailN);

    for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregLseMaxTmp, (__ubuf__ float*&)lseMaxTmp + i * dealCount);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(vregLseMax, vregLseMax, vregLseMaxTmp, pregTailN);
    }
    Reg::StoreAlign<T, StoreDist::DIST_NORM_B32>(lseMaxReduce, vregLseMax, pregTailN);

    for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregLseMaxTmp, (__ubuf__ float*&)lseMaxTmp + i * dealCount);
        Reg::Sub<T, Reg::MaskMergeMode::ZEROING>(vregLseMaxTmp, vregLseMaxTmp, vregLseMax, pregTailN);
        Reg::Exp<T, Reg::MaskMergeMode::ZEROING>(vregLseMaxTmp, vregLseMaxTmp, pregTailN);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregLseSumTmp, (__ubuf__ float*&)lseSumTmp + i * dealCount);
        Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vregLseSumTmp, vregLseSumTmp, vregLseMaxTmp, pregTailN);
        Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregLseSum, vregLseSum, vregLseSumTmp, pregTailN);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ float*&)lseSumTmp + i * dealCount, vregLseSumTmp,
                                                          pregTailN);
    }

    if constexpr (SINK_INJECT_STAGE == SinkInjectStage::AT_FD_EPILOGUE) {
        for (i = 0; i < static_cast<uint16_t>(learnableSinkFlag); ++i) {
            Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregLseSinkCast, (__ubuf__ float*&)lseSink);
            Reg::Sub<T, Reg::MaskMergeMode::ZEROING>(vregLseSinkCast, vregLseSinkCast, vregLseMax, pregTailN);
            Reg::Exp<T, Reg::MaskMergeMode::ZEROING>(vregLseSinkCast, vregLseSinkCast, pregTailN);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregLseSum, vregLseSum, vregLseSinkCast, pregTailN);
        }
    }

    for (i = 0; i < static_cast<uint16_t>(softmaxLseFlag); ++i) {
        Reg::RegTensor<float> vregMinValue;
        Reg::RegTensor<float> vregInfValue;
        Reg::MaskReg pregCompare;
        constexpr float infValue = 3e+99; // 3e+99 for float inf
        constexpr uint32_t tmpMin = 0xFF167699;
        float minValue = *((float*)&tmpMin);
        Reg::Duplicate<float, float>(vregInfValue, infValue);
        Reg::Duplicate<float, float>(vregMinValue, minValue);

        Reg::Log<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregLseSum, pregTailN);
        Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregRes, vregLseMax, pregTailN);
        // 如果 softmaxMax 等于负无穷，则将 lse 结果置为 inf
        Reg::Compare<float, CMPMODE::EQ>(pregCompare, vregLseMax, vregMinValue, pregTailN);
        Reg::Select<T>(vregRes, vregInfValue, vregRes, pregCompare);
        Reg::StoreAlign<T, StoreDist::DIST_NORM_B32>(lseOutUb, vregRes, pregTailN);
    }

    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregLseSumTmp, (__ubuf__ float*&)lseSumTmp + i * dealCount);
        Reg::Div<T, Reg::MaskMergeMode::ZEROING>(vregLseSumTmp, vregLseSumTmp, vregLseSum, pregTailN);
        Reg::StoreAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            (__ubuf__ float*&)lseUb, vregLseSumTmp, blockStride, repeatStride, pregTailN);
    }
}

} // namespace FaVectorApi
#endif
