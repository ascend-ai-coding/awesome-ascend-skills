/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "op_tiling/matmul/blaze_group_matmul_pertoken_quant_selector.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

constexpr AscendC::Reg::CastTrait GMM_PQ_SINGLE_FP32_TO_INT16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
constexpr AscendC::Reg::CastTrait GMM_PQ_SINGLE_INT16_TO_FP16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};
constexpr AscendC::Reg::CastTrait GMM_PQ_SINGLE_FP16_TO_INT8 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC};
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
constexpr AscendC::Reg::DivSpecificMode GMM_PQ_SINGLE_DIV_MODE = {
    AscendC::Reg::MaskMergeMode::ZEROING, false, AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};
#else
constexpr AscendC::Reg::DivSpecificMode GMM_PQ_SINGLE_DIV_MODE = {AscendC::Reg::MaskMergeMode::ZEROING, true};
#endif

// Pure FP32-workspace -> INT8 per-token quantization. The complete
// logical row is read from GM once and remains in UB while scale and y are
// produced. This class performs neither dequantization nor SwiGLU.
template <
    BlazeGroupMatmulPertokenQuant::ScaleClampMode ScaleMode_ = BlazeGroupMatmulPertokenQuant::ScaleClampMode::AFTER_DIV>
class BlockEpiloguePertokenQuantSinglePass {
public:
    using InputType = float;
    using OutputType = int8_t;
    using AuxOutputType = float;

    struct Params {
        int64_t n{0};
        float scaleMin{0.0f};
        int16_t quantMin{-127};
        int16_t quantMax{127};
    };

    __aicore__ inline void Init(const Params& params)
    {
        params_ = params;
        n_ = params.n;
        if ASCEND_IS_AIV {
            // The first waits consume these Init tokens; later iterations use
            // the tokens returned by the previous row's producer.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        }
    }

    __aicore__ inline ~BlockEpiloguePertokenQuantSinglePass()
    {
        if ASCEND_IS_AIV {
            // Local final drain only; Kernel owns cross-core and phase fences.
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        }
    }

    template <class WorkspaceTensor, class YTensor, class ScaleTensor>
    __aicore__ inline void operator()(
        const WorkspaceTensor& workspaceRows, const YTensor& yRows, const ScaleTensor& scaleRows, uint32_t rowCount)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (rowCount == 0 || n_ <= 0) {
            return;
        }
        const auto ubLayout = BlazeGroupMatmulPertokenQuant::MakeUbLayout(static_cast<uint64_t>(n_));
        const uint64_t inputBytes = ubLayout.inputBytes;
        const uint64_t outputBytes = ubLayout.outputBytes;
        if (ubLayout.requiredBytes > AscendC::TOTAL_UB_SIZE) {
            return;
        }

        for (uint32_t row = 0; row < rowCount; ++row) {
            QuantizeRow(workspaceRows, yRows, scaleRows, row, inputBytes, outputBytes);
        }
    }

private:
    template <class T>
    __aicore__ inline static auto MakeLayout(int64_t cols, int64_t pitch)
    {
        auto shape = AscendC::Te::MakeShape(
            AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, AscendC::Std::Int<1>{}),
            AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, cols));
        auto stride = AscendC::Te::MakeStride(
            AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, pitch),
            AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
        return AscendC::Te::MakePatternLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<T>>(
            shape, stride);
    }

    template <class T>
    __aicore__ inline static __ubuf__ T* Ub(uint64_t offset)
    {
        return reinterpret_cast<__ubuf__ T*>(asc_get_phy_buf_addr(0) + offset);
    }

    template <class T>
    __aicore__ inline static void CopyGmToUb(
        __gm__ T* src, uint64_t dstOffset, int64_t cols, int64_t srcPitch, int64_t dstPitch)
    {
        auto gm = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(src), MakeLayout<T>(cols, srcPitch));
        auto ub = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, T>(dstOffset), MakeLayout<T>(cols, dstPitch));
        AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{}), ub, gm);
    }

    template <class T>
    __aicore__ inline static void CopyUbToGm(
        __gm__ T* dst, uint64_t srcOffset, int64_t cols, int64_t dstPitch, int64_t srcPitch)
    {
        auto gm = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(dst), MakeLayout<T>(cols, dstPitch));
        auto ub = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, T>(srcOffset), MakeLayout<T>(cols, srcPitch));
        AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{}), gm, ub);
    }

    __aicore__ inline static void Compute(
        __ubuf__ float* input, __ubuf__ int8_t* output, __ubuf__ float* scale, float scaleMin, int16_t quantMin,
        int16_t quantMax, uint32_t elements)
    {
        __VEC_SCOPE__
        {
            const uint32_t vl = asc_get_vf_len() / sizeof(float);
            const uint32_t loops = (elements + vl - 1U) / vl;
            auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::RegTensor<float> maximum, data, absolute, reduced;
            AscendC::Reg::Duplicate(maximum, -__builtin_inff(), all);
            uint32_t remaining = elements;
            for (uint16_t i = 0; i < static_cast<uint16_t>(loops); ++i) {
                // UpdateMask updates remaining by reference; do not manually decrement it.
                auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                AscendC::Reg::Duplicate(absolute, -__builtin_inff(), all);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(data, input + i * vl);
                AscendC::Reg::Abs(absolute, data, mask);
                AscendC::Reg::Max(maximum, maximum, absolute, all);
            }

            AscendC::Reg::RegTensor<float> scaleReg, minimum, scaleBroadcast;
            AscendC::Reg::RegTensor<float> quantMaxReg;
            AscendC::Reg::ReduceMax(reduced, maximum, all);
            AscendC::Reg::Duplicate(minimum, scaleMin, all);
            if constexpr (ScaleMode_ == BlazeGroupMatmulPertokenQuant::ScaleClampMode::BEFORE_DIV) {
                AscendC::Reg::Max(reduced, reduced, minimum, all);
            }
            AscendC::Reg::Duplicate(quantMaxReg, static_cast<float>(quantMax), all);
            AscendC::Reg::Div<float, &GMM_PQ_SINGLE_DIV_MODE>(scaleReg, reduced, quantMaxReg, all);
            if constexpr (ScaleMode_ == BlazeGroupMatmulPertokenQuant::ScaleClampMode::AFTER_DIV) {
                AscendC::Reg::Max(scaleReg, scaleReg, minimum, all);
            }
            AscendC::Reg::Duplicate(scaleBroadcast, scaleReg, all);
            AscendC::Reg::UnalignReg unalign;
            AscendC::Reg::DataCopyUnAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                scale, scaleReg, unalign, 1);
            AscendC::Reg::DataCopyUnAlignPost(scale, unalign, 0);

            remaining = elements;
            for (uint16_t i = 0; i < static_cast<uint16_t>(loops); ++i) {
                // UpdateMask updates remaining by reference; do not manually decrement it.
                auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                AscendC::Reg::RegTensor<float> divided;
                AscendC::Reg::RegTensor<int16_t> rounded;
                AscendC::Reg::RegTensor<half> roundedHalf, lower, upper, clamped;
                AscendC::Reg::RegTensor<int8_t> quantized;
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(data, input + i * vl);
                AscendC::Reg::Div<float, &GMM_PQ_SINGLE_DIV_MODE>(divided, data, scaleBroadcast, mask);
                AscendC::Reg::Cast<int16_t, float, GMM_PQ_SINGLE_FP32_TO_INT16>(rounded, divided, mask);
                AscendC::Reg::Cast<half, int16_t, GMM_PQ_SINGLE_INT16_TO_FP16>(roundedHalf, rounded, mask);
                AscendC::Reg::Duplicate(lower, static_cast<half>(quantMin), mask);
                AscendC::Reg::Duplicate(upper, static_cast<half>(quantMax), mask);
                AscendC::Reg::Max(clamped, roundedHalf, lower, mask);
                AscendC::Reg::Min(clamped, clamped, upper, mask);
                AscendC::Reg::Cast<int8_t, half, GMM_PQ_SINGLE_FP16_TO_INT8>(quantized, clamped, mask);
                AscendC::Reg::DataCopy<int8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    output + i * vl, quantized, mask);
            }
        }
    }

    template <class WorkspaceTensor, class YTensor, class ScaleTensor>
    __aicore__ inline void QuantizeRow(
        const WorkspaceTensor& workspaceRows, const YTensor& yRows, const ScaleTensor& scaleRows, uint32_t row,
        uint64_t inputBytes, uint64_t outputBytes)
    {
        const uint64_t inputOffset = 0;
        const uint64_t outputOffset = inputBytes;
        const uint64_t scaleOffset = inputBytes + outputBytes;
        const int64_t inputPitch = static_cast<int64_t>(inputBytes / sizeof(float));
        const int64_t outputPitch = static_cast<int64_t>(outputBytes);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        auto workspaceRow = workspaceRows.Slice(
            AscendC::Te::MakeCoord(static_cast<int64_t>(row), int64_t{0}), AscendC::Te::MakeShape(int64_t{1}, n_));
        CopyGmToUb(workspaceRow.Data().Get(), inputOffset, n_, n_, inputPitch);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        Compute(
            Ub<float>(inputOffset), Ub<int8_t>(outputOffset), Ub<float>(scaleOffset), params_.scaleMin,
            params_.quantMin, params_.quantMax, static_cast<uint32_t>(n_));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        auto yRow = yRows.Slice(
            AscendC::Te::MakeCoord(static_cast<int64_t>(row), int64_t{0}), AscendC::Te::MakeShape(int64_t{1}, n_));
        CopyUbToGm(yRow.Data().Get(), outputOffset, n_, n_, outputPitch);
        auto scaleRow = scaleRows.Slice(
            AscendC::Te::MakeCoord(int64_t{0}, static_cast<int64_t>(row)),
            AscendC::Te::MakeShape(int64_t{1}, int64_t{1}));
        CopyUbToGm(
            scaleRow.Data().Get(), scaleOffset, 1, 1, BlazeGroupMatmulPertokenQuant::UB_ALIGN_BYTES / sizeof(float));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    Params params_{};
    int64_t n_{0};
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
