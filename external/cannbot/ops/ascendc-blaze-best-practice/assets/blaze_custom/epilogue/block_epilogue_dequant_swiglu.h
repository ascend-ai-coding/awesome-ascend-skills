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

#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

constexpr AscendC::Reg::CastTrait GMM_DQ_SWIGLU_I32_TO_F32 = {
    AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
constexpr AscendC::Reg::DivSpecificMode GMM_DQ_SWIGLU_DIV_MODE = {AscendC::Reg::MaskMergeMode::ZEROING, true};

// INT32 accumulator -> FP32 dequant -> FP32 SwiGLU -> FP32 workspace.
// The fixed actual-scenario contract is:
//   dequant = (float(acc) * xScale[row]) * weightScale[group, col].
class BlockEpilogueDequantSwiGlu {
public:
    struct Params {
        GM_ADDR weightScaleGmAddr{nullptr};
        GM_ADDR xScaleGmAddr{nullptr};
        int64_t n{0};
        int64_t workspaceN{0};
        // Total UB view for the accumulator and all three simultaneously-live
        // scale staging regions. The offsets below are byte offsets in this view.
        uint64_t ubViewBytes{0};
        uint64_t accumulatorPitch{0};
        uint64_t weightScaleActOffsetBytes{0};
        uint64_t weightScaleGateOffsetBytes{0};
        uint64_t xScaleOffsetBytes{0};
    };

    struct TileContext {
        uint32_t groupIdx{0};
        int64_t prefixM{0};
        int64_t mOffset{0};
        int64_t halfNOffset{0};
        int64_t curM{0};
        int64_t curH{0};
        int64_t packedHalfN{0};
        int64_t packedFullN{0};
        int64_t h{0};
    };

    __aicore__ inline void Init(const Params& params)
    {
        params_ = params;
        valid_ = params.weightScaleGmAddr != nullptr && params.xScaleGmAddr != nullptr && params.n > 0 &&
                 (params.n & 1) == 0 && params.accumulatorPitch > 0 && params.ubViewBytes > 0 &&
                 params.ubViewBytes <= AscendC::TOTAL_UB_SIZE && IsAligned(params.ubViewBytes) &&
                 IsAligned(params.weightScaleActOffsetBytes) && IsAligned(params.weightScaleGateOffsetBytes) &&
                 IsAligned(params.xScaleOffsetBytes) &&
                 params.weightScaleActOffsetBytes < params.weightScaleGateOffsetBytes &&
                 params.weightScaleGateOffsetBytes < params.xScaleOffsetBytes &&
                 params.xScaleOffsetBytes < params.ubViewBytes;
        workspaceN_ = params.workspaceN == 0 ? params.n / 2 : params.workspaceN;
        accumulatorPitch_ = static_cast<int64_t>(params.accumulatorPitch);
        accumulator_ = AscendC::LocalTensor<int32_t>(
            AscendC::TPosition::VECIN, 0, static_cast<uint32_t>(params.ubViewBytes / sizeof(int32_t)));
        weightScaleAct_ = AscendC::LocalTensor<float>(
            AscendC::TPosition::VECIN, static_cast<uint32_t>(params.weightScaleActOffsetBytes),
            static_cast<uint32_t>(
                RemainingBytes(params.ubViewBytes, params.weightScaleActOffsetBytes) / sizeof(float)));
        weightScaleGate_ = AscendC::LocalTensor<float>(
            AscendC::TPosition::VECIN, static_cast<uint32_t>(params.weightScaleGateOffsetBytes),
            static_cast<uint32_t>(
                RemainingBytes(params.ubViewBytes, params.weightScaleGateOffsetBytes) / sizeof(float)));
        xScale_ = AscendC::LocalTensor<float>(
            AscendC::TPosition::VECIN, static_cast<uint32_t>(params.xScaleOffsetBytes),
            static_cast<uint32_t>(RemainingBytes(params.ubViewBytes, params.xScaleOffsetBytes) / sizeof(float)));
        weightScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.weightScaleGmAddr));
        xScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.xScaleGmAddr));
    }

    __aicore__ inline auto AccumulatorTensor(int64_t curM, int64_t packedFullN) const
    {
        const int64_t fixpipeM = (curM + 1) / 2 * 2;
        auto shape = AscendC::Te::MakeShape(
            AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, fixpipeM),
            AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, packedFullN));
        auto stride = AscendC::Te::MakeStride(
            AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, packedFullN),
            AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
        auto layout =
            AscendC::Te::MakePatternLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>(
                shape, stride);
        return AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, int32_t>(0), layout);
    }

    __aicore__ inline void operator()(const TileContext& context, __gm__ float* workspaceGmAddr)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (workspaceGmAddr == nullptr) {
            return;
        }
        workspace_.SetGlobalBuffer(workspaceGmAddr);
        const int64_t halfM = (context.curM + AscendC::GetTaskRation() - 1) / AscendC::GetTaskRation();
        const int64_t localRows = AscendC::GetSubBlockIdx() == 0 ? halfM : context.curM - halfM;
        if (!valid_ || localRows <= 0 || context.curH <= 0 || context.packedHalfN < context.curH ||
            context.packedFullN <= 0 || accumulatorPitch_ < context.packedFullN || workspaceN_ < context.h ||
            !ValidateUbRegions(context, localRows)) {
            return;
        }
        const int64_t firstRow = context.prefixM + context.mOffset + AscendC::GetSubBlockIdx() * halfM;

        LoadScales(context, firstRow, localRows);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        Compute(context, localRows);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        StoreWorkspace(context, firstRow, localRows);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    }

private:
    static constexpr uint64_t UB_ALIGN_BYTES = 256UL;

    __aicore__ inline static bool IsAligned(uint64_t value)
    {
        return value % UB_ALIGN_BYTES == 0;
    }

    __aicore__ inline static uint64_t RemainingBytes(uint64_t total, uint64_t offset)
    {
        return total > offset ? total - offset : 0;
    }

    __aicore__ inline static bool RegionFits(
        uint64_t offset, uint64_t outer, uint64_t inner, uint64_t elementBytes, uint64_t limit)
    {
        if (offset > limit || outer == 0 || inner == 0 || elementBytes == 0) {
            return false;
        }
        const uint64_t capacityElements = (limit - offset) / elementBytes;
        return outer <= capacityElements / inner;
    }

    __aicore__ inline bool ValidateUbRegions(const TileContext& context, int64_t localRows) const
    {
        const uint64_t rows = static_cast<uint64_t>(localRows);
        const uint64_t packedFullN = static_cast<uint64_t>(context.packedFullN);
        const uint64_t curH = static_cast<uint64_t>(context.curH);
        const uint64_t actOffset = params_.weightScaleActOffsetBytes;
        const uint64_t gateOffset = params_.weightScaleGateOffsetBytes;
        const uint64_t xOffset = params_.xScaleOffsetBytes;
        return RegionFits(0, rows, packedFullN, sizeof(int32_t), actOffset) &&
               RegionFits(actOffset, 1, curH, sizeof(float), gateOffset) &&
               RegionFits(gateOffset, 1, curH, sizeof(float), xOffset) &&
               RegionFits(xOffset, 1, rows, sizeof(float), params_.ubViewBytes);
    }

    __aicore__ inline void LoadScales(const TileContext& context, int64_t firstRow, int64_t rows)
    {
        AscendC::DataCopyExtParams weightCopy{1, static_cast<uint32_t>(context.curH * sizeof(float)), 0, 0, 0};
        const uint64_t weightBase = static_cast<uint64_t>(context.groupIdx) * params_.n;
        AscendC::DataCopyPad(
            weightScaleAct_, weightScale_[weightBase + context.halfNOffset], weightCopy,
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        AscendC::DataCopyPad(
            weightScaleGate_, weightScale_[weightBase + context.h + context.halfNOffset], weightCopy,
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        AscendC::DataCopyExtParams tokenCopy{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(
            xScale_, xScaleGlobal_[firstRow], tokenCopy, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
    }

    __aicore__ inline void Compute(const TileContext& context, int64_t rows)
    {
        auto* packed = reinterpret_cast<__ubuf__ int32_t*>(accumulator_.GetPhyAddr());
        auto* output = reinterpret_cast<__ubuf__ float*>(packed);
        auto* actScale = reinterpret_cast<__ubuf__ float*>(weightScaleAct_.GetPhyAddr());
        auto* gateScale = reinterpret_cast<__ubuf__ float*>(weightScaleGate_.GetPhyAddr());
        auto* tokenScale = reinterpret_cast<__ubuf__ float*>(xScale_.GetPhyAddr());
        const uint32_t vl = asc_get_vf_len() / sizeof(float);
        const uint16_t loops = static_cast<uint16_t>((context.curH + vl - 1U) / vl);
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
                AscendC::Reg::RegTensor<float> xScale;
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(xScale, tokenScale + row);
                uint32_t remaining = static_cast<uint32_t>(context.curH);
                for (uint16_t i = 0; i < loops; ++i) {
                    auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                    const uint32_t offset = i * vl;
                    const uint32_t actOffset = row * context.packedFullN + offset;
                    const uint32_t gateOffset = row * context.packedFullN + context.packedHalfN + offset;
                    AscendC::Reg::RegTensor<int32_t> actI32, gateI32;
                    AscendC::Reg::RegTensor<float> act, gate, wsAct, wsGate;
                    AscendC::Reg::RegTensor<float> scaled, dequantAct, dequantGate;
                    AscendC::Reg::RegTensor<float> neg, exponent, denominator, swish, out;
                    AscendC::Reg::DataCopy(actI32, packed + actOffset);
                    AscendC::Reg::DataCopy(gateI32, packed + gateOffset);
                    AscendC::Reg::Cast<float, int32_t, GMM_DQ_SWIGLU_I32_TO_F32>(act, actI32, mask);
                    AscendC::Reg::Cast<float, int32_t, GMM_DQ_SWIGLU_I32_TO_F32>(gate, gateI32, mask);
                    AscendC::Reg::DataCopy(wsAct, actScale + offset);
                    AscendC::Reg::DataCopy(wsGate, gateScale + offset);
                    AscendC::Reg::Mul(scaled, act, xScale, mask);
                    AscendC::Reg::Mul(dequantAct, scaled, wsAct, mask);
                    AscendC::Reg::Mul(scaled, gate, xScale, mask);
                    AscendC::Reg::Mul(dequantGate, scaled, wsGate, mask);
                    AscendC::Reg::Muls(neg, dequantAct, -1.0f, mask);
                    AscendC::Reg::Exp(exponent, neg, mask);
                    AscendC::Reg::Adds(denominator, exponent, 1.0f, mask);
                    AscendC::Reg::Div<float, &GMM_DQ_SWIGLU_DIV_MODE>(swish, dequantAct, denominator, mask);
                    AscendC::Reg::Mul(out, swish, dequantGate, mask);
                    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                        output + actOffset, out, mask);
                }
            }
        }
    }

    __aicore__ inline void StoreWorkspace(const TileContext& context, int64_t firstRow, int64_t rows)
    {
        AscendC::LocalTensor<float> output(
            AscendC::TPosition::VECIN, 0, static_cast<uint32_t>(params_.ubViewBytes / sizeof(float)));
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(context.curH * sizeof(float)), 0, 0, 0};
        for (int64_t row = 0; row < rows; ++row) {
            const uint64_t dstOffset = static_cast<uint64_t>(firstRow + row) * workspaceN_ + context.halfNOffset;
            const uint64_t srcOffset = static_cast<uint64_t>(row) * context.packedFullN;
            AscendC::DataCopyPad(workspace_[dstOffset], output[srcOffset], copy);
        }
    }

    Params params_{};
    int64_t workspaceN_{0};
    int64_t accumulatorPitch_{0};
    bool valid_{false};
    AscendC::LocalTensor<int32_t> accumulator_;
    AscendC::LocalTensor<float> weightScaleAct_;
    AscendC::LocalTensor<float> weightScaleGate_;
    AscendC::LocalTensor<float> xScale_;
    AscendC::GlobalTensor<float> workspace_;
    AscendC::GlobalTensor<float> weightScale_;
    AscendC::GlobalTensor<float> xScaleGlobal_;
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
