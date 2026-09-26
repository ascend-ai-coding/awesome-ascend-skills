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

constexpr AscendC::Reg::DivSpecificMode GMM_SWIGLU_DIV_MODE = {AscendC::Reg::MaskMergeMode::ZEROING, true};

// Pure FP32 SwiGLU tile epilogue:
//   y = swish(act) * gate = act / (1 + exp(-act)) * gate.
// act/gate must already be paired in UB. No dequant or quant is performed.
class BlockEpilogueSwiGlu {
public:
    struct Params {
        uint64_t ubViewBytes{0};
        uint64_t stageRows{0};
        float nearZeroSwishThreshold{0.0f};
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
        const uint32_t ubElements = static_cast<uint32_t>(params.ubViewBytes / sizeof(float));
        cLocal_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, 0, ubElements);
        stageRows_ = static_cast<int64_t>(params.stageRows);
        nearZeroSwishThreshold_ = params.nearZeroSwishThreshold;
    }

    __aicore__ inline auto AccumulatorTensor(int64_t curM, int64_t packedFullN)
    {
        const int64_t fixpipeM = (curM + 1) / 2 * 2;
        auto layout =
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Std::Int<16>>(fixpipeM, packedFullN);
        return AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(0), layout);
    }

    __aicore__ inline static auto MakeNDExtLayout(int64_t rows, int64_t cols, int64_t pitch)
    {
        auto shape = AscendC::Te::MakeShape(
            AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, rows), AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, cols));
        auto stride = AscendC::Te::MakeStride(
            AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, pitch),
            AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
        return AscendC::Te::MakePatternLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>(
            shape, stride);
    }

    __aicore__ inline void operator()(const TileContext& context, __gm__ float* outputGmAddr)
    {
        if (outputGmAddr == nullptr) {
            return;
        }
        const int64_t halfM = (context.curM + AscendC::GetTaskRation() - 1) / AscendC::GetTaskRation();
        const int64_t localRows =
            (context.curM & 1L) != 0 ? halfM - static_cast<int64_t>(AscendC::GetSubBlockIdx()) : halfM;
        if (localRows <= 0 || stageRows_ <= 0 || context.curH <= 0 || context.packedHalfN < context.curH ||
            context.packedFullN < context.packedHalfN + context.curH) {
            return;
        }

        const int64_t subM0 =
            context.prefixM + context.mOffset + static_cast<int64_t>(AscendC::GetSubBlockIdx()) * halfM;
        const uint32_t vl = asc_get_vf_len() / sizeof(float);
        const uint16_t loops = static_cast<uint16_t>((context.curH + vl - 1U) / vl);
        auto* src = reinterpret_cast<__ubuf__ float*>(cLocal_.GetPhyAddr());

        for (int64_t stage = 0; stage < localRows; stage += stageRows_) {
            const int64_t remainingRows = localRows - stage;
            const int64_t rows = stageRows_ < remainingRows ? stageRows_ : remainingRows;
            ComputeRows(context, stage, rows, loops, vl, src);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            const uint64_t gmOffset = static_cast<uint64_t>(subM0 + stage) * context.h + context.halfNOffset;
            AscendC::GlobalTensor<float> y;
            y.SetGlobalBuffer(outputGmAddr);
            AscendC::LocalTensor<float> output(
                AscendC::TPosition::VECIN, 0, static_cast<uint32_t>(context.packedFullN * localRows));
            AscendC::DataCopyExtParams copy{
                static_cast<uint16_t>(rows), static_cast<uint32_t>(context.curH * sizeof(float)),
                static_cast<int64_t>((context.packedFullN - context.curH) * sizeof(float) / 32),
                static_cast<int64_t>((context.h - context.curH) * sizeof(float)), 0};
            AscendC::DataCopyPad(y[gmOffset], output[stage * context.packedFullN], copy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        }
    }

    __aicore__ inline ~BlockEpilogueSwiGlu() = default;

private:
    __aicore__ inline void ComputeRows(
        const TileContext& context, int64_t stage, int64_t rows, uint16_t loops, uint32_t vl, __ubuf__ float* src)
    {
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
                auto* actRow = src + (stage + static_cast<int64_t>(row)) * context.packedFullN;
                auto* gateRow = actRow + context.packedHalfN;
                auto* outRow = actRow;
                uint32_t remaining = static_cast<uint32_t>(context.curH);
                for (uint16_t i = 0; i < loops; ++i) {
                    // UpdateMask advances the referenced remaining count.
                    auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                    const uint32_t offset = i * vl;
                    AscendC::Reg::RegTensor<float> act, gate, negative;
                    AscendC::Reg::RegTensor<float> exponent, denominator, swish, result;
                    AscendC::Reg::RegTensor<float> absAct, actSquared;
                    AscendC::Reg::RegTensor<float> nearLinear, nearQuadratic, nearSwish;
                    AscendC::Reg::MaskReg nearZeroMask;
                    AscendC::Reg::DataCopy(act, actRow + offset);
                    AscendC::Reg::DataCopy(gate, gateRow + offset);
                    AscendC::Reg::Muls(negative, act, -1.0f, mask);
                    AscendC::Reg::Exp(exponent, negative, mask);
                    AscendC::Reg::Adds(denominator, exponent, 1.0f, mask);
                    AscendC::Reg::Div<float, &GMM_SWIGLU_DIV_MODE>(swish, act, denominator, mask);
                    AscendC::Reg::Abs(absAct, act, mask);
                    AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(
                        nearZeroMask, absAct, nearZeroSwishThreshold_, mask);
                    AscendC::Reg::Mul(actSquared, act, act, mask);
                    AscendC::Reg::Muls(nearLinear, act, 0.5f, mask);
                    AscendC::Reg::Muls(nearQuadratic, actSquared, 0.25f, mask);
                    AscendC::Reg::Add(nearSwish, nearLinear, nearQuadratic, mask);
                    AscendC::Reg::Select<float>(swish, nearSwish, swish, nearZeroMask);
                    AscendC::Reg::Mul(result, swish, gate, mask);
                    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                        outRow + offset, result, mask);
                }
            }
        }
    }

    AscendC::LocalTensor<float> cLocal_;
    int64_t stageRows_{0};
    float nearZeroSwishThreshold_{0.0f};
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
