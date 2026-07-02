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
 * \file scale_gelu_epilogue_regbase.h
 * \brief [MODIFY] AIV 侧 RegBase Epilogue — AIC+AIV 融合算子的核心定制点。
 *
 * [MODIFY] 本文件是 AIV 侧 eltwise 后处理逻辑的实现，是融合算子最核心的定制点。
 * 开发新融合算子时，您需要：
 *   1. 修改 L0CDataType（AIC matmul 输出类型：int32_t, float 等）
 *   2. 修改 OutputType（最终输出类型：bfloat16_t, half_t, float 等）
 *   3. 修改 ComputeType（中间计算类型：通常为 float）
 *   4. 修改 Params 结构体（GM 地址参数：scale、bias 等）
 *   5. 修改 Init 函数中的 UB 布局（buffer 分配和偏移计算）
 *   6. 修改 operator() 函数中的 eltwise 计算流水
 *   7. 保持 PipeBarrier<PIPE_ALL>() + CrossCoreSetFlag 序列不变（CV 同步关键）
 *
 * 当前示例为 quant_matmul_gelu 的 AIV 后处理：
 *   y = int32_result * scale[n] (perchannel) * pertokenScale[m] (per-token) + bias[n] (perchannel)
 *   out = gelu_tanh(y)  →  cast 为 bfloat16 写回 GM。
 *
 * Epilogue 类必须满足三接口合约：
 *   1) using Params = ...      — 参数类型
 *   2) void Init(Params, l1M, l1N, problemShape) — 初始化 UB 布局
 *   3) void operator()(BlockShape, gmOffset, flagId)  — 逐 tile 处理 + 写出 + SetFlag
 */

#ifndef EPILOGUE_SCALE_GELU_EPILOGUE_REGBASE_H
#define EPILOGUE_SCALE_GELU_EPILOGUE_REGBASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "epilogue/cv_sync_constants.h"
#include "kernel_utils/common_utils.h"

using namespace AscendC;

#ifdef __CCE_AICORE__

class ScaleGeluEpilogueRegBase {
public:
    using L0CDataType = int32_t;
    using OutputType = bfloat16_t;
    using ComputeType = float;

    static constexpr uint16_t ZERO_FLAG = 0;
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = CvSync::MODE;

    static constexpr float NEG_SQRT_EIGHT_OVER_PI = -1.595769121f * 0.044715f;
    static constexpr float TANH_APPROX_FACTOR = 1.0f / 0.044715f;

    struct Params {
        GM_ADDR x3GmAddr{nullptr};              // perchannel scale, float, shape (n)
        GM_ADDR pertokenScaleGmAddr{nullptr};   // per-token scale, float, shape (m)
        GM_ADDR biasGmAddr{nullptr};            // perchannel bias, float, shape (n)
        GM_ADDR outputGmAddr{nullptr};          // output, bf16, shape (m, n)
    };

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    AscendC::LocalTensor<L0CDataType> cLocal_{AscendC::TPosition::VECIN, 0, UB_SIZE};
    AscendC::LocalTensor<ComputeType> x3Buf_{AscendC::TPosition::VECIN, 0, UB_SIZE};
    AscendC::LocalTensor<ComputeType> pertokenBuf_{AscendC::TPosition::VECIN, 0, UB_SIZE};
    AscendC::LocalTensor<ComputeType> biasBuf_{AscendC::TPosition::VECIN, 0, UB_SIZE};
    AscendC::LocalTensor<OutputType> bf16Out_{AscendC::TPosition::VECIN, 0, UB_SIZE};

    AscendC::GlobalTensor<OutputType> outputGlobal_;
    AscendC::GlobalTensor<ComputeType> x3AsFloatGlobal_;
    AscendC::GlobalTensor<ComputeType> pertokenGlobal_;
    AscendC::GlobalTensor<ComputeType> biasGlobal_;

    int64_t stageRows_{0};
    int64_t nAlign_{0};
    int64_t nAlignBf16_{0};
    ProblemShape problemShape_;

    constexpr static AscendC::Reg::CastTrait castTraitI32ToF32 = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    constexpr static AscendC::Reg::CastTrait castTraitF32ToBF16 = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    __aicore__ inline void Init(
        Params const& params, int64_t l1M, int64_t l1N, ProblemShape& problemShape)
    {
        constexpr int64_t ALIGN_I32 = 32 / sizeof(L0CDataType);
        nAlign_ = ::CeilDiv(l1N, ALIGN_I32) * ALIGN_I32;
        constexpr int64_t ALIGN_BF16 = 32 / sizeof(OutputType);
        nAlignBf16_ = ::CeilDiv(l1N, ALIGN_BF16) * ALIGN_BF16;

        int64_t splitTaskRation = static_cast<int64_t>(AscendC::GetTaskRation());
        int64_t l1MSplit = ::CeilDiv(l1M, splitTaskRation);
        int64_t matmulAreaBytes = l1MSplit * nAlign_ * sizeof(L0CDataType);

        int64_t x3BufBytes = nAlign_ * sizeof(ComputeType);
        int64_t pertokenBufBytes = l1MSplit * sizeof(ComputeType);
        int64_t biasBufBytes = nAlign_ * sizeof(ComputeType);
        int64_t remainBytes = UB_SIZE - matmulAreaBytes - x3BufBytes - pertokenBufBytes - biasBufBytes;
        int64_t maxBf16Rows = remainBytes / (nAlign_ * sizeof(OutputType));
        stageRows_ = maxBf16Rows;

        int64_t x3BufElemOffset = l1MSplit * nAlign_;
        x3Buf_ = cLocal_[x3BufElemOffset].template ReinterpretCast<float>();

        int64_t pertokenElemOffset = x3BufElemOffset + nAlign_;
        pertokenBuf_ = cLocal_[pertokenElemOffset].template ReinterpretCast<float>();

        int64_t biasElemOffset = pertokenElemOffset + l1MSplit;
        biasBuf_ = cLocal_[biasElemOffset].template ReinterpretCast<float>();

        int64_t bf16OutByteOffset = matmulAreaBytes + x3BufBytes + pertokenBufBytes + biasBufBytes;
        bf16Out_ = cLocal_[bf16OutByteOffset / sizeof(L0CDataType)].template ReinterpretCast<OutputType>();

        problemShape_ = problemShape;
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ OutputType*>(params.outputGmAddr));
        x3AsFloatGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ComputeType*>(params.x3GmAddr));
        pertokenGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ComputeType*>(params.pertokenScaleGmAddr));
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ComputeType*>(params.biasGmAddr));

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
    }

    __aicore__ inline auto GetTensor() { return cLocal_; }

    __aicore__ inline void operator()(
        BlockShape const& blockShape, int64_t dstOffset, int64_t flagId = CvSync::AIV_TO_AIC_FLAG)
    {
        int64_t blockShapeM = Get<0>(blockShape);
        int64_t blockShapeN = Get<1>(blockShape);

        int64_t halfM = ::CeilDiv(blockShapeM, AscendC::GetTaskRation());
        blockShapeM = ((static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL)
                          ? (halfM - AscendC::GetSubBlockIdx()) : halfM;

        int64_t N = Get<MNK_N>(problemShape_);

        if (blockShapeM <= 0) {
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(flagId);
            return;
        }

        int64_t curStageRows = AscendC::Std::min(stageRows_, halfM);

        if (curStageRows <= 0) {
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(flagId);
            return;
        }

        int64_t nPos = dstOffset % N;
        int64_t mPos = dstOffset / N;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);

        {
            uint16_t nRows = 1;
            uint32_t rowBytes = static_cast<uint32_t>(blockShapeN * sizeof(ComputeType));
            uint32_t gmRowGap = static_cast<uint32_t>((N - blockShapeN) * sizeof(ComputeType));
            AscendC::DataCopyExtParams cp{nRows, rowBytes, gmRowGap, 0, 0};
            AscendC::DataCopyPadExtParams<ComputeType> pp{false, 0, 0, 0};
            AscendC::DataCopyPad(x3Buf_, x3AsFloatGlobal_[nPos], cp, pp);
        }

        {
            // perchannel bias (float): GM → biasBuf_
            uint16_t nRows = 1;
            uint32_t rowBytes = static_cast<uint32_t>(blockShapeN * sizeof(ComputeType));
            uint32_t gmRowGap = static_cast<uint32_t>((N - blockShapeN) * sizeof(ComputeType));
            AscendC::DataCopyExtParams cp{nRows, rowBytes, gmRowGap, 0, 0};
            AscendC::DataCopyPadExtParams<ComputeType> pp{false, 0, 0, 0};
            AscendC::DataCopyPad(biasBuf_, biasGlobal_[nPos], cp, pp);
        }

        {
            int64_t pertokenStart = mPos + AscendC::GetSubBlockIdx() * halfM;
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(halfM * sizeof(ComputeType)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<ComputeType> pp{false, 0, 0, 0};
            AscendC::DataCopyPad(pertokenBuf_, pertokenGlobal_[pertokenStart], cp, pp);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);

        __ubuf__ int32_t* srcAddr = (__ubuf__ int32_t*)cLocal_.GetPhyAddr();
        __ubuf__ float* x3Addr = (__ubuf__ float*)x3Buf_.GetPhyAddr();
        __ubuf__ float* pertokenAddr = (__ubuf__ float*)pertokenBuf_.GetPhyAddr();
        __ubuf__ float* biasAddr = (__ubuf__ float*)biasBuf_.GetPhyAddr();
        __ubuf__ bfloat16_t* dstAddr = (__ubuf__ bfloat16_t*)bf16Out_.GetPhyAddr();

        constexpr int64_t ALIGN_I32 = 32 / sizeof(L0CDataType);
        int64_t nAlign = ::CeilDiv(blockShapeN, ALIGN_I32) * ALIGN_I32;
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(ComputeType);
        uint16_t vfLoopNum = (static_cast<uint32_t>(nAlign) + VL - 1) / VL;

        int64_t stageOffset = 0;
        int64_t totalRows = halfM;

        while (stageOffset < totalRows) {
            int64_t rowsThisStage = AscendC::Std::min(curStageRows, totalRows - stageOffset);
            
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);

            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<int32_t> vregI32;
                AscendC::Reg::RegTensor<float> vregFloat;
                AscendC::Reg::RegTensor<float> vregX3;
                AscendC::Reg::RegTensor<float> vregBias;
                AscendC::Reg::RegTensor<float> vregToken;
                AscendC::Reg::RegTensor<float> vregY;
                AscendC::Reg::RegTensor<float> vregSqr;
                AscendC::Reg::RegTensor<float> vregCub;
                AscendC::Reg::RegTensor<float> vregGelu;
                AscendC::Reg::RegTensor<bfloat16_t> vregBf16;
                AscendC::Reg::MaskReg mask;

                for (uint16_t row = 0; row < static_cast<uint16_t>(rowsThisStage); row++) {
                    __ubuf__ int32_t* rowSrc = srcAddr + (stageOffset + row) * nAlign;
                    __ubuf__ bfloat16_t* rowDst = dstAddr + row * nAlign;

                    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                        vregToken, pertokenAddr + stageOffset + row);

                    for (uint16_t i = 0; i < vfLoopNum; i++) {
                        uint32_t active = static_cast<uint32_t>(nAlign) - static_cast<uint32_t>(i) * VL;
                        if (active > VL) {
                            active = VL;
                        }
                        mask = AscendC::Reg::UpdateMask<float>(active);

                        AscendC::Reg::DataCopy(vregI32, rowSrc + i * VL);
                        AscendC::Reg::DataCopy(vregX3, x3Addr + i * VL);
                        AscendC::Reg::DataCopy(vregBias, biasAddr + i * VL);

                        AscendC::Reg::Cast<float, int32_t, castTraitI32ToF32>(
                            vregFloat, vregI32, mask);

                        AscendC::Reg::Mul<float>(vregY, vregFloat, vregX3, mask);
                        AscendC::Reg::Mul<float>(vregY, vregY, vregToken, mask);
                        // + bias[n]（perchannel），在 gelu 之前
                        AscendC::Reg::Add<float>(vregY, vregY, vregBias, mask);

                        AscendC::Reg::Mul<float>(vregSqr, vregY, vregY, mask);
                        AscendC::Reg::Mul<float>(vregCub, vregSqr, vregY, mask);
                        AscendC::Reg::Axpy<float>(vregCub, vregY, TANH_APPROX_FACTOR, mask);
                        AscendC::Reg::Muls<float>(vregCub, vregCub, NEG_SQRT_EIGHT_OVER_PI, mask);
                        AscendC::Reg::Exp<float>(vregCub, vregCub, mask);
                        AscendC::Reg::Adds<float>(vregCub, vregCub, 1.0f, mask);
                        AscendC::Reg::Div<float>(vregGelu, vregY, vregCub, mask);

                        AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToBF16>(
                            vregBf16, vregGelu, mask);

                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            rowDst + i * VL, vregBf16, mask);
                    }
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);

            {
                int64_t gmRowOffset = dstOffset + stageOffset * N;
                gmRowOffset += AscendC::GetSubBlockIdx() * halfM * N;
                uint16_t nRows = static_cast<uint16_t>(rowsThisStage);
                uint32_t rowBytes = static_cast<uint32_t>(blockShapeN * sizeof(OutputType));
                uint32_t gmRowGap = static_cast<uint32_t>((N - blockShapeN) * sizeof(OutputType));
                int64_t ubRowGapBytes = (nAlign - blockShapeN) * static_cast<int64_t>(sizeof(OutputType));
                AscendC::DataCopyExtParams outParams{nRows, rowBytes, ubRowGapBytes, gmRowGap, 0};
                AscendC::DataCopyPad<OutputType>(outputGlobal_[gmRowOffset], bf16Out_, outParams);
            }
            
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);

            stageOffset += rowsThisStage;
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);
    }

    __host_aicore__ static Params InitParams(Params const& args) { return args; }

    __aicore__ ~ScaleGeluEpilogueRegBase()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
    }

};

#endif // __CCE_AICORE__

#endif // EPILOGUE_SCALE_GELU_EPILOGUE_REGBASE_H
