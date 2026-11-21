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


/*!
 * \file weight_quant_matmul_block_mmad.h
 * \brief AIC-side BlockMmad for weight-quant matmul (V+C fusion).
 *
 * In V+C weight-quant matmul, AIV first dequantizes the low-bit weight to
 * bf16/fp16 and writes the result into L1.  This BlockMmad then consumes the
 * dequantized B from L1 (NOT from GM) for MMAD.  The prologue (dequant +
 * UB/L1 management + AIV-side CV sync) lives in weight_quant_matmul_kernel.h.
 *
 * L1 layout (shared convention with kernel.h prologue):
 *   lower half: [B0 | A0]    upper half: [B1 | A1]
 *   bias (optional) at L1 tail, 64-byte aligned.
 *
 * CV sync (AIC side):
 *   Constructor pre-sets AIC_SYNC_AIV_FLAG for both L1 buffers × both AIV
 *   sub-blocks so the first K-L1 iteration does not block.
 *   Each K-L1 iteration: WaitFlag(AIV_SYNC_AIC_FLAG) → process →
 *   SetFlag(AIC_SYNC_AIV_FLAG).
 *   Destructor consumes remaining HardEvent flags (no CrossCore waits — all
 *   AIV_SYNC_AIC_FLAGs are consumed in the K-loop).
 */

#ifndef WEIGHT_QUANT_MATMUL_BLOCK_MMAD_H
#define WEIGHT_QUANT_MATMUL_BLOCK_MMAD_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"
#include "../policy/dispatch_policy.h"
#include "../epilogue/cv_sync_constants.h"

namespace Blaze {
namespace Gemm {
namespace Block {
using namespace AscendC;
using Blaze::Gemm::MNK_M;
using Blaze::Gemm::MNK_N;
using Blaze::Gemm::MNK_K;
using Blaze::Gemm::IDX_M_IDX;
using Blaze::Gemm::IDX_N_IDX;
using Blaze::Gemm::IDX_K_IDX;
using Blaze::Gemm::IDX_M_TILEIDX;
using Blaze::Gemm::IDX_N_TILEIDX;
using Blaze::Gemm::FINAL_ACCUMULATION;
using Blaze::Gemm::NON_FINAL_ACCUMULATION;
using Blaze::Gemm::ZERO_FLAG;
using Blaze::Gemm::FIRST_FLAG;
using Blaze::Gemm::DOUBLE_BUFFER_COUNT;
using Blaze::Gemm::CeilDiv;
using Blaze::Gemm::CeilAlign;

static constexpr uint64_t NO_FULL_LOAD_MODE = 0UL;

template <typename LayoutPtn, typename Type, bool TransVal>
struct L1LayoutHelper {
    static constexpr size_t C0 = 32 / sizeof(Type);
    using type = AscendC::Std::conditional_t<
        Blaze::Gemm::IsWeightNz<LayoutPtn>::value,
        AscendC::Te::FrameLayoutFormat<LayoutPtn, AscendC::Std::Int<C0>>,
        AscendC::Std::conditional_t<
            TransVal,
            AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0>>,
            AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0>>>>;
};

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

template <
    class AType_, class LayoutA_, class BType_, class LayoutB_,
    class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<
    WeightQuantMatmulPolicy<NO_FULL_LOAD_MODE>, AType_, LayoutA_, BType_, LayoutB_,
    CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using DequantBType = AType;
    using CType = CType_;
    using BiasType = float;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy = WeightQuantMatmulPolicy<NO_FULL_LOAD_MODE>;
    using ScheduleType = KernelWeightQuantMatmul;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA = Blaze::Gemm::IsTrans<LayoutA>::value;
    static constexpr bool transB = Blaze::Gemm::IsTrans<LayoutB>::value;

    static constexpr uint64_t L1_BUFFER_NUM = 2UL;
    static constexpr uint64_t L1_BUFFER_MASK = L1_BUFFER_NUM - 1UL;
    static constexpr uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
    using L0CType = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<AType, int8_t>, int32_t, float>;
    static constexpr uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr uint64_t BLOCK_CUBE = 16UL;
    static constexpr uint64_t BLOCK_CUBE_L0C = 16UL;

    static constexpr uint16_t CV_SYNC_MODE = CvSync::MODE;
    static constexpr int16_t AIV_SYNC_AIC_FLAG = CvSync::AIV_TO_AIC_FLAG;
    static constexpr int16_t AIC_SYNC_AIV_FLAG = CvSync::AIC_TO_AIV_FLAG;
    static constexpr int16_t FLAG_ID_MAX = 16;

    using MakeLayoutAL1 = typename L1LayoutHelper<LayoutA, AType, transA>::type;
    using MakeLayoutBL1 = typename L1LayoutHelper<LayoutB, DequantBType, transB>::type;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
    };

    __aicore__ inline BlockMmad()
    {
        #pragma unroll
        for (uint8_t i = 0; i < L1_BUFFER_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);

        AscendC::CrossCoreSetFlag<CV_SYNC_MODE, PIPE_MTE1>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreSetFlag<CV_SYNC_MODE, PIPE_MTE1>(AIC_SYNC_AIV_FLAG + 1);
        AscendC::CrossCoreSetFlag<CV_SYNC_MODE, PIPE_MTE1>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
        AscendC::CrossCoreSetFlag<CV_SYNC_MODE, PIPE_MTE1>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX + 1);

        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
        #pragma unroll
        for (uint8_t i = 0; i < L1_BUFFER_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
        AscendC::SetMMLayoutTransform(false);
    }

public:
    __aicore__ inline void Init(
        const TupleShape& problemShape, const TupleShape& l0TileShape,
        const L1Params& l1Params, bool enableL0cPingPong, bool hasBias = false)
    {
        m_ = AscendC::Te::Get<IDX_M_IDX>(problemShape);
        n_ = AscendC::Te::Get<IDX_N_IDX>(problemShape);
        k_ = AscendC::Te::Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        baseM_ = AscendC::Te::Get<IDX_M_IDX>(l0TileShape);
        baseN_ = AscendC::Te::Get<IDX_N_IDX>(l0TileShape);
        baseK_ = AscendC::Te::Get<IDX_K_IDX>(l0TileShape);
        enableL0cPingPong_ = enableL0cPingPong;
        hasBias_ = hasBias;

        aL1OneBuffer_ = baseM_ * kL1_ * sizeof(AType);
        bL1OneBuffer_ = baseN_ * kL1_ * sizeof(DequantBType);

        uint64_t l1HalfSize = AscendC::TOTAL_L1_SIZE >> 1;
        #pragma unroll
        for (uint64_t bufferId = 0; bufferId < L1_BUFFER_NUM; ++bufferId) {
            uint64_t l1HalfOffset = (bufferId & 1UL) * l1HalfSize;
            l1BufferBOffset_[bufferId] = l1HalfOffset;
            l1BufferAOffset_[bufferId] = l1HalfOffset + bL1OneBuffer_;
        }
        if (hasBias_) {
            biasL1Offset_ = AscendC::TOTAL_L1_SIZE - CeilAlign(baseN_ * sizeof(BiasType), 64UL);
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorA, typename TensorBias, typename TensorC>
    __aicore__ inline void operator()(
        TensorA gmA, TensorBias gmBias, TensorC gmC, BlockShape singleShape)
    {
        auto curM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
        auto curN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
        uint16_t l0cBufId = static_cast<uint16_t>(l0cPingPong_ & 0x1);
        uint64_t l0cOffset = l0cBufId * HALF_L0C_SIZE;
        auto layoutL0C = AscendC::Te::MakeFrameLayout<
            AscendC::Te::NZLayoutPtn, AscendC::Std::Int<BLOCK_CUBE_L0C>>(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, L0CType>(l0cOffset), layoutL0C);

        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});

        auto layoutBiasL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, curN);
        auto tensorBiasL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(biasL1Offset_), layoutBiasL1);
        auto copyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{});

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cBufId);

        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & L1_BUFFER_MASK;
            uint64_t kL1Offset = iter0 * kL1_;
            auto curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1Offset) : kL1_;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);

            auto layoutAL1 = MakeLayoutAL1{}(curM, curKL1);
            auto tensorAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[l1BufId]),
                layoutAL1);
            auto gmBlockA = gmA.Slice(
                AscendC::Te::MakeCoord(0, kL1Offset), AscendC::Te::MakeShape(curM, curKL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);

            if (hasBias_ && iter0 == 0) {
                AscendC::Te::Copy(copyGM2L1, tensorBiasL1, gmBias);
            }

            auto layoutBL1 = MakeLayoutBL1{}(curKL1, curN);
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, DequantBType>(l1BufferBOffset_[l1BufId]),
                layoutBL1);

            AscendC::CrossCoreWaitFlag<CV_SYNC_MODE, PIPE_MTE1>(AIV_SYNC_AIC_FLAG + l1BufId);
            AscendC::CrossCoreWaitFlag<CV_SYNC_MODE, PIPE_MTE1>(AIV_SYNC_AIC_FLAG + l1BufId + FLAG_ID_MAX);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t kL0Iter = CeilDiv(curKL1, baseK_);
            for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                auto kL0Offset = iter1 * baseK_;
                auto curKL0 = (kL0Offset + baseK_ > curKL1) ? (curKL1 - kL0Offset) : baseK_;
                uint64_t l0BufId = l0PingPong_ & 0x1;
                uint64_t l0Offset = HALF_L0_SIZE * l0BufId;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);

                auto layoutAL0 = AscendC::Te::MakeFrameLayout<
                    AscendC::Te::NZLayoutPtn, AscendC::Std::Int<BLOCK_CUBE>>(curM, curKL0);
                auto tensorAL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
                auto tensorBlockAL1 = tensorAL1.Slice(
                    AscendC::Te::MakeCoord(0, kL0Offset), AscendC::Te::MakeShape(curM, curKL0));
                AscendC::Te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);

                auto layoutBL0 = AscendC::Te::MakeFrameLayout<
                    AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<BLOCK_CUBE>>(curKL0, curN);
                auto tensorBL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, DequantBType>(l0Offset), layoutBL0);
                auto tensorBlockBL1 = tensorBL1.Slice(
                    AscendC::Te::MakeCoord(kL0Offset, 0), AscendC::Te::MakeShape(curKL0, curN));
                AscendC::Te::Copy(copyL12L0B, tensorBL0, tensorBlockBL1);

                if (hasBias_ && iter0 == 0 && iter1 == 0) {
                    auto layoutBiasBT = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, curN);
                    auto tensorBiasBT = AscendC::Te::MakeTensor(
                        AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, BiasType>(
                            baseN_ * l0cBufId * sizeof(BiasType)), layoutBiasBT);
                    AscendC::Te::Copy(copyL12BT, tensorBiasBT, tensorBiasL1);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);

                uint8_t mmadUnitFlag =
                    (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
                bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0);
                AscendC::Te::MmadParams mmadParams{
                    static_cast<uint16_t>(curM),
                    static_cast<uint16_t>(curN),
                    static_cast<uint16_t>(curKL0),
                    mmadUnitFlag,
                    mmadCmatrixInitVal};

                if (hasBias_ && mmadCmatrixInitVal) {
                    auto layoutBiasBT = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, curN);
                    auto tensorBiasBT = AscendC::Te::MakeTensor(
                        AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, BiasType>(
                            baseN_ * l0cBufId * sizeof(BiasType)), layoutBiasBT);
                    AscendC::Te::Mmad(
                        AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation>>{}.with(mmadParams),
                        tensorL0C, tensorAL0, tensorBL0, tensorBiasBT);
                } else {
                    AscendC::Te::Mmad(
                        AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation>>{}.with(mmadParams),
                        tensorL0C, tensorAL0, tensorBL0);
                }

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                l0PingPong_++;
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            AscendC::CrossCoreSetFlag<CV_SYNC_MODE, PIPE_MTE1>(AIC_SYNC_AIV_FLAG + l1BufId);
            AscendC::CrossCoreSetFlag<CV_SYNC_MODE, PIPE_MTE1>(AIC_SYNC_AIV_FLAG + l1BufId + FLAG_ID_MAX);

            abL1LoopCnt_++;
        }

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cBufId);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cBufId);
        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        copyL0C2GM.Call(gmC, tensorL0C, AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cBufId);

        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    uint64_t m_{0UL};
    uint64_t n_{0UL};
    uint64_t k_{0UL};
    uint64_t kL1Iter_{0UL};
    uint64_t kL1_{0UL};
    uint64_t baseM_{0UL};
    uint64_t baseN_{0UL};
    uint64_t baseK_{0UL};
    uint64_t abL1LoopCnt_{0UL};
    uint64_t l0PingPong_{0UL};
    uint64_t l0cPingPong_{0UL};
    bool enableL0cPingPong_{false};
    bool hasBias_{false};

    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t biasL1Offset_ = 0UL;
    uint64_t l1BufferAOffset_[L1_BUFFER_NUM] = {0UL};
    uint64_t l1BufferBOffset_[L1_BUFFER_NUM] = {0UL};
};
#endif
} // namespace Block
} // namespace Gemm
} // namespace Blaze

#endif // WEIGHT_QUANT_MATMUL_BLOCK_MMAD_H
