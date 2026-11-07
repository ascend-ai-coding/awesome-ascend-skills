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

#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

#include "blaze_custom/policy/dispatch_policy.h"

namespace Blaze {
namespace Gemm {
namespace Block {

// Packed dual-branch specialization. A is staged once for each K window while
// act/gate B are packed side by side and accumulated into one L0C tile.
template <
    uint64_t FullLoadMode_, bool AtomicAdd_, bool RefineNearZeroFp16_, class AType_, class LayoutA_,
    class BTypeTuple_, class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<
    MatmulDualBranchGlu<FullLoadMode_, AtomicAdd_, RefineNearZeroFp16_>, AType_, LayoutA_, BTypeTuple_, LayoutB_,
    CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    using DispatchPolicy = MatmulDualBranchGlu<FullLoadMode_, AtomicAdd_, RefineNearZeroFp16_>;
    using AType = AType_;
    static_assert(AscendC::Std::tuple_size_v<BTypeTuple_> == 2, "B type tuple must contain act and gate types");
    using BType = typename AscendC::Std::tuple_element<0, BTypeTuple_>::type;
    using GateBType = typename AscendC::Std::tuple_element<1, BTypeTuple_>::type;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using L0CType = CType_;
    using ScheduleType = typename DispatchPolicy::ScheduleType;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    static_assert(FullLoadMode_ == NONE_FULL_LOAD_MODE, "packed GLU projections support streaming mode only");
    static_assert(!AtomicAdd_, "packed GLU projections own complete accumulation");
    static_assert(AscendC::Std::is_same_v<BType, GateBType>, "act and gate B types must match");
    static_assert(
        (AscendC::Std::is_same_v<AType, int8_t> && AscendC::Std::is_same_v<BType, int8_t> &&
         AscendC::Std::is_same_v<CType, int32_t>) ||
            (AscendC::Std::is_same_v<AType, half> && AscendC::Std::is_same_v<BType, half> &&
             AscendC::Std::is_same_v<CType, float>),
        "packed GLU projections support int8/int8/int32 and fp16/fp16/fp32 only");

    struct Params {
        ProblemShape problemShape;
    };

    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(5);
            AscendC::SetMMLayoutTransform(true);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(5);
            AscendC::SetMMLayoutTransform(false);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        k_ = AscendC::Te::Get<IDX_K_IDX>(params.problemShape);
    }

    template <class TensorA, class TensorBAct, class TensorBGate, class TensorC>
    __aicore__ inline void operator()(
        const TensorA& gmA, const TensorBAct& gmBAct, const TensorBGate& gmBGate, TensorC ubC,
        const BlockShape& fullShape)
    {
        const uint64_t curM = AscendC::Te::Get<IDX_M_TILEIDX>(fullShape);
        const uint64_t curFullN = AscendC::Te::Get<IDX_N_TILEIDX>(fullShape);
        const uint64_t curHalfN = curFullN / 2;
        const uint64_t packedHalfN = CeilAlign(curHalfN, uint64_t{C0_SIZE_B});
        const uint64_t packedFullN = 2 * packedHalfN;
        const uint64_t curMAligned = CeilAlign(curM, 2UL);
        auto layoutL0C = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE_L0C>>(
            curMAligned, packedFullN);
        auto l0c = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, L0CType>(0), layoutL0C);
        IterateK(gmA, gmBAct, gmBGate, l0c, curM, curHalfN, packedHalfN, packedFullN);
        using OutLocation = AscendC::Te::GetMemLocation<TensorC>;
        if constexpr (AscendC::Std::is_same_v<OutLocation, AscendC::Te::Location::UB>) {
            CopyL0C2UbSplitM(ubC, l0c);
        } else if constexpr (AscendC::Std::is_same_v<OutLocation, AscendC::Te::Location::GM>) {
            auto copy = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
            auto actDst = ubC.Slice(AscendC::Te::MakeCoord(0UL, 0UL), AscendC::Te::MakeShape(curM, curHalfN));
            auto gateDst = ubC.Slice(AscendC::Te::MakeCoord(0UL, curHalfN), AscendC::Te::MakeShape(curM, curHalfN));
            auto actSrc = l0c.Slice(AscendC::Te::MakeCoord(0UL, 0UL), AscendC::Te::MakeShape(curM, curHalfN));
            auto gateSrc = l0c.Slice(AscendC::Te::MakeCoord(0UL, packedHalfN), AscendC::Te::MakeShape(curM, curHalfN));
            AscendC::Te::Copy(copy.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), actDst, actSrc);
            AscendC::Te::Copy(copy.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), gateDst, gateSrc);
        } else {
            static_assert(Blaze::Gemm::always_false_v<TensorC>, "dual-branch GLU output must be a GM or UB Tensor");
        }
    }

private:
    template <class TensorC, class TensorL0C>
    __aicore__ inline void CopyL0C2UbSplitM(const TensorC& ubC, const TensorL0C& l0c)
    {
        auto copyL0C2Ub = AscendC::Te::MakeCopy(
            AscendC::Te::CopyL0C2UB{}, Blaze::Gemm::Tile::CopyL0C2UBTraitMixSplitM{});
        AscendC::Te::Copy(copyL0C2Ub.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), ubC, l0c);
    }

    template <class TensorA, class TensorBAct, class TensorBGate, class TensorL0C>
    __aicore__ inline void IterateK(
        const TensorA& gmA, const TensorBAct& gmBAct, const TensorBGate& gmBGate, TensorL0C& l0c, uint64_t curM,
        uint64_t curHalfN, uint64_t packedHalfN, uint64_t packedFullN)
    {
        constexpr uint64_t A_STAGE_BYTES = BASE_M * K_L1 * sizeof(AType);
        constexpr uint64_t B_STAGE_BYTES = K_L1 * BASE_N * sizeof(BType);
        constexpr uint64_t STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;
        const uint64_t windows = CeilDiv(k_, uint64_t{K_L1});
        auto gm2l1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        for (uint64_t window = 0; window < windows; ++window) {
            const uint64_t curK = Min(k_ - window * K_L1, uint64_t{K_L1});
            const uint16_t stage = static_cast<uint16_t>(window & 1U);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(stage);
            const uint64_t stageOffset = stage * STAGE_BYTES;
            auto aL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(stageOffset), MakeLayoutAL1{}(curM, curK));
            auto aWindow = gmA.Slice(AscendC::Te::MakeCoord(0UL, window * K_L1), AscendC::Te::MakeShape(curM, curK));
            AscendC::Te::Copy(gm2l1, aL1, aWindow);
            auto bL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(stageOffset + A_STAGE_BYTES),
                MakeLayoutBL1{}(curK, packedFullN));
            AscendC::Te::Copy(
                gm2l1, bL1.Slice(AscendC::Te::MakeCoord(0UL, 0UL), AscendC::Te::MakeShape(curK, curHalfN)),
                gmBAct.Slice(AscendC::Te::MakeCoord(window * K_L1, 0UL), AscendC::Te::MakeShape(curK, curHalfN)));
            AscendC::Te::Copy(
                gm2l1, bL1.Slice(AscendC::Te::MakeCoord(0UL, packedHalfN), AscendC::Te::MakeShape(curK, curHalfN)),
                gmBGate.Slice(AscendC::Te::MakeCoord(window * K_L1, 0UL), AscendC::Te::MakeShape(curK, curHalfN)));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(stage);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(stage);
            IterateL0(aL1, bL1, l0c, curM, packedFullN, curK, window, windows);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(stage);
        }
    }

    template <class TensorAL1, class TensorBL1, class TensorL0C>
    __aicore__ inline void IterateL0(
        const TensorAL1& aL1, const TensorBL1& bL1, TensorL0C& l0c, uint64_t curM, uint64_t curN, uint64_t curKWindow,
        uint64_t window, uint64_t windows)
    {
        constexpr uint64_t HALF_L0_BYTES = BASE_M * BASE_K * sizeof(AType);
        const uint64_t loops = CeilDiv(curKWindow, uint64_t{BASE_K});
        auto l12a = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto l12b = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        for (uint16_t inner = 0; inner < static_cast<uint16_t>(loops); ++inner) {
            const uint64_t curK = Min(curKWindow - inner * BASE_K, uint64_t{BASE_K});
            const uint16_t stage = l0LoopCount_ & 1U;
            const uint64_t offset = stage * HALF_L0_BYTES;
            auto aL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(offset),
                AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(
                    curM, curK));
            auto bL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(offset),
                AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>(
                    curK, curN));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(4 + stage);
            AscendC::Te::Copy(
                l12a, aL0, aL1.Slice(AscendC::Te::MakeCoord(0UL, inner * BASE_K), AscendC::Te::MakeShape(curM, curK)));
            AscendC::Te::Copy(
                l12b, bL0, bL1.Slice(AscendC::Te::MakeCoord(inner * BASE_K, 0UL), AscendC::Te::MakeShape(curK, curN)));
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(stage);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(stage);
            AscendC::Te::MmadParams p;
            p.m = static_cast<uint16_t>(curM);
            p.n = static_cast<uint16_t>(curN);
            p.k = static_cast<uint16_t>(curK);
            p.unitFlag = (window + 1 == windows && inner + 1 == loops) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
            p.cmatrixInitVal = window == 0 && inner == 0;
            constexpr auto MMAD_ATOM =
                AscendC::Te::MakeMmad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});
            AscendC::Te::Mmad(MMAD_ATOM.with(p), l0c, aL0, bL0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(4 + stage);
            ++l0LoopCount_;
        }
    }

    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr uint64_t BASE_M = 256;
    static constexpr uint64_t BASE_N = 256;
    static constexpr uint64_t BASE_K = 128 / sizeof(AType);
    static constexpr uint64_t STEP_KA = 4;
    static constexpr uint64_t K_L1 = BASE_K * STEP_KA;
    static constexpr int32_t C0_SIZE = AscendC::Te::C0_ELEMENT<AType>;
    static constexpr int32_t C0_SIZE_B = AscendC::Te::C0_ELEMENT<BType>;
    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        TRANS_A, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;
    uint64_t k_{0};
    uint64_t l0LoopCount_{0};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
