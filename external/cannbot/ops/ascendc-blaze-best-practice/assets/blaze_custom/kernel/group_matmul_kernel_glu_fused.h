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

#ifndef GROUP_MATMUL_KERNEL_GLU_FUSED_H
#define GROUP_MATMUL_KERNEL_GLU_FUSED_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "tensor_api/tensor.h"

#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze_custom/epilogue/cv_sync_constants.h"
#include "blaze_custom/kernel/group_matmul_kernel_cv1_v2.h"
#include "blaze_custom/policy/dispatch_policy.h"
#include "op_tiling/matmul/blaze_group_matmul_tiling_data.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

// Group-aware fused GLU kernel. It owns group traversal, act/gate B views,
// MMAD/Epilogue ordering and CV handshakes. It deliberately does not own
// dequantization, per-token quantization, or operator-private UB derivation.
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<KernelMmadDualBranchGlu, typename BlockMmad_::DispatchPolicy::ScheduleType> &&
        !IsCv1V2EpiloguePipeline<BlockEpilogue_>::value>> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using BlockShape = typename BlockMmad::BlockShape;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Te::LayoutTraitDefault<BType>>;
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool REFINE_NEAR_ZERO_FP16 = BlockMmad::DispatchPolicy::REFINE_NEAR_ZERO_FP16;

    static_assert(AscendC::Std::is_same_v<ProblemShape, typename BlockMmad::ProblemShape>,
        "Kernel ProblemShape must match BlockMmad ProblemShape");
    static_assert(
        AscendC::Std::is_one_of_v<LayoutA, AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn> &&
            AscendC::Std::is_one_of_v<LayoutB, AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn>,
        "grouped GLU kernel supports contiguous ND/DN LayoutA and LayoutB only; NZ/ZN requires adaptation");
    static_assert(
        !REFINE_NEAR_ZERO_FP16 || (!TRANS_A && !TRANS_B),
        "near-zero FP16 refinement supports non-transposed LayoutA and LayoutB only");
    static_assert(CvSync::COUNT_ID_MAX > 0, "CvSync::COUNT_ID_MAX must be positive");
    static_assert(CvSync::COUNT_FLAG > 0, "CvSync::COUNT_FLAG must be positive");

    struct Params {
        GM_ADDR xGmAddr{nullptr};
        GM_ADDR weightGmAddr{nullptr};
        typename BlockEpilogue::Params epilogueParams{};
        uint64_t lda{0};
        uint64_t splitMRows{0};
        uint64_t cLocalPitch{0};
        // The concrete kernel entry owns GM-to-local tiling decoding through
        // GET_TILING_DATA_WITH_STRUCT/GET_TILING_DATA_MEMBER. This kernel only
        // consumes the resulting local POD; its lifetime must cover operator().
        const ::GroupMatmulTilingData* groupTilingData{nullptr};
        float refineAbsThreshold{0.0f};
    };

    __aicore__ inline void operator()(const Params& params, __gm__ float* outputGmAddr)
    {
        if (params.groupTilingData == nullptr) {
            return;
        }
        const ::GroupMatmulTilingData& groupTiling = *params.groupTilingData;
        const auto& tiling = groupTiling.matmul;
        if (tiling.n == 0 || (tiling.n & 1U) != 0 || tiling.k == 0 || groupTiling.groupNum == 0 ||
            groupTiling.groupListAddr == 0 || groupTiling.groupListType > 1 || params.splitMRows == 0 ||
            params.cLocalPitch == 0) {
            return;
        }
        AscendC::GlobalTensor<int64_t> groupList;
        groupList.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupTiling.groupListAddr));
        if (!ValidateGroupList(groupList, groupTiling, tiling.m)) {
            return;
        }
        BlockScheduler scheduler(typename BlockScheduler::Params{
            static_cast<int32_t>(tiling.baseM), static_cast<int32_t>(tiling.baseN), tiling.usedCoreNum,
            static_cast<uint32_t>(AscendC::GetTaskRation())});
        BlockEpilogue epilogue;
        epilogue.Init(params.epilogueParams);

        int64_t prefixM = 0;
        int64_t count = 0;
        for (uint32_t groupIdx = 0; groupIdx < groupTiling.groupNum; ++groupIdx) {
            const int64_t encoded = groupList.GetValue(groupIdx);
            const int64_t groupM = groupTiling.groupListType == 0 ? encoded - prefixM : encoded;
            const int64_t endM = groupTiling.groupListType == 0 ? encoded : prefixM + groupM;
            if (groupM == 0) {
                prefixM = endM;
                continue;
            }

            ProblemShape problemShape{groupM, tiling.n, tiling.k, 1};
            scheduler.UpdateNextProblem(problemShape);
            BlockMmad blockMmad;
            if ASCEND_IS_AIC {
                blockMmad.Init(typename BlockMmad::Params{problemShape});
            }
            auto xLayout = MakeXLayout(groupM, tiling.k, params.lda);
            auto* xPtr = reinterpret_cast<__gm__ AType*>(params.xGmAddr) + GroupOffset(prefixM, params.lda);
            auto gmX = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(xPtr), xLayout);
            auto weightLayout = MakeLayoutB{}(tiling.k, tiling.n);
            auto* weightPtr = reinterpret_cast<__gm__ BType*>(params.weightGmAddr) +
                              static_cast<uint64_t>(groupIdx) * tiling.k * tiling.n;
            auto gmWeight =
                AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(weightPtr), weightLayout);

            typename BlockScheduler::BlockCoord coord;
            bool outstanding = false;
            while (scheduler.GetTileIdx(coord)) {
                auto schedulerShape = scheduler.GetBlockShape(coord);
                const int64_t curM = AscendC::Te::Get<0>(schedulerShape);
                const int64_t curFullN = AscendC::Te::Get<1>(schedulerShape);
                const int64_t mOffset = AscendC::Te::Get<0>(coord) * static_cast<int64_t>(tiling.baseM);
                const int64_t fullNOffset = AscendC::Te::Get<1>(coord) * static_cast<int64_t>(tiling.baseN);
                const int64_t curH = curFullN / 2;
                const int64_t halfNOffset = fullNOffset / 2;
                constexpr int64_t B_C0 = AscendC::Te::C0_ELEMENT<BType>;
                const int64_t packedHalfN = (curH + B_C0 - 1) / B_C0 * B_C0;
                const int64_t packedFullN = 2 * packedHalfN;
                if (curM <= 0 || curH <= 0 || halfNOffset + curH > static_cast<int64_t>(tiling.n / 2) ||
                    static_cast<uint64_t>(packedFullN) > params.cLocalPitch ||
                    static_cast<uint64_t>(curM) > 2UL * params.splitMRows) {
                    return;
                }

                BlockShape blockShape{curM, curFullN, tiling.k, 0};
                typename BlockEpilogue::TileContext context{groupIdx,    prefixM,     mOffset,
                                                            halfNOffset, curM,        curH,
                                                            packedHalfN, packedFullN, static_cast<int64_t>(tiling.n / 2)};
                if ASCEND_IS_AIC {
                    if (outstanding) {
                        WaitForVector(count);
                    }
                    auto a = gmX.Slice(
                        AscendC::Te::MakeCoord(mOffset, static_cast<int64_t>(0)),
                        AscendC::Te::MakeShape(curM, static_cast<int64_t>(tiling.k)));
                    auto act = gmWeight.Slice(
                        AscendC::Te::MakeCoord(static_cast<int64_t>(0), halfNOffset),
                        AscendC::Te::MakeShape(static_cast<int64_t>(tiling.k), curH));
                    auto gate = gmWeight.Slice(
                        AscendC::Te::MakeCoord(
                            static_cast<int64_t>(0), static_cast<int64_t>(tiling.n / 2) + halfNOffset),
                        AscendC::Te::MakeShape(static_cast<int64_t>(tiling.k), curH));
                    auto accumulator = epilogue.AccumulatorTensor(curM, packedFullN);
                    blockMmad(a, act, gate, accumulator, blockShape);
                    ++count;
                    NotifyVector(count);
                    outstanding = true;
                }
                if ASCEND_IS_AIV {
                    ++count;
                    WaitForCube(count);
                    if constexpr (REFINE_NEAR_ZERO_FP16) {
                        RefineNearZero(context, xPtr, weightPtr, params.lda, tiling.k, params.refineAbsThreshold);
                        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                    }
                    epilogue(context, outputGmAddr);
                    NotifyCube(count);
                }
            }
            if ASCEND_IS_AIC {
                if (outstanding) {
                    WaitForVector(count);
                }
            }
            prefixM = endM;
        }
    }

private:
    __aicore__ inline static auto MakeXLayout(int64_t m, uint32_t k, uint64_t lda)
    {
        auto shape = AscendC::Te::MakeShape(
            AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, m), AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, k));
        if constexpr (TRANS_A) {
            auto stride = AscendC::Te::MakeStride(
                AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}),
                AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, lda));
            return AscendC::Te::MakePatternLayout<
                AscendC::Te::DNExtLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(shape, stride);
        } else {
            auto stride = AscendC::Te::MakeStride(
                AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, lda),
                AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
            return AscendC::Te::MakePatternLayout<
                AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(shape, stride);
        }
    }

    __aicore__ inline static uint64_t GroupOffset(int64_t prefixM, uint64_t lda)
    {
        if constexpr (TRANS_A) {
            return static_cast<uint64_t>(prefixM);
        }
        return static_cast<uint64_t>(prefixM) * lda;
    }

    template <class Context>
    __aicore__ inline static void RefineNearZero(
        const Context& context, __gm__ AType* xGroup, __gm__ BType* weightExpert, uint64_t lda, uint64_t k,
        float refineAbsThreshold)
    {
        static_assert(
            AscendC::Std::is_same_v<AType, half> && AscendC::Std::is_same_v<BType, half>,
            "near-zero refinement is defined for FP16 inputs");
        if (xGroup == nullptr || weightExpert == nullptr || lda == 0 || k == 0 || refineAbsThreshold <= 0.0f) {
            return;
        }
        const int64_t halfM = (context.curM + AscendC::GetTaskRation() - 1) / AscendC::GetTaskRation();
        const int64_t localRows =
            (context.curM & 1L) != 0 ? halfM - static_cast<int64_t>(AscendC::GetSubBlockIdx()) : halfM;
        if (localRows <= 0) {
            return;
        }
        AscendC::GlobalTensor<half> x;
        AscendC::GlobalTensor<half> weight;
        x.SetGlobalBuffer(xGroup);
        weight.SetGlobalBuffer(weightExpert);
        AscendC::LocalTensor<float> cLocal(
            AscendC::TPosition::VECIN, 0, static_cast<uint32_t>(localRows * context.packedFullN));
        const int64_t localRowBase = context.mOffset + static_cast<int64_t>(AscendC::GetSubBlockIdx()) * halfM;
        const uint64_t fullN = static_cast<uint64_t>(2 * context.h);
        for (int64_t row = 0; row < localRows; ++row) {
            const uint32_t rowOffset = static_cast<uint32_t>(row * context.packedFullN);
            const uint64_t xRow = static_cast<uint64_t>(localRowBase + row) * lda;
            for (int64_t col = 0; col < context.curH; ++col) {
                const uint32_t actOffset = rowOffset + static_cast<uint32_t>(col);
                const uint32_t gateOffset = rowOffset + static_cast<uint32_t>(context.packedHalfN + col);
                const float actCandidate = cLocal.GetValue(actOffset);
                const float gateCandidate = cLocal.GetValue(gateOffset);
                const bool actNeedsRefinement = actCandidate > -refineAbsThreshold && actCandidate < refineAbsThreshold;
                const bool gateNeedsRefinement =
                    gateCandidate > -refineAbsThreshold && gateCandidate < refineAbsThreshold;
                if (!actNeedsRefinement && !gateNeedsRefinement) {
                    continue;
                }
                float act = 0.0f;
                float gate = 0.0f;
                const uint64_t actColumn = static_cast<uint64_t>(context.halfNOffset + col);
                const uint64_t gateColumn = static_cast<uint64_t>(context.h) + actColumn;
                for (uint64_t inner = 0; inner < k; ++inner) {
                    const float xValue = static_cast<float>(x.GetValue(xRow + inner));
                    act += xValue * static_cast<float>(weight.GetValue(inner * fullN + actColumn));
                    gate += xValue * static_cast<float>(weight.GetValue(inner * fullN + gateColumn));
                }
                cLocal.SetValue(actOffset, act);
                cLocal.SetValue(gateOffset, gate);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline static bool ValidateGroupList(
        const AscendC::GlobalTensor<int64_t>& groupList, const GroupMatmulTilingData& groupTiling, uint32_t totalM)
    {
        int64_t accumulated = 0;
        for (uint32_t groupIdx = 0; groupIdx < groupTiling.groupNum; ++groupIdx) {
            const int64_t encoded = groupList.GetValue(groupIdx);
            if (groupTiling.groupListType == 0) {
                if (encoded < accumulated) {
                    return false;
                }
                accumulated = encoded;
            } else {
                if (encoded < 0) {
                    return false;
                }
                accumulated += encoded;
            }
        }
        return accumulated == static_cast<int64_t>(totalM);
    }

    static constexpr int16_t SECOND_VECTOR_OFFSET = 16;

    __aicore__ inline static int16_t Slot(int64_t count)
    {
        return static_cast<int16_t>(count / CvSync::COUNT_ID_MAX % CvSync::COUNT_FLAG);
    }

    __aicore__ inline static void WaitForVector(int64_t count)
    {
        const int16_t slot = Slot(count);
        AscendC::CrossCoreWaitFlag<CvSync::MODE, PIPE_FIX>(CvSync::AIV_TO_AIC_FLAG + slot);
        AscendC::CrossCoreWaitFlag<CvSync::MODE, PIPE_FIX>(
            CvSync::AIV_TO_AIC_FLAG + slot + SECOND_VECTOR_OFFSET);
    }

    __aicore__ inline static void NotifyVector(int64_t count)
    {
        const int16_t slot = Slot(count);
        AscendC::CrossCoreSetFlag<CvSync::MODE, PIPE_FIX>(CvSync::AIC_TO_AIV_FLAG + slot);
        AscendC::CrossCoreSetFlag<CvSync::MODE, PIPE_FIX>(
            CvSync::AIC_TO_AIV_FLAG + slot + SECOND_VECTOR_OFFSET);
    }

    __aicore__ inline static void WaitForCube(int64_t count)
    {
        if constexpr (REFINE_NEAR_ZERO_FP16) {
            AscendC::CrossCoreWaitFlag<CvSync::MODE, PIPE_S>(CvSync::AIC_TO_AIV_FLAG + Slot(count));
        } else {
            AscendC::CrossCoreWaitFlag<CvSync::MODE, PIPE_V>(CvSync::AIC_TO_AIV_FLAG + Slot(count));
        }
    }

    __aicore__ inline static void NotifyCube(int64_t count)
    {
        AscendC::CrossCoreSetFlag<CvSync::MODE, PIPE_MTE3>(CvSync::AIV_TO_AIC_FLAG + Slot(count));
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze

#endif // GROUP_MATMUL_KERNEL_GLU_FUSED_H
