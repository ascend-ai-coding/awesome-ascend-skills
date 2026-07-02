/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUP_MATMUL_KERNEL_H
#define GROUP_MATMUL_KERNEL_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "kernel_utils/common_utils.h"
#include "kernel_utils/layout_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "include/tensor_api/tensor.h"

#include "../block/group_matmul_block_scheduler.h"
#include "../epilogue/cv_sync_constants.h"

namespace Kernel {

template <class Epilogue>
struct GroupMatmulEpilogueTraits {
    static constexpr bool enabled = true;
    using Params = typename Epilogue::Params;
};

template <>
struct GroupMatmulEpilogueTraits<void> {
    static constexpr bool enabled = false;
    struct Params {};
};

// GroupMatmul is a delta over the Matmul kernel: this template owns groupList,
// prefix-M, per-group tensor views, and scheduler refresh. The default path is
// pure AIC direct writeback through BlockMmad. Pass a non-void Epilogue only
// for fused AIC/AIV kernels whose BlockMmad/Epilogue/CrossCore protocol is
// proven together by the caller. GroupMatmul fused epilogues are view-based:
// the kernel passes tile context, and the concrete Epilogue decides how many
// fused input/output tensors to build. Within that Epilogue adapter, group
// selection may move GM pointers to the current group base, but tile-local
// m/n offsets must be expressed by Slice. Ordinary Matmul epilogues that take
// only a linear gmOffset are not directly compatible.
template <class ProblemShape, class BlockMmad, class BlockScheduler, class Epilogue = void>
class GroupMatmulKernel {
public:
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = CvSync::MODE;
    static constexpr int16_t AIV_SYNC_AIC_FLAG = CvSync::AIV_TO_AIC_FLAG;
    static constexpr int16_t AIC_SYNC_AIV_FLAG = CvSync::AIC_TO_AIV_FLAG;
    static constexpr int16_t FLAG_ID_MAX = 16;
    static constexpr int16_t COUNT_ID_MAX = CvSync::COUNT_ID_MAX;
    static constexpr int16_t COUNT_FLAG = CvSync::COUNT_FLAG;
    static constexpr int64_t M_TAIL_ALIGN = 16;
    static constexpr int64_t N_TAIL_ALIGN = 16;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using L0CType = typename BlockMmad::L0CType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShapeType = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using EpilogueTraits = GroupMatmulEpilogueTraits<Epilogue>;
    using EpilogueParams = typename EpilogueTraits::Params;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB>;
    static constexpr bool HAS_EPILOGUE = EpilogueTraits::enabled;

    struct GroupTiling {
        uint32_t groupNum{0};
        uint32_t baseM{0};
        uint32_t baseN{0};
        uint32_t baseK{0};
        uint8_t dbL0C{0};
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmParams;
        L1Params l1Params;
        BlockSchedulerParams schedulerParams;
        GroupTiling groupTiling;
        EpilogueParams epilogueParams;
        GM_ADDR groupListGmAddr{nullptr};
    };

    struct TileContext {
        uint32_t groupIdx{0};
        uint32_t groupNum{0};
        int64_t prefixM{0};
        int64_t groupM{0};
        int64_t computeGroupM{0};
        int64_t mOffset{0};
        int64_t nOffset{0};
        int64_t curM{0};
        int64_t curN{0};
        int64_t writeM{0};
        int64_t totalM{0};
        int64_t totalN{0};
        int64_t totalK{0};
    };

    __aicore__ inline void operator()(const Params& params)
    {
        AscendC::GlobalTensor<int64_t> groupListGlobal;
        groupListGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.groupListGmAddr));

        int64_t prefixM = 0;
        int64_t count = 0;
        int64_t countId = 0;
        bool enableCVSync = false;

        BlockScheduler bs(params.schedulerParams);
        bs.SetTailAlign(M_TAIL_ALIGN, N_TAIL_ALIGN);

        for (uint32_t groupIdx = 0; groupIdx < params.groupTiling.groupNum; ++groupIdx) {
            int64_t groupM = groupListGlobal.GetValue(groupIdx);
            // groupList values are a caller contract. The host must not read them
            // for validation, but the device path should avoid casting a non-positive
            // value into a huge unsigned shape during bring-up.
            if (groupM <= 0) {
                continue;
            }

            int64_t computeGroupM = groupM;
            uint32_t curBaseM = CalcBalancedBaseM(computeGroupM, params.groupTiling.baseM);
            bs.UpdateBaseM(curBaseM);
            typename BlockScheduler::TupleShape bsProblemShape{
                computeGroupM, params.problemShape.n, params.problemShape.k, 1L};
            bs.UpdateNextProblem(bsProblemShape);
            if (IsLastGroupAndNeedSplit(bs, groupIdx, params.groupTiling.groupNum, params.problemShape.n, prefixM)) {
                bs.UpdateTailTile();
            }

            TupleShape groupProblemShape{computeGroupM, params.problemShape.n, params.problemShape.k};
            BlockShape l0TileShape{
                static_cast<int64_t>(curBaseM),
                static_cast<int64_t>(params.groupTiling.baseN),
                static_cast<int64_t>(params.groupTiling.baseK),
                0};
            bool enableL0cPingPong = (params.groupTiling.dbL0C > 1);
            BlockMmad blockMmadOp;
            if ASCEND_IS_AIC {
                blockMmadOp.Init(groupProblemShape, l0TileShape, params.l1Params, enableL0cPingPong);
            }

            if constexpr (HAS_EPILOGUE) {
                Epilogue epilogueOp;
                ProblemShapeType totalProblemShape{
                    params.problemShape.m, params.problemShape.n, params.problemShape.k, 1};
                epilogueOp.Init(params.epilogueParams, curBaseM, params.groupTiling.baseN, totalProblemShape);

                ProcessSingleGroup(params, bs, blockMmadOp, &epilogueOp, groupIdx, prefixM, groupM, computeGroupM,
                    curBaseM, count, countId, enableCVSync);
                DrainEpilogue(count, countId, enableCVSync);
            } else {
                ProcessSingleGroup(params, bs, blockMmadOp, nullptr, groupIdx, prefixM, groupM, computeGroupM,
                    curBaseM, count, countId, enableCVSync);
            }
            prefixM += groupM;
        }

        if constexpr (HAS_EPILOGUE) {
            DrainEpilogue(count, countId, enableCVSync);
        }
    }

private:
    __aicore__ inline uint32_t CalcBalancedBaseM(int64_t m, int64_t baseM) const
    {
        int64_t mCnt = CeilDiv(m, baseM);
        int64_t balanced = CeilAlign(CeilDiv(m, mCnt), M_TAIL_ALIGN);
        return static_cast<uint32_t>(balanced);
    }

    __aicore__ inline bool IsLastGroupAndNeedSplit(
        const BlockScheduler& bs, uint32_t groupIdx, uint32_t groupNum, int64_t n, int64_t prefixM) const
    {
        (void)bs;
        (void)groupIdx;
        (void)groupNum;
        (void)n;
        (void)prefixM;
        // The split-M scheduler keeps tail-split support, but the fused AIC/AIV epilogue path needs
        // a sync-safe split-tail protocol before enabling it for production.
        return false;
    }

    __aicore__ inline void ProcessSingleGroup(const Params& params, BlockScheduler& bs, BlockMmad& blockMmadOp,
        Epilogue* epilogueOp, uint32_t groupIdx, int64_t prefixM, int64_t groupM, int64_t computeGroupM,
        uint32_t curBaseM, int64_t& count, int64_t& countId, bool& enableCVSync)
    {
        auto layoutA = MakeLayoutA{}(computeGroupM, params.problemShape.k);
        auto layoutB = MakeLayoutB{}(params.problemShape.k, params.problemShape.n);
        auto layoutC = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(
            computeGroupM, params.problemShape.n);

        __gm__ AType* aGmPtr = reinterpret_cast<__gm__ AType*>(params.mmParams.aGmAddr);
        __gm__ BType* bGmPtr = reinterpret_cast<__gm__ BType*>(params.mmParams.bGmAddr);
        __gm__ CType* cGmPtr = reinterpret_cast<__gm__ CType*>(params.mmParams.cGmAddr);
        __gm__ AType* aGroupPtr = aGmPtr + prefixM * params.problemShape.k;
        __gm__ BType* bExpertPtr =
            bGmPtr + static_cast<int64_t>(groupIdx) * params.problemShape.n * params.problemShape.k;
        __gm__ CType* cGroupPtr = cGmPtr + prefixM * params.problemShape.n;

        auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGroupPtr), layoutA);
        auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bExpertPtr), layoutB);
        auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGroupPtr), layoutC);

        BlockCoord blockIdx;
        while (bs.GetTileIdx(blockIdx)) {
            BlockShape singleShape = bs.GetBlockShape(blockIdx);
            int64_t curM = Get<MNK_M>(singleShape);
            int64_t curN = Get<MNK_N>(singleShape);
            if (curM <= 0 || curN <= 0) {
                return;
            }

            int64_t mOffset = Get<MNK_M>(blockIdx) * static_cast<int64_t>(curBaseM) +
                              Get<MNK_K>(singleShape);
            int64_t nOffset = Get<MNK_N>(blockIdx) * static_cast<int64_t>(params.groupTiling.baseN) +
                              Get<MNK_B>(singleShape);

            if ASCEND_IS_AIC {
                if constexpr (HAS_EPILOGUE) {
                    if (enableCVSync) {
                        countId = count / COUNT_ID_MAX % COUNT_FLAG;
                        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + countId);
                        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                            AIV_SYNC_AIC_FLAG + countId + FLAG_ID_MAX);
                    }
                }

                auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(mOffset, 0),
                    AscendC::Te::MakeShape(curM, params.problemShape.k));
                auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(0, nOffset),
                    AscendC::Te::MakeShape(params.problemShape.k, curN));

                if constexpr (HAS_EPILOGUE) {
                    auto curMPad = (curM + 1L) & ~1L;
                    constexpr int64_t UB_N_ALIGN_ELEM = 32L / static_cast<int64_t>(sizeof(L0CType));
                    auto curNUbAlign = ((curN + UB_N_ALIGN_ELEM - 1L) / UB_N_ALIGN_ELEM) * UB_N_ALIGN_ELEM;
                    auto layoutUB = AscendC::Te::MakeFrameLayout<
                        AscendC::Te::NDExtLayoutPtn, AscendC::Std::Int<BlockMmad::BLOCK_CUBE_L0C>>(
                            curMPad, curNUbAlign);
                    auto ubBlockC = AscendC::Te::MakeTensor(
                        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, L0CType>(0), layoutUB);
                    blockMmadOp(gmBlockA, gmBlockB, ubBlockC, singleShape);
                } else {
                    auto gmBlockC = gmC.Slice(AscendC::Te::MakeCoord(mOffset, nOffset),
                        AscendC::Te::MakeShape(curM, curN));
                    blockMmadOp(gmBlockA, gmBlockB, gmBlockC, singleShape);
                }

                if constexpr (HAS_EPILOGUE) {
                    enableCVSync = true;
                    count++;
                    countId = count / COUNT_ID_MAX % COUNT_FLAG;
                    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + countId);
                    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIC_SYNC_AIV_FLAG + countId + FLAG_ID_MAX);
                }
            }

            if constexpr (HAS_EPILOGUE) {
                // Optional fused path: epilogue params are interpreted only after the
                // caller has chosen a mixed AIC/AIV launch and paired CrossCore protocol.
                // Pure GroupMatmul implementations should keep Epilogue=void. A non-void
                // Epilogue must provide operator()(blockShape, tileContext, flagId).
                // The Epilogue adapter owns its fused input count and tensor layouts.
                int64_t writeM = Min(curM, groupM - mOffset);
                if (writeM < 0) {
                    writeM = 0;
                }

                if ASCEND_IS_AIV {
                    count++;
                    countId = count / COUNT_ID_MAX % COUNT_FLAG;
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG + countId);

                    TileContext tileContext{
                        groupIdx,
                        params.groupTiling.groupNum,
                        prefixM,
                        groupM,
                        computeGroupM,
                        mOffset,
                        nOffset,
                        curM,
                        curN,
                        writeM,
                        params.problemShape.m,
                        params.problemShape.n,
                        params.problemShape.k};
                    (*epilogueOp)({writeM, curN, 1, 1}, tileContext, (AIV_SYNC_AIC_FLAG + countId));
                }
            }
        }
    }

    __aicore__ inline void DrainEpilogue(int64_t& count, int64_t& countId, bool& enableCVSync)
    {
        if ASCEND_IS_AIC {
            if (enableCVSync) {
                countId = count / COUNT_ID_MAX % COUNT_FLAG;
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + countId);
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                    AIV_SYNC_AIC_FLAG + countId + FLAG_ID_MAX);
                enableCVSync = false;
            }
        }
        count = 0;
        countId = 0;
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

} // namespace Kernel

#endif // GROUP_MATMUL_KERNEL_H
