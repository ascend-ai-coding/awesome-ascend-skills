/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_KERNEL_FUSED_H
#define MATMUL_KERNEL_FUSED_H

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

#include "../block/matmul_block_mmad.h"
#include "../block/matmul_block_scheduler.h"
#include "../utils/matmul_constant.h"
#include "../epilogue/cv_sync_constants.h"

namespace Kernel {

template <class ProblemShape, class BlockMmad, class BlockScheduler, class Epilogue>
class MatmulKernelFused {
public:
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = CvSync::MODE;
    static constexpr int16_t AIV_SYNC_AIC_FLAG = CvSync::AIV_TO_AIC_FLAG;
    static constexpr int16_t AIC_SYNC_AIV_FLAG = CvSync::AIC_TO_AIV_FLAG;
    static constexpr int16_t FLAG_ID_MAX = 16;
    static constexpr int16_t COUNT_ID_MAX = CvSync::COUNT_ID_MAX;
    static constexpr int16_t COUNT_FLAG = CvSync::COUNT_FLAG;

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, BlockScheduler, transA, transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;
    using EpilogueParams = typename Epilogue::Params;
    using ProblemShapeType = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB>;

    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint8_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        EpilogueParams epilogueParams;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        BlockSchedulerOp bs(params.problemShape, params.schParams);

        Epilogue epilogueOp;
        ProblemShapeType problemShape4 = {
            params.problemShape.m, params.problemShape.n, params.problemShape.k, 1};
        epilogueOp.Init(params.epilogueParams, params.qbmmParams.baseM, params.qbmmParams.baseN, problemShape4);

        BlockMmad blockMmadOp;
        if ASCEND_IS_AIC {
            TupleShape problemShape3 = {
                params.problemShape.m, params.problemShape.n, params.problemShape.k};
            BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
            bool enableL0cPingPong = (params.qbmmParams.dbL0C > 1);
            blockMmadOp.Init(problemShape3, l0TileShape, params.l1Params, enableL0cPingPong);
        }

        auto layoutA = MakeLayoutA{}(params.problemShape.m, params.problemShape.k);
        auto layoutB = MakeLayoutB{}(params.problemShape.k, params.problemShape.n);

        int64_t n = params.problemShape.n;
        int64_t count = 0;
        int64_t countId = 0;
        bool enableCVSync = false;
        constexpr int64_t kPos = 0L;

        __gm__ AType* aGmPtr = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        __gm__ BType* bGmPtr = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        __gm__ CType* cGmPtr = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);

        BlockCoord blockIdx;
        while (bs.GetTileIdx(blockIdx)) {
            int64_t mPos = Get<MNK_M>(blockIdx);
            int64_t nPos = Get<MNK_N>(blockIdx);
            BlockShape singleShape = bs.GetBlockShape(blockIdx);
            int64_t curM = Get<MNK_M>(singleShape);
            int64_t curN = Get<MNK_N>(singleShape);
            if (curM <= 0 || curN <= 0) { return; }

            int64_t offsetC = mPos * n + nPos;

            if ASCEND_IS_AIC {
                if (enableCVSync) {
                    countId = count / COUNT_ID_MAX % COUNT_FLAG;
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIV_SYNC_AIC_FLAG + countId);
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIV_SYNC_AIC_FLAG + countId + FLAG_ID_MAX);
                }

                auto gmA = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmPtr), layoutA);
                auto gmB = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmPtr), layoutB);
                auto layoutC = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(
                    params.problemShape.m, params.problemShape.n);
                // gmC 仅用于 BlockMmad 签名兼容——FixpOpti 中 BlockMmad 写 UB
                // 而非 GM，实际 GM 写回由 Epilogue 完成
                auto gmC = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmPtr), layoutC);

                auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(mPos, kPos),
                    AscendC::Te::MakeShape(curM, params.problemShape.k));
                auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(kPos, nPos),
                    AscendC::Te::MakeShape(params.problemShape.k, curN));
                auto gmBlockC = gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos),
                    AscendC::Te::MakeShape(curM, curN));

                blockMmadOp(gmBlockA, gmBlockB, gmBlockC, singleShape);

                enableCVSync = true;
                count++;
                countId = count / COUNT_ID_MAX % COUNT_FLAG;
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                    AIC_SYNC_AIV_FLAG + countId);
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                    AIC_SYNC_AIV_FLAG + countId + FLAG_ID_MAX);
            }

            if ASCEND_IS_AIV {
                count++;
                countId = count / COUNT_ID_MAX % COUNT_FLAG;
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(
                    AIC_SYNC_AIV_FLAG + countId);
                epilogueOp({curM, curN, 1, 1}, offsetC, (AIV_SYNC_AIC_FLAG + countId));
            }
        }

        if ASCEND_IS_AIC {
            if (enableCVSync) {
                countId = count / COUNT_ID_MAX % COUNT_FLAG;
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                    AIV_SYNC_AIC_FLAG + countId);
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                    AIV_SYNC_AIC_FLAG + countId + FLAG_ID_MAX);
            }
        }
    }
};

} // namespace Kernel

#endif // MATMUL_KERNEL_FUSED_H
