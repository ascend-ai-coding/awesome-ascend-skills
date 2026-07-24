/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// AIC/AIV 融合 Matmul Kernel —— CV 同步 + Epilogue 支持
// ----------------------------------------------------------------------------
// 本 kernel 编排 AIC（Cube）和 AIV（Vector）核心：
//   - AIC：运行 BlockMmad（L1→L0→MMAD→L0C→UB），通过 CrossCoreSetFlag 通知 AIV
//   - AIV：等待 AIC 信号，运行 Epilogue（UB→计算→GM），通过 CrossCoreSetFlag 回复 AIC
//
// CV 同步和 Epilogue 分发逻辑是通用的，适用于任何将 L0C 结果写入 UB 的 BlockMmad。
// 适配不同 BlockMmad 时，根据该 BlockMmad 的接口定义修改 [MODIFY] 标注处：
//   1. Init 参数（ProblemShape 维度、L1Params 字段、附加参数）
//   2. operator() 参数（GM Tensor 个数、Scale/Bias Tensor）
//   3. Params 结构体字段（与 BlockMmad::Params 对齐）
//   4. TensorC 内存位置（UB 或 GM，取决于 BlockMmad 的输出路径）
// ============================================================================

#ifndef MATMUL_KERNEL_FUSED_H
#define MATMUL_KERNEL_FUSED_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
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
    using L0CType = typename BlockMmad::L0CType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;
    using EpilogueParams = typename Epilogue::Params;
    using ProblemShapeType = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    // [MODIFY] C0 must match dtype for NZ/ZN fractal layouts (int8/fp8: C0=32, bf16/fp16: C0=16)
    static constexpr uint64_t A_C0 = 32 / sizeof(AType);
    static constexpr uint64_t B_C0 = 32 / sizeof(BType);
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<A_C0>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<B_C0>>;

    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint8_t dbL0C;
    };

    // [MODIFY] Params 字段需与所选 BlockMmad::Params 对齐。
    // 当前适用于 matmul_block_mmad.h（非 MX）。
    // 若使用其他 BlockMmad（如 MX 量化），根据其 Params 定义调整字段。
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
            // [MODIFY] 根据所选 BlockMmad 的 Init 接口定义调整参数。
            // 当前适用于 matmul_block_mmad.h：Init(TupleShape 3D, BlockShape, L1Params, bool)。
            // 若使用其他 BlockMmad，根据其 Init 签名调整（如 ProblemShape 维度、
            // L1Params 字段、附加参数 isBias/dbL0C/splitKNum 等）。
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
                auto curMPad = (curM + 1L) & ~1L;
                constexpr int64_t UB_N_ALIGN_ELEM = 32L / static_cast<int64_t>(sizeof(L0CType));
                auto curNUbAlign = ((curN + UB_N_ALIGN_ELEM - 1L) / UB_N_ALIGN_ELEM) * UB_N_ALIGN_ELEM;
                auto layoutUB = AscendC::Te::MakeFrameLayout<
                    AscendC::Te::NDExtLayoutPtn, AscendC::Std::Int<BlockMmad::BLOCK_CUBE_L0C>>(
                        curMPad, curNUbAlign);
                // [MODIFY] fused kernel 必须传 UB Tensor 以触发 CopyL0C2UB。
                // 若切换到其他 BlockMmad（如 MX），根据其 Location 分支调整。
                auto ubBlockC = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, L0CType>(0), layoutUB);

                auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(mPos, kPos),
                    AscendC::Te::MakeShape(curM, params.problemShape.k));
                auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(kPos, nPos),
                    AscendC::Te::MakeShape(params.problemShape.k, curN));

                // [MODIFY] 根据所选 BlockMmad 的 operator() 接口定义调整参数。
                // 当前适用于 matmul_block_mmad.h：operator()(gmA, gmB, ubBlockC, singleShape)。
                // 若使用其他 BlockMmad，根据其 operator() 签名调整（如 MX BlockMmad
                // 需要额外的 gmScaleA, gmScaleB, gmBias 等参数）。
                blockMmadOp(gmBlockA, gmBlockB, ubBlockC, singleShape);

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
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(
                    AIV_SYNC_AIC_FLAG + countId);
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
