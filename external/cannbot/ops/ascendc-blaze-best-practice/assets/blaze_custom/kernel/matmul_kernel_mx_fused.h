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
// AIC/AIV 融合 MX 量化 Matmul Kernel —— CV 同步 + Epilogue 支持
// ----------------------------------------------------------------------------
// 本 kernel 编排 AIC（Cube）和 AIV（Vector）核心，使用 blaze 库的 MX BlockMmad：
//   - AIC：运行 BlockMmad（L1→L0→MMAD→L0C→UB），通过 CrossCoreSetFlag 通知 AIV
//   - AIV：等待 AIC 信号，运行 Epilogue（UB→计算→GM），通过 CrossCoreSetFlag 回复 AIC
//
// 与 matmul_kernel_fused.h 的区别：
//   1. MX block 接口：operator()(gmA, gmB, gmScaleA, gmScaleB, gmBias, ubBlockC, singleShape)
//   2. Init 接受 4D ProblemShape，附加 isBias / dbL0C 参数
//   3. BlockScheduler 使用 blaze 库的 BlockSchedulerQuantBatchMatmulV3（直接传入，无 Selector）
//   4. C0_SIZE 由 IsFp4<AType> 决定（FP4: 64, FP8: 32）
//   5. L0CType 固定为 float（MX block 内部累加精度）
//   6. Scale Tensor 需要额外的 layout 和 slice
//   7. Kernel 传入 UB Tensor 触发 Blaze MX BlockMmad 的 L0C->UB 路径；当前该路径使用
//      CopyL0C2UBSplitMTrait / DUAL_DST_SPLIT_M，epilogue 必须按 GetTaskRation()/GetSubBlockIdx()
//      消费对应的 M 分片
//   8. MX block 内部使用独立的 L1 流水线事件标志（INPUT_BUFFER_FLAG / SCALE_BUFFER_FLAG /
//      M_MTE1_FLAG），与 CrossCore 同步标志互不干扰
//   9. 需要 blaze 库头文件在 include path 中（third_party/）
// ============================================================================

#ifndef MATMUL_KERNEL_MX_FUSED_H
#define MATMUL_KERNEL_MX_FUSED_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "tensor_api/tensor.h"

#include "../epilogue/cv_sync_constants.h"

namespace Kernel {

template <class ProblemShape, class BlockMmad, class BlockScheduler, class Epilogue>
class MxMatmulKernelFused {
public:
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = CvSync::MODE;
    static constexpr int16_t AIV_SYNC_AIC_FLAG = CvSync::AIV_TO_AIC_FLAG;
    static constexpr int16_t AIC_SYNC_AIV_FLAG = CvSync::AIC_TO_AIV_FLAG;
    static constexpr int16_t FLAG_ID_MAX = 16;
    static constexpr int16_t COUNT_ID_MAX = CvSync::COUNT_ID_MAX;
    static constexpr int16_t COUNT_FLAG = CvSync::COUNT_FLAG;

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;
    static constexpr bool weightNz = BlockMmad::weightNz;

    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using L0CType = float;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;

    static constexpr int64_t C0_SIZE = Blaze::Gemm::IsFp4<AType>() ? Blaze::Gemm::C0_SIZE_B4 : Blaze::Gemm::C0_SIZE_B8;
    static constexpr int32_t SCALE_C0 = 2;

    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockSchedulerParams = typename BlockScheduler::Params;
    using EpilogueParams = typename Epilogue::Params;

    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        typename BlockMmad::Params mmadParams;
        typename BlockMmad::L1Params l1Params;
        typename BlockScheduler::Params schParams;
        QBMMTiling qbmmParams;
        typename Epilogue::Params epilogueParams;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        BlockScheduler bs(params.problemShape, params.schParams);

        Epilogue epilogueOp;
        epilogueOp.Init(params.epilogueParams, params.qbmmParams.baseM, params.qbmmParams.baseN, params.problemShape);

        BlockMmad blockMmadOp;
        if ASCEND_IS_AIC {
            BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
            bool isBias = params.qbmmParams.isBias == 1;
            bool enableL0cPingPong = (params.qbmmParams.dbL0C > 1);
            blockMmadOp.Init(params.problemShape, l0TileShape, params.l1Params, isBias, enableL0cPingPong);
        }

        const auto m = AscendC::Te::Get<Blaze::Gemm::MNK_M>(params.problemShape);
        const auto n = AscendC::Te::Get<Blaze::Gemm::MNK_N>(params.problemShape);
        const auto k = AscendC::Te::Get<Blaze::Gemm::MNK_K>(params.problemShape);
        const auto scaleKLen =
            Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(Blaze::Gemm::MXFP_DIVISOR_SIZE)) *
            Blaze::Gemm::MXFP_MULTI_BASE_SIZE;

        auto layoutA = MakeLayoutA{}(m, k);
        auto layoutB = MakeLayoutB{}(k, n);
        auto layoutScaleA = MakeLayoutScaleA{}(m, scaleKLen);
        auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, n);
        auto layoutBias = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1L, n);

        __gm__ AType* aGmPtr = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        __gm__ BType* bGmPtr = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        __gm__ AscendC::fp8_e8m0_t* scaleAGmPtr =
            reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
        __gm__ AscendC::fp8_e8m0_t* scaleBGmPtr =
            reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
        __gm__ BiasType* biasGmPtr = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);

        auto gmA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmPtr), layoutA);
        auto gmB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmPtr), layoutB);
        auto gmScaleA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleAGmPtr), layoutScaleA);
        auto gmScaleB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBGmPtr), layoutScaleB);
        auto gmBias = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmPtr), layoutBias);

        const auto mTailTile = params.schParams.mTailTile;
        const auto nTailTile = params.schParams.nTailTile;
        if ((bs.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= AscendC::GetBlockNum()) {
            bs.UpdateTailTile(mTailTile, nTailTile);
        }

        int64_t count = 0;
        int64_t countId = 0;
        bool enableCVSync = false;
        constexpr int64_t kPos = 0L;
        int64_t mPos = 0L;
        int64_t nPos = 0L;

        BlockCoord blockIdx;
        while (bs.GetTileIdx(blockIdx)) {
            BlockShape singleShape = bs.template GetBlockShape<
                Blaze::Gemm::QuantMode::MX_PERGROUP_MODE,
                Blaze::Gemm::QuantMode::MX_PERGROUP_MODE,
                weightNz>(blockIdx);
            int64_t curM = AscendC::Te::Get<Blaze::Gemm::IDX_M_TILEIDX>(singleShape);
            int64_t curN = AscendC::Te::Get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape);
            if (curM <= 0 || curN <= 0) { return; }

            bs.GetTileCoord(blockIdx, mPos, nPos);

            if ASCEND_IS_AIC {
                if (enableCVSync) {
                    countId = count / COUNT_ID_MAX % COUNT_FLAG;
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIV_SYNC_AIC_FLAG + countId);
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIV_SYNC_AIC_FLAG + countId + FLAG_ID_MAX);
                }

                auto curMPad = (curM + 1L) & ~1L;
                constexpr int64_t UB_N_ALIGN_ELEM = 32L / static_cast<int64_t>(sizeof(L0CType));
                auto curNUbAlign =
                    ((curN + UB_N_ALIGN_ELEM - 1L) / UB_N_ALIGN_ELEM) * UB_N_ALIGN_ELEM;
                auto layoutUB = AscendC::Te::MakeFrameLayout<
                    AscendC::Te::NDExtLayoutPtn, AscendC::Std::Int<16>>(curMPad, curNUbAlign);
                auto ubBlockC = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, L0CType>(0), layoutUB);

                auto gmBlockA = gmA.Slice(
                    AscendC::Te::MakeCoord(mPos, kPos),
                    AscendC::Te::MakeShape(curM, k));
                auto gmBlockB = gmB.Slice(
                    AscendC::Te::MakeCoord(kPos, nPos),
                    AscendC::Te::MakeShape(k, curN));
                auto gmBlockScaleA = gmScaleA.Slice(
                    AscendC::Te::MakeCoord(mPos, kPos),
                    AscendC::Te::MakeShape(curM, scaleKLen));
                auto gmBlockScaleB = gmScaleB.Slice(
                    AscendC::Te::MakeCoord(kPos, nPos),
                    AscendC::Te::MakeShape(scaleKLen, curN));
                auto gmBlockBias = gmBias.Slice(
                    AscendC::Te::MakeCoord(0L, nPos),
                    AscendC::Te::MakeShape(1L, curN));

                blockMmadOp(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB,
                    gmBlockBias, ubBlockC, singleShape);

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
                int64_t offsetC = mPos * n + nPos;
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

#endif // MATMUL_KERNEL_MX_FUSED_H
