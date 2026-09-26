/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"

using AscendC::Coord;
using AscendC::SetMMLayoutTransform;
using AscendC::Shape;

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_matmul_basic.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"

#include "matmul_blaze_example_tiling_data.h"

__global__ __aicore__ void matmul_blaze_example(
    GM_ADDR a, GM_ADDR b, GM_ADDR c,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(MatmulBlazeExampleTilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulBlazeExampleTilingData, tilingData, tiling);

    using AType = bfloat16_t;
    using BType = bfloat16_t;
    using CType = bfloat16_t;
    using BiasType = float;

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::NDExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;

    constexpr uint64_t FUSED_OP_TYPE = 0;
    constexpr bool IS_ND_FORMAT = true;
    constexpr bool IS_FP32 = false;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<0, FUSED_OP_TYPE>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<
        ProblemShape, 0, IS_FP32, IS_ND_FORMAT>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<
        ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename KernelImpl::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;

    ProblemShape problemShape{
        static_cast<int64_t>(tilingData.m),
        static_cast<int64_t>(tilingData.n),
        static_cast<int64_t>(tilingData.k),
        1L};

    BlockMmadParams mmParams;
    mmParams.aGmAddr = a;
    mmParams.bGmAddr = b;
    mmParams.cGmAddr = c;
    mmParams.biasGmAddr = nullptr;
    mmParams.ml1 = tilingData.mL1;
    mmParams.nl1 = tilingData.nL1;
    mmParams.kl1 = tilingData.kL1;
    mmParams.ml0 = tilingData.baseM;
    mmParams.nl0 = tilingData.baseN;
    mmParams.kl0 = tilingData.baseK;
    mmParams.l1Stages = tilingData.l1BufferNum;
    mmParams.l0cStages = static_cast<uint16_t>(tilingData.l0cDB);

    BlockSchedulerParams schParams;
    schParams.mL1 = tilingData.mL1;
    schParams.nL1 = tilingData.nL1;
    schParams.kL1 = tilingData.kL1;
    schParams.baseM = tilingData.baseM;
    schParams.baseN = tilingData.baseN;
    schParams.baseK = tilingData.baseK;
    schParams.mTailCnt = tilingData.mTailCnt;
    schParams.nTailCnt = tilingData.nTailCnt;
    schParams.mBaseTailSplitCnt = tilingData.mBaseTailSplitCnt;
    schParams.nBaseTailSplitCnt = tilingData.nBaseTailSplitCnt;
    schParams.mTailMain = tilingData.mTailMain;
    schParams.nTailMain = tilingData.nTailMain;
    schParams.isHf32 = 0;
    schParams.l2CacheMode = Blaze::Gemm::L2_CACHE_DEFAULT;
    schParams.sliceM = 0;
    schParams.srcNdStride = 1;
    schParams.innerBatch = 1;

    BlockEpilogueParams epilogueParams;

    Params params;
    params.problemShape = problemShape;
    params.mmadParams = mmParams;
    params.epilogueParams = epilogueParams;
    params.schParams = schParams;

    KernelImpl kernel;
    kernel(params);
}
