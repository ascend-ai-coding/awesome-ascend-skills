/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_gelu_example_arch35.cpp
 * \brief quant_matmul_gelu_example 算子 Kernel 入口（arch35 架构，MIX：AIC matmul + AIV eltwise 融合）
 *
 * [MODIFY] 本文件是 AIC+AIV 融合算子 Kernel 入口的骨架。
 * 开发新融合算子时，按以下步骤修改：
 *   1. 修改函数名 quant_matmul_gelu_example → 您的算子名
 *   2. 修改 GM 参数列表（inputs + output + workspace + tiling）
 *   3. 修改 AType/BType/CType（矩阵乘输入/输出类型）
 *   4. 修改 layoutA/layoutB/layoutC（NDExt=行主序/DNExt=列主序/NZ=fractal）
 *   5. 修改 DispatchPolicy（NO_FULL_LOAD_MODE）
 *   6. 修改 EpilogueOp（核心定制点 — AIV 侧后处理逻辑）
 *   7. 修改 Params 构造（tiling 字段映射 + Epilogue 参数）
 *   8. 修改 REGISTER_TILING_DEFAULT / GET_TILING_DATA_WITH_STRUCT 的 TilingData 类型
 *
 *
 * 融合算子结构：
 *   - AIC: BlockMmad 做类型A × 类型B → L0CDataType matmul，结果经 CopyL0C2UB 落到 UB
 *   - AIV: EpilogueOp 读取 UB 的 L0CDataType 数据，做 eltwise 后处理，写回 GM
 *   - AIC/AIV 通过 MatmulKernelFused 内部 CrossCoreSetFlag/WaitFlag 同步
 *
 * NOTE: registry-invoke (opbuild) 下不使用 __mix__/__cube__ 属性；
 *       opbuild 默认对 AIC+AIV 双侧编译，MIX 行为由 MatmulKernelFused 内的
 *       `if ASCEND_IS_AIC` / `if ASCEND_IS_AIV` 分支天然形成。
 */

#include "kernel_operator.h"
#include "arch35/include/block/block_scheduler_policy.h"
#include "arch35/common/kernel_utils/layout_utils.h"
#include "arch35/common/kernel_utils/common_utils.h"
#include "arch35/include/kernel/matmul_kernel_fused.h"
#include "arch35/include/epilogue/scale_gelu_epilogue_regbase.h"  // [MODIFY] 替换为您选择的 Epilogue
#include "arch35/quant_matmul_gelu_example_tiling_data.h"

// ============================================================================
// Kernel 入口 —— MIX (AIC matmul + AIV eltwise)，单一 TilingKey
// [MODIFY] GM 参数顺序：inputs(...) + output(y) + workspace + tiling
// ============================================================================
__global__ __aicore__ void quant_matmul_gelu_example(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // ========================================================================
    // [MODIFY] 类型别名 — 根据您的算子输入 dtype 修改
    //   AType/BType: 矩阵乘输入类型
    //   CType: 仅用于 BlockMmad gmC 签名，实际写回由 Epilogue 完成
    // ========================================================================
    using AType = int8_t;
    using BType = int8_t;
    using CType = bfloat16_t;

    // ========================================================================
    // [MODIFY] Layout 配置 — 根据您的矩阵存储格式修改
    //   NDExtLayoutPtn = 行主序（不转置）  — A=(M,K) 行主序
    //   DNExtLayoutPtn = 列主序（转置语义） — B=(N,K) 行主序=列主序（物理不转置）
    //   NZLayoutPtn    = fractal/NZ 格式    — B 预排为 (K/16, N/16, 16, 16) fractal 块
    //
    //   常见配置:
    //     A 不转置 + B 转置(行主序=列主序): layoutA=NDExt, layoutB=DNExt  ← 当前
    //     A 不转置 + B 不转置:              layoutA=NDExt, layoutB=NDExt
    //     A 不转置 + B NZ fractal:           layoutA=NDExt, layoutB=NZ
    //     A 转置 + B 不转置:                 layoutA=DNExt, layoutB=NDExt
    // ========================================================================
    using layoutA = AscendC::Te::NDExtLayoutPtn;
    using layoutB = AscendC::Te::DNExtLayoutPtn;
    using layoutC = AscendC::Te::NDExtLayoutPtn;

    // ========================================================================
    // [MODIFY] DispatchPolicy 和 BlockScheduler
    //   NO_FULL_LOAD_MODE: A/B 均从 GM 流式加载到 L1
    // ========================================================================
    using BlockScheduler = MatmulSwatScheduler<NO_FULL_LOAD_MODE>;
    using DispatchPolicy = MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>;
    using ProblemShape = MatmulShape;

    using BlockMmad = Block::BlockMmad<DispatchPolicy, AType, layoutA, BType, layoutB, CType, layoutC>;

    // ========================================================================
    // [MODIFY] EpilogueOp — 核心定制点！选择您需要的 AIV 后处理 Epilogue 类
    //   当前 ScaleGeluEpilogueRegBase（见 op_kernel/arch35/include/epilogue/）:
    //     int32 → scale*pertoken+bias → gelu_tanh → bf16（使用 RegBase 指令）
    //   新增自定义 Epilogue 时，复制本文件作为起点，满足三接口合约:
    //     using Params, void Init(Params,l1M,l1N,problemShape), void operator()(BlockShape,gmOffset,flagId)
    // ========================================================================
    using EpilogueOp = ScaleGeluEpilogueRegBase;
    using MatmulKernelImpl = Kernel::MatmulKernelFused<ProblemShape, BlockMmad, BlockScheduler, EpilogueOp>;

    using Params = typename MatmulKernelImpl::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename MatmulKernelImpl::L1Params;
    using BlockSchedulerParams = typename MatmulKernelImpl::BlockSchedulerParams;
    using MatmulTiling = typename MatmulKernelImpl::QBMMTiling;
    using EpilogueParams = typename MatmulKernelImpl::EpilogueParams;

    // [MODIFY] TilingData 类型名
    REGISTER_TILING_DEFAULT(QuantMatmulGeluExampleTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantMatmulGeluExampleTilingData, tilingData, tiling);

    ProblemShape problemShape{tilingData.m, tilingData.n, tilingData.k, 1L};
    BlockMmadParams mmadParams{x1, x2, y};
    L1Params l1Params{static_cast<uint64_t>(tilingData.kL1)};
    BlockSchedulerParams schedulerParams{
        tilingData.baseM,
        tilingData.baseN,
        tilingData.mTailCnt,
        tilingData.nTailCnt,
        tilingData.mBaseTailSplitCnt,
        tilingData.nBaseTailSplitCnt,
        tilingData.mTailMain,
        tilingData.nTailMain};
    MatmulTiling qbmmParams{
        tilingData.baseM,
        tilingData.baseN,
        tilingData.baseK,
        tilingData.l0cDB};
    // [MODIFY] Epilogue 参数 — 与您的 EpilogueOp::Params 构造函数匹配
    //   ScaleGeluEpilogueRegBase::Params = {scale, pertokenScale, bias, output}
    EpilogueParams epilogueParams{scale, pertokenScale, bias, y};

    Params params{problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, epilogueParams};

    MatmulKernelImpl kernel;
    kernel(params);
}
