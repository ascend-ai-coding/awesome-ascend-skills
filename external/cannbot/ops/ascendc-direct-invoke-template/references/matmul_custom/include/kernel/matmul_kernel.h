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
 * \file matmul_kernel.h
 * \brief Top-level kernel template: builds GM tensors and drives BlockScheduler + BlockMmad.
 */

#ifndef MATMUL_KERNEL_H
#define MATMUL_KERNEL_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "kernel_utils/common_utils.h"
#include "kernel_utils/layout_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "include/tensor.h"

#include "../block/matmul_block_mmad.h"
#include "../block/matmul_block_scheduler.h"
#include "../utils/matmul_constant.h"

// ============================================================================
// Matmul Kernel —— Kernel 层模板：
//   1. 把 GM 原始指针包装成带 Layout 的 Tensor（NDLayout 行主序 / DNLayout 列主序）
//   2. 构造 BlockScheduler，循环取出当前 AIC 要处理的 (m, n) block 坐标
//   3. 按坐标切分 gmA/gmB/gmC，调用 BlockMmad 完成该 block 的累加+写回
//
// [MODIFY] 新算子如果需要额外的输入/输出（bias、scale…）：
//   - 在 BlockMmad::Params 里加地址字段；
//   - 在本文件 `ResetGmAddr` 与 `Process` 里构造对应的 GM Tensor 视图；
//   - 将新 Tensor 通过 `mmadOp_(...)` 传递给 BlockMmad。
// ============================================================================

namespace Kernel {

#define MATMUL_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockScheduler>
#define MATMUL_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler

using namespace AscendC;

MATMUL_KERNEL_CLASS_TEM_PARAMS
class MatmulKernel {
public:
    __aicore__ inline MatmulKernel()
    {}
    __aicore__ inline ~MatmulKernel()
    {}

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

    // LayoutA == ColumnMajor => host 落盘顺序 (K, M)，device 逻辑 (M, K)，用 DNLayoutFormat。
    // LayoutA == RowMajor    => host 落盘 (M, K)，device 直接按 ND 视图。
    using MakeLayoutA = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<LayoutA, layout::ColumnMajor>,
        AscendC::Te::DNLayoutFormat<AType>,
        AscendC::Te::NDLayoutFormat<AType>>;
    // LayoutB == ColumnMajor => host (N, K)，device 逻辑 (K, N)，用 DNLayoutFormat。
    // LayoutB == RowMajor    => host (K, N)，device 直接按 ND 视图。
    using MakeLayoutB = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<LayoutB, layout::ColumnMajor>,
        AscendC::Te::DNLayoutFormat<BType>,
        AscendC::Te::NDLayoutFormat<BType>>;

    struct MatmulTiling {
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
        MatmulTiling qbmmParams;
    };

public:
    __aicore__ inline void operator()(const Params& params);

private:
    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline void Process(const Params& params, BlockSchedulerOp& bs);
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }

private:
    BlockMmad mmadOp_;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
};

MATMUL_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void MatmulKernel<MATMUL_KERNEL_FUN_TEM_PARAMS>::operator()(const Params& params)
{
    // Matmul 只跑在 AIC 上；混合核启动时 AIV 直接返回。
    if ASCEND_IS_AIV {
        return;
    }

    ResetGmAddr(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    TupleShape problemShape_ = ToShapeTuple(params.problemShape);
    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0cPingPong = (params.qbmmParams.dbL0C > 1);
    mmadOp_.Init(problemShape_, l0TileShape, params.l1Params, enableL0cPingPong);
    Process(params, bs);
}

MATMUL_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void MatmulKernel<MATMUL_KERNEL_FUN_TEM_PARAMS>::ResetGmAddr(const Params& params)
{
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
}

MATMUL_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void MatmulKernel<MATMUL_KERNEL_FUN_TEM_PARAMS>::Process(
    const Params& params, BlockSchedulerOp& bs)
{
    auto layoutA = MakeLayoutA{}(params.problemShape.m, params.problemShape.k);
    auto layoutB = MakeLayoutB{}(params.problemShape.k, params.problemShape.n);
    auto layoutC = AscendC::Te::MakeNDLayout<CType>(params.problemShape.m, params.problemShape.n);

    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(aGmAddr_), layoutA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(bGmAddr_), layoutB);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(cGmAddr_), layoutC);

    BlockCoord blockIdx;
    constexpr int64_t kPos = 0L;
    while (bs.GetTileIdx(blockIdx)) {
        int64_t mPos = Get<MNK_M>(blockIdx);
        int64_t nPos = Get<MNK_N>(blockIdx);
        BlockShape singleShape = bs.GetBlockShape(blockIdx);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            return;
        }

        auto gmBlockA =
            gmA(AscendC::Te::MakeCoord(mPos, kPos),
                AscendC::Te::MakeShape(Get<MNK_M>(singleShape), params.problemShape.k));
        auto gmBlockB =
            gmB(AscendC::Te::MakeCoord(kPos, nPos),
                AscendC::Te::MakeShape(params.problemShape.k, Get<MNK_N>(singleShape)));
        auto gmBlockC =
            gmC(AscendC::Te::MakeCoord(mPos, nPos),
                AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));

        mmadOp_(gmBlockA, gmBlockB, gmBlockC, singleShape);
    }
}

} // namespace Kernel

#undef MATMUL_KERNEL_CLASS_TEM_PARAMS
#undef MATMUL_KERNEL_FUN_TEM_PARAMS

#endif // MATMUL_KERNEL_H
