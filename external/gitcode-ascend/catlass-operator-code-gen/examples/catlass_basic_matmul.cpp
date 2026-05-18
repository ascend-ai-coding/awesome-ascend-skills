/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/layout/layout.hpp"

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
namespace Catlass {
template <class InputType, class OutputType>
CATLASS_DEVICE void CatlassBasicMatmulTemplate(GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC)
{
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;

    using AType = Gemm::GemmType<InputType, LayoutA>;
    using BType = Gemm::GemmType<InputType, LayoutB>;
    using CType = Gemm::GemmType<OutputType, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

    LayoutA layoutA{problemShape.m(), problemShape.k()};
    LayoutB layoutB{problemShape.k(), problemShape.n()};
    LayoutC layoutC{problemShape.m(), problemShape.n()};
    typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};

    MatmulKernel kernel;
    kernel(params);
}
} // namespace Catlass

extern "C" __global__ __aicore__ void
catlass_basic_matmul(GM_ADDR self, GM_ADDR mat2, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    // user kernel impl
    Catlass::GemmCoord problemShape{tiling_data.m, tiling_data.n, tiling_data.k};
    if (TILING_KEY_IS(1)) {
        Catlass::CatlassBasicMatmulTemplate<half, half>(problemShape, self, mat2, out);
    } else if (TILING_KEY_IS(2)) {
        Catlass::CatlassBasicMatmulTemplate<int8_t, int32_t>(problemShape, self, mat2, out);
    }
}