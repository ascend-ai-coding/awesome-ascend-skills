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

#ifndef GROUP_MATMUL_KERNEL_CV1_V2_H
#define GROUP_MATMUL_KERNEL_CV1_V2_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "tensor_api/tensor.h"

#include "blaze/gemm/kernel/kernel_universal.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

constexpr AscendC::SyncAllConfig GROUP_MATMUL_CV1_V2_SYNC_ALL_CONFIG = {PIPE_ALL, PIPE_ALL};

template <class BlockEpilogue>
struct IsCv1V2EpiloguePipeline : public AscendC::Std::false_type {};

template <class BlockEpilogueV1, class BlockEpilogueV2>
struct IsCv1V2EpiloguePipeline<AscendC::Std::tuple<BlockEpilogueV1, BlockEpilogueV2>>
    : public AscendC::Std::true_type {};

// Generic GroupMatmul C+V1 + V2 composition. CV1 is any already-selected
// GemmUniversal specialization whose output is a full-row GM workspace. This
// layer owns only the global phase handoff, full-row AIV repartition and V2.
// It owns no Matmul formula, group traversal, branch view or activation logic.
template <
    class ProblemShape_, class BlockMmad_, class BlockEpilogueV1_, class BlockEpilogueV2_, class BlockScheduler_,
    typename Enable_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, AscendC::Std::tuple<BlockEpilogueV1_, BlockEpilogueV2_>, BlockScheduler_, Enable_> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogueV1 = BlockEpilogueV1_;
    using BlockEpilogueV2 = BlockEpilogueV2_;
    using BlockEpilogue = AscendC::Std::tuple<BlockEpilogueV1, BlockEpilogueV2>;
    using BlockScheduler = BlockScheduler_;
    using Cv1Kernel = GemmUniversal<ProblemShape, BlockMmad, BlockEpilogueV1, BlockScheduler>;
    using V2InputType = typename BlockEpilogueV2::InputType;
    using V2OutputType = typename BlockEpilogueV2::OutputType;
    using V2AuxOutputType = typename BlockEpilogueV2::AuxOutputType;

    struct Params {
        typename Cv1Kernel::Params cv1Params{};
        typename BlockEpilogueV2::Params epilogueV2Params{};
        int64_t realM{0};
        int64_t workspaceRowPitch{0};
        GM_ADDR epilogueV2OutputGmAddr{nullptr};
        GM_ADDR epilogueV2AuxOutputGmAddr{nullptr};
    };

    __aicore__ inline void operator()(const Params& params, __gm__ V2InputType* workspaceGmAddr)
    {
        {
            Cv1Kernel cv1Kernel;
            cv1Kernel(params.cv1Params, workspaceGmAddr);
        }

        AscendC::SyncAll<true, GROUP_MATMUL_CV1_V2_SYNC_ALL_CONFIG>();
        if ASCEND_IS_AIV {
            RunV2(params, workspaceGmAddr);
        }
    }

private:
    template <class ElementType>
    __aicore__ inline static auto MakeRowMajorLayout(int64_t rows, int64_t columns, int64_t rowPitch)
    {
        auto shape = AscendC::Te::MakeShape(
            AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, rows),
            AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, columns));
        auto stride = AscendC::Te::MakeStride(
            AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, rowPitch),
            AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
        return AscendC::Te::MakePatternLayout<
            AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<ElementType>>(shape, stride);
    }

    __aicore__ inline static void RunV2(const Params& params, __gm__ V2InputType* workspaceGmAddr)
    {
        const int64_t columns = params.epilogueV2Params.n;
        const uint64_t workers =
            static_cast<uint64_t>(AscendC::GetBlockNum()) * static_cast<uint64_t>(AscendC::GetTaskRation());
        if (workspaceGmAddr == nullptr || params.epilogueV2OutputGmAddr == nullptr ||
            params.epilogueV2AuxOutputGmAddr == nullptr || params.realM <= 0 || columns <= 0 ||
            params.workspaceRowPitch < columns || workers == 0) {
            return;
        }

        const uint64_t rank = static_cast<uint64_t>(AscendC::GetBlockIdx());
        if (rank >= workers) {
            return;
        }
        const uint64_t rows = static_cast<uint64_t>(params.realM);
        const int64_t rowStart = static_cast<int64_t>(rows * rank / workers);
        const int64_t rowEnd = static_cast<int64_t>(rows * (rank + 1U) / workers);
        const uint32_t rowCount = static_cast<uint32_t>(rowEnd - rowStart);
        if (rowCount == 0) {
            return;
        }

        auto workspace = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(workspaceGmAddr),
            MakeRowMajorLayout<V2InputType>(params.realM, columns, params.workspaceRowPitch));
        auto output = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ V2OutputType*>(params.epilogueV2OutputGmAddr)),
            MakeRowMajorLayout<V2OutputType>(params.realM, columns, columns));
        auto auxiliary = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ V2AuxOutputType*>(params.epilogueV2AuxOutputGmAddr)),
            MakeRowMajorLayout<V2AuxOutputType>(1, params.realM, params.realM));

        BlockEpilogueV2 epilogueV2;
        epilogueV2.Init(params.epilogueV2Params);
        epilogueV2(
            workspace.Slice(
                AscendC::Te::MakeCoord(rowStart, int64_t{0}), AscendC::Te::MakeShape(rowEnd - rowStart, columns)),
            output.Slice(
                AscendC::Te::MakeCoord(rowStart, int64_t{0}), AscendC::Te::MakeShape(rowEnd - rowStart, columns)),
            auxiliary.Slice(
                AscendC::Te::MakeCoord(int64_t{0}, rowStart),
                AscendC::Te::MakeShape(int64_t{1}, rowEnd - rowStart)),
            rowCount);
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze

#endif // GROUP_MATMUL_KERNEL_CV1_V2_H
