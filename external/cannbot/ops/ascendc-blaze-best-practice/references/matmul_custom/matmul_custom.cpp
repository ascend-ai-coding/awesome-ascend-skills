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
// Matmul Kernel 直调样例 —— BF16 in / BF16 out（dav-3510）
// ----------------------------------------------------------------------------
// 支持 NO_FULL_LOAD_MODE（通用 SWAT）与 A_FULL_LOAD_MODE（A 全载），
// 支持 A/B 的 ND/NZ 输入格式和 transA/transB 全部 16 种组合。
//
// 创建新算子时按下面 [MODIFY] 标记修改。搜索 `[MODIFY]` 即可定位每个改点；
// 按重要性分三档（必改 / 常改 / 选改）：
//
// === 必改（任何新算子都要动）===
//   [MODIFY] N1  函数名 / CMake 目标名 / run.sh OP_NAME 三处保持一致
//   [MODIFY] N2  AType / BType / CType（搭配 sizeA/sizeB/sizeC 字节数 +
//                matmul_tiling_constant.h::DATA_SIZE_FP16）
//   [MODIFY] N3  scripts/gen_data.py + verify_result.py 的 dtype / golden / 容差
//
// === 常改（按算子需求二选一）===
//   [MODIFY] C1  layoutA / layoutB 与 transA/transB
//   [MODIFY] C2  TilingData 增删字段（bias/scale 等额外输入）—— 见 `matmul_basic.md` §2.2
//
// === 选改（高级变种才需要）===
//   [MODIFY] A1  切到 A 全载（mode=a_full）
//   [MODIFY] A2  L1_BUFFER_NUM = 4 等更深流水（需同步 dispatch policy）
// ============================================================================

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "acl/acl.h"
#include "kernel_operator.h"

#include "block/block_scheduler_policy.h"
#include "host_utils/acl_utils.h"
#include "host_utils/common_utils.h"
#include "host_utils/io_utils.h"
#include "kernel_utils/layout_utils.h"
#include "kernel/matmul_kernel.h"
#include "tiling/matmul_tiling.h"
#include "tiling/matmul_tiling_data.h"

// ---------------- NO_FULL_LOAD Kernel 入口 ----------------
// [MODIFY N1] 函数名需与 CMake 目标名、run.sh 中 OP_NAME 保持一致。
// 模板参数 LayoutA / LayoutB 是 tensor_api 的 layout pattern：
//   - NDExtLayoutPtn → 行主序   - DNExtLayoutPtn → 列主序
//   - NZLayoutPtn    → NZ 分形  - ZNLayoutPtn    → ZN 分形
template <class LayoutA, class LayoutB>
__global__ __aicore__ __cube__ void matmul_custom(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dC,
    const MatmulTilingData tilingData)
{
    // [MODIFY N2] 输入/输出 dtype。
    using AType = bfloat16_t;
    using BType = bfloat16_t;
    using CType = bfloat16_t;

    using layoutA = LayoutA;
    using layoutC = AscendC::Te::NDExtLayoutPtn;

    using BlockScheduler = MatmulSwatScheduler<NO_FULL_LOAD_MODE>;
    using DispatchPolicy = MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>;
    using ProblemShape = MatmulShape;

    using BlockMmad = Block::BlockMmad<DispatchPolicy, AType, layoutA, BType, LayoutB, CType, layoutC>;
    using MatmulKernelImpl = Kernel::MatmulKernel<ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename MatmulKernelImpl::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename MatmulKernelImpl::L1Params;
    using BlockSchedulerParams = typename MatmulKernelImpl::BlockSchedulerParams;
    using MatmulTiling = typename MatmulKernelImpl::MatmulTiling;

    ProblemShape problemShape{tilingData.m, tilingData.n, tilingData.k, 1L};
    BlockMmadParams mmadParams{dA, dB, dC};
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
    Params params{problemShape, mmadParams, l1Params, schedulerParams, qbmmParams};
    MatmulKernelImpl kernel;
    kernel(params);
}

// ---------------- A_FULL_LOAD Kernel 入口 ----------------
// [MODIFY A1] A 全载模式：A 全部装入 L1 跨 N-tile 复用。
// 仅支持 transA=false, transB=false（由 if constexpr 在 dispatch 中保证）。
template <class LayoutA, class LayoutB>
__global__ __aicore__ __cube__ void matmul_a_full_load(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dC,
    const MatmulTilingData tilingData)
{
    using AType = bfloat16_t;
    using BType = bfloat16_t;
    using CType = bfloat16_t;

    using layoutA = LayoutA;
    using layoutC = AscendC::Te::NDExtLayoutPtn;

    using BlockScheduler = MatmulSwatScheduler<A_FULL_LOAD_MODE>;
    using DispatchPolicy = MatmulMultiBlockPolicy<A_FULL_LOAD_MODE>;
    using ProblemShape = MatmulShape;

    using BlockMmad = Block::BlockMmad<DispatchPolicy, AType, layoutA, BType, LayoutB, CType, layoutC>;
    using MatmulKernelImpl = Kernel::MatmulKernel<ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename MatmulKernelImpl::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename MatmulKernelImpl::L1Params;
    using BlockSchedulerParams = typename MatmulKernelImpl::BlockSchedulerParams;
    using MatmulTiling = typename MatmulKernelImpl::MatmulTiling;

    ProblemShape problemShape{tilingData.m, tilingData.n, tilingData.k, 1L};
    BlockMmadParams mmadParams{dA, dB, dC};
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
    Params params{problemShape, mmadParams, l1Params, schedulerParams, qbmmParams};
    MatmulKernelImpl kernel;
    kernel(params);
}

// NZ 物理排列 buffer 大小：(dim1/C0, dim0/16, 16, C0) 元素总数 × sizeof(dtype)
static uint64_t CalcNzSize(uint64_t dim0, uint64_t dim1, uint64_t c0)
{
    uint64_t dim0Blocks = (dim0 + 15) / 16;
    uint64_t dim1Blocks = (dim1 + c0 - 1) / c0;
    return dim1Blocks * dim0Blocks * 16 * c0 * sizeof(uint16_t);
}

// ---------------- Host 入口 ----------------
int main(int argc, char* argv[])
{
    uint64_t m = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    bool transA = false;
    bool transB = false;
    std::string aLayout = "nd";
    std::string bLayout = "nd";
    std::string mode = "no_full";

    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cerr << "Usage: " << argv[0] << " m k n [transA transB] [a_layout b_layout] [mode]" << std::endl;
        std::cerr << "  a_layout: nd (default) or nz" << std::endl;
        std::cerr << "  b_layout: nd (default) or nz" << std::endl;
        std::cerr << "  mode: no_full (default) or a_full" << std::endl;
        return 1;
    }
    if (argc < 4 || argc == 5 || argc > 9) {
        std::cerr << "ERROR: expected 3, 4, 6, 7, 8, or 9 arguments" << std::endl;
        return 1;
    }
    try {
        m = ParsePositiveUint64(argv[1], "m");
        k = ParsePositiveUint64(argv[2], "k");
        n = ParsePositiveUint64(argv[3], "n");
        if (argc >= 6) {
            transA = (std::string(argv[4]) == "true");
            transB = (std::string(argv[5]) == "true");
        }
        if (argc >= 7) {
            aLayout = argv[6];
        }
        if (argc >= 8) {
            bLayout = argv[7];
        }
        if (argc >= 9) {
            mode = argv[8];
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    constexpr int32_t deviceId = 0;
    constexpr uint64_t C0 = 16;  // [MODIFY N2] fp16/bf16: C0=16; fp8/int8: C0=32

    try {
        // [MODIFY A1] Host 侧 tiling：按 mode 选择 tiling 引擎。
        MatmulTilingData tilingData;
        if (mode == "a_full") {
            MatmulTilingAFullLoad tilingEngine;
            tilingEngine.GetTilingData(m, n, k, tilingData, transA, transB, aLayout == "nz", bLayout == "nz");
        } else {
            MatmulTilingSwat tilingEngine;
            tilingEngine.GetTilingData(m, n, k, tilingData, transA, transB, aLayout == "nz", bLayout == "nz");
        }

        AclRtSession aclSession(deviceId);
        aclSession.Init();
        aclrtStream stream = aclSession.GetStream();

        // [MODIFY N2] NZ 格式 size 按物理维度计算；ND 格式按逻辑维度。
        uint64_t sizeA = (aLayout == "nz")
            ? (transA ? CalcNzSize(k, m, C0) : CalcNzSize(m, k, C0))
            : m * k * sizeof(uint16_t);
        uint64_t sizeB = (bLayout == "nz")
            ? (transB ? CalcNzSize(n, k, C0) : CalcNzSize(k, n, C0))
            : k * n * sizeof(uint16_t);
        uint64_t sizeC = m * n * sizeof(uint16_t);

        ExampleIoPaths paths = GetExampleIoPaths();

        uint16_t* hA = nullptr;
        uint16_t* hB = nullptr;
        uint16_t* hC = nullptr;

        GM_ADDR dA = nullptr;
        GM_ADDR dB = nullptr;
        GM_ADDR dC = nullptr;

        CHECK_COND(
            aclrtMallocHost((void**)&hA, sizeA) == ACL_SUCCESS, "Failed to allocate host buffer for A.");
        std::unique_ptr<void, aclError (*)(void*)> hostA(hA, aclrtFreeHost);
        CHECK_COND(
            aclrtMallocHost((void**)&hB, sizeB) == ACL_SUCCESS, "Failed to allocate host buffer for B.");
        std::unique_ptr<void, aclError (*)(void*)> hostB(hB, aclrtFreeHost);
        CHECK_COND(
            aclrtMallocHost((void**)&hC, sizeC) == ACL_SUCCESS, "Failed to allocate host buffer for C.");
        std::unique_ptr<void, aclError (*)(void*)> hostC(hC, aclrtFreeHost);

        CHECK_COND(ReadExactFile(paths.inputDir + "/input_a.bin", hA, sizeA), "Failed to read input_a.bin.");
        CHECK_COND(ReadExactFile(paths.inputDir + "/input_b.bin", hB, sizeB), "Failed to read input_b.bin.");

        CHECK_COND(
            aclrtMalloc((void**)&dA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
            "Failed to allocate device buffer for A.");
        std::unique_ptr<void, aclError (*)(void*)> deviceA(dA, aclrtFree);
        CHECK_COND(
            aclrtMalloc((void**)&dB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
            "Failed to allocate device buffer for B.");
        std::unique_ptr<void, aclError (*)(void*)> deviceB(dB, aclrtFree);
        CHECK_COND(
            aclrtMalloc((void**)&dC, sizeC, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
            "Failed to allocate device buffer for C.");
        std::unique_ptr<void, aclError (*)(void*)> deviceC(dC, aclrtFree);

        CHECK_COND(
            aclrtMemcpyAsync(dA, sizeA, hA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
            "Failed to copy A to device.");
        CHECK_COND(
            aclrtMemcpyAsync(dB, sizeB, hB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
            "Failed to copy B to device.");

        // Lambda dispatch：按 aLayout/bLayout/transA/transB/mode 选择 kernel 实例化。
        auto launchKernel = [&](auto layoutATag, auto layoutBTag) {
            using LA = decltype(layoutATag);
            using LB = decltype(layoutBTag);
            if (mode == "a_full") {
                // A_FULL_LOAD 仅支持 transA=false && transB=false
                if constexpr (!TagToTrans<LA>::value && !TagToTrans<LB>::value) {
                    matmul_a_full_load<LA, LB>
                        <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dC, tilingData);
                }
            } else {
                matmul_custom<LA, LB>
                    <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dC, tilingData);
            }
        };

        auto dispatchB = [&](auto layoutATag) {
            if (bLayout == "nz") {
                if (transB) {
                    launchKernel(layoutATag, AscendC::Te::ZNLayoutPtn{});
                } else {
                    launchKernel(layoutATag, AscendC::Te::NZLayoutPtn{});
                }
            } else {
                if (transB) {
                    launchKernel(layoutATag, AscendC::Te::DNExtLayoutPtn{});
                } else {
                    launchKernel(layoutATag, AscendC::Te::NDExtLayoutPtn{});
                }
            }
        };

        if (aLayout == "nz") {
            if (transA) {
                dispatchB(AscendC::Te::ZNLayoutPtn{});
            } else {
                dispatchB(AscendC::Te::NZLayoutPtn{});
            }
        } else {
            if (transA) {
                dispatchB(AscendC::Te::DNExtLayoutPtn{});
            } else {
                dispatchB(AscendC::Te::NDExtLayoutPtn{});
            }
        }

        CHECK_COND(
            aclrtMemcpyAsync(hC, sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS,
            "Failed to copy C to host.");
        CHECK_COND(
            aclrtSynchronizeStream(stream) == ACL_SUCCESS,
            "Failed to synchronize stream.");

        CHECK_COND(WriteFile(paths.outputDir + "/output.bin", hC, sizeC), "Failed to write output.bin.");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
