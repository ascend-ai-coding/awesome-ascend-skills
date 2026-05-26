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
// Matmul FixpOpti Kernel 直调模板 —— BF16 in / BF16 out（dav-3510）
// ----------------------------------------------------------------------------
// FixpOpti（AIC+AIV 混合）：AIC 做 MMA + Fixpipe(L0C→UB)，AIV 做 Epilogue(Cast+写回)。
// SPLIT_M 拆分 M 到双 AIV，Epilogue 负责 float→bf16 Cast + DataCopyPad。
//
// [MODIFY] 标记体系：
//   N1 函数名 / CMake 目标名 / run.sh OP_NAME 三处同步
//   N2 AType / BType / CType + sizeA/B/C 字节数 + DATA_SIZE_FP16
//   N3 gen_data.py / verify_result.py 的 dtype / 容差
//   C1 layoutA / layoutB 与 transA/transB
//   C2 TilingData 增删字段（bias/scale 等额外输入）
//   A1 纯AIC↔FixpOpti 切换（dispatch mode / tiling 引擎）
//   E1 Epilogue 替换：IdentityEpilogue → 自定义 Epilogue
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
#include "kernel/matmul_kernel_fused.h"
#include "epilogue/identity_epilogue.h"
#include "tiling/matmul_tiling.h"
#include "tiling/matmul_tiling_data.h"

// ---------------- Kernel 入口 ----------------
// FixpOpti: __mix__(1,2) 表示每 block 1 AIC + 2 AIV。
// [MODIFY C1] LayoutA/LayoutB 由模板参数传入，支持 NN/NT/TN/TT 四种组合。
template <class LayoutA, class LayoutB>
__global__ __aicore__ __mix__(1, 2) void matmul_fixpopti(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dC,
    const MatmulTilingData tilingData)
{
    // [MODIFY N2] 输入/输出 dtype
    using AType = bfloat16_t;
    using BType = bfloat16_t;
    using CType = bfloat16_t;

    using layoutA = LayoutA;
    using layoutC = AscendC::Te::NDExtLayoutPtn;

    // [MODIFY A1] 选改：切 A 全载改为 A_FULL_LOAD_MODE + host 端 MatmulTilingAFullLoad
    using BlockScheduler = MatmulSwatScheduler<NO_FULL_LOAD_MODE>;
    using DispatchPolicy = MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>;
    using ProblemShape = MatmulShape;

    using BlockMmad = Block::BlockMmad<DispatchPolicy, AType, layoutA, BType, LayoutB, CType, layoutC>;
    // [MODIFY E1] 替换 IdentityEpilogue 为自定义 Epilogue
    using EpilogueOp = IdentityEpilogue<CType>;
    using MatmulKernelImpl = Kernel::MatmulKernelFused<ProblemShape, BlockMmad, BlockScheduler, EpilogueOp>;
    using Params = typename MatmulKernelImpl::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename MatmulKernelImpl::L1Params;
    using BlockSchedulerParams = typename MatmulKernelImpl::BlockSchedulerParams;
    using MatmulTiling = typename MatmulKernelImpl::QBMMTiling;
    using EpilogueParams = typename MatmulKernelImpl::EpilogueParams;

    ProblemShape problemShape{tilingData.m, tilingData.n, tilingData.k, 1L};
    BlockMmadParams mmadParams{dA, dB, dC};
    L1Params l1Params{static_cast<uint64_t>(tilingData.kL1)};
    BlockSchedulerParams schedulerParams{
        tilingData.baseM, tilingData.baseN,
        tilingData.mTailCnt, tilingData.nTailCnt,
        tilingData.mBaseTailSplitCnt, tilingData.nBaseTailSplitCnt,
        tilingData.mTailMain, tilingData.nTailMain};
    MatmulTiling qbmmParams{
        tilingData.baseM, tilingData.baseN,
        tilingData.baseK, tilingData.l0cDB};
    EpilogueParams epilogueParams{dC};
    Params params{problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, epilogueParams};
    MatmulKernelImpl kernel;
    kernel(params);
}

// ---------------- Host 入口 ----------------
int main(int argc, char* argv[])
{
    uint64_t m = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    // [MODIFY C1] 默认 NN 模式
    bool transA = false;
    bool transB = false;
    try {
        ParseArguments(argc, argv, m, k, n, transA, transB);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

    constexpr int32_t deviceId = 0;

    try {
        MatmulTilingData tilingData;
        // [MODIFY A1] 切 A 全载：替换为 MatmulTilingAFullLoad tilingEngine;
        MatmulTilingSwat tilingEngine;
        tilingEngine.GetTilingData(m, n, k, tilingData);

        AclRtSession aclSession(deviceId);
        aclSession.Init();
        aclrtStream stream = aclSession.GetStream();

        // [MODIFY N2] dtype byte size
        uint64_t sizeA = m * k * sizeof(uint16_t);
        uint64_t sizeB = k * n * sizeof(uint16_t);
        uint64_t sizeC = m * n * sizeof(uint16_t);

        ExampleIoPaths paths = GetExampleIoPaths();

        uint16_t* hA = nullptr;
        uint16_t* hB = nullptr;
        uint16_t* hC = nullptr;

        GM_ADDR dA = nullptr;
        GM_ADDR dB = nullptr;
        GM_ADDR dC = nullptr;

        CHECK_COND(aclrtMallocHost((void**)&hA, sizeA) == ACL_SUCCESS, "Failed to allocate host buffer for A.");
        std::unique_ptr<void, aclError (*)(void*)> hostA(hA, aclrtFreeHost);
        CHECK_COND(aclrtMallocHost((void**)&hB, sizeB) == ACL_SUCCESS, "Failed to allocate host buffer for B.");
        std::unique_ptr<void, aclError (*)(void*)> hostB(hB, aclrtFreeHost);
        CHECK_COND(aclrtMallocHost((void**)&hC, sizeC) == ACL_SUCCESS, "Failed to allocate host buffer for C.");
        std::unique_ptr<void, aclError (*)(void*)> hostC(hC, aclrtFreeHost);

        CHECK_COND(ReadExactFile(paths.inputDir + "/input_a.bin", hA, sizeA), "Failed to read input_a.bin.");
        CHECK_COND(ReadExactFile(paths.inputDir + "/input_b.bin", hB, sizeB), "Failed to read input_b.bin.");

        CHECK_COND(aclrtMalloc((void**)&dA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "Failed to alloc device A.");
        std::unique_ptr<void, aclError (*)(void*)> deviceA(dA, aclrtFree);
        CHECK_COND(aclrtMalloc((void**)&dB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "Failed to alloc device B.");
        std::unique_ptr<void, aclError (*)(void*)> deviceB(dB, aclrtFree);
        CHECK_COND(aclrtMalloc((void**)&dC, sizeC, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "Failed to alloc device C.");
        std::unique_ptr<void, aclError (*)(void*)> deviceC(dC, aclrtFree);

        CHECK_COND(aclrtMemcpyAsync(dA, sizeA, hA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
                   "Failed to copy A to device.");
        CHECK_COND(aclrtMemcpyAsync(dB, sizeB, hB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
                   "Failed to copy B to device.");

        // FixpOpti NN 模式：双端 NDExtLayoutPtn
        using NDExt = AscendC::Te::NDExtLayoutPtn;
        matmul_fixpopti<NDExt, NDExt>
            <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dC, tilingData);

        CHECK_COND(aclrtMemcpyAsync(hC, sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS,
                   "Failed to copy C to host.");
        CHECK_COND(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "Failed to synchronize stream.");

        CHECK_COND(WriteFile(paths.outputDir + "/output.bin", hC, sizeC), "Failed to write output.bin.");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
