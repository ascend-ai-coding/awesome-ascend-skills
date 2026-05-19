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
// Matmul Kernel 直调样例 —— BF16 in / BF16 out
// ----------------------------------------------------------------------------
// [MODIFY] 创建新算子时的修改点（搜索 [MODIFY] 标记）：
//   1. AType/BType/CType  —— 输入/输出数据类型（bf16、fp16、fp8…）
//   2. layoutA/layoutB    —— 输入矩阵 GM 排布（RowMajor / ColumnMajor）
//   3. Kernel 函数名       —— matmul_custom → <your_op>_custom
//   4. DispatchPolicy/Scheduler —— 默认 MatmulMultiBlockPolicy / MatmulSwatScheduler
//   5. TilingData 字段     —— 如需扩展（bias/scale/多 dtype），同步 matmul_tiling_data.h
//   6. Host 主流程         —— sizeA/sizeB/sizeC 按 dtype 字节数计算
//   7. CMakeLists.txt      —— 目标名 matmul_custom → <your_op>_custom
//   8. scripts/gen_data.py —— 输入分布与 golden 计算替换为真实参考实现
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

// ---------------- Kernel 入口 ----------------
// [MODIFY] 函数名 `matmul_custom` 需与 CMake 目标名、run.sh 中 OP_NAME 保持一致。
// 模板参数 LayoutB 在 host 侧运行时选择（transB=true => ColumnMajor, false => RowMajor）。
// 如需支持 transA，把模板签名改为 `template <class LayoutA, class LayoutB>`，
// 并在 host 侧按 4 种 (transA, transB) 组合实例化（详见 matmul_custom_launch_details.md §6）。
template <class LayoutB>
__global__ __aicore__ __cube__ void matmul_custom(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dC,
    const MatmulTilingData tilingData)
{
    // [MODIFY] 数据类型：BF16 in / BF16 out。替换为 half / fp8 / int8 等时，
    // 同步修改 gen_data.py 的 dtype、verify_result.py 的 dtype 与容差。
    // 注意 int8 输入需在 BlockMmad 中切换 L0CType=int32（详见 launch_details §8.1）。
    using AType = bfloat16_t;
    using BType = bfloat16_t;
    using CType = bfloat16_t;

    // [MODIFY] 逻辑 layout：A 行主序 / C 行主序（默认不支持 transA）；B 支持 Row/Col 两种。
    using layoutA = layout::RowMajor;
    using layoutC = layout::RowMajor;

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

// ---------------- Host 入口 ----------------
int main(int argc, char* argv[])
{
    uint64_t m = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    bool transA = false;
    bool transB = true;  // [CONFIG] 默认 B 采用 ColumnMajor（host 存 (N, K)）
    try {
        ParseArguments(argc, argv, m, k, n, transA, transB);
        if (transA) {
            throw std::invalid_argument("ERROR: transA=true is not supported by matmul_custom yet");
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

    constexpr int32_t deviceId = 0;

    try {
        // [MODIFY] Host 侧 tiling：沿用 MatmulTilingSwat（bf16/fp16 通用）。
        // 切换数据类型需要检查 tiling 内部的 DATA_SIZE_FP16 是否匹配。
        MatmulTilingData tilingData;
        MatmulTilingSwat tilingEngine;
        tilingEngine.GetTilingData(m, n, k, tilingData);

        AclRtSession aclSession(deviceId);
        aclSession.Init();
        aclrtStream stream = aclSession.GetStream();

        // [MODIFY] 按数据类型调整 byte size。bf16 = 2 字节；fp32 = 4；fp8 = 1。
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

        CHECK_COND(
            aclrtMallocHost((void**)&hA, sizeA) == ACL_SUCCESS, "Failed to allocate the host buffer for input A.");
        std::unique_ptr<void, aclError (*)(void*)> hostA(hA, aclrtFreeHost);
        CHECK_COND(
            aclrtMallocHost((void**)&hB, sizeB) == ACL_SUCCESS, "Failed to allocate the host buffer for input B.");
        std::unique_ptr<void, aclError (*)(void*)> hostB(hB, aclrtFreeHost);
        CHECK_COND(
            aclrtMallocHost((void**)&hC, sizeC) == ACL_SUCCESS, "Failed to allocate the host buffer for output C.");
        std::unique_ptr<void, aclError (*)(void*)> hostC(hC, aclrtFreeHost);

        CHECK_COND(ReadExactFile(paths.inputDir + "/input_a.bin", hA, sizeA), "Failed to read input_a.bin.");
        CHECK_COND(ReadExactFile(paths.inputDir + "/input_b.bin", hB, sizeB), "Failed to read input_b.bin.");

        CHECK_COND(
            aclrtMalloc((void**)&dA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
            "Failed to allocate the device buffer for input A.");
        std::unique_ptr<void, aclError (*)(void*)> deviceA(dA, aclrtFree);
        CHECK_COND(
            aclrtMalloc((void**)&dB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
            "Failed to allocate the device buffer for input B.");
        std::unique_ptr<void, aclError (*)(void*)> deviceB(dB, aclrtFree);
        CHECK_COND(
            aclrtMalloc((void**)&dC, sizeC, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
            "Failed to allocate the device buffer for output C.");
        std::unique_ptr<void, aclError (*)(void*)> deviceC(dC, aclrtFree);

        CHECK_COND(
            aclrtMemcpyAsync(dA, sizeA, hA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
            "Failed to copy input A from host to device.");
        CHECK_COND(
            aclrtMemcpyAsync(dB, sizeB, hB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
            "Failed to copy input B from host to device.");

        // [PATTERN] Kernel 直调 <<<usedCoreNum, nullptr, stream>>>。
        // 运行时根据 transB 选择对应的模板实例化。
        if (transB) {
            matmul_custom<layout::ColumnMajor>
                <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dC, tilingData);
        } else {
            matmul_custom<layout::RowMajor>
                <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dC, tilingData);
        }

        CHECK_COND(
            aclrtMemcpyAsync(hC, sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS,
            "Failed to copy output C from device to host.");
        CHECK_COND(
            aclrtSynchronizeStream(stream) == ACL_SUCCESS,
            "Failed to synchronize the ACL stream after kernel execution.");

        CHECK_COND(WriteFile(paths.outputDir + "/output.bin", hC, sizeC), "Failed to write output.bin.");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
