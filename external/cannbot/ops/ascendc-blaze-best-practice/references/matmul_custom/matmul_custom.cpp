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
// 单一模板同时支持 NO_FULL_LOAD_MODE（默认，通用 SWAT）与 A_FULL_LOAD_MODE（A 全载）。
// 默认走通用模式；切到 A 全载只需替换 3 行（见下方 [MODIFY A1] 精确 diff）。
//
// 创建新算子时按下面 [MODIFY] 标记修改。搜索 `[MODIFY]` 即可定位每个改点；
// 按重要性分三档（必改 / 常改 / 选改）：
//
// === 必改（任何新算子都要动）===
//   [MODIFY] N1  函数名 / CMake 目标名 / run.sh OP_NAME 三处保持一致
//   [MODIFY] N2  AType / BType / CType（搭配 sizeA/sizeB/sizeC 字节数 +
//                matmul_tiling_constant.h::DATA_SIZE_FP16）
//   [MODIFY] N3  scripts/gen_data.py + verify_result.py 的 dtype / golden / 容差
//                （A 全载场景：默认 trans_b=False；bf16 长 K 用 MERE/MARE 双门替代严格 allclose）
//
// === 常改（按算子需求二选一）===
//   [MODIFY] C1  layoutA / layoutB 与 transA/transB（通用模板默认 transA=false，
//                A 全载模板锁定 transA=transB=false）
//   [MODIFY] C2  TilingData 增删字段（bias/scale 等额外输入）—— 见 `matmul_basic.md` §2.2
//
// === 选改（高级变种才需要）===
//   [MODIFY] A1  切到 A 全载（NO_FULL_LOAD_MODE → A_FULL_LOAD_MODE）—— 见下方 launcher 注释；
//                适用 Align(m,16)*Align(k,16)*sizeof(A) ≤ ~256KB 且 N≫M，详见 `matmul_full_load.md`
//   [MODIFY] A2  L1_BUFFER_NUM = 4 等更深流水（需同步 dispatch policy）—— 见 `matmul_basic.md` §2.4
//
// 进阶细节（含排错、SFINAE、Tiling 算法）请翻 `matmul_pattern.md`（先选 pattern，再读
// `matmul_basic.md`（通用模板）或 `matmul_full_load.md`（A 全载模板））。
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
// [MODIFY N1] 函数名 `matmul_custom` 需与 CMake 目标名、run.sh 中 OP_NAME 保持一致。
// 模板参数 LayoutB 是 tensor_api 的 layout pattern：
//   - AscendC::Te::NDExtLayoutPtn  → 行主序（host 落盘 (K, N)）
//   - AscendC::Te::DNExtLayoutPtn  → 列主序（host 落盘 (N, K)，device 视图 (K, N)）
// host 侧根据 transB 选择对应 pattern 实例化；transA 路径同理（见下方 [MODIFY C1]）。
// [MODIFY C1] 如需支持 transA，把模板签名改为 `template <class LayoutA, class LayoutB>`，
// 并在 host 侧按 4 种 (transA, transB) 组合实例化（详见 launch_details §6 step-by-step）。
template <class LayoutB>
__global__ __aicore__ __cube__ void matmul_custom(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dC,
    const MatmulTilingData tilingData)
{
    // [MODIFY N2] 输入/输出 dtype。替换原则：
    //   - bf16/fp16/fp8 → BlockMmad 内部自动用 fp32 累加；
    //   - int8        → BlockMmad 内部自动用 int32 累加（见 matmul_block_mmad.h:77）；
    //   - 不支持的组合（如 int8→fp32 L0C）会被 check_data_type_3510.h static_assert 拒掉。
    using AType = bfloat16_t;
    using BType = bfloat16_t;
    using CType = bfloat16_t;

    // [MODIFY C1] A / C 侧 GM layout pattern：
    //   - NDExtLayoutPtn = 行主序（默认，host 落盘 (M, K) / (M, N)）
    //   - DNExtLayoutPtn = 列主序（transA / transC 时使用）
    // 默认锁 transA=false，所以 layoutA 固定 NDExtLayoutPtn；C 侧目前只支持 NDExtLayoutPtn。
    using layoutA = AscendC::Te::NDExtLayoutPtn;
    using layoutC = AscendC::Te::NDExtLayoutPtn;

    // [MODIFY A1] 选改：默认 SWAT (non-full-load) 流水。
    // 切到 A 全载（A 全部装入 L1 + 跨 N-tile 复用），把 mode 改为 A_FULL_LOAD_MODE，
    // **并按下面 Host 侧 [MODIFY A1] 同步切 tiling 引擎 + 锁 transA=transB=false**：
    //   - using BlockScheduler = MatmulSwatScheduler<A_FULL_LOAD_MODE>;
    //   - using DispatchPolicy = MatmulMultiBlockPolicy<A_FULL_LOAD_MODE>;
    //   - host 端：MatmulTilingAFullLoad tilingEngine; （替换 MatmulTilingSwat）
    // A 全载仅支持 transA=transB=false；适用判据 / L1 布局见 matmul_full_load.md §1。
    // 量化 / MX / Attention 等变种另在 matmul_block_mmad.h 加 SFINAE 特化（matmul_basic.md §2.3）。
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

    // [PITFALL] Params 是聚合初始化，字段顺序、个数必须与 matmul_kernel.h 中声明一致，
    // 否则触发 `excess elements in scalar initializer`。8 字段的 schedulerParams 不能省略。
    // [MODIFY C2] 新增 bias / scale 等额外输入：① TilingData 加字段 ② BlockMmadParams 加地址
    // ③ 这里把新地址塞进 mmadParams（详见 launch_details §8.2）。
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
    bool transB = true;  // 默认 B 采用列主序（host 文件存 (N, K)，对应 DNExtLayoutPtn）
    try {
        ParseArguments(argc, argv, m, k, n, transA, transB);
        // [MODIFY C1] 打开 transA 时删掉这条 throw，并让上面的 launcher 模板增加 LayoutA 参数。
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
        // Host 侧 tiling：默认 MatmulTilingSwat（通用 bf16/fp16）。
        // [MODIFY A1] 切到 A 全载：替换为 `MatmulTilingAFullLoad tilingEngine;`，并把上面 launcher
        // 的 DispatchPolicy / BlockScheduler mode 同步改为 A_FULL_LOAD_MODE（详见 matmul_full_load.md §1.3）。
        // [MODIFY N2] 切换数据类型时同步检查 matmul_tiling_constant.h::DATA_SIZE_FP16，
        // 否则 L1 预算会按错误字节数估算导致 OOM 或吃不满 buffer。
        MatmulTilingData tilingData;
        MatmulTilingSwat tilingEngine;
        tilingEngine.GetTilingData(m, n, k, tilingData);

        AclRtSession aclSession(deviceId);
        aclSession.Init();
        aclrtStream stream = aclSession.GetStream();

        // [MODIFY N2] dtype byte size：bf16/fp16=2，fp32=4，fp8/int8=1，fp4×2=0.5（按 packed 元素数计）。
        // 三处 sizeof 必须用 `sizeof(<对应 host 端 dtype>)` 而不是固定 uint16_t；改 dtype 时一并改 hA/hB/hC 指针类型。
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

        // Kernel 直调 <<<usedCoreNum, nullptr, stream>>>：
        //   - 第一参 usedCoreNum：tiling 给出的实际启动核数（纯 Cube 直接用，不乘 GetTaskRation）
        //   - 第二参 nullptr：tpipe 占位，Matmul 直调不需要
        //   - 第三参 stream：ACL stream
        // 运行时根据 transB 选择对应的 layout pattern 实例化（transA 路径见上面的 throw）。
        if (transB) {
            matmul_custom<AscendC::Te::DNExtLayoutPtn>
                <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dC, tilingData);
        } else {
            matmul_custom<AscendC::Te::NDExtLayoutPtn>
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
