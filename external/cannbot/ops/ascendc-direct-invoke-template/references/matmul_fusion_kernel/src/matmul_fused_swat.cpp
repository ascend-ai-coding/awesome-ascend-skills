// ============================================================
// [EXTEND] mxfp8 + eltwise 融合算子 Host Launcher（参考实现：Div）
//
// 本文件是**融合算子 Host Launcher 的参考样板**。默认实现为 mxfp8 + Div：
//   Output = matmul(A, B) / D
//   A, B   : mxfp8(E4M3) 量化矩阵，含 Scale 张量
//   D      : 第二路输入 [M, N], dtype = float (可按需替换为 half/其他)
//   Output : [M, N], dtype = float
//
// 适配其他 eltwise（Mul / Add / Relu / Cast / ...）的修改点：
//   1) 替换 #include "epilogue/div_epilogue.h" 为自实现 Epilogue 头文件
//   2) 替换 using MyEpilogue = DivEpilogue 为自实现类
//   3) 按需调整 DivisorType 的语义与 size 计算（sizeD）
//   4) 调整 Kernel Entry 的 dDivisor 参数语义（名字可保留或重命名）
//   5) 同步修改 scripts/gen_data.py 的 gen_fusion_inputs + compute_golden
//
// 关键架构约束（不可违反，详见 references/matmul_fusion_guide.md“机制说明”）：
//   - 使用 MIX 模式：无 __cube__，AIC/AIV 共享同一 Kernel 入口
//   - BlockMmad L0C 输出恒定走 Fixpipe→UB（SPLIT_M 双 AIV），单算子路径已移除
//   - Params 必须包含 6 子字段：
//       problemShape / mmadParams / l1Params / schParams / qbmmParams / epilogueParams
//     其中 mmadParams 仅 5 个 GM 地址：A / B / C / ScaleA / ScaleB
//   - CV 同步由 Kernel 层统一管理（不在 Epilogue 内部发 AIC→AIV Flag）
// ============================================================

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "acl/acl.h"

// [CONFIG] Include tiling headers BEFORE kernel_operator.h to avoid
// DataType ambiguity (project's enum class DataType vs AscendC::DataType)
#include "block/block_scheduler_policy.h"
#include "host_utils/acl_utils.h"
#include "host_utils/common_utils.h"
#include "host_utils/io_utils.h"
#include "kernel_utils/layout_utils.h"
#include "tiling/tiling_swat.h"
#include "tiling/tiling_data.h"

// Kernel headers (bring in AscendC::DataType)
#include "kernel_operator.h"
#include "kernel/matmul_kernel_fused.h"

// [EXTEND] 包含 DivEpilogue
#include "epilogue/div_epilogue.h"

// ============================================================
// [MODIFY] 数据类型别名 — 与 DESIGN.md 一致
// ============================================================
using AType = fp8_e4m3fn_t;           // mxfp8(E4M3)
using BType = fp8_e4m3fn_t;           // mxfp8(E4M3)
using CType = float;                  // matmul 内部累加类型 / Epilogue 全链路 float
using OutputType = float;             // 最终输出类型
using DivisorType = float;            // D 除数类型

// [EXTEND] 使用简化版 DivEpilogue (全 float, 无模板参数)
using MyEpilogue = DivEpilogue;

using layoutA = layout::RowMajor;
using layoutB = layout::ColumnMajor;
using layoutC = layout::RowMajor;

using BlockScheduler = QuantMatmulMxSwatScheduler<NO_FULL_LOAD_MODE>;

template <uint64_t Stages>
using DispatchPolicy = QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE, Stages>;

template <uint64_t Stages>
using BlockMmadFused = Block::BlockMmad<
    DispatchPolicy<Stages>, AType, layoutA, BType, layoutB, CType, layoutC>;

using ProblemShape = MatmulShape;

template <uint64_t Stages>
using FusedKernel = Kernel::MatmulKernelFused<
    ProblemShape, BlockMmadFused<Stages>, BlockScheduler, MyEpilogue>;

// ============================================================
// Kernel 函数声明（MIX 模式，无 __cube__）
// ============================================================
template <uint64_t Stages>
__global__ __aicore__ void FusedKernelEntry(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dScaleA, GM_ADDR dScaleB,
    GM_ADDR dDivisor, GM_ADDR dOutput,
    const QuantMatmulTilingData tilingData)
{
    FusedKernel<Stages> kernel;
    typename FusedKernel<Stages>::Params params;

    // 构建 problemShape
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, 1UL};

    // 构建 mmadParams（5 个 GM 地址；融合模式下 cGmAddr 不用于写回，仅签名兼容）
    params.mmadParams = {
        dA, dB, dOutput, dScaleA, dScaleB
    };

    // 构建 l1Params
    params.l1Params = {
        static_cast<uint64_t>(tilingData.stepK) * tilingData.baseK,
        tilingData.scaleKL1
    };

    // 构建 schParams
    params.schParams = {
        static_cast<int64_t>(tilingData.baseM),
        static_cast<int64_t>(tilingData.baseN),
        static_cast<int64_t>(tilingData.mTailTile),
        static_cast<int64_t>(tilingData.nTailTile),
        static_cast<int64_t>(tilingData.mBaseTailSplitCnt),
        static_cast<int64_t>(tilingData.nBaseTailSplitCnt),
        static_cast<int64_t>(tilingData.mTailMain),
        static_cast<int64_t>(tilingData.nTailMain)
    };

    // 构建 qbmmParams
    params.qbmmParams = {
        tilingData.baseM,
        tilingData.baseN,
        tilingData.baseK,
        tilingData.dbL0c
    };

    // 构建 epilogueParams (仅 GM 地址，per-tile 参数由 Kernel 层动态计算)
    params.epilogueParams = {
        dDivisor,           // divisorGmAddr
        dOutput             // outputGmAddr
    };

    kernel(params);
}

// ============================================================
// main — Host 侧入口
// ============================================================
int main(int argc, char* argv[])
{
    uint64_t m = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    try {
        if (argc < 4) {
            throw std::invalid_argument("ERROR: Expected at least 3 arguments: m k n");
        }
        m = ParsePositiveUint64(argv[1], "m");
        k = ParsePositiveUint64(argv[2], "k");
        n = ParsePositiveUint64(argv[3], "n");
        CheckUint32Shape(m, "m");
        CheckUint32Shape(k, "k");
        CheckUint32Shape(n, "n");
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    constexpr int32_t deviceId = 0;

    try {
        // ---- Tiling ----
        QuantMatmulTilingData tilingData;
        QuantMatmulTilingSwat<::DataType::FP8, ::DataType::FP8> tilingEngine;
        tilingEngine.GetTilingData(m, n, k, tilingData);

        // ---- ACL 初始化 ----
        AclRtSession aclSession(deviceId);
        aclSession.Init();
        aclrtStream stream = aclSession.GetStream();

        // ---- 内存大小计算 ----
        uint64_t sizeA = (m * k) * sizeof(uint8_t);
        uint64_t sizeB = (k * n) * sizeof(uint8_t);
        uint64_t sizeScaleA =
            (m * CeilDiv(k, TILING_MXFP_DIVISOR_SIZE) * TILING_MXFP_MULTI_BASE_SIZE) * sizeof(uint8_t);
        uint64_t sizeScaleB =
            (n * CeilDiv(k, TILING_MXFP_DIVISOR_SIZE) * TILING_MXFP_MULTI_BASE_SIZE) * sizeof(uint8_t);
        uint64_t sizeC = m * n * sizeof(OutputType);
        // [EXTEND] D 除数张量大小
        uint64_t sizeD = m * n * sizeof(DivisorType);

        ExampleIoPaths paths = GetExampleIoPaths();

        // ---- Host 内存分配 ----
        uint8_t* hA = nullptr;
        uint8_t* hB = nullptr;
        uint8_t* hScaleA = nullptr;
        uint8_t* hScaleB = nullptr;
        OutputType* hC = nullptr;
        // [EXTEND] D 除数 Host 内存
        DivisorType* hD = nullptr;

        CHECK_COND(aclrtMallocHost((void**)&hA, sizeA) == ACL_SUCCESS, "Alloc hA failed.");
        std::unique_ptr<void, aclError (*)(void*)> hostA(hA, aclrtFreeHost);
        CHECK_COND(aclrtMallocHost((void**)&hB, sizeB) == ACL_SUCCESS, "Alloc hB failed.");
        std::unique_ptr<void, aclError (*)(void*)> hostB(hB, aclrtFreeHost);
        CHECK_COND(aclrtMallocHost((void**)&hScaleA, sizeScaleA) == ACL_SUCCESS, "Alloc hScaleA failed.");
        std::unique_ptr<void, aclError (*)(void*)> hostScaleA(hScaleA, aclrtFreeHost);
        CHECK_COND(aclrtMallocHost((void**)&hScaleB, sizeScaleB) == ACL_SUCCESS, "Alloc hScaleB failed.");
        std::unique_ptr<void, aclError (*)(void*)> hostScaleB(hScaleB, aclrtFreeHost);
        CHECK_COND(aclrtMallocHost((void**)&hC, sizeC) == ACL_SUCCESS, "Alloc hC failed.");
        std::unique_ptr<void, aclError (*)(void*)> hostC(hC, aclrtFreeHost);
        // [EXTEND] D 除数 Host 内存分配
        CHECK_COND(aclrtMallocHost((void**)&hD, sizeD) == ACL_SUCCESS, "Alloc hD failed.");
        std::unique_ptr<void, aclError (*)(void*)> hostD(hD, aclrtFreeHost);

        // ---- 读取输入文件 ----
        CHECK_COND(ReadExactFile(paths.inputDir + "/input_a.bin", hA, sizeA), "Read input_a.bin failed.");
        CHECK_COND(ReadExactFile(paths.inputDir + "/input_b.bin", hB, sizeB), "Read input_b.bin failed.");
        CHECK_COND(ReadExactFile(paths.inputDir + "/input_scaleA.bin", hScaleA, sizeScaleA), "Read scaleA failed.");
        CHECK_COND(ReadExactFile(paths.inputDir + "/input_scaleB.bin", hScaleB, sizeScaleB), "Read scaleB failed.");
        // [EXTEND] 读取 D 除数输入
        CHECK_COND(ReadExactFile(paths.inputDir + "/input_d.bin", hD, sizeD), "Read input_d.bin failed.");

        // ---- Device 内存分配 ----
        GM_ADDR dA = nullptr;
        GM_ADDR dB = nullptr;
        GM_ADDR dScaleA = nullptr;
        GM_ADDR dScaleB = nullptr;
        GM_ADDR dC = nullptr;
        // [EXTEND] D 除数 Device 内存
        GM_ADDR dD = nullptr;

        CHECK_COND(aclrtMalloc((void**)&dA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "Alloc dA failed.");
        std::unique_ptr<void, aclError (*)(void*)> deviceA(dA, aclrtFree);
        CHECK_COND(aclrtMalloc((void**)&dB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "Alloc dB failed.");
        std::unique_ptr<void, aclError (*)(void*)> deviceB(dB, aclrtFree);
        CHECK_COND(aclrtMalloc((void**)&dScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "Alloc dScaleA.");
        std::unique_ptr<void, aclError (*)(void*)> deviceScaleA(dScaleA, aclrtFree);
        CHECK_COND(aclrtMalloc((void**)&dScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "Alloc dScaleB.");
        std::unique_ptr<void, aclError (*)(void*)> deviceScaleB(dScaleB, aclrtFree);
        CHECK_COND(aclrtMalloc((void**)&dC, sizeC, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "Alloc dC failed.");
        std::unique_ptr<void, aclError (*)(void*)> deviceC(dC, aclrtFree);
        // [EXTEND] D 除数 Device 内存分配
        CHECK_COND(aclrtMalloc((void**)&dD, sizeD, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "Alloc dD failed.");
        std::unique_ptr<void, aclError (*)(void*)> deviceD(dD, aclrtFree);

        // ---- H2D 拷贝 ----
        CHECK_COND(aclrtMemcpyAsync(dA, sizeA, hA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS, "H2D A.");
        CHECK_COND(aclrtMemcpyAsync(dB, sizeB, hB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS, "H2D B.");
        CHECK_COND(aclrtMemcpyAsync(dScaleA, sizeScaleA, hScaleA, sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS, "H2D scaleA.");
        CHECK_COND(aclrtMemcpyAsync(dScaleB, sizeScaleB, hScaleB, sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS, "H2D scaleB.");
        // [EXTEND] D 除数 H2D 拷贝
        CHECK_COND(aclrtMemcpyAsync(dD, sizeD, hD, sizeD, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS, "H2D D.");

        // ---- Kernel Launch ----
        if (tilingData.nBufferNum == 4U) {
            FusedKernelEntry<4><<<tilingData.usedCoreNum, nullptr, stream>>>(
                dA, dB, dScaleA, dScaleB, dD, dC, tilingData);
        } else {
            FusedKernelEntry<2><<<tilingData.usedCoreNum, nullptr, stream>>>(
                dA, dB, dScaleA, dScaleB, dD, dC, tilingData);
        }

        // ---- D2H 拷贝 + 写出 ----
        CHECK_COND(aclrtMemcpyAsync(hC, sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS, "D2H C.");
        CHECK_COND(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "Sync stream failed.");
        CHECK_COND(WriteFile(paths.outputDir + "/npu_out.bin", hC, sizeC), "Write npu_out.bin failed.");

        std::cout << "Fused matmul kernel executed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
