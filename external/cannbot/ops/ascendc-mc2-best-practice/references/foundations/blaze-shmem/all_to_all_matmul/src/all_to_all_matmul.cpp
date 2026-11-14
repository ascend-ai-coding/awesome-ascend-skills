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
// MC2 通算融合算子 Host 启动器 —— AllToAll + 量化 Matmul（MX FP8）多卡直调
// ----------------------------------------------------------------------------
// 该 host 程序做四件事：
//   1. fork rankNum 个子进程，每个子进程 = 一个 rank（一张卡）；
//   2. 每个子进程：ACL init + SHMEM init（含 UDMA 引擎选择）+ 分配 1 GB SHMEM；
//   3. 加载 A/B/Scale 输入数据到 device GM，按 mode 启动 kernel：
//      - precision 模式：单次 kernel + 比对，开发期精度验证用；
//      - perf 模式：10 轮（L2 cache flush + main kernel），供 msprof op 采集；
//   4. 等 all rank 退出，主进程汇总 success/failure。
//
// 创建新 MC2 算子时按下面 [MODIFY] 标记修改。搜索 `[MODIFY]` 即可定位每个改点；
// 按重要性分三档（必改 / 常改 / 选改）：
//
// === 必改（任何新算子都要动）===
//   [MODIFY] N1  函数名 / CMake 目标名 / run.sh OP_NAME 三处保持一致（全局替换）
//   [MODIFY] N2  kernel 入口名（与 include/kernel/*.h 中的 __global__ 函数一致）
//   [MODIFY] N3  scripts/gen_data.py + verify_result.py 的 dtype / golden / 容差
//
// === 常改（按算子需求二选一）===
//   [MODIFY] C1  mode 分支（如需新增 debug/dump 模式）
//   [MODIFY] C2  TilingData 字段（与 include/tiling/all_to_all_matmul_tiling_data.h 同步）
//   [MODIFY] C3  SHMEM 空间预算（默认 1 GB，rankNum 多或 M 大时调大）
//
// === 选改（高级变种才需要）===
//   [MODIFY] A1  perf 主循环的 L2 flush 大小（B 矩阵 > 256 MB 时调大）
//   [MODIFY] A2  boost 预热阶段（极致去 host bound，默认注释掉）
//
// 进阶细节请翻 references/：
//   - references/workflow_integration.md  各 Step 的 host 侧动作
//   - references/codebase_map.md          完整文件改造食谱
//   - references/profiling_mc2.md         perf 模式与 msprof op 调用细节
//
// [PITFALL] 多 rank 启动用 fork，子进程必须 exit() 而不是 return——
//           return 会让子进程继续跑 main() 后续逻辑，导致重复 fork。
// [PITFALL] SHMEM_SPACE_SIZE 是所有 rank 共享的"虚"空间，每个 rank 各自分配等大；
//           算空间预算时按 `rankSize * bufferSize * bufferBlockSize + scale` 估算。
// [PITFALL] perf 模式必须每轮主 kernel 前调用 heavy_add_kernel 刷 L2，否则
//           前一轮的 B 矩阵驻留 L2，带宽指标失真。
// ============================================================================

#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits.h>
#include <unistd.h>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <sys/wait.h>

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "acl/acl.h"
#include <cstdlib>
#include "utils.h"
#include "tiling/quant_matmul_mx_tiling_swat.h"
#include "tiling/all_to_all_matmul_tiling_data.h"
#include "kernel/all_to_all_matmul_impl.h"
#include "kernel/heavy_kernels.h"

static constexpr uint64_t SHMEM_SPACE_SIZE = 1024UL * 1024UL * 1024UL;
static constexpr uint64_t PACKAGE_SIZE = 512UL;

inline uint64_t CeilDiv(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}


// [MODIFY N1] 函数名（printUsage / parseArguments / runAllToAllMatmul / main）若改 OP_NAME，全部同步改名。
void printUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n rankNum [mode]" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A (total K, distributed across ranks)" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "  rankNum: number of ranks" << std::endl;
    std::cerr << "  mode: optional, 'precision' (default) | 'perf'" << std::endl;
    std::cerr << "        precision - 跑一次 kernel，写 npu_out.bin 供 verify_result.py 比对" << std::endl;
    std::cerr << "        perf       - boost 预热 + cache_flush 流水，10 轮主循环测性能" << std::endl;
    std::cerr << "Example: " << programName << " 100 200 64 4" << std::endl;
    std::cerr << "         " << programName << " 2048 8192 3584 4 perf" << std::endl;
}

void InitData(uint8_t **hostPtr, uint8_t **devicePtr, size_t aSize, std::string path = "")
{
    std::cout << path << std::endl;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**> (devicePtr), aSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(hostPtr), aSize));
    if (path.length() == 0) {
        return;
    }
    ReadFile(path, *hostPtr, aSize);
    ACL_CHECK(aclrtMemcpy(*devicePtr, aSize, *hostPtr, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
}

void parseArguments(int argc, char* argv[], int& m, int& k, int& n, int& rankNum, std::string& mode)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        exit(1);
    }
    if (argc < 5) {
        throw std::invalid_argument("ERROR: Lacks Arguments");
    }
    try {
        m = std::stoi(argv[1]);
        k = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
        rankNum = std::stoi(argv[4]);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("ERROR: m k n rankNum must be Integer");
    }

    mode = (argc >= 6) ? std::string(argv[5]) : std::string("precision");
    if (mode != "precision" && mode != "perf") {
        throw std::invalid_argument("ERROR: mode must be 'precision' or 'perf'");
    }

    if (m <= 0 || k <= 0 || n <= 0 || rankNum <= 0) {
        throw std::invalid_argument("ERROR: m k n rankNum must be positive");
    }

    if (k % rankNum != 0) {
        throw std::invalid_argument("ERROR: k must be divisible by rankNum");
    }

    uint32_t ka = k / rankNum;
    if (CeilDiv(ka, 64) % 2 != 0) {
        throw std::invalid_argument("ERROR: Ka (k/rankNum) should satisfy that CeilDiv(Ka, 64) is an even number");
    }
}

// [MODIFY N1] runAllToAllMatmul → run{OpName}（与 main 中的调用同步）
// [PITFALL] ipport 是 rank 间发现的 TCP 地址，多机训练时改为跨机 IP；单机多卡用 127.0.0.1 即可。
// [PITFALL] 端口 8998 是任意选的，被占用时改其他端口；所有 rank 必须用同一 ipport。
int runAllToAllMatmul(int rankNum, int rankId, int m, int k, int n, const std::string& mode) {
    const char *ipport = "tcp://127.0.0.1:8998";
    INFO_LOG("rankNum=%d, rankId=%d, ipport=%s", rankNum, rankId, ipport);

    uint32_t ka = k / rankNum;

    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = rankId;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    // ACL_CHECK(aclshmemx_set_conf_store_tls(false, nullptr, 0));

    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t defaultFlagUid;

    // [MODIFY C2] TilingData 字段若在 include/tiling/all_to_all_matmul_tiling_data.h 增删，
    //             此处的赋值必须同步更新。详见 references/codebase_map.md §3.5。
    allToAllMatmulTilingData tilingData;


    // [MODIFY C2] headMSize：单次通信切块的 M 行数。
    //   - 必须 ≥ Blaze BlockMmad 的 baseM（典型 128/256），且为其整数倍；
    //   - 单次 Put 数据量 = headMSize * kPerRank * sizeof(dtype)，应落在 UDMA 高效区间（数十~数百 KB）；
    //   - 512 在 MX FP8 + kPerRank=2048 场景下 Put ≈ 1 MB，UDMA 带宽利用率最高。
    // [PITFALL] headMSize 必须能整除 m（除非显式处理 tail），否则 tileCnt 计算会丢尾块。
    uint32_t headMSize = 512; // m / tileCnt;
    uint32_t tailMSize = 0;// m - headMSize * tileCnt;

    uint32_t tileCnt = (m - tailMSize) / headMSize;
    tilingData.commTilingData.tileCnt = tileCnt;
    // [MODIFY C1] bufferSize：通信流水深度。
    //   - 4 是参考工程默认值（经验最优），新算子一般不动；
    //   - 调大需要同步增大 SHMEM_SPACE_SIZE；调小会失去通算重叠效果。
    tilingData.commTilingData.bufferSize = 4;
    INFO_LOG("TileCnt=%u, HeadMSize=%u", tileCnt, headMSize);

    // [MODIFY N2] tiling 引擎类型与 include/kernel/qbmm_mx_kernel.h 的模板参数一致；
    //             非 MX 量化场景换成 MatmulTilingSwat（见 ascendc-blaze-best-practice）。
    QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;

    // 只需一份 tiling（按 headMSize 切块的 tile tiling）。
    tilingEngine.GetTilingData(headMSize, n, ka, false, true, tilingData.tileQbmmTilingData);

    uint64_t localMemSize = SHMEM_SPACE_SIZE;
    test_set_attr(rankId, rankNum, localMemSize, ipport, defaultFlagUid, &attributes);
    // [PITFALL] 必须选 ACLSHMEM_DATA_OP_UDMA 引擎，整个 kernel 内的 aclshmemx_udma_*
    //           API 才会生效。选错引擎 Put 调用会静默失败（返回成功但不搬数）。
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA; // 使用udma
    ACL_CHECK_WITH_RET(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes),
        ERROR_LOG("aclshmemx_init_attr failed"), return -1);

    // [PITFALL] aclshmem_align 的两个参数：PACKAGE_SIZE 是对齐粒度（512B，UDMA 要求），
    //           SHMEM_SPACE_SIZE 是分配大小（1 GB）。改大小前先核对空间预算。
    void *shmemSpace = aclshmem_align(PACKAGE_SIZE, SHMEM_SPACE_SIZE);

    std::vector<uint8_t> hostA(m * k, 0);
    std::vector<uint8_t> hostShmemA(m * k, 0);
    std::vector<uint8_t> hostB(k * n, 0);
    std::vector<uint8_t> hostScaleA(m * CeilDiv(k, 64) * 2, 0);
    std::vector<uint8_t> hostShmemScaleA(m * CeilDiv(k, 64) * 2, 0);
    std::vector<uint8_t> hostScaleB(n * CeilDiv(k, 64) * 2, 0);
    std::vector<half> hostOutput(m * n, 0);

    auto sizeA = static_cast<size_t>(1) * hostA.size() * sizeof(uint8_t);
    auto sizeB = static_cast<size_t>(1) * hostB.size() * sizeof(uint8_t);
    auto sizeScaleA = static_cast<size_t>(1) * hostScaleA.size() * sizeof(uint8_t);
    auto sizeScaleB = static_cast<size_t>(1) * hostScaleB.size() * sizeof(uint8_t);
    auto sizeOutput = static_cast<size_t>(1) * hostOutput.size() * sizeof(half);

    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    std::string baseDir = ".";
    if (len > 0) {
        exePath[len] = '\0';
        baseDir = exePath;
        size_t lastSlash = baseDir.find_last_of('/');
        if (lastSlash != std::string::npos) {
            baseDir.resize(lastSlash);
        }
    }
    std::string inputDir = baseDir + "/input/" + std::to_string(rankId);
    std::string outputDir = baseDir + "/output/" + std::to_string(rankId);
    ReadFile(inputDir + "/input_a.bin", hostA.data(), sizeA);
    ReadFile(inputDir + "/input_b.bin", hostB.data(), sizeB);
    ReadFile(inputDir + "/input_scaleA.bin", hostScaleA.data(), sizeScaleA);
    ReadFile(inputDir + "/input_scaleB.bin", hostScaleB.data(), sizeScaleB);

    GM_ADDR deviceA = nullptr;
    GM_ADDR deviceB = nullptr;
    GM_ADDR deviceScaleA = nullptr;
    GM_ADDR deviceScaleB = nullptr;
    GM_ADDR deviceOutput = nullptr;

    ACL_CHECK(aclrtMalloc((void**)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void**)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void**)&deviceScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void**)&deviceScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void**)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_ONLY));

    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleA, sizeScaleA, hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleB, sizeScaleB, hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE));

    // 默认 precision 模式：只跑一次 kernel，写 npu_out.bin 供 verify_result.py 比对。
    // perf 模式：先 boost 预热去 host bound，再 10 轮 (cache_flush.add_(1) + main kernel)
    // [MODIFY N2] kernel 入口名 AllToAllQuantMatmulKernelE4M3E4M3 → 与 include/kernel/*.h 的 __global__ 一致。
    //             新算子按 (AType, BType) 组合提供多个特化入口（参考工程提供 4 个 E4M3/E5M2 组合）。
    if (mode == "precision") {
        AllToAllQuantMatmulKernelE4M3E4M3<<<tilingData.tileQbmmTilingData.usedCoreNum, nullptr, stream>>>(
            shmemSpace, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
        ACL_CHECK(aclrtSynchronizeStream(stream));

        GM_ADDR shmemA = (GM_ADDR)shmemSpace + 1024 * 1024;
        GM_ADDR shmemAScale = shmemA + m * k;
        ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hostShmemA.data(), sizeA, shmemA, sizeA, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hostShmemScaleA.data(), sizeScaleA, shmemAScale, sizeScaleA, ACL_MEMCPY_DEVICE_TO_HOST));

        WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);
        WriteFile(outputDir + "/shmem_A.bin", hostShmemA.data(), sizeA);
        WriteFile(outputDir + "/shmem_scale.bin", hostShmemScaleA.data(), sizeScaleA);
    } else {
        // 1) boost buffer: 200MB bf16 (100M elements) —— 100 次 exp 去 host bound
        // 2) cache_flush buffer: 256MB bf16 (128M elements) —— 每轮 1 次 add_(1) 刷 L2
        // [MODIFY A1] perf 模式的两个 buffer 大小：
        //   - BOOST_ELEM_COUNT：去 host bound 用，200 MB 足够；
        //   - CACHE_FLUSH_ELEM_COUNT：刷 L2 用，必须 > L2 容量（950 L2 ≈ 192 MB），256 MB 留余量；
        //   若新算子的 B 矩阵 > 256 MB，CACHE_FLUSH_ELEM_COUNT 按 1.5x B 大小调大。
        // [PITFALL] HEAVY_BLOCK_NUM=56 必须等于 AIV 核数，少了刷不干净，多了报错。
        //           950 的 AIV 核数典型为 48~56，按实际 npu-smi 查询为准。
        constexpr int64_t HEAVY_BLOCK_NUM = 56;
        constexpr int64_t BOOST_ELEM_COUNT = 10000LL * 10000LL;
        size_t boostSize = static_cast<size_t>(BOOST_ELEM_COUNT) * sizeof(uint16_t);
        GM_ADDR boostIn = nullptr;
        GM_ADDR boostOut = nullptr;
        ACL_CHECK(aclrtMalloc((void**)&boostIn, boostSize, ACL_MEM_MALLOC_HUGE_ONLY));
        ACL_CHECK(aclrtMalloc((void**)&boostOut, boostSize, ACL_MEM_MALLOC_HUGE_ONLY));
        std::vector<uint16_t> boostHost(BOOST_ELEM_COUNT, 0x3F00);
        ACL_CHECK(aclrtMemcpy(boostIn, boostSize, boostHost.data(), boostSize, ACL_MEMCPY_HOST_TO_DEVICE));
        int64_t boostBlockLen = (BOOST_ELEM_COUNT + HEAVY_BLOCK_NUM - 1) / HEAVY_BLOCK_NUM;

        constexpr int64_t CACHE_FLUSH_ELEM_COUNT = 128LL * 1024 * 1024;
        size_t cacheFlushSize = static_cast<size_t>(CACHE_FLUSH_ELEM_COUNT) * sizeof(uint16_t);
        GM_ADDR cacheFlush = nullptr;
        ACL_CHECK(aclrtMalloc((void**)&cacheFlush, cacheFlushSize, ACL_MEM_MALLOC_HUGE_ONLY));
        std::vector<uint16_t> cacheFlushHost(CACHE_FLUSH_ELEM_COUNT, 0x0000);
        ACL_CHECK(aclrtMemcpy(cacheFlush, cacheFlushSize, cacheFlushHost.data(), cacheFlushSize, ACL_MEMCPY_HOST_TO_DEVICE));
        int64_t cacheFlushBlockLen = (CACHE_FLUSH_ELEM_COUNT + HEAVY_BLOCK_NUM - 1) / HEAVY_BLOCK_NUM;

        // 阶段 1：去 host bound（100 次 exp boost，最后 sync 一次）
        // constexpr int WARMUP_EXP_COUNT = 100;
        // for (int i = 0; i < WARMUP_EXP_COUNT; ++i) {
        //     heavy_exp_kernel<<<HEAVY_BLOCK_NUM, nullptr, stream>>>(
        //         boostIn, boostOut, BOOST_ELEM_COUNT, boostBlockLen);
        // }
        // ACL_CHECK(aclrtSynchronizeStream(stream));

        // 阶段 2：性能测试主循环（10 轮，每轮 1 次 cache_flush.add_(1) + 1 次 main kernel）。
        // msprof op 会以 kernel 为单位采集，每轮 main kernel 都会产生一条 OpBasicInfo 记录。
        // [PITFALL] 顺序不能反：必须先 heavy_add_kernel 刷 L2，再跑 main kernel。
        //           反了会让 main kernel 自己的 B 数据被 flush 掉，反过来污染指标。
        // [PITFALL] 每轮都要 aclrtSynchronizeStream，确保 heavy_add 与 main 不重叠；
        //           重叠了会让 heavy_add 的开销被算到 main kernel 头上。
        constexpr int PERF_LOOP_COUNT = 10;
        for (int i = 0; i < PERF_LOOP_COUNT; ++i) {
            heavy_add_kernel<<<HEAVY_BLOCK_NUM, nullptr, stream>>>(
                cacheFlush, CACHE_FLUSH_ELEM_COUNT, cacheFlushBlockLen);
            AllToAllQuantMatmulKernelE4M3E4M3<<<tilingData.tileQbmmTilingData.usedCoreNum, nullptr, stream>>>(
                shmemSpace, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
            ACL_CHECK(aclrtSynchronizeStream(stream));
        }

        aclrtFree(boostIn);
        aclrtFree(boostOut);
        aclrtFree(cacheFlush);
    }

    aclshmem_free(shmemSpace);
    aclrtFree(deviceA);
    aclrtFree(deviceScaleA);
    aclrtFree(deviceB);
    aclrtFree(deviceScaleB);
    aclrtFree(deviceOutput);

    ACL_CHECK(aclshmem_finalize());
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}

int main(int argc, char* argv[]) {
    int m, k, n, rankNum;
    std::string mode;
    try {
        parseArguments(argc, argv, m, k, n, rankNum, mode);
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return -1;
    }

    INFO_LOG("Master (PID=%d) will fork %d processes (mode=%s)", getpid(), rankNum, mode.c_str());

    // [PITFALL] fork 后子进程必须 exit(ret)，不能 return —— return 会回到 main 上层，
    //           导致子进程继续 fork，进程数指数爆炸。
    // [PITFALL] 所有 rank 必须同时启动（几乎同时）；启动间隔过大会导致 SHMEM 握手超时。
    //           大规模 rankNum 时可用进程组/信号量同步启动。
    std::vector<pid_t> pids(rankNum);
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = fork();

        if (pid < 0) {
            ERROR_LOG("Fork failed for rank %d", rankId);
            exit(-1);
        } else if (pid == 0) {
            int ret = runAllToAllMatmul(rankNum, rankId, m, k, n, mode);
            exit(ret);
        } else {
            pids[rankId] = pid;
            INFO_LOG("Forked Rank %d -> PID %d", rankId, pid);
        }
    }

    int status;
    bool all_success = true;
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = pids[rankId];
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            all_success = false;
            ERROR_LOG("Worker PID %d failed", pid);
        }
    }

    std::cout << "All workers finished. Status: " << (all_success ? "SUCCESS" : "FAILURE") << std::endl;
    return all_success ? 0 : -1;
}
