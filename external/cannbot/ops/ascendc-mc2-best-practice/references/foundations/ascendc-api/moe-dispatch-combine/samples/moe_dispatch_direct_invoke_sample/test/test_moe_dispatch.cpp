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
 * \file test_moe_dispatch.cpp
 * \brief
 */
#include <thread>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <getopt.h>
#include <fstream>
#include <unistd.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "hccl/hccl_comm.h"
#include "hccl/hccl.h"
#include "acl/acl.h"
#include "hccl/hcom.h"
#include "securec.h"
#include "tiling/hccl/hccl_tiling.h"
#include "tiling_data.h"

using namespace std;
using namespace MoeDispatchImpl;

// Global mutex for HcclAllocComResourceByTiling
static std::mutex g_hcclAllocMutex;

// Synchronization for kernel launch
static std::mutex g_syncMutex;
static std::condition_variable g_syncCond;
static std::atomic<int> g_initDoneCount(0);
static std::atomic<bool> g_canLaunch(false);

/**
 * @brief 简化的错误检查宏
 *
 * 用于 host 侧 ACL/HCCL 接口调用后的快速返回。
 */
#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int gRankSize = 2;
int64_t gBs = 4;
int64_t gH = 16;
int64_t gHcclBufferSize = 16; // 需要根据实际合理分配 win 区大小
std::string gOutputDir = ".";

/**
 * @brief 每个 rank 启动时需要携带的上下文
 */
struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
    void *mc2ContextAddr;
    void *tilingAddr;
};

int LaunchOneThread(Args &args);

/**
 * @brief Host 侧 launch wrapper
 *
 * 这个接口对应 AscendC 生成的 host stub，负责把原始指针和 stream
 * 传给 `<<<>>>` 直调 kernel。
 */
extern "C" void moe_dispatch_demo(uint32_t blockDim, void* stream,
    uint8_t* mc2Context, uint8_t* x, uint8_t* expertIds, uint8_t* expandX, uint8_t* expandIdx,
    uint8_t* expertTokenNums, uint8_t* epRecvCounts, uint8_t* workspaceGM,
    uint8_t* tilingGM);

extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);

/**
 * @brief 计算 shape 的元素总数
 */
int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

/**
 * @brief 将 float 转成 fp16 的简单工具函数
 *
 * 当前 sample 只在 host 侧造测试输入时使用，不依赖额外数学库。
 */
uint16_t FloatToFP16(float f)
{
    uint32_t x;
    /* use secure memcpy_s (provided by securec.h) to avoid unsafe memcpy usage flagged by codecheck */
    if (memcpy_s(&x, sizeof(x), &f, sizeof(f)) != 0) {
        /* fallback: ensure deterministic behavior on failure */
        x = 0;
    }
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exponent = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (x >> 13) & 0x3FF;
    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x400) >> (1 - exponent);
        return static_cast<uint16_t>(sign | mantissa);
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
}

/**
 * @brief 将结果二进制写到磁盘，便于脚本离线校验
 */
int WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        return -1;
    }
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) {
        return -1;
    }
    if (flock(fd, LOCK_EX) == -1) {
        close(fd);
        return -1;
    }
    ssize_t written = write(fd, static_cast<const char *>(buffer), size);
    flock(fd, LOCK_UN);
    close(fd);
    return (written == static_cast<ssize_t>(size)) ? ACL_SUCCESS : -1;
}

/**
 * @brief 解析命令行参数并更新全局配置
 */
void ParseCommandLine(int argc, char *argv[])
{
    static const struct option longOptions[] = {
        {"rank_size", 1, 0, 's'},
        {"bs", 1, 0, 'b'},
        {"h", 1, 0, 'h'},
        {"output_dir", 1, 0, 'o'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "s:b:h:o:", longOptions, nullptr)) != -1) {
        switch (opt) {
            case 's': gRankSize = atoi(optarg); break;
            case 'b': gBs = atoll(optarg); break;
            case 'h': gH = atoll(optarg); break;
            case 'o': gOutputDir = optarg; break;
            default: break;
        }
    }
}

/**
 * @brief 初始化 ACL/HCCL 运行时，以及每个 rank 的 device/context/stream/comm
 */
int InitializeRuntime(std::vector<int32_t> &devices,
    std::vector<aclrtContext> &contexts,
    std::vector<aclrtStream> &streams,
    std::vector<HcclComm> &comms)
{
    bool aclInited = false;
    int initializedRanks = 0;

    printf("[MAIN] Calling aclInit...\n");
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclInited = true;
    printf("[MAIN] aclInit success\n");

    for (int rank = 0; rank < gRankSize; ++rank) {
        devices[rank] = rank;
        printf("[MAIN] Setting device %d...\n", rank);
        ret = aclrtSetDevice(rank);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
        printf("[MAIN] Creating context for device %d...\n", rank);
        ret = aclrtCreateContext(&contexts[rank], rank);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
        printf("[MAIN] Creating stream for device %d...\n", rank);
        ret = aclrtCreateStream(&streams[rank]);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
        ++initializedRanks;
    }

    // 按 HcclCommInitAll 文档，单机单进程场景下由一个进程统一创建多卡通信域。
    printf("[MAIN] Calling HcclCommInitAll...\n");
    ret = HcclCommInitAll(static_cast<uint32_t>(gRankSize), devices.data(), comms.data());
    if (ret != ACL_SUCCESS) {
        goto cleanup;
    }
    printf("[MAIN] HcclCommInitAll success\n");
    return ACL_SUCCESS;

cleanup:
    for (int rank = initializedRanks - 1; rank >= 0; --rank) {
        aclrtSetCurrentContext(contexts[rank]);
        aclrtDestroyStream(streams[rank]);
        aclrtDestroyContext(contexts[rank]);
        aclrtResetDevice(rank);
    }
    if (aclInited) {
        aclFinalize();
    }
    return ret;
}

/**
 * @brief 启动所有 rank 线程并等待结束
 */
int LaunchRankThreads(std::vector<Args> &argsList, std::vector<int> &threadRets)
{
    std::vector<std::thread> threads;
    threads.reserve(gRankSize);
    for (int rank = 0; rank < gRankSize; ++rank) {
        printf("[MAIN] Starting thread for rank %d\n", rank);
        threads.emplace_back([&argsList, &threadRets, rank]() {
            printf("[Thread %d] Starting\n", rank);
            threadRets[rank] = LaunchOneThread(argsList[rank]);
            printf("[Thread %d] Done, ret=%d\n", rank, threadRets[rank]);
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    for (int rank = 0; rank < gRankSize; ++rank) {
        CHECK_RET(threadRets[rank] == ACL_SUCCESS, return threadRets[rank]);
    }
    return ACL_SUCCESS;
}

/**
 * @brief 按初始化逆序释放运行时资源
 */
void CleanupRuntime(const std::vector<aclrtContext> &contexts, const std::vector<aclrtStream> &streams)
{
    for (int rank = 0; rank < gRankSize; ++rank) {
        aclrtSetCurrentContext(contexts[rank]);
        aclrtDestroyStream(streams[rank]);
        aclrtDestroyContext(contexts[rank]);
        aclrtResetDevice(rank);
    }
    aclFinalize();
}

/**
 * @brief 构造并下发 dispatch 所需的 tiling 数据
 *
 * Host 侧只负责提供：
 * - 当前测试的 bs / h
 * - 当前 EP 通信域卡数
 * - 最大接收 token 数
 * - Mc2CcTilingConfig 生成的通信 tiling
 */
int CreateTilingDataAndContext(const char* hcomName, aclrtStream stream, void **deviceTilingAddr, void **deviceContextAddr, uint32_t usedCores)
{
    printf("[CreateTilingDataAndContext] Start, hcomName=%s\n", hcomName);
    MoeDispatchTilingData *tilingData = new MoeDispatchTilingData();
    if (tilingData == nullptr) {
        printf("[CreateTilingDataAndContext] Failed to allocate tilingData\n");
        return -1;
    }
    *tilingData = {};
    tilingData->tilingInfo.bs = gBs;
    tilingData->tilingInfo.h = gH;
    tilingData->tilingInfo.epWorldSize = gRankSize;
    tilingData->tilingInfo.aivNum = usedCores;
    tilingData->tilingInfo.maxRecvTokens = gBs * gRankSize;
    tilingData->tilingInfo.totalWinSize = gHcclBufferSize;
    tilingData->tilingInfo.topK = 1;
    printf("[CreateTilingDataAndContext] bs=%ld, h=%ld, rankSize=%d, maxRecvTokens=%ld, topK=%ld\n",
        gBs, gH, gRankSize, gBs * gRankSize, tilingData->tilingInfo.topK);

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(hcomName, 6, "AlltoAll=level0:fullmesh;level1:pairwise");
    mc2CcTilingConfig.SetCommEngine(3);

    int ret = mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    CHECK_RET(ret == ACL_SUCCESS, delete tilingData; return ret);
    ret = mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    CHECK_RET(ret == ACL_SUCCESS, delete tilingData; return ret);

    int tilingSize = sizeof(MoeDispatchTilingData);
    ret = aclrtMalloc(deviceTilingAddr, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, delete tilingData; return ret);
    ret = aclrtMemcpy(*deviceTilingAddr, tilingSize, tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, delete tilingData; return ret);

    HcclComm commHandle;
    ret = HcomGetCommHandleByGroup(hcomName, &commHandle);
    CHECK_RET(ret == ACL_SUCCESS, delete tilingData; return ret);

    void *mc2Context = nullptr;
    ret = HcclAllocComResourceByTiling(commHandle, stream, tilingData, &mc2Context);
    CHECK_RET(ret == ACL_SUCCESS, delete tilingData; return ret);
    CHECK_RET(mc2Context != nullptr, delete tilingData; return -1);

    // Copy A5 mc2Context from device to host for debugging
    constexpr size_t mc2ContextSize = sizeof(MoeDispatchImpl::HcclA5OpResParam);
    MoeDispatchImpl::HcclA5OpResParam *hostContext = new MoeDispatchImpl::HcclA5OpResParam();
    ret = aclrtMemcpy(hostContext, mc2ContextSize, mc2Context, mc2ContextSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
        printf("[CreateTilingDataAndContext] mc2Context content:\n");
        printf("  rankId=%u, rankDim=%u\n", hostContext->rankId, hostContext->rankDim);
        printf("  winSize=0x%lx\n", hostContext->winSize);
        printf("  windowsIn[0]=0x%lx, windowsIn[1]=0x%lx\n", hostContext->windowsIn[0], hostContext->windowsIn[1]);
        printf("  windowsOut[0]=0x%lx, windowsOut[1]=0x%lx\n", hostContext->windowsOut[0], hostContext->windowsOut[1]);
    }
    delete hostContext;

    // HcclAllocComResourceByTiling returns a device address directly
    // Use it as-is for kernel, no copy needed
    *deviceContextAddr = mc2Context;

    delete tilingData;
    return ACL_SUCCESS;
}

/**
 * @brief 单个 rank 执行 dispatch 时用到的 host/device 缓冲区
 */
struct LaunchBuffers {
    void *xDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *expandXDeviceAddr = nullptr;
    void *expandIdxDeviceAddr = nullptr;
    void *expertTokenNumsDeviceAddr = nullptr;
    void *epRecvCountsDeviceAddr = nullptr;
    std::string prefix;
    int64_t maxRecvTokens = 0;
    int64_t xElems = 0;
    int64_t expandXElems = 0;
    int64_t expandIdxElems = 0;
    std::vector<uint16_t> xHost;
    std::vector<int32_t> expertIdsHost;
    std::vector<uint16_t> expandXHost;
    std::vector<int32_t> expandIdxHost;
    std::vector<int64_t> expertTokenNumsHost;
    std::vector<int32_t> epRecvCountsHost;
};

/**
 * @brief 释放单个 rank 的 dispatch 资源
 */
void ReleaseLaunchResources(const Args &args, const LaunchBuffers &buffers, void *tilingAddr)
{
    printf("[LaunchOneThread] Cleaning up for rank %d\n", args.rankId);
    HcclCommDestroy(args.hcclComm);
    aclrtFree(buffers.xDeviceAddr);
    aclrtFree(buffers.expertIdsDeviceAddr);
    aclrtFree(buffers.expandXDeviceAddr);
    aclrtFree(buffers.expandIdxDeviceAddr);
    aclrtFree(buffers.expertTokenNumsDeviceAddr);
    aclrtFree(buffers.epRecvCountsDeviceAddr);
    // mc2ContextAddr is managed by HCCL, do not free it
    aclrtFree(tilingAddr);
}

/**
 * @brief 分配单个 rank 的 device buffer
 */
int AllocateLaunchResources(LaunchBuffers &buffers)
{
    int ret = aclrtMalloc(&buffers.xDeviceAddr, buffers.xElems * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMalloc(&buffers.expertIdsDeviceAddr, gBs * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMalloc(&buffers.expandXDeviceAddr, buffers.expandXElems * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMalloc(&buffers.expandIdxDeviceAddr, buffers.expandIdxElems * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMalloc(&buffers.expertTokenNumsDeviceAddr, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMalloc(&buffers.epRecvCountsDeviceAddr, gRankSize * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    return ACL_SUCCESS;
}

/**
 * @brief 生成单个 rank 的测试输入并初始化输出缓冲区
 * 注意：同一 token 不能被重复发送给同一个 expert。
 */
void PrepareLaunchHostBuffers(const Args &args, LaunchBuffers &buffers)
{
    buffers.maxRecvTokens = gBs * gRankSize;
    buffers.xElems = gBs * gH;
    buffers.expandXElems = buffers.maxRecvTokens * gH;
    buffers.expandIdxElems = buffers.maxRecvTokens * 3;

    buffers.xHost.resize(buffers.xElems);
    for (int64_t token = 0; token < gBs; ++token) {
        for (int64_t col = 0; col < gH; ++col) {
            buffers.xHost[token * gH + col] = FloatToFP16(static_cast<float>(args.rankId * 100 + token * 10 + col));
        }
    }
    buffers.expertIdsHost.resize(gBs);
    for (int64_t token = 0; token < gBs; ++token) {
        buffers.expertIdsHost[token] = static_cast<int32_t>((args.rankId + token) % gRankSize);
    }

    buffers.expandXHost.resize(buffers.expandXElems, 0);
    buffers.expandIdxHost.resize(buffers.expandIdxElems, 0);
    buffers.expertTokenNumsHost.resize(1, 0);
    buffers.epRecvCountsHost.resize(gRankSize, 0);
}

/**
 * @brief 将 host 数据拷贝到 device
 */
int CopyLaunchInputsToDevice(const LaunchBuffers &buffers)
{
    int ret = aclrtMemcpy(buffers.xDeviceAddr, buffers.xElems * sizeof(uint16_t),
        buffers.xHost.data(), buffers.xElems * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(buffers.expertIdsDeviceAddr, gBs * sizeof(int32_t),
        buffers.expertIdsHost.data(), gBs * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(buffers.expandXDeviceAddr, buffers.expandXElems * sizeof(uint16_t),
        buffers.expandXHost.data(), buffers.expandXElems * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(buffers.expandIdxDeviceAddr, buffers.expandIdxElems * sizeof(int32_t),
        buffers.expandIdxHost.data(), buffers.expandIdxElems * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(buffers.expertTokenNumsDeviceAddr, sizeof(int64_t),
        buffers.expertTokenNumsHost.data(), sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(buffers.epRecvCountsDeviceAddr, gRankSize * sizeof(int32_t),
        buffers.epRecvCountsHost.data(), gRankSize * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    return ret;
}

/**
 * @brief 等待所有线程就绪后再启动 kernel
 */
void WaitForLaunchBarrier(uint32_t rankId)
{
    printf("[LaunchOneThread] Rank %u initialization done, waiting for barrier\n", rankId);
    g_initDoneCount.fetch_add(1);
    {
        std::unique_lock<std::mutex> lock(g_syncMutex);
        g_syncCond.wait(lock, []{ return g_initDoneCount.load() == gRankSize; });
    }
    if (g_initDoneCount.load() == gRankSize) {
        g_canLaunch.store(true);
        g_syncCond.notify_all();
    }
    {
        std::unique_lock<std::mutex> lock(g_syncMutex);
        g_syncCond.wait(lock, []{ return g_canLaunch.load(); });
    }
}

/**
 * @brief 同步 stream 并回拷输出到 host
 */
int CollectLaunchOutputs(const Args &args, LaunchBuffers &buffers)
{
    int ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    printf("[LaunchOneThread] aclrtSynchronizeStreamWithTimeout for rank %d, ret=%d\n", args.rankId, ret);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    ret = aclrtMemcpy(buffers.expandXHost.data(), buffers.expandXElems * sizeof(uint16_t),
        buffers.expandXDeviceAddr, buffers.expandXElems * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(buffers.expandIdxHost.data(), buffers.expandIdxElems * sizeof(int32_t),
        buffers.expandIdxDeviceAddr, buffers.expandIdxElems * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(buffers.expertTokenNumsHost.data(), sizeof(int64_t),
        buffers.expertTokenNumsDeviceAddr, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(buffers.epRecvCountsHost.data(), gRankSize * sizeof(int32_t),
        buffers.epRecvCountsDeviceAddr, gRankSize * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    return ret;
}

/**
 * @brief 落盘当前 rank 的输入与输出结果
 */
void WriteLaunchArtifacts(const Args &args, const LaunchBuffers &buffers)
{
    std::string prefix = gOutputDir + "/";
    WriteFile(prefix + "input_rank" + std::to_string(args.rankId) + ".bin", buffers.xHost.data(), buffers.xElems * sizeof(uint16_t));
    WriteFile(prefix + "expert_ids_rank" + std::to_string(args.rankId) + ".bin", buffers.expertIdsHost.data(), gBs * sizeof(int32_t));
    WriteFile(prefix + "expand_x_rank" + std::to_string(args.rankId) + ".bin", buffers.expandXHost.data(), buffers.expandXElems * sizeof(uint16_t));
    WriteFile(prefix + "expand_idx_rank" + std::to_string(args.rankId) + ".bin", buffers.expandIdxHost.data(), buffers.expandIdxElems * sizeof(int32_t));
    WriteFile(prefix + "expert_token_nums_rank" + std::to_string(args.rankId) + ".bin", buffers.expertTokenNumsHost.data(), sizeof(int64_t));
    WriteFile(prefix + "ep_recv_counts_rank" + std::to_string(args.rankId) + ".bin", buffers.epRecvCountsHost.data(), gRankSize * sizeof(int32_t));

    std::ofstream summary(prefix + "summary_rank" + std::to_string(args.rankId) + ".txt");
    summary << "rank=" << args.rankId << "\n";
    summary << "bs=" << gBs << " h=" << gH << " rank_size=" << gRankSize << "\n";
    summary << "expert_token_nums=" << buffers.expertTokenNumsHost[0] << "\n";
}

/**
 * @brief 单个 rank 的完整执行流程
 *
 * 步骤：
 * 1. 绑定当前设备 context 与 stream
 * 2. 基于当前 rank 对应的 HCCL comm 名构造 tiling
 * 3. 申请输入/输出 device buffer
 * 4. 生成本 rank 的测试输入与路由 expertIds
 * 5. 调用 kernel launch wrapper 执行 dispatch
 * 6. 将输出回拷到 host 并 dump 到文件，供验证脚本读取
 */
int LaunchOneThread(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    LaunchBuffers buffers;
    void *mc2ContextAddr = nullptr;
    void *tilingAddr = nullptr;
    
    // 根据实际需要分配核数，当前 sample 为单核处理所有数据，生成新算子时需要根据需求调整分核策略和对应的核数。
    uint32_t usedCores = 1;
    // 构造 kernel 侧需要的通信/shape tiling 数据，以及 host 侧分配好的 mc2 context。
    ret = CreateTilingDataAndContext(hcomName, args.stream, &tilingAddr, &mc2ContextAddr, usedCores);
    if (ret != ACL_SUCCESS) {
        printf("[LaunchOneThread] CreateTilingDataAndContext failed, ret=%d\n", ret);
        goto cleanup;
    }

    PrepareLaunchHostBuffers(args, buffers);
    ret = AllocateLaunchResources(buffers);
    if (ret != ACL_SUCCESS) {
        printf("[LaunchOneThread] device allocation failed, ret=%d\n", ret);
        goto cleanup;
    }

    ret = CopyLaunchInputsToDevice(buffers);
    if (ret != ACL_SUCCESS) {
        printf("[LaunchOneThread] host to device copy failed, ret=%d\n", ret);
        goto cleanup;
    }

    // 通过 host wrapper 直接 launch 单算子 kernel
    
    // 同步点：等待所有线程都完成初始化
    WaitForLaunchBarrier(args.rankId);
    printf("[LaunchOneThread] Rank %d launching kernel\n", args.rankId);
    
    moe_dispatch_demo(usedCores, args.stream,
        (uint8_t*)mc2ContextAddr,
        (uint8_t*)buffers.xDeviceAddr,
        (uint8_t*)buffers.expertIdsDeviceAddr,
        (uint8_t*)buffers.expandXDeviceAddr,
        (uint8_t*)buffers.expandIdxDeviceAddr,
        (uint8_t*)buffers.expertTokenNumsDeviceAddr,
        (uint8_t*)buffers.epRecvCountsDeviceAddr,
        nullptr,
        (uint8_t*)tilingAddr);

    ret = CollectLaunchOutputs(args, buffers);
    if (ret != ACL_SUCCESS) {
        printf("[LaunchOneThread] collect outputs failed, ret=%d\n", ret);
        goto cleanup;
    }

    WriteLaunchArtifacts(args, buffers);

cleanup:
    ReleaseLaunchResources(args, buffers, tilingAddr);

    return ret;
}

/**
 * @brief 测试主入口
 *
 * Host 侧流程：
 * 1. 解析命令行参数
 * 2. 在单进程内初始化多卡 ACL context / stream
 * 3. 通过 HcclCommInitAll 一次性创建多卡 HCCL comm
 * 4. 为每张卡启动一个线程执行 dispatch 测试
 * 5. 清理 HCCL/ACL 资源
 */
int main(int argc, char *argv[])
{
    printf("[MAIN] Starting moe_dispatch test\n");
    ParseCommandLine(argc, argv);
    printf("[MAIN] gRankSize=%d, gBs=%ld, gH=%ld\n", gRankSize, gBs, gH);

    int ret = ACL_SUCCESS;

    std::vector<int32_t> devices(gRankSize);
    std::vector<aclrtContext> contexts(gRankSize);
    std::vector<aclrtStream> streams(gRankSize);
    std::vector<HcclComm> comms(gRankSize);
    std::vector<Args> argsList(gRankSize);
    std::vector<int> threadRets(gRankSize, ACL_SUCCESS);

    ret = InitializeRuntime(devices, contexts, streams, comms);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    for (int rank = 0; rank < gRankSize; ++rank) {
        argsList[rank].rankId = static_cast<uint32_t>(rank);
        argsList[rank].hcclComm = comms[rank];
        argsList[rank].stream = streams[rank];
        argsList[rank].context = contexts[rank];
    }

    ret = LaunchRankThreads(argsList, threadRets);
    CleanupRuntime(contexts, streams);
    return ret;
}