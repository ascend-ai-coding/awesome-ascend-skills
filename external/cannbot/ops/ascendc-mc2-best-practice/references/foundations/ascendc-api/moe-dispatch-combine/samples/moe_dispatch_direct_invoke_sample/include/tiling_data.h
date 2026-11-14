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
 * \file tiling_data.h
 * \brief
 */
#ifndef MOE_DISPATCH_TILING_DATA_H
#define MOE_DISPATCH_TILING_DATA_H

#include <stdint.h>
#include <kernel_tiling/kernel_tiling.h>

namespace MoeDispatchImpl {

// 最大 rank 数量和 expert 数量（用于 kernel 栈空间分配）
constexpr static int64_t MAX_RANK_NUM = 64;
constexpr static int64_t MAX_EXPERT_NUM = 8;

// HCCL 相关常量
constexpr uint32_t HCCL_MTE_MAX_RANK_NUM = 64;
constexpr uint32_t AICPU_MAX_RANK_NUM = 128 * 1024;

/**
 * @brief HCCL A5 通信资源参数
 *
 * 可由 HcclAllocComResourceByTiling 返回，包含 window 地址等信息
 */
struct HcclA5OpResParam {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
    uint64_t winSize;
    uint64_t windowsIn[HCCL_MTE_MAX_RANK_NUM];
    uint64_t windowsOut[HCCL_MTE_MAX_RANK_NUM];
    uint64_t xnAddr;
    uint64_t ckeAddr;
    uint64_t msAddr;
    uint64_t msSize;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

/**
 * @brief HCCL A3 通信资源参数
 *
 * 可由 HcclAllocComResourceByTiling 返回，包含 window 地址等信息
 */
struct HcclA3OpResParam {
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM];
};

/**
 * @brief Tiling 信息结构体
 *
 * 包含 dispatch 算子所需的基本参数
 */
struct MoeDispatchTilingInfo {
    uint64_t bs;            // 每个 rank 的 token 数量
    uint64_t h;             // 每个 token 的特征维度
    uint64_t epWorldSize;   // EP 并行的 rank 数量
    uint64_t aivNum;        // 使用的 AIV 核数量
    uint64_t maxRecvTokens; // 最大接收 token 数量
    uint64_t totalWinSize;  // 总 window 大小
    uint64_t topK;          // 每个 token 路由的 expert 数量
};

/**
 * @brief 完整的 Tiling 数据结构体
 *
 * 包含 MC2 通信所需的 tiling 信息和算子参数
 */
struct MoeDispatchTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDispatchTilingInfo tilingInfo;
};

} // namespace MoeDispatchImpl

#endif // MOE_DISPATCH_TILING_DATA_H
