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
 * \file moe_dispatch_base_compat.h
 * \brief
 */
#ifndef MOE_DISPATCH_BASE_COMPAT_H
#define MOE_DISPATCH_BASE_COMPAT_H

#include "basic_api/kernel_basic_intf.h"
#include "tiling_data.h"

namespace Mc2Kernel {

using namespace AscendC;

constexpr uint64_t A5_MTE_STATE_WIN_SIZE = 1024UL * 1024UL;

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
};

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
using HcclOpParam = MoeDispatchImpl::HcclA5OpResParam;

__aicore__ inline uint32_t GetRankId(__gm__ HcclOpParam *winContext)
{
    return winContext->rankId;
}

__aicore__ inline uint32_t GetRankDim(__gm__ HcclOpParam *winContext)
{
    return winContext->rankDim;
}

__aicore__ inline uint64_t GetWinSize(__gm__ HcclOpParam *winContext)
{
    return winContext->winSize;
}

__aicore__ inline GM_ADDR GetStatusDataSpaceGm(__gm__ HcclOpParam *winContext)
{
    return (GM_ADDR)(winContext->windowsIn[winContext->rankId]);
}

__aicore__ inline GM_ADDR GetBaseWindAddrByRankId(__gm__ HcclOpParam *winContext,
    const int32_t rankId, const int32_t curRankId)
{
    return (GM_ADDR)(winContext->windowsIn[rankId] + A5_MTE_STATE_WIN_SIZE);
}

__aicore__ inline GM_ADDR GetBaseWindStateAddrByRankId(__gm__ HcclOpParam *winContext,
    const int32_t rankId, const int32_t curRankId)
{
    return (GM_ADDR)(winContext->windowsIn[rankId]);
}
#else
using HcclOpParam = MoeDispatchImpl::HcclA3OpResParam;

__aicore__ inline uint32_t GetRankId(__gm__ HcclOpParam *winContext)
{
    return winContext->localUsrRankId;
}

__aicore__ inline uint32_t GetRankDim(__gm__ HcclOpParam *winContext)
{
    return winContext->rankSize;
}

__aicore__ inline uint64_t GetWinSize(__gm__ HcclOpParam *winContext)
{
    return winContext->winSize;
}

__aicore__ inline GM_ADDR GetStatusDataSpaceGm(__gm__ HcclOpParam *winContext)
{
    return (GM_ADDR)(winContext->localWindowsExp);
}

__aicore__ inline GM_ADDR GetBaseWindAddrByRankId(__gm__ HcclOpParam *winContext,
    const int32_t rankId, const int32_t curRankId)
{
    if (rankId == curRankId) {
        return (GM_ADDR)(winContext->localWindowsIn);
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext->remoteRes[rankId].nextDevicePtr))->windowsIn);
}

__aicore__ inline GM_ADDR GetBaseWindStateAddrByRankId(__gm__ HcclOpParam *winContext,
    const int32_t rankId, const int32_t curRankId)
{
    if (rankId == curRankId) {
        return (GM_ADDR)(winContext->localWindowsExp);
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext->remoteRes[rankId].nextDevicePtr))->windowsExp);
}
#endif

} // namespace Mc2Kernel

#endif // MOE_DISPATCH_BASE_COMPAT_H