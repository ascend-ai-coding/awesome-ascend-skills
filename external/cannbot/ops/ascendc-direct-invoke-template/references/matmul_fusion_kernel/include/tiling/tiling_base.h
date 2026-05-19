/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// [PATTERN] Tiling Base — 基础 Tiling 计算逻辑，通常不需要修改

/*!
 * \file tiling_base.h
 * \brief Base class that drives the common MX tiling workflow.
 */

#ifndef TILING_BASE_H
#define TILING_BASE_H

#include <cstdint>
#include <cstdio>

#include "host_utils/common_utils.h"
#include "tiling_common.h"
#include "tiling_data.h"

template <DataType aDataType, DataType bDataType>
class QuantMatmulTilingBase {
public:
    QuantMatmulTilingBase() = default;
    virtual ~QuantMatmulTilingBase() = default;

    void GetTilingData(uint64_t m, uint64_t n, uint64_t k, QuantMatmulTilingData& tilingData)
    {
        // Clear the cached state so one tiling object can safely be reused for
        // multiple shapes without leaking the previous decision.
        args_ = {};
        platformInfo_ = {};
        runInfo_ = {};

        // The public entry only prepares the shared state. Path-specific tiling
        // is delegated to one virtual operation so the derived class owns the
        // complete strategy-specific decision flow.
        InitCompileInfo();
        InitShapeArgs(m, n, k);
        DoOpTiling(tilingData);
        PrintTilingData(tilingData);
    }

protected:
    QuantMatmulArgs args_{};
    QuantMatmulPlatformInfo platformInfo_{};
    QuantMatmulRunInfo runInfo_{};

    virtual const char* TilingName() const = 0;

    virtual void DoOpTiling(QuantMatmulTilingData& tilingData) = 0;

private:
    void PrintTilingData(const QuantMatmulTilingData& tilingData) const
    {
        printf("[QuantMatmul Strategy]\n");
        printf("  strategy           : %s\n", TilingName());
        printf("[QuantMatmul Tiling Data]\n");
        printf("  usedCoreNum        : %u\n", tilingData.usedCoreNum);
        printf("  m                  : %u\n", tilingData.m);
        printf("  n                  : %u\n", tilingData.n);
        printf("  k                  : %u\n", tilingData.k);
        printf("  baseM              : %u\n", tilingData.baseM);
        printf("  baseN              : %u\n", tilingData.baseN);
        printf("  baseK              : %u\n", tilingData.baseK);
        printf("  scaleKL1           : %u\n", tilingData.scaleKL1);
        printf("  mTailTile          : %u\n", tilingData.mTailTile);
        printf("  nTailTile          : %u\n", tilingData.nTailTile);
        printf("  mBaseTailSplitCnt  : %u\n", tilingData.mBaseTailSplitCnt);
        printf("  nBaseTailSplitCnt  : %u\n", tilingData.nBaseTailSplitCnt);
        printf("  mTailMain          : %u\n", tilingData.mTailMain);
        printf("  nTailMain          : %u\n", tilingData.nTailMain);
        printf("  stepK              : %u\n", tilingData.stepK);
        printf("  nBufferNum         : %u\n", tilingData.nBufferNum);
        printf("  dbL0c              : %u\n", tilingData.dbL0c);
    }

    void InitCompileInfo()
    {
        // Query the platform once per tiling request so later decisions can
        // use the real cache sizes and core count of the current device.
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        platformInfo_.aicNum = ascendcPlatform->GetCoreNumAic();
        platformInfo_.aivNum = ascendcPlatform->GetCoreNumAiv();
        platformInfo_.socVersion = ascendcPlatform->GetSocVersion();
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::BT, platformInfo_.btSize);
    }

    void InitShapeArgs(uint64_t m, uint64_t n, uint64_t k)
    {
        args_.m = m;
        args_.n = n;
        args_.k = k;
    }
};

#endif // TILING_BASE_H
