/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file x_all_gather_formulaic_tiling.h
 * \brief Migrated verbatim from ops-transformer/mc2/x_all_gather_matmul/op_host/op_tiling/x_all_gather_formulaic_tiling.h.
 *        hccl_formulaic_tiling.h resolves against the vendored mc2_common tree.
 */
#ifndef __ALL_GATHER_FORMULAIC_TILING_H__
#define __ALL_GATHER_FORMULAIC_TILING_H__

#pragma once
#include "op_host/op_tiling/hccl_formulaic_tiling.h"
constexpr uint64_t LARGE_K_BOUNDARY = 8192;
constexpr uint64_t LARGE_N_BOUNDARY = 5120;
constexpr uint64_t SMALL_N_BOUNDARY = 2048;
constexpr uint64_t TINY_M = 512;
constexpr uint64_t MEDIAN_M = 4096;
constexpr double gatherLargerNKCommGrowRatio1 = 3;
constexpr double gatherLargerNKCommGrowRatio2 = 1.5;
constexpr uint64_t HUGE_K_BOUNDARY = 32768;

class XAllGatherPlusMM : public OneCalcOneCommBase
{
public:
    double frontMMTime_ = 0;
    bool strongTpBound_ = false;
    bool hasLocalAtFront_ = true;  // local提前计算

// Constructor
    explicit XAllGatherPlusMM(const mc2tiling::TilingArgs& args, uint32_t inputRankDim, KernelType inputKernelType,
                            SocVersion inputSocVersion)
        : OneCalcOneCommBase(args, inputRankDim, inputKernelType, inputSocVersion)
    {
        commPerf_.SetCommShapeLen(clusterInfo_.kValue);
        commPerf_.SetCommDTypeSize(clusterInfo_.inMatrixADtypeSize);
        rankTileNum_ = commPerf_.GetRankTileNum();
        tilingM_.SetMinLenByMax(commPerf_.GetLinearThresholdLen());
    }
    void EstimateKernelTime() override;
    void SelectTilingMethod() override;

private:
    virtual void SetCommTimeFactor() = 0;
    virtual bool GetAllowMoreCuts(bool smallMFlag) = 0;
};

#endif //__ALL_GATHER_FORMULAIC_TILING_H__
