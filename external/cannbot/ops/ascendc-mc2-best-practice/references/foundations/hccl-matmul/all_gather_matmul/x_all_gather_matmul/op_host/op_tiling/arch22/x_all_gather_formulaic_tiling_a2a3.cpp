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
 * \file x_all_gather_formulaic_tiling_a2a3.cpp
 * \brief Migrated verbatim from ops-transformer/mc2/x_all_gather_matmul/op_host/op_tiling/arch22/x_all_gather_formulaic_tiling_a2a3.cpp.
 *        Local include renamed to the all_gather_-prefixed header.
 */
#include "x_all_gather_formulaic_tiling_a2a3.h"

 void XAllGatherPlusMMA2A3::SetCommTimeFactor()
{
    // 通算并行时通信有膨胀，大K大N场景膨胀明显，做特殊处理
    bool medianMFlag = (clusterInfo_.mValue > SMALL_M) && (clusterInfo_.mValue <= MEDIAN_M);
    bool bwGrowthByUtil = matmulPerf_.cubeUtil_ < PART_L2_UTIL;
    bool bwGrowthByShape = clusterInfo_.kValue >= LARGE_K_BOUNDARY && clusterInfo_.nValue > LARGE_N_BOUNDARY;
    bool smallDim = rankDim_ <= MIN_COMM_RANKDIM || rankTileNum_ <= MatmulPerformance::SMALL_RANKTILE;
    bwGrowthByUtil = bwGrowthByUtil && !smallDim;
    bwGrowthByShape = bwGrowthByShape && !smallDim;
    if (bwGrowthByUtil) { // 0.85 is max cube utilizaion rate
        if (!medianMFlag) {
            tilingM_.SetMaxTileCnt(8U); // 8 is max tile cnt of M
        }
        commPerf_.ChangeCommTimeFactorByDivision(gatherLargerNKCommGrowRatio1); // 3x time of factor
    } else if (bwGrowthByShape) {
        if (!medianMFlag) {
            tilingM_.SetMaxTileCnt(8); // 8 is max tile cnt of M
        }
        commPerf_.ChangeCommTimeFactorByDivision(gatherLargerNKCommGrowRatio2); // 1.5x time of factor
    }
    commPerf_.ChangeCommTimeFactorByDivision(commGrowRatio); // 1.15x time of factor
    if (clusterInfo_.socType == SocVersion::SOC910_93) {
        commPerf_.ChangeCommTimeFactorByDivision(0.6);   // 0.6x time of factor
    }
}

bool XAllGatherPlusMMA2A3::GetAllowMoreCuts(bool smallMFlag)
{
    bool allowMoreCuts = (!tilingM_.cutRes.shortTileAtBack && smallMFlag) ||
        (strongTpBound_ && clusterInfo_.socType == SocVersion::SOC910_B);

    return allowMoreCuts;
}
