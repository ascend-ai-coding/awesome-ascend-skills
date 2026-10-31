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
 * \file x_all_gather_formulaic_tiling.cpp
 * \brief Migrated verbatim from ops-transformer/mc2/x_all_gather_matmul/op_host/op_tiling/x_all_gather_formulaic_tiling.cpp.
 *        Local include renamed to all_gather_-prefixed file; mc2_log.h resolves against the vendored mc2_common tree.
 */
#include "x_all_gather_formulaic_tiling.h"

#include <stdexcept>

void XAllGatherPlusMM::EstimateKernelTime()
{
    SetCommTimeFactor();

    // 预测计算、通信任务耗时
    double totalMatmulTime = EstimateTotalMatmulTime();
    double totalTpTime = EstimateTotalCommTime();
    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));
    double frontUtil = matmulPerf_.FindCubeUtilByL2Usage(clusterInfo_.mValue, 1); // Local部分matmul耗时预测
    if (frontUtil == 0.0) {
        throw std::invalid_argument("XAllGatherPlusMM::EstimateKernelTime: frontUtil is 0, invalid tiling baseM/baseN");
    }
    frontMMTime_ = matmulPerf_.MatmulTime(clusterInfo_.mValue, 1) * matmulPerf_.cubeUtil_ / frontUtil;
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;
    }

    // 根据通信、计算耗时比例，调整切分约束
    // 2x compute time
    strongTpBound_ = (totalTpTime > totalMatmulTime * 2U) && (clusterInfo_.kValue >= LARGE_K_BOUNDARY);
    // 2x matmulMinTileSize
    bool smallMFlag = (clusterInfo_.mValue < tilingM_.GetMinLen() * 2U) &&
                      (rankTileNum_ > MatmulPerformance::SMALL_RANKTILE);
    bool allowMoreCuts = GetAllowMoreCuts(smallMFlag);
    bool reduceAlignLen = clusterInfo_.nValue > SMALL_N_BOUNDARY && clusterInfo_.mValue <= TINY_M;
    if (allowMoreCuts) {
        if (reduceAlignLen) {
            tilingM_.SetAlignLength(tilingM_.GetAlignLength() / TWO);  // 0.5 is half of mAlignLen
        }
        tilingM_.SetMinLenByMin(tilingM_.GetAlignLength());
    }
    noCutFlag_ = totalTpTime < frontMMTime_ && clusterInfo_.mValue <= tilingM_.tileArgs.maxTileLen;
    if (clusterInfo_.kValue > HUGE_K_BOUNDARY) { // 特大shape切分后matmul性能好一点
        noCutFlag_ = false;
    }
}

void XAllGatherPlusMM::SelectTilingMethod()
{
    if (tilingM_.SetShortTileLen(noCutFlag_)) { // 如果shape太小就不切
        return;
    }
    // 流水配平，找到理论切分长度
    if (tilingM_.cutRes.shortTileAtBack) {
        bool smallFront = rankDim_ <= MIN_COMM_RANKDIM; // 2p时，front占了一半的计算量
        if (!smallFront) {
            tilingM_.cutRes.longTileLen = commPerf_.InverseCommTime(frontMMTime_);
        } else {
            ShortAtEndCalcBoundBalancing();
        }
    } else {
        ShortAtFrontCommBoundBalancing();
        if (strongTpBound_) {
            tilingM_.cutRes.longTileLen = std::min(tilingM_.cutRes.longTileLen, tilingM_.cutRes.shortTileLen);
        }
    }

    // 生成切分
    bool smallDimAlignUp = (rankDim_ <= MIN_COMM_RANKDIM) && tilingM_.cutRes.shortTileAtBack &&
        (clusterInfo_.nValue < SMALL_SHAPE_BAR || clusterInfo_.kValue < SMALL_SHAPE_BAR); // 2p，计算Bound，且N轴或者K轴很小
    bool goodLinearityShape = (clusterInfo_.kValue * clusterInfo_.nValue >= LARGE_NK_BAR_BASE * ONE_MBYTE);
    tilingM_.FitTileLengthDiscrete(smallDimAlignUp, goodLinearityShape, hasLocalAtFront_);
}
