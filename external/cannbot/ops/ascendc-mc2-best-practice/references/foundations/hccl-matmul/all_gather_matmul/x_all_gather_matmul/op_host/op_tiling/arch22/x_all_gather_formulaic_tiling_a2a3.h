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
 * \file x_all_gather_formulaic_tiling_a2a3.h
 * \brief Migrated verbatim from ops-transformer/mc2/x_all_gather_matmul/op_host/op_tiling/arch22/x_all_gather_formulaic_tiling_a2a3.h.
 *        Local include renamed to the all_gather_-prefixed base file. The arch22 (A2/A3) default SocVersion is
 *        SOC910_B; the A3 sub-branch inside SetCommTimeFactor is retained verbatim (dead code for an
 *        A2-only build, but trimming it risks altering the shared tiling behaviour).
 */
#ifndef __ALL_GATHER_FORMULAIC_TILING_A2A3_H__
#define __ALL_GATHER_FORMULAIC_TILING_A2A3_H__

#include "../x_all_gather_formulaic_tiling.h"

class XAllGatherPlusMMA2A3 : public XAllGatherPlusMM {
public:
    // Constructor
    explicit XAllGatherPlusMMA2A3(const mc2tiling::TilingArgs& args, uint32_t inputRankDim, KernelType inputKernelType,
                             SocVersion inputSocVersion = SocVersion::SOC910_B)
        : XAllGatherPlusMM(args, inputRankDim, inputKernelType, inputSocVersion)
    {
        commPerf_.SetCommShapeLen(clusterInfo_.kValue);
        commPerf_.SetCommDTypeSize(clusterInfo_.inMatrixADtypeSize);
        rankTileNum_ = commPerf_.GetRankTileNum();
        tilingM_.SetMinLenByMax(commPerf_.GetLinearThresholdLen());

        if (clusterInfo_.socType == SocVersion::SOC910_B) {
            tilingM_.SetMinLenByMax(matmulPerf_.GetLinearThresholdLen(rankDim_));
        } else {
            tilingM_.SetMinLenByMax(matmulPerf_.GetLinearThresholdLen(rankTileNum_));
        }
    }

private:
    void SetCommTimeFactor() override;
    bool GetAllowMoreCuts(bool smallMFlag) override;
};

#endif //__ALL_GATHER_FORMULAIC_TILING_A2A3_H__
