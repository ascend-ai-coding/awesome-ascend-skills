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
 * \file x_all_gather_matmul_tiling_a2a3.cpp
 * \brief A2-only (Ascend910B) tiling entry for XAllGatherMatmul. Migrated from
 *        ops-transformer/mc2/x_all_gather_matmul/op_host/op_tiling/arch22/x_all_gather_matmul_tiling_a2a3.cpp.
 *        The A3/doublering (SOC910_93) sub-branch has been REMOVED — this operator targets A2 only,
 *        where the topology is full-mesh -> SOC910_B; the A3 path (isA3=1 -> doublering -> SOC910_93)
 *        was runtime-dead on A2 and had no doublering kernel counterpart. The isA3/isStep fields were
 *        removed from XAllGatherSoc; the former A2/A3 ternaries below are collapsed to the A2 constant.
 */
#include "x_all_gather_formulaic_tiling_a2a3.h"
#include "x_all_gather_matmul_tiling_a2a3.h"

namespace optiling {
ge::graphStatus XAllGatherMatmulTilingA2A3::CheckValidRank(Mc2Tiling::XAllGatherMatmulTilingData* tilingData,
    const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK, gert::TilingContext *context,
    uint32_t rankSize)
{
    (void)tilingData;
    (void)context;
    // A2-only: validate rank size against the A2 rank set {2,4,8} (VALID_RANK key 0).
    auto it = std::find(VALID_RANK.at(0).begin(), VALID_RANK.at(0).end(), rankSize);
    if (it == VALID_RANK.at(0).end()) { return ge::GRAPH_FAILED; }
    return ge::GRAPH_SUCCESS;
}

void XAllGatherMatmulTilingA2A3::SetSocParam(Mc2Tiling::XAllGatherMatmulTilingData* tilingData, const char* group)
{
    (void)group;
    // A2-only: full-mesh, ND2NZ always on. isA3/isStep removed.
    tilingData->socParam.isND2NZ = 1U;
}

std::string XAllGatherMatmulTilingA2A3::GetAlgConfig(Mc2Tiling::XAllGatherMatmulTilingData* tilingData)
{
    (void)tilingData;
    return "AllGather=level0:fullmesh";
}

CutResult XAllGatherMatmulTilingA2A3::GetCutResult(Mc2Tiling::XAllGatherMatmulTilingData& tilingData,
    mc2tiling::TilingArgs& args)
{
    (void)tilingData;
    XAllGatherPlusMMA2A3 tileFormulate(args, args.rankDim, KernelType::ALL_GATHER, SocVersion::SOC910_B);
    tileFormulate.GetTiling();
    return tileFormulate.tilingM_.cutRes;
}

static ge::graphStatus XAllGatherMatmulTilingFuncA2A3(gert::TilingContext *context)
{
    XAllGatherMatmulTilingA2A3 impl;
    return impl.XAllGatherMatmulTilingFunc(context);
}

struct XAllGatherMatmulCompileInfo {};
static ge::graphStatus TilingParseForXAllGatherMatmul([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(XAllGatherMatmul)
    .Tiling(XAllGatherMatmulTilingFuncA2A3)
    .TilingParse<XAllGatherMatmulCompileInfo>(TilingParseForXAllGatherMatmul);
}
