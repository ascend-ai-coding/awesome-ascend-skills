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
 * \file x_all_gather_matmul_tiling_base.h
 * \brief Migrated from ops-transformer/mc2/x_all_gather_matmul/op_host/op_tiling/x_all_gather_matmul_tiling_base.h.
 *        Local includes renamed to all_gather_-prefixed files; mc2/common framework includes resolve against the
 *        vendored mc2_common/ tree via -I.
 */
#ifndef __ALL_GATHER_MATMUL_TILING_BASE_H__
#define __ALL_GATHER_MATMUL_TILING_BASE_H__

#include "vector"
#include "mc2_hcom_topo_info.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "x_all_gather_formulaic_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../op_kernel/x_all_gather_matmul_tiling_key.h"
#include "../../op_kernel/x_all_gather_matmul_tiling.h"

namespace optiling {
class XAllGatherMatmulTilingBase {
public:
    ge::graphStatus XAllGatherMatmulTilingFunc(gert::TilingContext *context);
    ge::graphStatus SetMatmulTilingXAllGatherMatmul(gert::TilingContext* context,
                                                   Mc2Tiling::XAllGatherMatmulTilingData& tilingData,
                                                   mc2tiling::TilingArgs& args);
    ge::graphStatus MCSpliteM(gert::TilingContext* ctx, Mc2Tiling::XAllGatherMatmulTilingData& tilingData,
                              mc2tiling::TilingArgs& args);
    ge::graphStatus GetXAllGatherFormulateTileCnt(const gert::TilingContext* ctx,
        Mc2Tiling::XAllGatherMatmulTilingData& tilingData, mc2tiling::TilingArgs& args);
    ge::graphStatus InitHcclParam(const gert::TilingContext *context, Mc2Tiling::XAllGatherMatmulTilingData* tilingData,
                                  const char* group);
    virtual CutResult GetCutResult(Mc2Tiling::XAllGatherMatmulTilingData& tilingData,
                                   mc2tiling::TilingArgs& args) = 0;
    virtual ge::graphStatus CheckValidRank(Mc2Tiling::XAllGatherMatmulTilingData* tilingData,
        const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK, gert::TilingContext *context,
        uint32_t rankSize) = 0;
    virtual void SetSocParam(Mc2Tiling::XAllGatherMatmulTilingData* tilingData, const char* group) = 0;
    virtual std::string GetAlgConfig(Mc2Tiling::XAllGatherMatmulTilingData* tilingData) = 0;
    virtual ~XAllGatherMatmulTilingBase() = default;
};
}

#endif //__ALL_GATHER_MATMUL_TILING_BASE_H__
