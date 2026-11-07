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
 * \file mc2_tiling_struct.h
 * \brief Trimmed vendored copy for all_gather_matmul.
 *
 *        The original mc2/common host header defined seven optiling tiling-data structs (MC2ServerCfg,
 *        MC2HcommCfg, RCSTiling, Mc2Msg, TileL2Tiling, TileInfo, MC2MatmulV3TilingData) via
 *        BEGIN_TILING_DATA_DEF/REGISTER_TILING_DATA_CLASS. None of them — nor their registered ...Op
 *        classes — are used by all_gather_matmul: this op's tiling data is the plain class
 *        Mc2Tiling::AllGatherMatmulTilingData (op_kernel/all_gather_matmul_tiling.h), retrieved host-side
 *        via context->GetTilingData<Mc2Tiling::AllGatherMatmulTilingData>() and kernel-side via
 *        REGISTER_TILING_DEFAULT(...). It does not nest any of the structs removed here. (The live
 *        Mc2Tiling::RCSTiling/Mc2Msg/TileL2Tiling/TileInfo used by this op live in the kernel-side copy
 *        mc2_common/op_kernel/mc2_tiling_struct.h, a different file/namespace — not affected here.)
 *
 *        All seven host optiling structs were dead code for this operator and have been removed. In
 *        particular MC2MatmulV3TilingData was the only user of TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling,
 *        matmulTiling); its expansion failed with "could not convert '{nullptr}' to
 *        AscendC::tiling::TCubeTiling" in translation units that did not first include a matmul tiling
 *        header. Removing it eliminates that error at its source, so the "tiling/tiling_api.h" include
 *        that briefly patched it is also dropped — nothing else in this file needs TCubeTiling.
 *
 *        Only the shared optiling constants are retained.
 */

#ifndef __MC2_TILING_STRUCT_H__
#define __MC2_TILING_STRUCT_H__

#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint8_t COMM_ALG_FULL_MESH = 1;
constexpr int64_t KVALUE_MIN = 256;
constexpr int64_t KVALUE_MAX = 65535;
} // namespace optiling

#endif
