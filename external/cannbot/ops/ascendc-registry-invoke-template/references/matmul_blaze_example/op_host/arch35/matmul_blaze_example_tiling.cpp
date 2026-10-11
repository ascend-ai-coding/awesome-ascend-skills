/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "matmul_blaze_example_tiling_data.h"
#include "matmul_blaze_example_tiling_key.h"
#include <algorithm>
#include <cstring>

namespace optiling {

static uint32_t AlignUp(uint64_t val, uint32_t align)
{
    return static_cast<uint32_t>(((val + align - 1) / align) * align);
}

static ge::graphStatus MatmulBlazeExampleTilingFunc(gert::TilingContext* context)
{
    auto aShape = context->GetInputShape(0)->GetStorageShape();
    auto bShape = context->GetInputShape(1)->GetStorageShape();
    uint64_t m = static_cast<uint64_t>(aShape.GetDim(0));
    uint64_t k = static_cast<uint64_t>(aShape.GetDim(1));
    uint64_t n = static_cast<uint64_t>(bShape.GetDim(1));

    OP_CHECK_IF(k != static_cast<uint64_t>(bShape.GetDim(0)),
        OP_LOGE(context, "MatmulBlazeExample: K mismatch. a.K=%lu, b.K=%lu", k, bShape.GetDim(0)),
        return ge::GRAPH_FAILED);

    constexpr uint32_t MAX_BASE = 128;
    constexpr uint32_t ALIGN = 16;
    constexpr uint32_t MAX_CORES = 48;

    MatmulBlazeExampleTilingData tilingData = {};
    tilingData.m = static_cast<uint32_t>(m);
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.k = static_cast<uint32_t>(k);

    tilingData.baseM = std::min(AlignUp(m, ALIGN), MAX_BASE);
    if (tilingData.baseM == 0) tilingData.baseM = ALIGN;
    tilingData.baseN = std::min(AlignUp(n, ALIGN), MAX_BASE);
    if (tilingData.baseN == 0) tilingData.baseN = ALIGN;
    tilingData.baseK = std::min(AlignUp(k, ALIGN), MAX_BASE);
    if (tilingData.baseK == 0) tilingData.baseK = ALIGN;

    tilingData.mL1 = tilingData.baseM;
    tilingData.nL1 = tilingData.baseN;
    tilingData.kL1 = tilingData.baseK;
    tilingData.mTailCnt = 1;
    tilingData.nTailCnt = 1;
    tilingData.mBaseTailSplitCnt = 1;
    tilingData.nBaseTailSplitCnt = 1;
    tilingData.mTailMain = 0;
    tilingData.nTailMain = 0;
    tilingData.l1BufferNum = 2;
    tilingData.l0cDB = 1;

    uint32_t mTiles = static_cast<uint32_t>((m + tilingData.baseM - 1) / tilingData.baseM);
    uint32_t nTiles = static_cast<uint32_t>((n + tilingData.baseN - 1) / tilingData.baseN);
    tilingData.usedCoreNum = std::min(mTiles * nTiles, MAX_CORES);
    if (tilingData.usedCoreNum == 0) tilingData.usedCoreNum = 1;

    auto* tilingPtr = context->GetTilingData<MatmulBlazeExampleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingPtr);
    OP_CHECK_IF(memcpy_s(tilingPtr, sizeof(MatmulBlazeExampleTilingData),
            &tilingData, sizeof(MatmulBlazeExampleTilingData)) != EOK,
        OP_LOGE(context, "memcpy_s failed"), return ge::GRAPH_FAILED);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

    context->SetBlockDim(static_cast<int64_t>(tilingData.usedCoreNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMatmulBlazeExample([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MatmulBlazeExampleCompileInfo {};

IMPL_OP_OPTILING(MatmulBlazeExample)
    .Tiling(MatmulBlazeExampleTilingFunc)
    .TilingParse<MatmulBlazeExampleCompileInfo>(TilingParseForMatmulBlazeExample);

} // namespace optiling
