/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_gelu_example_tiling.cpp
 * \brief quant_matmul_gelu_example 算子 Host 侧 Tiling 实现（arch35 架构）
 *
 * 形状提取：x1=(M,K), x2=(N,K) 转置语义 → M=x1[-2], K=x1[-1], N=x2[-2]。
 * 单一 TilingKey；2D GEMM（无 batch）；转置语义固定（无属性）。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/quant_matmul_gelu_example_tiling_data.h"
#include "../../op_kernel/arch35/quant_matmul_gelu_example_tiling_key.h"
#include "quant_matmul_gelu_example_tiling.h"

#include "matmul_tiling.h"
#include "platform/platform_ascendc.h"

namespace optiling {

class MatmulTilingSwatRegistry : public MatmulTilingSwat {
public:
    MatmulTilingSwatRegistry() = default;
    ~MatmulTilingSwatRegistry() override = default;

    void SetPlatformInfo(const MatmulPlatformInfo& info)
    {
        platformInfo_ = info;
    }

    void ComputeTiling(uint64_t m, uint64_t n, uint64_t k, MatmulTilingData& tilingData)
    {
        args_.m = m;
        args_.n = n;
        args_.k = k;
        args_.hasBias = false;
        args_.isATrans = false;
        args_.isBTrans = true;
        DoOpTiling(tilingData);
    }
};

static ge::graphStatus FillPlatformInfoFromContext(gert::TilingContext* context, MatmulPlatformInfo& platformInfo)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    platformInfo.aicNum = ascendcPlatform.GetCoreNumAic();
    platformInfo.aivNum = ascendcPlatform.GetCoreNumAiv();
    platformInfo.socVersion = ascendcPlatform.GetSocVersion();

    OP_CHECK_IF(platformInfo.aicNum == 0,
        OP_LOGE(context, "aicNum is 0"), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::BT, platformInfo.btSize);

    return ge::GRAPH_SUCCESS;
}

// ========================================================================
// [MODIFY] 形状提取逻辑。
//
//   当前为 2D GEMM x1=(M,K), x2=(N,K) 转置语义。
//   泛化变体：
//     - 3D BMM: 增加 batch 维度提取, M=x1[-2]*batch, 或按 batch 分核
//     - 动态 transpose: K 维度位置取决于 transpose 属性
//       transX1=false → x1=(M,K), K=x1[-1]
//       transX1=true  → x1=(K,M), K=x1[-2]
//     - 1D 向量乘: 降低维数要求（>=1 而非 ==2）
// ========================================================================
static ge::graphStatus GetShapeFromContext(gert::TilingContext* context,
    uint64_t& m, uint64_t& n, uint64_t& k)
{
    auto x1Desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    auto x2Desc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);

    auto x1Shape = context->GetInputShape(0)->GetStorageShape();
    auto x2Shape = context->GetInputShape(1)->GetStorageShape();

    auto x1DimNum = x1Shape.GetDimNum();
    auto x2DimNum = x2Shape.GetDimNum();

    // [MODIFY] 当前固定 x1=(M,K), x2=(N,K) 转置语义。
    //   如 transpose 由属性控制，需根据 isATrans/isBTrans 调整 K 的提取位置：
    //     isATrans=false → x1 的 K 在 x1[-1], M 在 x1[-2]
    //     isATrans=true  → x1 的 K 在 x1[-2], M 在 x1[-1]
    //     isBTrans=true  → x2 的 K 在 x2[-1], N 在 x2[-2]
    //     isBTrans=false → x2 的 K 在 x2[-1], N 在 x2[-2]（非转置，x2=(K,N)）
    //   注意：此时 x2 的逻辑形状为 (K,N)，物理形状为 (K,N)
    m = static_cast<uint64_t>(x1Shape.GetDim(x1DimNum - 2));
    k = static_cast<uint64_t>(x1Shape.GetDim(x1DimNum - 1));
    n = static_cast<uint64_t>(x2Shape.GetDim(x2DimNum - 2));

    OP_CHECK_IF(m < 16 || n < 16 || k < 16,
        OP_LOGE(context, "M/N/K must >= 16. M=%lu, N=%lu, K=%lu", m, n, k),
        return ge::GRAPH_FAILED);

    uint64_t x2K = static_cast<uint64_t>(x2Shape.GetDim(x2DimNum - 1));
    OP_CHECK_IF(k != x2K,
        OP_LOGE(context, "K mismatch. x1K=%lu, x2K=%lu (x2 shape=[N,K])", k, x2K),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

constexpr size_t WORKSPACE_NUM = 1;
constexpr size_t WS_SYS_SIZE = 0;

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus QuantMatmulGeluExampleTilingFunc(gert::TilingContext* context)
{
    MatmulPlatformInfo platformInfo;
    OP_CHECK_IF(FillPlatformInfoFromContext(context, platformInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "FillPlatformInfo failed"), return ge::GRAPH_FAILED);

    uint64_t m = 0, n = 0, k = 0;
    OP_CHECK_IF(GetShapeFromContext(context, m, n, k) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShape failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize failed"), return ge::GRAPH_FAILED);

    MatmulTilingSwatRegistry tilingEngine;
    tilingEngine.SetPlatformInfo(platformInfo);
    MatmulTilingData tilingData;
    tilingEngine.ComputeTiling(m, n, k, tilingData);

    // 组装 Custom TilingData（公共字段布局与 Blaze MatmulTilingData 一致）
    QuantMatmulGeluExampleTilingData finalTilingData;
    OP_CHECK_IF(memcpy_s(&finalTilingData, sizeof(MatmulTilingData), &tilingData, sizeof(MatmulTilingData)) != EOK,
        OP_LOGE(context, "memcpy_s for common fields failed"), return ge::GRAPH_FAILED);

    QuantMatmulGeluExampleTilingData* tilingPtr = context->GetTilingData<QuantMatmulGeluExampleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingPtr);
    OP_CHECK_IF(memcpy_s(tilingPtr, sizeof(QuantMatmulGeluExampleTilingData),
            &finalTilingData, sizeof(QuantMatmulGeluExampleTilingData)) != EOK,
        OP_LOGE(context, "memcpy_s failed"), return ge::GRAPH_FAILED);

    context->SetBlockDim(static_cast<int64_t>(tilingData.usedCoreNum));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForQuantMatmulGeluExample([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct QuantMatmulGeluExampleCompileInfo {};

IMPL_OP_OPTILING(QuantMatmulGeluExample)
    .Tiling(QuantMatmulGeluExampleTilingFunc)
    .TilingParse<QuantMatmulGeluExampleCompileInfo>(TilingParseForQuantMatmulGeluExample);

} // namespace optiling