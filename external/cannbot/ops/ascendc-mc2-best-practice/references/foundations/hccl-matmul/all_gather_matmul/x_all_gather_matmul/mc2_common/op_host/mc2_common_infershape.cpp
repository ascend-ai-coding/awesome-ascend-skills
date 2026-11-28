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
 * \file mc2_common_infershape.cpp
 * \brief
 */

#include "op_host/mc2_common_infershape.h"

using namespace ge;
namespace ops {
static ge::graphStatus CheckMatrixInputShapes(const gert::InferShapeContext* context, CommParas& commParas)
{
    commParas.x1MatrixShape = context->GetInputShape(0);
    if ((commParas.x1MatrixShape) == nullptr) { return ge::GRAPH_FAILED; }
    commParas.x2MatrixShape = context->GetInputShape(1);
    if ((commParas.x2MatrixShape) == nullptr) { return ge::GRAPH_FAILED; }
    if (commParas.x1MatrixShape->GetDimNum() != SUPPORT_DIM_SIZE) {
        return ge::GRAPH_FAILED;
    }
    if (commParas.x2MatrixShape->GetDimNum() != SUPPORT_DIM_SIZE) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ResolveRankSize(
    const gert::InferShapeContext* context, const char* groupStr, const int64_t* rankSizeAttr, int64_t& rankSize)
{
    if (*rankSizeAttr <= 0) {
#if defined(ENABLE_BUILT_IN)
        uint32_t rankNum = 0;
        if (Mc2Hcom::MC2HcomTopology::CommGetInstSizeByGroup(groupStr, &rankNum) != HCCL_SUCCESS || rankNum == 0) {
            return ge::GRAPH_FAILED;
        }
        rankSize = static_cast<int64_t>(rankNum);
#else

        return ge::GRAPH_FAILED;
#endif
    } else {
        rankSize = *rankSizeAttr;
    }
    return ge::GRAPH_SUCCESS;
}

static void FillMatmulDims(CommParas& commParas, bool isTransA, bool isTransB)
{
    commParas.dimM = !isTransA ? commParas.x1MatrixShape->GetDim(0) : commParas.x1MatrixShape->GetDim(1);
    commParas.dimKX1 = !isTransA ? commParas.x1MatrixShape->GetDim(1) : commParas.x1MatrixShape->GetDim(0);
    commParas.dimKX2 = !isTransB ? commParas.x2MatrixShape->GetDim(0) : commParas.x2MatrixShape->GetDim(1);
    commParas.dimN = !isTransB ? commParas.x2MatrixShape->GetDim(1) : commParas.x2MatrixShape->GetDim(0);
}

static ge::graphStatus CheckKDimMatch(const gert::InferShapeContext* context, const CommParas& commParas)
{
    if (commParas.dimKX1 != commParas.dimKX2) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// infershape 公共函数
ge::graphStatus CommonParamCheck(
    const gert::InferShapeContext* context, const size_t isTransAIndex, const size_t isTransBIndex, CommParas& commParas)
{
    if (CheckMatrixInputShapes(context, commParas) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if ((attrs) == nullptr) { return ge::GRAPH_FAILED; }
    const bool* isTransA = attrs->GetAttrPointer<bool>(isTransAIndex);
    const bool* isTransB = attrs->GetAttrPointer<bool>(isTransBIndex);
    const int64_t* rankSizeAttr = attrs->GetAttrPointer<int64_t>(RANK_SIZE);

    const char* groupStr = attrs->GetAttrPointer<char>(GROUP);
    if (groupStr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (ResolveRankSize(context, groupStr, rankSizeAttr, commParas.rankSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FillMatmulDims(commParas, *isTransA, *isTransB);

    return CheckKDimMatch(context, commParas);
}

enum class YShapeMode { AllGather, ReduceScatter };

static ge::graphStatus PrepareMatmulYShape(gert::InferShapeContext* context, CommParas& commParas, YShapeMode mode)
{
    if (commParas.dimM == -1) {
        commParas.rankSize = 1;
    }
    // 不支持 k = 0
    if (commParas.dimKX1 == 0) {
        commParas.dimM = commParas.dimN = 0;
        return ge::GRAPH_FAILED;
    }
    gert::Shape* yShape = context->GetOutputShape(0);
    if ((yShape) == nullptr) { return ge::GRAPH_FAILED; }
    yShape->SetDimNum(SUPPORT_DIM_SIZE);
    const int64_t dim0 = (mode == YShapeMode::AllGather)
        ? commParas.dimM * commParas.rankSize
        : commParas.dimM / commParas.rankSize;
    yShape->SetDim(0, dim0);
    yShape->SetDim(1, commParas.dimN);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulInferYShape(gert::InferShapeContext* context, CommParas& commParas)
{
    if (CommonParamCheck(context, AG_IS_TRANS_A, AG_IS_TRANS_B, commParas) != GRAPH_SUCCESS) { return GRAPH_FAILED; }
    return PrepareMatmulYShape(context, commParas, YShapeMode::AllGather);
}

ge::graphStatus AllGatherMatmulInferGatherOutShape(gert::InferShapeContext* context, const CommParas& commParas,
                                                   const size_t gatherIndex)
{
    if (context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const bool* isGatherOut = context->GetAttrs()->GetAttrPointer<bool>(gatherIndex);
    if ((isGatherOut) == nullptr) { return ge::GRAPH_FAILED; }
    gert::Shape* gatherOutShape = context->GetOutputShape(1);
    if ((gatherOutShape) == nullptr) { return ge::GRAPH_FAILED; }
    if (*isGatherOut) {
        gatherOutShape->SetDimNum(SUPPORT_DIM_SIZE);
        gatherOutShape->SetDim(0, commParas.dimM * commParas.rankSize);
        gatherOutShape->SetDim(1, commParas.dimKX1);
    } else {
        gatherOutShape->SetDimNum(1);
        gatherOutShape->SetDim(0, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulCommonInferShape(gert::InferShapeContext* context, const size_t gatherIndex)
{
    CommParas commParas;
    if (AllGatherMatmulInferYShape(context, commParas) != GRAPH_SUCCESS) { return GRAPH_FAILED; }
    
    if (AllGatherMatmulInferGatherOutShape(context, commParas, gatherIndex) != GRAPH_SUCCESS) { return GRAPH_FAILED; }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferMatmulReduceScatterCommon(gert::InferShapeContext* context)
{
    CommParas commParas;
    if (CommonParamCheck(context, RS_IS_TRANS_A, RS_IS_TRANS_B, commParas) != GRAPH_SUCCESS) { return GRAPH_FAILED; }
    return PrepareMatmulYShape(context, commParas, YShapeMode::ReduceScatter);
}
} // namespace ops
