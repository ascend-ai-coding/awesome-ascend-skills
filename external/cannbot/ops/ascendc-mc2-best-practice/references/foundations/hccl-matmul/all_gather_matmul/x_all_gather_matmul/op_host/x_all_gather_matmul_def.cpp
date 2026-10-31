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
 * \file x_all_gather_matmul_def.cpp
 * \brief Migrated from ops-transformer/mc2/x_all_gather_matmul/op_host/x_all_gather_matmul_def.cpp.
 *        Op class name XAllGatherMatmul (the x_ prefix added in the prior x_ops_lite_plus migration was dropped in this refactor).
 *        Per the non-quant A2 (Ascend910B) scope, only the ascend910b AICore config is registered
 *        (the source's ascend910_93 (A3) and ascend950 (A5) configs are dropped).
 */
#include "register/op_def_registry.h"

namespace ops {
class XAllGatherMatmul : public OpDef {
 public:
  explicit XAllGatherMatmul(const char *name) : OpDef(name) {
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .IgnoreContiguous();
    this->Input("bias")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("gather_out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

    this->Attr("group").AttrType(REQUIRED).String();
    this->Attr("is_trans_a").AttrType(OPTIONAL).Bool(false);
    this->Attr("is_trans_b").AttrType(OPTIONAL).Bool(false);
    this->Attr("gather_index").AttrType(OPTIONAL).Int(0);
    this->Attr("comm_turn").AttrType(OPTIONAL).Int(0);
    this->Attr("rank_size").AttrType(OPTIONAL).Int(0);
    this->Attr("is_gather_out").AttrType(OPTIONAL).Bool(true);

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("jitCompile.flag", "static_false")  // 动态shape,复用二进制,后续图支持后修改
        .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
    // 非量化、A2(Ascend910B)场景：仅注册 ascend910b
    this->AICore().AddConfig("ascend910b", aicore_config);
    this->MC2().HcclGroup("group");
  }
};

OP_ADD(XAllGatherMatmul);
}  // namespace ops
