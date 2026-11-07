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
 * \file euclidean_norm_def.cpp
 * \brief EuclideanNorm 算子定义
 *
 * y = sqrt( sum( x^2 ) ) along axes ；reduce 类算子，按 sum reducer
 * 路径实例化，pre-elewise=square、post-elewise=sqrt+Cast。
 */
#include "register/op_def_registry.h"

namespace ops {

// EuclideanNorm 算子定义：声明 x/axes/y 的 dtype 组合表、keep_dims attr 与 AICore 编译配置。
class EuclideanNorm : public OpDef {
public:
    explicit EuclideanNorm(const char* name)
        : OpDef(name)
    {
        // 输入 x / axes / 输出 y 的 dtype 表（共 8 行，按列对位组合）：
        //   行 i  : x[i],   axes[i],  y[i]
        //   x  ∈ {fp16, bf16, fp32, int32}，4 种 × 2 种 axes（int32/int64）= 8 行
        //   axes 按 IndexNumberType：int32 / int64。
        //   y dtype 与 x 一致。
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 输入 axes：列出要 reduce 的维度的 1-D 整数张量。
        //   infershape 和 tiling 都依赖 axes 的实际数值（否则推不出 output shape、做不出合轴决策），
        //   属"值依赖输入"，必须配 .ValueDepend(OPTIONAL) +
        //   IMPL_OP_INFERSHAPE.InputsDataDependency({1}) +
        //   IMPL_OP_OPTILING.TilingInputsDataDependency({1})。axes 索引为 1。
        this->Input("axes")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                       ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 输出 y：dtype 与 x 一致；shape 由 keep_dims 与 axes 共同决定（infershape 阶段算出）。
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // Attr keep_dims：true 时 reduce 后的轴保留为 size 1；false 时从 output shape 中删除。
        //   按 IR 注册顺序，索引 0。
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);

        // AI Core 编译配置（只支持 ascend950）。
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "euclidean_norm");

        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(EuclideanNorm);
} // namespace ops
