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
 * \file quant_matmul_gelu_example_def.cpp
 * \brief quant_matmul_gelu_example 算子定义 — AIC+AIV 融合（MIX）算子脚手架
 *
 * [MODIFY] 本文件是算子定义的骨架，用于声明输入输出和算子配置。
 * 开发新融合算子时，按以下步骤修改：
 *   1. 修改类名（QuantMatmulGeluExample → 您的算子名）
 *   2. 修改 Input/Output 名称、DataType、Format
 *   3. 添加/移除 AICoreConfig 对应支持的 SoC 版本
 *   4. 如需属性（transpose 等），添加 .Attr() 声明
 *
 * 当前示例为 5 输入 + 1 输出的量化矩阵乘 + 激活融合算子：
 *   - x1[m, k]            int8     (左矩阵)
 *   - x2[n, k]            int8     (右矩阵, 转置语义)
 *   - scale[n]            float32  (perchannel 反量化因子)
 *   - pertoken_scale[m]   float    (pertoken 反量化因子)
 *   - bias[n]             bfloat16 (perchannel bias, L2 层 cast 为 float 后下发)
 *   - y[m, n]             bfloat16 (out = gelu(matmul * scale * pertoken + bias))
 *
 */

#include "register/op_def_registry.h"

namespace ops {
class QuantMatmulGeluExample : public OpDef {
public:
    explicit QuantMatmulGeluExample(const char* name) : OpDef(name)
    {
        // [MODIFY] 输入定义 — 根据您的算子需求修改 Input 名称、DataType、Format
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pertoken_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        //   bias 外部接口 dtype 需与 L2 API 的 CheckDtypeValid 保持一致。
        //   框架会将自定义 AiCore 算子的 bf16 输入自动 cast 为 float，
        //   故 kernel 侧按 float 注册，L2 层显式 cast。
        //   如 bias 输入不需要 bf16→float 转换，直接注册为 float 即可。
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // [MODIFY] 输出定义 — 根据您的算子输出类型修改
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // [MODIFY] 仅支持 Ascend950 (DAV_3510)。如需支持更多 SoC，添加对应 AICoreConfig。
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};
OP_ADD(QuantMatmulGeluExample);
} // namespace ops