/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// 通用模板 - InferDataType 测试实现
// ============================================================================
// 文件命名：test_{op_name}_dtype_infershape.cpp
// 实际使用时需要根据算子定义替换：
// 1. 替换占位符 {{OpName}}, {{op_name}}
// 2. 替换 Tensor/属性名称（参考 template_param.h）
// ============================================================================

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "{{op_name}}_param.h"
#include <gtest/gtest.h>

namespace {
{
  OpName
}
} // namespace
UT {

  class {
    {
      OpName
    }
  } InferDTypeTest : public testing::TestWithParam < {
    {
      OpName
    }
  }
  InferDTypeUtParam>{protected : static void SetUpTestCase(){
      std::cout << "{{OpName}} InferDTypeTest SetUp" << std::endl;
}

static void TearDownTestCase() {
  std::cout << "{{OpName}} InferDTypeTest TearDown" << std::endl;
}
}
;

TEST_P(
    {
      {
        OpName
      }
    } InferDTypeTest,
    param) {
  auto param = GetParam();

  // ========== 构建输入数据类型列表（根据 inputInstance 筛选）==========
  std::vector<void *> inputDataTypes;
  if (param.inputInstance[0] == 1)
    inputDataTypes.emplace_back(&param.input0);
  if (param.inputInstance[1] == 1)
    inputDataTypes.emplace_back(&param.input1);
  if (param.inputInstance[2] == 1)
    inputDataTypes.emplace_back(&param.input_optional);
  // ... 添加所有输入 Tensor 的判断逻辑 ...

  // ========== 构建输出数据类型列表（根据 outputInstance 筛选）==========
  std::vector<void *> outputDataTypes;
  if (param.outputInstance[0] == 1)
    outputDataTypes.emplace_back(&param.output0);
  if (param.outputInstance[1] == 1)
    outputDataTypes.emplace_back(&param.output_optional);
  // ... 添加所有输出 Tensor 的判断逻辑 ...

  // ========== 构建 InferDataTypeContext ==========
  auto contextHolder =
      gert::InferDataTypeContextFaker()
          .SetOpType("{{OpName}}")
          .IrInstanceNum(param.inputInstance, param.outputInstance)
          .InputDataTypes(inputDataTypes)
          .OutputDataTypes(outputDataTypes)
          .NodeAttrs({
              {"attr_int",
               Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.attr_int)},
              {"attr_float",
               Ops::Transformer::AnyValue::CreateFrom<float>(param.attr_float)},
              {"attr_string",
               Ops::Transformer::AnyValue::CreateFrom<std::string>(
                   param.attr_string)},
              {"attr_bool",
               Ops::Transformer::AnyValue::CreateFrom<bool>(param.attr_bool)},
              // ... 添加所有属性 ...
          })
          .Build();

  // ========== 执行测试 ==========
  auto spaceRegistry =
      gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto inferDtypeFunc = spaceRegistry->GetOpImpl("{{OpName}}")->infer_datatype;
  ASSERT_EQ(
      inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()),
      param.expectResult);

  // ========== 验证输出数据类型（仅 SUCCESS 时）==========
  if (param.expectResult == ge::GRAPH_SUCCESS) {
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()
                  ->GetOutputDataType(0),
              param.output0);
    // EXPECT_EQ(
    //     contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1),
    //     param.output_optional
    // );
    // ... 验证所有输出 Tensor ...
  }
}

// ========== 从 CSV 加载测试用例 ==========
INSTANTIATE_TEST_SUITE_P(
    {{OpName}},
    {
      {
        OpName
      }
    } InferDTypeTest,
    testing::ValuesIn(GetCasesFromCsv<{
      {
        OpName
      }
    } InferDTypeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<{
      {
        OpName
      }
    } InferDTypeUtParam>);

} // namespace {{OpName}}UT