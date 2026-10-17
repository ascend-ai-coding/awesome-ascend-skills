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
// 通用模板 - InferShape 测试实现
// ============================================================================
// 文件命名：test_{op_name}_shape_infershape.cpp
// 实际使用时需要根据算子定义替换：
// 1. 替换占位符 {{OpName}}, {{op_name}}
// 2. 替换 Tensor/属性名称（参考 template_param.h）
// ============================================================================

#include "infer_shape_case_executor.h"
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
  } InferShapeTest : public testing::TestWithParam < {
    {
      OpName
    }
  }
  InferShapeUtParam>{protected : static void SetUpTestCase(){
      std::cout << "{{OpName}} InferShapeTest SetUp" << std::endl;
}

static void TearDownTestCase() {
  std::cout << "{{OpName}} InferShapeTest TearDown" << std::endl;
}
}
;

TEST_P(
    {
      {
        OpName
      }
    } InferShapeTest,
    param) {
  auto param = GetParam();

  // ========== 构建输入 Tensor 列表（根据 inputInstance 筛选）==========
  std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;

  // 根据 inputInstance 数组判断哪些 Tensor 存在
  if (param.inputInstance[0] == 1)
    inputTensorDesc.emplace_back(param.input0);
  if (param.inputInstance[1] == 1)
    inputTensorDesc.emplace_back(param.input1);
  if (param.inputInstance[2] == 1)
    inputTensorDesc.emplace_back(param.input_optional);
  // ... 添加所有输入 Tensor 的判断逻辑 ...

  // ========== 构建输出 Tensor 列表（根据 outputInstance 筛选）==========
  std::vector<gert::InfershapeContextPara::TensorDescription> outputTensorDesc;

  if (param.outputInstance[0] == 1)
    outputTensorDesc.emplace_back(param.output0);
  if (param.outputInstance[1] == 1)
    outputTensorDesc.emplace_back(param.output_optional);
  // ... 添加所有输出 Tensor 的判断逻辑 ...

  // ========== 构建 InferShapeContextPara ==========
  gert::InfershapeContextPara infershapeContextPara(
      "{{OpName}}", // 算子名称
      inputTensorDesc, outputTensorDesc,

      // ========== 属性列表（从 xxx_def.cpp 提取）==========
      {
          {"attr_int",
           Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.attr_int)},
          {"attr_float",
           Ops::Transformer::AnyValue::CreateFrom<float>(param.attr_float)},
          {"attr_string", Ops::Transformer::AnyValue::CreateFrom<std::string>(
                              param.attr_string)},
          {"attr_bool",
           Ops::Transformer::AnyValue::CreateFrom<bool>(param.attr_bool)},
          // ... 添加所有属性 ...
      });

  // ========== 执行测试 ==========
  ExecuteTestCase(infershapeContextPara, param.expectResult,
                  param.expectOutputShape);
}

// ========== 从 CSV 加载测试用例 ==========
INSTANTIATE_TEST_SUITE_P(
    {{OpName}},
    {
      {
        OpName
      }
    } InferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<{
      {
        OpName
      }
    } InferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<{
      {
        OpName
      }
    } InferShapeUtParam>);

} // namespace {{OpName}}UT