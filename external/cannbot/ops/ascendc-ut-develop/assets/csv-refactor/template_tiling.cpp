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
// 通用模板 - Tiling 测试实现
// ============================================================================
// 实际使用时需要根据算子定义替换：
// 1. 替换占位符 {{OpName}}, {{op_name}}
// 2. 替换 Tensor/属性名称（参考 template_param.h）
// ============================================================================

#include <gtest/gtest.h>
#include "../../{{op_name}}_param.h"
#include "tiling_case_executor.h"

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
  } Arch32TilingTest : public testing::TestWithParam < {
    {
      OpName
    }
  }
  TilingUtParam>{protected : static void SetUpTestCase(){
      std::cout << "{{OpName}} Arch32 TilingTest SetUp" << std::endl;
}

static void TearDownTestCase() {
  std::cout << "{{OpName}} Arch32 TilingTest TearDown" << std::endl;
}
}
;

TEST_P(
    {
      {
        OpName
      }
    } Arch32TilingTest,
    param) {
  auto param = GetParam();

  // ========== 定义 CompileInfo（如果需要）==========
  struct EmptyCompileInfo {
  } compileInfo;

  // ========== 定义 SocInfo（硬件信息）==========
  const std::string A2SocInfo =
      "{\n"
      "  \"hardware_info\": {\n"
      "    \"BT_SIZE\": 0,\n"
      "    \"load3d_constraints\": \"1\",\n"
      "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
      "    \"Intrinsic_data_move_l12ub\": true,\n"
      "    \"Intrinsic_data_move_l0c2ub\": true,\n"
      "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
      "    \"UB_SIZE\": 196608,\n"
      "    \"L2_SIZE\": 201326592,\n"
      "    \"L1_SIZE\": 524288,\n"
      "    \"L0A_SIZE\": 65536,\n"
      "    \"L0B_SIZE\": 65536,\n"
      "    \"L0C_SIZE\": 131072,\n"
      "    \"vector_core_cnt\": 40,\n"
      "    \"cube_core_cnt\": 20,\n"
      "    \"socVersion\": \"Ascend910_B3\"\n"
      "  }\n"
      "}";

  // ========== 构建 TilingContextPara ==========
  gert::TilingContextPara tilingContextPara(
      "{{OpName}}", // 算子名称

      // ========== 输入 Tensor 列表（按算子定义顺序）==========
      {
          param.input0,
          param.input1,
          // param.input_optional,  // 可选 Tensor，根据 inputInstance 判断
          // ... 添加所有输入 Tensor ...
      },

      // ========== 输出 Tensor 列表（按算子定义顺序）==========
      {
          param.output0,
          // param.output_optional,  // 可选 Tensor
          // ... 添加所有输出 Tensor ...
      },

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
      },

      // ========== 其他参数 ==========
      param.inputInstance,  // 输入 Tensor 实例标记
      param.outputInstance, // 输出 Tensor 实例标记
      &compileInfo,         // CompileInfo（可选）
      "Ascend910B",         // 芯片型号
      40,                   // 核数（可根据算子调整）
      196608,               // UB 大小
      4096,                 // 其他参数
      A2SocInfo             // SocInfo
  );

  // ========== 执行测试 ==========
  ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey,
                  param.expectTilingDataHash, {}, // expectWorkspaces（可选）
                  0,   // tilingDataReservedLen（可选）
                  true // useHashTilingData
  );
}

// ========== 从 CSV 加载测试用例 ==========
INSTANTIATE_TEST_SUITE_P(
    {{OpName}},
    {
      {
        OpName
      }
    } Arch32TilingTest,
    testing::ValuesIn(GetCasesFromCsv<{
      {
        OpName
      }
    } TilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<{
      {
        OpName
      }
    } TilingUtParam>);

} // namespace {{OpName}}UT