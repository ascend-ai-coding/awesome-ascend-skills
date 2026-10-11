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
// 通用模板 - 使用占位符表示算子参数，实际使用时需要根据算子定义替换
// ============================================================================
// 占位符替换说明：
// {{OP_NAME_UPPER}}  -> 算子名大写（下划线分隔） 示例：MY_OPERATOR
// {{OpName}}         -> 算子名驼峰（无分隔）     示例：MyOperator
// {{op_name}}        -> 算子名小写（下划线分隔） 示例：my_operator
//
// 继承说明：
// HostUtParamBase 是框架提供的基类（定义在 op_host_csv_case_loader.h），包含：
//   - case_name
//   - expectResult
//   - inputInstance
//   - outputInstance
// {{OpName}}HostUtParamBase 继承 HostUtParamBase，添加算子特定的属性字段
// ============================================================================

#ifndef{{OP_NAME_UPPER}}_PARAM_H
#define{{OP_NAME_UPPER}}_PARAM_H

#include "op_host_csv_case_loader.h"
#include <sstream>

namespace {
{
  OpName
}
} // namespace
UT {

  // ============================================================================
  // 算子基础参数结构 - 继承 HostUtParamBase，添加算子属性字段
  // ============================================================================
  struct {
    {
      OpName
    }
  } HostUtParamBase : public HostUtParamBase {
    // ========== 从 xxx_def.cpp 的 Attr 定义中提取的属性字段 ==========
    // 根据算子定义，添加所有属性，示例：

    // Int 类型属性：Attr("attr_int").AttrType(REQUIRED).Int(default_value)
    int64_t attr_int;

    // Float
    // 类型属性：Attr("attr_float").AttrType(OPTIONAL).Float(default_value)
    float attr_float;

    // String 类型属性：Attr("attr_string").AttrType(OPTIONAL).String("default")
    std::string attr_string;

    // Bool 类型属性：Attr("attr_bool").AttrType(OPTIONAL).Bool(default_value)
    bool attr_bool;

    // ... 根据算子定义添加所有属性字段 ...

    {
      {
        OpName
      }
    }
    HostUtParamBase(const csv_map &csvMap) : HostUtParamBase(csvMap) {
      // ========== 从 CSV 读取属性值 ==========
      // 根据属性类型使用相应的转换函数：

      // Int 类型：使用 std::stoll
      this->attr_int = std::stoll(ReadMap(csvMap, "attr_int"));

      // Float 类型：使用 std::stof
      this->attr_float = std::stof(ReadMap(csvMap, "attr_float"));

      // String 类型：直接读取
      this->attr_string = ReadMap(csvMap, "attr_string");

      // Bool 类型：CSV 中填写 0/1，使用 std::stoi
      this->attr_bool = std::stoi(ReadMap(csvMap, "attr_bool"));

      // ... 添加所有属性的读取逻辑 ...
    }
  };

  // ============================================================================
  // Tiling 测试参数结构 - 继承算子基础参数结构，添加 Tensor 字段
  // ============================================================================
  struct {
    {
      OpName
    }
  } TilingUtParam : public {
    {
      OpName
    }
  }
  HostUtParamBase {
    // ========== 输入 Tensor 描述 ==========
    // 根据算子定义的 Input 列表，添加所有输入 Tensor，示例：

    // REQUIRED Tensor：Input("input0").ParamType(REQUIRED)
    gert::TilingContextPara::TensorDescription input0 = TD_DEFAULT;

    // DYNAMIC Tensor：Input("input1").ParamType(DYNAMIC)
    gert::TilingContextPara::TensorDescription input1 = TD_DEFAULT;

    // OPTIONAL Tensor：Input("input_optional").ParamType(OPTIONAL)
    gert::TilingContextPara::TensorDescription input_optional = TD_DEFAULT;

    // ... 根据算子定义添加所有输入 Tensor ...

    // ========== 输出 Tensor 描述 ==========
    // 根据算子定义的 Output 列表，添加所有输出 Tensor，示例：

    // REQUIRED Tensor：Output("output0").ParamType(REQUIRED)
    gert::TilingContextPara::TensorDescription output0 = TD_DEFAULT;

    // OPTIONAL Tensor：Output("output_optional").ParamType(OPTIONAL)
    gert::TilingContextPara::TensorDescription output_optional = TD_DEFAULT;

    // ... 根据算子定义添加所有输出 Tensor ...

    // ========== Tiling 测试专用字段 ==========
    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    {
      {
        OpName
      }
    }
    TilingUtParam(const csv_map &csvMap) : {
      {
        OpName
      }
    }
    HostUtParamBase(csvMap) {
      // ========== 从 CSV 读取输入 Tensor 信息 ==========
      // 格式：GetTensorGE(csvMap, "{tensor_name}_shape", "{tensor_name}_dtype",
      //                   "{tensor_name}_format", this->{tensor_name})
      // 返回值：1 表示 Tensor 存在，0 表示 Tensor 不存在

      this->inputInstance.emplace_back(
          GetTensorGE(csvMap, "input0_shape", "input0_dtype", "input0_format",
                      this->input0));

      this->inputInstance.emplace_back(
          GetTensorGE(csvMap, "input1_shape", "input1_dtype", "input1_format",
                      this->input1));

      this->inputInstance.emplace_back(
          GetTensorGE(csvMap, "input_optional_shape", "input_optional_dtype",
                      "input_optional_format", this->input_optional));

      // ... 添加所有输入 Tensor 的读取逻辑 ...

      // ========== 从 CSV 读取输出 Tensor 信息 ==========

      this->outputInstance.emplace_back(
          GetTensorGE(csvMap, "output0_shape", "output0_dtype",
                      "output0_format", this->output0));

      this->outputInstance.emplace_back(
          GetTensorGE(csvMap, "output_optional_shape", "output_optional_dtype",
                      "output_optional_format", this->output_optional));

      // ... 添加所有输出 Tensor 的读取逻辑 ...

      // ========== 读取 Tiling 期望结果（仅 SUCCESS 时） ==========
      if (this->expectResult == ge::GRAPH_SUCCESS) {
        this->expectTilingKey = std::stoull(ReadMap(csvMap, "expectTilingKey"));
        this->expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
      }
    }
  };

  // ============================================================================
  // InferShape 测试参数结构 - 继承算子基础参数结构，添加 Tensor 字段
  // ============================================================================
  struct {
    {
      OpName
    }
  } InferShapeUtParam : public {
    {
      OpName
    }
  }
  HostUtParamBase {
    // ========== 输入 Tensor 描述 ==========
    gert::InfershapeContextPara::TensorDescription input0 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription input1 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription input_optional = ID_DEFAULT;
    // ... 根据算子定义添加所有输入 Tensor ...

    // ========== 输出 Tensor 描述 ==========
    gert::InfershapeContextPara::TensorDescription output0 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription output_optional = ID_DEFAULT;
    // ... 根据算子定义添加所有输出 Tensor ...

    // ========== InferShape 测试专用字段 ==========
    std::vector<std::vector<int64_t>> expectOutputShape;

    {
      {
        OpName
      }
    }
    InferShapeUtParam(const csv_map &csvMap) : {
      {
        OpName
      }
    }
    HostUtParamBase(csvMap) {
      // ========== 从 CSV 读取输入 Tensor 信息 ==========
      this->inputInstance.emplace_back(
          GetTensorGE(csvMap, "input0_shape", "input0_dtype", "input0_format",
                      this->input0));

      this->inputInstance.emplace_back(
          GetTensorGE(csvMap, "input1_shape", "input1_dtype", "input1_format",
                      this->input1));

      this->inputInstance.emplace_back(
          GetTensorGE(csvMap, "input_optional_shape", "input_optional_dtype",
                      "input_optional_format", this->input_optional));

      // ... 添加所有输入 Tensor 的读取逻辑 ...

      // ========== 从 CSV 读取输出 Tensor 信息 ==========
      this->outputInstance.emplace_back(
          GetTensorGE(csvMap, "output0_shape", "output0_dtype",
                      "output0_format", this->output0));

      this->outputInstance.emplace_back(
          GetTensorGE(csvMap, "output_optional_shape", "output_optional_dtype",
                      "output_optional_format", this->output_optional));

      // ... 添加所有输出 Tensor 的读取逻辑 ...

      // ========== 读取 InferShape 期望输出 Shape（仅 SUCCESS 时） ==========
      if (this->expectResult == ge::GRAPH_SUCCESS) {
        this->expectOutputShape = {
            GetShapeArr(ReadMap(csvMap, "output0_shape")),
            // ... 添加所有输出 Tensor 的期望 shape ...
        };
      }
    }
  };

  // ============================================================================
  // InferDataType 测试参数结构 - 继承算子基础参数结构，仅添加 dtype 字段
  // ============================================================================
  struct {
    {
      OpName
    }
  } InferDTypeUtParam : public {
    {
      OpName
    }
  }
  HostUtParamBase {
    // ========== 输入 Tensor 数据类型 ==========
    // 使用 ge::DataType 类型，不是 TensorDescription
    ge::DataType input0 = ge::DT_UNDEFINED;
    ge::DataType input1 = ge::DT_UNDEFINED;
    ge::DataType input_optional = ge::DT_UNDEFINED;
    // ... 根据算子定义添加所有输入 Tensor ...

    // ========== 输出 Tensor 数据类型 ==========
    ge::DataType output0 = ge::DT_UNDEFINED;
    ge::DataType output_optional = ge::DT_UNDEFINED;
    // ... 根据算子定义添加所有输出 Tensor ...

    {
      {
        OpName
      }
    }
    InferDTypeUtParam(const csv_map &csvMap) : {
      {
        OpName
      }
    }
    HostUtParamBase(csvMap) {
      // ========== 从 CSV 读取输入数据类型 ==========
      // 使用 GetDataTypeGE() 函数读取数据类型（不是 GetTensorGE）
      // 返回值：1 表示数据类型存在，0 表示数据类型为空

      this->inputInstance.emplace_back(
          GetDataTypeGE(csvMap, "input0_dtype", this->input0));
      this->inputInstance.emplace_back(
          GetDataTypeGE(csvMap, "input1_dtype", this->input1));
      this->inputInstance.emplace_back(
          GetDataTypeGE(csvMap, "input_optional_dtype", this->input_optional));

      // ... 添加所有输入 Tensor 的读取逻辑 ...

      // ========== 从 CSV 读取输出数据类型 ==========
      this->outputInstance.emplace_back(
          GetDataTypeGE(csvMap, "output0_dtype", this->output0));
      this->outputInstance.emplace_back(GetDataTypeGE(
          csvMap, "output_optional_dtype", this->output_optional));

      // ... 添加所有输出 Tensor 的读取逻辑 ...
    }
  };

} // namespace {{OpName}}UT

#endif // {{OP_NAME_UPPER}}_PARAM_H