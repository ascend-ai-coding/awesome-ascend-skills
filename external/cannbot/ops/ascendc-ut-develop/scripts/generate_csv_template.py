# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Ascend C Host UT CSV 重构辅助脚本

功能：
1. 从 xxx_def.cpp 自动提取算子定义信息（输入/输出/属性）
2. 自动生成 CSV 列名行模板
3. 自动生成 param.h 结构体框架
"""

import re
import sys
import logging
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TensorInfo:
    """Tensor 信息"""
    name: str
    param_type: str  # REQUIRED, DYNAMIC, OPTIONAL
    data_types: List[str]
    formats: List[str]


@dataclass
class AttrInfo:
    """属性信息"""
    name: str
    attr_type: str  # REQUIRED, OPTIONAL
    cpp_type: str   # Int, Float, String, Bool
    default_value: str


def parse_def_file(file_path: str) -> Tuple[List[TensorInfo], List[TensorInfo], List[AttrInfo]]:
    """
    解析 xxx_def.cpp 文件，提取算子定义信息
    
    Args:
        file_path: def.cpp 文件路径
    
    Returns:
        (inputs, outputs, attrs): 输入/输出/属性列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    inputs = []
    outputs = []
    attrs = []
    
    # 提取输入 Tensor
    input_pattern = r'this->Input\("([^"]+)"\)\s*\.ParamType\(([^)]+)\)'
    for match in re.finditer(input_pattern, content):
        name = match.group(1)
        param_type = match.group(2)
        
        # 提取 DataType（简化处理，实际可能需要更复杂的解析）
        dtype_pattern = rf'this->Input\("{name}"\).*?\.DataTypeList?\(\{{([^}}]+)\}}\)'
        dtype_match = re.search(dtype_pattern, content, re.DOTALL)
        data_types = []
        if dtype_match:
            dtypes_str = dtype_match.group(1)
            data_types = re.findall(r'ge::DT_\w+', dtypes_str)
        
        # 提取 Format
        format_pattern = rf'this->Input\("{name}"\).*?\.FormatList\(\{{([^}}]+)\}}\)'
        format_match = re.search(format_pattern, content, re.DOTALL)
        formats = []
        if format_match:
            formats_str = format_match.group(1)
            formats = re.findall(r'ge::FORMAT_\w+', formats_str)
        
        inputs.append(TensorInfo(name, param_type, data_types[:5], formats[:1]))  # 只取前5个dtype示例
    
    # 提取输出 Tensor
    output_pattern = r'this->Output\("([^"]+)"\)\s*\.ParamType\(([^)]+)\)'
    for match in re.finditer(output_pattern, content):
        name = match.group(1)
        param_type = match.group(2)
        
        dtype_pattern = rf'this->Output\("{name}"\).*?\.DataTypeList?\(\{{([^}}]+)\}}\)'
        dtype_match = re.search(dtype_pattern, content, re.DOTALL)
        data_types = []
        if dtype_match:
            dtypes_str = dtype_match.group(1)
            data_types = re.findall(r'ge::DT_\w+', dtypes_str)
        
        format_pattern = rf'this->Output\("{name}"\).*?\.FormatList\(\{{([^}}]+)\}}\)'
        format_match = re.search(format_pattern, content, re.DOTALL)
        formats = []
        if format_match:
            formats_str = format_match.group(1)
            formats = re.findall(r'ge::FORMAT_\w+', formats_str)
        
        outputs.append(TensorInfo(name, param_type, data_types[:5], formats[:1]))
    
    # 提取属性
    attr_pattern = r'this->Attr\("([^"]+)"\)\.AttrType\(([^)]+)\)\.(\w+)\(([^)]+)\)'
    for match in re.finditer(attr_pattern, content):
        name = match.group(1)
        attr_type = match.group(2)
        cpp_type = match.group(3)
        default_value = match.group(4)
        
        attrs.append(AttrInfo(name, attr_type, cpp_type, default_value))
    
    return inputs, outputs, attrs


def generate_csv_headers(inputs: List[TensorInfo], outputs: List[TensorInfo], 
                         attrs: List[AttrInfo], test_type: str = "infershape") -> str:
    """
    生成 CSV 列名行
    
    Args:
        inputs: 输入 Tensor 列表
        outputs: 输出 Tensor 列表
        attrs: 属性列表
        test_type: 测试类型（tiling 或 infershape）
    
    Returns:
        CSV 列名行字符串
    """
    headers = ["case_name", "expectResult"]
    
    # 输入 Tensor 列
    for tensor in inputs:
        headers.extend([f"{tensor.name}_shape", f"{tensor.name}_dtype", f"{tensor.name}_format"])
    
    # 输出 Tensor 列
    for tensor in outputs:
        headers.extend([f"{tensor.name}_shape", f"{tensor.name}_dtype", f"{tensor.name}_format"])
    
    # 属性列
    for attr in attrs:
        headers.append(attr.name)
    
    # 测试类型专用列
    if test_type == "tiling":
        headers.extend(["expectTilingKey", "expectTilingDataHash"])
    
    headers.append("case_annotation")
    
    return ",".join(headers)


def generate_param_struct(inputs: List[TensorInfo], outputs: List[TensorInfo], 
                         attrs: List[AttrInfo], op_name: str) -> str:
    """
    生成 param.h 结构体框架（简化版）
    
    Args:
        inputs: 输入 Tensor 列表
        outputs: 输出 Tensor 列表
        attrs: 属性列表
        op_name: 算子名称（小写）
    
    Returns:
        结构体代码框架
    """
    op_name_camel = "".join(word.capitalize() for word in op_name.split("_"))
    op_name_upper = op_name.upper()
    
    code = f"// {op_name_camel} 参数结构体框架\n"
    code += f"// 需要手动调整和补充完整\n\n"
    
    # 属性字段
    code += "// 属性字段\n"
    for attr in attrs:
        cpp_type_map = {
            "Int": "int64_t",
            "Float": "float",
            "String": "std::string",
            "Bool": "bool"
        }
        cpp_type = cpp_type_map.get(attr.cpp_type, "int64_t")
        code += f"{cpp_type} {attr.name};  // {attr.attr_type}, default: {attr.default_value}\n"
    
    # 输入 Tensor 字段
    code += "\n// 输入 Tensor 字段（Tiling 测试）\n"
    for tensor in inputs:
        code += f"gert::TilingContextPara::TensorDescription {tensor.name} = TD_DEFAULT;  // {tensor.param_type}\n"
    
    # 输出 Tensor 字段
    code += "\n// 输出 Tensor 字段（Tiling 测试）\n"
    for tensor in outputs:
        code += f"gert::TilingContextPara::TensorDescription {tensor.name} = TD_DEFAULT;  // {tensor.param_type}\n"
    
    return code


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    
    if len(sys.argv) < 2:
        logging.error("使用方法: python generate_csv_template.py <xxx_def.cpp>")
        logging.error("示例: python generate_csv_template.py fused_infer_attention_score_def.cpp")
        sys.exit(1)
    
    def_file = sys.argv[1]
    
    # 解析 def 文件
    inputs, outputs, attrs = parse_def_file(def_file)
    
    logging.info(f"\n找到 {len(inputs)} 个输入 Tensor:")
    for tensor in inputs:
        logging.info(f"  - {tensor.name}: {tensor.param_type}")
    
    logging.info(f"\n找到 {len(outputs)} 个输出 Tensor:")
    for tensor in outputs:
        logging.info(f"  - {tensor.name}: {tensor.param_type}")
    
    logging.info(f"\n找到 {len(attrs)} 个属性:")
    for attr in attrs:
        logging.info(f"  - {attr.name}: {attr.cpp_type} ({attr.attr_type})")
    
    # 生成 CSV 列名行
    logging.info("\n" + "=" * 80)
    logging.info("CSV 列名行（InferShape 测试）:")
    logging.info("=" * 80)
    csv_headers = generate_csv_headers(inputs, outputs, attrs, "infershape")
    logging.info(csv_headers)
    
    logging.info("\n" + "=" * 80)
    logging.info("CSV 列名行（Tiling 测试）:")
    logging.info("=" * 80)
    csv_headers_tiling = generate_csv_headers(inputs, outputs, attrs, "tiling")
    logging.info(csv_headers_tiling)
    
    # 生成参数结构体框架
    op_name = def_file.replace("_def.cpp", "")
    logging.info("\n" + "=" * 80)
    logging.info("参数结构体框架:")
    logging.info("=" * 80)
    param_struct = generate_param_struct(inputs, outputs, attrs, op_name)
    logging.info(param_struct)
    
    # 保存到文件
    output_file = f"{op_name}_csv_template.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CSV 列名行（InferShape 测试）:\n")
        f.write("=" * 80 + "\n")
        f.write(csv_headers + "\n\n")
        f.write("=" * 80 + "\n")
        f.write("CSV 列名行（Tiling 测试）:\n")
        f.write("=" * 80 + "\n")
        f.write(csv_headers_tiling + "\n\n")
        f.write("=" * 80 + "\n")
        f.write("参数结构体框架:\n")
        f.write("=" * 80 + "\n")
        f.write(param_struct + "\n")
    
    logging.info(f"\n已保存到文件: {output_file}")
    logging.info(f"\n模板中需要修改的地方已在注释中标注，完善模板后请删除所有的注释！")


if __name__ == "__main__":
    main()