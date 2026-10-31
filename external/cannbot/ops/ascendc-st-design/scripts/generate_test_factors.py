#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""
测试因子提取脚本

功能：从参数定义.yaml中提取测试因子，输出测试因子.yaml
输入：参数定义.yaml
输出：测试因子.yaml

使用方法：
    python generate_test_factors.py input.yaml output.yaml
    python generate_test_factors.py input.yaml  # 输出到标准输出
"""

import yaml
import sys
import argparse
import re
from typing import Dict, List, Any
from collections import OrderedDict

SPECIAL_STRINGS = {"inf", "-inf", "nan", "+inf", "+0", "-0"}

TENSOR_TYPES = {"aclTensor", "aclTensorList"}

ARRAY_TYPES = {"aclIntArray", "aclFloatArray", "aclBoolArray", "aclScalarList"}

SCALAR_TYPES = {
    "aclScalar",
    "int4_t", "int8_t", "int16_t", "int32_t", "int64_t",
    "uint1_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "bool", "float", "float16", "bfloat16", "float32", "double",
    "char", "string",
}

_FLOAT32_RANGES = [
    [0, 0.001], [0.001, 0.01], [0.01, 1], [1, 2], [2, 10],
    [10, 1000], [-0.001, 0], [-0.01, -0.001], [-1, -0.01],
    [-2, -1], [-10, -2], [-1000, -10], [-1, 1], [-0.01, 0.01],
    [-100, 100], [-3.4028235e38, 3.4028235e38], [0, 0],
    [-0.000030517578125, 0.000030517578125],
    [3.4028235e38, 3.4028235e38],
    [-3.4028235e38, -3.4028235e38],
    [-1.1754943508e-38, -1.1754943508e-38],
    [1.1754943508e-38, 1.1754943508e-38],
    ["inf", "inf"], ["-inf", "-inf"], ["nan", "nan"],
]

DEFAULT_VALUE_RANGES = {
    "float16": [
        [0, 0.001], [0.001, 0.01], [0.01, 1], [1, 2], [2, 10],
        [10, 1000], [-0.001, 0], [-0.01, -0.001], [-1, -0.01],
        [-2, -1], [-10, -2], [-1000, -10], [-1, 1], [-0.01, 0.01],
        [-100, 100], [0, 0], [-65504.0, 65504.0],
        [-0.0078125, 0.0078125], [65504.0, 65504.0],
        [-65504.0, -65504.0],
        [-6.103515625e-05, -6.103515625e-05],
        [6.103515625e-05, 6.103515625e-05],
        ["inf", "inf"], ["-inf", "-inf"], ["nan", "nan"],
    ],
    "float32": list(_FLOAT32_RANGES),
    "float64": [
        [0, 0.001], [0.001, 0.01], [0.01, 1], [1, 2], [2, 10],
        [10, 1000], [-0.001, 0], [-0.01, -0.001], [-1, -0.01],
        [-2, -1], [-10, -2], [-1000, -10], [-1, 1], [-0.01, 0.01],
        [-100, 100], [-3.4028235e38, 3.4028235e38], [0, 0],
        [3.4028235e38, 3.4028235e38],
        [-3.4028235e38, -3.4028235e38],
        [-0.000030517578125, 0.000030517578125],
        [-1.1754943508e-38, -1.1754943508e-38],
        [1.1754943508e-38, 1.1754943508e-38],
        [1.7976931348623157e308, 1.7976931348623157e308],
        [-1.7976931348623157e308, -1.7976931348623157e308],
        [-2.2250738585072014e-308, -2.2250738585072014e-308],
        [2.2250738585072014e-308, 2.2250738585072014e-308],
        ["inf", "inf"], ["-inf", "-inf"], ["nan", "nan"],
    ],
    "bfloat16": [
        [0, 0.001], [0.001, 0.01], [0.01, 1], [-1, 1], [1, 2],
        [2, 10], [10, 1000], [-0.001, 0], [-0.01, -0.001],
        [-1, -0.01], [-2, -1], [-10, -2], [-1000, -10], [-1, 1],
        [-0.01, 0.01], [-100, 100], [-3.38e38, 3.38e38], [0, 0],
        [-0.000030517578125, 0.000030517578125],
        [3.3895313892515355e38, 3.3895313892515355e38],
        [-3.3895313892515355e38, -3.3895313892515355e38],
        [-1.1754943508e-38, -1.1754943508e-38],
        [1.1754943508e-38, 1.1754943508e-38],
        ["inf", "inf"], ["-inf", "-inf"], ["nan", "nan"],
    ],
    "hf32": list(_FLOAT32_RANGES),
    "float4_e1m2": [
        [0, 0], [-0, -0], [0.25, 0.25], [-0.25, -0.25],
        [0.5, 0.5], [-0.5, -0.5], [0.75, 0.75], [-0.75, -0.75],
        [1, 1], [-1, -1], [1.25, 1.25], [-1.25, -1.25],
        [1.5, 1.5], [-1.5, -1.5], [1.75, 1.75], [-1.75, -1.75],
    ],
    "float4_e2m1": [
        [0.5, 0.5], [-0.5, -0.5], [1, 1], [-1, -1],
        [1.5, 1.5], [-1.5, -1.5], [2, 2], [-2, -2],
        [3, 3], [-3, -3], [4, 4], [-4, -4], [6, 6], [-6, -6],
    ],
    "float8_e4m3fn": [
        [-448, 448], [2e-6, 1], [-1, -2e-6],
        [2e-9, 1.75e-06], [-1.75e-06, 2e-9], [-0, 0],
    ],
    "float8_e5m2": [
        [-57344, 57344], [2e-14, 1], [-1, -2e-14],
        [2e-16, 1.5e-14], [-1.5e-14, 2e-16], [-0, 0],
    ],
    "float8_e8m0": [
        [-127, 127], [-10, 10], [-64, 64],
        [-100, 100], [0, 10], [-10, 0],
    ],
    "hifloat8": [
        [256, 32768], [-32768, -256],
        [0.000030517578125, 0.0078125],
        [-0.0078125, -0.000030517578125],
        [16, 256], [-256, 16], [-256, -16],
        [0.0078125, 0.125], [-0.125, -0.0078125],
        [4, 16], [-16, -4], [0.125, 0.5], [-0.5, -0.125],
        [2, 4], [-4, -2], [0.5, 1], [-1, -0.5],
        [1, 2], [-2, -1],
        [0.0000002384185791015625, 0.000030517578125], [-0, 0],
    ],
    "complex32": [
        [0, 0.001], [0.001, 0.01], [0.01, 1], [1, 2], [2, 10],
        [10, 1000], [-0.001, 0], [-0.01, -0.001], [-1, -0.01],
        [-2, -1], [-10, -2], [-1000, -10], [-1, 1],
        [-0.01, 0.01], [-100, 100],
        [-3.4028235e38, 3.4028235e38], [0, 0],
        [3.4028235e38, 3.4028235e38],
        [-3.4028235e38, -3.4028235e38],
        [-1.1754943508e-38, -1.1754943508e-38],
        [1.1754943508e-38, 1.1754943508e-38],
    ],
    "complex64": [
        [0, 0.001], [0.001, 0.01], [0.01, 1], [1, 2], [2, 10],
        [10, 1000], [-0.001, 0], [-0.01, -0.001], [-1, -0.01],
        [-2, -1], [-10, -2], [-1000, -10], [-1, 1],
        [-0.01, 0.01], [-100, 100],
        [-3.4028235e38, 3.4028235e38], [0, 0],
        [-0.000030517578125, 0.000030517578125],
        [3.4028235e38, 3.4028235e38],
        [-3.4028235e38, -3.4028235e38],
        [-1.1754943508e-38, -1.1754943508e-38],
        [1.1754943508e-38, 1.1754943508e-38],
    ],
    "complex128": [
        [0, 0.001], [0.001, 0.01], [0.01, 1], [1, 2], [2, 10],
        [10, 1000], [-0.001, 0], [-0.01, -0.001], [-1, -0.01],
        [-2, -1], [-10, -2], [-1000, -10], [-1, 1],
        [-0.01, 0.01], [-100, 100],
        [-3.4028235e38, 3.4028235e38], [0, 0],
        [3.4028235e38, 3.4028235e38],
        [-3.4028235e38, -3.4028235e38],
        [-0.000030517578125, 0.000030517578125],
        [-1.1754943508e-38, -1.1754943508e-38],
        [1.1754943508e-38, 1.1754943508e-38],
        [1.7976931348623157e308, 1.7976931348623157e308],
        [-1.7976931348623157e308, -1.7976931348623157e308],
        [-2.2250738585072014e-308, -2.2250738585072014e-308],
        [2.2250738585072014e-308, 2.2250738585072014e-308],
    ],
    "int4": [
        [0, 0], [-1, 0], [0, 1], [-1, 1], [-8, 7], [-8, -8],
        [7, 7], [-8, -1], [1, 7], [-4, 4], [-2, -1], [1, 2],
    ],
    "int8": [
        [0, 1], [1, 2], [2, 10], [-1, 0], [-2, -1], [-10, -2],
        [-1, 1], [-100, 100], [-10, 10], [0, 0],
        [-128, 127], [-128, -128], [127, 127],
    ],
    "int16": [
        [0, 1], [1, 2], [2, 10], [10, 1000], [-1, 0], [-2, -1],
        [-10, -2], [-1000, -10], [-1, 1], [-100, 100], [0, 0],
        [-32768, 32767], [-32768, -32768], [32767, 32767],
    ],
    "int32": [
        [0, 1], [1, 2], [2, 10], [10, 1000], [-1, 0], [-2, -1],
        [-10, -2], [-1000, -10], [-1, 1], [-100, 100], [0, 0],
        [-2147483648, 2147483647],
        [-2147483648, -2147483648],
        [2147483647, 2147483647],
    ],
    "int64": [
        [0, 1], [1, 2], [2, 10], [10, 1000], [-1, 0], [-2, -1],
        [-10, -2], [-1000, -10], [-1, 1], [-100, 100], [0, 0],
        [-9223372036854775808, 9223372036854775807],
        [-9223372036854775808, -9223372036854775808],
        [9223372036854775807, 9223372036854775807],
    ],
    "uint1": [[0, 1]],
    "uint8": [
        [0, 1], [1, 2], [2, 10], [0, 100], [0, 10],
        [0, 255], [0, 0], [255, 255],
    ],
    "uint16": [
        [0, 1], [1, 2], [2, 10], [10, 1000], [0, 100],
        [0, 65535], [0, 0], [65535, 65535],
    ],
    "uint32": [
        [0, 1], [1, 2], [2, 10], [10, 1000], [0, 100],
        [0, 4294967295], [0, 0], [4294967295, 4294967295],
    ],
    "uint64": [
        [0, 1], [1, 2], [2, 10], [10, 1000], [0, 100],
        [0, 18446744073709551615],
        [0, 0], [18446744073709551615, 18446744073709551615],
    ],
    "qint8": [[-128, 127]],
    "qint16": [
        [-1, 1], [1, 2], [2, 10], [10, 1000], [-2, -1],
        [-10, -2], [-1000, -10], [-1, 1], [0, 1], [1, 2],
        [2, 10], [10, 1000], [0, 100], [0, 65535],
        [0, 0], [65535, 65535],
        [-32768, 32767], [-32768, -32768], [32767, 32767],
    ],
    "qint32": [[-2147483648, 2147483647]],
    "quint8": [
        [0, 1], [1, 2], [2, 10], [0, 100], [0, 10],
        [0, 255], [0, 0], [255, 255],
    ],
    "quint16": [
        [0, 1], [1, 2], [2, 10], [10, 1000], [0, 100],
        [0, 65535], [0, 0], [65535, 65535],
    ],
    "bool": [[0, 1], [0, 0], [1, 1]],
    "char": [
        [0, 1], [1, 2], [2, 10], [-1, 0], [-2, -1], [-10, -2],
        [-1, 1], [-100, 100], [-10, 10], [0, 0],
        [-128, 127], [-128, -128], [127, 127],
    ],
    "string": [],
}


def get_default_value_range(dtype: str) -> List[List]:
    """
    获取 dtype 的默认 value_range

    Args:
        dtype: 数据类型字符串

    Returns:
        默认 value_range 列表
    """
    normalized = normalize_dtype(dtype) if 'normalize_dtype' in dir() else dtype.lower()
    return DEFAULT_VALUE_RANGES.get(normalized, DEFAULT_VALUE_RANGES.get(dtype, [[0, 100]]))


def normalize_dtype(dtype_str: str) -> str:
    """
    标准化 dtype 字符串

    Args:
        dtype_str: dtype 字符串

    Returns:
        标准化后的 dtype
    """
    dtype_map = {
        "float16": "float16", "fp16": "float16", "half": "float16",
        "float32": "float32", "float": "float32", "fp32": "float32",
        "float64": "float64", "double": "float64", "fp64": "float64",
        "bfloat16": "bfloat16", "bf16": "bfloat16",
        "hf32": "hf32",
        "float4_e1m2": "float4_e1m2",
        "float4_e2m1": "float4_e2m1",
        "float8_e4m3fn": "float8_e4m3fn",
        "float8_e5m2": "float8_e5m2",
        "float8_e8m0": "float8_e8m0",
        "hifloat8": "hifloat8",
        "complex32": "complex32",
        "complex64": "complex64",
        "complex128": "complex128",
        "int4": "int4",
        "int8": "int8", "s8": "int8",
        "int16": "int16", "s16": "int16",
        "int32": "int32", "s32": "int32", "int": "int32",
        "int64": "int64", "s64": "int64", "long": "int64",
        "uint1": "uint1",
        "uint8": "uint8", "u8": "uint8",
        "uint16": "uint16", "u16": "uint16",
        "uint32": "uint32", "u32": "uint32",
        "uint64": "uint64", "u64": "uint64",
        "qint8": "qint8",
        "qint16": "qint16",
        "qint32": "qint32",
        "quint8": "quint8",
        "quint16": "quint16",
        "bool": "bool", "boolean": "bool",
        "char": "char",
        "string": "string",
    }
    return dtype_map.get(dtype_str.lower(), dtype_str.lower())


def convert_special_value(value):
    """
    转换特殊值：只有 inf/-inf/nan 保持字符串，其余转为数值

    Args:
        value: 输入值

    Returns:
        转换后的值
    """
    if isinstance(value, str):
        if value.lower() in SPECIAL_STRINGS or value in SPECIAL_STRINGS:
            return value
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    return value


def convert_range_list(range_list):
    """
    转换范围列表中的值

    Args:
        range_list: 范围列表 [[min, max], ...]

    Returns:
        转换后的范围列表
    """
    result = []
    for item in range_list:
        if isinstance(item, list):
            result.append([convert_special_value(v) for v in item])
        else:
            result.append(convert_special_value(item))
    return result


def represent_str(dumper, data):
    """自定义字符串表示器，特殊字符串保持引号"""
    if data in SPECIAL_STRINGS:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def represent_list(dumper, data):
    """自定义列表表示器，处理嵌套列表"""
    if not data:
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    if all(isinstance(x, (int, float, str)) for x in data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)


def setup_yaml_ordered_dict():
    """配置YAML使用OrderedDict保持顺序"""

    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict)
    yaml.add_representer(str, represent_str)


def load_yaml(filepath: str) -> Dict:
    """加载YAML文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, filepath: str = None):
    """保存YAML数据到文件或标准输出"""
    setup_yaml_ordered_dict()

    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(
                data, f, allow_unicode=True, default_flow_style=False, sort_keys=False
            )
        print(f"测试因子已保存到: {filepath}")
    else:
        print(
            yaml.dump(
                data, allow_unicode=True, default_flow_style=False, sort_keys=False
            )
        )


def _extract_dtype_range_factors(
    name, param, io_type, factors, dtype_key="dtype_with_ranges"
):
    if dtype_key not in param:
        return
    dtypes = [item["dtype"] for item in param[dtype_key]]
    factors[f"{name}.dtype"] = dtypes
    if io_type != "input":
        return
    for dtype_item in param[dtype_key]:
        dtype = dtype_item["dtype"]
        value_range = dtype_item.get("value_range") or get_default_value_range(dtype)
        converted = convert_range_list(value_range)
        normalized = normalize_dtype(dtype)
        if converted:
            factors[f"{name}.value_range_{normalized}"] = converted


def extract_tensor_factors(param: Dict) -> Dict[str, List]:
    """
    提取 Tensor 类参数的测试因子

    支持 aclTensor 和 aclTensorList。

    Args:
        param: 参数定义字典

    Returns:
        测试因子字典
    """
    factors = OrderedDict()
    name = param["name"]
    io_type = param.get("io_type", "input")

    factors[f"{name}.exist"] = [True]
    if not param.get("required", True):
        factors[f"{name}.exist"] = [True, False]

    if "format" in param:
        format_value = param["format"]
        if isinstance(format_value, list):
            factors[f"{name}.format"] = format_value
        else:
            factors[f"{name}.format"] = [format_value]

    if "dimensions" in param:
        factors = check_dim(param, factors, name)

    if "length_ranges" in param:
        converted_range = convert_range_list(param["length_ranges"])
        factors[f"{name}.length_ranges"] = converted_range

    _extract_dtype_range_factors(name, param, io_type, factors)

    return factors


def check_dim(param, factors, name):
    dims = param["dimensions"]
    if isinstance(dims, list):
        fixed_dims = []
        for d in dims:
            if not isinstance(d, int):
                abs_d = int(d) if isinstance(d, (int, float)) else 1
                fixed_dims.append(abs_d)
            elif d < 0:  # 只过滤负数，允许0
                fixed_dims.append(abs(d))
            elif d > 8:
                fixed_dims.append(8)
            else:
                fixed_dims.append(d)
        factors[f"{name}.dimensions"] = fixed_dims
    else:
        d = dims
        if not isinstance(d, int) or d <= 0:
            abs_d = abs(int(d)) if isinstance(d, (int, float)) else 1
            factors[f"{name}.dimensions"] = [abs_d]
        elif d > 8:
            factors[f"{name}.dimensions"] = [8]
        else:
            factors[f"{name}.dimensions"] = [d]
    return factors


def extract_scalar_factors(param: Dict) -> Dict[str, List]:
    """
    提取 Scalar 类参数的测试因子

    支持 aclScalar 及 18 种标量类型：
        int4_t, int8_t, int16_t, int32_t, int64_t,
        uint1_t, uint8_t, uint16_t, uint32_t, uint64_t,
        bool, float, float16, bfloat16, float32, double,
        char, string

    Args:
        param: 参数定义字典

    Returns:
        测试因子字典
    """
    factors = OrderedDict()
    name = param["name"]
    io_type = param.get("io_type", "input")
    is_enum = param.get("is_enum", False)

    factors[f"{name}.exist"] = [True]
    if not param.get("required", True):
        factors[f"{name}.exist"] = [True, False]

    if "dtype_with_values" in param:
        dtypes = [item["dtype"] for item in param["dtype_with_values"]]
        factors[f"{name}.dtype"] = dtypes

        if is_enum:
            for dtype_item in param["dtype_with_values"]:
                if "value" in dtype_item:
                    converted_values = [
                        convert_special_value(v) for v in dtype_item["value"]
                    ]
                    factors[f"{name}.value"] = converted_values
                    break
        elif io_type == "input":
            for dtype_item in param["dtype_with_values"]:
                dtype = dtype_item["dtype"]

                if "value_range" in dtype_item:
                    value_range = dtype_item["value_range"]
                else:
                    value_range = get_default_value_range(dtype)

                converted_range = convert_range_list(value_range)
                normalized = normalize_dtype(dtype)
                if converted_range:
                    factors[f"{name}.value_range_{normalized}"] = converted_range

    return factors


def extract_array_factors(param: Dict) -> Dict[str, List]:
    """
    提取 Array 类参数的测试因子

    支持 aclIntArray, aclFloatArray, aclBoolArray, aclScalarList。

    Args:
        param: 参数定义字典

    Returns:
        测试因子字典
    """
    factors = OrderedDict()
    name = param["name"]
    io_type = param.get("io_type", "input")

    factors[f"{name}.exist"] = [True]
    if not param.get("required", True):
        factors[f"{name}.exist"] = [True, False]

    if "length_ranges" in param:
        converted_range = convert_range_list(param["length_ranges"])
        factors[f"{name}.length_ranges"] = converted_range

    _extract_dtype_range_factors(name, param, io_type, factors)

    return factors


def extract_factors_from_param(param: Dict) -> Dict[str, List]:
    """
    根据参数类型提取测试因子

    Args:
        param: 参数定义字典

    Returns:
        测试因子字典
    """
    param_type = param.get("type", "")

    if param_type in TENSOR_TYPES:
        return extract_tensor_factors(param)
    elif param_type in ARRAY_TYPES:
        return extract_array_factors(param)
    elif param_type in SCALAR_TYPES:
        return extract_scalar_factors(param)
    else:
        print(f"⚠️  未知的参数类型: {param_type}")
        return OrderedDict()


def extract_all_factors(params: List[Dict]) -> Dict:
    """
    从所有参数中提取测试因子

    Args:
        params: 参数定义列表

    Returns:
        完整的测试因子字典（仅包含 test_factors）
    """
    factors_section = OrderedDict()

    for param in params:
        param_name = param["name"]
        param_factors = extract_factors_from_param(param)

        if param_factors:
            factors_section[param_name] = OrderedDict(
                [("type", param.get("type", "")), ("factors", param_factors)]
            )

    return factors_section


def generate_factor_summary(factors: Dict) -> Dict:
    """
    生成测试因子摘要

    Args:
        factors: 测试因子字典

    Returns:
        摘要信息
    """
    total_params = len(factors)
    total_factors = sum(len(param_data["factors"]) for param_data in factors.values())

    summary = OrderedDict(
        [
            ("total_parameters", total_params),
            ("total_factors", total_factors),
            ("by_type", OrderedDict()),
            ("by_category", OrderedDict()),
        ]
    )

    for param_name, param_data in factors.items():
        param_type = param_data["type"]
        if param_type not in summary["by_type"]:
            summary["by_type"][param_type] = {"count": 0, "factors": 0}
        summary["by_type"][param_type]["count"] += 1
        summary["by_type"][param_type]["factors"] += len(param_data["factors"])

    category_count = OrderedDict()
    for param_name, param_data in factors.items():
        for factor_name in param_data["factors"].keys():
            if "." in factor_name:
                attribute_part = factor_name.split(".", 1)[1]
                if "_" in attribute_part:
                    category = attribute_part.rsplit("_", 1)[0]
                else:
                    category = attribute_part
                category_count[category] = category_count.get(category, 0) + 1

    summary["by_category"] = category_count

    return summary


def print_factor_summary(factors: Dict):
    """打印测试因子摘要"""
    summary = generate_factor_summary(factors)

    print("\n" + "=" * 60)
    print("测试因子提取摘要")
    print("=" * 60)
    print(f"参数总数: {summary['total_parameters']}")
    print(f"因子总数: {summary['total_factors']}")
    print()

    print("按参数类型统计:")
    for ptype, stats in summary["by_type"].items():
        print(f"  {ptype}: {stats['count']}个参数, {stats['factors']}个因子")
    print()

    print("按因子类别统计:")
    for category, count in summary["by_category"].items():
        print(f"  {category}: {count}个")
    print("=" * 60 + "\n")


def validate_yaml_structure(data: Dict) -> bool:
    """验证YAML结构是否正确"""
    if "parameters" not in data:
        print("错误: YAML文件缺少 'parameters' 字段")
        return False

    params = data["parameters"]
    if not isinstance(params, list):
        print("错误: 'parameters' 必须是列表")
        return False

    for i, param in enumerate(params):
        if "name" not in param:
            print(f"错误: 参数{i + 1}缺少 'name' 字段")
            return False
        if "type" not in param:
            print(
                f"错误: 参数{i + 1} ({param.get('name', 'unknown')}) 缺少 'type' 字段"
            )
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="从参数定义.yaml中提取测试因子",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取测试因子并保存到文件
  python generate_test_factors.py params.yaml factors.yaml

  # 提取测试因子并输出到标准输出
  python generate_test_factors.py params.yaml

  # 查看帮助
  python generate_test_factors.py --help
        """,
    )

    parser.add_argument("input", help="输入的参数定义YAML文件")
    parser.add_argument(
        "output", nargs="?", help="输出的测试因子YAML文件（可选，默认输出到标准输出）"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="静默模式，不打印摘要"
    )

    args = parser.parse_args()

    try:
        print(f"正在读取: {args.input}")
        data = load_yaml(args.input)

        if not validate_yaml_structure(data):
            sys.exit(1)

        print("正在提取测试因子...")
        factors = extract_all_factors(data["parameters"])

        if not args.quiet:
            print_factor_summary(factors)

        save_yaml(factors, args.output)

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{args.input}'")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"YAML解析错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
