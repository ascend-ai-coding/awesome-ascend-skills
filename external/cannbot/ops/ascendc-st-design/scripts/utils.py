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
公共工具函数模块

功能：
1. dtype字符串映射与规范化
2. dtype可转换性计算
3. dtype可推导组合计算
4. dtype推导计算
5. Shape broadcast关系计算
6. Shape broadcast结果计算
7. 随机Shape生成（基于对数分段）
"""

from typing import List, Set, Optional, Union, Tuple
import itertools
import random
import math


# ==================== dtype 映射定义 ====================

# dtype 标准名称到各种别名的映射
DTYPE_ALIASES: dict = {
    "float32": ["float32", "float", "acl_float", "FLOAT", "FLOAT32", "f32"],
    "float16": ["float16", "acl_float16", "FLOAT16", "FP16", "fp16", "half", "f16"],
    "bfloat16": ["bfloat16", "acl_bf16", "BF16", "bf16", "ACL_BF16"],
    "float64": ["float64", "double", "acl_double", "DOUBLE", "FLOAT64", "f64"],
    "int8": ["int8", "acl_int8", "INT8", "i8", "s8"],
    "uint8": ["uint8", "acl_uint8", "UINT8", "u8"],
    "int16": ["int16", "acl_int16", "INT16", "i16", "s16"],
    "uint16": ["uint16", "acl_uint16", "UINT16", "u16"],
    "int32": ["int32", "acl_int32", "INT32", "i32", "s32"],
    "uint32": ["uint32", "acl_uint32", "UINT32", "u32"],
    "int64": ["int64", "acl_int64", "INT64", "i64", "s64"],
    "uint64": ["uint64", "acl_uint64", "UINT64", "u64"],
    "bool": ["bool", "acl_bool", "BOOL", "boolean"],
    "complex32": ["complex32", "acl_complex32", "COMPLEX32", "c32"],
    "complex64": ["complex64", "acl_complex64", "COMPLEX64", "c64"],
    "complex128": ["complex128", "acl_complex128", "COMPLEX128", "c128"],
}

# 反向映射：别名 -> 标准名称
_ALIAS_TO_STANDARD: dict = {}
_STANDARD_DTYPES: set = set()

for standard, aliases in DTYPE_ALIASES.items():
    _STANDARD_DTYPES.add(standard)
    for alias in aliases:
        _ALIAS_TO_STANDARD[alias.lower()] = standard
        _ALIAS_TO_STANDARD[alias] = standard


def normalize_dtype(dtype_str: Optional[Union[str, int]]) -> Optional[str]:
    """
    将dtype字符串映射为标准名称

    Args:
        dtype_str: dtype字符串，可以是各种格式
            - "FLOAT" / "float" / "FLOAT32" / "float32" -> "float32"
            - "FLOAT16" / "float16" / "FP16" -> "float16"
            - "INT32" / "int32" -> "int32"
            ...

    Returns:
        标准化的dtype名称，如果无法识别则返回 None

    Examples:
        >>> normalize_dtype("FLOAT")
        'float32'
        >>> normalize_dtype("float16")
        'float16'
        >>> normalize_dtype("INT32")
        'int32'
        >>> normalize_dtype("unknown")
        None
    """
    if dtype_str is None:
        return None

    if isinstance(dtype_str, int):
        return None

    dtype_str = str(dtype_str).strip()

    result = _ALIAS_TO_STANDARD.get(dtype_str)
    if result:
        return result

    return _ALIAS_TO_STANDARD.get(dtype_str.lower())


def normalize_dtype_list(dtype_list: List[str]) -> List[str]:
    """
    批量规范化dtype列表

    Args:
        dtype_list: dtype字符串列表

    Returns:
        规范化后的dtype列表（过滤掉无法识别的）

    Examples:
        >>> normalize_dtype_list(["FLOAT", "FLOAT16", "INT32"])
        ['float32', 'float16', 'int32']
    """
    result: List[str] = []
    for dtype_str in dtype_list:
        normalized = normalize_dtype(dtype_str)
        if normalized is not None:
            result.append(normalized)
    return result


# ==================== dtype 类型分类 ====================

# 整数类型集合（标准名称）
INTEGER_DTYPES: Set[str] = {
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
}

# 浮点类型集合（标准名称）
FLOAT_DTYPES: Set[str] = {"float16", "float32", "float64", "bfloat16"}

# 复数类型集合（标准名称）
COMPLEX_DTYPES: Set[str] = {"complex32", "complex64", "complex128"}

# BOOL类型
BOOL_DTYPE: str = "bool"


def get_dtype_category(dtype: Optional[str]) -> Optional[str]:
    """
    获取dtype的类型分类

    Args:
        dtype: dtype字符串（可以是各种格式，会自动规范化）

    Returns:
        类型分类: "integer", "float", "complex", "bool" 或 None
    """
    if dtype is None:
        return None

    normalized = normalize_dtype(dtype)
    if normalized is None:
        return None

    if normalized in INTEGER_DTYPES:
        return "integer"
    elif normalized in FLOAT_DTYPES:
        return "float"
    elif normalized in COMPLEX_DTYPES:
        return "complex"
    elif normalized == BOOL_DTYPE:
        return "bool"

    return None


# ==================== dtype 可转换性计算 ====================


def get_convertible_source_dtypes(
    target_dtype: str, source_dtypes: List[str]
) -> List[str]:
    """
    根据 dtype 可转换规则，从原始 dtype 列表中筛选出可转换为目标 dtype 的列表

    可转换规则（参考：互转换关系.md）：
    1. 整数类型间可以转换，也支持往浮点、复数类型转换
    2. 浮点类型间可以转换，也支持往复数类型转换
    3. 复数类型间可以转换
    4. BOOL支持往整数、浮点、复数类型转换
    5. 其他场景不支持转换

    Args:
        target_dtype: 目标 dtype（需要转换到的类型）
        source_dtypes: 原始支持的 dtype 列表

    Returns:
        原始 dtype 列表中可转换为目标 dtype 的子列表

    Examples:
        >>> get_convertible_source_dtypes("float32", ["float32", "float16", "int32", "bool"])
        ['float32', 'float16', 'int32', 'bool']

        >>> get_convertible_source_dtypes("float32", ["float16", "int32", "int64", "complex64"])
        ['float16', 'int32', 'int64']

        >>> get_convertible_source_dtypes("int32", ["float32", "int8", "uint8", "bool"])
        ['int8', 'uint8', 'bool']

        >>> get_convertible_source_dtypes("complex64", ["float16", "float32", "int32", "bool"])
        ['float16', 'float32', 'int32', 'bool']

        >>> get_convertible_source_dtypes("bool", ["int32", "float32", "bool"])
        ['bool']
    """
    target_normalized = normalize_dtype(target_dtype)
    if target_normalized is None:
        return []

    normalized_sources = normalize_dtype_list(source_dtypes)

    target_category = get_dtype_category(target_normalized)
    if target_category is None:
        return []

    convertible: List[str] = []

    for source_dtype in normalized_sources:
        source_category = get_dtype_category(source_dtype)

        if source_category is None:
            continue

        if _can_convert(
            source_category, source_dtype, target_category, target_normalized
        ):
            convertible.append(source_dtype)

    return convertible


def _can_convert(
    source_category: str, source_dtype: str, target_category: str, target_dtype: str
) -> bool:
    """
    判断源 dtype 是否可以转换到目标 dtype

    Args:
        source_category: 源 dtype 的类型分类
        source_dtype: 源 dtype（已规范化）
        target_category: 目标 dtype 的类型分类
        target_dtype: 目标 dtype（已规范化）

    Returns:
        是否可以转换
    """
    if source_dtype == target_dtype:
        return True

    # 目标是整数：可以从整数、BOOL转换
    if target_category == "integer":
        return source_category in ("integer", "bool")

    # 目标是浮点：可以从整数、浮点、BOOL转换
    if target_category == "float":
        return source_category in ("integer", "float", "bool")

    # 目标是复数：可以从整数、浮点、复数、BOOL转换
    if target_category == "complex":
        return source_category in ("integer", "float", "complex", "bool")

    # 目标是BOOL：只能从BOOL转换（BOOL不接收其他类型的转换）
    if target_category == "bool":
        return source_category == "bool"

    return False


def can_convert_dtype(source_dtype: str, target_dtype: str) -> bool:
    """
    判断单个 dtype 是否可以转换到目标 dtype

    Args:
        source_dtype: 源 dtype
        target_dtype: 目标 dtype

    Returns:
        是否可以转换

    Examples:
        >>> can_convert_dtype("int32", "float32")
        True
        >>> can_convert_dtype("float32", "int32")
        False
        >>> can_convert_dtype("bool", "float32")
        True
        >>> can_convert_dtype("float32", "bool")
        False
    """
    source_normalized = normalize_dtype(source_dtype)
    target_normalized = normalize_dtype(target_dtype)

    if source_normalized is None or target_normalized is None:
        return False

    source_category = get_dtype_category(source_normalized)
    target_category = get_dtype_category(target_normalized)

    if source_category is None or target_category is None:
        return False

    return _can_convert(
        source_category, source_normalized, target_category, target_normalized
    )


# ==================== dtype 推导相关 ====================

# dtype推导表（参考：互推导关系.md）
# 格式: (dtype1, dtype2) -> result_dtype
# None 表示不能推导
# 表格行/列顺序: f32, f16, f64, bf16, s8, u8, s16, u16, s32, u32, s64, u64, bool, c32, c64, c128
DTYPE_INFER_TABLE: dict = {
    # f32 行
    ("float32", "float32"): "float32",
    ("float32", "float16"): "float32",
    ("float32", "float64"): "float64",
    ("float32", "bfloat16"): "float32",
    ("float32", "int8"): "float32",
    ("float32", "uint8"): "float32",
    ("float32", "int16"): "float32",
    ("float32", "uint16"): None,
    ("float32", "int32"): "float32",
    ("float32", "uint32"): None,
    ("float32", "int64"): "float32",
    ("float32", "uint64"): None,
    ("float32", "bool"): "float32",
    ("float32", "complex32"): "complex64",
    ("float32", "complex64"): "complex64",
    ("float32", "complex128"): "complex128",
    # f16 行
    ("float16", "float32"): "float32",
    ("float16", "float16"): "float16",
    ("float16", "float64"): "float64",
    ("float16", "bfloat16"): "float32",
    ("float16", "int8"): "float16",
    ("float16", "uint8"): "float16",
    ("float16", "int16"): "float16",
    ("float16", "uint16"): None,
    ("float16", "int32"): "float16",
    ("float16", "uint32"): None,
    ("float16", "int64"): "float16",
    ("float16", "uint64"): None,
    ("float16", "bool"): "float16",
    ("float16", "complex32"): "complex32",
    ("float16", "complex64"): "complex64",
    ("float16", "complex128"): "complex128",
    # f64 行
    ("float64", "float32"): "float64",
    ("float64", "float16"): "float64",
    ("float64", "float64"): "float64",
    ("float64", "bfloat16"): "float64",
    ("float64", "int8"): "float64",
    ("float64", "uint8"): "float64",
    ("float64", "int16"): "float64",
    ("float64", "uint16"): None,
    ("float64", "int32"): "float64",
    ("float64", "uint32"): None,
    ("float64", "int64"): "float64",
    ("float64", "uint64"): None,
    ("float64", "bool"): "float64",
    ("float64", "complex32"): "complex128",
    ("float64", "complex64"): "complex128",
    ("float64", "complex128"): "complex128",
    # bf16 行
    ("bfloat16", "float32"): "float32",
    ("bfloat16", "float16"): "float32",
    ("bfloat16", "float64"): "float64",
    ("bfloat16", "bfloat16"): "bfloat16",
    ("bfloat16", "int8"): "bfloat16",
    ("bfloat16", "uint8"): "bfloat16",
    ("bfloat16", "int16"): "bfloat16",
    ("bfloat16", "uint16"): None,
    ("bfloat16", "int32"): "bfloat16",
    ("bfloat16", "uint32"): None,
    ("bfloat16", "int64"): "bfloat16",
    ("bfloat16", "uint64"): None,
    ("bfloat16", "bool"): "bfloat16",
    ("bfloat16", "complex32"): "complex32",
    ("bfloat16", "complex64"): "complex64",
    ("bfloat16", "complex128"): "complex128",
    # s8 行
    ("int8", "float32"): "float32",
    ("int8", "float16"): "float16",
    ("int8", "float64"): "float64",
    ("int8", "bfloat16"): "bfloat16",
    ("int8", "int8"): "int8",
    ("int8", "uint8"): "int16",
    ("int8", "int16"): "int16",
    ("int8", "uint16"): None,
    ("int8", "int32"): "int32",
    ("int8", "uint32"): None,
    ("int8", "int64"): "int64",
    ("int8", "uint64"): None,
    ("int8", "bool"): "int8",
    ("int8", "complex32"): "complex32",
    ("int8", "complex64"): "complex64",
    ("int8", "complex128"): "complex128",
    # u8 行
    ("uint8", "float32"): "float32",
    ("uint8", "float16"): "float16",
    ("uint8", "float64"): "float64",
    ("uint8", "bfloat16"): "bfloat16",
    ("uint8", "int8"): "int16",
    ("uint8", "uint8"): "uint8",
    ("uint8", "int16"): "int16",
    ("uint8", "uint16"): None,
    ("uint8", "int32"): "int32",
    ("uint8", "uint32"): None,
    ("uint8", "int64"): "int64",
    ("uint8", "uint64"): None,
    ("uint8", "bool"): "uint8",
    ("uint8", "complex32"): "complex32",
    ("uint8", "complex64"): "complex64",
    ("uint8", "complex128"): "complex128",
    # s16 行
    ("int16", "float32"): "float32",
    ("int16", "float16"): "float16",
    ("int16", "float64"): "float64",
    ("int16", "bfloat16"): "bfloat16",
    ("int16", "int8"): "int16",
    ("int16", "uint8"): "int16",
    ("int16", "int16"): "int16",
    ("int16", "uint16"): None,
    ("int16", "int32"): "int32",
    ("int16", "uint32"): None,
    ("int16", "int64"): "int64",
    ("int16", "uint64"): None,
    ("int16", "bool"): "int16",
    ("int16", "complex32"): "complex32",
    ("int16", "complex64"): "complex64",
    ("int16", "complex128"): "complex128",
    # u16 行
    ("uint16", "float32"): None,
    ("uint16", "float16"): None,
    ("uint16", "float64"): None,
    ("uint16", "bfloat16"): None,
    ("uint16", "int8"): None,
    ("uint16", "uint8"): None,
    ("uint16", "int16"): None,
    ("uint16", "uint16"): "uint16",
    ("uint16", "int32"): None,
    ("uint16", "uint32"): None,
    ("uint16", "int64"): None,
    ("uint16", "uint64"): None,
    ("uint16", "bool"): None,
    ("uint16", "complex32"): None,
    ("uint16", "complex64"): None,
    ("uint16", "complex128"): None,
    # s32 行
    ("int32", "float32"): "float32",
    ("int32", "float16"): "float16",
    ("int32", "float64"): "float64",
    ("int32", "bfloat16"): "bfloat16",
    ("int32", "int8"): "int32",
    ("int32", "uint8"): "int32",
    ("int32", "int16"): "int32",
    ("int32", "uint16"): None,
    ("int32", "int32"): "int32",
    ("int32", "uint32"): None,
    ("int32", "int64"): "int64",
    ("int32", "uint64"): None,
    ("int32", "bool"): "int32",
    ("int32", "complex32"): "complex32",
    ("int32", "complex64"): "complex64",
    ("int32", "complex128"): "complex128",
    # u32 行
    ("uint32", "float32"): None,
    ("uint32", "float16"): None,
    ("uint32", "float64"): None,
    ("uint32", "bfloat16"): None,
    ("uint32", "int8"): None,
    ("uint32", "uint8"): None,
    ("uint32", "int16"): None,
    ("uint32", "uint16"): None,
    ("uint32", "int32"): None,
    ("uint32", "uint32"): "uint32",
    ("uint32", "int64"): None,
    ("uint32", "uint64"): None,
    ("uint32", "bool"): None,
    ("uint32", "complex32"): None,
    ("uint32", "complex64"): None,
    ("uint32", "complex128"): None,
    # s64 行
    ("int64", "float32"): "float32",
    ("int64", "float16"): "float16",
    ("int64", "float64"): "float64",
    ("int64", "bfloat16"): "bfloat16",
    ("int64", "int8"): "int64",
    ("int64", "uint8"): "int64",
    ("int64", "int16"): "int64",
    ("int64", "uint16"): None,
    ("int64", "int32"): "int64",
    ("int64", "uint32"): None,
    ("int64", "int64"): "int64",
    ("int64", "uint64"): None,
    ("int64", "bool"): "int64",
    ("int64", "complex32"): "complex32",
    ("int64", "complex64"): "complex64",
    ("int64", "complex128"): "complex128",
    # u64 行
    ("uint64", "float32"): None,
    ("uint64", "float16"): None,
    ("uint64", "float64"): None,
    ("uint64", "bfloat16"): None,
    ("uint64", "int8"): None,
    ("uint64", "uint8"): None,
    ("uint64", "int16"): None,
    ("uint64", "uint16"): None,
    ("uint64", "int32"): None,
    ("uint64", "uint32"): None,
    ("uint64", "int64"): None,
    ("uint64", "uint64"): "uint64",
    ("uint64", "bool"): None,
    ("uint64", "complex32"): None,
    ("uint64", "complex64"): None,
    ("uint64", "complex128"): None,
    # bool 行
    ("bool", "float32"): "float32",
    ("bool", "float16"): "float16",
    ("bool", "float64"): "float64",
    ("bool", "bfloat16"): "bfloat16",
    ("bool", "int8"): "int8",
    ("bool", "uint8"): "uint8",
    ("bool", "int16"): "int16",
    ("bool", "uint16"): None,
    ("bool", "int32"): "int32",
    ("bool", "uint32"): None,
    ("bool", "int64"): "int64",
    ("bool", "uint64"): None,
    ("bool", "bool"): "bool",
    ("bool", "complex32"): "complex32",
    ("bool", "complex64"): "complex64",
    ("bool", "complex128"): "complex128",
    # c32 行
    ("complex32", "float32"): "complex64",
    ("complex32", "float16"): "complex32",
    ("complex32", "float64"): "complex128",
    ("complex32", "bfloat16"): "complex32",
    ("complex32", "int8"): "complex32",
    ("complex32", "uint8"): "complex32",
    ("complex32", "int16"): "complex32",
    ("complex32", "uint16"): None,
    ("complex32", "int32"): "complex32",
    ("complex32", "uint32"): None,
    ("complex32", "int64"): "complex32",
    ("complex32", "uint64"): None,
    ("complex32", "bool"): "complex32",
    ("complex32", "complex32"): "complex32",
    ("complex32", "complex64"): "complex64",
    ("complex32", "complex128"): "complex128",
    # c64 行
    ("complex64", "float32"): "complex64",
    ("complex64", "float16"): "complex64",
    ("complex64", "float64"): "complex128",
    ("complex64", "bfloat16"): "complex64",
    ("complex64", "int8"): "complex64",
    ("complex64", "uint8"): "complex64",
    ("complex64", "int16"): "complex64",
    ("complex64", "uint16"): None,
    ("complex64", "int32"): "complex64",
    ("complex64", "uint32"): None,
    ("complex64", "int64"): "complex64",
    ("complex64", "uint64"): None,
    ("complex64", "bool"): "complex64",
    ("complex64", "complex32"): "complex64",
    ("complex64", "complex64"): "complex64",
    ("complex64", "complex128"): "complex128",
    # c128 行
    ("complex128", "float32"): "complex128",
    ("complex128", "float16"): "complex128",
    ("complex128", "float64"): "complex128",
    ("complex128", "bfloat16"): "complex128",
    ("complex128", "int8"): "complex128",
    ("complex128", "uint8"): "complex128",
    ("complex128", "int16"): "complex128",
    ("complex128", "uint16"): None,
    ("complex128", "int32"): "complex128",
    ("complex128", "uint32"): None,
    ("complex128", "int64"): "complex128",
    ("complex128", "uint64"): None,
    ("complex128", "bool"): "complex128",
    ("complex128", "complex32"): "complex128",
    ("complex128", "complex64"): "complex128",
    ("complex128", "complex128"): "complex128",
}


def infer_two_dtypes(dtype1: str, dtype2: str) -> Optional[str]:
    """
    根据两个dtype推导结果dtype（基于推导表）

    Args:
        dtype1: 第一个dtype
        dtype2: 第二个dtype

    Returns:
        推导后的dtype，如果不能推导则返回None

    Examples:
        >>> infer_two_dtypes("float16", "float32")
        'float32'
        >>> infer_two_dtypes("float16", "bfloat16")
        'float32'
        >>> infer_two_dtypes("float32", "uint16")
        None
    """
    d1 = normalize_dtype(dtype1)
    d2 = normalize_dtype(dtype2)

    if d1 is None or d2 is None:
        return None

    if d1 == d2:
        return d1

    result = DTYPE_INFER_TABLE.get((d1, d2))
    return result


def infer_dtypes(dtype_list: List[str]) -> Optional[str]:
    """
    根据dtype列表计算推导后的dtype
    多个dtype依次两两推导，得到最终结果

    Args:
        dtype_list: dtype列表

    Returns:
        推导后的dtype，如果无法推导则返回None

    Examples:
        >>> infer_dtypes(["float16", "float32"])
        'float32'
        >>> infer_dtypes(["float16", "float16", "float32"])
        'float32'
        >>> infer_dtypes(["float16", "bfloat16"])
        'float32'
        >>> infer_dtypes(["float16", "uint16"])
        None
    """
    if not dtype_list:
        return None

    normalized_list = normalize_dtype_list(dtype_list)
    if not normalized_list:
        return None

    if len(normalized_list) == 1:
        return normalized_list[0]

    result = normalized_list[0]
    for i in range(1, len(normalized_list)):
        result = infer_two_dtypes(result, normalized_list[i])
        if result is None:
            return None

    return result


def get_inferable_dtype_combinations(
    tensor_dtype_lists: List[List[str]],
) -> List[List[str]]:
    """
    根据多个tensor支持的dtype列表，计算所有支持推导的有效dtype组合

    Args:
        tensor_dtype_lists: 多个tensor参数各自支持的dtype列表
            例如: [["float16", "float32", "bfloat16"],
                   ["float16", "float32", "bfloat16"],
                   ["float16", "float32", "bfloat16"]]

    Returns:
        所有有效的dtype组合列表（每个组合中的dtype可以相互推导）
        例如: [["float16", "float16", "float16"],
               ["float16", "float16", "float32"],
               ["float16", "float16", "bfloat16"],
               ...]

    Examples:
        >>> dtypes1 = ["float16", "float32", "bfloat16"]
        >>> dtypes2 = ["float16", "float32", "bfloat16"]
        >>> result = get_inferable_dtype_combinations([dtypes1, dtypes2])
        >>> len(result) > 0
        True
        >>> # 验证每个组合都可以推导
        >>> all(infer_dtypes(combo) is not None for combo in result)
        True
    """
    if not tensor_dtype_lists:
        return []

    if len(tensor_dtype_lists) == 1:
        normalized = normalize_dtype_list(tensor_dtype_lists[0])
        return [[d] for d in normalized]

    # 规范化所有dtype列表
    normalized_lists = []
    for dtype_list in tensor_dtype_lists:
        normalized = normalize_dtype_list(dtype_list)
        if not normalized:
            return []
        normalized_lists.append(normalized)

    # 生成所有可能的组合
    all_combinations = list(itertools.product(*normalized_lists))

    # 筛选可以推导的组合
    valid_combinations = []
    for combo in all_combinations:
        if infer_dtypes(list(combo)) is not None:
            valid_combinations.append(list(combo))

    return valid_combinations


def infer_dtype(dtype1: Optional[str], dtype2: Optional[str] = None) -> Optional[str]:
    """
    类型推导：根据输入 dtype 推导结果 dtype
    （兼容旧接口，推荐使用 infer_dtypes 或 infer_two_dtypes）

    Args:
        dtype1: 第一个输入的 dtype
        dtype2: 第二个输入的 dtype（可选）

    Returns:
        推导后的 dtype

    Examples:
        >>> infer_dtype("float16", "float32")
        'float32'
        >>> infer_dtype("float16")
        'float16'
    """
    normalized_dtype1 = normalize_dtype(dtype1)

    if normalized_dtype1 is None:
        return None

    if dtype2 is None:
        return normalized_dtype1

    return infer_two_dtypes(normalized_dtype1, dtype2)


# ==================== 辅助函数 ====================


def get_all_supported_dtypes() -> List[str]:
    """获取所有支持的 dtype 列表"""
    return list(_STANDARD_DTYPES)


def is_valid_dtype(dtype: Optional[str]) -> bool:
    """检查 dtype 是否有效"""
    return normalize_dtype(dtype) is not None


def dtype_to_acl_format(dtype: Optional[str]) -> Optional[str]:
    """
    将标准 dtype 名称转换为 ACL 格式

    Args:
        dtype: 标准 dtype 名称

    Returns:
        ACL 格式的 dtype 名称

    Examples:
        >>> dtype_to_acl_format("float32")
        'ACL_FLOAT'
        >>> dtype_to_acl_format("float16")
        'ACL_FLOAT16'
        >>> dtype_to_acl_format("int32")
        'ACL_INT32'
    """
    normalized = normalize_dtype(dtype)
    if normalized is None:
        return None

    acl_map: dict = {
        "float32": "ACL_FLOAT",
        "float16": "ACL_FLOAT16",
        "float64": "ACL_DOUBLE",
        "bfloat16": "ACL_BF16",
        "int8": "ACL_INT8",
        "uint8": "ACL_UINT8",
        "int16": "ACL_INT16",
        "uint16": "ACL_UINT16",
        "int32": "ACL_INT32",
        "uint32": "ACL_UINT32",
        "int64": "ACL_INT64",
        "uint64": "ACL_UINT64",
        "bool": "ACL_BOOL",
        "complex32": "ACL_COMPLEX32",
        "complex64": "ACL_COMPLEX64",
        "complex128": "ACL_COMPLEX128",
    }

    return acl_map.get(normalized)


# ==================== Shape 相关函数 ====================

# 常量定义
MAX_SHAPE_PRODUCT = 2 * 1024 * 1024 * 1024  # 2G
MAX_DIM_VALUE = 2 * 1024 * 1024 * 1024  # 单轴最大值2G
MIN_DIM_VALUE = 1
MAX_DIMENSIONS = 8  # 最大维度数
NUM_LOG_SEGMENTS = 500  # 对数分段数量


def _calculate_shape_product(shape: List[int]) -> int:
    """计算shape的乘积"""
    product = 1
    for dim in shape:
        product *= dim
        if product > MAX_SHAPE_PRODUCT:
            return MAX_SHAPE_PRODUCT + 1
    return product


def generate_random_shape(
    dimensions: int, num_segments: int = NUM_LOG_SEGMENTS, seed: Optional[int] = None
) -> List[int]:
    """
    根据输入维度生成随机shape

    生成策略：
    1. 将shape乘积范围[1, 2G]按对数划分成num_segments段
    2. 随机选择一个分段，在该分段内随机生成目标乘积
    3. 将目标乘积分解到各个维度，确保维度值分布合理

    约束：
    - shape乘积 <= 2G
    - 单轴取值范围 [1, 2G]
    - 维度数 <= 8

    Args:
        dimensions: shape的维度数（1-8）
        num_segments: 对数分段数量，默认500
        seed: 随机种子（可选）

    Returns:
        随机生成的shape列表

    Examples:
        >>> shape = generate_random_shape(3, seed=42)
        >>> len(shape)
        3
        >>> _calculate_shape_product(shape) <= MAX_SHAPE_PRODUCT
        True
    """
    if seed is not None:
        random.seed(seed)

    # 参数校验
    if not isinstance(dimensions, int) or dimensions <= 0 or dimensions > MAX_DIMENSIONS:
        corrected = max(1, min(abs(int(dimensions)) if isinstance(dimensions, (int, float)) else 1, MAX_DIMENSIONS))
        dimensions = corrected
    else:
        dimensions = max(1, min(dimensions, MAX_DIMENSIONS))

    # 特殊情况：1维shape直接随机生成
    if dimensions == 1:
        # 直接在对数范围内随机
        log_min = 0  # log(1) = 0
        log_max = math.log2(MAX_SHAPE_PRODUCT)
        log_val = random.uniform(log_min, log_max)
        return [max(1, int(2**log_val))]

    # Step 1: 按对数划分段，随机选择一个分段
    log_min = 0  # log(1) = 0
    log_max = math.log2(MAX_SHAPE_PRODUCT)

    # 计算每段的对数范围
    log_segment_size = (log_max - log_min) / num_segments

    # 随机选择一个分段
    segment_idx = random.randint(0, num_segments - 1)
    segment_log_min = log_min + segment_idx * log_segment_size
    segment_log_max = segment_log_min + log_segment_size

    # 在分段内随机选择目标乘积的对数值
    target_log = random.uniform(segment_log_min, segment_log_max)
    target_product = int(2**target_log)

    # 确保目标乘积至少能分配到各个维度（每个维度至少为1）
    target_product = max(dimensions, target_product)
    target_product = min(target_product, MAX_SHAPE_PRODUCT)

    # Step 2: 将目标乘积分解到各个维度
    shape = _decompose_product_to_shape(target_product, dimensions)

    return shape


def _decompose_product_to_shape(target_product: int, dimensions: int) -> List[int]:
    """
    将目标乘积分解为指定维度的shape

    分解策略：
    1. 使用对数均匀分布来分配各维度的值
    2. 先生成各维度的对数值，然后转换为实际值
    3. 调整使乘积精确匹配目标值

    Args:
        target_product: 目标乘积
        dimensions: 维度数

    Returns:
        分解后的shape列表
    """
    if dimensions == 1:
        return [target_product]

    # 策略：使用随机分解
    # 1. 随机生成各个维度的"权重"
    # 2. 根据权重分配乘积

    shape = []

    # 方法1：逐步分解法
    remaining_product = target_product
    remaining_dims = dimensions

    for i in range(dimensions - 1):
        # 计算当前维度的最大可能值
        # 确保剩余乘积至少能分配给剩余维度（每维至少为1）
        max_for_this_dim = remaining_product // (remaining_dims)

        if max_for_this_dim <= 1:
            shape.append(1)
            remaining_dims -= 1
            continue

        # 使用对数均匀分布来选择当前维度的值
        # 这样可以产生更多样化的维度值分布
        log_min = 0  # log(1) = 0
        log_max = math.log2(max_for_this_dim)

        # 在对数空间均匀采样
        log_val = random.uniform(log_min, log_max)
        dim_value = max(1, min(max_for_this_dim, int(2**log_val)))

        shape.append(dim_value)
        remaining_product //= dim_value
        remaining_dims -= 1

    # 最后一个维度取剩余值
    shape.append(max(1, remaining_product))

    # 微调：如果乘积不匹配，调整最后一个维度
    actual_product = 1
    for d in shape[:-1]:
        actual_product *= d

    if actual_product > 0:
        last_dim = target_product // actual_product
        shape[-1] = max(1, last_dim)

    # 确保单轴值不超过最大值
    shape = [min(d, MAX_DIM_VALUE) for d in shape]

    return shape


def generate_random_shapes(
    dimensions: int,
    count: int = 10,
    num_segments: int = NUM_LOG_SEGMENTS,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """
    批量生成随机shape

    Args:
        dimensions: shape的维度数
        count: 需要生成的shape数量
        num_segments: 对数分段数量
        seed: 随机种子

    Returns:
        随机shape列表

    Examples:
        >>> shapes = generate_random_shapes(3, count=5, seed=42)
        >>> len(shapes)
        5
        >>> all(len(s) == 3 for s in shapes)
        True
    """
    if seed is not None:
        random.seed(seed)

    shapes = []
    for _ in range(count):
        shape = generate_random_shape(dimensions, num_segments)
        shapes.append(shape)

    return shapes


def generate_diverse_random_shapes(
    dimensions_list: List[int],
    count_per_dim: int = 5,
    num_segments: int = NUM_LOG_SEGMENTS,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """
    为多个维度生成多样化的随机shape

    Args:
        dimensions_list: 维度数列表，如[1, 2, 3, 4]
        count_per_dim: 每个维度生成的shape数量
        num_segments: 对数分段数量
        seed: 随机种子

    Returns:
        随机shape列表

    Examples:
        >>> shapes = generate_diverse_random_shapes([1, 2, 3], count_per_dim=2, seed=42)
        >>> len(shapes)
        6
    """
    if seed is not None:
        random.seed(seed)

    all_shapes = []
    for dims in dimensions_list:
        shapes = generate_random_shapes(dims, count_per_dim, num_segments)
        all_shapes.extend(shapes)

    return all_shapes


def generate_random_value_by_dtype(
    dtype: str,
    value_range: Optional[List[Union[int, float]]] = None,
    seed: Optional[int] = None
) -> Union[int, float, bool]:
    """
    按照数据类型和取值区间生成随机数

    Args:
        dtype: 数据类型（如 "float32", "int32", "bool" 等）
        value_range: 取值区间 [min, max]，如果为None则使用dtype的默认范围
        seed: 随机种子（可选）

    Returns:
        根据数据类型生成的随机值

    Examples:
        >>> generate_random_value_by_dtype("int32", [0, 100], seed=42)
        82
        >>> generate_random_value_by_dtype("float32", [0.0, 1.0], seed=42)
        0.6394267984578837
        >>> generate_random_value_by_dtype("bool", seed=42)
        False
    """
    if seed is not None:
        random.seed(seed)

    normalized = normalize_dtype(dtype)
    if normalized is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    dtype_ranges = {
        "int8": (-128, 127),
        "uint8": (0, 255),
        "int16": (-32768, 32767),
        "uint16": (0, 65535),
        "int32": (-2147483648, 2147483647),
        "uint32": (0, 4294967295),
        "int64": (-9223372036854775808, 9223372036854775807),
        "uint64": (0, 18446744073709551615),
        "float16": (-65504.0, 65504.0),
        "float32": (-3.4028235e38, 3.4028235e38),
        "float64": (-1.7976931348623157e308, 1.7976931348623157e308),
        "bfloat16": (-3.3895313892515355e38, 3.3895313892515355e38),
    }

    if normalized == "bool":
        if value_range is not None and len(value_range) == 2:
            min_val, max_val = value_range[0], value_range[1]
            if min_val == max_val:
                return bool(min_val)
            return random.choice([True, False])
        return random.choice([True, False])

    if value_range is None or len(value_range) != 2:
        value_range = dtype_ranges.get(normalized, (0, 100))

    min_val, max_val = value_range[0], value_range[1]

    def _parse_special_value(v):
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower == 'inf' or v_lower == '+inf':
                return float('inf')
            elif v_lower == '-inf':
                return float('-inf')
            elif v_lower == 'nan':
                return float('nan')
            elif v_lower == '+0':
                return 0.0
            elif v_lower == '-0':
                return -0.0
        return v

    min_val = _parse_special_value(min_val)
    max_val = _parse_special_value(max_val)

    if min_val == max_val:
        if isinstance(min_val, float) and (min_val != min_val):
            return float('nan')
        if normalized in INTEGER_DTYPES:
            return int(min_val)
        return float(min_val)

    if normalized in INTEGER_DTYPES:
        return random.randint(int(min_val), int(max_val))
    elif normalized in FLOAT_DTYPES:
        return random.uniform(float(min_val), float(max_val))
    elif normalized in COMPLEX_DTYPES:
        real = random.uniform(float(min_val), float(max_val))
        imag = random.uniform(float(min_val), float(max_val))
        return f"({real}+{imag}j)"
    else:
        return random.uniform(float(min_val), float(max_val))


def _can_broadcast_single_dim(source_dim: int, target_dim: int) -> bool:
    """
    判断单个维度是否可以broadcast

    规则：source_dim == target_dim 或 source_dim == 1
    """
    return source_dim == target_dim or source_dim == 1


def can_broadcast_to(source_shape: List[int], target_shape: List[int]) -> bool:
    """
    判断source_shape是否可以broadcast到target_shape

    Broadcast规则：
    1. 从后往前对齐维度
    2. 每个维度必须相等，或者source维度为1
    3. source维度数可以少于target

    Args:
        source_shape: 源shape
        target_shape: 目标shape

    Returns:
        是否可以broadcast

    Examples:
        >>> can_broadcast_to([1, 3], [2, 3])
        True
        >>> can_broadcast_to([3], [2, 3])
        True
        >>> can_broadcast_to([2, 3], [2, 3])
        True
        >>> can_broadcast_to([2, 4], [2, 3])
        False
    """
    source_rev = list(reversed(source_shape))
    target_rev = list(reversed(target_shape))

    for i in range(max(len(source_rev), len(target_rev))):
        if i >= len(source_rev):
            continue
        if i >= len(target_rev):
            return False

        s = source_rev[i]
        t = target_rev[i]

        if not _can_broadcast_single_dim(s, t):
            return False

    return True


def get_broadcast_result(shapes: List[List[int]]) -> Optional[List[int]]:
    """
    根据多个shape计算broadcast后的结果shape

    Broadcast规则：
    1. 从后往前对齐维度
    2. 每个维度取最大值
    3. 如果某个维度不兼容（都不为1且不相等），返回None

    Args:
        shapes: shape列表

    Returns:
        broadcast后的shape，如果不兼容则返回None

    Examples:
        >>> get_broadcast_result([[1, 3], [2, 3]])
        [2, 3]
        >>> get_broadcast_result([[3], [2, 3]])
        [2, 3]
        >>> get_broadcast_result([[2, 1], [1, 3]])
        [2, 3]
        >>> get_broadcast_result([[2, 3], [3, 4]])
        None
    """
    if not shapes:
        return None

    if len(shapes) == 1:
        return shapes[0]

    # 找到最大维度数
    max_dims = max(len(s) for s in shapes)

    result = []

    for i in range(max_dims):
        dim_values = []
        for shape in shapes:
            idx = len(shape) - 1 - i
            if idx >= 0:
                dim_values.append(shape[idx])

        # 检查兼容性并取最大值
        max_dim = 1
        for d in dim_values:
            if d != 1:
                if max_dim != 1 and max_dim != d:
                    return None
                max_dim = max(max_dim, d)

        result.append(max_dim)

    return list(reversed(result))


# def generate_broadcast_shapes(
#     source_shape: List[int], num_shapes: int = 5, seed: Optional[int] = None
# ) -> List[List[int]]:
#     """
#     根据输入shape生成满足broadcast关系的shape列表

#     Broadcast规则：从后往前对齐维度，每个维度必须相等或其中一方为1。

#     生成的shape包含以下场景（生成的shape与source_shape满足双向broadcast关系之一）：

#     | 场景 | 说明 | 示例source | 生成的shape | broadcast方向 |
#     |------|------|-----------|-------------|---------------|
#     | 相等 | 与输入shape完全相等 | [2,3,4] | [2,3,4] | 双向相等 |
#     | 维度少 | 维度数小于source，某些轴为1 | [2,3,4] | [4], [3,4], [1,4] | 生成→source |
#     | 维度多 | 维度数大于source | [2,3,4] | [6,2,3,4] | source→生成 |
#     | 维度相等 | 某些轴设为1 | [2,3,4] | [1,3,4], [2,1,4] | 生成→source |
#     | 扩展1轴 | source中为1的轴变非1 | [1,3,1] | [2,3,4], [5,3,8] | source→生成 |

#     约束：
#     - shape乘积小于2G
#     - 单轴取值范围1~2G
#     - 最大维度数8

#     Args:
#         source_shape: 输入shape
#         num_shapes: 需要生成的shape数量
#         seed: 随机种子（可选）

#     Returns:
#         满足broadcast关系的shape列表

#     Examples:
#         >>> # 无1轴的source
#         >>> shapes = generate_broadcast_shapes([2, 3, 4], num_shapes=5, seed=42)
#         >>> for s in shapes:
#         ...     assert can_broadcast_to(s, [2,3,4]) or can_broadcast_to([2,3,4], s)

#         >>> # 包含1轴的source
#         >>> shapes = generate_broadcast_shapes([1, 3, 1], num_shapes=5, seed=42)
#         >>> for s in shapes:
#         ...     assert can_broadcast_to(s, [1,3,1]) or can_broadcast_to([1,3,1], s)
#     """
#     if seed is not None:
#         random.seed(seed)

#     result = []
#     source_dims = len(source_shape)

#     # 辅助函数：验证shape是否满足乘积约束
#     def _validate_shape(shape: List[int]) -> Optional[List[int]]:
#         product = _calculate_shape_product(shape)
#         if product <= MAX_SHAPE_PRODUCT:
#             return shape
#         return None

#     # 场景1: 与输入shape相等
#     result.append(list(source_shape))

#     # 场景2: 维度比输入shape少（这些shape可以broadcast到source_shape）
#     if source_dims > 1:
#         for target_dims in range(1, source_dims):
#             shape = []
#             ones_count = random.randint(0, target_dims)
#             ones_positions = set(
#                 random.sample(range(target_dims), min(ones_count, target_dims))
#             )

#             for i in range(target_dims):
#                 source_idx = source_dims - target_dims + i
#                 if i in ones_positions:
#                     shape.append(1)
#                 else:
#                     shape.append(source_shape[source_idx])

#             validated = _validate_shape(shape)
#             if validated and validated not in result:
#                 result.append(validated)

#     # 场景3: 维度比输入shape多（source_shape可以broadcast到这些shape）
#     for extra_dims in range(1, min(4, MAX_DIMENSIONS - source_dims + 1)):
#         target_dims = source_dims + extra_dims
#         if target_dims > MAX_DIMENSIONS:
#             break

#         shape = []
#         # 前面添加额外维度（这些维度决定source_shape broadcast后的形状）
#         for i in range(extra_dims):
#             if random.random() < 0.3:
#                 shape.append(1)
#             else:
#                 shape.append(random.randint(2, 8))

#         # 添加原始维度
#         shape.extend(source_shape)

#         validated = _validate_shape(shape)
#         if validated and validated not in result:
#             result.append(validated)

#     # 场景4: 维度相等，某些轴为1（这些shape可以broadcast到source_shape）
#     for _ in range(3):
#         ones_count = random.randint(1, max(1, source_dims))
#         ones_positions = set(
#             random.sample(range(source_dims), min(ones_count, source_dims))
#         )

#         shape = []
#         for i in range(source_dims):
#             if i in ones_positions:
#                 shape.append(1)
#             else:
#                 shape.append(source_shape[i])

#         validated = _validate_shape(shape)
#         if validated and validated not in result:
#             result.append(validated)

#     # 场景5: 输入shape中轴包含1时，将1轴扩展为非1
#     # 这些shape可以让source_shape broadcast到
#     ones_in_source = [i for i, d in enumerate(source_shape) if d == 1]
#     if ones_in_source:
#         for _ in range(3):
#             shape = list(source_shape)
#             for idx in ones_in_source:
#                 if random.random() < 0.7:
#                     shape[idx] = random.randint(2, 16)

#             validated = _validate_shape(shape)
#             if validated and validated not in result:
#                 result.append(validated)

#     # 补充随机形状以满足数量要求
#     attempts = 0
#     while len(result) < num_shapes and attempts < num_shapes * 10:
#         attempts += 1

#         # 随机选择一种场景
#         scenario = random.choice(
#             ["fewer_dims", "more_dims", "equal_dims", "expand_ones"]
#         )

#         shape = None

#         if scenario == "fewer_dims" and source_dims > 1:
#             # 维度少，可以broadcast到source
#             target_dims = random.randint(1, source_dims - 1)
#             shape = []
#             for i in range(target_dims):
#                 source_idx = source_dims - target_dims + i
#                 if random.random() < 0.5:
#                     shape.append(1)
#                 else:
#                     shape.append(source_shape[source_idx])

#         elif scenario == "more_dims":
#             # 维度多，source可以broadcast到它
#             extra_dims = random.randint(1, min(4, MAX_DIMENSIONS - source_dims))
#             shape = []
#             for _ in range(extra_dims):
#                 shape.append(1 if random.random() < 0.3 else random.randint(2, 8))
#             shape.extend(source_shape)

#         elif scenario == "equal_dims":
#             # 维度相等，某些轴为1（可以broadcast到source）
#             shape = []
#             for i in range(source_dims):
#                 if random.random() < 0.3 and source_shape[i] != 1:
#                     shape.append(1)
#                 else:
#                     shape.append(source_shape[i])

#         else:  # expand_ones
#             # 只有当source有1轴时才执行
#             if ones_in_source:
#                 shape = list(source_shape)
#                 for idx in ones_in_source:
#                     if random.random() < 0.5:
#                         shape[idx] = random.randint(2, 16)
#             else:
#                 # 没有1轴时，跳过这个场景
#                 continue

#         if shape:
#             validated = _validate_shape(shape)
#             if validated and validated not in result:
#                 # 最终验证：确保满足broadcast关系
#                 if can_broadcast_to(validated, source_shape) or can_broadcast_to(
#                     source_shape, validated
#                 ):
#                     result.append(validated)

#     return result[:num_shapes]

def generate_broadcast_shapes(
    source_shape: List[int], seed: Optional[int] = None
) -> List[int]:
    """
    根据输入shape生成满足broadcast关系的shape

    Broadcast规则：从后往前对齐维度，每个维度必须相等或其中一方为1。

    Broadcast场景（生成的shape与source_shape满足双向broadcast关系之一）：
    | 场景 | 说明 | 示例source | 生成的shape | broadcast方向 |
    |------|------|-----------|-------------|---------------|
    | 相等 | 与输入shape完全相等 | [2,3,4] | [2,3,4] | 双向相等 |
    | 维度少 | 维度数小于source，某些轴为1 | [2,3,4] | [4], [3,4], [1,4] | 生成→source |
    | 维度多 | 维度数大于source | [2,3,4] | [6,2,3,4] | source→生成 |
    | 维度相等 | 某些轴设为1 | [2,3,4] | [1,3,4], [2,1,4] | 生成→source |
    | 扩展1轴 | source中为1的轴变非1 | [1,3,1] | [2,3,4], [5,3,8] | source→生成 |

    约束：
    - shape乘积小于2G
    - 单轴取值范围1~2G
    - 最大维度数8

    Args:
        source_shape: 输入shape
        seed: 随机种子（可选）

    Returns:
        满足broadcast关系的shape，从5种场景种随机选择一个场景生成并返回
    """
    # 1. 初始化随机种子
    if seed is not None:
        random.seed(seed)
    
    # 2. 校验输入合法性
    if not isinstance(source_shape, list) or not all(isinstance(x, int) and x >= 1 for x in source_shape):
        raise ValueError("source_shape必须是由正整数组成的列表")
    if len(source_shape) > MAX_DIMENSIONS:
        raise ValueError(f"source_shape维度数不能超过{MAX_DIMENSIONS}")
    
    # 3. 核心辅助函数：校验两个shape是否满足broadcast规则
    def is_broadcast_compatible(shape1: List[int], shape2: List[int]) -> bool:
        """检查shape1和shape2是否满足广播规则"""
        # 从后往前对齐维度
        for dim1, dim2 in zip(reversed(shape1), reversed(shape2)):
            if dim1 != dim2 and dim1 != 1 and dim2 != 1:
                return False
        return True
    
    # 4. 辅助函数：计算shape乘积
    def product(shape: List[int]) -> int:
        if not shape:
            return 1
        prod = 1
        for val in shape:
            prod *= val
            if prod > MAX_SHAPE_PRODUCT:
                break
        return prod
    
    # 5. 辅助函数：校验生成的shape是否满足所有约束（含广播兼容性）
    def is_valid(shape: List[int]) -> bool:
        # 约束1：维度数≤8
        if len(shape) > MAX_DIMENSIONS:
            return False
        # 约束2：单轴取值1~2G
        if any(val < MIN_DIM_VALUE or val > MAX_DIM_VALUE for val in shape):
            return False
        # 约束3：乘积<2G
        if product(shape) >= MAX_SHAPE_PRODUCT:
            return False
        # 约束4：必须与source_shape满足广播规则
        if not is_broadcast_compatible(shape, source_shape):
            return False
        return True
    
    # 6. 定义5种场景的生成函数（全场景边界修复）
    def _scene_equal() -> List[int]:
        """场景1：与source_shape完全相等"""
        return source_shape.copy()
    
    def _scene_less_dims() -> List[int]:
        """场景2：维度数小于source，从后往前对齐，可选轴设为1"""
        source_dim = len(source_shape)
        # 边界：source_dim=1时，无法生成更少维度，返回[1]（兼容广播）
        if source_dim == 1:
            return [1]
        # 随机选择生成的维度数（1 ~ source_dim-1）
        gen_dim = random.randint(1, source_dim - 1)
        # 从source_shape末尾取gen_dim个维度，随机将部分轴设为1
        gen_shape = source_shape[-gen_dim:].copy()
        # 随机选择0~gen_dim个轴设为1
        axes_to_1 = random.sample(range(gen_dim), k=random.randint(0, gen_dim))
        for idx in axes_to_1:
            gen_shape[idx] = 1
        return gen_shape
    
    def _scene_more_dims() -> List[int]:
        """场景3：维度数大于source（但不超过8），前面新增随机正整数轴（边界修复）"""
        source_dim = len(source_shape)
        # 边界：source_dim已达MAX_DIMENSIONS，无法生成更多维度，返回场景1结果
        if source_dim >= MAX_DIMENSIONS:
            return _scene_equal()
        # 随机选择生成的维度数（source_dim+1 ~ MAX_DIMENSIONS）
        gen_dim = random.randint(source_dim + 1, MAX_DIMENSIONS)
        # 前面新增的轴取1~MAX_DIM_VALUE之间的随机正整数（保证乘积<2G）
        prefix_dims = []
        remaining_product = MAX_SHAPE_PRODUCT // (product(source_shape) if source_shape else 1)
        for _ in range(gen_dim - source_dim):
            # 确保新增轴的取值不超过剩余乘积上限
            max_val = min(MAX_DIM_VALUE, remaining_product)
            val = random.randint(1, max_val)
            prefix_dims.append(val)
            remaining_product = max(1, remaining_product // val)
        # 拼接前缀轴和source_shape
        gen_shape = prefix_dims + source_shape.copy()
        return gen_shape
    
    def _scene_same_dims() -> List[int]:
        """场景4：维度数与source相等，随机将部分轴设为1"""
        gen_shape = source_shape.copy()
        # 至少选择1个轴设为1（避免与场景1重复）
        axes_to_1 = random.sample(range(len(gen_shape)), k=random.randint(1, len(gen_shape)))
        for idx in axes_to_1:
            gen_shape[idx] = 1
        return gen_shape
    
    def _scene_expand_1d() -> List[int]:
        """场景5：source中为1的轴替换为非1的随机正整数（保证广播兼容）"""
        # 先检查source_shape是否有1的轴，无则跳过该场景（避免生成不兼容shape）
        one_axes = [idx for idx, val in enumerate(source_shape) if val == 1]
        if not one_axes:
            # 无1轴时，场景5无法生成兼容shape，返回场景1的结果（兜底）
            return _scene_equal()
        
        gen_shape = source_shape.copy()
        # 将所有1的轴替换为非1的随机正整数（保证乘积<2G + 广播兼容）
        remaining_product = MAX_SHAPE_PRODUCT // (product([v for v in gen_shape if v != 1]) if gen_shape else 1)
        for idx in one_axes:
            max_val = min(MAX_DIM_VALUE, remaining_product)
            # 确保替换后的值>1且<=max_val
            val = random.randint(2, max_val) if max_val >= 2 else 2
            gen_shape[idx] = val
            remaining_product = max(1, remaining_product // val)
        return gen_shape
    
    # 7. 过滤不可用场景（避免选择无法生成的场景）
    source_dim = len(source_shape)
    available_scenes = []
    # 场景1：始终可用
    available_scenes.append(_scene_equal)
    # 场景2：source_dim>1时可用
    if source_dim > 1:
        available_scenes.append(_scene_less_dims)
    # 场景3：source_dim<MAX_DIMENSIONS时可用
    if source_dim < MAX_DIMENSIONS:
        available_scenes.append(_scene_more_dims)
    # 场景4：始终可用
    available_scenes.append(_scene_same_dims)
    # 场景5：source有1轴时可用
    if any(val == 1 for val in source_shape):
        available_scenes.append(_scene_expand_1d)
    
    # 8. 随机选择可用场景生成shape，若不满足约束则重试
    max_retries = 1000  # 最大重试次数，避免死循环
    retries = 0
    gen_shape = []
    
    while retries < max_retries:
        # 随机选择一个可用场景
        selected_scene = random.choice(available_scenes)
        gen_shape = selected_scene()
        # 校验约束（含广播兼容性）
        if is_valid(gen_shape):
            break
        retries += 1
    
    if retries >= max_retries:
        raise RuntimeError("超出最大重试次数，无法生成满足约束的shape")
    
    return gen_shape


def get_broadcastable_shapes(
    target_shape: List[int], num_shapes: int = 10, seed: Optional[int] = None
) -> List[List[int]]:
    """
    根据目标shape获取可单向推导(broadcast)至目标shape的shape列表

    这是generate_broadcast_shapes的语义化封装，返回所有可以broadcast到target_shape的shape

    Args:
        target_shape: 目标shape
        num_shapes: 需要生成的shape数量
        seed: 随机种子（可选）

    Returns:
        可以broadcast到target_shape的shape列表

    Examples:
        >>> shapes = get_broadcastable_shapes([2, 3, 4], num_shapes=5)
        >>> all(can_broadcast_to(s, [2, 3, 4]) for s in shapes)
        True
    """
    return generate_broadcast_shapes(target_shape, num_shapes, seed)


def generate_unidirectional_broadcast_shapes(
    shape1: List[int], seed: Optional[int] = None
) -> List[int]:
    """
    根据目标shape1获取可推导(broadcast)至目标shape1的shape2（仅保证shape2→shape1可广播）

    Broadcast规则：从后往前对齐维度，每个维度必须相等或其中一方为1。

    Broadcast场景：
    | 场景 | 说明 | 示例source | 生成的shape |
    |------|------|-----------|-------------|
    | 相等 | 与输入shape完全相等 | [2,3,4] | [2,3,4] |
    | 维度少 | 维度数小于source，某些轴为1 | [2,3,4] | [4], [3,4], [1,4] |
    | 维度相等 | 某些轴设为1 | [2,3,4] | [1,3,4], [2,1,4] | 生成→source |

    约束：
    - shape乘积小于2G
    - 单轴取值范围1~2G
    - 最大维度数8

    返回 ：
    - 仅保证shape2能broadcast至shape1（无需限制shape1反向广播）

    Args:
        shape1: 目标广播形状
        seed: 随机种子

    Returns:
        满足广播规则+约束的shape2
    """
    # 1. 初始化随机种子
    if seed is not None:
        random.seed(seed)
    
    # 2. 输入合法性校验
    if not isinstance(shape1, list) or not all(isinstance(x, int) and x >= 1 for x in shape1):
        raise ValueError("shape1必须是由正整数组成的列表")
    if len(shape1) > MAX_DIMENSIONS:
        raise ValueError(f"shape1维度数不能超过{MAX_DIMENSIONS}")
    
    # 3. 核心辅助函数：判断a是否能广播到b（标准广播规则）
    def can_broadcast(a: List[int], b: List[int]) -> bool:
        """检查a是否能广播到b（按numpy/pytorch广播规则）"""
        max_dim = max(len(a), len(b))
        # 补1对齐维度数
        a_padded = [1] * (max_dim - len(a)) + a
        b_padded = [1] * (max_dim - len(b)) + b
        
        # 逐维校验：相等 或 其中一方为1
        for dim_a, dim_b in zip(a_padded, b_padded):
            if dim_a != dim_b and dim_a != 1 and dim_b != 1:
                return False
        return True
    
    # 4. 辅助函数：计算shape乘积（用于约束校验）
    def calc_product(shape: List[int]) -> int:
        prod = 1
        for val in shape:
            prod *= val
            if prod > MAX_SHAPE_PRODUCT:
                break
        return prod
    
    # 5. 辅助函数：校验shape是否满足所有约束
    def is_valid_shape(shape: List[int]) -> bool:
        # 约束1：维度数≤8
        if len(shape) > MAX_DIMENSIONS:
            return False
        # 约束2：单轴取值1~2G
        if any(val < MIN_DIM_VALUE or val > MAX_DIM_VALUE for val in shape):
            return False
        # 约束3：乘积<2G
        if calc_product(shape) >= MAX_SHAPE_PRODUCT:
            return False
        return True
    
    # 6. 定义3种广播场景的生成函数
    def _scene_equal() -> List[int]:
        """场景1：与shape1完全相等"""
        return shape1.copy()
    
    def _scene_less_dims() -> List[int]:
        """场景2：维度数小于shape1，从后往前对齐，可选轴设为1"""
        shape1_dim = len(shape1)
        # 1维时特殊处理：返回[1]（唯一合法的少维度值）
        if shape1_dim == 1:
            return [1]
        # 多维时随机选择维度数（1 ~ shape1_dim-1）
        gen_dim = random.randint(1, shape1_dim - 1)
        # 从shape1末尾取gen_dim个维度，随机将部分轴设为1
        gen_shape = shape1[-gen_dim:].copy()
        # 随机选择0~gen_dim个轴设为1
        axes_to_1 = random.sample(range(gen_dim), k=random.randint(0, gen_dim))
        for idx in axes_to_1:
            gen_shape[idx] = 1
        return gen_shape
    
    def _scene_same_dims() -> List[int]:
        """场景3：维度数与shape1相等，随机将部分轴设为1"""
        gen_shape = shape1.copy()
        # 随机选择1~所有轴设为1（至少1个，避免与场景1重复）
        axes_to_1 = random.sample(range(len(gen_shape)), k=random.randint(1, len(gen_shape)))
        for idx in axes_to_1:
            gen_shape[idx] = 1
        return gen_shape
    
    # 7. 随机选择场景生成shape2，直到满足所有约束+广播规则
    scenes = [_scene_equal, _scene_less_dims, _scene_same_dims]
    max_retries = 1000  # 最大重试次数，避免死循环
    retries = 0
    shape2 = []
    
    while retries < max_retries:
        # 随机选择一个场景
        selected_scene = random.choice(scenes)
        shape2 = selected_scene()
        # 校验：约束合法 + shape2能广播到shape1
        if is_valid_shape(shape2) and can_broadcast(shape2, shape1):
            break
        retries += 1
    
    if retries >= max_retries:
        raise RuntimeError("超出最大重试次数，无法生成满足约束的广播shape2")
    
    return shape2


# ==================== 随机Shape生成测试 ====================


def _test_random_shape_generation():
    """测试随机shape生成功能"""
    print("\n" + "=" * 60)
    print("随机Shape生成功能测试")
    print("=" * 60)

    # 测试1: 基本功能测试
    print("\n1. 基本功能测试:")
    for dims in [1, 2, 3, 4, 5]:
        shape = generate_random_shape(dims, seed=42)
        product = _calculate_shape_product(shape)
        valid = product <= MAX_SHAPE_PRODUCT and len(shape) == dims
        status = "✓" if valid else "✗"
        print(
            f"  {status} 维度{dims}: {shape}, 乘积={product:,}, <2G: {product <= MAX_SHAPE_PRODUCT}"
        )

    # 测试2: 对数分段覆盖测试
    print("\n2. 对数分段覆盖测试 (生成100个shape，分析乘积分布):")
    shapes = generate_random_shapes(3, count=100, seed=42)
    products = [_calculate_shape_product(s) for s in shapes]

    # 按数量级分组统计
    ranges = [
        (1, 100, "<100"),
        (100, 1000, "100-1K"),
        (1000, 10000, "1K-10K"),
        (10000, 100000, "10K-100K"),
        (100000, 1000000, "100K-1M"),
        (1000000, 10000000, "1M-10M"),
        (10000000, 100000000, "10M-100M"),
        (100000000, MAX_SHAPE_PRODUCT, ">100M"),
    ]

    for min_val, max_val, label in ranges:
        count = sum(1 for p in products if min_val <= p < max_val)
        if count > 0:
            print(f"    {label}: {count}个")

    # 测试3: 维度值分布测试
    print("\n3. 维度值分布测试 (100个3维shape):")
    all_dims = []
    for s in shapes:
        all_dims.extend(s)

    dim_ranges = [
        (1, 1, "1"),
        (2, 10, "2-10"),
        (11, 100, "11-100"),
        (101, 1000, "101-1K"),
        (1001, 10000, "1K-10K"),
        (10001, MAX_DIM_VALUE, ">10K"),
    ]

    for min_val, max_val, label in dim_ranges:
        count = sum(1 for d in all_dims if min_val <= d <= max_val)
        pct = count / len(all_dims) * 100
        if count > 0:
            print(f"    维度值{label}: {count}个 ({pct:.1f}%)")

    # 测试4: 边界情况测试
    print("\n4. 边界情况测试:")
    # 测试1维
    shape = generate_random_shape(1, seed=42)
    print(f"  1维shape: {shape}, 乘积={_calculate_shape_product(shape)}")

    # 测试8维
    shape = generate_random_shape(8, seed=42)
    print(f"  8维shape: {shape}, 乘积={_calculate_shape_product(shape)}")

    # 测试5: 多样化shape生成
    print("\n5. 多样化shape生成测试:")
    shapes = generate_diverse_random_shapes([1, 2, 3, 4], count_per_dim=3, seed=42)
    for i, s in enumerate(shapes, 1):
        product = _calculate_shape_product(s)
        print(f"  {i}. {s} (乘积: {product:,})")

    # 测试6: 验证所有shape满足约束
    print("\n6. 约束验证测试 (1000个随机shape):")
    all_valid = True
    violations = []

    for dims in range(1, 9):
        for _ in range(125):  # 总共1000个
            shape = generate_random_shape(dims)
            product = _calculate_shape_product(shape)

            if product > MAX_SHAPE_PRODUCT:
                violations.append(("乘积超限", shape, product))
                all_valid = False

            if len(shape) != dims:
                violations.append(("维度不匹配", shape, dims))
                all_valid = False

            for d in shape:
                if d < 1 or d > MAX_DIM_VALUE:
                    violations.append(("维度值超限", shape, d))
                    all_valid = False

    if all_valid:
        print("  ✓ 所有1000个shape都满足约束条件")
    else:
        print(f"  ✗ 发现{len(violations)}个违规:")
        for v in violations[:5]:
            print(f"      {v}")

    # 测试7: 对数分段均匀性测试
    print("\n7. 对数分段均匀性测试 (500分段，每段采样1次):")
    segment_counts = [0] * 10  # 将500段分成10组统计

    for _ in range(500):
        shape = generate_random_shape(3)
        product = _calculate_shape_product(shape)
        log_product = math.log2(max(1, product))
        log_max = math.log2(MAX_SHAPE_PRODUCT)

        # 确定落在哪个组
        group = int(log_product / log_max * 10)
        group = min(9, group)
        segment_counts[group] += 1

    print("  各对数区间的shape数量分布:")
    for i, count in enumerate(segment_counts):
        log_start = i * math.log2(MAX_SHAPE_PRODUCT) / 10
        log_end = (i + 1) * math.log2(MAX_SHAPE_PRODUCT) / 10
        val_start = int(2**log_start)
        val_end = int(2**log_end)
        bar = "█" * (count // 5)
        print(f"    [{val_start:>10,} - {val_end:>10,}]: {count:3d} {bar}")


if __name__ == "__main__":
    # 运行所有测试
    print("=" * 60)
    print("Utils 模块功能测试")
    print("=" * 60)

    # dtype相关测试
    print("\n" + "=" * 60)
    print("dtype 映射测试")
    print("=" * 60)

    test_cases = ["FLOAT", "float", "FLOAT32", "float16", "FP16", "INT32", "BOOL"]
    for tc in test_cases:
        result = normalize_dtype(tc)
        print(f"normalize_dtype('{tc}') = '{result}'")

    # 随机shape生成测试
    _test_random_shape_generation()
