#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import sys
import csv
import logging
import re

_logger = logging.getLogger(__name__)


KERNEL_COLUMNS = [
    "testcase_name", "network_name", "op_name",
    "input_shapes", "input_dtypes", "input_formats",
    "output_shapes", "output_dtypes", "output_formats",
    "input_ori_shapes", "input_ori_formats",
    "output_ori_shapes", "output_ori_formats",
    "attributes", "input_data_ranges", "precision_tolerances",
    "absolute_precision",
    "output_inplace_indexes", "output_shape_unknown_indexes",
    "is_enabled", "remark", "soc_series", "priority",
    "dump_file_prefix", "manual_input_binaries", "manual_golden_binaries",
]

ACLNN_COLUMNS = [
    "testcase_name", "network_name", "api_name",
    "tensor_view_shapes", "tensor_dtypes", "tensor_formats",
    "tensor_storage_shapes", "tensor_view_offsets", "tensor_view_strides",
    "output_tensor_indexes", "output_inplace_indexes",
    "attributes", "scalar_dtypes", "scalar_data_ranges",
    "input_data_ranges", "precision_tolerances", "absolute_precision",
    "is_enabled", "remark", "soc_series", "priority",
    "dump_file_prefix", "manual_tensor_binaries", "manual_golden_binaries",
]

E2E_COLUMNS = [
    "testcase_name", "network_name", "api_name",
    "tensor_view_shapes", "tensor_dtypes", "tensor_formats",
    "tensor_storage_shapes", "tensor_view_offsets", "tensor_view_strides",
    "output_tensor_indexes", "attributes", "golden_api",
    "input_data_ranges", "precision_tolerances", "absolute_precision",
    "is_enabled", "remark", "soc_series", "priority",
]

KERNEL_REQUIRED = [
    "testcase_name", "op_name", "input_shapes", "input_dtypes", "output_dtypes", "output_shapes",
]

TUPLE_COLUMNS_KERNEL = [
    "input_dtypes", "input_formats", "output_dtypes", "output_formats",
]

COMMON_REQUIRED = ["testcase_name"]


def detect_mode(headers):
    if "api_name" in headers:
        api_values = [v.strip() for v in headers]
        return "e2e"
    return "kernel"


def get_expected_columns(mode):
    return {"kernel": KERNEL_COLUMNS, "aclnn": ACLNN_COLUMNS, "e2e": E2E_COLUMNS}[mode]


def get_required_columns(mode):
    if mode == "kernel":
        return KERNEL_REQUIRED
    return COMMON_REQUIRED


def get_tuple_columns(mode):
    if mode == "kernel":
        return TUPLE_COLUMNS_KERNEL
    return []


def check_encoding(file_path):
    passed = True
    try:
        with open(file_path, "rb") as f:
            raw = f.read()
        if raw[:3] == b"\xef\xbb\xbf":
            passed = False
            _logger.info("  [FAIL] 编码为 UTF-8 BOM，应为 UTF-8（不带 BOM）")
        else:
            raw.decode("utf-8")
            _logger.info("  [PASS] 编码为 UTF-8（不带 BOM）")
    except UnicodeDecodeError:
        passed = False
        _logger.info("  [FAIL] 无法以 UTF-8 解码")
    return passed


def check_header(headers, expected):
    passed = True
    if headers != expected:
        passed = False
        missing = set(expected) - set(headers)
        extra = set(headers) - set(expected)
        order_diff = [i for i, (e, a) in enumerate(zip(expected, headers)) if e != a]
        parts = []
        if missing:
            parts.append(f"缺少: {missing}")
        if extra:
            parts.append(f"多余: {extra}")
        if order_diff:
            parts.append(f"顺序不一致的列索引: {order_diff}")
        _logger.info("  [FAIL] 表头列名/顺序不正确: %s", '; '.join(parts))
    else:
        _logger.info("  [PASS] 表头列名和顺序正确 (%d 列)", len(expected))
    return passed


def check_row_count(row_count):
    passed = True
    if row_count == 0:
        passed = False
        _logger.info("  [FAIL] CSV 数据行数为 0")
    else:
        _logger.info("  [PASS] CSV 数据行数: %d", row_count)
    return passed


def check_testcase_names(rows):
    passed = True
    names = [row["testcase_name"] for row in rows]
    pattern = re.compile(r"^(case|network)\d+(_\w+)?$")
    non_matching = [(i, n) for i, n in enumerate(names) if not pattern.match(n)]
    if non_matching:
        passed = False
        for i, n in non_matching[:5]:
            _logger.info("  [FAIL] testcase_name 格式错误: 行%d '%s'", i, n)
        if len(non_matching) > 5:
            _logger.info("  ... 共 %d 处格式错误", len(non_matching))
    unique = set(names)
    if len(unique) != len(names):
        passed = False
        dupes = [n for n in unique if names.count(n) > 1]
        _logger.info("  [FAIL] testcase_name 存在重复: %s", dupes[:5])
    if passed:
        _logger.info("  [PASS] testcase_name 唯一且格式正确 (%d 条)", len(names))
    return passed


def check_op_name(rows):
    passed = True
    pattern = re.compile(r"^[a-z][a-z0-9_]*$")
    values = set()
    for i, row in enumerate(rows):
        val = row["op_name"].strip()
        values.add(val)
        if not val:
            passed = False
            _logger.info("  [FAIL] op_name 为空: 行%d", i)
        elif not pattern.match(val):
            passed = False
            _logger.info("  [FAIL] op_name 格式错误: 行%d '%s'", i, val)
    if passed:
        _logger.info("  [PASS] op_name 格式正确: %s", values)
    return passed


def check_required_fields(rows, required):
    passed = True
    for col in required:
        if col not in rows[0]:
            continue
        empty_count = sum(1 for row in rows if not row[col].strip())
        if empty_count > 0:
            passed = False
            _logger.info("  [FAIL] 必填字段 '%s' 有 %d 行为空", col, empty_count)
    if passed:
        _logger.info("  [PASS] 所有必填字段均非空")
    return passed


def check_precision_tolerances(rows):
    passed = True
    for i, row in enumerate(rows):
        val = row["precision_tolerances"].strip()
        if val.lower() == "none" or val == "":
            continue
        if not val.startswith("(("):
            passed = False
            _logger.info("  [FAIL] precision_tolerances 格式错误: 行%d '%s'", i, val)
        else:
            stripped = val.rstrip()
            if not (stripped.endswith("))") or stripped.endswith("),)")):
                passed = False
                _logger.info("  [FAIL] precision_tolerances 格式错误: 行%d '%s'", i, val)
    if passed:
        non_none = 0
        for row in rows:
            val = row["precision_tolerances"].strip()
            if val and val.lower() != "none":
                non_none += 1
        none_count = len(rows) - non_none
        _logger.info(
            "  [PASS] precision_tolerances 格式正确 (%d 条有值, %d 条为 None)",
            non_none, none_count
        )
    return passed


def _count_outer_elements(s):
    if not s:
        return 0
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return 0
    inner = s[1:-1]
    depth = 0
    count = 1
    for ch in inner:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            count += 1
    return count


def check_tuple_length_consistency(rows, tuple_cols, input_ref_col, output_ref_col=None):
    if output_ref_col is None:
        output_ref_col = input_ref_col
    passed = True
    for i, row in enumerate(rows):
        for col in tuple_cols:
            val = row[col].strip()
            if val.lower() == "none" or val == "" or val == "None":
                continue
            is_output = col.startswith("output_")
            ref = output_ref_col if is_output else input_ref_col
            ref_str = row[ref].strip()
            if ref_str.lower() == "none" or ref_str == "":
                continue
            ref_count = _count_outer_elements(ref_str)
            col_count = _count_outer_elements(val)
            if col_count != ref_count:
                passed = False
                _logger.info(
                    "  [FAIL] 行%d: '%s' tuple 长度(%d) != '%s' 长度(%d)",
                    i, col, col_count, ref, ref_count
                )
    if passed:
        _logger.info("  [PASS] tuple 字段长度与 shapes 一致")
    return passed


def check_mode_detection(headers):
    mode = detect_mode(headers)
    _logger.info(
        "  [PASS] 模式识别: %s (表头%s api_name)",
        mode, '含' if mode != 'kernel' else '不含'
    )
    return mode


def validate(csv_path):
    _logger.info("=== TTK CSV 校验: %s ===\n", csv_path)
    results = {}

    results["encoding"] = check_encoding(csv_path)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    mode = check_mode_detection(headers)
    expected = get_expected_columns(mode)
    required = get_required_columns(mode)
    tuple_cols = get_tuple_columns(mode)
    input_ref_col = "input_shapes" if mode == "kernel" else "tensor_view_shapes"
    output_ref_col = "output_shapes" if mode == "kernel" else "tensor_view_shapes"

    results["header"] = check_header(headers, expected)
    results["row_count"] = check_row_count(len(rows))

    if len(rows) == 0:
        _logger.info("\n=== 校验完成: CSV 无数据行，跳过字段校验 ===")
        all_pass = all(results.values())
        _logger.info("\n总计: %s", 'PASS' if all_pass else 'FAIL')
        return all_pass

    results["testcase_names"] = check_testcase_names(rows)
    if "op_name" in rows[0]:
        results["op_name"] = check_op_name(rows)
    results["required_fields"] = check_required_fields(rows, required)
    results["precision_tolerances"] = check_precision_tolerances(rows)
    results["tuple_length"] = check_tuple_length_consistency(rows, tuple_cols, input_ref_col, output_ref_col)

    all_pass = all(results.values())
    _logger.info("\n=== 校验完成: %s ===", 'PASS' if all_pass else 'FAIL')
    return all_pass


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s")
    if len(sys.argv) != 2:
        _logger.info("用法: python ttk_validate_csv.py <csv_file>")
        sys.exit(1)
    csv_file = sys.argv[1]
    ok = validate(csv_file)
    sys.exit(0 if ok else 1)
