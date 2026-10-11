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
"""Format JSON: arrays inline, objects indented, per_dtype entries multi-line.

Usage:
    python3 format_json.py <file_path>

Rules:
- Array of simple values: single line
- Array of objects: each object single line (inline), one per line
- Object (as dict value): multi-line, 2-space indent, each key on its own line
- Exception: per_dtype.{dtype} entry objects are multi-line (each key on its own line)
- Exception: array element objects containing nested dicts/lists are multi-line
"""

import json
import logging
import sys

_logger = logging.getLogger(__name__)


def _is_simple(v):
    return not isinstance(v, (dict, list))


def _has_nested(obj):
    """Check if a dict contains any nested dict or list values."""
    for v in obj.values():
        if isinstance(v, (dict, list)):
            if isinstance(v, list) and all(_is_simple(x) for x in v):
                continue
            return True
    return False


def _obj_inline(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))


def _fmt(v, level):
    if isinstance(v, dict):
        return _fmt_dict(v, level)
    if isinstance(v, list):
        return _fmt_list(v, level)
    return json.dumps(v, ensure_ascii=False)


def _fmt_dict(obj, level):
    if not obj:
        return "{}"
    ind = "  " * level
    inner = "  " * (level + 1)
    parts = []
    for k, v in obj.items():
        if k == "per_dtype" and isinstance(v, dict):
            parts.append(f'{inner}"{k}": {_fmt_per_dtype_dict(v, level + 1)}')
        else:
            parts.append(f'{inner}"{k}": {_fmt(v, level + 1)}')
    return "{\n" + ",\n".join(parts) + f"\n{ind}}}"


def _fmt_per_dtype_dict(obj, level):
    """Format the per_dtype dict: each dtype value is an array of multi-line entry objects."""
    if not obj:
        return "{}"
    ind = "  " * level
    inner = "  " * (level + 1)
    parts = []
    for k, v in obj.items():
        if isinstance(v, list):
            parts.append(f'{inner}"{k}": {_fmt_per_dtype_array(v, level + 1)}')
        else:
            parts.append(f'{inner}"{k}": {_fmt(v, level + 1)}')
    return "{\n" + ",\n".join(parts) + f"\n{ind}}}"


def _fmt_per_dtype_array(arr, level):
    """per_dtype.{dtype} array: entry objects are multi-line."""
    if not arr:
        return "[]"
    ind = "  " * level
    inner = "  " * (level + 1)
    parts = []
    for x in arr:
        if isinstance(x, dict):
            parts.append(f"{inner}{_fmt_dict(x, level + 1)}")
        else:
            parts.append(f"{inner}{_fmt(x, level + 1)}")
    return "[\n" + ",\n".join(parts) + f"\n{ind}]"


def _fmt_list(arr, level):
    if not arr:
        return "[]"
    if all(_is_simple(x) for x in arr):
        return json.dumps(arr, ensure_ascii=False, separators=(", ", ": "))
    ind = "  " * level
    inner = "  " * (level + 1)
    parts = []
    for x in arr:
        if isinstance(x, dict):
            if _has_nested(x):
                parts.append(f"{inner}{_fmt_dict(x, level + 1)}")
            else:
                parts.append(f"{inner}{_obj_inline(x)}")
        else:
            parts.append(f"{inner}{_fmt(x, level + 1)}")
    return "[\n" + ",\n".join(parts) + f"\n{ind}]"


def format_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = _fmt_dict(data, 0) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(result)
    with open(path, "r", encoding="utf-8") as f:
        json.load(f)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s")
    if len(sys.argv) != 2:
        _logger.error("Usage: %s <file_path>", sys.argv[0])
        sys.exit(1)
    format_json_file(sys.argv[1])
