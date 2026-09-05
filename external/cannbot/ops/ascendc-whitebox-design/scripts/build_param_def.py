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
"""Build S2P2_param_def.json from S2P2_param_def_groups.json.

Usage:
  python3 build_param_def.py \
    --groups S2P2_param_def_groups.json \
    --output S2P2_param_def.json

The groups file contains all derivation data (LLM output).
This script assembles the complete S2P2_param_def.json with
proper field ordering and compact JSON formatting.
"""

import json
import argparse
import logging

_logger = logging.getLogger(__name__)


def expand_group(g):
    group = {}
    group["id"] = g["id"]
    group["mode"] = g["mode"]

    if "group_dims" in g:
        group["group_dims"] = g["group_dims"]

    per_dtype_spec = g["per_dtype"]
    if not isinstance(per_dtype_spec, dict):
        raise ValueError(f"group '{g['id']}': per_dtype must be a dict, got {type(per_dtype_spec).__name__}")

    group["per_dtype"] = per_dtype_spec

    if "constraint_note" in g:
        group["constraint_note"] = g["constraint_note"]
    return group


def _is_scalar(v):
    return isinstance(v, (int, float, str, bool)) or v is None


def _encode_value(v, indent_level, indent_str="  "):
    prefix = indent_str * indent_level
    inner_prefix = indent_str * (indent_level + 1)

    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return json.dumps(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(_is_scalar(item) for item in v):
            inner = ", ".join(_encode_value(item, 0) for item in v)
            return f"[{inner}]"
        if all(isinstance(item, dict) for item in v):
            parts = []
            for item in v:
                parts.append(f"{inner_prefix}{_encode_value(item, indent_level + 1)}")
            return "[\n" + ",\n".join(parts) + f"\n{prefix}]"
        parts = []
        for item in v:
            parts.append(f"{inner_prefix}{_encode_value(item, indent_level + 1)}")
        return "[\n" + ",\n".join(parts) + f"\n{prefix}]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        parts = []
        for k, val in v.items():
            encoded_val = _encode_value(val, indent_level + 1)
            parts.append(f'{inner_prefix}{json.dumps(k)}: {encoded_val}')
        return "{\n" + ",\n".join(parts) + f"\n{prefix}}}"
    return json.dumps(v, ensure_ascii=False)


def dump_compact_json(data):
    return _encode_value(data, 0) + "\n"


def main():
    logging.basicConfig(format="%(message)s")
    parser = argparse.ArgumentParser(description="Build S2P2_param_def.json from groups data")
    parser.add_argument("--groups", required=True, help="Path to S2P2_param_def_groups.json")
    parser.add_argument("--output", required=True, help="Path to output S2P2_param_def.json")
    args = parser.parse_args()

    with open(args.groups, "r") as f:
        groups_data = json.load(f)

    result = {
        "platform": groups_data["platform"],
        "platform_cores": groups_data["platform_cores"],
        "tiling_keys": groups_data["tiling_keys"],
        "dtype_tensors": groups_data["dtype_tensors"],
        "groups": [expand_group(g) for g in groups_data["groups"]],
    }

    json_str = dump_compact_json(result)
    with open(args.output, "w") as f:
        f.write(json_str)

    parsed = json.loads(json_str)
    if parsed != result:
        raise ValueError("Round-trip check failed")

    _logger.info("Generated %s (%d bytes, %d lines)", args.output, len(json_str), json_str.count(chr(10)))
    _logger.info("Platform: %s", result['platform'])
    _logger.info("Cores: %s", result['platform_cores'])
    _logger.info("Tiling keys: %s", result['tiling_keys'])
    _logger.info("Dtype tensors: %s", result['dtype_tensors'])
    _logger.info("Groups: %d", len(result['groups']))
    for g in result["groups"]:
        dtypes = list(g["per_dtype"].keys())
        entry_counts = {dt: len(entries) for dt, entries in g["per_dtype"].items()}
        _logger.info("  %s: dtypes=%s, entries=%s", g['id'], dtypes, entry_counts)


if __name__ == "__main__":
    main()
