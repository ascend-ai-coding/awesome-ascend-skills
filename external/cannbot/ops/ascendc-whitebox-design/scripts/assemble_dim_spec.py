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
"""
assemble_dim_spec.py — 通用 dim_spec 组装脚本

从 S2P2_analysis_data.json + S2P1_operator_model.json 组装 S2P2_dim_spec.json。
脚本完全通用，不含任何算子特定逻辑。所有算子特定信息通过 analysis_data.json 声明。

功能：
1. 读取 analysis_data.json（LLM 生成的紧凑分析数据）
2. 读取 operator_model.json（提取 platform/dtypes/dtype_tensors）
3. 识别 "_auto" 标记，按 _dtype_adjust 规则计算维度值
4. 展开 _template → 按 dtype 复制 per_dtype entries（支持 dtype 特定覆盖）
5. 组装顶层结构，写入 S2P2_dim_spec.json

analysis_data.json 中的声明式字段：

  _dtype_adjust: {
    "<dim_name>": {"type": "min_bytes", "bytes": <int>}
  }
    声明哪些维度需要根据 dtype size 计算值。
    type=min_bytes: value = ceil(bytes / dtypeSize)

  per_dtype._template: [...]
    所有 dtype 共用的 entries，脚本自动展开。

  per_dtype.<dtype>: [...]
    dtype 特定 entries，覆盖 _template 展开结果。

  dims 中的 "_auto" 标记：
    {"lo": "_auto", "hi": 1024, "count": 5}
    表示 lo 需要从 _dtype_adjust 规则自动计算。

用法：
    python3 assemble_dim_spec.py \
        --input S2P2_analysis_data.json \
        --operator-model S2P1_operator_model.json \
        --output S2P2_dim_spec.json
"""

import argparse
import json
import copy
import logging
import math

_logger = logging.getLogger(__name__)


DTYPE_SIZE = {
    "float16": 2,
    "float32": 4,
    "bfloat16": 2,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "bool": 1,
    "complex64": 8,
    "complex128": 16,
}


def resolve_auto(dims, dtype, dtype_adjust_rules):
    if not dtype_adjust_rules:
        return dims

    result = copy.deepcopy(dims)
    dtype_size = DTYPE_SIZE.get(dtype, 2)

    for dim_name, rule in dtype_adjust_rules.items():
        if dim_name not in result:
            continue

        dim_spec = result[dim_name]
        rule_type = rule.get("type", "min_bytes")

        if rule_type == "min_bytes":
            byte_threshold = rule["bytes"]
            computed_value = math.ceil(byte_threshold / dtype_size)

            if dim_spec.get("lo") == "_auto":
                dim_spec["lo"] = computed_value

            if dim_spec.get("hi") == "_auto":
                dim_spec["hi"] = computed_value

    return result


def _expand_no_template(per_dtype):
    result = {}
    for dtype_key, entries in per_dtype.items():
        if not dtype_key.startswith("_"):
            result[dtype_key] = [copy.deepcopy(e) for e in entries]
    return result


def _expand_dtype_entries(template_entries, dtype, dtype_adjust_rules):
    entries = []
    for entry in template_entries:
        new_entry = copy.deepcopy(entry)
        if "dims" in new_entry:
            new_entry["dims"] = resolve_auto(new_entry["dims"], dtype, dtype_adjust_rules)
        entries.append(new_entry)
    return entries


def expand_template(per_dtype, dtypes, dtype_adjust_rules):
    if "_template" not in per_dtype:
        return _expand_no_template(per_dtype)

    template_entries = per_dtype["_template"]
    result = {}
    for dtype in dtypes:
        if dtype in per_dtype:
            result[dtype] = [copy.deepcopy(e) for e in per_dtype[dtype]]
        else:
            result[dtype] = _expand_dtype_entries(template_entries, dtype, dtype_adjust_rules)
    return result


def extract_platform_info(operator_model):
    platform = operator_model.get("platform", "")
    platform_cores = 0
    if "core_count=" in platform:
        try:
            raw = platform.split("core_count=")[1]
            for sep in [",", " ", "}"]:
                raw = raw.split(sep)[0]
            platform_cores = int(raw)
        except (IndexError, ValueError):
            pass

    dtype_tensors = []
    for inp in operator_model.get("inputs", []):
        if inp.get("dtype", {}).get("values"):
            dtype_tensors.append({
                "tensor": inp["name"],
                "param": f"{inp['name']}_dtype"
            })

    dtypes = []
    for inp in operator_model.get("inputs", []):
        vals = inp.get("dtype", {}).get("values", [])
        if vals:
            dtypes = vals
            break

    return platform, platform_cores, dtype_tensors, dtypes


def assemble(analysis_data, operator_model):
    platform, platform_cores, dtype_tensors, dtypes = extract_platform_info(
        operator_model
    )

    tiling_keys = analysis_data.get("tiling_keys", [])
    dtype_adjust_rules = analysis_data.get("_dtype_adjust", {})

    groups = []
    for group in analysis_data.get("groups", []):
        new_group = {
            "id": group["id"],
            "mode": group["mode"],
            "constraint_note": group.get("constraint_note", ""),
        }

        if "group_dims" in group:
            new_group["group_dims"] = group["group_dims"]

        if "per_dtype" in group:
            new_group["per_dtype"] = expand_template(
                group["per_dtype"], dtypes, dtype_adjust_rules
            )

        groups.append(new_group)

    return {
        "platform": platform,
        "platform_cores": platform_cores,
        "tiling_keys": tiling_keys,
        "dtype_tensors": dtype_tensors,
        "dtypes": dtypes,
        "groups": groups,
    }


def main():
    logging.basicConfig(format="%(message)s")
    parser = argparse.ArgumentParser(description="Assemble S2P2_dim_spec.json")
    parser.add_argument("--input", required=True, help="S2P2_analysis_data.json path")
    parser.add_argument(
        "--operator-model", required=True, help="S2P1_operator_model.json path"
    )
    parser.add_argument("--output", required=True, help="Output S2P2_dim_spec.json path")
    args = parser.parse_args()

    with open(args.input) as f:
        analysis_data = json.load(f)

    with open(args.operator_model) as f:
        operator_model = json.load(f)

    result = assemble(analysis_data, operator_model)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    total_entries = sum(
        len(entries)
        for g in result["groups"]
        for entries in g["per_dtype"].values()
    )
    _logger.info(
        "Generated %s: %d groups, %d tiling keys, %d per_dtype entries",
        args.output, len(result['groups']),
        len(result['tiling_keys']), total_entries
    )


if __name__ == "__main__":
    main()
