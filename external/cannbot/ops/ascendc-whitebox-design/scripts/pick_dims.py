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
"""Generate S2P2_param_def_groups.json from S2P2_dim_spec.json.

Usage:
  python3 pick_dims.py \
    --input S2P2_dim_spec.json \
    --output S2P2_param_def_groups.json

Reads the dim spec (lo/hi/count ranges written by LLM),
randomly picks non-repeating integer values, and produces
the groups.json consumed by build_param_def.py.
"""

import json
import argparse
import logging
import random
import zlib
import math

_logger = logging.getLogger(__name__)


DTYPE_SIZE = {
    "float16": 2, "float32": 4, "bfloat16": 2,
    "uint8": 1, "int8": 1, "int16": 2, "int32": 4,
    "int64": 8, "bool": 1, "complex64": 8, "complex128": 16,
}


def make_seed(group_id, dtype_name=""):
    return zlib.crc32(f"{group_id}:{dtype_name}".encode()) % 100000


def pick_range(lo, hi, count, rng):
    available = hi - lo + 1
    if available <= count:
        return sorted(range(lo, hi + 1))
    return sorted(rng.sample(range(lo, hi + 1), count))


def resolve_dim(spec, rng):
    if "values" in spec:
        return spec["values"]
    if "value" in spec:
        return [spec["value"]]
    return pick_range(spec["lo"], spec["hi"], spec["count"], rng)


def _build_ctx(base_name, bv, dtype_size, param_lists, i):
    ctx = {base_name: bv, "_dtype_size": dtype_size}
    for pname, pvals in param_lists.items():
        ctx[pname] = pvals[i]
    return ctx


def _resolve_derived(derived_spec, ctx, rng):
    eval_globals = {"__builtins__": {}, "math": math, "abs": abs, "min": min, "max": max}
    if isinstance(derived_spec, str):
        return eval(derived_spec, eval_globals, ctx)
    dim_lo = derived_spec["lo"]
    dim_hi = derived_spec["hi"]
    if derived_spec.get("min") is not None:
        computed_lo = eval(derived_spec["min"], eval_globals, ctx)
    else:
        computed_lo = dim_lo
    if derived_spec.get("max") is not None:
        computed_hi = eval(derived_spec["max"], eval_globals, ctx)
    else:
        computed_hi = dim_hi
    final_lo = max(int(computed_lo), dim_lo)
    final_hi = min(int(computed_hi), dim_hi)
    if final_lo > final_hi:
        final_lo = dim_lo
        final_hi = dim_hi
    return rng.randint(final_lo, final_hi)


def resolve_compound(spec, rng, dtype):
    base_name = spec["base"]["name"]
    base_count = spec["base"]["count"]
    base_values = pick_range(spec["base"]["lo"], spec["base"]["hi"], base_count, rng)
    dtype_size = DTYPE_SIZE.get(dtype, 2)

    param_lists = {}
    if "params" in spec:
        for pname, pspec in spec["params"].items():
            prng = random.Random(rng.randint(0, 99999))
            param_lists[pname] = [prng.randint(pspec["lo"], pspec["hi"]) for _ in range(base_count)]

    results = []
    for i, bv in enumerate(base_values):
        ctx = _build_ctx(base_name, bv, dtype_size, param_lists, i)
        entry = {base_name: bv}
        for dim_name, derived_spec in spec.get("derived", {}).items():
            entry[dim_name] = _resolve_derived(derived_spec, ctx, rng)
        results.append(entry)
    return results


def process_group(group, dtypes):
    gid = group["id"]
    result = {"id": gid, "mode": group["mode"]}

    if "group_dims" in group:
        rng = random.Random(make_seed(gid, "__group_dims__"))
        result["group_dims"] = {}
        for dim_name, spec in group["group_dims"].items():
            result["group_dims"][dim_name] = resolve_dim(spec, rng)

    result["per_dtype"] = {}
    for dtype_name in dtypes:
        if dtype_name not in group["per_dtype"]:
            continue
        entries = group["per_dtype"][dtype_name]
        rng = random.Random(make_seed(gid, dtype_name))
        result_entries = []
        for entry in entries:
            re = {"path": entry["path"], "key": entry["key"]}
            for dim_name, spec in entry.get("dims", {}).items():
                re[dim_name] = resolve_dim(spec, rng)
            for comp_name, spec in entry.get("compound_dims", {}).items():
                re[comp_name] = resolve_compound(spec, rng, dtype_name)
            result_entries.append(re)
        result["per_dtype"][dtype_name] = result_entries

    if "constraint_note" in group:
        result["constraint_note"] = group["constraint_note"]
    return result


def main():
    logging.basicConfig(format="%(message)s")
    parser = argparse.ArgumentParser(description="Generate param_def_groups from dim_spec")
    parser.add_argument("--input", required=True, help="Path to S2P2_dim_spec.json")
    parser.add_argument("--output", required=True, help="Path to output S2P2_param_def_groups.json")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        spec = json.load(f)

    dtypes = spec.get("dtypes", [])
    groups_out = [process_group(g, dtypes) for g in spec["groups"]]

    result = {
        "platform": spec["platform"],
        "platform_cores": spec["platform_cores"],
        "tiling_keys": spec["tiling_keys"],
        "dtype_tensors": spec["dtype_tensors"],
        "groups": groups_out,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    _logger.info("Generated %s (%d groups)", args.output, len(groups_out))
    for g in groups_out:
        dtypes_list = list(g["per_dtype"].keys())
        counts = {d: len(e) for d, e in g["per_dtype"].items()}
        _logger.info("  %s: dtypes=%s, entries=%s", g['id'], dtypes_list, counts)


if __name__ == "__main__":
    main()
