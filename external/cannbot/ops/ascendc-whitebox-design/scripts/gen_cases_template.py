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
"""S2P2 -> S2P2_cases.json (JSON-driven, no hardcoded values)

Usage: python3 S2P2_gen_cases.py
Output: S2P2_cases.json
"""

import json
import os
import random
import argparse
import zlib
from collections import Counter
import logging

_logger = logging.getLogger(__name__)

random.seed(42)
logging.basicConfig(format="%(message)s")

# ═══════════════════════════════════════════════════════════════
# Section 1: 配置
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAM_DEF_PATH = os.path.join(SCRIPT_DIR, "S2P2_param_def.json")
OUT = os.path.join(SCRIPT_DIR, "S2P2_cases.json")

with open(PARAM_DEF_PATH, encoding="utf-8") as _f:
    _param_def = json.load(_f)

DTYPE_PARAM = _param_def["dtype_tensors"][0]["param"]

_a = 0
for _g in _param_def["groups"]:
    for _dtype, _entries in _g["per_dtype"].items():
        _a += len(_entries)

_default_cap = max(min(10, 100 // _a), 1)
_pa = argparse.ArgumentParser()
_pa.add_argument("--cap", type=int, default=_default_cap)
_cap = _pa.parse_args().cap
_logger.info("a=%d, default_cap=%d, cap=%d", _a, _default_cap, _cap)

# ═══════════════════════════════════════════════════════════════
# Section 2: 工具函数
# ═══════════════════════════════════════════════════════════════

ENTRY_RESERVED = {"path", "key"}


def compress_per_dtype(dim_dicts, cap):
    """生成恰好 cap 个 per_dtype 组合。每个维度独立随机选值，固定 seed 保证可复现。"""
    if not dim_dicts:
        return [{}]
    names = list(dim_dicts.keys())
    base_seed = zlib.crc32(str(sorted(names)).encode()) % 100000
    rng = random.Random(base_seed)
    results, seen = [], set()
    max_iter = cap * 30
    i = 0
    while len(results) < cap and i < max_iter:
        combo = {}
        for name in names:
            combo.update(rng.choice(dim_dicts[name]))
        key = tuple(sorted(combo.items()))
        if key not in seen:
            seen.add(key)
            results.append(combo)
        i += 1
    return results


def compress_group_pool(dim_dicts):
    """多 group 级维度 → 单 POOL。各维度独立 shuffle（不同 seed），同位配对。单维度直接返回。"""
    if len(dim_dicts) == 1:
        return list(dim_dicts.values())[0]
    rng = random.Random()
    shuffled = {}
    min_len = min(len(v) for v in dim_dicts.values())
    for name, values in dim_dicts.items():
        rng.seed(zlib.crc32(name.encode()) % 100000)
        s = values[:]
        rng.shuffle(s)
        shuffled[name] = s[:min_len]
    results = []
    for i in range(min_len):
        combo = {}
        for name in shuffled:
            combo.update(shuffled[name][i])
        results.append(combo)
    return results


def shuffled_pool(base, seed):
    """返回打乱后的池和位置指针。每个 group 独立 seed。"""
    rng = random.Random(seed)
    p = base[:]
    rng.shuffle(p)
    return p, 0


def extract_entry_dims(entry):
    """从 per_dtype entry 中提取维度，构建 dim_dicts 供 compress_per_dtype 使用。

    - flat array (如 ["float16"], [1]) → [{field: v} for v in values]
    - compound dict list (如 [{"{key1}": v1, "{key2}": v2}, ...]) → 直接使用
    """
    dim_dicts = {}
    for field, value in entry.items():
        if field in ENTRY_RESERVED:
            continue
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            dim_dicts[field] = value
        elif isinstance(value, list):
            dim_dicts[field] = [{field: v} for v in value]
    return dim_dicts


def extract_group_dims(group):
    """从 group["group_dims"] 提取 group 级维度。"""
    dim_dicts = {}
    for field, value in group.get("group_dims", {}).items():
        if isinstance(value, list):
            dim_dicts[field] = [{field: v} for v in value]
    return dim_dicts


# ═══════════════════════════════════════════════════════════════
# Section 3: 生成逻辑
# ═══════════════════════════════════════════════════════════════

all_cases = []

for group_idx, _group in enumerate(_param_def["groups"]):
    gid = _group["id"]
    _seed = (group_idx + 1) * 100

    group_dim_dicts = extract_group_dims(_group)
    if group_dim_dicts:
        pool_base = compress_group_pool(group_dim_dicts)
    else:
        pool_base = [{}]

    pool, pos = shuffled_pool(pool_base, _seed)

    for dtype_name, entries in _group["per_dtype"].items():
        for _entry in entries:
            entry_path = _entry["path"]
            entry_key = _entry["key"]
            _dim_dicts = extract_entry_dims(_entry)
            pairs = compress_per_dtype(_dim_dicts, _cap)

            for _p in pairs:
                if pos >= len(pool):
                    pool, pos = shuffled_pool(pool_base, _seed)
                gp = pool[pos]
                pos += 1
                case = {
                    "_group": gid,
                    DTYPE_PARAM: dtype_name,
                    "path": entry_path,
                    "key": entry_key,
                    **_p,
                    **gp,
                }
                all_cases.append(case)

# ═══════════════════════════════════════════════════════════════
# Section 4: 后处理
# ═══════════════════════════════════════════════════════════════

_seen = set()
unique = []
for c in all_cases:
    k = tuple(sorted(c.items()))
    if k not in _seen:
        _seen.add(k)
        unique.append(c)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(unique, f, indent=2, ensure_ascii=False)

cnt = Counter(c["_group"] for c in unique)
_logger.info("Generated %d cases -> %s", len(unique), OUT)
for g in sorted(cnt.keys()):
    _logger.info("  %s: %d", g, cnt[g])
