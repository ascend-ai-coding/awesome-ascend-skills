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
# ----------------------------------------------------------------------------------------------------------
"""
query_strategies.py — 策略卡片程序化筛选工具

读取 plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/*.md 头部的 YAML frontmatter，
根据瓶颈类型、算子族、复杂度等条件筛选候选策略 ID。

用法:
    # 按瓶颈筛选
    python3 query_strategies.py --bottleneck mte2_stall --limit 15

    # 按算子族筛选
    python3 query_strategies.py --op-family attention

    # 复合筛选（AND）
    python3 query_strategies.py --bottleneck mte2_stall --op-family normalization \
        --complexity-max L1 --exclude-ids P1,P19 --output /tmp/candidates.json

    # 冲突排除
    python3 query_strategies.py --bottleneck mte2_stall --exclude-conflicts-of P14

    # 校验所有卡片 frontmatter
    python3 query_strategies.py --validate-all

输出 JSON:
    {
      "matched": ["P1", "P7", "P10"],
      "scored": [{"id": "P1", "score": 5, "hit_bottlenecks": [...], "hit_op_families": [...]}],
      "filter_applied": {...},
      "total_before_filter": 103,
      "total_after_filter": 3
    }

设计原则:
- 不引入 PyYAML 依赖（手写简化 YAML parser）
- 筛选结果带打分（命中瓶颈数×2 + 命中算子族数×1 + 协同命中×1）
- --exclude-conflicts-of 支持多个 ID（逗号分隔）
"""
import argparse
import glob
import json
import logging
import os
import re
import sys
from dataclasses import dataclass


LOGGER = logging.getLogger(__name__)

# 数据输出专用 logger：CLI 结果（JSON/表格/报告）走 stdout（agent 调用协议通道），
# 与 LOGGER（stderr 进度/警告）分离，避免 lint G.LOG.02 误报 print。
DATA_LOGGER = logging.getLogger(f"{__name__}.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)

VALID_BOTTLENECKS = {
    "mte2_stall", "mte3_stall", "tiling_imbalance", "scalar_loading",
    "scalar_compute", "compute_bound", "near_optimal", "no_overlap",
    "partial_overlap", "undersize_transfer", "icache_miss", "bus_contention",
    "l2_cache_thrash", "ub_memory_pressure",
}

VALID_OP_FAMILIES = {
    "elementwise", "normalization", "reduction", "softmax", "attention",
    "flash_attention", "cv_fusion", "matmul", "moe", "quantization",
    "pooling_gather", "optimizer", "index_scatter", "broadcast_mask",
    "special", "datapath",
}

VALID_COMPLEXITIES = ["L0", "L1", "L2"]


def _append_list_item(result: dict, key: str, line: str):
    """处理 frontmatter 中的列表项行（以 - 开头）。"""
    item = line.strip()[2:].strip()
    # 去除可能的 quote
    item = item.strip('"').strip("'")
    if item:
        result.setdefault(key, []).append(item)


def _parse_kv_line(result: dict, line: str) -> str | None:
    """解析 key: value 行，返回新的 current_list_key（无则 None）。"""
    m = re.match(r'^([\w_]+)\s*:\s*(.*)$', line)
    if not m:
        return None
    key, val = m.group(1), m.group(2).strip()

    if not val:
        # key 后无值（后续是 - 列表）
        result[key] = []
        return key
    if val.startswith('[') and val.endswith(']'):
        # 行内列表 [a, b, c]
        inner = val[1:-1].strip()
        items = [x.strip().strip('"').strip("'") for x in inner.split(',')] if inner else []
        result[key] = [x for x in items if x]
        return None
    # 单值
    result[key] = val.strip('"').strip("'")
    return None


def parse_frontmatter(content: str) -> dict | None:
    """从文件内容提取 YAML frontmatter（简化 parser，只支持本项目的字段格式）。

    支持：
        key: value
        key: [a, b, c]
        key:
          - a
          - b
    """
    if not content.startswith('---\n'):
        return None
    end_match = re.search(r'\n---\n', content[4:])
    if not end_match:
        return None
    yaml_block = content[4:4 + end_match.start()]

    result = {}
    current_list_key = None
    for line in yaml_block.split('\n'):
        # 跳过空行和注释
        if not line.strip() or line.strip().startswith('#'):
            continue

        # 列表项（以 - 开头，可缩进）
        if re.match(r'^\s+-\s', line):
            if current_list_key:
                _append_list_item(result, current_list_key, line)
            continue

        current_list_key = _parse_kv_line(result, line)

    return result


def load_all_cards(cards_dir: str) -> dict:
    """扫描所有卡片，返回 {id: metadata_dict}。缺失 frontmatter 的卡片跳过。"""
    index = {}
    errors = []
    card_files = sorted(glob.glob(os.path.join(cards_dir, "*.md")))
    # 排除 SCHEMA.md / README.md 等辅助文件
    card_files = [f for f in card_files if not os.path.basename(f) in ("SCHEMA.md", "README.md")]

    for card_path in card_files:
        try:
            with open(card_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (OSError, UnicodeDecodeError) as e:
            errors.append({"file": card_path, "error": f"read_error: {e}"})
            continue

        fm = parse_frontmatter(content)
        if fm is None:
            # 没有 frontmatter，跳过（向后兼容，不算 error）
            continue

        sid = fm.get('id')
        if not sid:
            errors.append({"file": card_path, "error": "missing_id"})
            continue

        # 标准化列表字段
        fm['bottlenecks'] = fm.get('bottlenecks', []) or []
        fm['op_families'] = fm.get('op_families', []) or []
        fm['conflicts_with'] = fm.get('conflicts_with', []) or []
        fm['synergizes_with'] = fm.get('synergizes_with', []) or []
        fm['complexity'] = fm.get('complexity', 'L1')
        fm['_card_path'] = card_path

        index[sid] = fm

    return {"index": index, "errors": errors}


def validate_all(cards_dir: str) -> int:
    """校验所有卡片 frontmatter。返回错误数。"""
    data = load_all_cards(cards_dir)
    index = data["index"]
    errors = list(data["errors"])

    all_ids = set(index.keys())
    for sid, meta in index.items():
        # bottlenecks 合法性
        for b in meta['bottlenecks']:
            if b not in VALID_BOTTLENECKS:
                errors.append({"id": sid, "error": f"invalid_bottleneck: {b}"})
        # op_families 合法性
        for f in meta['op_families']:
            if f not in VALID_OP_FAMILIES:
                errors.append({"id": sid, "error": f"invalid_op_family: {f}"})
        # complexity 合法性
        if meta['complexity'] not in VALID_COMPLEXITIES:
            errors.append({"id": sid, "error": f"invalid_complexity: {meta['complexity']}"})
        # conflicts_with / synergizes_with 必须指向存在的 ID
        for cid in meta['conflicts_with']:
            if cid not in all_ids:
                errors.append({"id": sid, "error": f"conflicts_with_unknown_id: {cid}"})
        for sid2 in meta['synergizes_with']:
            if sid2 not in all_ids:
                errors.append({"id": sid, "error": f"synergizes_with_unknown_id: {sid2}"})
        # id 与文件名前缀一致性
        fname = os.path.basename(meta['_card_path'])
        prefix = fname.split('_', 1)[0].replace('.md', '')
        if prefix != sid:
            errors.append({"id": sid, "error": f"id_filename_mismatch: file={fname}"})

    DATA_LOGGER.info("Validation: %d cards loaded, %d errors", len(index), len(errors))
    for e in errors:
        DATA_LOGGER.info("  ❌ %s", e)
    return len(errors)


@dataclass
class QueryFilter:
    """query 的筛选条件封装。"""
    cards_dir: str
    bottleneck: str = None
    op_family: str = None
    complexity_max: str = None
    exclude_ids: list = None
    exclude_conflicts_of: list = None
    limit: int = 20


_COMPLEXITY_ORDER = {"L0": 0, "L1": 1, "L2": 2}


def _id_sort_key(sid: str):
    m = re.match(r'([A-Z]+)(\d+)', sid)
    if m:
        return (m.group(1), int(m.group(2)))
    return (sid, 0)


def _score_candidate(sid: str, meta: dict, qf: QueryFilter, ceiling: int) -> dict | None:
    """对单张卡片执行硬条件过滤并打分。未通过硬条件返回 None。"""
    if _COMPLEXITY_ORDER.get(meta['complexity'], 99) > ceiling:
        return None

    # bottleneck 硬条件（如指定）
    hit_bottlenecks = []
    if qf.bottleneck:
        if qf.bottleneck not in meta['bottlenecks']:
            return None
        hit_bottlenecks.append(qf.bottleneck)

    # op_family 硬条件（如指定）
    hit_op_families = []
    if qf.op_family:
        if qf.op_family in meta['op_families']:
            hit_op_families.append(qf.op_family)
        elif 'datapath' in meta['op_families']:
            # datapath 视为通用命中（适用于所有算子族）
            hit_op_families.append('datapath')
        else:
            return None

    # 打分：瓶颈×2 + 算子族×1 + 精确匹配 op_family 额外+1
    score = len(hit_bottlenecks) * 2 + len(hit_op_families)
    if qf.op_family and qf.op_family in meta['op_families']:
        score += 1  # 精确命中 op_family 额外加分（而非 datapath 回退）

    return {
        "id": sid,
        "score": score,
        "complexity": meta['complexity'],
        "hit_bottlenecks": hit_bottlenecks,
        "hit_op_families": hit_op_families,
        "bottlenecks": meta['bottlenecks'],
        "op_families": meta['op_families'],
        "conflicts_with": meta['conflicts_with'],
        "synergizes_with": meta['synergizes_with'],
    }


def query(qf: QueryFilter) -> dict:
    """筛选策略。硬条件 AND 匹配，软条件参与打分。"""
    data = load_all_cards(qf.cards_dir)
    index = data["index"]
    exclude_ids = set(qf.exclude_ids or [])

    # 展开 exclude_conflicts_of: 所有被这些 ID 冲突的策略都要排除
    for ref_id in (qf.exclude_conflicts_of or []):
        if ref_id in index:
            for conflict_id in index[ref_id].get('conflicts_with', []):
                exclude_ids.add(conflict_id)

    ceiling = _COMPLEXITY_ORDER.get(qf.complexity_max, 99) if qf.complexity_max else 99

    candidates = []
    for sid, meta in index.items():
        # 排除 ID
        if sid in exclude_ids:
            continue
        scored = _score_candidate(sid, meta, qf, ceiling)
        if scored is not None:
            candidates.append(scored)

    # 按得分降序排序，同分按 ID 升序
    candidates.sort(key=lambda c: (-c['score'], _id_sort_key(c['id'])))
    candidates = candidates[:qf.limit]

    return {
        "matched": [c['id'] for c in candidates],
        "scored": candidates,
        "filter_applied": {
            "bottleneck": qf.bottleneck,
            "op_family": qf.op_family,
            "complexity_max": qf.complexity_max,
            "exclude_ids": sorted(exclude_ids),
        },
        "total_before_filter": len(index),
        "total_after_filter": len(candidates),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="策略卡片程序化筛选工具")
    parser.add_argument("--cards-dir",
                        default="plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/")
    parser.add_argument("--bottleneck", default=None, help="瓶颈类型（单值）")
    parser.add_argument("--op-family", default=None, help="算子族（单值）")
    parser.add_argument("--complexity-max", default=None, choices=VALID_COMPLEXITIES,
                        help="复杂度上限（L0/L1/L2）")
    parser.add_argument("--exclude-ids", default="", help="排除的策略 ID（逗号分隔）")
    parser.add_argument("--exclude-conflicts-of", default="",
                        help="排除与这些 ID 冲突的策略（逗号分隔）")
    parser.add_argument("--limit", type=int, default=20, help="返回候选数量上限")
    parser.add_argument("--output", default="-", help="输出 JSON 路径（- 表示 stdout）")
    parser.add_argument("--validate-all", action="store_true", help="校验所有卡片")

    args = parser.parse_args()

    # 参数校验
    if args.bottleneck and args.bottleneck not in VALID_BOTTLENECKS:
        LOGGER.error("ERROR: invalid bottleneck '%s'. Valid: %s",
                     args.bottleneck, sorted(VALID_BOTTLENECKS))
        sys.exit(2)
    if args.op_family and args.op_family not in VALID_OP_FAMILIES:
        LOGGER.error("ERROR: invalid op_family '%s'. Valid: %s",
                     args.op_family, sorted(VALID_OP_FAMILIES))
        sys.exit(2)

    if args.validate_all:
        err_count = validate_all(args.cards_dir)
        sys.exit(1 if err_count > 0 else 0)

    exclude_ids = [x.strip() for x in args.exclude_ids.split(',') if x.strip()]
    exclude_conflicts_of = [x.strip() for x in args.exclude_conflicts_of.split(',') if x.strip()]

    result = query(QueryFilter(
        cards_dir=args.cards_dir,
        bottleneck=args.bottleneck,
        op_family=args.op_family,
        complexity_max=args.complexity_max,
        exclude_ids=exclude_ids,
        exclude_conflicts_of=exclude_conflicts_of,
        limit=args.limit,
    ))

    out = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output == "-":
        DATA_LOGGER.info("%s", out)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(out)
        LOGGER.info("Query result written: %s (%d matched)",
                    args.output, result['total_after_filter'])


if __name__ == "__main__":
    main()
