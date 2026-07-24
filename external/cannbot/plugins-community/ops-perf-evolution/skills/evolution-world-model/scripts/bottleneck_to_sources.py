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
bottleneck_to_sources.py — Stage 3 CLI 入口

把 Stage 2 LLM 给出的 bottleneck_labels 列表反查 INDEX.json，
输出 candidate_source_keys 供 wm_ops select 使用。

设计文档：docs/design/knowledge-strategy-architecture-v3.2.md §3.4 Stage 3
关联库函数：profiling_evidence.match_strategies_by_labels

用法：
    # 从 JSON 文件读 labels
    python3 bottleneck_to_sources.py --labels-file diagnosis.json

    # 命令行直接传 labels
    python3 bottleneck_to_sources.py --labels mte2_stall no_overlap

    # 限制候选数量
    python3 bottleneck_to_sources.py --labels mte2_stall --limit 10

    # 输出到指定文件（默认 stdout）
    python3 bottleneck_to_sources.py --labels mte2_stall --output candidates.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# 数据输出专用 logger：CLI 结果（JSON/表格/报告）走 stdout（agent 调用协议通道），
# 与 LOGGER（stderr 进度/警告）分离，避免 lint G.LOG.02 误报 print。
DATA_LOGGER = logging.getLogger(f"{__name__}.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)

# 让 sibling import 能 work（脚本既可以在 evolution-world-model/scripts/ 跑，
# 也可以在 skill 副本跑）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from profiling_evidence import match_strategies_by_labels, validate_labels


def _resolve_labels(args) -> list:
    """解析 labels 来源（--labels 或 --labels-file），失败返回 None。"""
    if args.labels:
        return args.labels
    try:
        data = json.loads(args.labels_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        LOGGER.error("ERROR: cannot read labels-file: %s", e)
        return None
    # 支持两种字段名
    labels = data.get("bottleneck_labels") or data.get("labels") or []
    if not labels:
        LOGGER.error("ERROR: labels-file must contain non-empty "
                     "'bottleneck_labels' or 'labels' field")
        return None
    return labels


def _render_summary(labels: list, result: dict) -> str:
    """渲染人类友好的摘要文本。"""
    text = (
        f"Input labels: {labels}\n"
        f"Unknown labels: {result.get('unknown_labels', [])}\n"
        f"Candidate count: {len(result['candidate_source_keys'])}\n"
        f"Top candidates (by hit count):\n"
    )
    for sid, sk in zip(result["candidate_ids"], result["candidate_source_keys"]):
        text += f"  - {sid:5}  {sk}\n"
    return text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 bottleneck_labels 反查 evolution-strategies INDEX.json",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--labels",
        nargs="+",
        help="直接给出 bottleneck_labels 列表（空格分隔）",
    )
    src.add_argument(
        "--labels-file",
        type=Path,
        help="JSON 文件，含 bottleneck_labels 字段（list of str）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="candidate_source_keys 返回数量上限（默认全返回）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出文件路径（默认 stdout）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：unknown labels 时退出码 2（默认 0 但在输出中标注）",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="人类友好的摘要（默认输出完整 JSON）",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()

    labels = _resolve_labels(args)
    if labels is None:
        return 1

    # 校验
    valid = validate_labels(labels)
    if not valid["valid"] and args.strict:
        LOGGER.error("ERROR: unknown labels in strict mode: %s", valid['unknown'])
        LOGGER.error("       valid vocabulary size = %d", valid['vocabulary_size'])
        return 2

    # 反查
    result = match_strategies_by_labels(labels, limit=args.limit, include_unknown=True)

    # 输出
    if args.summary:
        text = _render_summary(labels, result)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            DATA_LOGGER.info("%s", text)
    else:
        output = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            DATA_LOGGER.info("%s", output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
