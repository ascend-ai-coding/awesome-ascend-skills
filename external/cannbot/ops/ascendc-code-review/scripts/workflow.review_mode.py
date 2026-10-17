#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""
检视模式路由器。读 workflow.review-thresholds.yaml，按变更行数/文件数判定 mode，
输出 JSON {mode, routing, guidance}。code-fetch / file-review 入口调用，阈值单一来源在 yaml。

四档中 minimal/compact/standard 按行数分档，oversized 另需文件数同时达阈值。
routing 恒为 pr-review——oversized 不切换流程，只让调用方向用户给出拆分建议。
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

CFG = Path(__file__).parent / "workflow.review-thresholds.yaml"


def load_cfg():
    """读 workflow.review-thresholds.yaml。"""
    try:
        import yaml  # PyYAML 6.x
    except ImportError as exc:  # pragma: no cover
        raise ImportError(f"PyYAML is required (pip install pyyaml): {exc}") from exc
    with open(CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_oversized(lines, files, oversized_cfg):
    """超大 PR 双高判定：文件数 > min_files 且 行数 >= min_lines。files 为空则不触发
    （文件检视不传 --files，故永不取到本档）。"""
    if files is None:
        return False
    return files > oversized_cfg["min_files"] and lines >= oversized_cfg["min_lines"]


def pick_mode(lines, modes_cfg):
    """按行数选 minimal/compact/standard 条目。"""
    if lines < modes_cfg["minimal"]["max_lines"]:
        return "minimal", modes_cfg["minimal"]
    if lines < modes_cfg["compact"]["max_lines"]:
        return "compact", modes_cfg["compact"]
    return "standard", modes_cfg["standard"]


def decide(lines, files, cfg):
    """返回 (mode, routing, guidance)。"""
    if is_oversized(lines, files, cfg["oversized"]):
        oversized = cfg["oversized"]
        return "oversized", oversized["routing"], oversized["guidance"]
    name, entry = pick_mode(lines, cfg["review_modes"])
    return name, entry["routing"], entry["guidance"]


def main():
    ap = argparse.ArgumentParser(description="检视模式路由器")
    ap.add_argument("--lines", type=int, required=True,
                    help="变更行数（PR=diff 新增+删除；文件检视=代码总行数。"
                         "统计口径见 workflow.review-thresholds.yaml 顶部「统计口径（行数）」段）")
    ap.add_argument("--files", type=int, default=None, help="变更文件数（文件检视模式可省略）")
    args = ap.parse_args()
    try:
        cfg = load_cfg()
    except ImportError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    mode, routing, guidance = decide(args.lines, args.files, cfg)
    # 机器可读 JSON 走 stdout（供调用方 $(...) 捕获，与 create_review_dir.py 约定一致）；人读日志走 stderr
    logger.info("mode=%s routing=%s lines=%d files=%s", mode, routing, args.lines, args.files)
    print(json.dumps({"mode": mode, "routing": routing, "guidance": guidance}, ensure_ascii=False))


if __name__ == "__main__":
    main()
