#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND OF WARRANTIES OR CONDITIONS OF ANY KIND,
# INCLUDING WITHOUT LIMITATION, NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""
超大 PR 判定器。读 workflow.review-thresholds.yaml 的 oversized 双高阈值，
输出 JSON {"oversized": bool}。code-fetch 调用（文件检视不传 --files，永不判为超大）。
命中后调用方仅向用户输出拆分建议（不阻断），不改变编排。
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
    （文件检视不传 --files，故永不判为超大）。"""
    if files is None:
        return False
    return files > oversized_cfg["min_files"] and lines >= oversized_cfg["min_lines"]


def main():
    ap = argparse.ArgumentParser(description="超大 PR 判定器")
    ap.add_argument("--lines", type=int, required=True,
                    help="变更行数（PR=diff 新增+删除。"
                         "统计口径见 workflow.review-thresholds.yaml 顶部「统计口径（行数）」段）")
    ap.add_argument("--files", type=int, default=None, help="变更文件数（文件检视模式可省略）")
    args = ap.parse_args()
    try:
        cfg = load_cfg()
    except ImportError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    oversized = is_oversized(args.lines, args.files, cfg["oversized"])
    # 机器可读 JSON 走 stdout（供调用方 $(...) 捕获，与 create_review_dir.py 约定一致）；人读日志走 stderr
    logger.info("oversized=%s lines=%d files=%s", oversized, args.lines, args.files)
    print(json.dumps({"oversized": oversized}, ensure_ascii=False))


if __name__ == "__main__":
    main()
