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
精选数据集（二筛）

按 (pr_number, commit_id) 复合分组聚合 clean 数据，筛选组内评论数 >= min_comments
的组，每组输出为一个 JSON 对象（组级元信息 + comments 数组），写入 refined/。

不重算 commit_id（clean 里的 commit_id 已是修复后可信值，重算会引入定位 bug）。

用法:
    python3 scripts/refine_dataset.py                 # 处理全部四仓
    python3 scripts/refine_dataset.py --repo ops-cv   # 单仓
"""

import logging
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

REPO_BARE_DIRS = {
    'ops-transformer': '/tmp/pr_recover_test/repo.git',
    'ops-nn': '/tmp/ops_nn_repo.git',
    'ops-cv': '/tmp/ops_cv_repo.git',
    'ops-math': '/tmp/ops_math_repo.git',
}

# 评论级保留字段（其余字段丢弃，避免冗余）
COMMENT_FIELDS = [
    'comment_id', 'author', 'created_at', 'diff_file', 'new_line',
    'body', 'quality', 'reason', 'analysis_source', 'resolved',
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"警告: {path.name} 第 {num} 行 JSON 解析失败: {e}")
    return records


def build_group(records: List[Dict[str, Any]],
                group_key: Tuple[int, str], seq: int) -> Dict[str, Any]:
    """将一个 (pr, commit) 组的评论打包成单个 JSON 对象"""
    pr_number, commit_id = group_key
    first = records[0]

    comments = []
    for i, r in enumerate(records):
        c = {k: r[k] for k in COMMENT_FIELDS if k in r and r[k] is not None}
        # clean 数据无稳定 comment_id，按 (pr, commit, 序号) 生成
        if not c.get('comment_id'):
            c['comment_id'] = f"{commit_id[:12]}_{pr_number}_{i+1}"
        comments.append(c)

    # 按 created_at 排序，保证可复现
    comments.sort(key=lambda c: c.get('created_at', ''))

    return {
        'repo': first.get('repo', ''),
        'pr_number': pr_number,
        'commit_id': commit_id,
        'comment_count': len(comments),
        'diff_file': first.get('diff_file', ''),
        'comments': comments,
    }


def refine_repo(input_path: Path, output_path: Path,
                min_comments: int = 10) -> Dict[str, int]:
    """按 (pr, commit) 聚合，保留评论数 >= min_comments 的组"""
    records = load_jsonl(input_path)

    groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    no_commit = 0
    for r in records:
        cid = r.get('commit_id', '')
        if cid:
            groups[(r.get('pr_number'), cid)].append(r)
        else:
            no_commit += 1

    kept = {k: v for k, v in groups.items() if len(v) >= min_comments}
    kept_count = sum(len(v) for v in kept.values())

    # 组按 (pr, commit) 排序，保证输出稳定可复现
    sorted_keys = sorted(kept.keys(),
                         key=lambda k: (k[0] or 0, k[1]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for seq, k in enumerate(sorted_keys, 1):
            obj = build_group(kept[k], k, seq)
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    return {
        'total': len(records),
        'no_commit': no_commit,
        'groups': len(groups),
        'kept_groups': len(kept),
        'kept_comments': kept_count,
    }


def main():
    parser = argparse.ArgumentParser(description='精选数据集（按 (pr,commit) 二筛）')
    parser.add_argument('--repo', choices=list(REPO_BARE_DIRS.keys()),
                        help='只处理指定仓库（默认全部四仓）')
    parser.add_argument('--min-comments', type=int, default=10,
                        help='最小评论数过滤，>=该值才入选（默认 10）')
    args = parser.parse_args()

    repos = [args.repo] if args.repo else list(REPO_BARE_DIRS.keys())
    clean_dir = Path('history_real_comment/clean')
    refined_dir = Path('history_real_comment/refined')

    totals = {k: 0 for k in
              ['total', 'no_commit', 'groups', 'kept_groups', 'kept_comments']}

    for repo in repos:
        inp = clean_dir / f'{repo}.jsonl'
        outp = refined_dir / f'{repo}.jsonl'
        if not inp.exists():
            logging.info(f"跳过 {repo}: clean 数据不存在 {inp}")
            continue
        s = refine_repo(inp, outp, args.min_comments)
        for k in totals:
            totals[k] += s[k]
        logging.info(f"{repo}: {s['kept_groups']} 组 / {s['kept_comments']} 条入选 "
              f"(共 {s['total']} 条, {s['groups']} 组) → {outp}")

    logging.info(f"\n合计: {totals['kept_groups']} 组 / {totals['kept_comments']} 条入选 "
          f"(共 {totals['total']} 条)")


if __name__ == '__main__':
    main()
