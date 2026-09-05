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
恢复到 PR 评论发生时的代码状态

从 clean JSONL 数据中读取记录，用与 gen_proposer_sample.py 一致的 commit 定位
算法（commit_date %ci + amend 检测）恢复到评论时的代码状态。

工作原理：
  1. ensure_bare_repo: 自动 clone bare repo 到 {base_dir}/.repos/{repo}.git
  2. ensure_pr_ref: fetch PR ref（refs/merge-requests/{pr}/head）
  3. get_commit_for_comment: 用 commit_date 定位评论时的 commit（与 gen_proposer_sample.py 一致）
  4. post_fix 检查: commit_date 晚于评论时间则拒绝（amend 后的修复版）
  5. cp -r bare_repo → target/.git + git checkout commit

用法:
    # 从 clean 数据恢复单条
    python3 scripts/restore_pr_repo.py \
        --input history_real_comment/clean/ops-cv.jsonl \
        --pr 345 --commit f0f5426862ef \
        --output-dir /tmp/restored

    # 只提取单个文件（快速验证）
    python3 scripts/restore_pr_repo.py \
        --input history_real_comment/clean/ops-cv.jsonl \
        --pr 345 --commit f0f5426862ef \
        --extract-only

    # 批量恢复
    python3 scripts/restore_pr_repo.py \
        --input history_real_comment/clean/ops-cv.jsonl \
        --batch --output-dir /tmp/restored_cv
"""

import logging
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from gen_proposer_sample import (
    REPO_GITCODE_URL,
    get_bare_dir,
    ensure_bare_repo,
    ensure_pr_ref,
    get_commit_for_comment,
    normalize_time,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)


def find_record(input_path: Path, pr_number: int, commit_id: str) -> Optional[dict]:
    """从 JSONL 中查找匹配的记录"""
    commit_prefix = commit_id[:12]
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get('pr_number') == pr_number
                    and r.get('commit_id', '').startswith(commit_prefix)):
                return r
    return None


def resolve_commit(repo: str, pr_number: int, created_at: str,
                   diff_file: str, base_dir: Path) -> str:
    """用 gen_proposer_sample.py 的算法定位评论时的 commit

    包含 post_fix 检查：若 commit_date 晚于评论时间（amend 后的修复版），
    返回空字符串。
    """
    bare_dir = ensure_bare_repo(repo, base_dir)
    if not bare_dir:
        logging.error(f"  错误: bare repo 不可用: {repo}")
        return ''

    if not ensure_pr_ref(repo, pr_number, base_dir):
        logging.error(f"  错误: 无法 fetch PR ref: {repo}#{pr_number}")
        return ''

    commit = get_commit_for_comment(repo, pr_number, created_at, diff_file, base_dir)
    if not commit:
        logging.error(f"  错误: 无法定位 commit")
        return ''

    r = subprocess.run(
        ['git', 'log', '-1', '--format=%ci', commit],
        cwd=bare_dir, capture_output=True, text=True,
    )
    if r.returncode == 0 and r.stdout.strip():
        if normalize_time(r.stdout.strip()) > normalize_time(created_at):
            logging.warning(f"  警告: commit {commit} 是 post_fix（commit_date 晚于评论时间）")
            return ''

    return commit


def restore_full_repo(repo_name: str, commit: str, target_dir: Path,
                      base_dir: Path) -> bool:
    """恢复完整仓库到指定 commit 的工作目录

    Args:
        repo_name: 仓库名 (ops-transformer/ops-nn/ops-cv/ops-math)
        commit: 目标 commit hash (完整或前缀)
        target_dir: 输出目录（会被清空重建）
        base_dir: 项目根目录（用于定位 bare repo）

    Returns:
        True 表示成功
    """
    bare_dir = get_bare_dir(repo_name, base_dir)
    if not bare_dir.exists():
        logging.error(f"错误: bare repo 不存在: {bare_dir}")
        return False

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    git_dir = target_dir / '.git'

    logging.info(f"  复制 bare repo ({bare_dir}) ...")
    shutil.copytree(bare_dir, git_dir, symlinks=True)

    subprocess.run(
        ['git', 'config', '--local', 'core.bare', 'false'],
        cwd=target_dir, capture_output=True, check=True
    )

    logging.info(f"  checkout {commit[:12]} ...")
    result = subprocess.run(
        ['git', 'checkout', commit],
        cwd=target_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        logging.error(f"  checkout 失败: {result.stderr.strip()}")
        shutil.rmtree(target_dir)
        return False

    head = subprocess.run(
        ['git', 'log', '--oneline', '-1'],
        cwd=target_dir, capture_output=True, text=True
    )
    file_count = len(list(target_dir.rglob('*')))
    git_count = len(list(git_dir.rglob('*')))
    logging.info(f"  ✓ HEAD: {head.stdout.strip()}")
    logging.info(f"  ✓ 文件数: {file_count - git_count}")

    return True


def extract_file(repo_name: str, commit: str, diff_file: str,
                 target_dir: Path, base_dir: Path) -> bool:
    """只提取单个文件（快速验证用，不需要完整 checkout）

    Args:
        repo_name: 仓库名
        commit: 目标 commit
        diff_file: 文件在 commit 中的路径
        target_dir: 输出目录（文件将保存在 target_dir/repo_name/diff_file）
        base_dir: 项目根目录

    Returns:
        True 表示成功
    """
    bare_dir = get_bare_dir(repo_name, base_dir)
    if not bare_dir.exists():
        logging.error(f"错误: bare repo 不存在: {bare_dir}")
        return False

    check = subprocess.run(
        ['git', 'cat-file', '-t', f'{commit}:{diff_file}'],
        cwd=bare_dir, capture_output=True, text=True
    )
    if check.returncode != 0:
        logging.info(f"  ✗ 文件在 commit 中不存在: {diff_file}")
        return False

    out_path = target_dir / repo_name / diff_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ['git', 'show', f'{commit}:{diff_file}'],
        cwd=bare_dir, capture_output=True, text=True
    )
    out_path.write_text(result.stdout, encoding='utf-8')
    line_count = result.stdout.count('\n')
    logging.info(f"  ✓ {diff_file} ({line_count} lines, commit {commit[:12]})")
    return True


def verify_record(record: dict, base_dir: Path) -> dict:
    """验证单条记录的 commit → 文件 → 行号可达性

    Returns:
        {'match': bool, 'offset': int, 'detail': str, 'resolved_commit': str}
    """
    repo = record.get('repo', '')
    pr_number = record.get('pr_number', 0)
    created_at = record.get('created_at', '')
    diff_file = record.get('diff_file', '')
    new_line = record.get('new_line')

    commit = resolve_commit(repo, pr_number, created_at, diff_file, base_dir)
    if not commit:
        return {'match': False, 'offset': 999, 'detail': '无法定位 commit',
                'resolved_commit': ''}

    bare_dir = get_bare_dir(repo, base_dir)

    check = subprocess.run(
        ['git', 'cat-file', '-t', f'{commit}:{diff_file}'],
        cwd=bare_dir, capture_output=True, text=True
    )
    if check.returncode != 0:
        return {'match': False, 'offset': 999, 'detail': f'文件不存在: {diff_file}',
                'resolved_commit': commit}

    result = subprocess.run(
        ['git', 'show', f'{commit}:{diff_file}'],
        cwd=bare_dir, capture_output=True, text=True
    )
    total_lines = result.stdout.count('\n')

    if new_line and new_line > total_lines:
        return {'match': False, 'offset': new_line - total_lines,
                'detail': f'new_line={new_line} 超出文件 ({total_lines} lines)',
                'resolved_commit': commit}

    return {'match': True, 'offset': 0, 'detail': f'OK ({total_lines} lines)',
            'resolved_commit': commit}


def main():
    parser = argparse.ArgumentParser(
        description='恢复 PR 评论时的代码状态（与 gen_proposer_sample.py 一致的 commit 定位）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s -i clean/ops-cv.jsonl --pr 345 --commit f0f5426862ef -o /tmp/restored
  %(prog)s -i clean/ops-cv.jsonl --pr 345 --commit f0f5426862ef --extract-only
  %(prog)s -i clean/ops-cv.jsonl --batch -o /tmp/restored_cv
        """)

    parser.add_argument('--input', '-i', type=Path, required=True,
                        help='clean JSONL 文件路径')
    parser.add_argument('--output-dir', '-o', type=Path,
                        default=Path('/tmp/restored_repos'),
                        help='输出目录 (默认 /tmp/restored_repos)')
    parser.add_argument('--pr', type=int, help='PR 编号')
    parser.add_argument('--commit', type=str, help='commit hash (至少 6 位，用于定位记录)')
    parser.add_argument('--batch', action='store_true',
                        help='批量恢复模式 (恢复 JSONL 中所有记录)')
    parser.add_argument('--extract-only', action='store_true',
                        help='只提取目标文件，不做完整 checkout')
    parser.add_argument('--max', type=int, default=0,
                        help='批量模式最大恢复数量 (0=全部)')
    parser.add_argument('--verify', action='store_true',
                        help='只验证不恢复 (检查文件+行号可达性)')
    parser.add_argument('--base-dir', type=Path, default=None,
                        help='项目根目录 (默认: 本文件的上级目录)')

    args = parser.parse_args()

    base_dir = args.base_dir or Path(__file__).resolve().parent.parent

    if not args.input.exists():
        logging.error(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)

    repo_name = None
    for name in REPO_GITCODE_URL:
        if name in str(args.input):
            repo_name = name
            break
    if not repo_name:
        logging.error("错误: 无法从文件名推断仓库，请确认文件名包含仓库名")
        sys.exit(1)

    if args.verify:
        if not args.pr or not args.commit:
            parser.error('--verify 需要 --pr 和 --commit')
        record = find_record(args.input, args.pr, args.commit)
        if not record:
            logging.info(f"未找到 PR#{args.pr} commit={args.commit} 的记录")
            sys.exit(1)
        result = verify_record(record, base_dir)
        print(json.dumps({**result, 'record': {
            'pr': record['pr_number'], 'file': record['diff_file'],
            'line': record['new_line'], 'body': record['body'][:100]
        }}, ensure_ascii=False, indent=2))  # noqa
        sys.exit(0 if result['match'] else 1)

    if args.extract_only:
        if not args.pr or not args.commit:
            parser.error('--extract-only 需要 --pr 和 --commit')
        record = find_record(args.input, args.pr, args.commit)
        if not record:
            logging.info(f"未找到 PR#{args.pr} commit={args.commit} 的记录")
            sys.exit(1)
        commit = resolve_commit(repo_name, args.pr,
                                record.get('created_at', ''),
                                record.get('diff_file', ''), base_dir)
        if not commit:
            sys.exit(1)
        ok = extract_file(repo_name, commit, record['diff_file'],
                          args.output_dir, base_dir)
        sys.exit(0 if ok else 1)

    if args.batch:
        with open(args.input, 'r', encoding='utf-8') as f:
            lines = [l for l in f if l.strip()]

        records = [json.loads(l) for l in lines
                   if json.loads(l).get('commit_id')]
        if args.max > 0:
            records = records[:args.max]

        logging.info(f"批量恢复 {len(records)} 条记录 ...")
        success = 0
        skipped = 0
        seen = set()
        for i, r in enumerate(records):
            pr = r['pr_number']
            created_at = r.get('created_at', '')
            diff_file = r.get('diff_file', '')

            commit = resolve_commit(repo_name, pr, created_at, diff_file, base_dir)
            if not commit:
                skipped += 1
                continue

            dedup_key = (repo_name, pr, commit)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            target = args.output_dir / f"{repo_name}_pr{pr}_{commit[:12]}"
            logging.info(f"[{i+1}/{len(records)}] PR#{pr} commit={commit[:12]}")
            if restore_full_repo(repo_name, commit, target, base_dir):
                success += 1

        logging.info(f"\n完成: {success} 成功, {skipped} 跳过 (post_fix/无法定位)")
        sys.exit(0 if success > 0 else 1)

    if not args.pr or not args.commit:
        parser.error('需要 --pr 和 --commit')
    record = find_record(args.input, args.pr, args.commit)
    if not record:
        logging.info(f"未找到 PR#{args.pr} commit={args.commit} 的记录")
        sys.exit(1)

    commit = resolve_commit(repo_name, args.pr,
                            record.get('created_at', ''),
                            record.get('diff_file', ''), base_dir)
    if not commit:
        sys.exit(1)

    if commit != record.get('commit_id', '')[:12]:
        logging.warning(f"  警告: 重新定位 commit={commit} 与记录中 commit_id="
              f"{record.get('commit_id', '')[:12]} 不同")

    target = args.output_dir / f"{repo_name}_pr{args.pr}_{commit[:12]}"
    logging.info(f"恢复 {repo_name} PR#{args.pr} → {target}")
    ok = restore_full_repo(repo_name, commit, target, base_dir)
    if ok:
        logging.info(f"成功: {target}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
