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
"""准备评测数据：扫描 ground-truth 报告 → 恢复代码 + 生成 diff → 输出 manifest

与 gen_bench_config.py 配合使用：
  1. 本脚本（耗时）：恢复代码、生成 diff、提取代码文件清单 → manifest.json
  2. gen_bench_config.py（秒级）：读 manifest → 组装 prompt → 输出 batch_review config

用法:
    python scripts/prepare_bench_data.py --reports-dir benchmark_tasks/top20_transformer_filtered
    python scripts/prepare_bench_data.py --reports-dir benchmark_tasks/ops-cv-ground-truth-reports
"""
import logging
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from gen_proposer_sample import (
    get_bare_dir,
    ensure_bare_repo,
    ensure_pr_ref,
    get_commit_for_comment,
    normalize_time,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

DEFAULT_REPOS = ["ops-cv", "ops-math", "ops-nn", "ops-transformer"]
FILENAME_RE = re.compile(r'(?:\d+_)?([\w-]+)_pr_(\d+)_([0-9a-f]+)\.md')

CODE_EXTS = {'.cpp', '.h', '.hpp', '.cc', '.c', '.cu', '.cuh', '.py'}


def is_code_file(path: str) -> bool:
    return Path(path).suffix.lower() in CODE_EXTS


def find_created_at(repo: str, pr_number: int, commit_hash: str,
                    reports_dir: Path) -> str:
    """从 clean JSONL 找到该 PR 任意评论的 created_at（用于 post_fix 校验）"""
    clean_path = Path('history_real_comment/clean') / f'{repo}.jsonl'
    if not clean_path.exists():
        return ''
    commit_prefix = commit_hash[:12]
    with open(clean_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get('pr_number') == pr_number
                    and r.get('commit_id', '').startswith(commit_prefix)):
                return r.get('created_at', '')
    return ''


def resolve_commit(repo: str, pr_number: int, commit_hash: str,
                   created_at: str, base_dir: Path) -> str:
    """用 gen_proposer_sample.py 的算法定位 commit + post_fix 校验"""
    bare_dir = ensure_bare_repo(repo, base_dir)
    if not bare_dir:
        return ''

    if not ensure_pr_ref(repo, pr_number, base_dir):
        return ''

    diff_file = ''
    clean_path = Path('history_real_comment/clean') / f'{repo}.jsonl'
    commit_prefix = commit_hash[:12]
    if clean_path.exists():
        with open(clean_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if (r.get('pr_number') == pr_number
                        and r.get('commit_id', '').startswith(commit_prefix)):
                    diff_file = r.get('diff_file', '')
                    if diff_file:
                        break

    if not diff_file:
        return commit_hash

    commit = get_commit_for_comment(repo, pr_number, created_at, diff_file, base_dir)
    if not commit:
        return ''

    r = subprocess.run(
        ['git', 'log', '-1', '--format=%ci', commit],
        cwd=bare_dir, capture_output=True, text=True,
    )
    if r.returncode == 0 and r.stdout.strip() and created_at:
        if normalize_time(r.stdout.strip()) > normalize_time(created_at):
            return ''

    return commit


def generate_diff(bare_dir: Path, commit: str, output_path: Path) -> bool:
    """生成 commit 相对其父的 diff"""
    result = subprocess.run(
        ['git', 'diff', f'{commit}^..{commit}'],
        cwd=bare_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ['git', 'show', '--format=', commit],
            cwd=bare_dir, capture_output=True, text=True,
        )
    if result.returncode != 0:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.stdout, encoding='utf-8')
    return bool(result.stdout.strip())


def get_changed_code_files(bare_dir: Path, commit: str) -> list:
    """提取 commit 变更的代码文件清单"""
    result = subprocess.run(
        ['git', 'diff', '--name-only', f'{commit}^..{commit}'],
        cwd=bare_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ['git', 'show', '--name-only', '--format=', commit],
            cwd=bare_dir, capture_output=True, text=True,
        )
    files = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line and is_code_file(line):
            files.append(line)
    return files


def main():
    parser = argparse.ArgumentParser(description='准备评测数据：恢复代码 + 生成 diff')
    parser.add_argument('--reports-dir', default='benchmark_tasks/top20_transformer_filtered',
                        help='ground-truth 报告目录（直接包含 .md 文件）')
    parser.add_argument('--output', default='manifest.json', help='输出 manifest 路径')
    parser.add_argument('--data-dir', default='bench_data',
                        help='代码和 diff 的存放目录')
    parser.add_argument('--base-dir', default=None, help='项目根目录')
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).resolve().parent.parent
    base_dir = base_dir.resolve()

    triples = []
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        logging.error(f'错误：报告目录不存在: {reports_dir}')
        sys.exit(1)
    for md in sorted(reports_dir.glob('*.md')):
        m = FILENAME_RE.search(md.name)
        if not m:
            continue
        triples.append((m.group(1), str(int(m.group(2))), m.group(3)))

    if not triples:
        logging.error('错误：未扫描到任何报告')
        sys.exit(1)

    logging.info(f'扫描到 {len(triples)} 份报告')

    from collections import Counter
    pr_counts = Counter((repo, pr) for repo, pr, _ in triples)

    data_dir = base_dir / args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    failed = []
    for idx, (repo, pr, commit_hash) in enumerate(triples, 1):
        logging.info(f'[{idx}/{len(triples)}] {repo} PR {pr} commit {commit_hash[:12]}')

        created_at = find_created_at(repo, int(pr), commit_hash, Path(args.reports_dir))

        commit = resolve_commit(repo, int(pr), commit_hash, created_at, base_dir)
        if not commit:
            logging.error(f'  跳过: 无法定位 commit (post_fix 或 fetch 失败)')
            failed.append({'repo': repo, 'pr': pr, 'commit': commit_hash,
                           'reason': 'resolve_commit_failed'})
            continue

        if commit != commit_hash[:12]:
            logging.warning(f'  警告: 重新定位 commit {commit} != 记录 {commit_hash[:12]}')

        bare_dir = get_bare_dir(repo, base_dir)

        if pr_counts[(repo, pr)] > 1:
            dir_name = f'{repo}-pr-{pr}-{commit[:7]}'
        else:
            dir_name = f'{repo}-pr-{pr}'
        repo_dir = data_dir / dir_name / 'repo'
        diff_path = data_dir / dir_name / 'changes.diff'

        if not repo_dir.exists():
            import shutil
            repo_dir.mkdir(parents=True, exist_ok=True)
            git_dir = repo_dir / '.git'
            shutil.copytree(bare_dir, git_dir, symlinks=True)
            subprocess.run(['git', 'config', '--local', 'core.bare', 'false'],
                           cwd=repo_dir, capture_output=True)
            subprocess.run(['git', 'checkout', commit],
                           cwd=repo_dir, capture_output=True, text=True)

        if not diff_path.exists():
            generate_diff(bare_dir, commit, diff_path)

        code_files = get_changed_code_files(bare_dir, commit)

        entry = {
            'repo': repo,
            'pr_number': pr,
            'commit_hash': commit,
            'repo_dir': str(repo_dir.relative_to(base_dir)),
            'diff_file': str(diff_path.relative_to(base_dir)),
            'code_files': code_files,
        }
        entries.append(entry)
        logging.info(f'  ✓ {len(code_files)} 个代码文件, diff {diff_path.stat().st_size} bytes')

    manifest = {
        'total': len(entries),
        'failed': len(failed),
        'entries': entries,
        'failures': failed,
    }

    output_path = base_dir / args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logging.error(f'\n完成: {len(entries)} 成功, {len(failed)} 失败')
    logging.info(f'manifest: {output_path}')
    if failed:
        logging.error('失败项:')
        for fail in failed:
            logging.info(f'  {fail["repo"]} PR {fail["pr"]} {fail["commit"][:12]}: {fail["reason"]}')


if __name__ == '__main__':
    main()
