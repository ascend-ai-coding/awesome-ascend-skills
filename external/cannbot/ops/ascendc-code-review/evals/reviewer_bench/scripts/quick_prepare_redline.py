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
"""快速准备红线 PR 测试数据（绕过 bare repo PR ref 问题）

直接 clone 仓库 → fetch PR merge ref → checkout → 生成 diff → 输出 manifest

用法:
    python scripts/quick_prepare_redline.py --top 3
    python scripts/quick_prepare_redline.py --prs ops-math:93,ops-math:387
"""
import logging
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

DEFAULT_REPO_URLS = {
    'ops-math': 'https://gitcode.com/cann/ops-math.git',
    'ops-nn': 'https://gitcode.com/cann/ops-nn.git',
    'ops-cv': 'https://gitcode.com/cann/ops-cv.git',
    'ops-transformer': 'https://gitcode.com/cann/ops-transformer.git',
}

FILENAME_RE = re.compile(r'(?:\d+_)?([\w-]+)_pr_(\d+)_([0-9a-f]+)\.md')
CODE_EXTS = {'.cpp', '.h', '.hpp', '.cc', '.c', '.cu', '.cuh', '.py'}


def parse_report_summary(md_path: Path) -> list:
    """从 ground-truth 报告提取 PR 信息和文件数"""
    text = md_path.read_text(encoding='utf-8', errors='replace')
    m = FILENAME_RE.search(md_path.name)
    if not m:
        return []
    repo = m.group(1)
    pr_number = int(m.group(2))
    commit = m.group(3)

    fc_match = re.search(r'代码文件.*?(\d+)\s*个文件', text)
    file_count = int(fc_match.group(1)) if fc_match else 99

    return [(repo, pr_number, commit, file_count, md_path.name)]


def _rmtree_force(path: Path):
    """强制删除目录（处理 git 只读文件 + Windows 长路径）"""
    import stat
    import os as _os
    if sys.platform == 'win32':
        empty = path.parent / f'_empty_{path.name}'
        empty.mkdir(exist_ok=True)
        subprocess.run(['robocopy', str(empty), str(path), '/MIR', '/NFL', '/NDL', '/NJH', '/NJS'],
                       capture_output=True, timeout=120)
        try:
            empty.rmdir()
        except Exception:
            pass
        try:
            _os.rmdir(str(path))
        except Exception:
            pass
    else:
        def onerror(func, fpath, exc_info):
            _os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        shutil.rmtree(path, onerror=onerror)


def clone_and_prepare(repo: str, pr_number: int, base_commit: str,
                      data_dir: Path, base_dir: Path, force: bool = False,
                      repo_urls: dict = None) -> dict:
    """clone 仓库 → fetch PR ref → checkout PR head → 生成 PR 全量 diff → 重置 master 到 PR base

    GT 报告文件名中的 commit 是 PR **base**（检视基准点），不是 PR head。
    PR head 通过 fetch refs/merge-requests/N/head 获得。
    diff 范围 = base_commit..pr_head，覆盖 PR 全部变更。
    """
    if repo_urls is None:
        repo_urls = DEFAULT_REPO_URLS
    repo_url = repo_urls.get(repo)
    if not repo_url:
        return {}

    dir_name = f'{repo}-pr-{pr_number}'
    repo_dir = data_dir / dir_name / 'repo'
    diff_path = data_dir / dir_name / 'changes.diff'

    if not force and repo_dir.exists() and diff_path.exists() and diff_path.stat().st_size > 0:
        code_files = _get_code_files_from_diff(diff_path)
        if code_files:
            return {
                'repo': repo, 'pr_number': str(pr_number),
                'commit_hash': base_commit,
                'repo_dir': str(repo_dir.relative_to(base_dir)),
                'diff_file': str(diff_path.relative_to(base_dir)),
                'code_files': code_files,
            }

    bare_dir = data_dir / '_bare' / f'{repo}.git'
    bare_dir.parent.mkdir(parents=True, exist_ok=True)

    if not bare_dir.exists():
        logging.info(f'  clone bare {repo}...')
        r = subprocess.run(
            ['git', 'clone', '--bare', repo_url, str(bare_dir)],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            logging.error(f'  clone 失败: {r.stderr[:200]}')
            return {}

    logging.info(f'  fetch PR {pr_number} merge ref...')
    mr_ref = f'mr_{pr_number}'
    subprocess.run(
        ['git', 'fetch', 'origin', f'refs/merge-requests/{pr_number}/head:{mr_ref}'],
        cwd=bare_dir, capture_output=True, text=True, timeout=120,
    )

    pr_head = None
    check_mr = subprocess.run(
        ['git', 'rev-parse', '--verify', mr_ref],
        cwd=bare_dir, capture_output=True, text=True,
    )
    if check_mr.returncode == 0:
        pr_head = check_mr.stdout.strip()
        logging.info(f'  PR head: {pr_head[:12]}')
    else:
        logging.info(f'  无法获取 PR head ref mr_{pr_number}')
        return {}

    check_base = subprocess.run(
        ['git', 'cat-file', '-t', base_commit[:12]],
        cwd=bare_dir, capture_output=True, text=True,
    )
    if check_base.returncode != 0:
        logging.info(f'  PR base commit {base_commit[:12]} 不存在于 bare repo，跳过')
        return {}
    logging.info(f'  PR base: {base_commit[:12]}')

    if not repo_dir.exists() or force:
        if repo_dir.exists():
            _rmtree_force(repo_dir)
        repo_dir.mkdir(parents=True, exist_ok=True)
        git_dir = repo_dir / '.git'
        logging.info(f'  复制 bare repo...')
        shutil.copytree(bare_dir, git_dir, symlinks=True)
        subprocess.run(['git', 'config', '--local', 'core.bare', 'false'],
                       cwd=repo_dir, capture_output=True)
        logging.info(f'  checkout PR head {pr_head[:12]}...')
        subprocess.run(['git', 'checkout', pr_head],
                       cwd=repo_dir, capture_output=True, text=True)
        logging.info(f'  reset master -> PR base {base_commit[:12]}...')
        subprocess.run(['git', 'branch', '-f', 'master', base_commit],
                       cwd=repo_dir, capture_output=True, text=True)

    if not diff_path.exists() or diff_path.stat().st_size == 0 or force:
        logging.info(f'  生成 PR 全量 diff ({base_commit[:12]}..{pr_head[:12]})...')
        r = subprocess.run(
            ['git', 'diff', f'{base_commit}..{pr_head}'],
            cwd=bare_dir, capture_output=True, text=True, encoding='utf-8', errors='replace',
        )
        if r.returncode != 0 or not r.stdout.strip():
            logging.error(f'  diff 生成失败: {r.stderr[:200]}')
            return {}
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(r.stdout, encoding='utf-8')
        file_count = len(re.findall(r'^diff --git', r.stdout, re.MULTILINE))
        logging.info(f'  diff: {file_count} 文件')

    code_files = _get_code_files_from_diff(diff_path)

    return {
        'repo': repo, 'pr_number': str(pr_number),
        'commit_hash': pr_head,
        'base_commit': base_commit,
        'repo_dir': str(repo_dir.relative_to(base_dir)),
        'diff_file': str(diff_path.relative_to(base_dir)),
        'code_files': code_files,
    }


def _get_code_files_from_diff(diff_path: Path) -> list:
    if not diff_path.exists():
        return []
    raw = diff_path.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        text = raw.decode('utf-16', errors='replace')
    elif raw[:3] == b'\xef\xbb\xbf':
        text = raw[3:].decode('utf-8', errors='replace')
    else:
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            text = raw.decode('gbk', errors='replace')
    files = []
    for m in re.finditer(r'^diff --git a/(.+?) b/(.+?)$', text, re.MULTILINE):
        f = m.group(2)
        if Path(f).suffix.lower() in CODE_EXTS:
            files.append(f)
    return list(dict.fromkeys(files))


def main():
    parser = argparse.ArgumentParser(description='快速准备红线 PR 测试数据')
    parser.add_argument('--reports-dir', default='benchmark_tasks/top20_redline_and_topk_filtered')
    parser.add_argument('--top', type=int, default=3, help='取最简单的 N 个')
    parser.add_argument('--prs', type=str, default=None,
                        help='手动指定 PR（格式: repo:pr,repo:pr）')
    parser.add_argument('--data-dir', default='bench_data_redline')
    parser.add_argument('--output', default='manifest_redline.json')
    parser.add_argument('--force', action='store_true', help='强制重新生成（覆盖已有数据）')
    parser.add_argument('--repo-url', action='append', default=[],
                        help='自定义仓库 URL（格式: repo:url，可多次指定，覆盖默认值）')
    args = parser.parse_args()

    repo_urls = dict(DEFAULT_REPO_URLS)
    for item in args.repo_url:
        parts = item.split(':', 1)
        if len(parts) == 2:
            repo_urls[parts[0]] = parts[1]

    base_dir = Path(__file__).resolve().parent.parent
    reports_dir = base_dir / args.reports_dir
    data_dir = base_dir / args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.prs:
        triples = []
        for item in args.prs.split(','):
            parts = item.strip().split(':')
            if len(parts) == 2:
                repo, pr_num = parts[0], int(parts[1])
                base_commit = ''
                pr_padded = f'{pr_num:04d}'
                for pattern in [f'*{repo}_pr_{pr_num}_*.md', f'*{repo}_pr_{pr_padded}_*.md']:
                    matches = list(reports_dir.glob(pattern))
                    if matches:
                        m = FILENAME_RE.search(matches[0].name)
                        if m:
                            base_commit = m.group(3)
                        break
                if not base_commit:
                    logging.warning(f'  警告: 未找到 {repo} PR {pr_num} 的 GT 报告，无法确定 base commit')
                triples.append((repo, pr_num, base_commit, 0, ''))
    else:
        all_reports = []
        for md in sorted(reports_dir.glob('*.md')):
            all_reports.extend(parse_report_summary(md))
        all_reports.sort(key=lambda x: x[3])
        triples = all_reports[:args.top]

    logging.info(f'准备 {len(triples)} 个 PR')

    entries = []
    for idx, (repo, pr, commit, fc, _) in enumerate(triples, 1):
        logging.info(f'\n[{idx}/{len(triples)}] {repo} PR {pr} (commit {commit[:12] if commit else "?"})')
        entry = clone_and_prepare(repo, pr, commit, data_dir, base_dir, force=args.force, repo_urls=repo_urls)
        if entry:
            entries.append(entry)
            logging.info(f'  ✓ {len(entry["code_files"])} 个代码文件')
        else:
            logging.error(f'  ✗ 失败')

    manifest = {'total': len(entries), 'failed': len(triples) - len(entries), 'entries': entries}
    out_path = base_dir / args.output
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    logging.error(f'\n完成: {len(entries)} 成功, {len(triples) - len(entries)} 失败')
    logging.info(f'manifest: {out_path}')

    for e in entries:
        logging.info(f'  {e["repo"]} PR {e["pr_number"]}: {len(e["code_files"])} 文件, '
              f'diff {Path(e["diff_file"]).stat().st_size if Path(base_dir / e["diff_file"]).exists() else 0} bytes')


if __name__ == '__main__':
    main()
