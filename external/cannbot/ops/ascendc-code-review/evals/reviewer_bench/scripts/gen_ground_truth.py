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
从 refined JSONL 生成真值检视报告（Markdown 格式）

格式对齐 fia_block_vec_flashdecode.h_review_summary.md：
- 去掉"修复建议"和"正确示例"
- 保留代码片段（从 bare repo 提取被检视代码）
- 每条人工评论 → 一条"发现问题"
- 增加数据审查：过滤无用/刷评论数据

bare repo 自动管理：
- 首次运行时自动 clone 到 .repos/{repo}.git
- 自动 fetch PR ref（refs/merge-requests/{pr}/head）

用法:
    python scripts/gen_ground_truth.py                          # 全部四仓
    python scripts/gen_ground_truth.py --repo ops-transformer   # 单仓
    python scripts/gen_ground_truth.py --dry-run                # 只统计不写文件
"""

import logging
import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

REPO_LABEL = {
    'ops-transformer': 'ops-transformer',
    'ops-nn': 'ops-nn',
    'ops-cv': 'ops-cv',
    'ops-math': 'ops-math',
}

REPO_GITCODE_URL = {
    'ops-transformer': 'https://gitcode.com/cann/ops-transformer.git',
    'ops-nn': 'https://gitcode.com/cann/ops-nn.git',
    'ops-cv': 'https://gitcode.com/cann/ops-cv.git',
    'ops-math': 'https://gitcode.com/cann/ops-math.git',
}

MIN_BODY_LEN = 10
MAX_SAME_AUTHOR_RATIO = 0.8
MIN_VALID_COMMENTS = 3
SPAM_MIN_COMMENTS = 5
SPAM_UNIQUE_RATIO = 0.5

REPLY_PATTERNS = [
    '已修正', '已修改', '已修复', '已解决', '已删除', '已添加', '已更新',
    '已改', '已优化', '已调整', '已补齐', '已替换', '已重构',
    '感谢指正', '感谢提醒', '好的，', '收到，',
    '已 merge', '已merge', '已合并', '已提交', '已推送',
]
REPLY_EXACT = {'已修正', '已修改', '已修复', '已解决', '收到', '好的', '了解', '同意', '好的，已修改'}


# ── bare repo 管理 ────────────────────────────────────────────────────────

def get_bare_dir(repo: str, base_dir: Path) -> Path:
    return base_dir / '.repos' / f'{repo}.git'


def ensure_bare_repo(repo: str, base_dir: Path) -> Optional[Path]:
    bare_dir = get_bare_dir(repo, base_dir)
    if bare_dir.exists():
        return bare_dir
    url = REPO_GITCODE_URL.get(repo)
    if not url:
        return None
    bare_dir.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f'  clone {repo} → {bare_dir} ...')
    r = subprocess.run(
        ['git', 'clone', '--bare', url, str(bare_dir)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        logging.error(f'  clone 失败: {r.stderr.strip()}')
        return None
    return bare_dir


def ensure_pr_ref(repo: str, pr_number: int, base_dir: Path) -> bool:
    bare_dir = get_bare_dir(repo, base_dir)
    if not bare_dir.exists():
        return False
    ref = f'pr_{pr_number}'
    check = subprocess.run(
        ['git', 'rev-parse', '--verify', ref],
        cwd=bare_dir, capture_output=True, text=True,
    )
    if check.returncode == 0:
        return True
    subprocess.run(
        ['git', 'fetch', 'origin',
         f'refs/merge-requests/{pr_number}/head:{ref}'],
        cwd=bare_dir, capture_output=True, text=True, timeout=120,
    )
    verify = subprocess.run(
        ['git', 'rev-parse', '--verify', ref],
        cwd=bare_dir, capture_output=True, text=True,
    )
    return verify.returncode == 0


# ── 代码片段提取 ──────────────────────────────────────────────────────────

_file_cache: Dict[Tuple[str, str, str], List[str]] = {}


def get_file_lines(repo: str, commit: str, file_path: str,
                   base_dir: Path) -> List[str]:
    cache_key = (repo, commit, file_path)
    if cache_key in _file_cache:
        return _file_cache[cache_key]
    bare_dir = get_bare_dir(repo, base_dir)
    if not bare_dir.exists():
        _file_cache[cache_key] = []
        return []
    r = subprocess.run(
        ['git', 'show', f'{commit}:{file_path}'],
        cwd=bare_dir, capture_output=True,
    )
    if r.returncode != 0:
        _file_cache[cache_key] = []
        return []
    try:
        text = r.stdout.decode('utf-8', errors='replace')
    except Exception:
        _file_cache[cache_key] = []
        return []
    lines = text.split('\n')
    lines = [l.rstrip('\r') for l in lines]
    _file_cache[cache_key] = lines
    return lines


def guess_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return {
        '.h': 'cpp', '.hpp': 'cpp', '.cpp': 'cpp', '.cc': 'cpp', '.c': 'c',
        '.py': 'python', '.cmake': 'cmake', '.txt': 'text',
        '.sh': 'bash', '.md': 'markdown', '.yaml': 'yaml', '.yml': 'yaml',
    }.get(ext, '')


def extract_code_snippet(repo: str, commit: str, file_path: str,
                         line_start: int, line_end: int,
                         base_dir: Path, total_lines: int = 10) -> str:
    lines = get_file_lines(repo, commit, file_path, base_dir)
    if not lines:
        return ''
    total = len(lines)
    if line_start <= 0:
        line_start = 1
    if line_end <= 0:
        line_end = line_start
    center = (line_start + line_end) // 2
    half = total_lines // 2
    start = max(1, center - half)
    end = min(total, start + total_lines - 1)
    snippet_lines = []
    for i in range(start, end + 1):
        snippet_lines.append(f'{i:4d} | {lines[i - 1]}')
    return '\n'.join(snippet_lines)


# ── 评论过滤 ──────────────────────────────────────────────────────────────

def is_reply(body: str) -> bool:
    stripped = body.strip()
    if stripped in REPLY_EXACT:
        return True
    for pat in REPLY_PATTERNS:
        if stripped.startswith(pat) and len(stripped) < 30:
            return True
    if (stripped.startswith('@') or stripped.startswith('[@')) and len(stripped) < 80:
        return True
    if stripped.startswith('>') and '@' in stripped:
        return True
    return False


def _is_cut_heading(stripped: str, kws: List[str]) -> bool:
    s = stripped.lstrip('#> `')
    s = s.lstrip('#> `')
    if s.startswith('**'):
        s = s[2:].lstrip()
    return any(s.startswith(kw) for kw in kws)


def strip_fix_suggestion(body: str) -> str:
    cut_keywords = [
        '建议修复', '修复建议', '修复意见',
        '正确示例', '正确写法', '修复方案',
    ]
    lines = body.split('\n')
    result = []
    in_cut = False
    for line in lines:
        stripped = line.strip()
        if in_cut:
            if stripped.startswith('####') or stripped.startswith('### '):
                in_cut = False
                result.append(line)
            else:
                continue
        else:
            if _is_cut_heading(stripped, cut_keywords):
                in_cut = True
                continue
            result.append(line)
    return '\n'.join(result).rstrip()


def load_refined(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def is_valid_comment(c: Dict[str, Any]) -> Tuple[bool, str]:
    body = (c.get('body') or '').strip()
    if not body:
        return False, 'empty_body'
    if len(body) < MIN_BODY_LEN:
        return False, 'too_short'
    if not c.get('diff_file'):
        return False, 'no_file'
    if not c.get('new_line'):
        return False, 'no_line'
    if c.get('quality') != 'accepted':
        return False, f"quality={c.get('quality', '')}"
    lower = body.lower()
    low_value = ['lgtm', '+1', 'good', 'nice', 'ok', '收到', '了解', '同意', '赞同']
    if lower in low_value:
        return False, 'low_value'
    if is_reply(body):
        return False, 'author_reply'
    if body.startswith('<table') or ('pipelinedetail' in lower and 'pipelineRunId' in lower):
        return False, 'bot_content'
    return True, ''


def filter_group(comments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = Counter()
    valid = []
    for c in comments:
        ok, reason = is_valid_comment(c)
        if ok:
            valid.append(c)
        else:
            stats[reason] += 1
    if not valid:
        return [], stats
    seen_bodies = set()
    deduped = []
    for c in valid:
        body_key = c['body'].strip()
        if body_key in seen_bodies:
            stats['duplicate_body'] += 1
            continue
        seen_bodies.add(body_key)
        deduped.append(c)
    author_counts = Counter(c.get('author', '') for c in deduped)
    spam_authors = set()
    for author, count in author_counts.items():
        if count > SPAM_MIN_COMMENTS and count / len(deduped) >= MAX_SAME_AUTHOR_RATIO:
            author_bodies = set(
                c['body'].strip() for c in deduped if c.get('author', '') == author
            )
            unique_ratio = len(author_bodies) / count
            if unique_ratio < SPAM_UNIQUE_RATIO:
                spam_authors.add(author)
                stats['spam_author'] += count
    if spam_authors:
        before = len(deduped)
        deduped = [c for c in deduped if c.get('author', '') not in spam_authors]
        stats['spam_removed'] = before - len(deduped)
    seen_keys = set()
    final = []
    for c in deduped:
        key = (c.get('author', ''), c.get('diff_file', ''), c.get('new_line', 0))
        if key in seen_keys:
            stats['dup_author_file_line'] += 1
            continue
        seen_keys.add(key)
        final.append(c)
    stats['valid'] = len(final)
    return final, stats


def parse_line_value(line_val: Any) -> Tuple[int, int]:
    if not line_val:
        return (0, 0)
    if isinstance(line_val, int):
        return (line_val, line_val)
    s = str(line_val).strip()
    if not s:
        return (0, 0)
    if '-' in s:
        parts = s.split('-', 1)
        try:
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return (0, 0)
    try:
        v = int(s)
        return (v, v)
    except ValueError:
        return (0, 0)


def guess_side(file_path: str) -> str:
    if '/op_kernel/' in file_path or '/op_kernel\\' in file_path:
        return 'Kernel侧'
    if '/op_host/' in file_path or '/op_host\\' in file_path:
        return 'Tiling侧'
    if '/op_graph/' in file_path or '/op_graph\\' in file_path:
        return 'Host侧'
    if '/op_tiling/' in file_path or '/op_tiling\\' in file_path:
        return 'Tiling侧'
    if '/op_api/' in file_path or '/op_api\\' in file_path:
        return 'Host侧'
    if '/tests/' in file_path or '/tests\\' in file_path:
        return '测试侧'
    return '通用'


def extract_operator(file_path: str) -> str:
    parts = file_path.replace('\\', '/').split('/')
    for i, p in enumerate(parts):
        if p in ('op_kernel', 'op_host', 'op_graph', 'op_tiling', 'op_api', 'tests'):
            if i > 0:
                return parts[i - 1]
    if parts:
        return parts[-1].split('.')[0]
    return 'unknown'


# ── 报告生成 ──────────────────────────────────────────────────────────────

def generate_report(group: Dict[str, Any], comments: List[Dict[str, Any]],
                    base_dir: Path) -> str:
    repo = group.get('repo', '')
    pr_number = group.get('pr_number', 0)
    commit_id = group.get('commit_id', '')[:12]

    files = sorted(set(c.get('diff_file', '') for c in comments if c.get('diff_file')))
    primary_file = files[0] if files else ''
    side = guess_side(primary_file)
    now = datetime.now().strftime('%Y-%m-%d')

    by_file: Dict[str, List[Dict]] = defaultdict(list)
    for c in comments:
        by_file[c.get('diff_file', '')].append(c)

    lines = []
    lines.append('# 代码检视报告')
    lines.append('')
    lines.append('## 检视概览')
    lines.append(f'- **仓库**: {repo}')
    lines.append(f'- **PR编号**: {pr_number}')
    lines.append(f'- **Commit**: {commit_id}')
    lines.append(f'- **代码文件**: {", ".join(files) if len(files) <= 3 else f"{len(files)} 个文件"}')
    lines.append(f'- **代码侧别**: {side}')
    lines.append(f'- **检视来源**: 人工检视评论（GitCode PR）')
    total = sum(1 for c in comments
                if strip_fix_suggestion((c.get('body') or '').strip()))
    lines.append(f'- **总评论数**: {total} 条')
    lines.append(f'- **检视时间**: {now}')
    lines.append('')

    lines.append('## 检视统计')
    lines.append('')
    lines.append('| 状态 | 条数 | 占比 |')
    lines.append('|-----|------|------|')
    lines.append(f'| FAIL（发现问题） | {total} | 100% |')
    lines.append('')
    lines.append('---')
    lines.append('')

    lines.append('## 发现问题')
    lines.append('')

    idx = 0
    for file_path in sorted(by_file.keys()):
        file_comments = by_file[file_path]
        file_side = guess_side(file_path)
        lines.append(f'### 文件: {file_path}（{file_side}）')
        lines.append('')

        for c in file_comments:
            author = c.get('author', 'unknown')
            line_start, line_end = parse_line_value(c.get('new_line'))
            body = c.get('body', '').strip()
            body = strip_fix_suggestion(body)
            if not body:
                continue
            idx += 1
            created = c.get('created_at', '')[:10]

            line_range = str(line_start) if line_start == line_end else f'{line_start}-{line_end}'

            lines.append(f'#### [{idx}] 人工检视意见')
            lines.append('')
            lines.append(f'- **评论者**: {author}')
            lines.append(f'- **文件**: {file_path}')
            lines.append(f'- **行号**: {line_range}')
            lines.append(f'- **评论时间**: {created}')
            lines.append(f'- **问题描述**:')
            lines.append('')
            for body_line in body.split('\n'):
                lines.append(f'  > {body_line}')
            lines.append('')

            snippet = extract_code_snippet(
                repo, commit_id, file_path, line_start, line_end, base_dir
            )
            if snippet:
                lang = guess_language(file_path)
                lines.append(f'- **代码片段**（行{line_range}）:')
                lines.append(f'```{lang}')
                lines.append(snippet)
                lines.append('```')
                lines.append('')

            lines.append('---')
            lines.append('')

    lines.append('## 被检视代码')
    lines.append('')
    lines.append(f'> 本报告基于 PR {pr_number}（commit {commit_id}）的人工检视评论生成')
    lines.append('')
    if len(files) <= 5:
        for fp in files:
            lines.append(f'- `{fp}`')
    else:
        for fp in files[:5]:
            lines.append(f'- `{fp}`')
        lines.append(f'- ... 共 {len(files)} 个文件')
    lines.append('')

    return '\n'.join(lines)


# ── 主流程 ────────────────────────────────────────────────────────────────

def process_repo(repo: str, base_dir: Path, dry_run: bool) -> Dict[str, Any]:
    refined_path = base_dir / 'history_real_comment' / 'refined' / f'{repo}.jsonl'
    output_dir = base_dir / 'benchmark_tasks' / f'{repo}-ground-truth-reports'

    if not refined_path.exists():
        logging.info(f'  跳过 {repo}: refined 数据不存在')
        return {}

    bare_dir = ensure_bare_repo(repo, base_dir)
    if not bare_dir:
        logging.info(f'  跳过 {repo}: bare repo 不可用')
        return {}

    groups = load_refined(refined_path)
    logging.info(f'  {repo}: {len(groups)} 组 refined 数据')

    total_stats = Counter()
    generated = 0
    skipped = 0
    fetched_prs = set()

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for g in groups:
        comments = g.get('comments', [])
        valid_comments, stats = filter_group(comments)
        total_stats.update(stats)

        if len(valid_comments) < MIN_VALID_COMMENTS:
            total_stats['skip_low_count'] += 1
            skipped += 1
            continue

        pr_number = g.get('pr_number', 0)
        commit_id = g.get('commit_id', '')[:12]

        if pr_number not in fetched_prs:
            ensure_pr_ref(repo, pr_number, base_dir)
            fetched_prs.add(pr_number)

        if dry_run:
            generated += 1
            logging.info(f'    [DRY] pr={pr_number} commit={commit_id} valid={len(valid_comments)}/{len(comments)}')
            continue

        _file_cache.clear()

        report = generate_report(g, valid_comments, base_dir)

        filename = f'{repo}_pr_{pr_number:04d}_{commit_id}.md'
        filepath = output_dir / filename
        filepath.write_text(report, encoding='utf-8')
        generated += 1

    result = {
        'repo': repo,
        'groups': len(groups),
        'generated': generated,
        'skipped': skipped,
        'stats': dict(total_stats),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='从 refined JSONL 生成真值检视报告')
    parser.add_argument('--repo', choices=list(REPO_LABEL.keys()),
                        help='只处理指定仓库（默认全部）')
    parser.add_argument('--dry-run', action='store_true',
                        help='只统计不写文件')
    args = parser.parse_args()

    repos = [args.repo] if args.repo else list(REPO_LABEL.keys())
    base_dir = Path(__file__).resolve().parent.parent

    logging.info('=== 生成真值检视报告 ===')
    all_stats = Counter()
    for repo in repos:
        logging.info(f'\n[{repo}]')
        result = process_repo(repo, base_dir, args.dry_run)
        if not result:
            continue
        logging.info(f'  生成: {result["generated"]}  跳过: {result["skipped"]}')
        logging.info(f'  审查统计: {result["stats"]}')
        all_stats.update(result['stats'])

    logging.info(f'\n=== 汇总 ===')
    logging.info(f'审查过滤统计: {dict(all_stats)}')


if __name__ == '__main__':
    main()
