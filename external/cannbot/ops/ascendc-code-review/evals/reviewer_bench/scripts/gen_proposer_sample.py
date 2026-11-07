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
"""从现有 refined 数据生成带提出人的样例报告（无需 API Token）

核心逻辑：
  1. proposer 推断：同一 (diff_file, new_line) 的评论属于同一讨论线程，
     线程中最早评论者 = proposer（提出人）
  2. commit 定位：每条评论按 created_at 时间，用 git log --before 找到
     评论发生时的代码状态（修改前的代码），确保看到的是有问题代码
  3. 过滤不完整评论：无位置信息、无代码片段、过短、回复、低价值等

用法:
    python3 scripts/gen_proposer_sample.py --repo ops-cv --limit 3
"""

import logging
import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

# ── 常量 ──────────────────────────────────────────────────────────────────

MIN_BODY_LEN = 10

REPLY_PATTERNS = [
    '已修正', '已修改', '已修复', '已解决', '已删除', '已添加', '已更新',
    '已改', '已优化', '已调整', '已补齐', '已替换', '已重构',
    '已核对', '已确认', '已验证', '已检查', '已同步', '已排查',
    '已测试', '已编译', '已构建', '已运行',
    '感谢指正', '感谢提醒', '好的，', '收到，',
    '已 merge', '已merge', '已合并', '已提交', '已推送',
]
REPLY_EXACT = {'已修正', '已修改', '已修复', '已解决', '收到', '好的',
               '了解', '同意', '好的，已修改'}

LOW_VALUE_EXACT = {'lgtm', '+1', 'good', 'nice', 'ok', '收到', '了解',
                   '同意', '赞同', '👍', '/lgtm', '/approve', '/check-cla'}


def normalize_time(t: str) -> str:
    """统一时间格式为 'YYYY-MM-DD HH:MM:SS'，用于字符串比较

    git %ci 格式:  '2026-03-17 21:06:55 +0800'
    created_at 格式: '2026-03-17T14:35:23.123+08:00'
    统一截取前19字符并将 T 替换为空格
    """
    t = t.strip().replace('T', ' ')
    return t[:19]

REPO_GITCODE_URL = {
    'ops-transformer': 'https://gitcode.com/cann/ops-transformer.git',
    'ops-nn': 'https://gitcode.com/cann/ops-nn.git',
    'ops-cv': 'https://gitcode.com/cann/ops-cv.git',
    'ops-math': 'https://gitcode.com/cann/ops-math.git',
}


# ── Bare repo 管理 ────────────────────────────────────────────────────────

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
    logging.info(f'  clone {repo} -> {bare_dir} ...')
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


# ── 获取 PR 作者 ──────────────────────────────────────────────────────────

_pr_author_cache: Dict[Tuple[str, int], str] = {}


def get_pr_author(repo: str, pr_number: int, base_dir: Path) -> str:
    """获取 PR 作者（代码提交者），通过 PR 分支最新 commit 的 author

    返回 author name（可能和 GitCode username 不一致）。
    调用方应同时用 name 和 email local-part 做匹配。
    """
    cache_key = (repo, pr_number)
    if cache_key in _pr_author_cache:
        return _pr_author_cache[cache_key]

    bare_dir = get_bare_dir(repo, base_dir)
    if not bare_dir.exists():
        return ''

    ref = f'pr_{pr_number}'
    r = subprocess.run(
        ['git', 'log', '-1', '--format=%an|%ae', ref],
        cwd=bare_dir, capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        _pr_author_cache[cache_key] = ''
        return ''

    parts = r.stdout.strip().split('|', 1)
    name = parts[0].strip()
    email_local = ''
    if len(parts) == 2 and '@' in parts[1]:
        email_local = parts[1].strip().split('@')[0]

    # 返回 name 和 email local part，用 | 分隔
    result = f'{name}|{email_local}' if email_local else name
    _pr_author_cache[cache_key] = result
    return result


# ── 按评论时间定位 commit（修改前的代码） ────────────────────────────────

_commit_cache: Dict[Tuple[str, int, str, str], str] = {}


def get_commit_for_comment(repo: str, pr_number: int, created_at: str,
                           diff_file: str, base_dir: Path) -> str:
    """找到评论发生时的代码 commit（确保看到修改前/有问题的代码）

    核心问题：GitCode PR 的 commit 可能被 amend，amend 后 commit_date 更新为
    amend 时间，但代码内容是修复后的。必须用 commit_date（%ci）而非
    author_date（%ai）来判断，因为 commit_date 反映代码实际写入仓库的时间。

    策略：
    1. 列出所有修改该文件的 commit（含 commit_date %ci）
    2. 找到 commit_date <= 评论时间的最后一个 commit（评论者当时看到的代码）
    3. 如果所有 commit 的 commit_date 都晚于评论时间（被 amend 过），
       返回第一个 commit，由调用方通过 commit_date 检查过滤为 post_fix
    """
    cache_key = (repo, pr_number, created_at[:19], diff_file)
    if cache_key in _commit_cache:
        return _commit_cache[cache_key]

    bare_dir = get_bare_dir(repo, base_dir)
    if not bare_dir.exists():
        return ''

    ref = f'pr_{pr_number}'
    check = subprocess.run(
        ['git', 'rev-parse', '--verify', ref],
        cwd=bare_dir, capture_output=True, text=True,
    )
    if check.returncode != 0:
        _commit_cache[cache_key] = ''
        return ''

    # 列出所有修改该文件的 commit，按时间正序（旧→新），用 commit_date
    r = subprocess.run(
        ['git', 'log', '--reverse', '--format=%H|%ci', ref, '--', diff_file],
        cwd=bare_dir, capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        _commit_cache[cache_key] = ''
        return ''

    commits = []
    for line in r.stdout.strip().split('\n'):
        parts = line.split('|', 1)
        if len(parts) == 2:
            commits.append((parts[0].strip()[:12], parts[1].strip()))

    if not commits:
        _commit_cache[cache_key] = ''
        return ''

    # 找到 commit_date <= 评论时间的最后一个 commit
    result = ''
    norm_created = normalize_time(created_at)
    for commit_hash, commit_date in commits:
        if normalize_time(commit_date) <= norm_created:
            result = commit_hash
        else:
            break

    if result:
        _commit_cache[cache_key] = result
        return result

    # 所有 commit 的 commit_date 都晚于评论时间 → 被 amend 过
    # 返回第一个 commit，由 filter_comments 的 post_fix 检查过滤
    first_commit = commits[0][0]
    _commit_cache[cache_key] = first_commit
    return first_commit


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
                         base_dir: Path, total_lines: int = 20) -> str:
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


# ── 工具函数 ──────────────────────────────────────────────────────────────

def parse_line_value(line_val) -> Tuple[int, int]:
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
    fp = file_path.replace('\\', '/')
    if '/op_kernel/' in fp:
        return 'Kernel侧'
    if '/op_host/' in fp:
        return 'Tiling侧'
    if '/op_graph/' in fp:
        return 'Host侧'
    if '/op_tiling/' in fp:
        return 'Tiling侧'
    if '/op_api/' in fp:
        return 'Host侧'
    if '/tests/' in fp:
        return '测试侧'
    return '通用'


def load_refined(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ── 推断提出人 ────────────────────────────────────────────────────────────

def infer_proposers(comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """同一 (diff_file, new_line) 的评论属于同一讨论线程，
    线程中最早评论者 = proposer。"""
    threads = defaultdict(list)
    for c in comments:
        key = (c.get('diff_file', ''), c.get('new_line', 0))
        threads[key].append(c)

    thread_proposer = {}
    for key, thread_comments in threads.items():
        sorted_comments = sorted(
            thread_comments,
            key=lambda c: c.get('created_at', '')
        )
        if sorted_comments:
            thread_proposer[key] = sorted_comments[0].get('author', '')

    result = []
    for c in comments:
        key = (c.get('diff_file', ''), c.get('new_line', 0))
        c = dict(c)
        c['proposer'] = thread_proposer.get(key, c.get('author', ''))
        result.append(c)
    return result


# ── 评论过滤 ──────────────────────────────────────────────────────────────

def is_reply(body: str) -> bool:
    stripped = body.strip()
    if stripped in REPLY_EXACT:
        return True
    for pat in REPLY_PATTERNS:
        if stripped.startswith(pat):
            if len(stripped) < 30:
                return True
            rest = stripped[len(pat):]
            if not rest or rest[0] in ':：,，。.、\n ;；':
                return True
    if (stripped.startswith('@') or stripped.startswith('[@')) and len(stripped) < 80:
        return True
    if stripped.startswith('>') and '@' in stripped:
        return True
    return False


def is_low_value(body: str) -> bool:
    stripped = body.strip().lower()
    if stripped in LOW_VALUE_EXACT:
        return True
    if stripped.startswith('/lgtm') or stripped.startswith('/approve'):
        return True
    if stripped.startswith('/check-cla') or stripped.startswith('/modify'):
        return True
    if stripped.startswith('compile') and len(stripped) < 30:
        return True
    return False


def is_bot_content(body: str) -> bool:
    lower = body.lower()
    if '<table' in body and '<tr>' in body:
        return True
    ci_keywords = ['compile_ascend', 'check_pr', 'pre_compile',
                   'pipelinedetail', 'pipelinerunid']
    ci_count = sum(1 for kw in ci_keywords if kw in lower)
    if ci_count >= 2:
        return True
    if ('success' in lower or 'failed' in lower) and (
        'compile_ascend' in lower or 'check_pr' in lower
    ):
        return True
    urls = re.findall(r'https?://\S+', body)
    if len(urls) > 5:
        return True
    return False


def filter_comments(comments: List[Dict[str, Any]], repo: str,
                    pr_number: int, pr_author: str, base_dir: Path
                    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """过滤不完整的评论，并为每条评论定位正确的 commit

    过滤规则：
    1. 无 diff_file 或 new_line → no_position
    2. body 过短（<10字符）→ too_short
    3. 低价值内容（lgtm/approve/compile等）→ low_value
    4. 回复类评论（已修改/已修复等）→ reply
    5. bot/CI 内容 → bot_content
    6. proposer == PR作者 → author_is_proposer（PR作者自己提的不是检视意见）
    7. 无法定位 commit → no_commit
    8. commit_date 晚于评论时间 → post_fix_commit（代码是修复后的）
    9. 无法提取代码片段 → no_snippet
    """
    stats = defaultdict(int)
    valid = []

    for c in comments:
        body = (c.get('body') or '').strip()

        # 1. 位置信息
        if not c.get('diff_file') or not c.get('new_line'):
            stats['no_position'] += 1
            continue

        # 2. body 长度
        if len(body) < MIN_BODY_LEN:
            stats['too_short'] += 1
            continue

        # 3. 低价值
        if is_low_value(body):
            stats['low_value'] += 1
            continue

        # 4. 回复
        if is_reply(body):
            stats['reply'] += 1
            continue

        # 5. bot 内容
        if is_bot_content(body):
            stats['bot_content'] += 1
            continue

        # 6. proposer == PR作者 → 扔掉（PR作者自己提的不是检视意见）
        proposer = c.get('proposer', '')
        if pr_author:
            # pr_author 格式: "name|email_local" 或 "name"
            parts = pr_author.split('|')
            author_names = set(p.strip().lower() for p in parts if p.strip())
            if proposer.lower() in author_names:
                stats['author_is_proposer'] += 1
                continue

        # 7. 定位 commit（评论时间前的代码状态）
        commit = get_commit_for_comment(
            repo, pr_number,
            c.get('created_at', ''),
            c.get('diff_file', ''),
            base_dir,
        )
        if not commit:
            stats['no_commit'] += 1
            continue
        c['resolved_commit'] = commit

        # 8. 检查 commit 的 commit_date 是否晚于评论时间
        #    如果是，说明 commit 被 amend 过，代码是修复后的，无法作为 ground truth
        bare_dir = get_bare_dir(repo, base_dir)
        r = subprocess.run(
            ['git', 'log', '-1', '--format=%ci', commit],
            cwd=bare_dir, capture_output=True, text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            if normalize_time(r.stdout.strip()) > normalize_time(c.get('created_at', '')):
                stats['post_fix_commit'] += 1
                continue

        # 9. 代码片段可提取
        line_start, _ = parse_line_value(c.get('new_line'))
        snippet = extract_code_snippet(
            repo, commit, c.get('diff_file', ''),
            line_start, line_start, base_dir,
        )
        if not snippet:
            stats['no_snippet'] += 1
            continue

        stats['valid'] += 1
        valid.append(c)

    return valid, dict(stats)


# ── Markdown 报告生成 ────────────────────────────────────────────────────

def generate_report(repo: str, pr_number: int, pr_author: str,
                    comments: List[Dict[str, Any]],
                    base_dir: Path) -> str:
    # pr_author 格式: "name|email_local" 或 "name"，取 name 显示
    author_display = pr_author.split('|')[0].strip() if pr_author else 'unknown'
    files = sorted(set(
        c.get('diff_file', '') for c in comments if c.get('diff_file')
    ))
    primary_file = files[0] if files else ''
    side = guess_side(primary_file)
    now = datetime.now().strftime('%Y-%m-%d')

    by_file = defaultdict(list)
    for c in comments:
        by_file[c.get('diff_file', '')].append(c)

    lines = []
    lines.append('# 代码检视报告')
    lines.append('')
    lines.append('## 检视概览')
    lines.append(f'- **仓库**: {repo}')
    lines.append(f'- **PR编号**: {pr_number}')
    lines.append(f'- **PR作者**: {author_display}')
    lines.append(f'- **代码文件**: '
                 f'{", ".join(files) if len(files) <= 3 else f"{len(files)} 个文件"}')
    lines.append(f'- **代码侧别**: {side}')
    lines.append('- **检视来源**: 人工检视评论（GitCode PR）')
    lines.append(f'- **总评论数**: {len(comments)} 条')
    lines.append(f'- **检视时间**: {now}')
    lines.append('')

    lines.append('## 检视统计')
    lines.append('')
    lines.append('| 状态 | 条数 | 占比 |')
    lines.append('|-----|------|------|')
    lines.append(f'| FAIL（发现问题） | {len(comments)} | 100% |')
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
            proposer = c.get('proposer', 'unknown')
            commit = c.get('resolved_commit', '')
            line_start, line_end = parse_line_value(c.get('new_line'))
            body = c.get('body', '').strip()
            if not body:
                continue
            idx += 1
            created = c.get('created_at', '')[:10]
            line_range = (str(line_start) if line_start == line_end
                          else f'{line_start}-{line_end}')

            lines.append(f'#### [{idx}] 人工检视意见')
            lines.append('')
            lines.append(f'- **提出人**: {proposer}')
            lines.append(f'- **作者**: {author_display}')
            lines.append(f'- **文件**: {file_path}')
            lines.append(f'- **行号**: {line_range}')
            lines.append(f'- **评论时间**: {created}')
            lines.append(f'- **Commit**: {commit}')
            lines.append('- **问题描述**:')
            lines.append('')
            for body_line in body.split('\n'):
                lines.append(f'  > {body_line}')
            lines.append('')

            snippet = extract_code_snippet(
                repo, commit, file_path, line_start, line_end, base_dir,
                total_lines=20,
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
    lines.append(f'> 本报告基于 PR {pr_number} 的人工检视评论生成'
                 f'（已过滤 PR 作者自己的评论、回复及修复后的 commit）')
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

def generate_reports(
    repo: str = 'ops-cv',
    limit: int = 3,
    min_comments: int = 3,
    base_dir: str | Path | None = None,
    refined_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """从 refined 数据生成带提出人的样例报告（可被外部调用）

    Args:
        repo:         仓库名称 (ops-transformer / ops-nn / ops-cv / ops-math)
        limit:        生成报告数量
        min_comments: 最小有效评论数
        base_dir:     项目根目录（默认: 本文件的上级目录）
        refined_dir:  refined 数据目录（默认: {base_dir}/history_real_comment/refined）
        output_dir:   报告输出目录（默认: {base_dir}/benchmark_tasks/{repo}-proposer-sample）

    Returns:
        dict:
          output_dir  — 输出目录 (Path)
          files       — 生成的报告文件路径列表 (list[Path])
          filter_stats — 过滤统计 (dict[str, int])
    """
    # 清空调用间可能残留的缓存
    _pr_author_cache.clear()
    _commit_cache.clear()
    _file_cache.clear()

    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    base_dir = Path(base_dir).resolve()

    if refined_dir is None:
        refined_path = base_dir / 'history_real_comment' / 'refined' / f'{repo}.jsonl'
    else:
        refined_path = Path(refined_dir) / f'{repo}.jsonl'

    if output_dir is None:
        out_dir = base_dir / 'benchmark_tasks' / f'{repo}-proposer-sample'
    else:
        out_dir = Path(output_dir)

    if not refined_path.exists():
        logging.error(f'错误: {refined_path} 不存在')
        return {'output_dir': out_dir, 'files': [], 'filter_stats': {}}

    groups = load_refined(refined_path)
    logging.info(f'{repo}: {len(groups)} 组 refined 数据')

    bare_dir = ensure_bare_repo(repo, base_dir)
    if not bare_dir:
        logging.error('错误: bare repo 不可用，无法提取代码片段')
        return {'output_dir': out_dir, 'files': [], 'filter_stats': {}}
    logging.info(f'bare repo: {bare_dir}')

    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []
    generated = 0
    total_stats: dict[str, int] = defaultdict(int)
    fetched_prs: set[int] = set()

    for g in groups:
        if generated >= limit:
            break

        comments = g.get('comments', [])
        if len(comments) < min_comments:
            continue

        pr_number = g.get('pr_number', 0)
        group_commit = g.get('commit_id', '')[:12]

        # fetch PR ref
        if pr_number not in fetched_prs:
            ensure_pr_ref(repo, pr_number, base_dir)
            fetched_prs.add(pr_number)

        # 获取 PR 作者
        pr_author = get_pr_author(repo, pr_number, base_dir)

        # 推断提出人
        annotated = infer_proposers(comments)

        # 过滤 + 定位 commit（proposer == PR作者的扔掉）
        _file_cache.clear()
        valid, stats = filter_comments(
            annotated, repo, pr_number, pr_author, base_dir
        )
        for k, v in stats.items():
            total_stats[k] += v

        if len(valid) < min_comments:
            total_stats['skip_low_count'] += 1
            continue

        report = generate_report(repo, pr_number, pr_author, valid, base_dir)
        filename = f'{repo}_pr_{pr_number:04d}_{group_commit}.md'
        filepath = out_dir / filename
        filepath.write_text(report, encoding='utf-8')
        generated_files.append(filepath)
        generated += 1

        proposer_count = len(set(c.get('proposer', '') for c in valid))
        commits = set(c.get('resolved_commit', '') for c in valid)
        author_display = pr_author.split('|')[0].strip() if pr_author else '?'
        logging.info(f'  [{generated}] PR {pr_number} author={author_display}: '
              f'{len(valid)} 条有效评论 (原 {len(comments)}), '
              f'{proposer_count} 提出人, '
              f'{len(commits)} commits '
              f'-> {filepath.name}')

    logging.info(f'\n生成 {generated} 份报告 -> {out_dir}')
    logging.info(f'过滤统计: {dict(total_stats)}')

    return {
        'output_dir': out_dir,
        'files': generated_files,
        'filter_stats': dict(total_stats),
    }


def main():
    parser = argparse.ArgumentParser(
        description='从 refined 数据生成带提出人的样例报告'
    )
    parser.add_argument('--repo', default='ops-cv',
                        choices=['ops-transformer', 'ops-nn', 'ops-cv', 'ops-math'])
    parser.add_argument('--limit', type=int, default=3,
                        help='生成报告数量（默认3）')
    parser.add_argument('--min-comments', type=int, default=3,
                        help='最小有效评论数（默认3）')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    result = generate_reports(
        repo=args.repo,
        limit=args.limit,
        min_comments=args.min_comments,
        base_dir=base_dir,
    )
    sys.exit(0 if result['files'] else 1)


if __name__ == '__main__':
    main()
