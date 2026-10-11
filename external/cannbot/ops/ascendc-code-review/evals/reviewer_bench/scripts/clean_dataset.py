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
LLM 驱动的数据清洗脚本

从 fetch_review_comments.py 提取的行内评论中，通过 DeepSeek API 语义分析清洗数据。
保留 keyword 匹配作为 LLM 调用失败的 fallback。

输入：PR 级 JSONL 文件（每行一个 PR，含 comments 数组）
输出：扁平化 JSONL 文件，每条评论增加 quality 和 reason 字段

使用方式：
    python3 scripts/clean_dataset.py
    python3 scripts/clean_dataset.py --input history_real_comment/raw/ops-transformer.jsonl \
                                     --output history_real_comment/clean/ops-transformer.jsonl
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)


# ── Load .env ───────────────────────────────────────────────────────────

def load_dotenv(env_path: Path = None):
    """从 .env 文件加载环境变量（不覆盖已存在的）"""
    if env_path is None:
        # 从脚本位置向上查找 .env
        script_dir = Path(__file__).resolve().parent
        for parent in [script_dir, script_dir.parent]:
            candidate = parent / '.env'
            if candidate.exists():
                env_path = candidate
                break
    if env_path and env_path.exists():
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()


# ── 提示词加载 ──────────────────────────────────────────────────────────

def load_prompt_template(prompt_file: str = None) -> str:
    """从 prompts/ 目录加载提示词模板

    优先使用 prompt_file 参数指定的路径，其次从项目根目录的
    prompts/clean_dataset_prompt.md 读取。
    """
    base_dir = Path(__file__).resolve().parent.parent

    if prompt_file:
        p = Path(prompt_file)
        if not p.is_absolute():
            p = base_dir / p
    else:
        p = base_dir / 'prompts' / 'clean_dataset_prompt.md'

    if not p.exists():
        logging.error(f"错误：提示词文件不存在: {p}")
        sys.exit(1)

    return p.read_text(encoding='utf-8').strip()


def fill_prompt(template: str, comment: Dict[str, Any]) -> str:
    """用评论数据填充提示词模板中的占位符"""
    return template \
        .replace("{pr_number}", str(comment.get('pr_number', ''))) \
        .replace("{author}", str(comment.get('author', ''))) \
        .replace("{file_path}", str(comment.get('file_path', '未知'))) \
        .replace("{line}", str(comment.get('line', '未知'))) \
        .replace("{severity}", str(comment.get('severity', ''))) \
        .replace("{category}", str(comment.get('category', ''))) \
        .replace("{resolved}", str(comment.get('resolved', False))) \
        .replace("{body}", str(comment.get('body', '')))


# ── Keyword Fallback ────────────────────────────────────────────────────

LOW_VALUE_KEYWORDS = ['lgtm', '+1', '👍', 'good', 'nice', 'ok', '收到', '了解']


def keyword_analyze(comment: Dict[str, Any]) -> Dict[str, str]:
    """Keyword-based fallback for quality analysis."""
    body = (comment.get('body') or '').lower()

    # 评论太短且无实质内容
    if len(body.strip()) < 5:
        return {"quality": "rejected", "reason": "评论内容过短"}

    # 纯低价值关键词
    if body.strip() in [kw.lower() for kw in LOW_VALUE_KEYWORDS]:
        return {"quality": "rejected", "reason": "无实质技术内容"}

    is_resolved = comment.get('resolved', False)

    if is_resolved:
        return {"quality": "accepted", "reason": "已闭环修复"}

    # 有实质性技术内容的长评论默认 accepted
    if len(body) > 50:
        return {"quality": "accepted", "reason": "具有实质性技术内容"}

    return {"quality": "pending", "reason": "无法通过关键词判断"}


# ── LLM Analysis ────────────────────────────────────────────────────────

def create_client(api_key: str, base_url: str = "https://api.deepseek.com") -> OpenAI:
    """创建 OpenAI 兼容客户端"""
    return OpenAI(api_key=api_key, base_url=base_url)


def analyze_comment(client: OpenAI, model: str, comment: Dict[str, Any],
                    prompt_template: str, max_retries: int = 3) -> Dict[str, str]:
    """
    通过 LLM 分析单条评论质量。

    Returns:
        {"quality": "accepted|rejected|pending", "reason": "..."}
    失败时返回 None，由调用方决定 fallback。
    """
    prompt = fill_prompt(prompt_template, comment)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            content = resp.choices[0].message.content.strip()
            # 提取 JSON（兼容 markdown code block 包裹）
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(content)
            if result.get("quality") in ("accepted", "rejected", "pending"):
                return result
            logging.warning(f"  警告: LLM 返回无效 quality 值: {result}")
        except json.JSONDecodeError:
            logging.warning(f"  警告: LLM 返回非 JSON (attempt {attempt+1}): {content[:100]}")
        except Exception as e:
            logging.warning(f"  警告: API 调用失败 (attempt {attempt+1}): {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # exponential backoff

    return None


# ── Data Flattening ─────────────────────────────────────────────────────

# 仓库 → bare repo 路径映射
REPO_BARE_DIRS = {
    'ops-transformer': '/tmp/pr_recover_test/repo.git',
    'ops-nn': '/tmp/ops_nn_repo.git',
    'ops-cv': '/tmp/ops_cv_repo.git',
    'ops-math': '/tmp/ops_math_repo.git',
}


def get_commit_id(repo: str, pr_number: int, comment_time: str, diff_file: str,
                  commit_cache: dict) -> str:
    """通过评论时间 + 文件路径定位对应的 commit hash

    优先使用 git log --before=<评论时间> <ref> -- <diff_file> 查找包含
    目标文件的最近 commit。如果文件在评论时间前不存在（RC2：文件仅存在于
    diff 中，尚未 commit），回退到原始算法（不限文件路径）。

    结果缓存在 commit_cache 中避免重复 git 调用。
    """
    cache_key = (repo, pr_number, comment_time[:19], diff_file)
    if cache_key in commit_cache:
        return commit_cache[cache_key]

    bare_dir = REPO_BARE_DIRS.get(repo)
    if not bare_dir:
        return ''

    import subprocess
    from pathlib import Path
    if not Path(bare_dir).exists():
        return ''

    ref = f"pr_{pr_number}"
    # 先检查 ref 是否存在
    check = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=bare_dir, capture_output=True, text=True
    )
    if check.returncode != 0:
        # ref 不存在，尝试 fetch
        subprocess.run(
            ["git", "fetch", "origin", f"refs/merge-requests/{pr_number}/head:{ref}"],
            cwd=bare_dir, capture_output=True, text=True, timeout=30
        )

    def _run_log(args: list) -> str:
        r = subprocess.run(args, cwd=bare_dir, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()[:12]
        return ''

    # 策略 1: 查找评论时间之前、包含目标文件的最近 commit（根因 1 修复）
    commit_id = _run_log(
        ["git", "log", f"--before={comment_time}", "--format=%H", "-1", ref, "--", diff_file]
    )
    if commit_id:
        commit_cache[cache_key] = commit_id
        return commit_id

    # 策略 2: 文件在评论时间之前不存在于任何 commit（RC2）
    # 查找 PR 分支上首次引入该文件的 commit
    commit_id = _run_log(
        ["git", "log", "--reverse", "--format=%H", "-1", ref, "--", diff_file]
    )
    if commit_id:
        commit_cache[cache_key] = commit_id
        return commit_id

    # 策略 3: 兜底 — 原始算法（不限文件路径）
    commit_id = _run_log(
        ["git", "log", f"--before={comment_time}", "--format=%H", "-1", ref]
    )
    commit_cache[cache_key] = commit_id
    return commit_id


def flatten_pr_record(record: Dict[str, Any]) -> list:
    """将 PR 级记录扁平化为独立评论列表"""
    pr_number = record.get('pr_number', '')
    repo = record.get('repo', '')
    comments = []
    for c in record.get('comments', []):
        comments.append({
            'pr_number': pr_number,
            'repo': repo,
            'comment_id': c.get('id'),
            'author': c.get('author', ''),
            'body': c.get('body', ''),
            'resolved': c.get('resolved', False),
            'created_at': c.get('created_at', ''),
            'severity': c.get('severity', ''),
            'category': c.get('category', ''),
            # 行内评论位置信息
            'diff_file': c.get('diff_file', ''),
            'new_line': c.get('new_line'),
            'line': c.get('line'),
            'file_path': c.get('file_path', ''),
            'new_path': c.get('new_path', ''),
            'old_path': c.get('old_path', ''),
            'is_expired_suggestion': c.get('is_expired_suggestion'),
        })
    return comments


# ── Main Pipeline ────────────────────────────────────────────────────────

def clean_dataset(input_path: Path, output_path: Path, api_key: str, model: str,
                  prompt_template: str):
    """主流程：读取 → LLM 分析 → 计算 commit_id → 写入"""
    client = create_client(api_key)
    commit_cache = {}  # (repo, pr_number, time) -> commit_id 缓存

    stats = {'total': 0, 'accepted': 0, 'rejected': 0, 'pending': 0,
             'llm_success': 0, 'fallback': 0, 'has_commit': 0}

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue

            for comment in flatten_pr_record(record):
                stats['total'] += 1

                # 计算 commit_id
                commit_id = get_commit_id(
                    comment.get('repo', ''),
                    comment.get('pr_number', 0),
                    comment.get('created_at', ''),
                    comment.get('diff_file', ''),
                    commit_cache
                )
                comment['commit_id'] = commit_id
                if commit_id:
                    stats['has_commit'] += 1

                # 尝试 LLM 分析
                result = analyze_comment(client, model, comment, prompt_template)

                if result is not None:
                    stats['llm_success'] += 1
                    source = 'llm'
                else:
                    # LLM 失败，使用 keyword fallback
                    result = keyword_analyze(comment)
                    stats['fallback'] += 1
                    source = 'keyword'

                quality = result['quality']
                stats[quality] += 1

                # rejected 的评论不写入输出
                if quality == 'rejected':
                    continue

                # 无位置信息（diff_file 或 new_line 为空）的评论不写入
                if not comment.get('diff_file') or comment.get('new_line') is None:
                    stats['no_position'] = stats.get('no_position', 0) + 1
                    continue

                # 写入时附加清洗结果
                comment['quality'] = quality
                comment['reason'] = result.get('reason', '')
                comment['analysis_source'] = source

                # 重排字段顺序：repo → pr_number → commit_id → ...，删除 comment_id
                ordered = {
                    'repo': comment.get('repo', ''),
                    'pr_number': comment['pr_number'],
                    'commit_id': comment.get('commit_id', ''),
                    'author': comment.get('author', ''),
                    'created_at': comment.get('created_at', ''),
                    'diff_file': comment.get('diff_file', ''),
                    'new_line': comment.get('new_line'),
                    'body': comment.get('body', ''),
                    'quality': comment['quality'],
                    'reason': comment['reason'],
                    'analysis_source': comment['analysis_source'],
                }
                fout.write(json.dumps(ordered, ensure_ascii=False) + '\n')

                # 进度显示
                if stats['total'] % 10 == 0:
                    logging.info(f"已处理 {stats['total']} 条 "
                          f"(accepted={stats['accepted']}, "
                          f"rejected={stats['rejected']}, "
                          f"pending={stats['pending']}, "
                          f"commit={stats['has_commit']})")

                # 速率限制
                time.sleep(0.5)

    return stats


# ── Commit ID 重算 ──────────────────────────────────────────────────────

def recalc_commit_ids(input_path: Path, output_path: Path) -> dict:
    """对已清洗数据重算 commit_id（不需要 LLM，只修正 git log 算法）

    读取 clean JSONL，用修复后的 get_commit_id() 重新计算每条记录的
    commit_id，写入输出。
    """
    commit_cache = {}
    stats = {'total': 0, 'changed': 0, 'new_has_commit': 0, 'old_has_commit': 0,
             'strategy1': 0, 'strategy2': 0, 'strategy3': 0}

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            stats['total'] += 1

            old_commit = record.get('commit_id', '')
            if old_commit:
                stats['old_has_commit'] += 1

            # 用修复后的算法重算 commit_id
            new_commit = get_commit_id(
                record.get('repo', ''),
                record.get('pr_number', 0),
                record.get('created_at', ''),
                record.get('diff_file', ''),
                commit_cache
            )

            if new_commit:
                stats['new_has_commit'] += 1
            if new_commit != old_commit:
                stats['changed'] += 1

            record['commit_id'] = new_commit
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')

            if stats['total'] % 200 == 0:
                logging.info(f"已处理 {stats['total']} 条 (changed={stats['changed']})")

    return stats


def main():
    parser = argparse.ArgumentParser(description='LLM 驱动的数据清洗脚本')
    parser.add_argument('--input', type=Path,
                        default=Path('history_real_comment/raw/ops-transformer.jsonl'),
                        help='输入的 JSONL 文件')
    parser.add_argument('--output', type=Path,
                        default=Path('history_real_comment/clean/ops-transformer.jsonl'),
                        help='输出的清洗后 JSONL 文件')
    parser.add_argument('--api-key', type=str,
                        default=os.environ.get('DEEPSEEK_API_KEY', ''),
                        help='DeepSeek API key (默认读取 DEEPSEEK_API_KEY 环境变量)')
    parser.add_argument('--model', type=str, default='deepseek-chat',
                        help='模型名称 (默认 deepseek-chat)')
    parser.add_argument('--prompt-file', default='prompts/clean_dataset_prompt.md',
                        help='评论质量评估提示词文件（含 {pr_number} {author} 等占位符）')
    parser.add_argument('--recalc-commits', action='store_true',
                        help='仅重算已有 clean 数据的 commit_id（不需要 LLM）')

    args = parser.parse_args()

    if not args.input.exists():
        logging.error(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)

    if args.recalc_commits:
        # 仅重算 commit_id 模式
        logging.info(f"重算 commit_id: {args.input} -> {args.output}")
        stats = recalc_commit_ids(args.input, args.output)
        logging.info(f"\n=== 重算完成 ===")
        logging.info(f"总记录数: {stats['total']}")
        logging.info(f"  commit_id 变更: {stats['changed']} ({stats['changed']/max(stats['total'],1)*100:.1f}%)")
        logging.info(f"  旧有 commit: {stats['old_has_commit']}")
        logging.info(f"  新有 commit: {stats['new_has_commit']}")
        return

    if not args.api_key:
        logging.error("错误: 未提供 API key。请通过 --api-key 或 DEEPSEEK_API_KEY 环境变量设置")
        sys.exit(1)

    # 加载提示词模板
    prompt_template = load_prompt_template(args.prompt_file)

    logging.info(f"开始清洗数据集: {args.input}")
    logging.info(f"输出到: {args.output}")
    logging.info(f"模型: {args.model}")
    logging.info(f"提示词: {args.prompt_file}")

    stats = clean_dataset(args.input, args.output, args.api_key, args.model,
                          prompt_template)

    logging.info(f"\n=== 清洗完成 ===")
    logging.info(f"总记录数: {stats['total']}")
    logging.info(f"  ✓ accepted: {stats['accepted']}")
    logging.info(f"  ✗ rejected: {stats['rejected']}")
    logging.info(f"  ? pending:  {stats['pending']}")
    logging.info(f"  commit_id:  {stats['has_commit']}")
    logging.info(f"LLM 成功: {stats['llm_success']}, Fallback: {stats['fallback']}")


if __name__ == '__main__':
    main()