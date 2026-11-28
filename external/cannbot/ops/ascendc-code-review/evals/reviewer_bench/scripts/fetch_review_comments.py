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
"""批量获取 GitCode PR 检视评论

通过 GitCode API v1 获取指定仓库的全部 merge request discussions（评论），
筛选出人工检视评论，提取对应代码行信息。

支持仓库:
    - ops-transformer (默认)
    - ops-nn
    - ops-cv
    - ops-math

用法:
    python3 fetch_review_comments.py --pr 3604                          # 单个 PR (默认 ops-transformer)
    python3 fetch_review_comments.py --pr 100 --repo ops-nn             # 单个 PR (指定仓库)
    python3 fetch_review_comments.py --all --repo ops-cv                # 全部 PR
    python3 fetch_review_comments.py --start 1 --end 100 --repo ops-math

环境变量:
    GITCODE_TOKEN - GitCode Personal Access Token
"""

import logging
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
import ssl

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

# === 配置 ===
GITCODE_API = "https://gitcode.com/api/v1"

# 四大算子仓 project ID 映射
REPO_PROJECT_IDS = {
    "ops-transformer": 7673863,
    "ops-nn": 7665709,
    "ops-cv": 7657293,
    "ops-math": 7649531,
}

TOKEN = os.environ.get("GITCODE_TOKEN", "")

if not TOKEN:
    logging.error("错误: 请设置环境变量 GITCODE_TOKEN")
    sys.exit(1)

# 绕过 WAF 必需的完整浏览器 headers
HEADERS = {
    "PRIVATE-TOKEN": TOKEN,
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Origin": "https://gitcode.com",
    "Referer": "https://gitcode.com/cann/ops-transformer",
}

ctx = ssl.create_default_context()
total_requests = 0


def api_get(path, params=None, max_retries=4):
    """请求 GitCode API v1"""
    global total_requests
    url = f"{GITCODE_API}/{path}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        url += "?" + qs

    for attempt in range(max_retries):
        total_requests += 1
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                return json.loads(resp.read().decode()), resp.status
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                err = json.loads(body)
            except Exception:
                err = {"_raw": body[:300]}
            if e.code == 429:
                wait = min(60, 2 ** attempt)
                logging.info(f"  限流(429)，等待 {wait}s...")
                time.sleep(wait)
                continue
            if e.code == 418:
                logging.info(f"  WAF拦截(418)，等待 {5 * (attempt + 1)}s...")
                time.sleep(5 * (attempt + 1))
                continue
            if e.code == 404:
                return None, 404
            logging.error(f"  HTTP {e.code} (尝试 {attempt+1}): {err.get('error_message', str(err)[:100])}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return err, e.code
        except Exception as e:
            logging.error(f"  网络错误 (尝试 {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            return None, 0
    return None, 0


def get_all_discussions(mr_iid, project_id):
    """获取 MR 全部 discussions，提取嵌套评论"""
    all_review_notes = []
    end_id = 0
    page = 0

    while True:
        params = {"per_page": 100, "end_id": end_id}
        data, code = api_get(f"projects/{project_id}/merge_requests/{mr_iid}/discussions", params)
        if code != 200 or data is None:
            break

        items = data.get("content", {}).get("data", [])
        for item in items:
            # 跳过系统动作型记录（commit, label变更等）
            if item.get("action"):
                continue

            # 提取嵌套 notes 中的真实评论（保留回复链）
            nested = item.get("notes") or item.get("comments") or []
            for note in nested:
                body = (note.get("body") or "").strip()
                if not body:
                    continue

                # 取作者：嵌套 note 优先，其次外层
                author = note.get("author") or item.get("author") or {}

                all_review_notes.append({
                    "id": note.get("id", item.get("id")),
                    "body": body,
                    "author": author.get("username", ""),
                    "author_name_cn": author.get("name_cn", ""),
                    "created_at": note.get("created_at") or item.get("created_at", ""),
                    "severity": item.get("severity_cn", ""),
                    "category": item.get("review_categories_cn", ""),
                    "resolved": item.get("resolved", False),
                    # 行内评论位置信息（note 级别）
                    "diff_file": note.get("diff_file", ""),
                    "new_line": note.get("new_line"),
                    "line": note.get("line"),
                    "file_path": note.get("file_path", ""),
                    "new_path": note.get("new_path", ""),
                    "old_path": note.get("old_path", ""),
                    "is_expired_suggestion": note.get("is_expired_suggestion"),
                })

        page += 1
        new_end_id = data.get("content", {}).get("end_id", 0)
        if new_end_id <= end_id or len(items) < 100:
            break
        end_id = new_end_id

        time.sleep(0.25 if page % 3 else 0.5)

    return all_review_notes


def is_human_review(note):
    """判断是否是人工检视评论（note 已由 get_all_discussions 提取为 dict）

    过滤规则：
    1. 评论过短（<10字符）
    2. 机器人账号（后缀 -robot/-bot/-ci，或已知 bot 用户名）
    3. 自动化消息模式（流水线触发、标签变更等）
    4. CI/CD 执行结果（编译状态表格、流水线报告）
    5. HTML 表格消息（通常来自 bot）
    6. 纯链接消息（>5 个 URL）
    """
    import re

    body = note.get("body", "").strip()
    if len(body) < 10:
        return False

    author = note.get("author", "")

    # === 规则 2: 排除机器人账号 ===
    bot_suffixes = ("-robot", "-bot", "-ci")
    if author.lower().endswith(bot_suffixes):
        return False

    bot_authors = {"cann-robot", "ascend-robot", "gitlab-ci", "pipeline-bot", "ci-bot"}
    if author.lower() in bot_authors:
        return False

    # === 规则 3: 排除自动化消息模式 ===
    bot_patterns = ("流水线任务触发成功", "source code change are detected",
                    "add label", "delete label", "Add reviewers", "Add approvers",
                    "create merge request", "virtual merging success",
                    "changed this line on", "Thanks for your pull-request",
                    "代码屏蔽问题待您审核")
    for pattern in bot_patterns:
        if pattern in body:
            return False

    # === 规则 4: 排除 CI/CD 执行结果 ===
    ci_keywords = ["Compile_Ascend", "Check_Pr", "Pre_Compile", "pre_comment",
                   "pipelineDetail", "pipelineRunId", "pipeline 指导"]
    ci_count = sum(1 for kw in ci_keywords if kw in body)
    if ci_count >= 2:
        return False

    # 编译结果表格（包含 SUCCESS/FAILED + 任务名称）
    if ("SUCCESS" in body or "FAILED" in body) and ("Compile_Ascend" in body or "Check_Pr" in body):
        return False

    # === 规则 5: 排除 HTML 表格消息 ===
    if "<table" in body and "<tr>" in body:
        return False

    # === 规则 6: 排除纯链接消息（>5 个 URL） ===
    urls = re.findall(r'https?://\S+', body)
    if len(urls) > 5:
        return False

    return True


def format_note(note):
    """格式化一条评论"""
    username = note.get("author", "unknown")
    name_cn = note.get("author_name_cn", "")
    display_name = f"{username}({name_cn})" if name_cn else username

    body = (note.get("body") or "").strip()
    created = (note.get("created_at") or "")[:19]

    # 尝试从 body 中提取代码位置引用
    import re
    file_line_refs = re.findall(r'([\w./_-]+\.(?:cpp|h|hpp|asc|py|cmake|txt))[:：]\s*(\d+)', body)
    loc = ""
    if file_line_refs:
        loc = " | ".join(f"{f}:{l}" for f, l in file_line_refs[:3])

    return f"[{created}] {display_name}{' ['+loc+']' if loc else ''}: {body[:300]}"


def main():
    parser = argparse.ArgumentParser(description="批量获取 GitCode PR 检视评论")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pr", type=int, help="单个 PR 编号")
    group.add_argument("--all", action="store_true", help="全部 PR")
    group.add_argument("--start", type=int, help="起始 PR 编号（需配合 --end）")
    parser.add_argument("--end", type=int, help="结束 PR 编号")
    parser.add_argument("--repo", default="ops-transformer",
                        choices=list(REPO_PROJECT_IDS.keys()),
                        help="仓库名称（默认 ops-transformer）")
    parser.add_argument("--output", default=None,
                        help="JSONL 输出文件（默认 history_real_comment/raw/{repo}.jsonl）")
    parser.add_argument("--text-output", help="可读文本输出文件")
    parser.add_argument("--dry-run", action="store_true", help="仅统计不拉取")
    parser.add_argument("--delay", type=float, default=0.4, help="请求间隔秒数 (默认0.4)")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    args = parser.parse_args()

    project_id = REPO_PROJECT_IDS[args.repo]

    # 默认输出路径: history_real_comment/raw/{repo}.jsonl
    if args.output is None:
        args.output = f"history_real_comment/raw/{args.repo}.jsonl"

    if args.start is not None and args.end is None:
        parser.error("--start 需配合 --end")
    if args.end is not None and args.start is None:
        parser.error("--end 需配合 --start")

    # 确定 PR 范围
    if args.pr:
        pr_list = [args.pr]
    elif args.all:
        logging.info(f"获取 {args.repo} 最新 MR 编号...")
        data, code = api_get(
            f"projects/{project_id}/merge_requests",
            {"per_page": 1, "state": "all", "order_by": "created_at", "sort": "desc"}
        )
        if code == 200 and data and data.get("content"):
            latest = data["content"][0]["iid"]
            logging.info(f"最新 MR: !{latest}")
            pr_list = list(range(1, latest + 1))
        else:
            logging.error(f"获取失败 code={code}")
            sys.exit(1)
    else:
        pr_list = list(range(args.start, args.end + 1))

    total_prs = len(pr_list)
    logging.info(f"仓库: {args.repo} (project_id={project_id})")
    logging.info(f"待处理: {total_prs} 个 MR")
    logging.info(f"JSONL: {args.output}")
    if args.text_output:
        logging.info(f"文本:  {args.text_output}")
    logging.info(f"间隔:  {args.delay}s/请求")
    logging.info("")

    stats = {"scanned": 0, "with_reviews": 0, "total_comments": 0, "human_comments": 0, "skipped": 0}
    start_time = time.time()

    text_fp = open(args.text_output, "w", encoding="utf-8") if args.text_output else None

    try:
        with open(args.output, "w", encoding="utf-8") as out:
            for i, pr_num in enumerate(pr_list):
                stats["scanned"] += 1

                if i % 50 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    rate = stats["scanned"] / elapsed * 60 if elapsed > 0 else 0
                    logging.info(
                        f"  [{i}/{total_prs}] 扫描:{stats['scanned']} | "
                        f"有检视:{stats['with_reviews']} | "
                        f"评论:{stats['human_comments']} | "
                        f"跳:{stats['skipped']} | "
                        f"速率:{rate:.0f}/min | "
                        f"请求:{total_requests}")

                if args.dry_run:
                    time.sleep(0.1)
                    continue

                notes = get_all_discussions(pr_num, project_id)
                stats["total_comments"] += len(notes)

                human_notes = [n for n in notes if is_human_review(n)]
                if human_notes:
                    stats["with_reviews"] += 1
                    stats["human_comments"] += len(human_notes)

                    record = {
                        "pr_number": pr_num,
                        "repo": args.repo,
                        "human_comment_count": len(human_notes),
                        "total_discussion_count": len(notes),
                        "comments": [
                            {
                                "id": n.get("id"),
                                "author": n.get("author", ""),
                                "author_name_cn": n.get("author_name_cn", ""),
                                "body": n.get("body", ""),
                                "created_at": n.get("created_at", ""),
                                "severity": n.get("severity", ""),
                                "category": n.get("category", ""),
                                "resolved": n.get("resolved", False),
                                "diff_file": n.get("diff_file", ""),
                                "new_line": n.get("new_line"),
                                "line": n.get("line"),
                                "file_path": n.get("file_path", ""),
                                "new_path": n.get("new_path", ""),
                                "old_path": n.get("old_path", ""),
                                "is_expired_suggestion": n.get("is_expired_suggestion"),
                            }
                            for n in human_notes
                        ],
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()

                    if text_fp:
                        text_fp.write(f"\n{'=' * 80}\n")
                        text_fp.write(f"PR !{pr_num} ({args.repo})\n")
                        text_fp.write(f"人工评论: {len(human_notes)} / 总讨论: {len(notes)}\n")
                        text_fp.write(f"{'=' * 80}\n")
                        for n in human_notes:
                            text_fp.write(format_note(n) + "\n")
                        text_fp.flush()

                time.sleep(args.delay)

    except KeyboardInterrupt:
        logging.info(f"\n\n用户中断")
    finally:
        if text_fp:
            text_fp.close()

    elapsed = time.time() - start_time
    logging.info(f"\n{'=' * 60}")
    logging.info(f"仓库: {args.repo}")
    logging.info(f"完成! 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logging.info(f"扫描 MR: {stats['scanned']} | 跳过: {stats['skipped']}")
    logging.info(f"有检视评论的 MR: {stats['with_reviews']}")
    logging.info(f"总讨论数: {stats['total_comments']}")
    logging.info(f"人工检视评论: {stats['human_comments']}")
    logging.info(f"API 请求数: {total_requests}")
    logging.info(f"输出: {args.output}")
    if args.text_output:
        logging.info(f"文本:  {args.text_output}")


if __name__ == "__main__":
    main()
