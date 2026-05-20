#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""克隆 GitCode PR 完整源码到本地（无论 PR 是否合入）

用法:
    python clone_pr_source.py --repo https://gitcode.com/cann/ops-transformer --pr 4356 --clone-dir ./pr_source/4356

与 get_gitcode_pr_diff.py 互补：
    - get_gitcode_pr_diff.py → 获取 diff 文件
    - clone_pr_source.py      → 获取完整源码（用于子 Agent 追溯变量定义/初始化/上游校验）
"""

import argparse
import logging
import os
import re
import subprocess
import sys

ALLOWED_GITCODE_DOMAIN = "gitcode.com"

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr
)
logger = logging.getLogger(__name__)


def parse_repo_url(url: str) -> tuple[str, str, str]:
    """解析仓库链接，返回 (owner, repo, .git URL)"""
    if not url.startswith(f"https://{ALLOWED_GITCODE_DOMAIN}/"):
        raise ValueError(f"只支持 {ALLOWED_GITCODE_DOMAIN} 仓库，当前 URL: {url}")

    url = url.rstrip("/")
    url = re.sub(r"/pulls/\d+$", "", url)
    url = re.sub(r"\.git$", "", url)

    match = re.search(r"gitcode\.com/([^/]+)/([^/]+)", url)
    if not match:
        raise ValueError(f"无法解析 owner/repo: {url}")

    return match.group(1), match.group(2), f"{url}.git"


def run_git(cmd: list[str], cwd: str | None = None) -> bool:
    """执行 git 命令，成功返回 True，失败打印日志返回 False"""
    try:
        subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.debug("命令失败: %s — %s", " ".join(cmd), e.stderr.strip())
        return False


def clone_repo(repo_url: str, clone_dir: str) -> None:
    """浅克隆仓库到指定目录"""
    logger.info("克隆 %s 到 %s ...", repo_url, clone_dir)
    if not run_git(["git", "clone", "--depth=1", repo_url, clone_dir]):
        raise RuntimeError("克隆仓库失败")


def fetch_pr_ref(pr_number: int, clone_dir: str) -> None:
    """获取 PR 代码：优先 merge ref（已合入），回退 head ref（未合入）"""
    merge_ref = f"refs/merge-requests/{pr_number}/merge"
    head_ref = f"refs/merge-requests/{pr_number}/head"

    logger.info("获取 PR #%d 代码...", pr_number)

    if run_git(["git", "fetch", "origin", f"{merge_ref}:pr_ref"], cwd=clone_dir):
        run_git(["git", "checkout", "pr_ref"], cwd=clone_dir)
        logger.info("已 checkout PR #%d merge 提交", pr_number)
        return

    logger.info("merge ref 不存在，尝试 head ref...")
    if run_git(["git", "fetch", "origin", f"{head_ref}:pr_ref"], cwd=clone_dir):
        run_git(["git", "checkout", "pr_ref"], cwd=clone_dir)
        logger.info("已 checkout PR #%d head 提交", pr_number)
        return

    raise RuntimeError(f"无法获取 PR #{pr_number} 代码（merge 和 head ref 均不存在）")


def main():
    parser = argparse.ArgumentParser(description="克隆 GitCode PR 完整源码")
    parser.add_argument("--repo", required=True, help="仓库链接")
    parser.add_argument("--pr", required=True, type=int, help="PR 编号")
    parser.add_argument("--clone-dir", required=True, help="克隆目标目录")
    args = parser.parse_args()

    if os.path.exists(args.clone_dir):
        logger.info("目录已存在，跳过克隆: %s", args.clone_dir)
        return

    try:
        _, _, repo_url = parse_repo_url(args.repo)
        clone_repo(repo_url, args.clone_dir)
        fetch_pr_ref(args.pr, args.clone_dir)
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
