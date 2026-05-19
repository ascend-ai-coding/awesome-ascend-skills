#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""获取 GitCode PR Diff（无 token）

用法:
    python get_gitcode_pr_diff.py --repo <url> --pr <num> [--output <file>] [--stat] [--file-filter <pattern>]

支持的仓库:
    - cann/ops-transformer
    - cann/ops-math
    - cann/ops-nn
    - cann/ops-cv
    - 其他 gitcode.com/cann/* 下的仓库

示例:
    # ops-transformer 仓库
    python get_gitcode_pr_diff.py --repo https://gitcode.com/cann/ops-transformer --pr 3228

    # ops-math 仓库
    python get_gitcode_pr_diff.py --repo https://gitcode.com/cann/ops-math --pr 123 --stat

    # ops-nn 仓库
    python get_gitcode_pr_diff.py --repo https://gitcode.com/cann/ops-nn --pr 456 \
    --file-filter "*.asc" --output diff.txt

与 ascendc-ops-reviewer 集成:
    python skills/ascendc-code-review/scripts/get_gitcode_pr_diff.py --repo <url> --pr <num>
"""

import argparse
import fnmatch
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile

# 常量定义
ALLOWED_GITCODE_DOMAIN = "gitcode.com"
TEMP_DIR_PREFIX = "gitcode_pr_"

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr
)
logger = logging.getLogger(__name__)


def parse_repo_url(url: str) -> tuple[str, str]:
    """解析仓库链接，返回 (owner, repo)

    支持格式:
        - https://gitcode.com/owner/repo
        - https://gitcode.com/owner/repo.git
        - https://gitcode.com/owner/repo/pulls/123

    Args:
        url: 仓库链接

    Returns:
        tuple[str, str]: (owner, repo)

    Raises:
        ValueError: 当 URL 格式不正确时抛出
    """
    # 验证 URL 格式 - 只允许 https://gitcode.com 开头
    if not url.startswith(f"https://{ALLOWED_GITCODE_DOMAIN}/"):
        raise ValueError(f"只支持 {ALLOWED_GITCODE_DOMAIN} 仓库，当前 URL: {url}")

    url = url.rstrip("/")

    # 使用更精确的正则移除末尾的 .git 和 /pulls/xxx
    url = re.sub(r"/pulls/\d+$", "", url)
    url = re.sub(r"\.git$", "", url)

    # 提取 owner/repo
    match = re.search(r"gitcode\.com/([^/]+)/([^/]+)", url)
    if not match:
        raise ValueError(f"无法从 URL 解析 owner/repo: {url}")

    owner = match.group(1)
    repo = match.group(2)

    return owner, repo


def run_git_command(
    cmd: list[str], cwd: str | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    """执行 git 命令

    Args:
        cmd: git 命令列表
        cwd: 工作目录
        check: 是否检查返回码

    Returns:
        subprocess.CompletedProcess: 命令执行结果

    Raises:
        subprocess.CalledProcessError: 当命令执行失败且 check=True 时抛出
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check,
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error("Git 命令失败: %s", " ".join(cmd))
        logger.error("工作目录: %s", cwd or "当前目录")
        logger.error("错误信息: %s", e.stderr)
        raise


def _extract_first_diff(diff_content: str) -> str:
    """提取 git show -m 输出中的第一个 diff

    git show -m 会为每个 parent 生成一个 diff，
    merge commit 有两个 parent，我们只需要第一个。

    Args:
        diff_content: git show -m 的完整输出

    Returns:
        str: 第一个 diff 的内容
    """
    if not diff_content:
        return diff_content

    lines = diff_content.splitlines(keepends=True)
    first_diff_lines: list[str] = []

    for line in lines:
        if line.startswith("commit ") and first_diff_lines:
            break
        first_diff_lines.append(line)

    if len(first_diff_lines) < len(lines):
        return "".join(first_diff_lines)

    return diff_content


def _apply_file_filter(diff_content: str, file_filter: str) -> str:
    """应用文件路径过滤到 diff 内容

    Args:
        diff_content: diff 内容
        file_filter: 文件路径过滤模式（通配符）

    Returns:
        str: 过滤后的 diff 内容
    """
    filtered_lines: list[str] = []
    current_file: str | None = None
    include_file = False

    for line in diff_content.splitlines(keepends=True):
        if line.startswith("diff --git"):
            match = re.search(r"diff --git a/(.*?) b/(.*)", line)
            if match:
                current_file = match.group(2)
                include_file = bool(
                    current_file and fnmatch.fnmatch(current_file, file_filter)
                )
            else:
                current_file = None
                include_file = False

        if include_file:
            filtered_lines.append(line)

    return "".join(filtered_lines)


def _cleanup_temp_dir(temp_dir: str) -> None:
    """清理临时目录

    Args:
        temp_dir: 临时目录路径
    """
    try:
        shutil.rmtree(temp_dir)
    except OSError as e:
        logger.warning("清理临时目录失败: %s", e)


def get_pr_diff_git(
    repo_url: str,
    pr_number: int,
    file_filter: str | None = None,
    stat_only: bool = False,
) -> str:
    """通过 git 命令获取 PR diff

    使用 GitCode 的 merge 引用获取正确的 PR diff：
    - refs/merge-requests/{PR}/merge 指向虚拟合并提交
    - 使用 git show merge_commit 获取 PR 实际变更

    Args:
        repo_url: 仓库链接（.git 格式）
        pr_number: PR 编号
        file_filter: 文件路径过滤模式（可选）
        stat_only: 是否仅返回统计信息

    Returns:
        str: PR diff 内容

    Raises:
        subprocess.CalledProcessError: 当 git 命令执行失败时抛出
    """
    # 参数验证
    if not isinstance(pr_number, int) or pr_number <= 0:
        raise ValueError(f"PR 编号必须是正整数，当前值: {pr_number}")

    temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    repo_dir = os.path.join(temp_dir, "repo")

    try:
        logger.info("正在克隆仓库...")
        run_git_command(
            [
                "git",
                "clone",
                "--bare",
                repo_url,
                repo_dir,
            ]
        )

        # 获取 PR merge 引用（虚拟合并提交）
        logger.info("正在获取 PR #%d merge 引用...", pr_number)
        merge_ref = f"mr_{pr_number}_merge"
        run_git_command(
            [
                "git",
                "fetch",
                "origin",
                f"refs/merge-requests/{pr_number}/merge:{merge_ref}",
            ],
            cwd=repo_dir,
        )

        # 使用 git show 获取 PR diff（显示 merge 提交的变更）
        # 这会自动显示 PR 分支相对于 target 分支的实际变更
        logger.info("正在生成 diff...")

        if stat_only:
            result = run_git_command(
                ["git", "show", "-m", "--stat", merge_ref],
                cwd=repo_dir,
            )
        else:
            result = run_git_command(
                ["git", "show", "-m", merge_ref],
                cwd=repo_dir,
            )

        diff_content = result.stdout

        diff_content = _extract_first_diff(diff_content)

        if file_filter and diff_content and not stat_only:
            diff_content = _apply_file_filter(diff_content, file_filter)

        return diff_content

    finally:
        _cleanup_temp_dir(temp_dir)


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器

    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        description="获取 GitCode PR 的 diff 内容（无需 token）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # ops-transformer 仓库
  %(prog)s --repo https://gitcode.com/cann/ops-transformer --pr 3228
  %(prog)s --repo https://gitcode.com/cann/ops-transformer --pr 3228 --stat

  # ops-math 仓库
  %(prog)s --repo https://gitcode.com/cann/ops-math --pr 123 --output pr_123.diff

  # ops-nn 仓库
  %(prog)s --repo https://gitcode.com/cann/ops-nn --pr 456 --file-filter "*.asc"

  # ops-cv 仓库
  %(prog)s --repo https://gitcode.com/cann/ops-cv --pr 789 --stat --verbose
        """,
    )
    parser.add_argument(
        "--repo", required=True, help="仓库链接，如 https://gitcode.com/owner/repo"
    )
    parser.add_argument("--pr", required=True, type=int, help="PR 编号")
    parser.add_argument("--output", help="输出文件路径（默认输出到 stdout）")
    parser.add_argument(
        "--file-filter", help="文件路径过滤，支持通配符（如 *.asc、**/*.py）"
    )
    parser.add_argument("--stat", action="store_true", help="仅显示变更统计信息")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    return parser


def validate_and_get_repo_url(repo_url_str: str) -> tuple[str, str, str]:
    """验证并构建仓库 URL

    Args:
        repo_url_str: 用户提供的仓库链接字符串

    Returns:
        tuple[str, str, str]: (owner, repo, 完整的 .git URL)

    Raises:
        ValueError: 当 URL 格式不正确时抛出
    """
    owner, repo = parse_repo_url(repo_url_str)
    repo_url = f"https://{ALLOWED_GITCODE_DOMAIN}/{owner}/{repo}.git"
    return owner, repo, repo_url


def setup_logging(verbose: bool, owner: str, repo: str, pr_number: int) -> None:
    """设置日志级别

    Args:
        verbose: 是否显示详细日志
        owner: 仓库 owner
        repo: 仓库名称
        pr_number: PR 编号
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("仓库: %s/%s", owner, repo)
        logger.debug("PR: #%d", pr_number)


def write_output(diff_content: str, output_path: str | None, verbose: bool) -> None:
    """输出 diff 结果

    Args:
        diff_content: diff 内容
        output_path: 输出文件路径（None 表示输出到 stdout）
        verbose: 是否显示详细日志
    """
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(diff_content)
        if verbose:
            logger.debug("已写入: %s", output_path)
    else:
        print(diff_content)


def main() -> None:
    """主函数 - 解析命令行参数并获取 PR diff"""
    parser = create_argument_parser()
    args = parser.parse_args()

    owner, repo, repo_url = validate_and_get_repo_url(args.repo)
    setup_logging(args.verbose, owner, repo, args.pr)

    try:
        diff_content = get_pr_diff_git(
            repo_url=repo_url,
            pr_number=args.pr,
            file_filter=args.file_filter,
            stat_only=args.stat,
        )
        if not diff_content:
            logger.info("未找到变更或 diff 为空")
            sys.exit(0)

        write_output(diff_content, args.output, args.verbose)
    except Exception as e:
        logger.error("获取 diff 失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
