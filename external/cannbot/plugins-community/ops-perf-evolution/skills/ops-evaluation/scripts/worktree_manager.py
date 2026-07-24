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
# ----------------------------------------------------------------------------------------------------------
"""
Git Worktree 管理模块 — 为 ops-evo 并行构建提供文件系统隔离。

每个进化变体在独立的 git worktree 中执行构建，避免 build/ 和 build_out/
目录冲突。Worktree 共享 .git 对象库，磁盘增量仅为工作目录文件。

用法:
    from worktree_manager import create_build_worktree, remove_worktree, apply_modifications

    # 创建 worktree
    wt_path = create_build_worktree(
        repo_root="/path/to/ops-nn",
        worktree_base="output/session/worktrees",
        task_id="round_1_p0",
    )

    # 应用修改
    apply_modifications(
        worktree_path=wt_path,
        op_path_relative="nn/FastGELU",
        modified_files_dir="output/session/round_1/parallel_0/modified_files",
    )

    # 清理
    remove_worktree(wt_path)
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 数据输出专用 logger：CLI 结果（JSON/表格/报告）走 stdout（agent 调用协议通道），
# 与 LOGGER（stderr 进度/警告）分离，避免 lint G.LOG.02 误报 print。
DATA_LOGGER = logging.getLogger(f"{__name__}.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)

# git 可执行文件绝对路径（避免依赖 PATH 查找）
GIT_BIN = shutil.which("git") or "/usr/bin/git"


def create_build_worktree(
    repo_root: str,
    worktree_base: str,
    task_id: str,
    commit: str = "HEAD",
) -> str:
    """创建 git worktree 用于隔离构建。

    Args:
        repo_root: ops 仓的 git 根目录
        worktree_base: worktree 存放的父目录
        task_id: 任务标识（如 "round_1_p0"），用作 worktree 目录名和分支名
        commit: 基于哪个 commit 创建（默认 HEAD）

    Returns:
        worktree 的绝对路径

    Raises:
        RuntimeError: git worktree 创建失败
    """
    repo_root = os.path.abspath(repo_root)
    worktree_base = os.path.abspath(worktree_base)
    os.makedirs(worktree_base, exist_ok=True)

    worktree_path = os.path.join(worktree_base, task_id)

    # 如果已存在，先清理
    if os.path.exists(worktree_path):
        logger.warning(f"Worktree already exists, removing: {worktree_path}")
        remove_worktree(worktree_path, repo_root=repo_root)

    # 创建 detached worktree（不创建新分支，避免分支名冲突）
    cmd = [
        GIT_BIN, "-C", repo_root,
        "worktree", "add", "--detach",
        worktree_path, commit,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create worktree at {worktree_path}:\n{result.stderr}"
        )

    logger.info(f"Created worktree: {worktree_path} (from {commit})")
    return worktree_path


def remove_worktree(
    worktree_path: str,
    repo_root: Optional[str] = None,
    force: bool = True,
) -> bool:
    """清理 git worktree。

    Args:
        worktree_path: worktree 目录路径
        repo_root: 主仓根目录（若为 None，尝试自动检测）
        force: 是否强制删除（即使有未提交修改）

    Returns:
        是否成功清理
    """
    worktree_path = os.path.abspath(worktree_path)

    if not os.path.exists(worktree_path):
        return True

    # 尝试通过 git worktree remove 正式清理
    if repo_root:
        cmd = [GIT_BIN, "-C", repo_root, "worktree", "remove"]
        if force:
            cmd.append("--force")
        cmd.append(worktree_path)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            logger.info(f"Removed worktree: {worktree_path}")
            return True
        logger.warning(f"git worktree remove failed: {result.stderr}")

    # 降级：直接删除目录 + prune
    try:
        shutil.rmtree(worktree_path)
        if repo_root:
            subprocess.run(
                [GIT_BIN, "-C", repo_root, "worktree", "prune"],
                capture_output=True, timeout=30,
            )
        logger.info(f"Force-removed worktree directory: {worktree_path}")
        return True
    except OSError as e:
        logger.error(f"Failed to remove worktree {worktree_path}: {e}")
        return False


def apply_modifications(
    worktree_path: str,
    op_path_relative: str,
    modified_files_dir: str,
) -> int:
    """将修改后的文件应用到 worktree 中的算子目录。

    Args:
        worktree_path: worktree 根目录
        op_path_relative: 算子在仓中的相对路径（如 "nn/FastGELU"）
        modified_files_dir: 修改文件所在目录（含 op_kernel/, op_host/ 子目录）

    Returns:
        复制的文件数量
    """
    target_base = os.path.join(worktree_path, op_path_relative)
    source_base = os.path.abspath(modified_files_dir)

    if not os.path.isdir(source_base):
        logger.warning(f"Modified files directory not found: {source_base}")
        return 0

    copied = 0
    for subdir in ("op_kernel", "op_host"):
        src_dir = os.path.join(source_base, subdir)
        dst_dir = os.path.join(target_base, subdir)

        if not os.path.isdir(src_dir):
            continue

        os.makedirs(dst_dir, exist_ok=True)

        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(dst_dir, filename)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
                copied += 1

    logger.info(
        f"Applied {copied} modified files to worktree: "
        f"{target_base}"
    )
    return copied


def get_worktree_list(repo_root: str) -> list:
    """列出仓库的所有 worktree。"""
    result = subprocess.run(
        [GIT_BIN, "-C", repo_root, "worktree", "list", "--porcelain"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return []

    worktrees = []
    current = {}
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            if current:
                worktrees.append(current)
            current = {"path": line[len("worktree "):]}
        elif line.startswith("HEAD "):
            current["head"] = line[len("HEAD "):]
        elif line == "detached":
            current["detached"] = True
        elif line.startswith("branch "):
            current["branch"] = line[len("branch "):]
    if current:
        worktrees.append(current)

    return worktrees


def cleanup_session_worktrees(
    repo_root: str,
    worktree_base: str,
    keep_tasks: Optional[list] = None,
) -> int:
    """清理一个 session 的所有 worktree，可选保留指定任务的。

    Args:
        repo_root: 主仓根目录
        worktree_base: worktree 存放的父目录
        keep_tasks: 要保留的 task_id 列表（如 ["round_2_p1"]）

    Returns:
        清理的 worktree 数量
    """
    if not os.path.isdir(worktree_base):
        return 0

    keep_set = set(keep_tasks or [])
    cleaned = 0

    for entry in os.listdir(worktree_base):
        if entry in keep_set:
            continue
        wt_path = os.path.join(worktree_base, entry)
        if os.path.isdir(wt_path):
            if remove_worktree(wt_path, repo_root=repo_root):
                cleaned += 1

    return cleaned


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Git Worktree Manager for ops-evo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = subparsers.add_parser("create", help="Create a build worktree")
    p_create.add_argument("--repo-root", required=True, help="ops 仓的 git 根目录")
    p_create.add_argument("--worktree-base", required=True, help="worktree 存放的父目录")
    p_create.add_argument("--task-id", required=True, help="任务标识（用作目录名）")
    p_create.add_argument("--commit", default="HEAD", help="基于哪个 commit 创建")

    # remove
    p_remove = subparsers.add_parser("remove", help="Remove a worktree")
    p_remove.add_argument("--worktree-path", required=True, help="worktree 目录路径")
    p_remove.add_argument("--repo-root", default=None, help="主仓根目录")

    # cleanup
    p_cleanup = subparsers.add_parser("cleanup", help="Cleanup session worktrees")
    p_cleanup.add_argument("--repo-root", required=True, help="主仓根目录")
    p_cleanup.add_argument("--worktree-base", required=True, help="worktree 存放的父目录")
    p_cleanup.add_argument("--keep", default=None, help="要保留的 task_id（逗号分隔）")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    if args.command == "create":
        wt_path = create_build_worktree(
            repo_root=args.repo_root,
            worktree_base=args.worktree_base,
            task_id=args.task_id,
            commit=args.commit,
        )
        DATA_LOGGER.info("%s", wt_path)

    elif args.command == "remove":
        ok = remove_worktree(args.worktree_path, repo_root=args.repo_root)
        DATA_LOGGER.info("%s", "OK" if ok else "FAILED")

    elif args.command == "cleanup":
        keep = args.keep.split(",") if args.keep else None
        n = cleanup_session_worktrees(
            repo_root=args.repo_root,
            worktree_base=args.worktree_base,
            keep_tasks=keep,
        )
        DATA_LOGGER.info("Cleaned %d worktrees", n)


if __name__ == "__main__":
    main()
