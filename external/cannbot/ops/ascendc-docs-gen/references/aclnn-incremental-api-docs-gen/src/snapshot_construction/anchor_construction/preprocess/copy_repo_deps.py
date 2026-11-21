# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""DEPRECATED: This module is no longer used after the version-split removal refactor.

Include scanning now works directly against the real directory structure. Kept for reference only.
"""

import logging
import shutil
from pathlib import Path

from src.snapshot_construction.anchor_construction.common.include_analyzer import (
    SOURCE_EXTENSIONS,
    build_file_index,
    build_transitive,
    classify_include,
    extract_includes,
)

logger = logging.getLogger(__name__)


def collect_repo_deps(staged_op_dir, real_op_dir, real_ops_root) -> set[str]:
    """预扫描：收集 ② repo_other 传递依赖的 display 路径集合。

    读 staged 文件内容（与源同内容），但 classify_include 用真实 op_dir/ops_root
    （相对 include 路径是为真实位置设计的）。
    """
    staged_op_dir = Path(staged_op_dir)
    real_op_dir = Path(real_op_dir)
    real_ops_root = Path(real_ops_root)

    deps: set[str] = set()
    transitive_cache: dict = {}
    file_index = build_file_index(real_ops_root)

    for staged_file in staged_op_dir.rglob("*"):
        if not staged_file.is_file() or staged_file.suffix not in SOURCE_EXTENSIONS:
            continue

        rel = staged_file.relative_to(staged_op_dir)
        real_file_dir = real_op_dir / rel.parent

        direct_self, direct_repo_other, direct_external = [], [], []
        for inc, qt in extract_includes(staged_file):
            if qt == "angle":
                continue
            cat, display = classify_include(inc, real_file_dir, real_ops_root, real_op_dir, file_index=file_index)
            if cat == "self":
                direct_self.append(display)
            elif cat == "repo_other":
                direct_repo_other.append({"原始引用": inc, "解析路径": display})
            else:
                direct_external.append(inc)

        if not direct_self and not direct_repo_other:
            continue

        transitive = build_transitive(
            direct_self,
            direct_repo_other,
            direct_external,
            real_ops_root,
            real_op_dir,
            file_cache=transitive_cache,
            file_index=file_index,
        )
        for item in transitive["②_仓内其他"]:
            deps.add(item["解析路径"])

    return deps


def copy_repo_deps(deps: set[str], real_ops_root, operator_storage_path) -> int:
    """把 ② 依赖文件按相对于 ops_root 的路径复制到 operator_storage 下。

    display 路径形如 "ops-math/common/inc/aicpu/aicpu_op_def.h"，
    去掉仓名前缀后拼到 operator_storage 下。
    """
    real_ops_root = Path(real_ops_root)
    operator_storage_path = Path(operator_storage_path)
    prefix = f"{real_ops_root.name}/"
    copied = 0

    for display in deps:
        if not display.startswith(prefix):
            logger.warning(f"② 依赖 display 路径未匹配仓根前缀，跳过: {display}")
            continue
        start = len(prefix)
        rel_path = display[start:]
        src = real_ops_root / rel_path
        dst = operator_storage_path / rel_path

        if not src.is_file():
            logger.warning(f"② 依赖文件不存在，跳过: {src}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        logger.debug(f"复制 ② 依赖: {src} -> {dst}")

    return copied


def stage_repo_deps(staged_op_dir, real_op_dir, real_ops_root, operator_storage_path) -> int:  # DEPRECATED
    """便捷入口：预扫描 + 复制。返回复制的文件数。"""
    deps = collect_repo_deps(staged_op_dir, real_op_dir, real_ops_root)
    if not deps:
        logger.info("无 ② 仓内依赖需要复制")
        return 0
    copied = copy_repo_deps(deps, real_ops_root, operator_storage_path)
    logger.info(f"已复制 {copied} 个 ② 仓内依赖文件到 {operator_storage_path}")
    return copied
