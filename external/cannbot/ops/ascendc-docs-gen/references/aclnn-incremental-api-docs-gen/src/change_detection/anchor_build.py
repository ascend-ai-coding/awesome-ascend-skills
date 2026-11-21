# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
import os
from pathlib import Path

from src.snapshot_construction.snapshot import load_snapshot
from src.logger import info, debug, warning, error


def anchor_rebuild(
    operator_name,
    operator_path,
    change_operator_name,
    data_path,
    filter_list,
    work_dir,
    file_dir_list=None,
    old_tree_id_path=None,
    md_file=None,
    source_files=None,
):
    """
    执行算子重建流程（包含额外文件处理）。

    Args:
        operator_name: 算子名称,结构化的
        operator_path: 算子目录路径,结构化的
        change_operator_name: 改变后的原始目录。
        data_path: 输出数据路径
        filter_list: 文档部分过滤列表
        file_dir_list: 变更文件列表（用于增量 tree_id）
        old_tree_id_path: 旧版本 tree_id 路径（用于增量）
        file_list: 额外关联文件列表（绝对路径），需要同样进行 anchor 构建
        md_file: Optional markdown source file path for md_stem derivation.
        source_files: Optional list of source files to filter anchors.
    """
    import shutil

    # 1. 创建算子存储目录
    operator_path = os.path.join(work_dir, operator_path)
    operator_storage_path = f"{data_path}/{operator_name}"
    if os.path.exists(operator_storage_path):
        shutil.rmtree(operator_storage_path)
    shutil.copytree(operator_path, operator_storage_path)

    # 2. source_files 未指定但 md_file 已知时，按 md_stem 解析源文件，使 anchor 范围与 main_snapshot 对齐
    if source_files is None and md_file is not None:
        from src.snapshot_construction.anchor_construction.common.md_resolver import (
            resolve_md_to_source_files,
        )

        md_path = Path(operator_path) / "docs" / md_file
        resolved = resolve_md_to_source_files(md_path, Path(operator_path))
        source_files = [str(f) for f in resolved]
        info(
            f"anchor_rebuild: auto-resolved {len(source_files)} source files from md_stem '{md_file}'"
        )
        debug(f"anchor_rebuild: resolved source_files = {source_files}")

    # 3. 执行 load_snapshot
    # Dependency expansion needs the checkout root to resolve shared files.
    echo_info = load_snapshot(
        operator_path,
        data_path,
        str(work_dir),
        doc_part_control=filter_list,
        file_dir_list=file_dir_list,
        old_tree_id_path=old_tree_id_path,
        md_file=md_file,
        source_files=source_files,
    )

    # 4. 返回统计信息和额外文件判断结果
    return echo_info
