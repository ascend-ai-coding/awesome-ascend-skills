#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
为 organized_docs 添加 aclnn_{算子名}.h 的代码功能描述

功能：
1. 从 organized_docs 读取每个 markdown 文档
2. 找到对应的 op_api_tree.json 文件
3. 从 JSON 的 file_tree 中扫描 aclnn_*.h 文件名，提取真实算子名
4. 从中提取 aclnn_{算子名}.h 的 response（代码功能描述）
5. 将该描述插入到文档中"当前文件路径"的后面
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from src.logger import debug, error, info, warning

PROGRESS_INTERVAL = 10  # Log progress every N files


def extract_file_path(content):
    """
    Extract the file path from markdown content.

    Args:
        content: The markdown file content

    Returns:
        The extracted file path or None
    """
    match = re.search(r"当前文件路径:\s*`([^`]+)`", content)
    if match:
        return match.group(1)
    return None


def extract_operator_name_from_json(json_path):
    """
    Derive the real operator name from the op_api_tree.json content.
    Scans file_tree keys for aclnn_*.h pattern to find the true operator name,
    avoiding contamination from commit hash suffixes or version names in directory paths.

    Args:
        json_path: Path to the op_api_tree.json file

    Returns:
        The operator name or None
        Example: finds key 'aclnn_abs.h' -> returns 'abs'
                 finds key 'aclnn_moe_distribute_dispatch_v2.h' -> returns 'moe_distribute_dispatch_v2'
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_tree = data.get("file_tree", {})

        # Strategy 1: Find aclnn_*.h key and extract operator name from filename
        for key in file_tree:
            if key.startswith("aclnn_") and key.endswith(".h"):
                start = len("aclnn_")
                end = -len(".h")
                return key[start:end]

        # Strategy 2: Use the 'operator' field written by generate_directory_trees.py
        operator = data.get("operator")
        if operator:
            return operator

    except Exception as e:
        warning(f"Error extracting operator name from JSON {json_path}: {e}")

    return None


def _extract_operator_name_from_path(file_path: str) -> str:
    """从绝对路径中提取算子名。

    在路径中查找 op_api/op_host 等目标文件夹，取其父目录名作为算子名。
    """
    target_folders_in_path = {"op_api", "op_graph", "op_host", "op_kernel"}
    posix_path = Path(file_path).as_posix()
    for folder in target_folders_in_path:
        idx = posix_path.find(f"/{folder}/")
        if idx >= 0:
            prefix = posix_path[:idx]
            return Path(prefix).name
    return ""


def _search_op_api_tree_by_operator(directory_trees_dir: str, operator_name: str) -> Optional[str]:
    """在 directory_trees 目录中搜索包含指定算子名的 op_api_tree.json。"""
    dir_path = Path(directory_trees_dir)
    if not dir_path.exists():
        return None

    for root, dirs, files in os.walk(dir_path):
        if "op_api_tree.json" in files:
            normalized_root = Path(root).as_posix()
            if f"/{operator_name}/" in normalized_root or normalized_root.endswith(f"/{operator_name}"):
                return str(Path(root) / "op_api_tree.json")
    return None


def find_json_path(file_path, directory_trees_dir, operator_level_path=None) -> Optional[str]:
    """
    根据绝对路径和 directory_trees 目录结构，查找 op_api_tree.json 文件。

    Args:
        file_path: 文件的绝对路径（从 Markdown 中提取）
        directory_trees_dir: directory_trees 目录路径
        operator_level_path: 算子级别路径（用于跨版本回退搜索）

    Returns:
        op_api_tree.json 文件路径，或 None
    """
    try:
        # 从绝对路径提取算子名
        operator_name = _extract_operator_name_from_path(file_path)
        if not operator_name:
            debug(f"无法从路径 {file_path} 中提取算子名")
            return None

        # Step 1: 在当前版本的 directory_trees 中搜索
        result = _search_op_api_tree_by_operator(directory_trees_dir, operator_name)
        if result:
            return result

        # Step 2: 跨版本回退搜索
        if operator_level_path and Path(operator_level_path).is_dir():
            for version_dir in sorted([p.name for p in Path(operator_level_path).iterdir()]):
                version_trees_dir = str(Path(operator_level_path) / version_dir / "tree_id" / "directory_trees")
                if not Path(version_trees_dir).is_dir():
                    continue
                result = _search_op_api_tree_by_operator(version_trees_dir, operator_name)
                if result:
                    debug(f"在相邻版本 '{version_dir}' 中找到 {operator_name} 的 op_api_tree.json")
                    return result

        debug(f"未找到 {operator_name} 的 op_api_tree.json")

    except Exception as e:
        warning(f"搜索 JSON 文件时出错 ({file_path}): {e}")

    return None


def extract_aclnn_h_response(json_path, operator_name):
    """
    Extract the response for aclnn_{operator_name}.h from the JSON file.

    Args:
        json_path: Path to the op_api_tree.json file
        operator_name: The operator name

    Returns:
        The response string or None
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Navigate to file_tree
        file_tree = data.get("file_tree", {})
        print("FILE_TREE:", file_tree)
        # Look for aclnn_{operator_name}.h
        aclnn_h_file = f"aclnn_{operator_name}.h"

        if aclnn_h_file in file_tree:
            return file_tree[aclnn_h_file].get("response")
        else:
            # 无精准匹配时，模糊匹配以 aclnn_ 开头、以 .h 结尾的文件
            for key, value in file_tree.items():
                if key.startswith("aclnn_") and key.endswith(".h"):
                    return value.get("response")

    except Exception as e:
        error(f"Error reading JSON file {json_path}: {e}")

    return None


def insert_aclnn_h_description(content, response, operator_name):
    """
    Insert the aclnn_{operator_name}.h description into the markdown content.

    Args:
        content: The original markdown content
        response: The response from the JSON (with escaped \n)
        operator_name: The operator name

    Returns:
        The updated markdown content
    """
    # Convert escaped newlines to actual newlines
    response = response.replace("\\n", "\n")

    # Find the line containing "所属_directory"
    lines = content.split("\n")

    insertion_point = -1
    for i, line in enumerate(lines):
        # Find the line after "所属_directory" (after the line containing "- 所属目录:")
        if "## 一、算子整体说明" in line:
            # Insert after this line (at i+1), but need to skip empty lines
            insertion_point = i + 1
            break

    end_point = -1
    for i, line in enumerate(lines):
        # Find the line after "所属_directory" (after the line containing "- 所属目录:")
        if "## 二、代码整体目录及说明" in line:
            # Insert after this line (at i+1), but need to skip empty lines
            end_point = i
            break

    if insertion_point == -1:
        warning("Could not find '所属目录' line")
        return content

    # Skip any empty lines after the insertion point
    while insertion_point < len(lines) and lines[insertion_point].strip() == "":
        insertion_point += 1

    header_content = response.split("## 代码功能")[-1].split("## 关键结构")[0].strip()
    # Build the new section to insert
    new_section = f"ACLNN API 头文件描述 (aclnn_{operator_name}.h)：{header_content}\n"

    # Insert the new section
    updated_lines = lines[:insertion_point] + [new_section] + lines[end_point:]
    updated_content = "\n".join(updated_lines)

    return updated_content


def process_md_file(md_path, directory_trees_dir, operator_level_path=None):
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        file_path = extract_file_path(content)
        if not file_path:
            debug(f"No file path found in {Path(md_path).name}")
            return False

        json_path = find_json_path(file_path, directory_trees_dir, operator_level_path)
        if not json_path:
            debug(
                f"No op_api_tree.json for {file_path} in {Path(md_path).name} — "
                f"likely a non-op_api file (op_kernel/op_host) without ACLNN headers"
            )
            return False

        # Derive operator name from the JSON content (not from the path)
        operator_name = extract_operator_name_from_json(json_path)
        if not operator_name:
            debug(f"No operator name found in {json_path} for {Path(md_path).name}")
            return False

        # Extract the aclnn_{operator_name}.h response
        response = extract_aclnn_h_response(json_path, operator_name)
        if not response:
            warning(f"Could not find aclnn_{operator_name}.h response in {json_path} for {Path(md_path).name}")
            return False

        # Insert the aclnn_h description
        updated_content = insert_aclnn_h_description(content, response, operator_name)

        # Write the updated content back to the file
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(updated_content)

        return True

    except Exception as e:
        error(f"Error processing {md_path}: {e}")
        return False


def main(out_path):
    organized_docs_dir = out_path / "organized_docs"
    directory_trees_dir = out_path / "directory_trees"

    # Derive operator-level path for cross-version fallback:
    # out_path = {operator_name}/{version}/tree_id/
    # operator_level_path = {operator_name}/
    operator_level_path = out_path.parent.parent if hasattr(out_path, "parent") else None

    info("=" * 60)
    info("Starting to add ACLNN API header descriptions")
    if operator_level_path:
        info(f"Operator-level path: {operator_level_path} (cross-version fallback enabled)")
    info("=" * 60)

    md_files = [str(f) for f in Path(organized_docs_dir).glob("*.md")]
    total_files = len(md_files)

    info(f"Found {total_files} markdown files to process")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for i, md_file in enumerate(md_files):
        filename = Path(md_file).name

        if (i + 1) % PROGRESS_INTERVAL == 0 or i == 0:
            info(f"Processing {i + 1}/{total_files}: {filename}")

        result = process_md_file(md_file, directory_trees_dir, operator_level_path)

        if result:
            processed_count += 1
        else:
            skipped_count += 1

    # Generate final summary
    info("=" * 60)
    info("Processing complete")
    info(f"Total files: {total_files}")
    info(f"Successfully processed: {processed_count}")
    info(f"Skipped: {skipped_count}")
    info(f"Errors: {error_count}")
    info("=" * 60)
