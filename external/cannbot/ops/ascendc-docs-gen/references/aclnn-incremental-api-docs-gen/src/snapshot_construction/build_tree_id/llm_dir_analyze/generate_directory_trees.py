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
生成目录树结构脚本
遍历 file_tree_structure 文件夹，在每一层目录中创建目录描述文件，
包含：1. 当前目录名称 2. 目录中包含文件树（嵌套JSON结构）
"""

import json
from pathlib import Path

from src.logger import debug, error, info, warning


def build_file_tree_structure(directory_path, relative_path=""):
    """递归构建文件树结构的JSON

    Args:
        directory_path: 当前目录的Path对象
        relative_path: 相对于根目录的路径（用于构建完整路径）

    Returns:
        dict: 文件树结构
    """
    tree = {}

    # 遍历当前目录中的所有项目
    for item in sorted(directory_path.iterdir()):
        if item.is_file() and item.suffix == ".json":
            # 读取JSON文件获取file_path
            try:
                with open(item, "r", encoding="utf-8") as f:
                    data = json.load(f)

                file_path = data.get("file_path", "")
                # 提取文件名（绝对路径用 Path.name）
                file_name = Path(file_path).name if file_path else item.name

                # 叶子节点：使用file_path字段中的文件名作为key，包含完整的JSON内容
                tree[file_name] = data

            except (json.JSONDecodeError, IOError) as e:
                # 如果读取失败，使用原始文件名
                tree[item.name] = {"error": f"Failed to read JSON: {e}"}

        elif item.is_dir():
            # 非叶子节点：递归处理子目录
            subtree = build_file_tree_structure(item, f"{relative_path}/{item.name}")
            if subtree:  # 只添加非空的子树
                tree[item.name] = subtree

    return tree


def generate_directory_description_files(out_path):
    """为每一层目录生成描述文件，保存到新目录中"""
    source_dir = out_path / "file_tree_structure"
    output_dir = out_path / "directory_trees"

    if not source_dir.exists():
        error(f"错误: {source_dir} 目录不存在")
        return

    # 创建输出目录（如果不存在）
    output_dir.mkdir(exist_ok=True)

    # 统计信息
    created_count = 0
    skipped_count = 0
    error_count = 0

    info(f"源目录: {source_dir}")
    info(f"输出目录: {output_dir}")

    # 先清理输出目录中的内容（保留目录结构，但删除之前的_tree.json文件）
    # 注意：这里只删除_tree.json文件，保留目录结构
    for tree_file in output_dir.rglob("*_tree.json"):
        tree_file.unlink()

    # 遍历所有目录（包括子目录）
    for directory in sorted(source_dir.rglob("*")):
        if not directory.is_dir():
            continue

        # 获取相对于source_dir的路径
        relative_path = directory.relative_to(source_dir)

        # 在输出目录中创建对应的目录结构
        output_directory = output_dir / relative_path
        output_directory.mkdir(parents=True, exist_ok=True)

        # 构建当前目录的文件树
        file_tree = build_file_tree_structure(directory)

        # 如果文件树为空，跳过
        if not file_tree:
            skipped_count += 1
            continue

        # Extract operator name from relative path (first component is always the operator name)
        operator_name = relative_path.parts[0] if relative_path.parts else directory.name

        # 创建目录描述
        directory_info = {
            "directory_name": directory.name,
            "operator": operator_name,
            "file_tree": file_tree,
        }

        # 生成输出文件名（在输出目录中）
        output_file = output_directory / f"{directory.name}_tree.json"

        # 写入JSON文件
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(directory_info, f, ensure_ascii=False, indent=2)

            created_count += 1
            info(f"已创建: {str(output_file)}")

        except Exception as e:
            error(f"错误: 创建 {str(output_file)} 时失败: {e}")
            error_count += 1

    # 打印统计信息
    info("=" * 50)
    info("处理完成:")
    info(f"  创建描述文件: {created_count} 个")
    info(f"  跳过空目录: {skipped_count} 个")
    info(f"  错误: {error_count} 个")
    info(f"  结果目录: {output_dir.resolve()}")
    info("=" * 50)


def main(out_path):
    """入口函数，兼容 run_pipeline.py 调用"""
    generate_directory_description_files(out_path)


if __name__ == "__main__":
    main()
