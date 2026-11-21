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
算子文档生成工具
从 llm_responses 和 file_tree_structure 中整理信息，为每个叶子节点生成完整的算子代码说明文档
"""

import json
import os
from collections import defaultdict
from pathlib import Path

from src.logger import debug, error, info, warning


def extract_relative_within_operator(abs_path: str, operator_name: str) -> str:
    """从绝对路径中提取算子目录内部的相对路径。

    例如: abs_path='D:/data/ops-math/math/abs/op_api/aclnn_abs.h', operator_name='abs'
    → 返回 'op_api/aclnn_abs.h'
    """
    posix_path = Path(abs_path).as_posix()
    idx = posix_path.find(f"/{operator_name}/")
    if idx >= 0:
        start = idx + len(f"/{operator_name}/")
        return posix_path[start:]
    return Path(abs_path).name


def load_llm_responses(llm_dir):
    """
    加载目录描述，返回 {目录路径: 描述} 的字典

    Args:
        llm_dir: llm_responses 目录路径

    Returns:
        dir_desc: 字典，键为目录路径，值为目录功能描述
    """
    dir_desc = {}
    for root, dirs, files in os.walk(llm_dir):
        for file in files:
            if file.endswith(".json"):
                path = Path(root) / file
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    prompt_path = Path(data["prompt_path"]).as_posix()
                    # 去掉最后的文件名，得到目录路径
                    pp = Path(prompt_path)
                    if pp.suffix == ".txt":
                        dir_path = str(pp.parent)
                    else:
                        dir_path = prompt_path
                    dir_desc[dir_path] = data["response"]
    return dir_desc


def load_file_descriptions(fs_dir):
    """
    加载文件描述，返回按算子分组的字典

    Args:
        fs_dir: file_tree_structure 目录路径

    Returns:
        files_by_operator: 字典，键为算子名，值为该算子的所有叶子节点信息列表
        all_files: 所有叶子节点的完整信息列表
    """
    files_by_operator = defaultdict(list)
    all_files = []

    for root, dirs, files in os.walk(fs_dir):
        for file in files:
            if file.endswith(".json"):
                path = Path(root) / file
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    file_path = Path(data["file_path"]).as_posix()
                    operator = data["operator"]
                    operator_root_dir = data.get("operator_root_dir", data.get("category", operator))

                    files_by_operator[operator].append(
                        {
                            "file_path": file_path,
                            "filename": Path(file_path).name,
                            "category": operator_root_dir,
                            "operator_root_dir": operator_root_dir,
                            "response": data["response"],
                        }
                    )
                    # all_files 中也补充 operator_root_dir
                    data_with_root = dict(data)
                    data_with_root["operator_root_dir"] = operator_root_dir
                    all_files.append(data_with_root)

    return dict(files_by_operator), all_files


def build_dirs_for_operator(operator, operator_root_dir, all_leaves):
    """
    收集指定算子涉及的所有目录

    Args:
        operator: 算子名称
        operator_root_dir: 算子根目录描述的 key
        all_leaves: 所有叶子节点的原始数据

    Returns:
        all_dirs: 该算子涉及的目录列表（按深度排序）
    """
    all_dirs = set()

    for leaf in all_leaves:
        if leaf["operator"] == operator:
            file_path = leaf["file_path"]
            # 从绝对路径中提取算子目录内部的相对路径
            relative = extract_relative_within_operator(file_path, operator)
            # 根目录用 operator_root_dir
            all_dirs.add(operator_root_dir)
            parts = relative.split("/")
            # 添加中间目录
            for i in range(len(parts) - 1):
                dir_path = operator_root_dir + "/" + "/".join(parts[: i + 1])
                all_dirs.add(dir_path)

    # 按深度排序（浅层在前）
    sorted_dirs = sorted(all_dirs, key=lambda x: x.count("/"))

    return sorted(sorted_dirs)


def build_directory_tree(sorted_dirs, dir_desc):
    """
    构建目录树结构（带功能标注）

    Args:
        sorted_dirs: 排序后的目录列表
        dir_desc: 目录描述字典

    Returns:
        tree: 树结构字典
        tree_label: 树节点标签（功能描述）
    """
    tree = {}
    tree_label = {}

    if not sorted_dirs:
        return tree, tree_label

    # 根目录
    root = sorted_dirs[0]
    tree[root] = []
    tree_label[root] = dir_desc.get(root, "")

    # 构建树
    for dir_path in sorted_dirs[1:]:
        # 用 Path 取父目录路径（兼容跨平台）
        parent = str(Path(dir_path).parent)

        if parent in tree:
            # 直接添加到父节点
            tree[parent].append(dir_path)
        else:
            # 查找最近的父目录
            # 简化处理：找到以 parent 结尾的目录
            found_parent = None
            for existing_dir in tree:
                if dir_path.startswith(existing_dir + "/"):
                    found_parent = existing_dir
                    break

            if found_parent is not None:
                if found_parent not in tree:
                    tree[found_parent] = []

                # 避免重复添加
                if dir_path not in tree[found_parent]:
                    tree[found_parent].append(dir_path)
            else:
                # 作为根的新子节点
                tree[root].append(dir_path)

        # 初始化当前目录的子节点列表
        if dir_path not in tree:
            tree[dir_path] = []

        # 添加功能标签
        tree_label[dir_path] = dir_desc.get(dir_path, "")

    return tree, tree_label


def format_tree_to_markdown(tree, tree_label, operator_files):
    """
    将目录树格式化为Markdown

    Args:
        tree: 树结构字典
        tree_label: 树节点标签
        operator_files: 该算子的所有文件列表

    Returns:
        markdown_text: 格式化的目录树Markdown文本
    """
    lines = []

    def print_tree(node, prefix="", is_last=True):
        if node not in tree:
            return

        children = tree[node]
        label = tree_label.get(node, "")

        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            connector = "└── " if is_last_child else "├── "

            # 添加功能标注
            if label:
                lines.append(f"{prefix}{connector}{child}（{label}）")
            else:
                lines.append(f"{prefix}{connector}{child}")

            # 检查该目录下是否有叶子文件
            child_files = [f for f in operator_files if f["file_path"].startswith(f"{child}/")]
            if child_files:
                # 找到最深一层
                max_depth = max(f["file_path"].count("/") for f in child_files)
                deepest_files = [f for f in child_files if f["file_path"].count("/") == max_depth]

                # 只列出直接子文件（当前目录的下一层）
                # fmt: off
                direct_files = [
                    f
                    for f in deepest_files
                    if f["file_path"].count("/") == path_depth(child, operator_files) + 1
                ]
                # fmt: on

                for j, f in enumerate(direct_files):
                    file_is_last = j == len(direct_files) - 1
                    file_connector = "└── " if file_is_last else "├── "
                    lines.append(f"{prefix}│   {file_connector}{f['filename']}")

        # 递归处理子节点
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            new_prefix = prefix + "    " if is_last_child else prefix + "│   "
            print_tree(child, new_prefix, is_last_child)

    def path_depth(dir_path, files):
        """获取目录在文件路径中的深度"""
        for f in files:
            if f["file_path"].startswith(f"{dir_path}/"):
                # 返回该目录下第一层文件的深度
                start = len(dir_path) + 1
                rel_path = f["file_path"][start:]
                parts = rel_path.split("/")
                return len(parts)
        return 0

    # 找到根节点
    if tree:
        root = next(iter(tree))
        root_label = tree_label.get(root, "")
        if root_label:
            lines.append(f"{root}（{root_label}）")
        else:
            lines.append(root)
        print_tree(root)

    return "\n".join(lines)


def path_depth_by_files(dir_path, operator_files):
    """
    获取目录下叶子文件相对于该目录的深度（0表示下一层就是叶子）
    """
    min_depth = float("inf")
    for f in operator_files:
        if f["file_path"].startswith(f"{dir_path}/"):
            start = len(dir_path) + 1
            rel_path = f["file_path"][start:]
            parts = rel_path.split("/")
            depth = len(parts)
            if depth < min_depth:
                min_depth = depth

    return min_depth if min_depth != float("inf") else 0


def format_tree_to_markdown_v2(tree, tree_label, operator_files):
    """
    改进的目录树格式化函数
    """
    lines = []

    # 找到根节点
    if not tree:
        return ""

    root = next(iter(tree))
    root_label = tree_label.get(root, "")
    if root_label:
        lines.append(f"{root}（{root_label}）")
    else:
        lines.append(root)

    stack = [(root, "", True, 0)]  # (node, prefix, is_last, level)

    while stack:
        node, prefix, is_last, level = stack.pop()

        children = tree.get(node, [])
        tree_label.get(node, "")

        # 根据深度决定是否继续递归
        node.count("/")

        for i, child in enumerate(children):
            child_is_last = i == len(children) - 1
            connector = "└── " if child_is_last else "├── "

            child_label = tree_label.get(child, "")
            if child_label:
                lines.append(f"{prefix}{connector}{child}（{child_label}）")
            else:
                lines.append(f"{prefix}{connector}{child}")

            # 检查该目录的子目录中是否还有叶子文件
            max_depth = 0

            for f in operator_files:
                if f["file_path"].startswith(f"{child}/"):
                    rel_depth = f["file_path"].count("/") - 1
                    if rel_depth > max_depth:
                        max_depth = rel_depth
                    # 叶子文件的完整深度减去当前目录深度
                    file_total_depth = f["file_path"].count("/")
                    if file_total_depth == child.count("/") + 1:
                        pass

            # 如果当前目录有直接叶子文件，列出它们
            direct_leaf_files = [
                f
                for f in operator_files
                if f["file_path"].startswith(f"{child}/") and f["file_path"].count("/") == child.count("/") + 1
            ]

            for j, leaf in enumerate(direct_leaf_files):
                leaf_is_last = j == len(direct_leaf_files) - 1
                leaf_connector = "└── " if leaf_is_last else "├── "
                lines.append(f"{prefix}│   {leaf_connector}{leaf['filename']}")

            # 如果还有子目录，添加到栈中
            # 检查是否还有更深层的目录
            deeper_dirs = []
            for potential_dir in tree:
                if potential_dir.startswith(f"{child}/"):
                    deeper_depth = potential_dir.count("/")
                    if deeper_depth > child.count("/"):
                        if deeper_depth == child.count("/") + 1:  # 直接子目录
                            deeper_dirs.append(potential_dir)

            # 简化处理：只处理下一层子目录
            if deeper_dirs:
                prefix + ("    " if child_is_last else "│   ")
                # 找到下一层子目录
                [d for d in tree[child] if d.count("/") == child.count("/") + 1]

                # 实际上我们应该遍历 tree[child] 的子节点
                # 重新思考：tree[child] 包含的是 child 的一级子目录
                if child in tree:
                    # 反向遍历以保持正确顺序
                    for k, subchild in enumerate(reversed(tree[child])):
                        subchild_is_last = k == len(tree[child]) - 1
                        subchild_prefix = prefix + ("    " if child_is_last else "│   ")
                        subchild_connector = "└── " if subchild_is_last else "├── "
                        subchild_label = tree_label.get(subchild, "")

                        if subchild_label:
                            lines.append(f"{subchild_prefix}{subchild_connector}{subchild}（{subchild_label}）")
                        else:
                            lines.append(f"{subchild_prefix}{subchild_connector}{subchild}")

                        # 检查子目录的叶子文件
                        subchild_leaf_files = [
                            f
                            for f in operator_files
                            if f["file_path"].startswith(f"{subchild}/")
                            and f["file_path"].count("/") == subchild.count("/") + 1
                        ]

                        for leaf_idx, leaf in enumerate(subchild_leaf_files):
                            leaf_is_last = leaf_idx == len(subchild_leaf_files) - 1
                            leaf_connector = "└── " if leaf_is_last else "├── "
                            lines.append(f"{subchild_prefix}│   {leaf_connector}{leaf['filename']}")

                        # 进一步递归
                        # 检查 subchild 是否有子节点
                        if subchild in tree:
                            # 添加到栈中继续处理
                            new_sub_prefix = subchild_prefix + ("    " if subchild_is_last else "│   ")
                            for m, sub2child in enumerate(reversed(tree[subchild])):
                                sub2child_is_last = m == len(tree[subchild]) - 1
                                sub2child_label = tree_label.get(sub2child, "")
                                sub2_branch = "└── " if sub2child_is_last else "├── "

                                if sub2child_label:
                                    lines.append(f"{new_sub_prefix}{sub2_branch}{sub2child}（{sub2child_label}）")
                                else:
                                    lines.append(f"{new_sub_prefix}{sub2_branch}{sub2child}")

                                # 检查子2目录的叶子文件
                                sub2child_leaf_files = [
                                    f
                                    for f in operator_files
                                    if f["file_path"].startswith(f"{sub2child}/")
                                    and f["file_path"].count("/") == sub2child.count("/") + 1
                                ]

                                for n, leaf in enumerate(sub2child_leaf_files):
                                    leaf_is_last = n == len(sub2child_leaf_files) - 1
                                    lines.append(
                                        f"{new_sub_prefix}│   {'└── ' if leaf_is_last else '├── '}{leaf['filename']}"
                                    )

                                # 继续下一层...
                                if sub2child in tree:
                                    new_sub2_prefix = new_sub_prefix + ("    " if sub2child_is_last else "│   ")
                                    for sub3_idx, sub3child in enumerate(reversed(tree[sub2child])):
                                        sub3child_is_last = sub3_idx == len(tree[sub2child]) - 1
                                        sub3child_label = tree_label.get(sub3child, "")
                                        sub3_branch = "└── " if sub3child_is_last else "├── "

                                        if sub3child_label:
                                            lines.append(
                                                f"{new_sub2_prefix}{sub3_branch}{sub3child}（{sub3child_label}）"
                                            )
                                        else:
                                            lines.append(f"{new_sub2_prefix}{sub3_branch}{sub3child}")

                                        # 检查子3目录的叶子文件
                                        sub3child_leaf_files = [
                                            f
                                            for f in operator_files
                                            if f["file_path"].startswith(f"{sub3child}/")
                                            and f["file_path"].count("/") == sub3child.count("/") + 1
                                        ]

                                        for p, leaf in enumerate(sub3child_leaf_files):
                                            leaf_is_last = p == len(sub3child_leaf_files) - 1
                                            leaf_branch = "└── " if leaf_is_last else "├── "
                                            lines.append(f"{new_sub2_prefix}│   {leaf_branch}{leaf['filename']}")

                                        # 继续下一层...
                                        if sub3child in tree:
                                            new_sub3_prefix = new_sub2_prefix + (
                                                "    " if sub3child_is_last else "│   "
                                            )
                                            for q, sub4child in enumerate(reversed(tree[sub3child])):
                                                sub4child_is_last = q == len(tree[sub3child]) - 1
                                                sub4child_label = tree_label.get(sub4child, "")
                                                sub4_branch = "└── " if sub4child_is_last else "├── "

                                                if sub4child_label:
                                                    lines.append(
                                                        f"{new_sub3_prefix}{sub4_branch}{sub4child}（{sub4child_label}）"
                                                    )
                                                else:
                                                    lines.append(f"{new_sub3_prefix}{sub4_branch}{sub4child}")

                                                # 检查子4目录的叶子文件
                                                sub4child_leaf_files = [
                                                    f
                                                    for f in operator_files
                                                    if f["file_path"].startswith(f"{sub4child}/")
                                                    and f["file_path"].count("/") == sub4child.count("/") + 1
                                                ]

                                                for r, leaf in enumerate(sub4child_leaf_files):
                                                    leaf_is_last = r == len(sub4child_leaf_files) - 1
                                                    leaf_branch = "└── " if leaf_is_last else "├── "
                                                    lines.append(
                                                        f"{new_sub3_prefix}│   {leaf_branch}{leaf['filename']}"
                                                    )

                                                # 继续下一层...
                                                if sub4child in tree:
                                                    new_sub4_prefix = new_sub3_prefix + (
                                                        "    " if sub4child_is_last else "│   "
                                                    )
                                                    for s, sub5child in enumerate(reversed(tree[sub4child])):
                                                        sub5child_is_last = s == len(tree[sub4child]) - 1
                                                        sub5child_label = tree_label.get(sub5child, "")
                                                        sub5_branch = "└── " if sub5child_is_last else "├── "

                                                        if sub5child_label:
                                                            sub5_line = f"{sub5_branch}{sub5child}（{sub5child_label}）"
                                                            lines.append(f"{new_sub4_prefix}{sub5_line}")
                                                        else:
                                                            lines.append(f"{new_sub4_prefix}{sub5_branch}{sub5child}")

                                                        # 检查子5目录的叶子文件
                                                        sub5child_leaf_files = [
                                                            f
                                                            for f in operator_files
                                                            if f["file_path"].startswith(f"{sub5child}/")
                                                            and f["file_path"].count("/") == sub5child.count("/") + 1
                                                        ]

                                                        for t, leaf in enumerate(sub5child_leaf_files):
                                                            leaf_is_last = t == len(sub5child_leaf_files) - 1
                                                            leaf_branch = "└── " if leaf_is_last else "├── "
                                                            lines.append(
                                                                f"{new_sub4_prefix}│   {leaf_branch}{leaf['filename']}"
                                                            )

    return "\n".join(lines)


def build_directory_tree_v2(sorted_dirs, dir_desc):
    """
    改进的目录树构建函数
    """
    tree = defaultdict(list)
    tree_label = {}

    if not sorted_dirs:
        return dict(tree), tree_label

    # 首先添加所有节点
    for dir_path in sorted_dirs:
        tree[dir_path] = []
        tree_label[dir_path] = dir_desc.get(dir_path, "")

    # 建立父子关系
    for dir_path in sorted_dirs[1:]:  # 跳过根节点
        parts = dir_path.split("/")
        parent = "/".join(parts[0:-1])

        if parent in tree:
            if dir_path not in tree.get(parent, []):
                tree[parent].append(dir_path)

    return dict(tree), tree_label


def get_ordered_operator_files(operator_files):
    """按路径排序算子文件列表"""
    return sorted(operator_files, key=lambda x: x["file_path"])


def generate_document(
    operator,
    leaf_info,
    operator_root_desc,
    sorted_dirs,
    tree_label,
    operator_files,
    operator_root_dir,
):
    """
    为单个叶子节点生成完整的Markdown文档

    Args:
        operator: 算子名称
        leaf_info: 叶子节点信息字典
        operator_root_desc: 算子根目录描述
        sorted_dirs: 排序后的目录列表
        tree_label: 树节点标签
        operator_files: 该算子的所有文件列表

    Returns:
        markdown_content: 生成的Markdown文档内容
    """
    leaf_path = leaf_info["file_path"]
    leaf_info["filename"]
    # 从绝对路径提取算子目录内部的相对目录路径
    relative_within = extract_relative_within_operator(leaf_path, operator)
    parent_dir = str(Path(relative_within).parent)
    leaf_dir_path = operator_root_dir + "/" + parent_dir if parent_dir and parent_dir != "." else operator_root_dir

    # 获取叶子所在目录的标签
    tree_label.get(leaf_dir_path, "")

    # 构建目录树（只包含到该叶子节点为止的路径）
    # 实际上我们需要完整的树，只是高亮到叶子节点

    # 使用完整树
    tree = defaultdict(list)
    for dir_path in sorted_dirs:
        if dir_path not in tree:
            tree[dir_path] = []
        tree_label[dir_path] = tree_label.get(dir_path, "")

    # 建立父子关系
    for dir_path in sorted_dirs[1:]:
        parts = dir_path.split("/")
        parent = "/".join(parts[0:-1])
        if parent in tree and dir_path not in tree[parent]:
            tree[parent].append(dir_path)

    # 格式化树
    tree_lines = []
    root = sorted_dirs[0]
    root_label = tree_label.get(root, "")
    if root_label:
        tree_lines.append(f"{root}（{root_label}）")
    else:
        tree_lines.append(root)

    def print_tree_recursive(node, prefix="", is_last=True):
        if node not in tree:
            return

        children = sorted(tree[node])
        for i, child in enumerate(children):
            child_is_last = i == len(children) - 1
            connector = "└── " if child_is_last else "├── "

            child_label = tree_label.get(child, "")
            if child_label:
                tree_lines.append(f"{prefix}{connector}{child}（{child_label}）")
            else:
                tree_lines.append(f"{prefix}{connector}{child}")

            # 检查该目录是否有叶子文件
            leaf_files = [
                f
                for f in operator_files
                if f["file_path"].startswith(f"{child}/") and f["file_path"].count("/") == child.count("/") + 1
            ]

            for j, leaf in enumerate(leaf_files):
                leaf_is_last = j == len(leaf_files) - 1
                leaf_connector = "└── " if leaf_is_last else "├── "
                tree_lines.append(f"{prefix}│   {leaf_connector}{leaf['filename']}")

            # 递归
            new_prefix = prefix + ("    " if child_is_last else "│   ")
            print_tree_recursive(child, new_prefix, child_is_last)

    print_tree_recursive(root)

    tree_markdown = "\n".join(tree_lines)

    # 生成文件概览：只显示当前文件及其上层目录路径
    file_overview_parts = [
        f"- 当前文件路径: `{leaf_path}`",
        f"- 所属目录: `{leaf_dir_path}`",
    ]
    file_overview_markdown = "\n".join(file_overview_parts)

    # 获取该文件的详细说明
    leaf_detail = leaf_info["response"]
    leaf_detail = "\n#### ".join([item.strip() for item in leaf_detail.split("##")])

    # 构建文档
    markdown_content = f"""# {operator} 算子代码说明

## 一、算子整体说明

{operator_root_desc}

## 二、代码整体目录及说明

{tree_markdown}

## 三、代码文件概览

{file_overview_markdown}

### 详细说明
{leaf_detail}
"""

    return markdown_content


def escape_filename(filename):
    """
    清理文件名，移除特殊字符
    """
    # 替换路径分隔符为下划线（兼容 Windows/Linux）
    filename = filename.replace("/", "_").replace("\\", "_")
    return filename


def main(out_path):
    # 设置路径（均指向项目根目录下的 data/）
    llm_dir = out_path / "llm_responses"
    fs_dir = out_path / "file_tree_structure"
    output_dir = out_path / "organized_docs"

    # 创建输出目录
    output_dir.mkdir(exist_ok=True)

    info("=" * 60)
    info("算子文档生成工具")
    info("=" * 60)

    # 1. 加载目录描述
    info(f"[1/4] 加载目录描述从 {llm_dir}...")
    dir_desc = load_llm_responses(llm_dir)
    info(f"     加载了 {len(dir_desc)} 个目录描述")

    # 2. 加载文件描述
    info(f"[2/4] 加载文件描述从 {fs_dir}...")
    files_by_operator, all_files = load_file_descriptions(fs_dir)
    info(f"     加载了 {len(all_files)} 个文件描述，涉及 {len(files_by_operator)} 个算子")

    # 3. 生成文档
    info(f"[3/4] 生成文档到 {output_dir}...")
    total_docs = 0

    # 用于跟踪每个(operator, filename)对的计数器，处理重复
    file_count = defaultdict(int)

    for operator, operator_files in files_by_operator.items():
        # 从文件数据中获取 operator_root_dir
        operator_root_dir = ""
        for f in operator_files:
            ord_val = f.get("operator_root_dir", "")
            if ord_val:
                operator_root_dir = ord_val
                break
        # 获取算子根目录描述（用 operator_root_dir 替代硬编码 math/）
        operator_root_path = operator_root_dir if operator_root_dir else operator
        operator_root_desc = dir_desc.get(operator_root_path, "")

        # 构建该算子的完整目录树
        sorted_dirs = build_dirs_for_operator(operator, operator_root_path, all_files)
        tree_label = {d: dir_desc.get(d, "") for d in sorted_dirs}

        if not operator_root_desc and sorted_dirs:
            operator_root_desc = tree_label.get(sorted_dirs[0], "")

        # 为每个叶子节点生成文档
        for leaf_info in operator_files:
            doc_content = generate_document(
                operator=operator,
                leaf_info=leaf_info,
                operator_root_desc=operator_root_desc,
                sorted_dirs=sorted_dirs,
                tree_label=tree_label,
                operator_files=operator_files,
                operator_root_dir=operator_root_path,
            )

            # 保存文档
            leaf_filename = leaf_info["filename"]
            safe_filename = escape_filename(leaf_filename)

            # 处理重复文件名
            file_key = f"{operator}_{safe_filename}"
            count = file_count.get(file_key, 0)
            file_count[file_key] = count + 1

            if count > 0:
                output_filename = f"{operator}_{safe_filename}_{count}.md"
            else:
                output_filename = f"{operator}_{safe_filename}.md"

            output_path = output_dir / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(doc_content)

            total_docs += 1

            if total_docs % 100 == 0:
                info(f"     已生成 {total_docs} 个文档...")

    # 4. 完成
    info("[4/4] 完成！")
    info("生成统计:")
    info(f"  - 算子数量: {len(files_by_operator)}")
    info(f"  - 文档总数: {total_docs}")
    info(f"  - 输出目录: {output_dir}")
    info("=" * 60)


if __name__ == "__main__":
    main()
