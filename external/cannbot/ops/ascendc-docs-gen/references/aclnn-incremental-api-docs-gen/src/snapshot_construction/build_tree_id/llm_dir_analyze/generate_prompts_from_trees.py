#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
汇总所有目录信息，生成统一的 prompt 文件，一次性请求 LLM 总结每个目录的信息。

原有方式：每个目录独立生成一个 prompt，分别调用 LLM，效率低。
当前方式：所有目录汇总到一个 prompt，一次 LLM 调用返回所有目录的说明。
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List

from src.logger import info, debug, warning, error


def extract_sections_from_response(response: str) -> Dict[str, str]:
    """
    从 response 中提取各个部分的内容

    Args:
        response: 完整的响应字符串

    Returns:
        包含代码功能、关键结构、关键函数的字典
    """
    sections = {"code_function": "", "key_structures": "", "key_functions": ""}

    try:
        # 提取代码功能
        code_function_match = re.search(
            r"## 代码功能\n(.*?)(?=## 关键结构|## 关键函数|$)", response, re.DOTALL
        )
        if code_function_match:
            content = code_function_match.group(1).strip()
            # 处理多行：移除行首的 "- " 或换行，合并为单行
            sections["code_function"] = re.sub(r"\s*-\s*", "；", content).replace(
                "\n", " "
            )

        # 提取关键结构
        key_structures_match = re.search(
            r"## 关键结构\n(.*?)(?=## 关键函数|$)", response, re.DOTALL
        )
        if key_structures_match:
            content = key_structures_match.group(1).strip()
            sections["key_structures"] = re.sub(r"\s*-\s*", "；", content).replace(
                "\n", " "
            )

        # 提取关键函数
        key_functions_match = re.search(r"## 关键函数\n(.*?)$", response, re.DOTALL)
        if key_functions_match:
            content = key_functions_match.group(1).strip()
            sections["key_functions"] = re.sub(r"\s*-\s*", "；", content).replace(
                "\n", " "
            )
    except Exception as e:
        error(f"提取 sections 时出错: {e}")

    return sections


def extract_leaf_nodes(
    file_tree: Dict[str, Any], output_list: List[Dict[str, Any]], prefix: str = ""
) -> None:
    """
    递归提取叶子节点信息

    Args:
        file_tree: 文件树字典结构
        output_list: 输出列表，用于存储提取的叶子节点信息
        prefix: 路径前缀（用于显示相对路径）
    """
    for key, value in file_tree.items():
        # 如果是叶子节点（值包含 response 字段）
        if isinstance(value, dict) and "response" in value and "file_path" in value:
            sections = extract_sections_from_response(value["response"])

            output_list.append(
                {
                    "filename": key,
                    "file_path": value.get("file_path", ""),
                    "code_function": sections["code_function"],
                    "key_structures": sections["key_structures"],
                    "key_functions": sections["key_functions"],
                }
            )
        # 如果是目录节点（值是字典但不包含 response 字段）
        elif isinstance(value, dict):
            extract_leaf_nodes(value, output_list, f"{prefix}{key}/")


def main(out_path):
    """主函数：汇总所有目录信息，生成一个组合 prompt"""
    directory_trees_dir = out_path / "directory_trees"
    output_dir = out_path / "new_prompts"
    output_dir.mkdir(exist_ok=True)

    # 查找所有 _tree.json 文件
    json_files = sorted(directory_trees_dir.glob("**/*.json"))

    if not json_files:
        error(f"错误: 在 {directory_trees_dir} 中未找到任何 JSON 文件")
        return

    info(f"找到 {len(json_files)} 个目录描述文件")

    # 构建每个目录的信息块
    dir_sections = []

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        directory_name = data.get("directory_name", "")
        file_tree = data.get("file_tree", {})

        if not directory_name or not file_tree:
            warning(f"警告: {json_file} 缺少必要字段，跳过")
            continue

        # 相对于 directory_trees 的路径（用于 Step 4 的目录路径映射）
        rel_dir_path = str(json_file.parent.relative_to(directory_trees_dir))

        # 提取叶子节点信息（来自 Step 2 的 LLM 分析结果）
        leaf_nodes = []
        extract_leaf_nodes(file_tree, leaf_nodes)

        dir_sections.append(
            {
                "rel_path": rel_dir_path,
                "dir_name": directory_name,
                "leaf_nodes": leaf_nodes,
            }
        )

    if not dir_sections:
        error("错误: 没有有效的目录信息可处理")
        return

    # 构建组合 prompt（精简版：去掉了冗余的文件树和关键结构/函数，只保留文件名+功能）
    lines = [
        "你是一名算子代码架构分析专家。",
        "以下列出算子各文件夹及所含文件的功能，请对每个文件夹生成一条职责说明。",
        "",
        "要求：",
        "- 中文，每条不超过80字",
        "- 说明目录职责，不解释具体实现",
        "- 内容如：算子API入口、ACLNN接口实现等",
        "",
        '以纯JSON返回，key为目录相对路径，value为"## "+说明：',
        '{"路径A": "## 说明", "路径B": "## 说明"}',
        "",
        "---目录信息---",
        "",
    ]

    for sec in dir_sections:
        # 每个目录一行标题 + 每文件一行"文件名: 功能"
        lines.append(f"【{sec['dir_name']}】{sec['rel_path']}")
        for node in sec["leaf_nodes"]:
            desc = (
                node["code_function"][:120]
                if len(node["code_function"]) > 120
                else node["code_function"]
            )
            lines.append(f"  {node['filename']}: {desc}")
        lines.append("")

    combined_prompt = "\n".join(lines)

    # 保存组合 prompt
    output_file = output_dir / "combined_directory_prompt.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_prompt)

    info(f"生成组合 prompt: {output_file}")
    info(f"共包含 {len(dir_sections)} 个目录")
    info(f"prompt 总长度: {len(combined_prompt)} 字符")


if __name__ == "__main__":
    main()
