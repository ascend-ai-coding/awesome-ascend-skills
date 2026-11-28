#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Python脚本：算子文档整理
功能：将organized_docs中的文档根据directory_trees进行映射整理，生成算子整体说明文档
"""

import os
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.logger import info, debug, warning, error


def load_directory_trees(
    directory_trees_dir: Path,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    加载directory_trees中的所有JSON文件，构建文件名到代码描述的映射

    Args:
        directory_trees_dir: directory_trees文件夹路径

    Returns:
        (文件名到描述的字典, 操作符名称到目录路径的字典)
    """
    file_to_description = {}
    operator_to_tree_path = {}

    # 遍历directory_trees文件夹
    for root, dirs, files in os.walk(directory_trees_dir):
        for file in files:
            if file.endswith("_tree.json"):
                json_path = Path(root) / file
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # 提取操作符名称 (从JSON中的operator字段或directory_name字段)
                    operator_name = data.get("operator") or data.get(
                        "directory_name", ""
                    )
                    if operator_name and operator_name not in operator_to_tree_path:
                        operator_to_tree_path[operator_name] = str(json_path)

                    # 递归提取文件描述，传递操作符名称
                    extract_file_descriptions(
                        data.get("file_tree", {}), file_to_description, operator_name
                    )

                except Exception as e:
                    warning(f"Failed to load {json_path}: {e}")

    return file_to_description, operator_to_tree_path


def extract_file_descriptions(
    tree: dict, mapping: Dict[str, str], operator: str = ""
) -> None:
    """
    从directory trees结构中递归提取文件的描述

    Args:
        tree: JSON中的file_tree结构
        mapping: 将填充的映射字典
        operator: 当前操作符名称（从JSON operator字段获取）
    """
    if isinstance(tree, dict):
        for key, value in tree.items():
            if isinstance(value, dict):
                # 检查是否是文件节点（包含response字段）
                if "response" in value:
                    # 提取文件名和描述
                    file_path = value.get("file_path", "")
                    response = value.get("response", "")
                    file_operator = value.get("operator", "") or operator

                    # 从绝对路径中提取文件名
                    if file_path:
                        normalized_path = Path(file_path).as_posix()
                        file_name = Path(normalized_path).name
                        if file_name:
                            # 提取"代码功能"部分
                            code_desc = extract_code_function_description(response)
                            # 使用操作符名+文件名作为key来避免冲突
                            key = f"{file_operator}:{file_name}"
                            mapping[key] = code_desc
                else:
                    # 递归处理子目录
                    extract_file_descriptions(value, mapping, operator)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        extract_file_descriptions(item, mapping, operator)


def extract_code_function_description(response: str) -> str:
    """
    从response字段中提取"代码功能"部分描述

    Args:
        response: 完整的markdown响应文本

    Returns:
        提取的代码功能描述
    """
    # 查找"## 代码功能"部分
    match = re.search(r"##\s*代码功能\s*\n(.*?)(?=##|$)", response, re.DOTALL)
    if match:
        desc = match.group(1).strip()
        # 限制长度，提取主要描述
        # 只取第一句话或前500个字符
        if "。" in desc:
            desc = desc.split("。")[0] + "。"
        elif "\n" in desc:
            desc = desc.split("\n")[0]
        elif len(desc) > 500:
            desc = desc[:500] + "..."
        return desc
    return response[:200] + "..." if len(response) > 200 else response


def extract_operator_from_filename(
    filename: str, first_doc_path: Path
) -> Optional[str]:
    """
    从文件名中提取算子名称

    Args:
        filename: 文件名 (如 "leaky_relu_grad_aclnn_leaky_relu_backward.cpp.md")
        first_doc_path: 第一个文档文件路径（用于从文档内容中提取操作符名）

    Returns:
        算子名称 (如 "leaky_relu_grad")
    """
    # 首先尝试从文档内容中读取标题
    if first_doc_path and first_doc_path.exists():
        try:
            with open(first_doc_path, "r", encoding="utf-8") as f:
                content = f.read()
                # 提取标题 "# {operator_name} 算子代码说明"
                match = re.search(r"^#\s+(\S+)\s+算子代码说明", content, re.MULTILINE)
                if match:
                    return match.group(1)
        except:
            pass

    # 备选方案：从文件名推断
    # 文件名格式: {operator_name}_{file_path}.md
    if filename.endswith(".md"):
        filename = filename[:-3]

    # 分析：操作符名到文件名的分割可以根据以下规则
    # 寻找最后出现的.cpp、.h等扩展名
    # 检查每一部分是否看起来像文件路径片段
    parts = filename.split("_")

    # 从后往前找最后一个有效的文件扩展名
    for i in range(len(parts) - 1, 0, -1):
        part = parts[i]
        # 如果这个部分包含.cpp、.h等扩展名（部分匹配）
        # 或者如果这个部分很短且看起来像扩展名（如 'cpp', 'h'）
        if i == len(parts) - 1 and "." in part:
            # 这是文件名部分（带扩展名）
            # 操作符名是前面的部分
            return "_".join(parts[:i])
        elif i == len(parts) - 1 and part in ["cpp", "h", "hpp", "cu", "c"]:
            # 这是扩展名部分
            return "_".join(parts[:i])

    # 如果没找到，尝试白名单方式
    return None


def parse_directory_tree(content: str) -> Tuple[List[str], int]:
    """
    解析目录树内容，解析每一行并返回(行列表, 用作缩进的空格数)

    Args:
        content: 目录树内容字符串

    Returns:
        (解析后的行列表, 缩进空格数)
    """
    lines = content.strip().split("\n")
    if not lines:
        return [], 0

    # 计算最小的缩进空格数（通常是2）
    indent_spaces = 2  # 默认值
    for line in lines if lines else []:
        stripped = line.lstrip()
        if stripped and stripped.startswith(("├", "└", "│")):
            indent_spaces = len(line) - len(stripped.lstrip("├─│"))
            break

    return lines, indent_spaces


def get_tree_level(line: str, indent_size: int = 2) -> int:
    """
    计算目录树行的层级

    Args:
        line: 目录树行
        indent_size: 每层缩进的空格数

    Returns:
        层级数 (0是根目录)
    """
    stripped = line.lstrip()
    if not stripped:
        return -1

    # 计算前导空格数（排除特殊字符）
    leading_space = len(line) - len(line.lstrip(" ├│\n\t\r"))
    if stripped.startswith(("├", "└")):
        return (leading_space // indent_size) + 1
    elif stripped.startswith("│"):
        return leading_space // indent_size + 1
    elif stripped.startswith(" ") and not stripped.startswith(("├", "└", "│")):
        # 可能是目录路径行
        if "（" in stripped and "）" in stripped:
            # 检测到目录说明，比如 "activation/leaky_relu_grad（## ACLNN接口实现）"
            # 计算层级
            path_part = stripped.split("（")[0].strip()
            # 计算路径的层级
            return path_part.count("/") + 1
    return 0


def enhance_tree_with_descriptions(
    tree_content: str, file_to_desc: Dict[str, str], operator: str = ""
) -> str:
    """
    为目录树中的每个文件添加功能描述

    Args:
        tree_content: 原始目录树内容
        file_to_desc: 文件名到描述的映射 (格式: {operator:filename -> description})
        operator: 当前操作符名称

    Returns:
        增强后的目录树内容
    """
    lines = tree_content.strip().split("\n")
    if not lines:
        return tree_content

    enhanced_lines = []

    for line in lines:
        enhanced_line = line

        # 检查是否是文件行（包含文件扩展名）
        # 跳过包含说明如（## ...）的行
        if "（" not in line and any(
            ext in line for ext in [".cpp", ".h", ".cu", ".hpp"]
        ):
            # 提取文件名
            # 1. 先找到├或└或│后的部分
            stripped = line.lstrip("│ \t\r")
            if stripped.startswith(("├──", "└──")):
                file_part = stripped[3:]  # 去掉 ├──
                # 提取纯文件名（去掉目录路径）
                if "/" in file_part:
                    file_name = Path(file_part).name
                else:
                    file_name = file_part.strip()

                # 如果找到描述，添加到行尾
                if file_name:
                    # 使用操作符名:文件名的格式查找
                    key = f"{operator}:{file_name}"
                    if key in file_to_desc:
                        desc = file_to_desc[key]
                        enhanced_line = line + " " + desc
                    elif file_name in file_to_desc:
                        # 降级查找（兼容性）
                        desc = file_to_desc[file_name]
                        enhanced_line = line + " " + desc

        enhanced_lines.append(enhanced_line)

    return "\n".join(enhanced_lines)


def create_operator_summary(
    operator_name: str,
    representative_file: str,
    organized_dir: Path,
    output_dir: Path,
    file_to_desc: Dict[str, str],
) -> None:
    """
    为每个算子创建整体说明文档

    Args:
        operator_name: 算子名称
        representative_file: 代表文件路径
        organized_dir: organized_docs目录
        output_dir: 输出目录
        file_to_desc: 文件描述映射
    """
    # 读取代表文件
    input_path = organized_dir / representative_file
    if not input_path.exists():
        warning(f"Representative file not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 提取文档的各个部分
    parts = extract_document_sections(content)

    # 构建新文档
    output_content = []

    # 标题
    output_content.append(f"# {operator_name} 算子代码说明\n")

    # 一、算子整体说明
    if "overall_description" in parts:
        output_content.append("## 一、算子整体说明\n")
        output_content.append(parts["overall_description"])
        output_content.append("\n")
    else:
        output_content.append("## 一、算子整体说明\n\n")

    # 二、代码整体目录及说明（增强版本）
    if "directory_tree" in parts:
        output_content.append("## 二、代码整体目录及说明\n")
        enhanced_tree = enhance_tree_with_descriptions(
            parts["directory_tree"], file_to_desc, operator_name
        )
        output_content.append(enhanced_tree)
        output_content.append("\n")

    # 保存文件
    output_path = output_dir / f"{operator_name}算子说明.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(output_content)

    info(f"Created: {output_path}")


def extract_document_sections(content: str) -> Dict[str, str]:
    """
    从文档中提取各个部分

    Args:
        content: 文档内容

    Returns:
        包含各部分的字典:
        - title: 标题
        - overall_description: 算子整体说明
        - directory_tree: 目录树
        - code_overview: 代码概览
    """
    parts = {}

    lines = content.split("\n")

    # 状态变量
    current_section = None
    section_content = []

    for line in lines:
        # 检测章节标题
        if re.match(r"^##\s+", line):
            # 如果之前有章节，保存内容
            if current_section and section_content:
                parts[current_section] = "\n".join(section_content)

            # 开始新章节
            if "一、算子整体说明" in line or "算子整体说明" in line:
                current_section = "overall_description"
            elif "二、代码整体目录及说明" in line or "代码整体目录" in line:
                current_section = "directory_tree"
            elif "三、代码文件概览" in line or "代码文件概览" in line:
                current_section = "code_overview"
            else:
                current_section = None

            section_content = []
        elif current_section:
            section_content.append(line)

    # 保存最后一个章节
    if current_section and section_content:
        parts[current_section] = "\n".join(section_content)

    return parts


def main(out_path):
    """主函数"""
    # 设置路径（均指向项目根目录下的 data/）
    organized_dir = out_path / "organized_docs"
    directory_trees_dir = out_path / "directory_trees"
    output_dir = out_path / "organized_docs_with_descriptions"

    # 1. 创建输出目录及子文件夹
    output_dir.mkdir(exist_ok=True)
    operator_summary_dir = output_dir / "算子说明"
    original_docs_dir = output_dir / "原始文档"
    operator_summary_dir.mkdir(exist_ok=True)
    original_docs_dir.mkdir(exist_ok=True)
    info(f"输出目录: {output_dir}")
    info(f"算子说明目录: {operator_summary_dir}")
    info(f"原始文档目录: {original_docs_dir}")

    # 2. 加载directory_trees数据
    info("加载directory_trees数据...")
    file_to_desc, operator_to_tree = load_directory_trees(directory_trees_dir)
    info(f"加载了 {len(file_to_desc)} 个文件的描述")
    info(f"识别了 {len(operator_to_tree)} 个操作符目录")

    # 3. 扫描organized_docs，识别算子和代表文件
    info("扫描organized_docs...")
    operator_to_file = {}  # {operator_name: list_of_files}

    for file in sorted(organized_dir.glob("*.md")):
        # 首次发现的文件用于从文档内容中提取操作符名
        operator_name = extract_operator_from_filename(file.name, file)
        if operator_name:
            if operator_name not in operator_to_file:
                operator_to_file[operator_name] = []
            operator_to_file[operator_name].append(file.name)

    info(f"识别了 {len(operator_to_file)} 个算子")

    # 4. 生成算子整体说明文档
    info("生成算子整体说明文档...")
    for operator_name, files in sorted(operator_to_file.items()):
        # 保留第一个作为代表文件
        representative_file = files[0]
        create_operator_summary(
            operator_name,
            representative_file,
            organized_dir,
            operator_summary_dir,
            file_to_desc,
        )

    # 5. 复制所有原始文档到原始文档子文件夹
    info("复制原始文档...")
    copied_count = 0
    for file in organized_dir.glob("*.md"):
        dest = original_docs_dir / file.name
        shutil.copy2(file, dest)
        copied_count += 1

    info(f"复制了 {copied_count} 个原始文档到: {original_docs_dir}")
    info(f"完成！输出目录: {output_dir}")

    for file in sorted([p.name for p in Path(organized_dir).iterdir()]):
        debug(file)
        with open(Path(organized_dir) / file, "r", encoding="utf-8") as f:
            file_content = f.read()
        file_new_content = "##" + file_content.split("## 三、")[1]
        with open(Path(original_docs_dir) / file, "w", encoding="utf-8") as f:
            f.write(file_new_content)


if __name__ == "__main__":
    main()
