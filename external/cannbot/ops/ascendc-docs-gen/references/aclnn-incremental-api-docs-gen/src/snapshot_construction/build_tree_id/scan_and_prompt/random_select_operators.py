#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
扫描算子源码目录，读取所有文件内容并保存为 JSON。

核心改动：file_path 字段存储绝对路径，新增 operator_root_dir 字段。
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

from src.logger import info, warning

# 跳过的目录
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "build",
    "venv",
    ".venv",
    ".idea",
    "output",
    "build_out",
}

# 包含的文件扩展名
INCLUDE_EXTENSIONS = {
    # 源码
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".hxx",
    ".cu",
    ".cuh",
    # 配置
    ".json",
    ".ini",
    ".yaml",
    ".yml",
    ".cmake",
    # 文档
    ".md",
    ".txt",
    ".rst",
    # 脚本
    ".sh",
    ".py",
    ".bat",
    # CMake
    "CMakeLists.txt",
    # 其他
    ".cfg",
    ".conf",
    ".toml",
}

# 跳过的文件名
SKIP_FILES = {".DS_Store", "Thumbs.db"}


def should_include_file(filepath: Path) -> bool:
    """判断文件是否应包含在输出中。"""
    if filepath.name in SKIP_FILES:
        return False
    if (
        filepath.suffix.lower() in INCLUDE_EXTENSIONS
        or filepath.name == "CMakeLists.txt"
    ):
        return True
    return False


def read_file_content(
    filepath: Path, max_size: int = 10 * 1024 * 1024
) -> Optional[str]:
    """读取文件内容，超过大小限制时跳过。"""
    try:
        if filepath.stat().st_size > max_size:
            return f"[FILE TOO LARGE: {filepath.stat().st_size} bytes, skipping]"
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"[READ ERROR: {str(e)}]"


def scan_operator_files(operator_dir: Path) -> Dict[str, str]:
    """
    递归扫描算子目录中的所有文件。

    返回字典的 key 为文件绝对路径的 POSIX 格式字符串，
    value 为文件内容。
    """
    files = {}

    for root, dirs, filenames in os.walk(operator_dir):
        root_path = Path(root)

        # 过滤跳过的目录（原地修改 dirs 列表）
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in filenames:
            filepath = root_path / filename

            if should_include_file(filepath):
                # 使用绝对路径作为 key（不再做 relative_to）
                abs_path = str(filepath.resolve().as_posix())
                files[abs_path] = read_file_content(filepath)

    return files


def process_project_single(operator_path: Path) -> Dict:
    """处理单个算子目录，返回其数据字典。"""

    if not operator_path.exists():
        raise FileNotFoundError(f"算子目录不存在: {operator_path}")

    operators_data = []
    info(f"  处理: {operator_path.name}")
    files = scan_operator_files(operator_path)
    operators_data.append({"name": operator_path.name, "files": files})

    # operator_root_dir：算子根目录描述的 key，直接用算子名
    return {
        "operator_root_dir": operator_path.name,
        "selected_operators": operators_data,
    }


def main(operator_path: Path, out_path: Path):
    """主入口：扫描算子并保存为 JSON。"""
    out_path.mkdir(exist_ok=True)

    result = process_project_single(operator_path)

    output_file = out_path / "operators.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    info(f"已保存 {len(result['selected_operators'])} 个算子到 {output_file}")
    info(f"全部完成! JSON 文件保存到 {out_path}")


if __name__ == "__main__":
    main()
