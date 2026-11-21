#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
准备批量分析 prompt：从 operators.json 提取代码文件，生成统一的 batch_prompts.json。

核心改动：filepath 字段存储绝对路径。
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any

from src.logger import info, warning

# 输入文件名
INPUT_FILES = ["operators.json"]

# 源码扩展名
CODE_EXTENSIONS = {".cpp", ".cc", ".c", ".h", ".hpp", ".cuh", ".cu"}

# 只包含这些子目录下的文件
TARGET_FOLDERS = {"op_api", "op_graph", "op_host", "op_kernel"}


def is_code_file(filepath: str) -> bool:
    """判断文件是否为源码文件。"""
    return any(filepath.lower().endswith(ext) for ext in CODE_EXTENSIONS)


def is_in_target_folder(file_path: str) -> bool:
    """判断文件路径是否在 op_api/op_graph/op_host/op_kernel 下。

    file_path 现为绝对路径，如 D:/data/ops-math/math/abs/op_api/aclnn_abs.h。
    在 Path.parts 中查找目标文件夹名即可。
    """
    parts = Path(file_path).parts
    return any(folder in parts for folder in TARGET_FOLDERS)


def prepare_tasks(out_path) -> List[Dict[str, Any]]:
    """从 operators.json 中提取代码文件任务。"""
    tasks = []
    task_id = 0

    for filename in INPUT_FILES:
        filepath = out_path / filename
        if not filepath.exists():
            warning(f"{filename} 不存在，跳过")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 读取 operator_root_dir（新增字段）
        operator_root_dir = data.get("operator_root_dir", "")
        operators = data.get("selected_operators", [])

        info(f"处理 {filename}: {len(operators)} 个算子")

        for operator in operators:
            op_name = operator.get("name", "unknown")
            files = operator.get("files", {})

            code_files_count = 0
            for file_path, content in files.items():
                if (
                    is_code_file(file_path)
                    and content
                    and is_in_target_folder(file_path)
                ):
                    tasks.append(
                        {
                            "task_id": task_id,
                            "operator_name": op_name,
                            "operator_root_dir": operator_root_dir,
                            "filepath": file_path,
                            "code": content,
                            "file_size_chars": len(content),
                        }
                    )
                    task_id += 1
                    code_files_count += 1

            info(f"  - {op_name}: {code_files_count} 个代码文件")

    return tasks


def save_tasks(out_path, tasks: List[Dict[str, Any]]):
    """Save tasks to JSON file for reference."""
    tasks_file = out_path / "code_file_tasks.json"
    with open(tasks_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    info(f"Saved {len(tasks)} tasks to {tasks_file}")


def generate_prompt_json(prompt_dir, tasks: List[Dict[str, Any]]):
    """生成统一的 JSON 文件，包含所有任务的元数据和代码内容，以 task_id 为 key。"""
    prompt_dir.mkdir(exist_ok=True)

    generated_time = time.strftime("%Y-%m-%d %H:%M:%S")
    prompts_dict = {}
    for task in tasks:
        tid = str(task["task_id"])
        prompts_dict[tid] = {
            "task_id": task["task_id"],
            "operator_name": task["operator_name"],
            "operator_root_dir": task["operator_root_dir"],
            "filepath": task["filepath"],
            "code_size": task["file_size_chars"],
            "generated": generated_time,
            "code": task["code"],
        }

    output_file = prompt_dir / "batch_prompts.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"prompts": prompts_dict}, f, ensure_ascii=False, indent=2)

    return len(prompts_dict)


def main(out_path: Path):
    """Main function to prepare batch analysis."""

    prompt_dir = out_path / "batch_prompts_by_file"
    results_dir = out_path / "analysis_results"
    prompt_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    info("=" * 80)
    info("Preparing Code File Analysis Prompts (Unified JSON)")
    info("=" * 80)

    tasks = prepare_tasks(out_path)

    if not tasks:
        warning("No tasks to process!")
        return

    info(f"Total code files to process: {len(tasks)}")

    size_stats = {
        "small": sum(1 for t in tasks if t["file_size_chars"] < 10000),
        "medium": sum(1 for t in tasks if 10000 <= t["file_size_chars"] < 50000),
        "large": sum(1 for t in tasks if t["file_size_chars"] >= 50000),
    }
    info(f"  - Small files (<10KB): {size_stats['small']}")
    info(f"  - Medium files (10-50KB): {size_stats['medium']}")
    info(f"  - Large files (>=50KB): {size_stats['large']}")

    save_tasks(out_path, tasks)

    info("Generating unified prompt JSON...")
    created = generate_prompt_json(prompt_dir, tasks)

    info("=" * 80)
    info(
        f"PROMPT PREPARATION COMPLETE: {created} prompts in {prompt_dir / 'batch_prompts.json'}"
    )
    info("=" * 80)
    info(f"Prompt data saved to: {prompt_dir / 'batch_prompts.json'}")
    info(f"Results will be saved to: {results_dir}")


if __name__ == "__main__":
    main()
