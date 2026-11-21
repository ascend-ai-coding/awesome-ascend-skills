#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
统一入口脚本：串联 scan→analyze_file→analyze_dir→generate→enrich 的完整算子文档生成流水线
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Optional
from src.logger import info, debug, warning, error, section as log_section

# 导入各步骤模块
from src.snapshot_construction.build_tree_id.scan_and_prompt import (
    random_select_operators,
    prepare_code_analysis_batch,
)
from src.snapshot_construction.build_tree_id.llm_file_analyze import llm_processor
from src.snapshot_construction.build_tree_id.llm_dir_analyze import (
    build_file_tree,
    generate_directory_trees,
    generate_prompts_from_trees,
    send_prompts_to_llm,
)
from src.snapshot_construction.build_tree_id.generate_docs import generate_operator_docs
from src.snapshot_construction.build_tree_id.enrich_docs import (
    add_aclnn_h_description,
    organize_operator_docs,
)
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

console = Console()


def _step_separator(step_name: str) -> None:
    info("=" * 70)
    info(f"  {step_name}")
    info("=" * 70)
    console.print(f"        [dim]{step_name}[/dim]")


def run_step_1(operator_path: Path, out_path: Path) -> None:
    """Step 1: 随机选取算子并生成分析 prompt 文件 (1a -> 1b)"""
    info("Step 1: select_operators")

    # 1a: random_select_operators.py — 从 ops 仓随机选取算子，输出 JSON
    info("[1a] random_select_operators.py — 随机选取算子")
    random_select_operators.main(operator_path, out_path)

    # 1b: prepare_code_analysis_batch.py — 基于 JSON 生成统一的 prompt JSON
    info("[1b] prepare_code_analysis_batch.py — 生成统一 prompt JSON")
    debug(f"  输入: {out_path}/<project>-operators.json")
    debug(f"  输出: {out_path}/batch_prompts_by_file/batch_prompts.json")
    debug(f"        {out_path}/code_file_tasks.json")
    prepare_code_analysis_batch.main(out_path)

    info("[Step 1 完成]")


def run_step_2(operator_path, out_path) -> None:
    """Step 2: 文件级信息提取 - 对 ops 源码中的每个代码文件调用 LLM API"""
    info("Step 2: level_1_info_extract")
    info("调用 llm_processor.main() ...")
    llm_processor.main(operator_path, out_path)
    info("[Step 2 完成]")


def run_step_3(out_path) -> None:
    """Step 3: 目录级信息提取 (3a -> 3b -> 3c -> 3d)"""
    info("Step 3: file_level_info_extract")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        disable=not sys.stdout.isatty(),
    ) as progress:
        task = progress.add_task("      处理进度", total=4)

        # 3a: build_file_tree.py — 将 step_2 输出复制为 file_tree_structure/
        info("[3a] build_file_tree.py — 构建文件树结构")
        debug(f"  输入: {out_path / 'results_by_file'}")
        debug(f"  输出: {out_path / 'file_tree_structure'}")
        build_file_tree.main(out_path)
        progress.update(
            task, advance=1, description="      处理进度 [dim][构造文件树][/dim]"
        )

        # 3b: generate_directory_trees.py — 为每层目录生成 _tree.json
        info("[3b] generate_directory_trees.py — 生成目录描述文件")
        debug(f"  输入: {out_path / 'file_tree_structure'}")
        debug(f"  输出: {out_path / 'directory_trees'}")
        generate_directory_trees.main(out_path)
        progress.update(
            task, advance=1, description="      处理进度 [dim][提取目录层级][/dim]"
        )

        # 3c: generate_prompts_from_trees.py — 结合目录描述生成 prompt 文件
        info("[3c] generate_prompts_from_trees.py — 生成目录层级提取 prompt")
        debug(f"  输入: {out_path / 'directory_trees'}")
        debug(f"  输出: {out_path / 'new_prompts'}")
        generate_prompts_from_trees.main(out_path)
        progress.update(
            task, advance=1, description="      处理进度 [dim][获取目录描述][/dim]"
        )

        # 3d: send_prompts_to_llm.py — 调用 LLM API 获取目录描述
        info("[3d] send_prompts_to_llm.py — 发送 prompts 到 LLM API")
        debug(f"  输入: {out_path / 'new_prompts'}")
        debug(f"  输出: {out_path / 'llm_responses'}")
        send_prompts_to_llm.main(out_path)
        progress.update(
            task, advance=1, description="      处理进度 [dim][解析完成][/dim]"
        )

    info("[Step 3 完成]")


def run_step_4(out_path) -> None:
    """Step 4: 生成每个代码文件的 Markdown 文档"""
    info("Step 4: final_identification_info")
    info("调用 generate_operator_docs.main() ...")
    debug("输入:")
    debug(f"  - llm_responses: {out_path}/llm_responses/")
    debug(f"  - file_tree_structure: {out_path}/file_tree_structure/")
    debug("输出:")
    debug(f"  - organized_docs: {out_path}/organized_docs/")
    generate_operator_docs.main(out_path)
    info("[Step 4 完成]")


def run_step_5(out_path) -> None:
    """Step 5: 为 Markdown 文档补充 aclnn_*.h 描述，生成算子整体说明 (5a -> 5b)"""
    info("Step 5: add_ori_descriptions")

    # 5a: add_aclnn_h_description.py — 为 organized_docs 中的 md 文件补充描述
    info("[5a] add_aclnn_h_description.py — 补充 ACLNN API 头文件描述")
    debug("  输入: data/organized_docs/  +  data/directory_trees/")
    debug("  输出: data/organized_docs/  (in-place 更新)")
    add_aclnn_h_description.main(out_path)

    # 5b: organize_operator_docs.py — 生成算子整体说明文档
    info("[5b] organize_operator_docs.py — 生成算子整体说明文档")
    debug("  输入: data/organized_docs/  (已更新)  +  data/directory_trees/")
    debug("  输出: data/organized_docs_with_descriptions/")
    organize_operator_docs.main(out_path)

    info("[Step 5 完成]")


def _pre_populate_results(
    out_path: Path,
    old_tree_id_path: Path,
    file_filter: list,
) -> None:
    """
    Step 1 与 Step 2 之间执行：从旧 tree_id 预填充未变更文件的 LLM 结果。

    利用 llm_processor 已有的 is_result_exists 跳过机制，
    已存在结果的文件不再重复调 LLM，从而只对 file_filter 中的变更文件发起 LLM 调用。

    Args:
        out_path:      tree_id 输出目录
        old_tree_id_path: 旧版本完整的 tree_id 目录路径
        file_filter:   变更文件绝对路径列表（仅这些文件会重新调 LLM）
    """
    batch_prompts_path = out_path / "batch_prompts_by_file" / "batch_prompts.json"
    old_results_dir = old_tree_id_path / "results_by_file"
    new_results_dir = out_path / "results_by_file"

    if not batch_prompts_path.exists() or not old_results_dir.exists():
        info(
            "  [_pre_populate_results] 跳过: batch_prompts.json 或旧 results_by_file 不存在"
        )
        return

    new_results_dir.mkdir(parents=True, exist_ok=True)

    with open(batch_prompts_path, "r", encoding="utf-8") as f:
        batch_prompts = json.load(f)

    # batch_prompts.json 中 filepath 现为绝对路径，直接与 file_filter 对比
    copied_count = 0
    for task_id, task_data in batch_prompts.get("prompts", {}).items():
        filepath_abs = task_data.get("filepath", "")
        if not filepath_abs:
            continue

        # 变更文件 —— 不预填充，由 step_2 重新调 LLM
        if file_filter and filepath_abs in file_filter:
            continue

        # 用 task_id 前缀匹配旧结果文件
        task_id_prefix = f"{int(task_id):03d}_"
        for old_file in old_results_dir.iterdir():
            if old_file.is_file() and old_file.name.startswith(task_id_prefix):
                new_file = new_results_dir / old_file.name
                if not new_file.exists():
                    shutil.copy2(old_file, new_file)
                    copied_count += 1

    info(f"  [_pre_populate_results] 预填充 {copied_count} 个未变更文件的 LLM 结果")


def main(
    operator_path: Path,
    out_path: Path,
    file_filter: Optional[list] = None,
    old_tree_id_path: Optional[Path] = None,
) -> None:
    """
    算子文档生成流水线入口。

    Args:
        operator_path: 算子源码目录路径
        out_path:      输出目录路径（tree_id 输出位置）
        file_filter:   变更文件绝对路径列表（可选，仅用于增量场景）
        old_tree_id_path: 旧版本完整的 tree_id 目录路径（可选，用于预填充旧结果）
    """
    info("=" * 70)
    info("  算子文档生成流水线 (step_1 -> step_2 -> step_3 -> step_4 -> step_5)")
    info("=" * 70)
    info("=" * 70)

    run_step_1(operator_path, out_path)

    # 增量模式：在 step_1 扫描完毕后，预填充未变更文件的旧 LLM 结果
    if file_filter and old_tree_id_path:
        _pre_populate_results(out_path, old_tree_id_path, file_filter)

    run_step_2(operator_path, out_path)
    run_step_3(out_path)
    run_step_4(out_path)
    run_step_5(out_path)

    info("=" * 70)
    info("  流水线执行完毕!")
    info("=" * 70)
    info("输出目录汇总:")
    info(f"  Step 1: {out_path}/")
    info("    ├── operators.json")
    info("    ├── code_file_tasks.json")
    info("    ├── batch_prompts_by_file/batch_prompts.json")
    info("    └── analysis_results/")
    info(f"  Step 2: {out_path}/results_by_file/")
    info(f"  Step 3: {out_path}/")
    info("    ├── file_tree_structure/")
    info("    ├── directory_trees/")
    info("    ├── new_prompts/")
    info("    └── llm_responses/")
    info(f"  Step 4: {out_path}/organized_docs/")
    info(f"  Step 5: {out_path}/organized_docs_with_descriptions/")


if __name__ == "__main__":
    main()
