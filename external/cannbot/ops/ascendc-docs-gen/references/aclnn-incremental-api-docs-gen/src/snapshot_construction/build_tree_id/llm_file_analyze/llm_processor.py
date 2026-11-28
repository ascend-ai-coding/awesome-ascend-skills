# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

from src.snapshot_construction.pub_ref import PubRefConfig
from src.llm.client import LLMClient
from src.logger import info, debug, warning, error


def build_prompt(template: str, code: str, folder_tree: str) -> str:
    """
    用代码内容和目录树替换prompt模板中的{code}和{folder_tree}占位符。

    Args:
        template: prompt模板内容
        code: 代码内容
        folder_tree: 目录树字符串

    Returns:
        完整的prompt字符串
    """
    result = template.replace("{code}", code)
    result = result.replace("{folder_tree}", folder_tree)
    return result


def generate_result_filename(
    category: str, operator: str, file_path: str, task_id: int = None
) -> str:
    """
    根据元数据生成结果文件名。

    Args:
        category: 项目分类
        operator: 算子名称
        file_path: 原始文件路径
        task_id: 任务ID，用于确保唯一性

    Returns:
        结果文件名（不含路径）
    """
    # 从绝对路径中提取文件名（不含扩展名）
    file_name = Path(file_path).stem if file_path else "unknown"

    # 使用task_id确保唯一性（task_id是输入文件序号，从0到819，保证不重复）
    # 构造结果文件名：{task_id:03d}_{category}_{operator}_{file_name}_result.json
    if task_id is not None:
        result_filename = f"{task_id:03d}_{category}_{operator}_{file_name}_result.json"
    else:
        result_filename = f"{category}_{operator}_{file_name}_result.json"
    return result_filename


def is_result_exists(result_path: str) -> bool:
    """
    检查结果文件是否已存在。

    Args:
        result_path: 结果文件路径

    Returns:
        存在返回True，否则返回False
    """
    return Path(result_path).exists()


def save_result(result_path: str, metadata: Dict, response: str) -> None:
    """
    保存LLM结果到JSON文件。

    Args:
        result_path: 结果文件路径
        metadata: 元数据字典
        response: LLM响应内容
    """
    # 绝对路径统一为 POSIX 格式
    file_path = (
        Path(metadata.get("file_path", "")).as_posix()
        if metadata.get("file_path")
        else ""
    )

    result = {
        "category": metadata.get("category", ""),
        "operator": metadata.get("operator", ""),
        "operator_root_dir": metadata.get("operator_root_dir", ""),
        "file_path": file_path,
        "response": response,
    }

    # 确保目录存在
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)

    # 使用线程锁确保文件写入安全
    file_lock = threading.Lock()
    with file_lock:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def generate_folder_tree(operator_path: str) -> str:
    """
    根据算子的分类和名称，从源码目录生成完整的目录树字符串。

    Returns:
        目录树字符串，格式类似：
        upsample_trilinear3d_backward/
        ├── op_api/
        │   ├── aclnn_upsample_trilinear_3d_backward.h
        │   └── aclnn_upsample_trilinear_3d_backward.cpp
        ...
    """
    # 构造算子目录路径

    def build_tree(directory: str, prefix: str = "", is_last: bool = True) -> list:
        """递归构建目录树。"""
        lines = []
        try:
            dir_path = Path(directory)
            # 获取目录下的所有文件和子目录，排序
            items = sorted([p.name for p in dir_path.iterdir()])

            # 过滤掉隐藏文件（以.开头）
            items = [item for item in items if not item.startswith(".")]

            for i, item in enumerate(items):
                item_path = dir_path / item
                is_last_item = i == len(items) - 1

                # 确定连接符
                if is_last_item:
                    connector = "└── "
                    child_prefix = prefix + "    "
                else:
                    connector = "├── "
                    child_prefix = prefix + "│   "

                if item_path.is_dir():
                    lines.append(f"{prefix}{connector}{item}/")
                    lines.extend(build_tree(str(item_path), child_prefix, is_last_item))
                else:
                    lines.append(f"{prefix}{connector}{item}")
        except FileNotFoundError:
            # 目录不存在，返回空列表
            return []

        return lines

    # 生成目录树
    if Path(operator_path).exists():
        tree_lines = [f"{operator_path}/"]
        tree_lines.extend(build_tree(operator_path))
        return "\n".join(tree_lines)
    else:
        # 如果算子目录不存在，返回提示信息
        return f"{operator_path}/ (目录不存在: {operator_path})"


def process_single_file(
    task_data: Dict,
    output_dir: Path,
    template: str,
    llm_client: LLMClient,
    stats: Dict,
    stats_lock: threading.Lock,
    idx: int,
    total: int,
    operator_path: Path,
) -> Tuple[str, str]:
    """
    处理单个任务的函数，用于多线程执行。

    Args:
        task_data: 任务数据字典（来自 batch_prompts.json）
        output_dir: 输出目录
        template: prompt模板
        llm_client: LLMClient实例
        stats: 统计信息字典
        stats_lock: 统计信息的线程锁
        idx: 当前处理索引
        total: 任务总数

    Returns:
        (status, message) 状态和消息
    """
    try:
        # 从 JSON 数据读取元数据
        category = task_data.get("operator_root_dir", "")
        operator = task_data.get("operator_name", "")
        operator_root_dir = task_data.get("operator_root_dir", "")
        file_path = task_data.get("filepath", "")
        code = task_data.get("code", "")
        task_id = task_data.get("task_id", idx - 1)

        # 生成目录树
        folder_tree = generate_folder_tree(operator_path=operator_path)

        # 生成结果文件名
        result_filename = generate_result_filename(
            category, operator, file_path, task_id
        )
        result_path = str(output_dir / result_filename)

        # 检查结果是否已存在
        if is_result_exists(result_path):
            with stats_lock:
                stats["skipped"] += 1
                stats["completed"] += 1
            debug(
                f"[Thread-{threading.get_ident()}] [{idx}/{total}] ✅ 跳过 (结果已存在): {result_filename}"
            )
            return ("skipped", f"跳过 (结果已存在): {result_filename}")

        # 构建完整prompt
        full_prompt = build_prompt(template, code, folder_tree)

        # 调用LLM API
        debug(
            f"[Thread-{threading.get_ident()}] [{idx}/{total}] 🔄 处理中: {task_id:03d}_{category}_{operator}"
        )
        response = llm_client.chat(
            [{"role": "user", "content": full_prompt}], max_tokens=2000
        )

        # 保存结果（保持与下游 build_file_tree 兼容的输出格式）
        metadata = {
            "category": category,
            "operator": operator,
            "operator_root_dir": operator_root_dir,
            "file_path": file_path,
            "code": code,
        }
        save_result(result_path, metadata, response)

        with stats_lock:
            stats["processed"] += 1
            stats["completed"] += 1
        debug(
            f"[Thread-{threading.get_ident()}] [{idx}/{total}] ✅ 成功: {result_filename}"
        )
        return ("success", f"成功: {result_filename}")

    except Exception as e:
        with stats_lock:
            stats["failed"] += 1
            stats["completed"] += 1
        error(
            f"[Thread-{threading.get_ident()}] [{idx}/{total}] ❌ 失败: {task_data.get('filepath', 'unknown')} - {str(e)}"
        )
        return "failed", f"失败: {task_data.get('filepath', 'unknown')} - {str(e)}"


def main(operator_path, out_path: Path):
    """主函数：执行LLM批量处理流程。"""

    info("=" * 60)
    info("LLM API 自动化处理脚本 (版本2)")
    info("=" * 60)

    # 创建LLMClient
    llm_client = LLMClient(model=PubRefConfig.MODEL)

    if not PubRefConfig.INCREMENTAL_LLM_KEY:
        error("错误: 未找到 OPENAI_API_KEY 环境变量")
        error("请设置环境变量: export INCREMENTAL_LLM_KEY='your-api-key'")
        return

    info("API配置:")
    info(f"  模型: {llm_client.model}")
    info(f"  Base URL: {llm_client.client.base_url}")
    info(f"  API Key: {'已设置' if PubRefConfig.INCREMENTAL_LLM_KEY else '未设置'}")

    # 读取prompt模板 - 使用新模板
    template_path = Path(__file__).parent / "aclnn_prompt模板.md"
    if not template_path.exists():
        error(f"错误: 找不到模板文件 {template_path}")
        return

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    info(f"已加载模板文件: {template_path}")

    # 设置目录路径
    input_dir = out_path / "batch_prompts_by_file"
    output_dir = out_path / "results_by_file"  # 使用新文件夹保存结果

    if not input_dir.exists():
        error(f"错误: 找不到输入目录 {input_dir}")
        return

    # 读取统一的 batch_prompts.json
    prompt_json_path = input_dir / "batch_prompts.json"
    if not prompt_json_path.exists():
        error(f"错误: 找不到 {prompt_json_path}")
        return

    with open(prompt_json_path, "r", encoding="utf-8") as f:
        batch_data = json.load(f)

    prompts_dict = batch_data.get("prompts", {})
    if not prompts_dict:
        error("错误: batch_prompts.json 中无 prompts 数据")
        return

    # 按 task_id 排序
    sorted_items = sorted(prompts_dict.items(), key=lambda x: int(x[0]))
    info(f"找到 {len(sorted_items)} 个待处理任务")

    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)

    # 统计信息
    stats = {
        "total": len(sorted_items),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "completed": 0,
    }
    stats_lock = threading.Lock()

    # 设置并发数量
    max_workers = 5
    info(f"使用 {max_workers} 并发线程处理")
    info(f"预计处理时间约: ~{len(sorted_items) // max_workers} 批次")

    # 使用ThreadPoolExecutor进行多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [
            executor.submit(
                process_single_file,
                task_data,  # 直接传 dict，不再传 Path
                output_dir,
                template,
                llm_client,
                stats,
                stats_lock,
                idx,
                len(sorted_items),
                operator_path,
            )
            for idx, (tid, task_data) in enumerate(sorted_items, 1)
        ]

        # 使用Rich显示进度条
        console = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=console,
            disable=not sys.stdout.isatty(),
        ) as progress:
            task = progress.add_task("      处理进度", total=len(sorted_items))
            for future in as_completed(futures):
                try:
                    future.result()
                    progress.update(
                        task,
                        advance=1,
                        description=(
                            f"      处理进度 "
                            f"[green]成功:{stats['processed']}[/] "
                            f"[yellow]跳过:{stats['skipped']}[/] "
                            f"[red]失败:{stats['failed']}[/] "
                            f"[cyan]完成:{stats['completed']}[/]"
                        ),
                    )
                except Exception:
                    progress.update(task, advance=1)

    # 输出处理统计信息
    info("=" * 60)
    info("处理统计:")
    info(f"  总任务数: {stats['total']}")
    info(f"  处理成功: {stats['processed']}")
    info(f"  跳过已存在: {stats['skipped']}")
    info(f"  处理失败: {stats['failed']}")
    info(f"  完成总数: {stats['completed']}")
    info("=" * 60)


if __name__ == "__main__":
    main()
