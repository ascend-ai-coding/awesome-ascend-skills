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
Anchor Redirect Pipeline - 统一入口函数

按顺序执行四个重定向阶段:
1. file 阶段: 文件级重定向 (type="file")
2. func 阶段: 函数级重定向 (type="function")
3. note 阶段: 注释重定向 (type="note")
4. section 阶段: 章节重定向 (type="section")
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.logger import debug, error, info, warning

# 抑制子模块中 logging.INFO/WARNING/ERROR 的输出，只保留 CRITICAL
logging.basicConfig(level=logging.CRITICAL, format="%(message)s")
for logger_name in ["", __name__]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)


def run_anchor_redirect_pipeline(
    anchor_path: str,
    origin_map_base: str,
    change_map_base: str,
    origin_repo_path: str,
    change_repo_path: str,
    output_dir: str = "anchor_redirect_output",
    verbose: bool = False,
    extra_files_info: dict = None,
    filter_list: list = None,
) -> dict:
    """
    顺序执行 file → func → note → section 四个重定向阶段。

    参数:
        anchor_path: anchor JSON 文件路径（四个阶段共用）
        origin_map_base: 变更前 code_map 目录 (file/func 共用)
        change_map_base: 变更后 code_map 目录 (file/func 共用)
        change_repo_path: 变更后算子仓库路径 (note/section 用作 data_dir)
        output_dir: 输出目录，默认 "anchor_redirect_output"
        verbose: 是否输出详细日志
        extra_files_info: 额外文件信息 {'internal': [...], 'external': [...]}
                         外部文件在重定向时会跳过处理
        filter_list: anchor 文件名过滤列表（不带 .json 后缀，如 ['summary', 'function_proto']）
                    仅处理文件名 stem 命中的 anchor 文件，未命中的直接跳过；None/空列表表示不过滤

    返回:
        汇总报告字典
    """
    # 打印额外文件信息
    if extra_files_info:
        console = Console()
        print("=" * 60)
        print("[run_anchor_redirect_pipeline] 额外文件信息:")
        internal_files = extra_files_info.get("internal", [])
        external_files = extra_files_info.get("external", [])
        print(f"  内部文件（参与重定向）: {len(internal_files)} 个")
        for i, f in enumerate(internal_files, 1):
            print(f"    [{i}] {f}")
        print(f"  外部文件（跳过重定向）: {len(external_files)} 个")
        for i, f in enumerate(external_files, 1):
            print(f"    [{i}] {f}")
        print("=" * 60)

    console = Console()
    output_dir = os.path.join(output_dir, "redirect")
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    info(f"重定向输出路径：{output_dir}")

    # 构建外部文件的绝对路径集合（用于过滤）
    external_files_abs = set()
    if extra_files_info and extra_files_info.get("external"):
        for ext_file in extra_files_info["external"]:
            external_files_abs.add(str(Path(ext_file).resolve()))

    if external_files_abs:
        info(f"[过滤] 外部文件数量: {len(external_files_abs)} 个，对应的 anchor 将被跳过")

    # 读取初始 anchor JSON 并过滤外部文件的 anchor
    anchor_files = os.listdir(anchor_path)

    # 根据 filter_list 过滤 anchor 文件
    # filter_list 元素对应 anchor 文件名（不带 .json 后缀，如 'summary' 对应 'summary.json'）
    # 在 filter_list 中的保留，不在的直接跳过；None/空列表表示不过滤（全部保留）
    if filter_list:
        filter_set = set(filter_list)
        before_count = len(anchor_files)
        anchor_files = [f for f in anchor_files if Path(f).stem in filter_set]
        skipped_count = before_count - len(anchor_files)
        if skipped_count > 0:
            info(f"[过滤] filter_list 过滤: 跳过 {skipped_count} 个 anchor 文件，保留 {len(anchor_files)} 个")
        if not anchor_files:
            warning(f"[过滤] filter_list={filter_list} 过滤后未保留任何 anchor 文件")

    filtered_anchor_dir = os.path.join(output_dir, "filtered_anchors")
    os.makedirs(filtered_anchor_dir, exist_ok=True)

    total_skipped = 0
    total_retained = 0

    for anchor_file in anchor_files:
        with open(os.path.join(anchor_path, anchor_file), "r", encoding="utf-8") as f:
            data = json.load(f)
        anchors = data.get("anchor_tuple", [])

        # 过滤外部文件的 anchor
        if external_files_abs:
            filtered_anchors = []
            skipped_count = 0

            for anchor in anchors:
                # 获取 anchor 的 file_path
                anchor_info = anchor.get("anchor_information", {})
                level1 = anchor_info.get("level1", {})
                file_path = level1.get("file_path", "")

                if file_path:
                    # 将 anchor 的 file_path 转换为绝对路径
                    anchor_file_abs = str(Path(file_path).resolve())

                    # 判断是否在外部文件列表中
                    if anchor_file_abs in external_files_abs:
                        skipped_count += 1
                        continue  # 跳过外部文件的 anchor

                filtered_anchors.append(anchor)

            # 更新 anchors
            anchors = filtered_anchors
            data["anchor_tuple"] = filtered_anchors

            if skipped_count > 0:
                info(f"  {anchor_file}: 跳过 {skipped_count} 个外部文件 anchor，保留 {len(filtered_anchors)} 个")

            total_skipped += skipped_count
            total_retained += len(filtered_anchors)

        # 将过滤后的 anchor 数据保存到临时目录
        filtered_anchor_file = os.path.join(filtered_anchor_dir, anchor_file)
        with open(filtered_anchor_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        debug(f"  {anchor_file}加载 {len(anchors)} 个 anchor（已保存过滤后的版本）")

        # 统计 anchor 类型
        type_counts = {}
        for a in anchors:
            t = a.get("anchor_information", {}).get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items()))
        # print(f"  类型分布: {type_parts}")

        report = {
            "anchor_path": anchor_path,
            "origin_map_base": origin_map_base,
            "change_map_base": change_map_base,
            "change_repo_path": change_repo_path,
            "output_dir": output_dir,
            "stages": {},
        }
        debug(json.dumps(report, indent=2, ensure_ascii=False))

    if external_files_abs and total_skipped > 0:
        info(f"[过滤汇总] 总计跳过 {total_skipped} 个外部文件 anchor，保留 {total_retained} 个")

    console.print("    [dim]【File 阶段】文件级重定向[/dim]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    ) as file_progress:
        file_task = file_progress.add_task("    处理进度", total=len(anchor_files))
        for anchor_file in anchor_files:
            # =====================
            # Stage 1: file 阶段
            # =====================
            import src.change_detection.anchor_redirect_file.ancor_match_file_0422 as file_module

            file_progress.update(file_task, description=f"    处理进度 [dim]({anchor_file})[/dim]")

            # 使用过滤后的 anchor 文件（如果有外部文件）
            if external_files_abs:
                anchor_input_path = os.path.join(filtered_anchor_dir, anchor_file)
            else:
                anchor_input_path = os.path.join(anchor_path, anchor_file)

            report_return, file_data = file_module.process(
                anchor_input_path,
                origin_map_base,
                change_map_base,
                change_repo_path,
                output_dir,
            )

            # 同步 anchor 数据
            with open(os.path.join(output_dir, f"file_{anchor_file}"), "w", encoding="utf-8") as f:
                json.dump(file_data, f, ensure_ascii=False, indent=2)
            file_progress.update(file_task, advance=1)
    info("[Stage 1] file 阶段 - 完成")

    info("====" * 20)

    # for anchor_file in anchor_files:
    # =====================
    # Stage 2: func 阶段
    # =====================
    console.print("    [dim]【Function 阶段】函数级重定向[/dim]")

    # 始终使用 filtered_anchors/（已含 filter_list 过滤 + 剔除外部文件 anchor）
    func_anchor_path = filtered_anchor_dir

    anchor_redirect_file_try_func_process(
        func_anchor_path,
        origin_map_base,
        change_map_base,
        change_repo_path,
        output_dir,
        verbose,
    )

    info("[Stage 2] func 阶段 - 完成")
    info("====" * 20)
    # =====================
    # Stage 3: note 阶段
    # =====================
    console.print("    [dim]【Note 阶段】注释级重定向[/dim]")

    # 始终使用 filtered_anchors/（已含 filter_list 过滤 + 剔除外部文件 anchor）
    note_anchor_path = filtered_anchor_dir

    try:
        import src.change_detection.anchor_redirect_note.anchor_retarget_note as note_module

        note_module.process_anchor_data(note_anchor_path, change_repo_path, output_dir, verbose)

        info("[Stage 3] note 阶段 - 完成")
        # print(result)
        # print(f"  处理: {stats['processed']} 个 | 找到: {stats['found']} 个 | 未找到: {stats['not_found']} 个")
    except Exception as e:
        report["stages"]["note"] = {
            "status": f"error: {e}",
        }
        # 打印异常详细信息：文件名、行号、堆栈
        import sys
        import traceback

        traceback.print_exc()

        # 单独获取 当前异常的 文件名、行号
        exc_type, exc_value, exc_tb = sys.exc_info()
        fname = exc_tb.tb_frame.f_code.co_filename  # 报错文件
        line_no = exc_tb.tb_lineno  # 报错行号

        error(f"\n报错文件：{fname}")
        error(f"报错行号：{line_no}")
        error(f"错误信息：{e}")
        info("[Stage 3] note 阶段 - 完成")
        error(f"  错误: {e}")
    info("====" * 20)
    # =====================
    # Stage 4: section 阶段
    # =====================
    console.print("    [dim]【Section 阶段】章节级重定向[/dim]")

    # 始终使用 filtered_anchors/（已含 filter_list 过滤 + 剔除外部文件 anchor）
    section_anchor_path = filtered_anchor_dir

    try:
        import src.change_detection.anchor_redirect_section.anchor_retarget_section as section_module

        section_module.process_anchor_data(section_anchor_path, change_repo_path, output_dir, verbose)

        info("[Stage 4] section 阶段 - 完成")
        # print(result)
        # print(f"  处理: {stats['processed']} 个 | 找到: {stats['found']} 个 | 未找到: {stats['not_found']} 个")
    except Exception as e:
        report["stages"]["section"] = {
            "status": f"error: {e}",
        }
        info("[Stage 4] section 阶段 - 完成")
        error(f"  错误: {e}")
        # 打印异常详细信息：文件名、行号、堆栈
        import sys
        import traceback

        traceback.print_exc()

    # 保存汇总报告
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    info("Pipeline 完成")
    info(f"  汇总报告: {summary_path}")

    return report


def anchor_redirect_file_try_func_process(
    anchor_path: str,
    origin_map_base: str,
    change_map_base: str,
    change_repo_path: str,
    output_dir: str,
    verbose: bool,
):
    """尝试调用 func 模块的 process 函数，失败时返回 None 并打印错误。"""
    import src.change_detection.anchor_redirect_func.function_redirect_v1 as func_module

    anchor_path_list = os.listdir(anchor_path)
    function_data_result = []
    for anchor in anchor_path_list:
        anchor_json_path = os.path.join(anchor_path, anchor)
        func_data = func_module.process(
            anchor_json_path,
            change_map_base,
            change_map_base,
            change_repo_path.split("op_host")[0],
            output_dir,
            verbose,
        )
        function_data_result.append(func_data)

    # print(function_data_result)
    return function_data_result


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Anchor Redirect Pipeline - 顺序执行 file→func→note→section 四个重定向阶段"
    )
    parser.add_argument(
        "--anchor-path",
        default="./out/base/anchor/",
        type=str,
        # required=True,
        help="anchor JSON 文件路径（四个阶段共用）",
    )
    parser.add_argument(
        "--origin-map-base",
        default="out/base/code_map/base",
        type=str,
        # required=True,
        help="变更前 code_map 目录（file/func 共用）",
    )
    parser.add_argument(
        "--change-map-base",
        default="out/base/code_map/base",
        type=str,
        # required=True,
        help="变更后 code_map 目录（file/func 共用）",
    )
    parser.add_argument(
        "--origin-repo-path",
        default="./out/operator_storage/roi_align/base/op_host/op_api/aclnn_roi_align.cpp",
        type=str,
        # required=True,
        help="变更前算子仓库路径（note/section 用作 data_dir）",
    )
    parser.add_argument(
        "--change-repo-path",
        default="./out/operator_storage/roi_align/base",
        type=str,
        # required=True,
        help="变更后算子仓库路径（note/section 用作 data_dir）",
    )
    # parser.add_argument()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="输出目录（默认: anchor_redirect_output）",
    )
    parser.add_argument("--verbose", action="store_true", default=True, help="显示详细输出")
    return parser.parse_args()


def main():
    args = parse_args()

    run_anchor_redirect_pipeline(
        anchor_path=args.anchor_path,
        origin_map_base=args.origin_map_base,
        change_map_base=args.change_map_base,
        origin_repo_path=args.origin_repo_path,
        change_repo_path=args.change_repo_path,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
