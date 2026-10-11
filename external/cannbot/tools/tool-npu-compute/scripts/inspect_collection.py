#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""递归识别 npu-compute 报告解包后的结果文件。"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cli_support import add_format_argument, configure_cli_loggers, emit_result

LOGGER, OUTPUT_LOGGER = configure_cli_loggers(__name__)

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
RESULT_SUBDIRECTORY_PATTERN = re.compile(
    r"^collection-p[0-9]+-(?:[0-9]{4,}|[A-Za-z0-9]{6})$"
)
RESULT_SUBDIRECTORY_DESCRIPTIONS = [
    "collection-p<process-id>-<sequence>",
    "collection-p<process-id>-<unique-suffix>",
]


class InspectionError(Exception):
    """表示解包结果无法检查。"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="递归检查 npu-compute 报告解包后的结果。"
    )
    parser.add_argument("result_path", type=Path, help="报告解包后的结果目录")
    add_format_argument(parser)
    return parser.parse_args()


def load_catalogs():
    catalogs = []
    for name in (
        "metric-catalog.json",
        "hardware-info-catalog.json",
        "pipe-trace-catalog.json",
        "summary-catalog.json",
    ):
        with (ASSET_ROOT / name).open(encoding="utf-8") as input_file:
            catalogs.append(json.load(input_file))
    return catalogs


def known_result_files(
    metric_catalog, hardware_catalog, pipe_trace_catalog, summary_catalog
):
    return {
        hardware_catalog["output_file"],
        pipe_trace_catalog["output_file"],
        summary_catalog["output_file"],
        *(section["output_file"] for section in metric_catalog["sections"].values()),
    }


def inspect(
    path,
    metric_catalog,
    hardware_catalog,
    pipe_trace_catalog,
    summary_catalog,
):
    if path.is_symlink() or not path.is_dir():
        raise InspectionError("解包结果路径必须是目录")

    recognized_names = known_result_files(
        metric_catalog, hardware_catalog, pipe_trace_catalog, summary_catalog
    )
    root_files = []
    result_directories = {}
    unknown_files = []

    def walk(directory, destination):
        try:
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
        except OSError as error:
            relative = directory.relative_to(path).as_posix() or "."
            raise InspectionError(
                f"读取解包结果目录失败：{relative}: {error}"
            ) from error

        for entry in entries:
            relative = entry.relative_to(path).as_posix()
            if entry.is_symlink():
                unknown_files.append(relative)
                continue
            if entry.is_dir():
                if RESULT_SUBDIRECTORY_PATTERN.fullmatch(entry.name):
                    nested_files = []
                    result_directories[relative] = nested_files
                    walk(entry, nested_files)
                else:
                    unknown_files.append(relative + "/")
                    walk(entry, None)
                continue
            if (
                entry.is_file()
                and destination is not None
                and entry.name in recognized_names
            ):
                destination.append(entry.name)
            else:
                unknown_files.append(relative)

    try:
        walk(path, root_files)
    except OSError as error:
        raise InspectionError(f"检查解包结果失败：{error}") from error

    collection_results = [
        {
            "relative_path": relative,
            "recognized_files": sorted(files),
        }
        for relative, files in sorted(result_directories.items())
    ]
    return {
        "root": str(path),
        "root_result": {
            "relative_path": ".",
            "recognized_files": sorted(root_files),
        },
        "collection_results": collection_results,
        "unknown_files": sorted(unknown_files),
        "result_subdirectory_patterns": RESULT_SUBDIRECTORY_DESCRIPTIONS,
    }


def render_file_list(names):
    return "、".join(f"`{name}`" for name in names) if names else "无"


def render_markdown(result):
    lines = [
        "# 解包结果检查",
        "",
        f"- **结果路径：** `{result['root']}`",
        f"- **根级文件：** {render_file_list(result['root_result']['recognized_files'])}",
    ]
    lines.extend(("", "## 子结果集合", ""))
    if result["collection_results"]:
        lines.extend(("| 相对路径 | 已识别文件 |", "|---|---|"))
        for item in result["collection_results"]:
            lines.append(
                f"| `{item['relative_path']}` | "
                f"{render_file_list(item['recognized_files'])} |"
            )
    else:
        lines.append("无")

    lines.extend(("", "## 未识别项目", ""))
    lines.append(render_file_list(result["unknown_files"]))
    return "\n".join(lines)


def main():
    logging.basicConfig(format="%(message)s")
    arguments = parse_arguments()
    try:
        result = inspect(arguments.result_path, *load_catalogs())
    except (InspectionError, json.JSONDecodeError, KeyError, OSError) as error:
        LOGGER.error("错误：%s", error)
        return 2

    emit_result(OUTPUT_LOGGER, arguments.format, result, render_markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
