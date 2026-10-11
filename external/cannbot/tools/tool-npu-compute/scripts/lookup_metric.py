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
"""查询 Section 指标目录，输出字段定义。"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cli_support import add_format_argument, configure_cli_loggers, emit_result

LOGGER, OUTPUT_LOGGER = configure_cli_loggers(__name__)

CATALOG_PATH = Path(__file__).resolve().parents[1] / "assets" / "metric-catalog.json"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="查询 npu-compute Section CSV 字段定义。"
    )
    parser.add_argument("--section", required=True, help="Section 名称，区分大小写")
    parser.add_argument(
        "--field", help="字段名称，区分大小写；省略时返回 Section 的全部字段"
    )
    add_format_argument(parser)
    return parser.parse_args()


def load_catalog():
    with CATALOG_PATH.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def fail(message, choices):
    LOGGER.error("错误：%s", message)
    LOGGER.error("可选值：%s", ", ".join(choices))
    return 2


def query(catalog, section_name, field_name):
    sections = catalog["sections"]
    if section_name not in sections:
        return None, fail(f"未知 Section：{section_name}", sections)

    section = sections[section_name]
    result = {
        "section": section_name,
        "output_file": section["output_file"],
    }
    if field_name is None:
        result["fields"] = section["fields"]
        return result, 0

    fields = {field["name"]: field for field in section["fields"]}
    if field_name not in fields:
        return None, fail(f"{section_name} 中不存在字段：{field_name}", fields)
    result["field"] = fields[field_name]
    return result, 0


def markdown_text(value):
    if isinstance(value, list):
        return "；".join(value)
    return str(value)


def escape_table_cell(value):
    return markdown_text(value).replace("|", "\\|").replace("\n", " ")


def render_field_markdown(result):
    field = result["field"]
    return "\n".join(
        (
            f"# {result['section']} / `{field['name']}`",
            "",
            f"- **输出文件：** `{result['output_file']}`",
            f"- **含义：** {field['meaning']}",
            f"- **适用 Core：** {markdown_text(field['core_types'])}",
            f"- **单位：** {field['unit']}",
            f"- **依赖：** {markdown_text(field['dependencies'])}",
            f"- **计算规则：** {field['formula']}",
            f"- **`NA` 条件：** {markdown_text(field['na_conditions'])}",
            f"- **分析说明：** {field['analysis_notes']}",
        )
    )


def render_section_markdown(result):
    lines = [
        f"# {result['section']} 字段",
        "",
        f"输出文件：`{result['output_file']}`",
        "",
        "| 字段 | 含义 | 适用 Core | 单位 | 依赖 | 计算规则 | `NA` 条件 | 分析说明 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for field in result["fields"]:
        values = (
            f"`{field['name']}`",
            field["meaning"],
            field["core_types"],
            field["unit"],
            field["dependencies"],
            field["formula"],
            field["na_conditions"],
            field["analysis_notes"],
        )
        lines.append(
            "| " + " | ".join(escape_table_cell(value) for value in values) + " |"
        )
    return "\n".join(lines)


def main():
    logging.basicConfig(format="%(message)s")
    arguments = parse_arguments()
    try:
        catalog = load_catalog()
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.error("错误：读取指标目录失败：%s", error)
        return 1

    result, status = query(catalog, arguments.section, arguments.field)
    if status != 0:
        return status
    renderer = render_section_markdown if arguments.field is None else render_field_markdown
    emit_result(OUTPUT_LOGGER, arguments.format, result, renderer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
