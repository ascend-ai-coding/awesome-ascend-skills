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
import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cli_support import add_format_argument, configure_cli_loggers, emit_result

LOGGER, OUTPUT_LOGGER = configure_cli_loggers(__name__)

CATALOG_PATH = Path(__file__).resolve().parents[1] / "assets" / "metric-catalog.json"


class InspectionError(Exception):
    pass


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="检查 npu-compute Section CSV 的结构和数据可用性。"
    )
    parser.add_argument("csv_path", type=Path, help="需要检查的 CSV 文件")
    add_format_argument(parser)
    return parser.parse_args()


def load_catalog():
    with CATALOG_PATH.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def read_rows(path):
    try:
        with path.open(newline="", encoding="utf-8") as input_file:
            return list(csv.reader(input_file, strict=True))
    except csv.Error as error:
        raise InspectionError(f"CSV 格式错误：{error}") from error
    except (OSError, UnicodeError) as error:
        raise InspectionError(f"读取 CSV 失败：{error}") from error


def identify_section(path, header, sections):
    by_filename = {
        section["output_file"]: (name, section) for name, section in sections.items()
    }
    if path.name in by_filename:
        return by_filename[path.name]

    matches = []
    for name, section in sections.items():
        expected_header = [field["name"] for field in section["fields"]]
        if header == expected_header:
            matches.append((name, section))
    if len(matches) == 1:
        return matches[0]
    raise InspectionError("无法根据文件名或表头识别 Section")


def validate_header(header, section_name, section):
    expected = [field["name"] for field in section["fields"]]
    duplicate_fields = [name for name, count in Counter(header).items() if count > 1]
    unknown_fields = [name for name in header if name not in expected]
    missing_fields = [name for name in expected if name not in header]
    errors = []
    if duplicate_fields:
        errors.append(f"重复字段：{', '.join(duplicate_fields)}")
    if unknown_fields:
        errors.append(f"未知字段：{', '.join(unknown_fields)}")
    if missing_fields:
        errors.append(f"缺少字段：{', '.join(missing_fields)}")
    if not errors and header != expected:
        errors.append(f"字段顺序与 {section_name} 定义不一致")
    if errors:
        raise InspectionError("\n".join(errors))


def inspect(path, catalog):
    rows = read_rows(path)
    if not rows:
        raise InspectionError("CSV 文件为空")

    header = rows[0]
    section_name, section = identify_section(path, header, catalog["sections"])
    validate_header(header, section_name, section)
    expected_count = len(header)
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != expected_count:
            raise InspectionError(
                f"第 {line_number} 行列数为 {len(row)}，预期为 {expected_count}"
            )

    data_rows = rows[1:]
    row_count = len(data_rows)
    columns = []
    for index, field in enumerate(section["fields"]):
        values = [row[index] for row in data_rows]
        na_count = sum(value == "NA" for value in values)
        empty_count = sum(value == "" for value in values)
        valid_count = row_count - na_count - empty_count
        availability_rate = valid_count / row_count if row_count else 0.0
        columns.append(
            {
                "name": field["name"],
                "meaning": field["meaning"],
                "unit": field["unit"],
                "valid_count": valid_count,
                "na_count": na_count,
                "availability_rate": availability_rate,
            }
        )
    return {
        "file": str(path),
        "section": section_name,
        "row_count": row_count,
        "columns": columns,
    }


def escape_table_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(result):
    lines = [
        f"# {result['section']} CSV 检查",
        "",
        f"- **文件：** `{result['file']}`",
        f"- **数据行数：** {result['row_count']}",
        "",
        "| 字段 | 含义 | 单位 | 有效值 | `NA` | 有效率 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for column in result["columns"]:
        values = (
            f"`{column['name']}`",
            column["meaning"],
            column["unit"],
            column["valid_count"],
            column["na_count"],
            f"{column['availability_rate']:.2%}",
        )
        lines.append(
            "| " + " | ".join(escape_table_cell(value) for value in values) + " |"
        )
    return "\n".join(lines)


def main():
    logging.basicConfig(format="%(message)s")
    arguments = parse_arguments()
    try:
        result = inspect(arguments.csv_path, load_catalog())
    except (InspectionError, json.JSONDecodeError, OSError) as error:
        LOGGER.error("错误：%s", error)
        return 2

    emit_result(OUTPUT_LOGGER, arguments.format, result, render_markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
