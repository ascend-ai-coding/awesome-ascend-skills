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
"""检查并解释 npu-compute summary.jsonl。"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cli_support

LOGGER, OUTPUT_LOGGER = cli_support.configure_cli_loggers(__name__)

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
SUMMARY_CATALOG_PATH = ASSET_ROOT / "summary-catalog.json"
METRIC_CATALOG_PATH = ASSET_ROOT / "metric-catalog.json"


class InspectionError(Exception):
    """表示 summary.jsonl 不符合公开结构。"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="检查并解释 npu-compute summary.jsonl。"
    )
    parser.add_argument("jsonl_path", type=Path, help="需要检查的 summary JSONL 文件")
    cli_support.add_format_argument(parser)
    return parser.parse_args()


def load_catalogs():
    with SUMMARY_CATALOG_PATH.open(encoding="utf-8") as input_file:
        summary_catalog = json.load(input_file)
    with METRIC_CATALOG_PATH.open(encoding="utf-8") as input_file:
        metric_catalog = json.load(input_file)
    return summary_catalog, metric_catalog


def load_records(path, record_types):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise InspectionError(f"读取 summary.jsonl 失败：{error}") from error

    records = []
    categories = set()
    op_info_seen = False
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if op_info_seen:
            raise InspectionError("OpInfoSummary 必须是最后一条记录")
        category, record = cli_support.parse_jsonl_record(
            line, line_number, InspectionError
        )
        if category not in record_types:
            raise InspectionError(f"未知类别：{category}")
        if category in categories:
            raise InspectionError(f"重复类别：{category}")
        records.append((line_number, category, record))
        categories.add(category)
        op_info_seen = category == "OpInfoSummary"

    if not records:
        raise InspectionError("summary.jsonl 没有记录")
    if records[-1][1] != "OpInfoSummary":
        raise InspectionError("最后一条记录必须是 OpInfoSummary")
    return records


def section_fields(category, definition, metric_catalog):
    source = definition["metric_fields"]
    if source["catalog"] != "metric-catalog.json":
        raise InspectionError(f"{category} 使用未知字段目录：{source['catalog']}")
    section = metric_catalog["sections"].get(source["section"])
    if section is None:
        raise InspectionError(f"{category} 的指标字段定义不存在")
    excluded = set(source["excluded_fields"])
    return [field for field in section["fields"] if field["name"] not in excluded]


def validate_field_names(category, record, expected_names):
    actual_names = list(record)
    unknown = [name for name in actual_names if name not in expected_names]
    missing = [name for name in expected_names if name not in actual_names]
    errors = []
    if unknown:
        errors.append(f"{category} 未知字段：{', '.join(unknown)}")
    if missing:
        errors.append(f"{category} 缺少字段：{', '.join(missing)}")
    if errors:
        raise InspectionError("\n".join(errors))
    if actual_names != expected_names:
        raise InspectionError(f"{category} 字段顺序不一致")


def matches_type(value, json_type):
    if json_type == "null":
        return value is None
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def validate_value(category, name, value, json_types):
    if not any(matches_type(value, json_type) for json_type in json_types):
        expected = " 或 ".join(json_types)
        raise InspectionError(f"{category}.{name} 类型应为 {expected}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise InspectionError(f"{category}.{name} 必须是有限数值")
    if isinstance(value, str) and not value:
        raise InspectionError(f"{category}.{name} 不能为空")


def inspect_section(category, record, definition, metric_catalog):
    fields = section_fields(category, definition, metric_catalog)
    expected_names = ["category", *(field["name"] for field in fields)]
    validate_field_names(category, record, expected_names)
    validate_value(category, "category", record["category"], ["string"])

    output_fields = []
    for field in fields:
        value = record[field["name"]]
        validate_value(category, field["name"], value, ["number", "null"])
        output_fields.append(
            {
                "name": field["name"],
                "value": value,
                "json_types": ["number", "null"],
                "meaning": field["meaning"],
                "unit": field["unit"],
                "nullable_when": field["na_conditions"],
                "analysis": field["analysis_notes"],
            }
        )
    return {
        "category": category,
        "description": definition["description"],
        "data_level": definition["data_level"],
        "fields": output_fields,
    }


def inspect_op_info(record, definition, selected_fields):
    fields = definition["fields"]
    expected_names = [field["name"] for field in fields]
    validate_field_names("OpInfoSummary", record, expected_names)

    output_fields = []
    for field in fields:
        value = record[field["name"]]
        validate_value("OpInfoSummary", field["name"], value, field["json_types"])
        output_fields.append({**field, "value": value})
    return {
        "description": definition["description"],
        "data_level": definition["data_level"],
        "selected_values": {name: record[name] for name in selected_fields},
        "fields": output_fields,
    }


def inspect(path, summary_catalog, metric_catalog):
    record_types = summary_catalog["record_types"]
    records = load_records(path, record_types)
    section_records = []
    op_info_summary = None
    for _, category, record in records:
        definition = record_types[category]
        if category == "OpInfoSummary":
            op_info_summary = inspect_op_info(
                record, definition, summary_catalog["op_info_selected_fields"]
            )
        else:
            section_records.append(
                inspect_section(category, record, definition, metric_catalog)
            )
    return {
        "file": str(path),
        "record_count": len(records),
        "section_categories": [record["category"] for record in section_records],
        "section_records": section_records,
        "op_info_summary": op_info_summary,
    }


def escape_table_cell(value):
    if value is None:
        return "null"
    if isinstance(value, list):
        value = "；".join(str(item) for item in value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_fields(fields):
    lines = [
        "| 字段 | 实际值 | 类型 | 含义 | 单位 | 可为 null 的条件 | 分析说明 |",
        "|---|---|---|---|---|---|---|",
    ]
    for field in fields:
        values = (
            f"`{field['name']}`",
            field["value"],
            "/".join(field["json_types"]),
            field["meaning"],
            field["unit"],
            field["nullable_when"],
            field["analysis"],
        )
        lines.append(
            "| " + " | ".join(escape_table_cell(value) for value in values) + " |"
        )
    return lines


def render_markdown(result):
    lines = [
        "# summary.jsonl 检查",
        "",
        f"- **文件：** `{result['file']}`",
        f"- **记录数：** {result['record_count']}",
        f"- **Section 汇总：** {', '.join(result['section_categories'])}",
    ]
    for record in result["section_records"]:
        lines.extend(("", f"## {record['category']}", "", record["description"], ""))
        lines.extend(render_fields(record["fields"]))
    op_info = result["op_info_summary"]
    lines.extend(("", "## OpInfoSummary", "", op_info["description"], ""))
    lines.extend(render_fields(op_info["fields"]))
    return "\n".join(lines)


def main():
    logging.basicConfig(format="%(message)s")
    arguments = parse_arguments()
    try:
        summary_catalog, metric_catalog = load_catalogs()
        result = inspect(arguments.jsonl_path, summary_catalog, metric_catalog)
    except (InspectionError, json.JSONDecodeError, KeyError, OSError) as error:
        LOGGER.error("错误：%s", error)
        return 2

    cli_support.emit_result(OUTPUT_LOGGER, arguments.format, result, render_markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
