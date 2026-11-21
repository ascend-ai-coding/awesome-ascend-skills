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
"""按 Set 或 Section 名称查询结果目录。"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cli_support import add_format_argument, configure_cli_loggers, emit_result

LOGGER, OUTPUT_LOGGER = configure_cli_loggers(__name__)

CATALOG_PATH = Path(__file__).resolve().parents[1] / "assets" / "section-catalog.json"


class SelectionAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        selections = list(getattr(namespace, self.dest, None) or [])
        selection_kind = option_string.removeprefix("--")
        selections.append((selection_kind, values))
        setattr(namespace, self.dest, selections)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="查询 npu-compute Set、Section、结果文件和解析入口。"
    )
    parser.add_argument(
        "--set",
        dest="selections",
        action=SelectionAction,
        metavar="NAME",
        help="Set 名称，区分大小写；可与 --section 交错并重复指定",
    )
    parser.add_argument(
        "--section",
        dest="selections",
        action=SelectionAction,
        metavar="NAME",
        help="Section 名称，区分大小写；可与 --set 交错并重复指定",
    )
    add_format_argument(parser)
    return parser.parse_args()


def load_catalog():
    with CATALOG_PATH.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def report_unknown(selection_kind, name, choices):
    label = "Set" if selection_kind == "set" else "Section"
    LOGGER.error("错误：未知 %s：%s", label, name)
    LOGGER.error("可选值：%s", ", ".join(choices))


def query(catalog, selections):
    sections = catalog["sections"]
    if not selections:
        return (
            {
                "sets": catalog["sets"],
                "sections": sections,
                "default_results": catalog["default_results"],
            },
            0,
            False,
        )

    sections_by_name = {section["name"]: section for section in sections}
    selected_sections = []
    seen = set()
    for selection_kind, name in selections:
        if selection_kind == "set":
            member_names = catalog["sets"].get(name)
            if member_names is None:
                report_unknown("set", name, catalog["sets"])
                return None, 2, False
        else:
            if name not in sections_by_name:
                report_unknown("section", name, sections_by_name)
                return None, 2, False
            member_names = [name]

        for member_name in member_names:
            if member_name in seen:
                continue
            seen.add(member_name)
            selected_sections.append(sections_by_name[member_name])

    single_section = len(selections) == 1 and selections[0][0] == "section"
    if single_section:
        return selected_sections[0], 0, True
    return {"sections": selected_sections}, 0, False


def render_entry(entry):
    return "\n".join(
        (
            f"## {entry['name']}",
            "",
            f"- **结果文件：** `{entry['result_file']}`",
            f"- **结果类型：** `{entry['result_kind']}`",
            f"- **说明：** `{entry['reference']}`",
            f"- **检查脚本：** `{entry['inspector']}`",
        )
    )


def render_set(set_name, members):
    return "\n".join((f"## {set_name}", "", *(f"- `{name}`" for name in members)))


def render_markdown(result, single_section):
    if single_section:
        return render_entry(result)

    blocks = []
    if "sets" in result:
        blocks.append("# npu-compute Set")
        blocks.extend(
            render_set(set_name, members)
            for set_name, members in result["sets"].items()
        )
    blocks.append("# npu-compute Section")
    blocks.extend(render_entry(entry) for entry in result["sections"])
    if "default_results" in result:
        blocks.append("# 默认结果")
        blocks.extend(render_entry(entry) for entry in result["default_results"])
    return "\n\n".join(blocks)


def main():
    logging.basicConfig(format="%(message)s")
    arguments = parse_arguments()
    try:
        catalog = load_catalog()
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.error("错误：读取 Section 目录失败：%s", error)
        return 1
    result, status, single_section = query(catalog, arguments.selections)
    if status != 0:
        return status
    emit_result(
        OUTPUT_LOGGER,
        arguments.format,
        result,
        lambda value: render_markdown(value, single_section),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
