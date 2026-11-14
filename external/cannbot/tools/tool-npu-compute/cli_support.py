# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Shared command-line logging and result rendering helpers."""

import json
import logging
import sys
from collections.abc import Callable


def configure_cli_loggers(module_name):
    logger = logging.getLogger(module_name)
    output_logger = logging.getLogger(f"{module_name}.output")
    output_logger.setLevel(logging.INFO)
    output_logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    output_logger.handlers[:] = [handler]
    return logger, output_logger


def add_format_argument(parser):
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="输出格式，默认为 markdown",
    )


def parse_jsonl_record(line, line_number, error_type):
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise error_type(f"第 {line_number} 行 JSON 格式错误：{error.msg}") from error
    if not isinstance(record, dict):
        raise error_type(f"第 {line_number} 行必须是 JSON 对象")
    if "category" not in record:
        raise error_type(f"第 {line_number} 行缺少 category")
    category = record.get("category")
    if not isinstance(category, str) or not category:
        raise error_type(f"第 {line_number} 行 category 必须是非空字符串")
    return category, record


def emit_result(output_logger, output_format, result, markdown_renderer: Callable):
    if output_format == "json":
        output_logger.info("%s", json.dumps(result, ensure_ascii=False, indent=2))
    else:
        output_logger.info("%s", markdown_renderer(result))
