#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import os
import re
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup_logging, validate_path

setup_logging()

logger = logging.getLogger(__name__)


def read_cce_file(cce_path):
    with open(cce_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    func_hash = None
    for line in lines:
        match = re.search(r'//\s*funcHash:\s*(\d+)', line)
        if match:
            func_hash = match.group(1)
            break

    if not func_hash:
        return None, None

    line_code_map = {idx: line.strip() for idx, line in enumerate(lines, start=1)}

    return func_hash, line_code_map


def get_cce_op(line_code_map):
    cce_op = {}
    target_ops = {'set_flag', 'wait_flag', 'pipe_barrier', 'SUBKERNEL'}

    for idx, line in line_code_map.items():
        is_t_op = line.startswith('T') and (('<' in line and '>' in line) or ('(' in line and ')' in line))
        has_target_op = any(op in line for op in target_ops)
        if is_t_op or has_target_op:
            cce_op[idx] = line

    return cce_op


def extract_operation_type(cce_op_val):
    cce_op_type = []

    for val in cce_op_val:
        if val.startswith('T') and '<' in val and '>' in val:
            cce_op_type.append(val.split("<")[0])
        elif val.startswith('T') and '(' in val and ')' in val:
            cce_op_type.append(val.split("(")[0])
        elif (val.startswith('w') or val.startswith('s')) and '(' in val:
            cce_op_type.append(val.split("(")[0])
        else:
            cce_op_type.append(val)

    return cce_op_type


def find_source_location(cce_path, json_path, cce_line_number):
    func_hash, line_code_map = read_cce_file(cce_path)

    if not func_hash or not line_code_map:
        logger.info("错误：无法找到 funcHash 或无法解析 CCE 文件")
        return None

    if cce_line_number not in line_code_map:
        logger.info("错误：CCE 文件中没有第 %d 行", cce_line_number)
        return None

    cce_line = line_code_map[cce_line_number]
    cce_op = get_cce_op(line_code_map)
    cce_op_line = list(cce_op.keys())
    cce_op_type = extract_operation_type(list(cce_op.values()))

    if cce_line_number not in cce_op_line:
        return {
            'matched': False,
            'reason': '该代码为框架自动生成代码，非客户前端编写的代码，无源码与之映射',
            'cce_line_code': cce_line
        }

    cce_op_index = cce_op_line.index(cce_line_number)
    cce_op_name = cce_op_type[cce_op_index]

    with open(json_path, 'r', encoding='utf-8') as f:
        program_data = json.load(f)

    func_data = next((func for func in program_data.get('functions', [])
                      if func.get('hash') == func_hash), None)

    if not func_data:
        logger.info("错误：未找到 hash 为 %s 的函数", func_hash)
        return None

    program_op = func_data.get('operations', [])

    cce_count = len(cce_op_line)
    json_count = len(program_op)

    logger.info("")
    logger.info("[统计信息]")
    logger.info("  CCE 文件中操作数: %d 个", cce_count)
    logger.info("  program.json 中 操作数: %d 个", json_count)

    if cce_count != json_count:
        return {
            'matched': False,
            'reason': 'CCE 文件与 program.json 操作数不一致',
            'cce_line_code': cce_line
        }

    matched_op = program_op[cce_op_index]

    return {
        'matched': True,
        'cce_line_code': cce_line,
        'operation_type': cce_op_name,
        'operation_index': cce_op_index + 1,
        'opcode': matched_op.get('opcode'),
        'source_file': matched_op.get('file'),
        'source_line': matched_op.get('line'),
    }


def print_source_code_line(file_path, line_number):
    if not file_path or not line_number:
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if line_number < 1 or line_number > len(lines):
            return

        start = max(0, line_number - 3)
        end = min(len(lines), line_number + 2)

        logger.info("")
        logger.info("[源代码] %s:%d", file_path, line_number)
        logger.info("-" * 80)
        for i in range(start, end):
            marker = ">>>" if i == line_number - 1 else "   "
            logger.info("%s %4d: %s", marker, i + 1, lines[i].rstrip())
        logger.info("-" * 80)
    except Exception as e:
        logger.error("[ERROR] 无法读取源代码文件: %s", e)


def main():
    parser = argparse.ArgumentParser(description="将 CCE 问题行映射到前端源代码")
    parser.add_argument('cce_file', help="CCE 文件路径")
    parser.add_argument('program_json', help="program.json 文件路径")
    parser.add_argument('cce_line_number', type=int, help="CCE 文件中的问题行号")
    args = parser.parse_args()

    cce_path = os.path.abspath(args.cce_file)
    json_path = os.path.abspath(args.program_json)
    cce_line_number = args.cce_line_number

    cce_path = os.path.abspath(cce_path)
    json_path = os.path.abspath(json_path)

    valid, error_msg = validate_path(cce_path, "CCE 文件")
    if not valid:
        logger.info(error_msg)
        sys.exit(1)

    valid, error_msg = validate_path(json_path, "program.json 文件")
    if not valid:
        logger.info(error_msg)
        sys.exit(1)

    logger.info("CCE 文件: %s", cce_path)
    logger.info("问题行号: %d", cce_line_number)
    logger.info("-" * 80)

    result = find_source_location(cce_path, json_path, cce_line_number)

    if result is None:
        logger.info("")
        logger.info("✗ 无法映射到源代码")
        sys.exit(1)

    logger.info("")
    logger.info("[CCE 问题代码]")
    logger.info("  %s", result['cce_line_code'])

    if result['matched']:
        logger.info("")
        logger.info("✓ 操作: %s (第 %d 个)", result['operation_type'], result['operation_index'])
        logger.info("✓ 匹配: %s", result['opcode'])

        if result['source_file'] and result['source_line']:
            logger.info("✓ 源代码: %s:%d", result['source_file'], result['source_line'])
            print_source_code_line(result['source_file'], result['source_line'])
        else:
            logger.info("✗ 该代码为框架自动生成代码，非客户前端编写的代码，无源码与之映射")
    else:
        logger.info("")
        logger.info("✗ 无法匹配")
        logger.info("  原因: %s", result['reason'])


if __name__ == '__main__':
    main()
