#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
plog 日志解析脚本（运行时错误码版本）
用途：解析 Ascend plog 日志，提取错误码和运行时异常信息

使用方法：
    python3 parse_plog.py <plog_file_path>
    python3 parse_plog.py  # 使用最新日志
"""

import os
import sys
import re
import glob
from typing import Dict, Optional


class PlogParser:
    """plog 日志解析器（运行时错误码版本）"""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.errors = []
        self.warnings = []

    def parse(self) -> Dict:
        if not os.path.exists(self.log_path):
            return {"error": f"日志文件不存在: {self.log_path}"}

        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            self._parse_line(line.strip(), line_num)

        return {
            "log_file": self.log_path,
            "total_lines": len(lines),
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": self._generate_summary()
        }

    @staticmethod
    def _classify_error(line: str) -> str:
        if re.search(r'ACLNN_ERR_PARAM', line):
            return "参数错误"
        elif re.search(r'ACLNN_ERR_RUNTIME', line):
            return "Runtime错误"
        elif re.search(r'ACLNN_ERR_INNER_TILING', line):
            return "Tiling错误"
        elif re.search(r'ACLNN_ERR_INNER_FIND_KERNEL', line):
            return "Kernel查找错误"
        elif re.search(r'ACLNN_ERR_INNER_OPP', line):
            return "环境配置错误"
        else:
            return "其他错误"

    def _parse_line(self, line: str, line_num: int):
        if re.search(r'\[ERROR\]', line, re.IGNORECASE):
            self.errors.append({
                "line": line_num,
                "content": line,
                "type": self._classify_error(line)
            })

        elif re.search(r'\[WARN\]', line, re.IGNORECASE):
            self.warnings.append({
                "line": line_num,
                "content": line
            })

    def _generate_summary(self) -> str:
        summary_lines = []
        summary_lines.append(f"错误总数: {len(self.errors)}")
        summary_lines.append(f"警告总数: {len(self.warnings)}")

        if self.errors:
            error_types = {}
            for err in self.errors:
                error_type = err["type"]
                error_types[error_type] = error_types.get(error_type, 0) + 1

            summary_lines.append("\n错误类型分布:")
            for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
                summary_lines.append(f"  - {etype}: {count}")

        return "\n".join(summary_lines)


def find_latest_plog() -> Optional[str]:
    log_dir = os.path.expanduser("~/ascend/log/debug/plog")
    if not os.path.exists(log_dir):
        return None

    log_files = glob.glob(os.path.join(log_dir, "plog-pid_*.log"))
    if not log_files:
        return None

    log_files.sort(key=os.path.getmtime, reverse=True)
    return log_files[0]


def print_report(result: Dict):
    print("=" * 60)
    print("plog 日志解析报告（运行时错误码）")
    print("=" * 60)
    print(f"日志文件: {result['log_file']}")
    print(f"总行数: {result['total_lines']}")
    print()
    print(result['summary'])
    print()

    if result['errors']:
        print("=" * 60)
        print("错误详情 (前10条)")
        print("=" * 60)
        for err in result['errors'][:10]:
            print(f"[Line {err['line']}] [{err['type']}]")
            print(f"  {err['content'][:200]}")
            print()


def main():
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = find_latest_plog()
        if not log_path:
            print("错误: 未找到 plog 日志文件")
            print("用法: python3 parse_plog.py <plog_file_path>")
            sys.exit(1)
        print(f"使用最新日志: {log_path}")

    parser = PlogParser(log_path)
    result = parser.parse()

    if "error" in result:
        print(f"错误: {result['error']}")
        sys.exit(1)

    print_report(result)

    if result['errors']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
