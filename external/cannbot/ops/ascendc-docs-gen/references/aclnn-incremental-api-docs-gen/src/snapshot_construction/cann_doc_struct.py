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
将md文件夹中的Markdown文件格式化为JSON并写入json文件夹
"""

import glob
import json
import os
import re
import shutil
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from src.logger import debug, error, info, warning


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.current_table = []
        self.current_row = []
        self.current_cell = []
        self.in_thead = False
        self.in_tbody = False
        self.in_th = False
        self.in_td = False
        self.headers = []

    def handle_starttag(self, tag, attrs):
        if tag == "thead":
            self.in_thead = True
        elif tag == "tbody":
            self.in_tbody = True
        elif tag == "tr":
            self.current_row = []
        elif tag == "th":
            self.in_th = True
            self.current_cell = []
        elif tag == "td":
            self.in_td = True
            self.current_cell = []

    def handle_endtag(self, tag):
        if tag == "thead":
            self.in_thead = False
        elif tag == "tbody":
            self.in_tbody = False
        elif tag == "tr":
            if self.in_thead:
                self.headers = self.current_row.copy()
            elif self.in_tbody:
                self.current_table.append(self.current_row.copy())
            self.current_row = []
        elif tag == "th":
            cell_text = "".join(self.current_cell).strip()
            self.current_row.append(cell_text)
            self.in_th = False
        elif tag == "td":
            cell_text = "".join(self.current_cell).strip()
            self.current_row.append(cell_text)
            self.in_td = False
        elif tag == "table":
            if self.headers and self.current_table:
                self.tables.append({"headers": self.headers, "rows": self.current_table})

    def handle_data(self, data):
        if self.in_th or self.in_td:
            self.current_cell.append(data)


def html_table_to_json(html_table: str) -> list:
    """
    将HTML表格转换为JSON格式

    Args:
        html_table: HTML表格字符串

    Returns:
        JSON字符串
    """
    parser = TableParser()
    parser.feed(html_table)

    if not parser.tables:
        return []

    result = []
    for table in parser.tables:
        for row in table["rows"]:
            row_dict = {}
            for i, header in enumerate(table["headers"]):
                if i < len(row):
                    row_dict[header] = row[i]
                else:
                    row_dict[header] = ""
            result.append(row_dict)

    return result


def extract_support_section(content: str) -> Dict[str, Any]:
    """提取产品支持情况部分"""
    result = {}
    # 查找"产品支持情况"标题下的表格
    # (?=^## ) 行首锚定: 防止在 ### / #### 等更深级标题内部误匹配 (需 re.MULTILINE)
    pattern = r"## 产品支持情况\s*\n+(.*?)(?=^## |\Z)"
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    if match:
        result["origin_txt"] = match.group(0).strip()
        result["support_matrix"] = match.group(1).strip()
    return result


def extract_summary_section(content: str) -> Dict[str, Any]:
    """提取功能说明部分"""
    result = {}
    # 查找"功能说明"部分
    pattern = r"## 功能说明\s*\n+(.*?)(?=^## |\Z)"
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    if match:
        result["origin_txt"] = match.group(0).strip()
        section_content = match.group(1).strip()
        # 提取接口功能
        func_match = re.search(r"[*-]*接口功能[*-]*[：:]\s*(.+)", section_content)
        if func_match:
            result["summary_clarify"] = func_match.group(1).strip()

        # 提取计算公式 - 支持 $$...$$ 和 $...$ 格式
        formula_pattern = r"[*-]*计算公式[*-]*[：:](.*?)(?=^## |\Z)"
        formula_match = re.search(formula_pattern, section_content, re.DOTALL | re.MULTILINE)
        if formula_match:
            result["formula"] = formula_match.group(1).strip()

    return result


def extract_function_proto_section(content: str) -> Dict[str, Any]:
    """提取函数原型部分"""
    result = {}
    # 查找"函数原型"部分
    pattern = r"## 函数原型\s*\n+(.*?)(?=^## |\Z)"
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    if match:
        result["origin_txt"] = match.group(0).strip()
        section_content = match.group(1).strip()

        # 提取两段式接口说明 - 提取第一个代码块之前的文字
        code_start = section_content.find("```")
        if code_start != -1:
            text_before_code = section_content[:code_start].strip()
            result["function_proto_clarify"] = text_before_code

        # 提取所有函数签名（代码块中的Cpp代码）
        code_blocks = re.findall(r"```\w*\s*\n(.*?)```", section_content, re.DOTALL)
        result["sub_function_sign"] = code_blocks

    return result


def is_second_stage_function(func_name: str) -> bool:
    """判断是否是第二段接口函数（不以GetWorkspaceSize结尾的）"""
    return not func_name.endswith("GetWorkspaceSize")


def extract_sub_functions(content: str) -> List[Dict[str, Any]]:
    """提取所有子函数的详细说明"""
    sub_functions = []

    # 查找所有## 子函数名（如 aclnnAbsGetWorkspaceSize, aclnnAbs）
    func_pattern = r"## (aclnn\w+)\s*\n+(.*?)(?=^## |\Z)"
    matches = re.finditer(func_pattern, content, re.DOTALL | re.MULTILINE)

    for match in matches:
        func_name = match.group(1)
        func_content = match.group(2).strip()
        origin_txt = match.group(0).strip()

        sub_func = {
            "changeconfig": func_name,
            "origin_txt": origin_txt,
        }

        # 提取参数说明表格 - 查找整个表格（从<table到</table>）
        table_pattern = r"(<table.*?</table>)"
        table_match = re.search(table_pattern, func_content, re.DOTALL)
        if table_match:
            sub_func["param_matrix_table"] = table_match.group(1).strip()
            if sub_func["param_matrix_table"]:
                sub_func["param_matrix"] = html_table_to_json(sub_func["param_matrix_table"])

            # 提取表格后面的补充说明（如产品限制说明）
            start = table_match.end()
            after_table = func_content[start:]
            # 提取到返回值之前的内容
            return_match = re.search(r"\*\*返回值", after_table)
            if return_match:
                desc_text = after_table[: return_match.start()].strip()
                # 清理，去掉无意义的末尾
                if desc_text.endswith("-") or desc_text.endswith("- "):
                    desc_text = desc_text[:-1].strip()
                if desc_text:
                    sub_func["describe"] = desc_text

        # 提取返回值说明
        return_pattern = r"\*\*返回值[：:]?\s*\n+(.*?)(?=第一段接口|^## |\Z)"
        return_match = re.search(return_pattern, func_content, re.DOTALL | re.MULTILINE)
        if return_match:
            return_content = return_match.group(1).strip()
            sub_func["fixconfig_1"] = "返回值"
            sub_func["fixconfig_2"] = return_content
        else:
            # 备用：直接从aclnnStatus开始提取
            aclnn_status_match = re.search(r"(aclnnStatus.*?参见.*?aclnn返回码.*?)。", func_content)
            if aclnn_status_match:
                sub_func["fixconfig_1"] = "返回值"
                sub_func["fixconfig_2"] = aclnn_status_match.group(1).strip()

        # 只有第一段接口（有GetWorkspaceSize）才会有错误码校验
        if is_second_stage_function(func_name):
            sub_functions.append(sub_func)
            continue

        # 检查是否有错误码校验逻辑（仅对第一段接口）
        if "第一段接口" in func_content and "出现以下场景时报错" in func_content:
            sub_func["changeconfig_error"] = "第一段接口会完成入参校验，出现以下场景时报错："

            # 提取错误码表格 - 找到"第一段接口"之后的内容中的表格
            error_section_pattern = r"第一段接口.*?报错[：:]\s*\n*(.*?)(?=^## |\Z)"
            error_match = re.search(error_section_pattern, func_content, re.DOTALL | re.MULTILINE)
            if error_match:
                error_content = error_match.group(1).strip()
                # 提取表格
                error_table_match = re.search(r"(<table.*?</table>)", error_content, re.DOTALL)
                if error_table_match:
                    sub_func["errorcode"] = error_table_match.group(1).strip()

        sub_functions.append(sub_func)

    return sub_functions


def extract_constraint_section(content: str) -> Dict[str, Any]:
    """提取约束说明部分"""
    result = {}
    # 查找"约束说明"部分
    pattern = r"## 约束说明\s*\n+(.*?)(?=^## |\Z)"
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    if match:
        result["origin_txt"] = match.group(0).strip()
        result["constrain_clarify"] = match.group(1).strip()
    return result


def extract_example_section(content: str) -> Dict[str, Any]:
    """提取调用示例部分"""
    result = {}
    # 查找"调用示例"部分
    pattern = r"## 调用示例\s*\n+(.*?)(?=^## |\Z)"
    section_match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    if section_match:
        result["origin_txt"] = section_match.group(0).strip()
        section_content = section_match.group(1).strip()

        examples = []
        # 按代码块分割，每个代码块前是对应描述
        # 分割出所有代码块及其前面的内容
        code_block_pattern = r"(.*?)(```.*?```)"

        for block_match in re.finditer(code_block_pattern, section_content, re.DOTALL):
            desc_text = block_match.group(1).strip()
            code_block = block_match.group(2)

            # 提取代码内容（去掉代码块标记）
            code_content_match = re.search(r"```.*?\n([\s\S]*?)```", code_block)
            if code_content_match:
                code = code_content_match.group(1).strip()
                examples.append({"describe": desc_text, "code": code})

        if examples:
            result["fixconfig"] = "调用示例"
            result["content"] = examples

    return result


def parse_md_file(file_path: str) -> Dict[str, Any]:
    """解析单个md文件并返回JSON结构"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    result = {}

    # 1. support部分
    support = extract_support_section(content)
    if support:
        result["support"] = {
            "fixconfig": "产品支持情况",
            **support,
        }

    # 2. summary部分
    summary = extract_summary_section(content)
    if summary:
        result["summary"] = {
            "fixconfig": "功能说明",
            "summary_fixconfig_function": "接口功能",
            "summary_clarify": summary.get("summary_clarify", ""),
            "summary_fixconfig_formula": "计算公式",
            "summary_formula": summary.get("formula", ""),
            "origin_txt": summary["origin_txt"],
        }

    # 3. function_proto部分
    func_proto = extract_function_proto_section(content)
    if func_proto:
        result["function_proto"] = {
            "fixconfig": "函数原型",
            "function_proto_clarify": func_proto.get("function_proto_clarify", ""),
            "sub_function_sign": func_proto.get("sub_function_sign", []),
            "origin_txt": func_proto["origin_txt"],
        }

    # 4. sub_function_hash部分
    sub_functions = extract_sub_functions(content)
    for idx, sub_func in enumerate(sub_functions, 1):
        func_key = sub_func.get("changeconfig", f"sub_function_{idx}")

        # 获取返回值说明
        fixconfig_1 = sub_func.get("fixconfig_1", "")
        fixconfig_2 = sub_func.get("fixconfig_2", "")

        # 构建基础字段
        result_entry = {
            "changeconfig": sub_func.get("changeconfig", ""),
            f"subfunction_{func_key}_fixconfig": "接口函数参数说明",
            f"subfunction_{func_key}_param_matrix": sub_func.get("param_matrix", ""),
            f"subfunction_{func_key}_param_matrix_table": sub_func.get("param_matrix_table", ""),
            f"subfunction_{func_key}_describe": sub_func.get("describe", ""),
            f"subfunction_{func_key}_fixconfig_1": fixconfig_1,
            f"subfunction_{func_key}_fixconfig_2": fixconfig_2,
            "origin_txt": sub_func["origin_txt"],
        }

        # 添加错误码相关信息（仅第一段接口）
        if "changeconfig_error" in sub_func:
            result_entry[f"subfunction_{func_key}_changeconfig"] = sub_func["changeconfig_error"]
            result_entry[f"subfunction_{func_key}_clarify.errorcode"] = ""
            result_entry[f"subfunction_{func_key}.errorcode"] = sub_func.get("errorcode", "")

        result[f"sub_function_hash_{idx}"] = result_entry

    # 5. constraint部分
    constraint = extract_constraint_section(content)
    if constraint:
        result["constraint"] = {
            "fixconfig": "约束说明",
            "constrain_clarify": constraint.get("constrain_clarify", ""),
            "origin_txt": constraint.get("origin_txt", ""),
        }

    # 6. example部分
    example = extract_example_section(content)
    if example:
        result["example"] = example

    return result


def main(md_path, json_path):
    # 设置路径
    md_folder = md_path  # os.path.join(os.path.dirname(__file__), md_path)
    json_folder = json_path  # os.path.join(os.path.dirname(__file__), json_path)

    # 创建json文件夹（如果不存在）
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    doc_folder = os.path.join(os.path.dirname(json_folder), "doc")
    os.makedirs(doc_folder, exist_ok=True)

    # 获取所有md文件
    md_files = glob.glob(os.path.join(md_folder, "**/aclnn*.md"))
    md_files.sort()
    if not md_files:
        warning(f"{md_path} 未找到文件")
    if len(md_files) > 1:
        info(f"{md_path}  找到 {len(md_files)} 个md文件")

    doc_json_dict = {}
    # 处理每个文件
    for md_file in md_files:
        # 解析md文件
        json_data = parse_md_file(md_file)
        doc_json_dict[md_file] = json_data
        # 生成输出文件名
        basename = os.path.basename(md_file)
        json_filename = basename.replace(".md", ".json")
        if json_folder:
            json_file = os.path.join(json_folder, json_filename)
            # 写入JSON文件
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            shutil.copy2(md_file, os.path.join(doc_folder, basename))

    return doc_json_dict


if __name__ == "__main__":
    # 自测入口: 用法 main('<算子目录>', 'out/doc_json'), 需通过 TEST_OPERATOR_PATH 或
    # CANN_OPS_ROOT 环境变量提供本机 ops 检出路径, 不在代码中硬编码。
    ops_root = os.environ.get("CANN_OPS_ROOT")
    if not ops_root:
        raise SystemExit("请设置 CANN_OPS_ROOT 环境变量指向本机 ops 仓检出根目录")
    main(
        os.path.join(ops_root, "ops-transformer", "mc2", "all_gather_matmul_v2"),
        "out/doc_json",
    )
