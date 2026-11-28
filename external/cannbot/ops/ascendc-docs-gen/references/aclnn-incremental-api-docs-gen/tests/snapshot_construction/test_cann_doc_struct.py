# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Tests for cann_doc_struct module.

Tests cover:
- TableParser: HTML table parsing (thead/tbody, multiple tables, empty cells, no headers, nested content)
- html_table_to_json: HTML table string to list of dicts
- extract_support_section: 产品支持情况 section extraction
- extract_summary_section: 功能说明 section extraction
- extract_function_proto_section: 函数原型 section extraction
- is_second_stage_function: function name classification
- extract_sub_functions: sub-function detail extraction
- extract_constraint_section: 约束说明 section extraction
- extract_example_section: 调用示例 section extraction
- parse_md_file: complete markdown file parsing
- main: batch processing of md files
"""

import json

import pytest

from src.snapshot_construction.cann_doc_struct import (
    TableParser,
    html_table_to_json,
    extract_support_section,
    extract_summary_section,
    extract_function_proto_section,
    is_second_stage_function,
    extract_sub_functions,
    extract_constraint_section,
    extract_example_section,
    parse_md_file,
    main,
)


# =============================================================================
# Test data: HTML tables
# =============================================================================

SIMPLE_TABLE_HTML = """\
<table>
<thead>
<tr><th>参数</th><th>类型</th></tr>
</thead>
<tbody>
<tr><td>self</td><td>输入张量</td></tr>
<tr><td>out</td><td>输出张量</td></tr>
</tbody>
</table>
"""

MULTI_TABLE_HTML = """\
<table>
<thead>
<tr><th>A</th></tr>
</thead>
<tbody>
<tr><td>1</td></tr>
</tbody>
</table>
<table>
<thead>
<tr><th>B</th></tr>
</thead>
<tbody>
<tr><td>2</td></tr>
</tbody>
</table>
"""

TABLE_EMPTY_CELLS_HTML = """\
<table>
<thead>
<tr><th>A</th><th>B</th></tr>
</thead>
<tbody>
<tr><td></td><td>val</td></tr>
</tbody>
</table>
"""

TABLE_NO_HEADERS_HTML = """\
<table>
<tbody>
<tr><td>A</td><td>B</td></tr>
</tbody>
</table>
"""

TABLE_NESTED_CONTENT_HTML = """\
<table>
<thead>
<tr><th>描述</th></tr>
</thead>
<tbody>
<tr><td>line1
line2</td></tr>
</tbody>
</table>
"""

TABLE_MISSING_CELLS_HTML = """\
<table>
<thead>
<tr><th>A</th><th>B</th><th>C</th></tr>
</thead>
<tbody>
<tr><td>1</td></tr>
</tbody>
</table>
"""


# =============================================================================
# Test data: Markdown content
# =============================================================================

SUPPORT_SECTION_MD = """\
# aclnnAbs

## 产品支持情况
<table>
<thead>
<tr><th>产品</th><th>版本</th></tr>
</thead>
<tbody>
<tr><td>Atlas 800</td><td>A2</td></tr>
</tbody>
</table>

## 功能说明
some content
"""

SUPPORT_ONLY_MD = """\
# aclnnAbs

## 产品支持情况
<table>
<thead>
<tr><th>产品</th><th>版本</th></tr>
</thead>
<tbody>
<tr><td>Atlas 800</td><td>A2</td></tr>
</tbody>
</table>
"""

NO_SUPPORT_SECTION_MD = """\
# aclnnAbs

## 功能说明
some content
"""

SUMMARY_WITH_FORMULA_MD = """\
# aclnnAbs

## 功能说明
- 接口功能：计算输入张量的绝对值。

- 计算公式：$$y = |x|$$

## 函数原型
next section
"""

SUMMARY_NO_FORMULA_MD = """\
# aclnnAbs

## 功能说明
- 接口功能：计算输入张量的绝对值。

## 函数原型
next section
"""

NO_SUMMARY_SECTION_MD = """\
# aclnnAbs

## 函数原型
next section
"""

FUNCTION_PROTO_MD = """\
# aclnnAbs

## 函数原型
两段式接口说明文字

```cpp
aclnnStatus aclnnAbsGetWorkspaceSize(const aclTensor *self, uint64_t *workspaceSize)
```

## 约束说明
next section
"""

NO_FUNCTION_PROTO_MD = """\
# aclnnAbs

## 约束说明
some content
"""

SUB_FUNCTIONS_MD = """\
# aclnnAbs

## aclnnAbsGetWorkspaceSize

<table>
<thead>
<tr><th>参数</th><th>类型</th></tr>
</thead>
<tbody>
<tr><td>self</td><td>输入张量</td></tr>
</tbody>
</table>

产品限制说明

**返回值
aclnnStatus 返回值参见 aclnn 返回码。

第一段接口会完成入参校验，出现以下场景时报错：
<table>
<thead>
<tr><th>错误码</th><th>描述</th></tr>
</thead>
<tbody>
<tr><td>ACLNN_ERR_PARAM</td><td>参数错误</td></tr>
</tbody>
</table>

## aclnnAbs

<table>
<thead>
<tr><th>参数</th><th>类型</th></tr>
</thead>
<tbody>
<tr><td>out</td><td>输出张量</td></tr>
</tbody>
</table>

**返回值
aclnnStatus 返回值参见 aclnn 返回码。

## 约束说明
some constraint
"""

NO_SUB_FUNCTIONS_MD = """\
# aclnnAbs

## 约束说明
some constraint
"""

CONSTRAINT_SECTION_MD = """\
# aclnnAbs

## 约束说明
输入张量的数据类型支持 float16、float32。

## 调用示例
next section
"""

NO_CONSTRAINT_SECTION_MD = """\
# aclnnAbs

## 调用示例
next section
"""

EXAMPLE_SECTION_MD = """\
# aclnnAbs

## 调用示例
示例代码描述

```cpp
// 示例代码
aclnnAbs(addr, executor, stream);
```
"""

NO_EXAMPLE_SECTION_MD = """\
# aclnnAbs

## 约束说明
some constraint
"""

EXAMPLE_MULTI_CODE_BLOCKS_MD = """\
# aclnnAbs

## 调用示例
描述一

```cpp
code block 1
```

描述二

```python
code block 2
```
"""

EXAMPLE_NO_CODE_BLOCKS_MD = """\
# aclnnAbs

## 调用示例
这是一段描述但没有代码块。
"""

COMPLETE_OPERATOR_MD = """\
# aclnnAbs

## 产品支持情况
<table>
<thead>
<tr><th>产品</th><th>版本</th></tr>
</thead>
<tbody>
<tr><td>Atlas 800</td><td>A2</td></tr>
</tbody>
</table>

## 功能说明
- 接口功能：计算输入张量的绝对值。

- 计算公式：$$y = |x|$$

## 函数原型
两段式接口说明文字

```cpp
aclnnStatus aclnnAbsGetWorkspaceSize(const aclTensor *self, uint64_t *workspaceSize)
```

## aclnnAbsGetWorkspaceSize

<table>
<thead>
<tr><th>参数</th><th>类型</th></tr>
</thead>
<tbody>
<tr><td>self</td><td>输入张量</td></tr>
</tbody>
</table>

产品限制说明

**返回值
aclnnStatus 返回值参见 aclnn 返回码。

第一段接口会完成入参校验，出现以下场景时报错：
<table>
<thead>
<tr><th>错误码</th><th>描述</th></tr>
</thead>
<tbody>
<tr><td>ACLNN_ERR_PARAM</td><td>参数错误</td></tr>
</tbody>
</table>

## aclnnAbs

<table>
<thead>
<tr><th>参数</th><th>类型</th></tr>
</thead>
<tbody>
<tr><td>out</td><td>输出张量</td></tr>
</tbody>
</table>

**返回值
aclnnStatus 返回值参见 aclnn 返回码。

## 约束说明
输入张量的数据类型支持 float16、float32。

## 调用示例
示例代码描述

```cpp
// 示例代码
aclnnAbs(addr, executor, stream);
```
"""

PARTIAL_OPERATOR_MD = """\
# aclnnAbs

## 功能说明
- 接口功能：计算输入张量的绝对值。

## 约束说明
输入张量的数据类型支持 float16、float32。
"""


# =============================================================================
# Tests for TableParser
# =============================================================================


class TestTableParser:
    """Tests for TableParser class."""

    def test_parser_initial_state(self):
        """测试TableParser初始状态"""
        parser = TableParser()
        assert parser.tables == []
        assert parser.current_table == []
        assert parser.current_row == []
        assert parser.current_cell == []
        assert parser.headers == []
        assert parser.in_thead is False
        assert parser.in_tbody is False
        assert parser.in_th is False
        assert parser.in_td is False

    def test_simple_table_with_thead_tbody(self):
        """测试带thead和tbody的简单表格"""
        parser = TableParser()
        parser.feed(SIMPLE_TABLE_HTML)
        assert len(parser.tables) == 1
        table = parser.tables[0]
        assert table["headers"] == ["参数", "类型"]
        assert len(table["rows"]) == 2
        assert table["rows"][0] == ["self", "输入张量"]
        assert table["rows"][1] == ["out", "输出张量"]

    def test_multiple_tables(self):
        """测试多个表格的解析"""
        parser = TableParser()
        parser.feed(MULTI_TABLE_HTML)
        assert len(parser.tables) == 2

    def test_table_with_empty_cells(self):
        """测试包含空单元格的表格"""
        parser = TableParser()
        parser.feed(TABLE_EMPTY_CELLS_HTML)
        assert len(parser.tables) == 1
        table = parser.tables[0]
        assert table["rows"][0] == ["", "val"]

    def test_table_no_headers(self):
        """测试没有thead的表格不保存"""
        parser = TableParser()
        parser.feed(TABLE_NO_HEADERS_HTML)
        assert len(parser.tables) == 0

    def test_table_with_nested_content(self):
        """测试包含多行内容的表格单元格"""
        parser = TableParser()
        parser.feed(TABLE_NESTED_CONTENT_HTML)
        assert len(parser.tables) == 1
        table = parser.tables[0]
        assert table["rows"][0] == ["line1\nline2"]


# =============================================================================
# Tests for html_table_to_json
# =============================================================================


class TestHtmlTableToJson:
    """Tests for html_table_to_json function."""

    def test_normal_table_to_dict_list(self):
        """测试正常表格转换为字典列表"""
        result = html_table_to_json(SIMPLE_TABLE_HTML)
        assert len(result) == 2
        assert result[0] == {"参数": "self", "类型": "输入张量"}
        assert result[1] == {"参数": "out", "类型": "输出张量"}

    def test_empty_string_returns_empty_list(self):
        """测试空字符串返回空列表"""
        result = html_table_to_json("")
        assert result == []

    def test_invalid_html_returns_empty_list(self):
        """测试无效HTML返回空列表"""
        result = html_table_to_json("not a table at all")
        assert result == []

    def test_table_with_missing_cells_fills_empty(self):
        """测试单元格不足时用空字符串填充"""
        result = html_table_to_json(TABLE_MISSING_CELLS_HTML)
        assert len(result) == 1
        assert result[0] == {"A": "1", "B": "", "C": ""}

    def test_table_with_empty_cells_to_json(self):
        """测试空单元格在JSON中的表现"""
        result = html_table_to_json(TABLE_EMPTY_CELLS_HTML)
        assert len(result) == 1
        assert result[0] == {"A": "", "B": "val"}

    def test_multiple_tables_to_json(self):
        """测试多个表格转换为JSON"""
        result = html_table_to_json(MULTI_TABLE_HTML)
        assert len(result) >= 2


# =============================================================================
# Tests for extract_support_section
# =============================================================================


class TestExtractSupportSection:
    """Tests for extract_support_section function."""

    def test_content_with_support_section(self):
        """测试包含产品支持情况的内容"""
        result = extract_support_section(SUPPORT_SECTION_MD)
        assert "origin_txt" in result
        assert "support_matrix" in result
        assert "产品支持情况" in result["origin_txt"]
        assert "<table>" in result["support_matrix"]

    def test_content_without_support_section(self):
        """测试不包含产品支持情况的内容"""
        result = extract_support_section(NO_SUPPORT_SECTION_MD)
        assert result == {}

    def test_support_section_origin_txt_contains_header(self):
        """测试origin_txt包含章节标题"""
        result = extract_support_section(SUPPORT_SECTION_MD)
        assert result["origin_txt"].startswith("## 产品支持情况")


# =============================================================================
# Tests for extract_summary_section
# =============================================================================


class TestExtractSummarySection:
    """Tests for extract_summary_section function."""

    def test_content_with_summary_and_formula(self):
        """测试包含功能说明和计算公式的内容"""
        result = extract_summary_section(SUMMARY_WITH_FORMULA_MD)
        assert "origin_txt" in result
        assert "summary_clarify" in result
        assert "formula" in result
        assert result["summary_clarify"] == "计算输入张量的绝对值。"
        assert "$$y = |x|$$" in result["formula"]

    def test_content_without_summary(self):
        """测试不包含功能说明的内容"""
        result = extract_summary_section(NO_SUMMARY_SECTION_MD)
        assert result == {}

    def test_summary_without_formula(self):
        """测试有功能说明但无计算公式的内容"""
        result = extract_summary_section(SUMMARY_NO_FORMULA_MD)
        assert "origin_txt" in result
        assert "summary_clarify" in result
        assert "formula" not in result

    def test_summary_clarify_extraction(self):
        """测试接口功能说明的提取"""
        result = extract_summary_section(SUMMARY_WITH_FORMULA_MD)
        assert result["summary_clarify"] == "计算输入张量的绝对值。"


# =============================================================================
# Tests for extract_function_proto_section
# =============================================================================


class TestExtractFunctionProtoSection:
    """Tests for extract_function_proto_section function."""

    def test_content_with_function_proto(self):
        """测试包含函数原型章节的内容"""
        result = extract_function_proto_section(FUNCTION_PROTO_MD)
        assert "origin_txt" in result
        assert "function_proto_clarify" in result
        assert "sub_function_sign" in result

    def test_content_without_function_proto(self):
        """测试不包含函数原型章节的内容"""
        result = extract_function_proto_section(NO_FUNCTION_PROTO_MD)
        assert result == {}

    def test_function_proto_clarify(self):
        """测试函数原型说明文字的提取"""
        result = extract_function_proto_section(FUNCTION_PROTO_MD)
        assert result["function_proto_clarify"] == "两段式接口说明文字"

    def test_sub_function_sign(self):
        """测试子函数签名的提取"""
        result = extract_function_proto_section(FUNCTION_PROTO_MD)
        assert isinstance(result["sub_function_sign"], list)
        assert len(result["sub_function_sign"]) == 1
        assert "aclnnAbsGetWorkspaceSize" in result["sub_function_sign"][0]


# =============================================================================
# Tests for is_second_stage_function
# =============================================================================


class TestIsSecondStageFunction:
    """Tests for is_second_stage_function function."""

    def test_get_workspace_size_returns_false(self):
        """测试以GetWorkspaceSize结尾的函数返回False"""
        assert is_second_stage_function("aclnnAbsGetWorkspaceSize") is False

    def test_non_get_workspace_size_returns_true(self):
        """测试不以GetWorkspaceSize结尾的函数返回True"""
        assert is_second_stage_function("aclnnAbs") is True

    def test_empty_string_returns_true(self):
        """测试空字符串返回True"""
        assert is_second_stage_function("") is True


# =============================================================================
# Tests for extract_sub_functions
# =============================================================================


class TestExtractSubFunctions:
    """Tests for extract_sub_functions function."""

    def test_multiple_sub_functions(self):
        """测试提取多个子函数"""
        result = extract_sub_functions(SUB_FUNCTIONS_MD)
        assert len(result) == 2
        assert result[0]["changeconfig"] == "aclnnAbsGetWorkspaceSize"
        assert result[1]["changeconfig"] == "aclnnAbs"

    def test_first_stage_function_has_error_code(self):
        """测试第一段接口包含错误码信息"""
        result = extract_sub_functions(SUB_FUNCTIONS_MD)
        first_stage = result[0]
        assert first_stage["changeconfig"] == "aclnnAbsGetWorkspaceSize"
        assert "changeconfig_error" in first_stage
        assert "errorcode" in first_stage

    def test_second_stage_function_no_error_code(self):
        """测试第二段接口不包含错误码信息"""
        result = extract_sub_functions(SUB_FUNCTIONS_MD)
        second_stage = result[1]
        assert second_stage["changeconfig"] == "aclnnAbs"
        assert "changeconfig_error" not in second_stage
        assert "errorcode" not in second_stage

    def test_function_with_param_matrix_table(self):
        """测试包含参数矩阵表格的子函数"""
        result = extract_sub_functions(SUB_FUNCTIONS_MD)
        assert "param_matrix_table" in result[0]
        assert "param_matrix" in result[0]
        assert isinstance(result[0]["param_matrix"], list)
        assert len(result[0]["param_matrix"]) == 1
        assert result[0]["param_matrix"][0] == {"参数": "self", "类型": "输入张量"}

    def test_function_with_return_value(self):
        """测试包含返回值说明的子函数"""
        result = extract_sub_functions(SUB_FUNCTIONS_MD)
        assert result[0]["fixconfig_1"] == "返回值"
        assert "aclnnStatus" in result[0]["fixconfig_2"]

    def test_no_sub_functions_returns_empty(self):
        """测试无子函数时返回空列表"""
        result = extract_sub_functions(NO_SUB_FUNCTIONS_MD)
        assert result == []

    def test_sub_function_changeconfig(self):
        """测试子函数名称的提取"""
        result = extract_sub_functions(SUB_FUNCTIONS_MD)
        assert result[0]["changeconfig"] == "aclnnAbsGetWorkspaceSize"
        assert result[1]["changeconfig"] == "aclnnAbs"

    def test_sub_function_origin_txt(self):
        """测试子函数的origin_txt字段"""
        result = extract_sub_functions(SUB_FUNCTIONS_MD)
        assert "origin_txt" in result[0]
        assert "aclnnAbsGetWorkspaceSize" in result[0]["origin_txt"]
        assert "origin_txt" in result[1]
        assert "aclnnAbs" in result[1]["origin_txt"]

    def test_first_stage_function_describe(self):
        """测试第一段接口的describe字段"""
        result = extract_sub_functions(SUB_FUNCTIONS_MD)
        assert "describe" in result[0]
        assert result[0]["describe"] == "产品限制说明"

    def test_second_stage_function_no_describe(self):
        """测试第二段接口没有describe字段"""
        result = extract_sub_functions(SUB_FUNCTIONS_MD)
        assert "describe" not in result[1]

    def test_second_stage_function_has_return_value(self):
        """测试第二段接口也有返回值说明"""
        result = extract_sub_functions(SUB_FUNCTIONS_MD)
        assert result[1]["fixconfig_1"] == "返回值"
        assert "aclnnStatus" in result[1]["fixconfig_2"]

    def test_sub_functions_with_sample_md(self, sample_md_content):
        """测试使用sample_md_content提取子函数"""
        result = extract_sub_functions(sample_md_content)
        assert len(result) == 2
        assert result[0]["changeconfig"] == "aclnnAbsGetWorkspaceSize"
        assert result[1]["changeconfig"] == "aclnnAbs"


# =============================================================================
# Tests for extract_constraint_section
# =============================================================================


class TestExtractConstraintSection:
    """Tests for extract_constraint_section function."""

    def test_content_with_constraint(self):
        """测试包含约束说明的内容"""
        result = extract_constraint_section(CONSTRAINT_SECTION_MD)
        assert "origin_txt" in result
        assert "constrain_clarify" in result
        assert "float16" in result["constrain_clarify"]

    def test_content_without_constraint(self):
        """测试不包含约束说明的内容"""
        result = extract_constraint_section(NO_CONSTRAINT_SECTION_MD)
        assert result == {}


# =============================================================================
# Tests for extract_example_section
# =============================================================================


class TestExtractExampleSection:
    """Tests for extract_example_section function."""

    def test_content_with_example(self):
        """测试包含调用示例的内容"""
        result = extract_example_section(EXAMPLE_SECTION_MD)
        assert "origin_txt" in result
        assert "fixconfig" in result
        assert result["fixconfig"] == "调用示例"
        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["describe"] == "示例代码描述"
        assert "aclnnAbs" in result["content"][0]["code"]

    def test_content_without_example(self):
        """测试不包含调用示例的内容"""
        result = extract_example_section(NO_EXAMPLE_SECTION_MD)
        assert result == {}

    def test_example_multiple_code_blocks(self):
        """测试包含多个代码块的调用示例"""
        result = extract_example_section(EXAMPLE_MULTI_CODE_BLOCKS_MD)
        assert "content" in result
        assert len(result["content"]) == 2
        assert result["content"][0]["describe"] == "描述一"
        assert result["content"][0]["code"] == "code block 1"
        assert result["content"][1]["describe"] == "描述二"
        assert result["content"][1]["code"] == "code block 2"

    def test_example_no_code_blocks(self):
        """测试调用示例中没有代码块的情况"""
        result = extract_example_section(EXAMPLE_NO_CODE_BLOCKS_MD)
        assert "origin_txt" in result
        assert "fixconfig" not in result
        assert "content" not in result


# =============================================================================
# Tests for parse_md_file
# =============================================================================


class TestParseMdFile:
    """Tests for parse_md_file function."""

    def test_parse_complete_md_file(self, tmp_path):
        """测试解析完整的算子md文件"""
        md_file = tmp_path / "aclnn_abs.md"
        md_file.write_text(COMPLETE_OPERATOR_MD, encoding="utf-8")
        result = parse_md_file(str(md_file))
        assert isinstance(result, dict)
        assert "support" in result
        assert "summary" in result
        assert "function_proto" in result
        assert "constraint" in result
        assert "example" in result

    def test_parse_partial_md_file(self, tmp_path):
        """测试解析部分章节的md文件"""
        md_file = tmp_path / "aclnn_abs.md"
        md_file.write_text(PARTIAL_OPERATOR_MD, encoding="utf-8")
        result = parse_md_file(str(md_file))
        assert isinstance(result, dict)
        assert "summary" in result
        assert "constraint" in result
        assert "support" not in result
        assert "function_proto" not in result
        assert "example" not in result

    def test_parse_file_with_sub_functions(self, tmp_path):
        """测试解析包含子函数的md文件"""
        md_file = tmp_path / "aclnn_abs.md"
        md_file.write_text(COMPLETE_OPERATOR_MD, encoding="utf-8")
        result = parse_md_file(str(md_file))
        assert "sub_function_hash_1" in result
        assert "sub_function_hash_2" in result
        assert (
            result["sub_function_hash_1"]["changeconfig"] == "aclnnAbsGetWorkspaceSize"
        )
        assert result["sub_function_hash_2"]["changeconfig"] == "aclnnAbs"

    def test_parse_with_sample_md_content(self, tmp_path, sample_md_content):
        """测试使用conftest提供的sample_md_content解析"""
        md_file = tmp_path / "aclnn_abs.md"
        md_file.write_text(sample_md_content, encoding="utf-8")
        result = parse_md_file(str(md_file))
        assert isinstance(result, dict)
        assert "support" in result
        assert "summary" in result

    def test_parse_md_file_support_only(self, tmp_path):
        """测试只包含产品支持情况的md文件"""
        md_file = tmp_path / "aclnn_abs.md"
        md_file.write_text(SUPPORT_ONLY_MD, encoding="utf-8")
        result = parse_md_file(str(md_file))
        assert "support" in result
        assert "summary" not in result
        assert "constraint" not in result


# =============================================================================
# Tests for main
# =============================================================================


class TestMain:
    """Tests for main function."""

    def test_main_creates_json_files(self, tmp_path):
        """测试main函数创建JSON文件"""
        md_folder = tmp_path / "md_input"
        md_folder.mkdir()
        sub_dir = md_folder / "sub"
        sub_dir.mkdir()
        md_file = sub_dir / "aclnn_abs.md"
        md_file.write_text(COMPLETE_OPERATOR_MD, encoding="utf-8")

        json_folder = tmp_path / "json_output"

        result = main(str(md_folder), str(json_folder))

        assert isinstance(result, dict)
        assert len(result) == 1
        assert json_folder.exists()
        json_files = list(json_folder.glob("*.json"))
        assert len(json_files) == 1
        assert json_files[0].name == "aclnn_abs.json"

    def test_main_empty_folder_returns_empty_dict(self, tmp_path):
        """测试空文件夹返回空字典"""
        md_folder = tmp_path / "md_empty"
        md_folder.mkdir()
        json_folder = tmp_path / "json_output"

        result = main(str(md_folder), str(json_folder))

        assert result == {}

    def test_main_json_output_is_valid(self, tmp_path):
        """测试输出的JSON文件是有效的JSON"""
        md_folder = tmp_path / "md_input"
        md_folder.mkdir()
        sub_dir = md_folder / "sub"
        sub_dir.mkdir()
        md_file = sub_dir / "aclnn_abs.md"
        md_file.write_text(COMPLETE_OPERATOR_MD, encoding="utf-8")

        json_folder = tmp_path / "json_output"

        main(str(md_folder), str(json_folder))

        json_files = list(json_folder.glob("*.json"))
        assert len(json_files) == 1

        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert "support" in data

    def test_main_creates_output_folder_if_not_exists(self, tmp_path):
        """测试输出文件夹不存在时自动创建"""
        md_folder = tmp_path / "md_input"
        md_folder.mkdir()
        sub_dir = md_folder / "sub"
        sub_dir.mkdir()
        md_file = sub_dir / "aclnn_abs.md"
        md_file.write_text(COMPLETE_OPERATOR_MD, encoding="utf-8")

        json_folder = tmp_path / "nested" / "json_output"

        main(str(md_folder), str(json_folder))

        assert json_folder.exists()

    def test_main_multiple_md_files(self, tmp_path):
        """测试处理多个md文件"""
        md_folder = tmp_path / "md_input"
        md_folder.mkdir()
        sub_dir = md_folder / "sub"
        sub_dir.mkdir()

        md_file1 = sub_dir / "aclnn_abs.md"
        md_file1.write_text(COMPLETE_OPERATOR_MD, encoding="utf-8")

        md_file2 = sub_dir / "aclnn_add.md"
        md_file2.write_text(PARTIAL_OPERATOR_MD, encoding="utf-8")

        json_folder = tmp_path / "json_output"

        result = main(str(md_folder), str(json_folder))

        assert len(result) == 2
        json_files = list(json_folder.glob("*.json"))
        assert len(json_files) == 2

    def test_main_copies_original_md_to_doc_folder(self, tmp_path):
        """测试main函数将原始md文档复制到doc_json同级的doc文件夹"""
        md_folder = tmp_path / "md_input"
        md_folder.mkdir()
        sub_dir = md_folder / "sub"
        sub_dir.mkdir()
        md_file = sub_dir / "aclnn_abs.md"
        md_file.write_text(COMPLETE_OPERATOR_MD, encoding="utf-8")

        json_folder = tmp_path / "json_output"

        main(str(md_folder), str(json_folder))

        doc_folder = tmp_path / "doc"
        copied_md = doc_folder / "aclnn_abs.md"
        assert copied_md.exists()
        assert copied_md.read_text(encoding="utf-8") == COMPLETE_OPERATOR_MD
        assert sorted(p.name for p in json_folder.iterdir()) == ["aclnn_abs.json"]
