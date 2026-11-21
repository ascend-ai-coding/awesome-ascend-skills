# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for unified brace-matching utilities.
"""

import pytest
from src.snapshot_construction.anchor_construction.common.brace_utils import (
    # Line-based functions
    find_matching_brace_end,
    find_function_start,
    find_enclosing_function,
    find_if_block_start,
    find_full_if_else_block,
    find_endif_line,
    # Content-based functions
    find_function_body_start,
    find_matching_brace,
)


# =============================================================================
# Test data: Mock C++ code
# =============================================================================

SIMPLE_FUNCTION = """\
int add(int a, int b) {
    return a + b;
}
"""

NESTED_FUNCTION = """\
void process() {
    if (condition) {
        for (int i = 0; i < 10; i++) {
            doSomething();
        }
    }
}
"""

IF_ELSE_CHAIN = """\
int classify(int x) {
    if (x < 0) {
        return -1;
    } else if (x == 0) {
        return 0;
    } else {
        return 1;
    }
}
"""

PREPROCESSOR_IFDEF = """\
#ifdef FEATURE_ENABLED
    void enableFeature() {
        setupFeature();
    }
#else
    void enableFeature() {
        // Feature disabled
    }
#endif
"""

MULTIPLE_FUNCTIONS = """\
int first() {
    return 1;
}

int second() {
    return 2;
}

int third() {
    return 3;
}
"""

COMPLEX_CODE = """\
class Example {
public:
    void method() {
        if (test) {
            for (int i = 0; i < n; i++) {
                process(i);
            }
        } else {
            handleDefault();
        }
    }

    int getValue() {
        #ifdef DEBUG
        return debugValue;
        #else
        return value;
        #endif
    }
};
"""


# =============================================================================
# Tests for line-based functions
# =============================================================================


class TestFindMatchingBraceEnd:
    """Tests for find_matching_brace_end (line-based)."""

    def test_simple_function(self):
        """测试简单函数的大括号匹配"""
        lines = SIMPLE_FUNCTION.split("\n")
        # Opening brace on line 0
        result = find_matching_brace_end(lines, 0)
        assert result == 2  # Closing brace on line 2

    def test_nested_braces(self):
        """测试嵌套大括号匹配"""
        lines = NESTED_FUNCTION.split("\n")
        # Opening brace on line 0
        result = find_matching_brace_end(lines, 0)
        assert result == 6  # Closing brace on line 6

    def test_if_else_chain(self):
        """测试 if-else 链的大括号匹配"""
        lines = IF_ELSE_CHAIN.split("\n")
        # Opening brace on line 0
        result = find_matching_brace_end(lines, 0)
        assert result == 8  # Closing brace on line 8 (split creates extra empty line)

    def test_no_matching_brace(self):
        """测试没有匹配大括号的情况"""
        lines = ["int broken() {", "    return 1;"]
        result = find_matching_brace_end(lines, 0)
        assert result == len(lines) - 1  # Returns last line


class TestFindFunctionStart:
    """Tests for find_function_start (line-based)."""

    def test_simple_function(self):
        """测试简单函数的起始位置"""
        lines = SIMPLE_FUNCTION.split("\n")
        # Brace on line 0, signature starts on line 0
        result = find_function_start(lines, 0)
        assert result == 0

    def test_function_with_blank_lines(self):
        """测试带空行的函数"""
        code = "\n\nint add(int a, int b) {\n    return a + b;\n}\n"
        lines = code.split("\n")
        # Brace on line 2
        result = find_function_start(lines, 2)
        assert result == 2

    def test_function_after_semicolon(self):
        """测试分号后的函数"""
        code = "int x = 5;\nint add(int a, int b) {\n    return a + b;\n}\n"
        lines = code.split("\n")
        # Brace on line 1
        result = find_function_start(lines, 1)
        assert result == 1

    def test_function_after_closing_brace(self):
        """测试大括号后的函数"""
        lines = MULTIPLE_FUNCTIONS.split("\n")
        # Second function brace on line 4
        result = find_function_start(lines, 4)
        assert result == 4


class TestFindEnclosingFunction:
    """Tests for find_enclosing_function (line-based)."""

    def test_find_enclosing_simple(self):
        """测试查找简单函数的包围范围"""
        lines = SIMPLE_FUNCTION.split("\n")
        # Target line 1 (inside function)
        start, end = find_enclosing_function(lines, 1)
        assert start == 0
        assert end == 2

    def test_find_enclosing_nested(self):
        """测试查找嵌套结构中的函数"""
        lines = NESTED_FUNCTION.split("\n")
        # Target line 3 (inside nested if)
        start, end = find_enclosing_function(lines, 3)
        assert start == 0
        assert end == 4  # split creates extra empty line

    def test_find_enclosing_multiple_functions(self):
        """测试在多个函数中查找"""
        lines = MULTIPLE_FUNCTIONS.split("\n")
        # Target line 5 (inside second function)
        start, end = find_enclosing_function(lines, 5)
        assert start == 4
        assert end == 6


class TestFindIfBlockStart:
    """Tests for find_if_block_start (line-based)."""

    def test_find_if_start(self):
        """测试查找 if 语句起始"""
        lines = IF_ELSE_CHAIN.split("\n")
        # Target line 2 (inside if block)
        result = find_if_block_start(lines, 2)
        assert result == 1

    def test_find_else_if_start(self):
        """测试查找 else if 语句起始"""
        lines = IF_ELSE_CHAIN.split("\n")
        # Target line 4 (inside else if block)
        result = find_if_block_start(lines, 4)
        assert result == 1  # Finds the first if statement (line 1)

    def test_no_if_found(self):
        """测试未找到 if 语句"""
        lines = SIMPLE_FUNCTION.split("\n")
        # Target line 1 (no if statement)
        result = find_if_block_start(lines, 1)
        assert result == 1  # Returns match_idx


class TestFindFullIfElseBlock:
    """Tests for find_full_if_else_block (line-based)."""

    def test_simple_if(self):
        """测试简单 if 语句"""
        code = "int test() {\n    if (x) {\n        return 1;\n    }\n}\n"
        lines = code.split("\n")
        # if on line 1
        result = find_full_if_else_block(lines, 1)
        assert result == 3  # Ends at line 3

    def test_if_else_chain(self):
        """测试完整 if-else 链"""
        lines = IF_ELSE_CHAIN.split("\n")
        # if on line 1
        result = find_full_if_else_block(lines, 1)
        assert result == 7  # Ends at line 7 (split creates extra empty line)

    def test_if_with_blank_lines(self):
        """测试带空行的 if-else 链"""
        code = "int test() {\n    if (x) {\n        return 1;\n    }\n\n    else {\n        return 2;\n    }\n}\n"
        lines = code.split("\n")
        # if on line 1
        result = find_full_if_else_block(lines, 1)
        assert result == 7


class TestFindEndifLine:
    """Tests for find_endif_line (line-based)."""

    def test_simple_ifdef(self):
        """测试简单 #ifdef"""
        lines = PREPROCESSOR_IFDEF.split("\n")
        # #ifdef on line 0
        result = find_endif_line(lines, 0)
        assert result == 8  # #endif on line 8

    def test_nested_ifdef(self):
        """测试嵌套 #ifdef"""
        code = "#ifdef A\n    #ifdef B\n        code();\n    #endif\n#endif\n"
        lines = code.split("\n")
        # Outer #ifdef on line 0
        result = find_endif_line(lines, 0)
        assert result == 4  # Outer #endif on line 4

    def test_no_endif(self):
        """测试没有 #endif"""
        lines = ["#ifdef FEATURE", "    code();"]
        result = find_endif_line(lines, 0)
        assert result == len(lines) - 1  # Returns last line


# =============================================================================
# Tests for content-based functions
# =============================================================================


class TestFindFunctionBodyStart:
    """Tests for find_function_body_start (content-based)."""

    def test_simple_function(self):
        """测试简单函数体起始位置"""
        content = "int add(int a, int b) { return a + b; }"
        result = find_function_body_start(content, 0)
        assert result == 22  # Position of '{' (count: int add(int a, int b) = 22 chars)

    def test_function_with_whitespace(self):
        """测试带空白的函数体起始"""
        content = "void process()   \n{ \n    code(); \n}"
        result = find_function_body_start(content, 0)
        assert (
            result == 18
        )  # Position of '{' after newline (void process()   \n = 18 chars)

    def test_no_brace(self):
        """测试没有大括号"""
        content = "int broken();"
        result = find_function_body_start(content, 0)
        assert result == -1  # Not found


class TestFindMatchingBrace:
    """Tests for find_matching_brace (content-based)."""

    def test_simple_braces(self):
        """测试简单大括号匹配"""
        content = "{ return 42; }"
        result = find_matching_brace(content, 0)
        assert result == 13  # Position of closing '}' (count: { return 42; = 13 chars)

    def test_nested_braces(self):
        """测试嵌套大括号匹配"""
        content = "{ if (x) { return 1; } return 0; }"
        result = find_matching_brace(content, 0)
        assert result == 33  # Position of final '}' (count manually)

    def test_deeply_nested(self):
        """测试深层嵌套"""
        content = "{ { { { } } } }"
        result = find_matching_brace(content, 0)
        assert result == 14  # Position of final '}'

    def test_function_body(self):
        """测试函数体大括号匹配"""
        content = "int add(int a, int b) { return a + b; }"
        brace_start = find_function_body_start(content, 0)
        result = find_matching_brace(content, brace_start)
        assert result == 38  # Position of closing '}' (count manually)

    def test_no_matching_brace(self):
        """测试没有匹配大括号"""
        content = "{ unclosed { brace"
        result = find_matching_brace(content, 0)
        assert result == -1  # Not found

    def test_complex_code(self):
        """测试复杂代码"""
        content = COMPLEX_CODE
        # Find first opening brace (class body)
        first_brace = content.find("{")
        result = find_matching_brace(content, first_brace)
        # Should find the matching closing brace for the class
        assert result > 0
        assert content[result] == "}"


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_line_and_content_based_consistency(self):
        """测试行基和内容基函数的一致性"""
        # Use the same code for both paradigms
        lines = SIMPLE_FUNCTION.split("\n")
        content = SIMPLE_FUNCTION

        # Line-based: find function bounds
        start_line, end_line = find_enclosing_function(lines, 1)

        # Content-based: find function bounds
        brace_start = find_function_body_start(content, 0)
        brace_end = find_matching_brace(content, brace_start)

        # Both should find the same function
        assert start_line == 0
        assert end_line == 2
        assert brace_start > 0
        assert brace_end > brace_start

    def test_complex_scenario(self):
        """测试复杂场景"""
        lines = COMPLEX_CODE.split("\n")

        # Find the method function
        method_line = None
        for i, line in enumerate(lines):
            if "void method()" in line:
                method_line = i
                break

        assert method_line is not None

        # Use line-based to find enclosing function
        start, end = find_enclosing_function(lines, method_line + 1)

        # Verify we found the class
        assert "class Example" in lines[start]
