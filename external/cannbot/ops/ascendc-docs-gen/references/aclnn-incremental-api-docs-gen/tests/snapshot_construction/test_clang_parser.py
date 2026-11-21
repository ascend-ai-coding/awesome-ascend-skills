# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for ClangParser class — pure methods only.

Tests build_clang_args(), make_symbol_id(), and should_skip().
Does NOT test parse_file, get_parent_function, or extract_callee_from_tokens
(those require real libclang AST nodes and are integration tests).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.snapshot_construction.code_graph.clang_parser import ClangParser

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def parser():
    """创建 ClangParser 实例，mock 掉 libclang 的 Index.create()"""
    with patch("src.snapshot_construction.code_graph.clang_parser.Index") as mock_index:
        mock_index.create.return_value = MagicMock()
        p = ClangParser("/tmp/fake_project")
        yield p


# =============================================================================
# Tests for build_clang_args()
# =============================================================================


class TestBuildClangArgs:
    """测试 build_clang_args() 方法"""

    def test_returns_list(self, parser):
        """测试返回值是列表类型"""
        args = parser.build_clang_args()
        assert isinstance(args, list)

    def test_contains_language_cpp(self, parser):
        """测试包含 -x c++ 语言标识"""
        args = parser.build_clang_args()
        assert "-x" in args
        idx = args.index("-x")
        assert args[idx + 1] == "c++"

    def test_contains_std_cpp17(self, parser):
        """测试包含 -std=c++17 标准"""
        args = parser.build_clang_args()
        assert "-std=c++17" in args

    def test_contains_aclnn_api_macro(self, parser):
        """测试包含 -DACLNN_API= 宏定义"""
        args = parser.build_clang_args()
        assert "-DACLNN_API=" in args

    def test_contains_include_path(self, parser):
        """测试包含 -I 加项目根目录"""
        args = parser.build_clang_args()
        expected = "-I" + parser.project_root
        assert expected in args

    def test_total_args_count(self, parser):
        """测试参数总数为 5"""
        args = parser.build_clang_args()
        assert len(args) == 5


# =============================================================================
# Tests for make_symbol_id()
# =============================================================================


class TestMakeSymbolId:
    """测试 make_symbol_id() 方法"""

    def test_simple_name(self, parser):
        """测试简单函数名直接返回"""
        assert parser.make_symbol_id("myFunc") == "myFunc"

    def test_empty_string(self, parser):
        """测试空字符串返回空字符串"""
        assert parser.make_symbol_id("") == ""

    def test_name_with_namespace(self, parser):
        """测试带命名空间的名称（当前实现直接返回原名）"""
        assert parser.make_symbol_id("ns::Func") == "ns::Func"


# =============================================================================
# Tests for should_skip()
# =============================================================================


class TestShouldSkip:
    """测试 should_skip() 方法"""

    def test_path_with_tests(self, parser):
        """测试包含 /tests/ 的路径返回 True"""
        assert parser.should_skip("/home/user/tests/file.cpp") is True

    def test_path_with_examples(self, parser):
        """测试包含 /examples/ 的路径返回 True"""
        assert parser.should_skip("/home/user/examples/file.cpp") is True

    def test_normal_path(self, parser):
        """测试普通路径返回 False"""
        assert parser.should_skip("/home/user/src/file.cpp") is False

    def test_windows_path_with_tests(self, parser):
        """测试 Windows 风格反斜杠路径含 tests 返回 True"""
        assert parser.should_skip("C:\\Users\\project\\tests\\file.cpp") is True

    def test_windows_path_without_tests(self, parser):
        """测试 Windows 风格反斜杠路径不含 tests/examples 返回 False"""
        assert parser.should_skip("C:\\Users\\project\\src\\file.cpp") is False

    def test_path_with_tests_substring_not_slash_delimited(self, parser):
        """测试路径含 tests 子串但无斜线分隔时返回 False"""
        assert parser.should_skip("/home/usertests/file.cpp") is False
