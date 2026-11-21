#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for the unified SkipInfo class.

This test module covers all 4 existing SKIP_TYPES configurations from:
- extract_files.py
- extract_functions.py
- extract_section.py
- extract_notes.py
"""

import pytest
from src.snapshot_construction.anchor_construction.common.skip_info import SkipInfo


# Test data: All 4 existing SKIP_TYPES configurations
SKIP_TYPES_FILES = {
    "FILE_READ_ERROR": "File read error",
    "DIRECTORY_NOT_EXISTS": "Operator directory does not exist",
    "NO_MATCHING_FILES": "No matching files found",
}

SKIP_TYPES_FUNCTIONS = {
    "FILE_READ_ERROR": "File read error",
    "NO_FUNCTION_BODY": "No function body found",
    "NO_MATCHING_BRACE": "No matching brace found",
    "PATH_NOT_EXISTS": "Operator path does not exist",
    "NO_CPP_FILES": "No aclnn_*.cpp files found",
}

SKIP_TYPES_SECTION = {
    "FILE_READ_ERROR": "File read error",
    "PATH_NOT_EXISTS": "Operator path does not exist",
    "NO_CPP_FILES": "No aclnn_*.cpp files found",
    "NO_DEF_H_FILE": "No def.h file found",
    "NO_SECTIONS_FOUND": "No matching sections found",
}

SKIP_TYPES_NOTES = {
    "FILE_READ_ERROR": "File read error",
    "NO_PARAM_OR_RETURN": "Comment block without @param or @return",
    "PATH_NOT_EXISTS": "Operator path does not exist",
    "NO_HEADER_FILES": "No header files found",
    "NO_VALID_BLOCKS": "No valid comment blocks extracted",
}


class TestSkipInfoInitialization:
    """测试 SkipInfo 初始化"""

    def test_init_with_files_skip_types(self):
        """测试使用 extract_files.py 的 SKIP_TYPES 初始化"""
        skip_info = SkipInfo(SKIP_TYPES_FILES)

        assert skip_info.skip_types == SKIP_TYPES_FILES
        assert len(skip_info.skip_counts) == 3
        assert len(skip_info.skip_details) == 3

        # 所有计数应初始化为 0
        for count in skip_info.skip_counts.values():
            assert count == 0

        # 所有详情列表应初始化为空
        for details in skip_info.skip_details.values():
            assert details == []

    def test_init_with_functions_skip_types(self):
        """测试使用 extract_functions.py 的 SKIP_TYPES 初始化"""
        skip_info = SkipInfo(SKIP_TYPES_FUNCTIONS)

        assert skip_info.skip_types == SKIP_TYPES_FUNCTIONS
        assert len(skip_info.skip_counts) == 5
        assert len(skip_info.skip_details) == 5

    def test_init_with_section_skip_types(self):
        """测试使用 extract_section.py 的 SKIP_TYPES 初始化"""
        skip_info = SkipInfo(SKIP_TYPES_SECTION)

        assert skip_info.skip_types == SKIP_TYPES_SECTION
        assert len(skip_info.skip_counts) == 5
        assert len(skip_info.skip_details) == 5

    def test_init_with_notes_skip_types(self):
        """测试使用 extract_notes.py 的 SKIP_TYPES 初始化"""
        skip_info = SkipInfo(SKIP_TYPES_NOTES)

        assert skip_info.skip_types == SKIP_TYPES_NOTES
        assert len(skip_info.skip_counts) == 5
        assert len(skip_info.skip_details) == 5


class TestRecordSkip:
    """测试 record_skip 方法"""

    def test_record_skip_basic(self):
        """测试基本的跳过记录"""
        skip_info = SkipInfo(SKIP_TYPES_FILES)

        skip_info.record_skip(
            "FILE_READ_ERROR", "/path/to/file.cpp", "Permission denied"
        )

        assert skip_info.skip_counts["FILE_READ_ERROR"] == 1
        assert len(skip_info.skip_details["FILE_READ_ERROR"]) == 1
        assert skip_info.skip_details["FILE_READ_ERROR"][0] == {
            "item": "/path/to/file.cpp",
            "reason": "Permission denied",
        }

    def test_record_skip_without_reason(self):
        """测试不带原因的跳过记录"""
        skip_info = SkipInfo(SKIP_TYPES_FUNCTIONS)

        skip_info.record_skip("NO_FUNCTION_BODY", "aclnn_add")

        assert skip_info.skip_counts["NO_FUNCTION_BODY"] == 1
        assert skip_info.skip_details["NO_FUNCTION_BODY"][0] == {
            "item": "aclnn_add",
            "reason": "",
        }

    def test_record_skip_multiple_times(self):
        """测试多次记录同一类型的跳过"""
        skip_info = SkipInfo(SKIP_TYPES_SECTION)

        skip_info.record_skip("NO_SECTIONS_FOUND", "file1.cpp")
        skip_info.record_skip("NO_SECTIONS_FOUND", "file2.cpp", "No keywords")
        skip_info.record_skip("NO_SECTIONS_FOUND", "file3.cpp")

        assert skip_info.skip_counts["NO_SECTIONS_FOUND"] == 3
        assert len(skip_info.skip_details["NO_SECTIONS_FOUND"]) == 3

    def test_record_skip_multiple_types(self):
        """测试记录多种类型的跳过"""
        skip_info = SkipInfo(SKIP_TYPES_NOTES)

        skip_info.record_skip("FILE_READ_ERROR", "file1.h", "Error 1")
        skip_info.record_skip("NO_PARAM_OR_RETURN", "file2.h", "Error 2")
        skip_info.record_skip("NO_VALID_BLOCKS", "file3.h", "Error 3")

        assert skip_info.skip_counts["FILE_READ_ERROR"] == 1
        assert skip_info.skip_counts["NO_PARAM_OR_RETURN"] == 1
        assert skip_info.skip_counts["NO_VALID_BLOCKS"] == 1

    def test_record_skip_unknown_type(self):
        """测试记录未知类型的跳过（应被忽略）"""
        skip_info = SkipInfo(SKIP_TYPES_FILES)

        # 未知类型不应增加计数
        skip_info.record_skip("UNKNOWN_TYPE", "item", "reason")

        assert skip_info.get_total_skips() == 0
        assert "UNKNOWN_TYPE" not in skip_info.skip_counts


class TestGetTotalSkips:
    """测试 get_total_skips 方法"""

    def test_get_total_skips_empty(self):
        """测试空 SkipInfo 的总跳过数"""
        skip_info = SkipInfo(SKIP_TYPES_FILES)

        assert skip_info.get_total_skips() == 0

    def test_get_total_skips_single_type(self):
        """测试单一类型的总跳过数"""
        skip_info = SkipInfo(SKIP_TYPES_FUNCTIONS)

        skip_info.record_skip("NO_FUNCTION_BODY", "func1")
        skip_info.record_skip("NO_FUNCTION_BODY", "func2")
        skip_info.record_skip("NO_FUNCTION_BODY", "func3")

        assert skip_info.get_total_skips() == 3

    def test_get_total_skips_multiple_types(self):
        """测试多种类型的总跳过数"""
        skip_info = SkipInfo(SKIP_TYPES_SECTION)

        skip_info.record_skip("FILE_READ_ERROR", "file1")
        skip_info.record_skip("NO_CPP_FILES", "dir1")
        skip_info.record_skip("NO_SECTIONS_FOUND", "file2")
        skip_info.record_skip("NO_DEF_H_FILE", "dir2")

        assert skip_info.get_total_skips() == 4

    def test_get_total_skips_after_merge(self):
        """测试合并后的总跳过数"""
        skip_info1 = SkipInfo(SKIP_TYPES_NOTES)
        skip_info1.record_skip("FILE_READ_ERROR", "file1")
        skip_info1.record_skip("NO_HEADER_FILES", "dir1")

        skip_info2 = SkipInfo(SKIP_TYPES_NOTES)
        skip_info2.record_skip("FILE_READ_ERROR", "file2")
        skip_info2.record_skip("NO_VALID_BLOCKS", "file3")

        skip_info1.merge(skip_info2)

        assert skip_info1.get_total_skips() == 4


class TestMerge:
    """测试 merge 方法"""

    def test_merge_basic(self):
        """测试基本的合并操作"""
        skip_info1 = SkipInfo(SKIP_TYPES_FILES)
        skip_info1.record_skip("FILE_READ_ERROR", "file1")
        skip_info1.record_skip("NO_MATCHING_FILES", "dir1")

        skip_info2 = SkipInfo(SKIP_TYPES_FILES)
        skip_info2.record_skip("FILE_READ_ERROR", "file2")
        skip_info2.record_skip("DIRECTORY_NOT_EXISTS", "dir2")

        skip_info1.merge(skip_info2)

        assert skip_info1.skip_counts["FILE_READ_ERROR"] == 2
        assert skip_info1.skip_counts["NO_MATCHING_FILES"] == 1
        assert skip_info1.skip_counts["DIRECTORY_NOT_EXISTS"] == 1
        assert skip_info1.get_total_skips() == 4

    def test_merge_preserves_details(self):
        """测试合并保留详情"""
        skip_info1 = SkipInfo(SKIP_TYPES_FUNCTIONS)
        skip_info1.record_skip("NO_FUNCTION_BODY", "func1", "reason1")

        skip_info2 = SkipInfo(SKIP_TYPES_FUNCTIONS)
        skip_info2.record_skip("NO_FUNCTION_BODY", "func2", "reason2")
        skip_info2.record_skip("NO_MATCHING_BRACE", "func3", "reason3")

        skip_info1.merge(skip_info2)

        details = skip_info1.skip_details["NO_FUNCTION_BODY"]
        assert len(details) == 2
        assert details[0] == {"item": "func1", "reason": "reason1"}
        assert details[1] == {"item": "func2", "reason": "reason2"}

    def test_merge_different_skip_types(self):
        """测试合并不同 skip_types 的 SkipInfo"""
        skip_info1 = SkipInfo(SKIP_TYPES_FILES)
        skip_info1.record_skip("FILE_READ_ERROR", "file1")

        # SKIP_TYPES_FUNCTIONS 有额外的类型
        skip_info2 = SkipInfo(SKIP_TYPES_FUNCTIONS)
        skip_info2.record_skip("FILE_READ_ERROR", "file2")
        skip_info2.record_skip("NO_FUNCTION_BODY", "func1")  # 这个类型不在 FILES 中

        skip_info1.merge(skip_info2)

        # 只有共同类型会被合并
        assert skip_info1.skip_counts["FILE_READ_ERROR"] == 2
        assert skip_info1.get_total_skips() == 2  # NO_FUNCTION_BODY 被忽略

    def test_merge_empty_skip_info(self):
        """测试合并空的 SkipInfo"""
        skip_info1 = SkipInfo(SKIP_TYPES_SECTION)
        skip_info1.record_skip("FILE_READ_ERROR", "file1")

        skip_info2 = SkipInfo(SKIP_TYPES_SECTION)  # 空

        skip_info1.merge(skip_info2)

        assert skip_info1.get_total_skips() == 1

    def test_merge_into_empty_skip_info(self):
        """测试合并到空的 SkipInfo"""
        skip_info1 = SkipInfo(SKIP_TYPES_NOTES)  # 空

        skip_info2 = SkipInfo(SKIP_TYPES_NOTES)
        skip_info2.record_skip("FILE_READ_ERROR", "file1")
        skip_info2.record_skip("NO_HEADER_FILES", "dir1")

        skip_info1.merge(skip_info2)

        assert skip_info1.get_total_skips() == 2


class TestGetSkipCount:
    """测试 get_skip_count 方法"""

    def test_get_skip_count_existing_type(self):
        """测试获取存在的类型的计数"""
        skip_info = SkipInfo(SKIP_TYPES_FILES)
        skip_info.record_skip("FILE_READ_ERROR", "file1")
        skip_info.record_skip("FILE_READ_ERROR", "file2")

        assert skip_info.get_skip_count("FILE_READ_ERROR") == 2

    def test_get_skip_count_nonexistent_type(self):
        """测试获取不存在的类型的计数"""
        skip_info = SkipInfo(SKIP_TYPES_FUNCTIONS)

        assert skip_info.get_skip_count("NONEXISTENT_TYPE") == 0


class TestGetSkipDetails:
    """测试 get_skip_details 方法"""

    def test_get_skip_details_existing_type(self):
        """测试获取存在的类型的详情"""
        skip_info = SkipInfo(SKIP_TYPES_SECTION)
        skip_info.record_skip("NO_SECTIONS_FOUND", "file1", "reason1")
        skip_info.record_skip("NO_SECTIONS_FOUND", "file2", "reason2")

        details = skip_info.get_skip_details("NO_SECTIONS_FOUND")

        assert len(details) == 2
        assert details[0] == {"item": "file1", "reason": "reason1"}
        assert details[1] == {"item": "file2", "reason": "reason2"}

    def test_get_skip_details_nonexistent_type(self):
        """测试获取不存在的类型的详情"""
        skip_info = SkipInfo(SKIP_TYPES_NOTES)

        details = skip_info.get_skip_details("NONEXISTENT_TYPE")

        assert details == []


class TestToDictFromDict:
    """测试 to_dict 和 from_dict 方法"""

    def test_to_dict(self):
        """测试导出为字典"""
        skip_info = SkipInfo(SKIP_TYPES_FILES)
        skip_info.record_skip("FILE_READ_ERROR", "file1", "reason1")
        skip_info.record_skip("NO_MATCHING_FILES", "dir1", "reason2")

        data = skip_info.to_dict()

        assert "skip_types" in data
        assert "skip_counts" in data
        assert "skip_details" in data
        assert data["skip_types"] == SKIP_TYPES_FILES
        assert data["skip_counts"]["FILE_READ_ERROR"] == 1
        assert data["skip_counts"]["NO_MATCHING_FILES"] == 1

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "skip_types": SKIP_TYPES_FUNCTIONS,
            "skip_counts": {
                "FILE_READ_ERROR": 2,
                "NO_FUNCTION_BODY": 1,
                "NO_MATCHING_BRACE": 0,
                "PATH_NOT_EXISTS": 0,
                "NO_CPP_FILES": 0,
            },
            "skip_details": {
                "FILE_READ_ERROR": [
                    {"item": "file1", "reason": "error1"},
                    {"item": "file2", "reason": "error2"},
                ],
                "NO_FUNCTION_BODY": [{"item": "func1", "reason": ""}],
                "NO_MATCHING_BRACE": [],
                "PATH_NOT_EXISTS": [],
                "NO_CPP_FILES": [],
            },
        }

        skip_info = SkipInfo.from_dict(data)

        assert skip_info.skip_types == SKIP_TYPES_FUNCTIONS
        assert skip_info.skip_counts["FILE_READ_ERROR"] == 2
        assert skip_info.skip_counts["NO_FUNCTION_BODY"] == 1
        assert len(skip_info.skip_details["FILE_READ_ERROR"]) == 2

    def test_round_trip(self):
        """测试 to_dict -> from_dict 往返"""
        skip_info1 = SkipInfo(SKIP_TYPES_SECTION)
        skip_info1.record_skip("FILE_READ_ERROR", "file1", "error1")
        skip_info1.record_skip("NO_SECTIONS_FOUND", "file2", "error2")

        data = skip_info1.to_dict()
        skip_info2 = SkipInfo.from_dict(data)

        assert skip_info2.skip_types == skip_info1.skip_types
        assert skip_info2.skip_counts == skip_info1.skip_counts
        assert skip_info2.skip_details == skip_info1.skip_details
        assert skip_info2.get_total_skips() == skip_info1.get_total_skips()


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_skip_types(self):
        """测试空的 skip_types"""
        skip_info = SkipInfo({})

        assert skip_info.get_total_skips() == 0
        assert skip_info.skip_counts == {}
        assert skip_info.skip_details == {}

    def test_record_skip_with_empty_item(self):
        """测试记录空 item"""
        skip_info = SkipInfo(SKIP_TYPES_FILES)

        skip_info.record_skip("FILE_READ_ERROR", "", "some reason")

        assert skip_info.skip_counts["FILE_READ_ERROR"] == 1
        assert skip_info.skip_details["FILE_READ_ERROR"][0]["item"] == ""

    def test_record_skip_with_empty_reason(self):
        """测试记录空 reason"""
        skip_info = SkipInfo(SKIP_TYPES_FUNCTIONS)

        skip_info.record_skip("NO_FUNCTION_BODY", "func1", "")

        assert skip_info.skip_details["NO_FUNCTION_BODY"][0]["reason"] == ""

    def test_merge_with_self(self):
        """测试与自己合并"""
        skip_info = SkipInfo(SKIP_TYPES_NOTES)
        skip_info.record_skip("FILE_READ_ERROR", "file1")

        skip_info.merge(skip_info)

        # 计数应该翻倍
        assert skip_info.skip_counts["FILE_READ_ERROR"] == 2
        assert skip_info.get_total_skips() == 2

    def test_large_number_of_skips(self):
        """测试大量跳过记录"""
        skip_info = SkipInfo(SKIP_TYPES_SECTION)

        for i in range(100):
            skip_info.record_skip("FILE_READ_ERROR", f"file{i}", f"error{i}")

        assert skip_info.skip_counts["FILE_READ_ERROR"] == 100
        assert skip_info.get_total_skips() == 100
        assert len(skip_info.skip_details["FILE_READ_ERROR"]) == 100


class TestAllSkipTypesConfigurations:
    """测试所有 4 种 SKIP_TYPES 配置的完整工作流"""

    def test_files_workflow(self):
        """测试 extract_files.py 的完整工作流"""
        skip_info = SkipInfo(SKIP_TYPES_FILES)

        # 模拟文件提取过程
        skip_info.record_skip("DIRECTORY_NOT_EXISTS", "/path/to/op1")
        skip_info.record_skip(
            "FILE_READ_ERROR", "/path/to/op2/file.cpp", "Permission denied"
        )
        skip_info.record_skip("NO_MATCHING_FILES", "/path/to/op3")

        assert skip_info.get_total_skips() == 3
        assert skip_info.get_skip_count("DIRECTORY_NOT_EXISTS") == 1
        assert skip_info.get_skip_count("FILE_READ_ERROR") == 1
        assert skip_info.get_skip_count("NO_MATCHING_FILES") == 1

    def test_functions_workflow(self):
        """测试 extract_functions.py 的完整工作流"""
        skip_info1 = SkipInfo(SKIP_TYPES_FUNCTIONS)
        skip_info1.record_skip("NO_FUNCTION_BODY", "aclnn_add")
        skip_info1.record_skip("NO_MATCHING_BRACE", "aclnn_sub")

        skip_info2 = SkipInfo(SKIP_TYPES_FUNCTIONS)
        skip_info2.record_skip("FILE_READ_ERROR", "/path/file.cpp")
        skip_info2.record_skip("NO_CPP_FILES", "/path/op1")

        skip_info1.merge(skip_info2)

        assert skip_info1.get_total_skips() == 4

    def test_section_workflow(self):
        """测试 extract_section.py 的完整工作流"""
        skip_info = SkipInfo(SKIP_TYPES_SECTION)

        # 模拟段落提取过程
        skip_info.record_skip("PATH_NOT_EXISTS", "/path/to/op1")
        skip_info.record_skip("NO_DEF_H_FILE", "/path/to/op2")
        skip_info.record_skip(
            "NO_SECTIONS_FOUND", "/path/to/op3/file.cpp", "No keywords matched"
        )

        data = skip_info.to_dict()
        skip_info_restored = SkipInfo.from_dict(data)

        assert skip_info_restored.get_total_skips() == 3
        assert skip_info_restored.get_skip_count("NO_SECTIONS_FOUND") == 1

    def test_notes_workflow(self):
        """测试 extract_notes.py 的完整工作流"""
        skip_info = SkipInfo(SKIP_TYPES_NOTES)

        # 模拟注释提取过程
        skip_info.record_skip("NO_HEADER_FILES", "/path/to/op1")
        skip_info.record_skip(
            "NO_PARAM_OR_RETURN", "/path/to/op2/file.h", "Only @brief"
        )
        skip_info.record_skip("NO_VALID_BLOCKS", "/path/to/op3/file.h")

        assert skip_info.get_total_skips() == 3
        assert len(skip_info.get_skip_details("NO_PARAM_OR_RETURN")) == 1
