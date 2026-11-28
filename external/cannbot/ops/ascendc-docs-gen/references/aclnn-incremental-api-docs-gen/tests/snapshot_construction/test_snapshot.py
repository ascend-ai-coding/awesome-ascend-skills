# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for snapshot_construction.snapshot module.

Tests cover:
- statistic_anchor_info: anchor statistics collection from output directory,
  including file suffix matching, doc_part mapping, anchor type aggregation,
  missing/duplicate file handling, and empty anchor_tuple.
- load_snapshot: snapshot orchestration with all dependencies mocked,
  including doc_part_control conditional dispatch, default (None) behavior,
  expand_file_deps invocation, and return value propagation.
"""

import json
import os
from unittest.mock import patch

import pytest

from src.snapshot_construction.snapshot import load_snapshot, statistic_anchor_info
from src.snapshot_construction.anchor_construction.common.path_config import PathConfig


# =============================================================================
# Helper functions
# =============================================================================


def make_anchor_item(anchor_type, file_path):
    """Create a single anchor_tuple item with given type and file_path."""
    return {
        "anchor_information": {
            "type": anchor_type,
            "level1": {"file_path": file_path},
        }
    }


def make_anchor_json(anchor_items):
    """Create anchor JSON data with given anchor_tuple items."""
    return {"anchor_tuple": anchor_items}


def write_anchor_file(directory, filename, anchor_items):
    """Write an anchor JSON file to the given directory and return its path."""
    filepath = os.path.join(str(directory), filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(make_anchor_json(anchor_items), f)
    return filepath


# =============================================================================
# Tests for statistic_anchor_info
# =============================================================================


class TestStatisticAnchorInfo:
    """测试 statistic_anchor_info 函数"""

    def test_all_five_files_returns_five_stats(self, tmp_path):
        """测试所有 5 个 anchor 文件存在时返回 5 个统计字典"""
        items = [make_anchor_item("file", "path/to/file.cpp")]
        write_anchor_file(tmp_path, PathConfig.SUB_FUNCTION_HASH_FILE, items)
        write_anchor_file(tmp_path, PathConfig.CONSTRAINT_FILE, items)
        write_anchor_file(tmp_path, PathConfig.SUPPORT_FILE, items)
        write_anchor_file(tmp_path, PathConfig.SUMMARY_FILE, items)
        write_anchor_file(tmp_path, PathConfig.FUNCTION_PROTO_FILE, items)

        result = statistic_anchor_info(str(tmp_path))

        assert len(result) == 5

    def test_stat_dict_has_required_keys(self, tmp_path):
        """测试每个统计字典包含必需的键"""
        items = [make_anchor_item("file", "path/to/file.cpp")]
        write_anchor_file(tmp_path, PathConfig.SUB_FUNCTION_HASH_FILE, items)

        result = statistic_anchor_info(str(tmp_path))

        assert len(result) == 1
        stat = result[0]
        assert "doc_part" in stat
        assert "file" in stat
        assert "function" in stat
        assert "section" in stat
        assert "note" in stat

    def test_missing_file_skipped_no_error(self, tmp_path):
        """测试缺失的文件被跳过，不报错"""
        items = [make_anchor_item("file", "path/to/file.cpp")]
        # Only write 2 out of 5 files
        write_anchor_file(tmp_path, PathConfig.SUB_FUNCTION_HASH_FILE, items)
        write_anchor_file(tmp_path, PathConfig.CONSTRAINT_FILE, items)

        result = statistic_anchor_info(str(tmp_path))

        assert len(result) == 2

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """测试空目录返回空列表"""
        result = statistic_anchor_info(str(tmp_path))

        assert result == []

    def test_doc_part_mapping_correct(self, tmp_path):
        """测试 doc_part 映射正确（sub_function_hash, constraint, support, summary, function_proto）"""
        items = [make_anchor_item("file", "path/to/file.cpp")]
        write_anchor_file(tmp_path, PathConfig.SUB_FUNCTION_HASH_FILE, items)
        write_anchor_file(tmp_path, PathConfig.CONSTRAINT_FILE, items)
        write_anchor_file(tmp_path, PathConfig.SUPPORT_FILE, items)
        write_anchor_file(tmp_path, PathConfig.SUMMARY_FILE, items)
        write_anchor_file(tmp_path, PathConfig.FUNCTION_PROTO_FILE, items)

        result = statistic_anchor_info(str(tmp_path))

        doc_parts = {r["doc_part"] for r in result}
        assert doc_parts == {
            "sub_function_hash",
            "constraint",
            "support",
            "summary",
            "function_proto",
        }

    def test_anchor_type_aggregation(self, tmp_path):
        """测试 anchor 类型聚合，file_path 进入正确的类型列表"""
        items = [
            make_anchor_item("file", "path/file1.cpp"),
            make_anchor_item("function", "path/func1.cpp"),
            make_anchor_item("section", "path/section1.md"),
            make_anchor_item("note", "path/note1.txt"),
        ]
        write_anchor_file(tmp_path, PathConfig.SUB_FUNCTION_HASH_FILE, items)

        result = statistic_anchor_info(str(tmp_path))

        assert len(result) == 1
        stat = result[0]
        assert stat["file"] == ["path/file1.cpp"]
        assert stat["function"] == ["path/func1.cpp"]
        assert stat["section"] == ["path/section1.md"]
        assert stat["note"] == ["path/note1.txt"]

    def test_multiple_files_same_suffix_skipped(self, tmp_path):
        """测试同后缀多文件被跳过（len != 1）"""
        items = [make_anchor_item("file", "path/to/file.cpp")]
        write_anchor_file(tmp_path, PathConfig.SUB_FUNCTION_HASH_FILE, items)
        # Write a second file whose name also ends with the suffix
        write_anchor_file(tmp_path, "dup_" + PathConfig.SUB_FUNCTION_HASH_FILE, items)

        result = statistic_anchor_info(str(tmp_path))

        # sub_function_hash suffix has 2 matching files -> skipped
        # No other suffix files exist -> all skipped
        assert result == []

    def test_empty_anchor_tuple_type_lists_empty(self, tmp_path):
        """测试空 anchor_tuple 时类型列表为空"""
        write_anchor_file(tmp_path, PathConfig.SUB_FUNCTION_HASH_FILE, [])

        result = statistic_anchor_info(str(tmp_path))

        assert len(result) == 1
        stat = result[0]
        assert stat["file"] == []
        assert stat["function"] == []
        assert stat["section"] == []
        assert stat["note"] == []

    def test_multiple_items_same_type_aggregated(self, tmp_path):
        """测试同类型多个 anchor 项被聚合到同一列表"""
        items = [
            make_anchor_item("file", "path/file1.cpp"),
            make_anchor_item("file", "path/file2.cpp"),
            make_anchor_item("function", "path/func1.cpp"),
        ]
        write_anchor_file(tmp_path, PathConfig.CONSTRAINT_FILE, items)

        result = statistic_anchor_info(str(tmp_path))

        assert len(result) == 1
        stat = result[0]
        assert stat["file"] == ["path/file1.cpp", "path/file2.cpp"]
        assert stat["function"] == ["path/func1.cpp"]


# =============================================================================
# Tests for load_snapshot
# =============================================================================


@pytest.fixture
def mock_deps():
    """Patch all load_snapshot dependencies."""
    with (
        patch("src.snapshot_construction.snapshot.cann_doc_struct") as cann_doc,
        patch("src.snapshot_construction.snapshot.single_code_map") as code_map,
        patch("src.snapshot_construction.snapshot.run_pipeline") as run_pipeline,
        patch(
            "src.snapshot_construction.snapshot.run_pipe_line_single"
        ) as run_pipe_single,
        patch(
            "src.snapshot_construction.snapshot.get_constraint_evidence"
        ) as get_constraint,
        patch(
            "src.snapshot_construction.snapshot.summary_function_proto_main"
        ) as summary_fp,
        patch("src.snapshot_construction.snapshot.extract_evidence") as extract_ev,
        patch("src.snapshot_construction.snapshot.expand_file_deps") as expand_deps,
        patch("src.snapshot_construction.snapshot.DataLoader") as data_loader,
        patch("src.snapshot_construction.snapshot.statistic_anchor_info") as stat_info,
    ):
        stat_info.return_value = []
        yield {
            "cann_doc_struct": cann_doc,
            "single_code_map": code_map,
            "run_pipeline": run_pipeline,
            "run_pipe_line_single": run_pipe_single,
            "get_constraint_evidence": get_constraint,
            "summary_function_proto_main": summary_fp,
            "extract_evidence": extract_ev,
            "expand_file_deps": expand_deps,
            "DataLoader": data_loader,
            "statistic_anchor_info": stat_info,
        }


class TestLoadSnapshot:
    """测试 load_snapshot 函数"""

    def test_calls_cann_doc_struct_main(self, mock_deps, tmp_path):
        """测试调用 cann_doc_struct.main"""
        load_snapshot(str(tmp_path), str(tmp_path), str(tmp_path), doc_part_control=[])

        mock_deps["cann_doc_struct"].main.assert_called_once()

    def test_calls_single_code_map_main(self, mock_deps, tmp_path):
        """测试调用 single_code_map.main"""
        load_snapshot(str(tmp_path), str(tmp_path), str(tmp_path), doc_part_control=[])

        mock_deps["single_code_map"].main.assert_called_once()

    def test_calls_run_pipeline_main(self, mock_deps, tmp_path):
        """测试调用 run_pipeline.main"""
        load_snapshot(str(tmp_path), str(tmp_path), str(tmp_path), doc_part_control=[])

        mock_deps["run_pipeline"].main.assert_called_once()

    def test_sub_function_hash_calls_run_pipe_line_single(self, mock_deps, tmp_path):
        """测试 doc_part_control 包含 sub_function_hash 时调用 run_pipe_line_single"""
        load_snapshot(
            str(tmp_path),
            str(tmp_path),
            str(tmp_path),
            doc_part_control=["sub_function_hash"],
        )

        mock_deps["run_pipe_line_single"].assert_called_once()

    def test_constraint_calls_get_constraint_evidence(self, mock_deps, tmp_path):
        """测试 doc_part_control 包含 constraint 时调用 get_constraint_evidence"""
        load_snapshot(
            str(tmp_path),
            str(tmp_path),
            str(tmp_path),
            doc_part_control=["constraint"],
        )

        mock_deps["get_constraint_evidence"].assert_called_once()

    def test_function_proto_calls_summary_function_proto_main(
        self, mock_deps, tmp_path
    ):
        """测试 doc_part_control 包含 function_proto 时调用 summary_function_proto_main"""
        load_snapshot(
            str(tmp_path),
            str(tmp_path),
            str(tmp_path),
            doc_part_control=["function_proto"],
        )

        mock_deps["summary_function_proto_main"].assert_called_once()

    def test_summary_calls_summary_function_proto_main(self, mock_deps, tmp_path):
        """测试 doc_part_control 包含 summary 时调用 summary_function_proto_main（因 or 条件）"""
        load_snapshot(
            str(tmp_path),
            str(tmp_path),
            str(tmp_path),
            doc_part_control=["summary"],
        )

        mock_deps["summary_function_proto_main"].assert_called_once()

    def test_support_calls_extract_evidence(self, mock_deps, tmp_path):
        """测试 doc_part_control 包含 support 时调用 extract_evidence"""
        load_snapshot(
            str(tmp_path),
            str(tmp_path),
            str(tmp_path),
            doc_part_control=["support"],
        )

        mock_deps["extract_evidence"].assert_called_once()

    def test_default_doc_part_control_builds_all_parts(self, mock_deps, tmp_path):
        """测试默认 doc_part_control (None) 构建所有 5 个部分"""
        load_snapshot(str(tmp_path), str(tmp_path), str(tmp_path))

        # All doc_part-specific functions should be called
        mock_deps["run_pipe_line_single"].assert_called_once()
        mock_deps["get_constraint_evidence"].assert_called_once()
        mock_deps["summary_function_proto_main"].assert_called_once()
        mock_deps["extract_evidence"].assert_called_once()

    def test_calls_expand_file_deps(self, mock_deps, tmp_path):
        """测试调用 expand_file_deps"""
        load_snapshot(str(tmp_path), str(tmp_path), str(tmp_path), doc_part_control=[])

        mock_deps["expand_file_deps"].assert_called_once()

    def test_returns_statistic_anchor_info_result(self, mock_deps, tmp_path):
        """测试返回 statistic_anchor_info 的结果"""
        expected = [
            {"doc_part": "test", "file": [], "function": [], "section": [], "note": []}
        ]
        mock_deps["statistic_anchor_info"].return_value = expected

        result = load_snapshot(
            str(tmp_path), str(tmp_path), str(tmp_path), doc_part_control=[]
        )

        assert result == expected

    def test_empty_doc_part_control_skips_all_anchor_builders(
        self, mock_deps, tmp_path
    ):
        """测试空 doc_part_control 跳过所有 anchor 构建步骤"""
        load_snapshot(str(tmp_path), str(tmp_path), str(tmp_path), doc_part_control=[])

        mock_deps["run_pipe_line_single"].assert_not_called()
        mock_deps["get_constraint_evidence"].assert_not_called()
        mock_deps["summary_function_proto_main"].assert_not_called()
        mock_deps["extract_evidence"].assert_not_called()

    def test_calls_data_loader(self, mock_deps, tmp_path):
        """测试调用 DataLoader 实例化"""
        load_snapshot(str(tmp_path), str(tmp_path), str(tmp_path), doc_part_control=[])

        mock_deps["DataLoader"].assert_called_once()

    def test_expand_file_deps_receives_ops_root(self, mock_deps, tmp_path):
        """测试 expand_file_deps 接收 ops_root 参数"""
        ops_root = str(tmp_path / "ops_root")
        load_snapshot(str(tmp_path), str(tmp_path), ops_root, doc_part_control=[])

        mock_deps["expand_file_deps"].assert_called_once()
        call_args = mock_deps["expand_file_deps"].call_args
        # ops_root is the first positional argument
        assert call_args.args[0] == ops_root
