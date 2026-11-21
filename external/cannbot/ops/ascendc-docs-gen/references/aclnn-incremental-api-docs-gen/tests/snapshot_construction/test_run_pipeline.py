# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for run_pipeline module - the unified entry script for operator document generation pipeline.

Tests cover:
- _step_separator: console output helper (does not crash)
- _pre_populate_results: file-copy logic for incremental mode
  (skip cases, normal copy, file filter, existing results, task_id prefix,
   empty prompts, missing filepath)
- run_step_1 through run_step_5: thin wrappers mocked to verify submodule calls
- main: orchestration logic verifying step ordering and _pre_populate_results
  invocation conditions
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.snapshot_construction.build_tree_id.run_pipeline import (
    _pre_populate_results,
    _step_separator,
    main,
    run_step_1,
    run_step_2,
    run_step_3,
    run_step_4,
    run_step_5,
)

# =============================================================================
# Tests for _step_separator
# =============================================================================


class TestStepSeparator:
    """测试 _step_separator 步骤分隔符函数"""

    def test_step_separator_does_not_raise(self):
        """测试调用 _step_separator 不抛出异常"""
        _step_separator("Step 1: select_operators")

    def test_step_separator_with_empty_name(self):
        """测试调用 _step_separator 传空字符串不抛出异常"""
        _step_separator("")


# =============================================================================
# Tests for _pre_populate_results
# =============================================================================


class TestPrePopulateResults:
    """测试 _pre_populate_results 预填充未变更文件逻辑"""

    def test_skip_when_batch_prompts_missing(self, tmp_path):
        """测试 batch_prompts.json 不存在时跳过，不创建 results_by_file"""
        out_path = tmp_path / "out"
        old_tree_id_path = tmp_path / "old"
        old_results = old_tree_id_path / "results_by_file"
        old_results.mkdir(parents=True)
        (old_results / "001_result.json").write_text("{}")

        # batch_prompts.json 不存在
        _pre_populate_results(out_path, old_tree_id_path, file_filter=[])

        # new results_by_file 不应被创建
        assert not (out_path / "results_by_file").exists()

    def test_skip_when_old_results_dir_missing(self, tmp_path):
        """测试旧 results_by_file 不存在时跳过"""
        out_path = tmp_path / "out"
        prompts_dir = out_path / "batch_prompts_by_file"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "batch_prompts.json").write_text(json.dumps({"prompts": {"1": {"filepath": "/a.cpp"}}}))

        old_tree_id_path = tmp_path / "old"  # results_by_file 不存在

        _pre_populate_results(out_path, old_tree_id_path, file_filter=[])

        assert not (out_path / "results_by_file").exists()

    def test_copies_unchanged_files(self, tmp_path):
        """测试正常复制未变更文件的 LLM 结果"""
        out_path = tmp_path / "out"
        prompts_dir = out_path / "batch_prompts_by_file"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "batch_prompts.json").write_text(
            json.dumps(
                {
                    "prompts": {
                        "1": {"filepath": "/abs/path/file1.cpp"},
                        "2": {"filepath": "/abs/path/file2.cpp"},
                    }
                }
            )
        )

        old_tree_id_path = tmp_path / "old"
        old_results = old_tree_id_path / "results_by_file"
        old_results.mkdir(parents=True)
        (old_results / "001_result.json").write_text('{"result":"file1"}')
        (old_results / "002_result.json").write_text('{"result":"file2"}')

        _pre_populate_results(out_path, old_tree_id_path, file_filter=[])

        new_results = out_path / "results_by_file"
        assert (new_results / "001_result.json").exists()
        assert (new_results / "002_result.json").exists()
        assert (new_results / "001_result.json").read_text() == '{"result":"file1"}'

    def test_skips_files_in_filter(self, tmp_path):
        """测试 file_filter 中的变更文件不被复制"""
        out_path = tmp_path / "out"
        prompts_dir = out_path / "batch_prompts_by_file"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "batch_prompts.json").write_text(
            json.dumps(
                {
                    "prompts": {
                        "1": {"filepath": "/abs/path/changed.cpp"},
                        "2": {"filepath": "/abs/path/unchanged.cpp"},
                    }
                }
            )
        )

        old_tree_id_path = tmp_path / "old"
        old_results = old_tree_id_path / "results_by_file"
        old_results.mkdir(parents=True)
        (old_results / "001_result.json").write_text('{"r":"1"}')
        (old_results / "002_result.json").write_text('{"r":"2"}')

        _pre_populate_results(
            out_path,
            old_tree_id_path,
            file_filter=["/abs/path/changed.cpp"],
        )

        new_results = out_path / "results_by_file"
        assert not (new_results / "001_result.json").exists()
        assert (new_results / "002_result.json").exists()

    def test_does_not_overwrite_existing_result(self, tmp_path):
        """测试已存在的新结果文件不被覆盖"""
        out_path = tmp_path / "out"
        prompts_dir = out_path / "batch_prompts_by_file"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "batch_prompts.json").write_text(
            json.dumps(
                {
                    "prompts": {
                        "1": {"filepath": "/abs/path/file1.cpp"},
                    }
                }
            )
        )

        old_tree_id_path = tmp_path / "old"
        old_results = old_tree_id_path / "results_by_file"
        old_results.mkdir(parents=True)
        (old_results / "001_result.json").write_text('{"old":"data"}')

        # 预先创建 new results_by_file 中的同名文件
        new_results = out_path / "results_by_file"
        new_results.mkdir(parents=True)
        (new_results / "001_result.json").write_text('{"new":"data"}')

        _pre_populate_results(out_path, old_tree_id_path, file_filter=[])

        # 已存在的文件不应被覆盖
        assert (new_results / "001_result.json").read_text() == '{"new":"data"}'

    def test_task_id_prefix_zero_padded(self, tmp_path):
        """测试 task_id 前缀格式化为 3 位零填充"""
        out_path = tmp_path / "out"
        prompts_dir = out_path / "batch_prompts_by_file"
        prompts_dir.mkdir(parents=True)
        # task_id "1" -> 前缀 "001_"; task_id "10" -> 前缀 "010_"
        (prompts_dir / "batch_prompts.json").write_text(
            json.dumps(
                {
                    "prompts": {
                        "1": {"filepath": "/abs/file1.cpp"},
                        "10": {"filepath": "/abs/file10.cpp"},
                    }
                }
            )
        )

        old_tree_id_path = tmp_path / "old"
        old_results = old_tree_id_path / "results_by_file"
        old_results.mkdir(parents=True)
        (old_results / "001_file1.json").write_text("{}")
        (old_results / "010_file10.json").write_text("{}")

        _pre_populate_results(out_path, old_tree_id_path, file_filter=[])

        new_results = out_path / "results_by_file"
        assert (new_results / "001_file1.json").exists()
        assert (new_results / "010_file10.json").exists()

    def test_empty_prompts_no_copy(self, tmp_path):
        """测试空 prompts 字典不复制任何文件"""
        out_path = tmp_path / "out"
        prompts_dir = out_path / "batch_prompts_by_file"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "batch_prompts.json").write_text(json.dumps({"prompts": {}}))

        old_tree_id_path = tmp_path / "old"
        old_results = old_tree_id_path / "results_by_file"
        old_results.mkdir(parents=True)
        (old_results / "001_result.json").write_text("{}")

        _pre_populate_results(out_path, old_tree_id_path, file_filter=[])

        new_results = out_path / "results_by_file"
        # 目录会被创建，但里面不应有任何文件
        assert new_results.exists()
        assert not any(new_results.iterdir())

    def test_missing_filepath_skipped(self, tmp_path):
        """测试缺少 filepath 字段的 task 被跳过"""
        out_path = tmp_path / "out"
        prompts_dir = out_path / "batch_prompts_by_file"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "batch_prompts.json").write_text(
            json.dumps(
                {
                    "prompts": {
                        "1": {"filepath": "/abs/file1.cpp"},
                        "2": {},  # 缺少 filepath
                    }
                }
            )
        )

        old_tree_id_path = tmp_path / "old"
        old_results = old_tree_id_path / "results_by_file"
        old_results.mkdir(parents=True)
        (old_results / "001_file1.json").write_text("{}")
        (old_results / "002_file2.json").write_text("{}")

        _pre_populate_results(out_path, old_tree_id_path, file_filter=[])

        new_results = out_path / "results_by_file"
        assert (new_results / "001_file1.json").exists()
        assert not (new_results / "002_file2.json").exists()


# =============================================================================
# Tests for run_step_1 through run_step_5
# =============================================================================


class TestRunStep1:
    """测试 run_step_1 调用子模块"""

    @patch("src.snapshot_construction.build_tree_id.run_pipeline.prepare_code_analysis_batch")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.random_select_operators")
    def test_run_step_1_calls_submodules(self, mock_random, mock_prepare):
        """测试 run_step_1 调用 random_select_operators 和 prepare_code_analysis_batch"""
        operator_path = Path("/fake/ops")
        out_path = Path("/fake/out")
        run_step_1(operator_path, out_path)

        mock_random.main.assert_called_once_with(operator_path, out_path)
        mock_prepare.main.assert_called_once_with(out_path)


class TestRunStep2:
    """测试 run_step_2 调用 llm_processor"""

    @patch("src.snapshot_construction.build_tree_id.run_pipeline.llm_processor")
    def test_run_step_2_calls_llm_processor(self, mock_llm):
        """测试 run_step_2 调用 llm_processor.main"""
        operator_path = Path("/fake/ops")
        out_path = Path("/fake/out")
        run_step_2(operator_path, out_path)

        mock_llm.main.assert_called_once_with(operator_path, out_path)


class TestRunStep3:
    """测试 run_step_3 调用目录级子模块"""

    @patch("src.snapshot_construction.build_tree_id.run_pipeline.send_prompts_to_llm")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.generate_prompts_from_trees")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.generate_directory_trees")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.build_file_tree")
    def test_run_step_3_calls_all_submodules(self, mock_build, mock_gen_dir, mock_gen_prompts, mock_send):
        """测试 run_step_3 依次调用 build_file_tree, generate_directory_trees,
        generate_prompts_from_trees, send_prompts_to_llm
        """
        out_path = Path("/fake/out")
        run_step_3(out_path)

        mock_build.main.assert_called_once_with(out_path)
        mock_gen_dir.main.assert_called_once_with(out_path)
        mock_gen_prompts.main.assert_called_once_with(out_path)
        mock_send.main.assert_called_once_with(out_path)


class TestRunStep4:
    """测试 run_step_4 调用 generate_operator_docs"""

    @patch("src.snapshot_construction.build_tree_id.run_pipeline.generate_operator_docs")
    def test_run_step_4_calls_generate_operator_docs(self, mock_gen_docs):
        """测试 run_step_4 调用 generate_operator_docs.main"""
        out_path = Path("/fake/out")
        run_step_4(out_path)

        mock_gen_docs.main.assert_called_once_with(out_path)


class TestRunStep5:
    """测试 run_step_5 调用补充描述子模块"""

    @patch("src.snapshot_construction.build_tree_id.run_pipeline.organize_operator_docs")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.add_aclnn_h_description")
    def test_run_step_5_calls_both_submodules(self, mock_add, mock_organize):
        """测试 run_step_5 调用 add_aclnn_h_description 和 organize_operator_docs"""
        out_path = Path("/fake/out")
        run_step_5(out_path)

        mock_add.main.assert_called_once_with(out_path)
        mock_organize.main.assert_called_once_with(out_path)


# =============================================================================
# Tests for main
# =============================================================================


class TestMain:
    """测试 main 流水线编排逻辑"""

    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_5")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_4")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_3")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_2")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_1")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline._pre_populate_results")
    def test_main_without_filter_no_pre_populate(
        self,
        mock_pre_populate,
        mock_step1,
        mock_step2,
        mock_step3,
        mock_step4,
        mock_step5,
    ):
        """测试无 file_filter 和 old_tree_id_path 时不调用 _pre_populate_results"""
        operator_path = Path("/fake/ops")
        out_path = Path("/fake/out")
        main(operator_path, out_path)

        mock_step1.assert_called_once_with(operator_path, out_path)
        mock_step2.assert_called_once_with(operator_path, out_path)
        mock_step3.assert_called_once_with(out_path)
        mock_step4.assert_called_once_with(out_path)
        mock_step5.assert_called_once_with(out_path)
        mock_pre_populate.assert_not_called()

    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_5")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_4")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_3")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_2")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_1")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline._pre_populate_results")
    def test_main_with_filter_and_old_tree_calls_pre_populate(
        self,
        mock_pre_populate,
        mock_step1,
        mock_step2,
        mock_step3,
        mock_step4,
        mock_step5,
    ):
        """测试同时提供 file_filter 和 old_tree_id_path 时在 step_1 之后、step_2 之前调用 _pre_populate_results"""
        operator_path = Path("/fake/ops")
        out_path = Path("/fake/out")
        file_filter = ["/abs/path/changed.cpp"]
        old_tree_id_path = Path("/fake/old")

        # 跟踪调用顺序
        call_order = []
        mock_step1.side_effect = lambda *a, **kw: call_order.append("step_1")
        mock_step2.side_effect = lambda *a, **kw: call_order.append("step_2")
        mock_pre_populate.side_effect = lambda *a, **kw: call_order.append("pre_populate")

        main(
            operator_path,
            out_path,
            file_filter=file_filter,
            old_tree_id_path=old_tree_id_path,
        )

        mock_pre_populate.assert_called_once_with(out_path, old_tree_id_path, file_filter)
        # 验证 _pre_populate_results 在 step_1 和 step_2 之间调用
        assert call_order == ["step_1", "pre_populate", "step_2"]

    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_5")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_4")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_3")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_2")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_1")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline._pre_populate_results")
    def test_main_with_only_filter_no_pre_populate(
        self,
        mock_pre_populate,
        mock_step1,
        mock_step2,
        mock_step3,
        mock_step4,
        mock_step5,
    ):
        """测试仅有 file_filter 无 old_tree_id_path 时不调用 _pre_populate_results"""
        operator_path = Path("/fake/ops")
        out_path = Path("/fake/out")
        main(operator_path, out_path, file_filter=["/abs/path/changed.cpp"])

        mock_pre_populate.assert_not_called()
        mock_step1.assert_called_once()
        mock_step5.assert_called_once()

    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_5")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_4")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_3")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_2")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline.run_step_1")
    @patch("src.snapshot_construction.build_tree_id.run_pipeline._pre_populate_results")
    def test_main_with_only_old_tree_no_pre_populate(
        self,
        mock_pre_populate,
        mock_step1,
        mock_step2,
        mock_step3,
        mock_step4,
        mock_step5,
    ):
        """测试仅有 old_tree_id_path 无 file_filter 时不调用 _pre_populate_results"""
        operator_path = Path("/fake/ops")
        out_path = Path("/fake/out")
        main(operator_path, out_path, old_tree_id_path=Path("/fake/old"))

        mock_pre_populate.assert_not_called()
        mock_step1.assert_called_once()
        mock_step5.assert_called_once()
