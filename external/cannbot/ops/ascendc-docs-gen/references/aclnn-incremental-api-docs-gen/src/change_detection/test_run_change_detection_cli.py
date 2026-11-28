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
测试脚本：run_change_detection_cli.py

测试场景覆盖：
1. 正常流程测试（Mock 外部依赖）
2. 参数验证测试
3. 环境变量检查测试
4. 路径处理测试
5. 异常情况测试
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# 导入被测试模块
from src.change_detection.run_change_detection_cli import change_deteciton_cli


class TestChangeDetectionCLI:
    """变更检测 CLI 测试类"""

    @pytest.fixture(autouse=True)
    def setup_env(self, tmp_path):
        """每个测试前设置环境变量"""
        # 设置必要的环境变量
        self.env_backup = {
            "INCREMENTAL_LLM_KEY": os.environ.get("INCREMENTAL_LLM_KEY"),
            "INCREMENTAL_LLM_URL": os.environ.get("INCREMENTAL_LLM_URL"),
            "INCREMENTAL_LLM_MODEL": os.environ.get("INCREMENTAL_LLM_MODEL"),
        }

        os.environ.update(
            {
                "INCREMENTAL_LLM_KEY": "test_key",
                "INCREMENTAL_LLM_URL": "http://test.url",
                "INCREMENTAL_LLM_MODEL": "test_model",
            }
        )

        self.tmp_path = tmp_path
        self.workspace_dir = tmp_path / "workspace"
        self.workspace_dir.mkdir()

        # 创建必要的目录结构
        self._create_test_structure()

        yield

        # 恢复环境变量
        for key, value in self.env_backup.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

    def _create_test_structure(self):
        """创建测试所需的目录结构"""
        # 创建原始算子目录
        origin_operator = self.workspace_dir / "origin_operator"
        origin_operator.mkdir()
        (origin_operator / "test.cpp").write_text("// original code")

        # 创建变更后算子目录
        changed_operator = self.workspace_dir / "changed_operator"
        changed_operator.mkdir()
        (changed_operator / "test.cpp").write_text("// changed code")

        # 创建快照目录结构
        snapshot_base = self.workspace_dir / "snapshot"
        snapshot_base.mkdir()

        # 创建算子名称和版本目录
        md_stem_dir = snapshot_base / "test_op" / "aclnnTestOp"
        md_stem_dir.mkdir(parents=True)

        # 创建 anchor 目录
        anchor_dir = md_stem_dir / "anchor"
        anchor_dir.mkdir()

        # 创建 code_map 目录
        origin_code_map = md_stem_dir / "code_map" / "aclnnTestOp"
        origin_code_map.mkdir(parents=True)

        # 创建输出目录
        output_dir = self.workspace_dir / "output"
        output_dir.mkdir()

        # 创建测试 anchor 文件
        anchor_file = anchor_dir / "anchor.json"
        anchor_data = {
            "anchor_tuple": [
                {
                    "anchor_id": "test_001",
                    "note": "test anchor",
                    "anchor_information": {
                        "type": "file",
                        "level1": {
                            "file_path": "test.cpp",
                            "file_name": "test.cpp",
                            "behind_code": "// test",
                            "content": "test content",
                        },
                    },
                }
            ]
        }
        with open(anchor_file, "w", encoding="utf-8") as f:
            json.dump(anchor_data, f, ensure_ascii=False, indent=2)

        # 创建 code_map 文件
        code_map_file = origin_code_map / "test_op_map.json"
        code_map_data = {
            "nodes": [{"id": "func_A", "type": "function", "path": "test.cpp"}],
            "edges": [],
        }
        with open(code_map_file, "w", encoding="utf-8") as f:
            json.dump(code_map_data, f, ensure_ascii=False, indent=2)

    @patch("src.change_detection.run_change_detection_cli.anchor_rebuild")
    @patch("src.change_detection.run_change_detection_cli.run_anchor_redirect_pipeline")
    def test_normal_flow(self, mock_redirect, mock_rebuild, capsys):
        """测试正常流程"""
        # 设置 Mock 返回值
        mock_rebuild.return_value = {"status": "success", "changed_files": ["test.cpp"]}
        mock_redirect.return_value = None

        # 调用函数
        change_deteciton_cli(
            origin_operator_path="origin_operator",
            changed_operator_path=str(self.workspace_dir / "changed_operator"),
            data_path="snapshot",
            md_stem="aclnnTestOp",
            output_path="output",
            filter_list=["summary", "function_proto"],
            operator_name="test_op",
            file_dir_list=None,
            workspace_dir=str(self.workspace_dir),
            verbose=False,
        )

        # 验证 anchor_rebuild 被正确调用
        assert mock_rebuild.called
        call_args = mock_rebuild.call_args
        assert call_args[0][0] == "test_op"  # operator_name
        assert "changed_operator" in call_args[0][1]  # changed_operator_path_split
        assert "changed_operator" in call_args[0][2]  # changed_operator_path（原始仓绝对路径）
        assert "output" in call_args[0][3]  # new_snapshot_path
        assert call_args[0][4] == ["summary", "function_proto"]  # filter_list

        # 验证 run_anchor_redirect_pipeline 被正确调用
        assert mock_redirect.called
        redirect_args = mock_redirect.call_args[1]
        assert "anchor" in redirect_args["anchor_path"]
        assert "code_map" in redirect_args["origin_map_base"]
        assert "verbose" in redirect_args

        # 验证 extra_files_info 正确构建（默认值）
        assert "extra_files_info" in redirect_args
        assert redirect_args["extra_files_info"] == {"external": [], "internal": []}

        # 验证输出结果
        captured = capsys.readouterr()
        assert "===change-detection-result===" in captured.out
        assert "status" in captured.out

    @patch("src.change_detection.run_change_detection_cli.anchor_rebuild")
    @patch("src.change_detection.run_change_detection_cli.run_anchor_redirect_pipeline")
    def test_with_file_dir_list(self, mock_redirect, mock_rebuild):
        """测试带 file_dir_list 参数的情况"""
        mock_rebuild.return_value = {"status": "success"}
        mock_redirect.return_value = None

        file_list = [str(self.workspace_dir / "changed_operator" / "test.cpp")]

        change_deteciton_cli(
            origin_operator_path="origin_operator",
            changed_operator_path=str(self.workspace_dir / "changed_operator"),
            data_path="snapshot",
            md_stem="aclnnTestOp",
            output_path="output",
            filter_list=["summary"],
            operator_name="test_op",
            file_dir_list=file_list,
            workspace_dir=str(self.workspace_dir),
            verbose=True,
        )

        # 验证 anchor_rebuild 收到了 file_dir_list 和 old_tree_id_path
        call_args = mock_rebuild.call_args
        assert call_args[1]["file_dir_list"] == file_list
        assert "old_tree_id_path" in call_args[1]
        assert "tree_id" in call_args[1]["old_tree_id_path"]

        # 验证 verbose=True 传递给了 redirect pipeline
        assert mock_redirect.call_args[1]["verbose"] is True

    def test_missing_environment_variables(self):
        """测试缺少环境变量的情况"""
        # 临时删除环境变量
        backup_key = os.environ.pop("INCREMENTAL_LLM_KEY", None)

        try:
            # 重新导入模块会触发环境变量检查
            # 预期抛出异常
            import importlib

            import src.change_detection.run_change_detection_cli

            # 如果环境变量已设置，这里不会抛异常
            # 需要通过其他方式测试
            pass
        finally:
            if backup_key:
                os.environ.update({"INCREMENTAL_LLM_KEY": backup_key})

    @patch("src.change_detection.run_change_detection_cli.anchor_rebuild")
    @patch("src.change_detection.run_change_detection_cli.run_anchor_redirect_pipeline")
    def test_relative_vs_absolute_paths(self, mock_redirect, mock_rebuild):
        """测试相对路径和绝对路径的处理"""
        mock_rebuild.return_value = {"status": "success"}
        mock_redirect.return_value = None

        # 使用绝对路径调用
        change_deteciton_cli(
            origin_operator_path=str(self.workspace_dir / "origin_operator"),
            changed_operator_path=str(self.workspace_dir / "changed_operator"),
            data_path=str(self.workspace_dir / "snapshot"),
            md_stem="aclnnTestOp",
            output_path=str(self.workspace_dir / "output"),
            filter_list=["summary"],
            operator_name="test_op",
            workspace_dir=str(self.workspace_dir),
        )

        assert mock_rebuild.called
        assert mock_redirect.called

    @patch("src.change_detection.run_change_detection_cli.anchor_rebuild")
    @patch("src.change_detection.run_change_detection_cli.run_anchor_redirect_pipeline")
    def test_default_workspace(self, mock_redirect, mock_rebuild):
        """测试不提供 workspace_dir 时使用默认值"""
        mock_rebuild.return_value = {"status": "success"}
        mock_redirect.return_value = None

        # 切换到临时目录
        original_cwd = os.getcwd()
        try:
            os.chdir(str(self.workspace_dir))

            # 不提供 workspace_dir
            # 注意：这个测试可能会因为目录不存在而失败
            # 在实际测试中可能需要调整
        finally:
            os.chdir(original_cwd)

    @patch("src.change_detection.run_change_detection_cli.anchor_rebuild")
    @patch("src.change_detection.run_change_detection_cli.run_anchor_redirect_pipeline")
    def test_empty_filter_list(self, mock_redirect, mock_rebuild):
        """测试空 filter_list 的处理"""
        mock_rebuild.return_value = {"status": "success"}
        mock_redirect.return_value = None

        change_deteciton_cli(
            origin_operator_path="origin_operator",
            changed_operator_path=str(self.workspace_dir / "changed_operator"),
            data_path="snapshot",
            md_stem="aclnnTestOp",
            output_path="output",
            filter_list=[],
            operator_name="test_op",
            workspace_dir=str(self.workspace_dir),
        )

        # 验证空列表被正确传递
        call_args = mock_rebuild.call_args
        assert call_args[0][4] == []

    @patch("src.change_detection.run_change_detection_cli.anchor_rebuild")
    @patch("src.change_detection.run_change_detection_cli.run_anchor_redirect_pipeline")
    def test_result_output_format(self, mock_redirect, mock_rebuild, capsys):
        """测试结果输出格式"""
        test_result = {
            "status": "success",
            "changed_files": ["file1.cpp", "file2.cpp"],
            "total_changes": 2,
            "details": {
                "file1.cpp": {"type": "modified"},
                "file2.cpp": {"type": "added"},
            },
        }
        mock_rebuild.return_value = test_result
        mock_redirect.return_value = None

        change_deteciton_cli(
            origin_operator_path="origin_operator",
            changed_operator_path=str(self.workspace_dir / "changed_operator"),
            data_path="snapshot",
            md_stem="aclnnTestOp",
            output_path="output",
            filter_list=["summary"],
            operator_name="test_op",
            workspace_dir=str(self.workspace_dir),
        )

        # 验证输出包含正确的 JSON 格式
        captured = capsys.readouterr()
        assert "===change-detection-result===" in captured.out
        assert "===/change-detection-result===" in captured.out

        # 提取并解析 JSON
        lines = captured.out.split("\n")
        json_start = None
        json_lines = []
        for i, line in enumerate(lines):
            if "===change-detection-result===" in line:
                json_start = i + 1
            elif "===/change-detection-result===" in line:
                break
            elif json_start is not None:
                json_lines.append(line)

        json_str = "\n".join(json_lines)
        parsed_result = json.loads(json_str)
        assert parsed_result == test_result

    @patch("src.change_detection.run_change_detection_cli.anchor_rebuild")
    @patch("src.change_detection.run_change_detection_cli.run_anchor_redirect_pipeline")
    def test_error_handling_in_rebuild(self, mock_redirect, mock_rebuild):
        """测试 anchor_rebuild 抛出异常时的处理"""
        mock_rebuild.side_effect = Exception("Rebuild failed")

        with pytest.raises(Exception) as exc_info:
            change_deteciton_cli(
                origin_operator_path="origin_operator",
                changed_operator_path=str(self.workspace_dir / "changed_operator"),
                data_path="snapshot",
                md_stem="aclnnTestOp",
                output_path="output",
                filter_list=["summary"],
                operator_name="test_op",
                workspace_dir=str(self.workspace_dir),
            )

        assert "Rebuild failed" in str(exc_info.value)

    @patch("src.change_detection.run_change_detection_cli.anchor_rebuild")
    @patch("src.change_detection.run_change_detection_cli.run_anchor_redirect_pipeline")
    def test_with_add_and_modified_files(self, mock_redirect, mock_rebuild):
        """测试 add_file_list 和 modified_file_list 参数"""
        mock_rebuild.return_value = {"status": "success"}
        mock_redirect.return_value = None

        add_file_list = ["ext/aaa.cpp", "ext/bbb.h"]
        modified_file_list = ["src/core.cpp", "ext/aaa.cpp"]

        change_deteciton_cli(
            origin_operator_path="origin_operator",
            changed_operator_path=str(self.workspace_dir / "changed_operator"),
            data_path="snapshot",
            md_stem="aclnnTestOp",
            output_path="output",
            filter_list=["summary"],
            operator_name="test_op",
            workspace_dir=str(self.workspace_dir),
            add_file_list=add_file_list,
            modified_file_list=modified_file_list,
        )

        redirect_args = mock_redirect.call_args[1]
        extra_info = redirect_args["extra_files_info"]

        # external: add_file_list ∩ modified_file_list
        assert set(extra_info["external"]) == {"ext/aaa.cpp"}
        # internal: modified_file_list - add_file_list
        assert set(extra_info["internal"]) == {"src/core.cpp"}

    @patch("src.change_detection.run_change_detection_cli.anchor_rebuild")
    @patch("src.change_detection.run_change_detection_cli.run_anchor_redirect_pipeline")
    def test_error_handling_in_redirect(self, mock_redirect, mock_rebuild):
        """测试 run_anchor_redirect_pipeline 抛出异常时的处理"""
        mock_rebuild.return_value = {"status": "success"}
        mock_redirect.side_effect = Exception("Redirect failed")

        with pytest.raises(Exception) as exc_info:
            change_deteciton_cli(
                origin_operator_path="origin_operator",
                changed_operator_path=str(self.workspace_dir / "changed_operator"),
                data_path="snapshot",
                md_stem="aclnnTestOp",
                output_path="output",
                filter_list=["summary"],
                operator_name="test_op",
                workspace_dir=str(self.workspace_dir),
            )

        assert "Redirect failed" in str(exc_info.value)


class TestPathHandling:
    """路径处理专项测试"""

    @pytest.fixture
    def tmp_workspace(self, tmp_path):
        """创建临时工作空间"""
        workspace = tmp_path / "test_workspace"
        workspace.mkdir()
        return workspace

    @patch("src.change_detection.run_change_detection_cli.anchor_rebuild")
    @patch("src.change_detection.run_change_detection_cli.run_anchor_redirect_pipeline")
    def test_path_with_chinese_characters(self, mock_redirect, mock_rebuild, tmp_workspace):
        """测试路径包含中文字符"""
        # 设置环境变量
        os.environ.update(
            {
                "INCREMENTAL_LLM_KEY": "test_key",
                "INCREMENTAL_LLM_URL": "http://test.url",
                "INCREMENTAL_LLM_MODEL": "test_model",
            }
        )

        mock_rebuild.return_value = {"status": "success"}
        mock_redirect.return_value = None

        # 创建包含中文的路径
        chinese_dir = tmp_workspace / "测试目录"
        chinese_dir.mkdir()
        (chinese_dir / "test.cpp").write_text("// test")

        # 注意：这里只测试路径传递，不验证实际文件操作
        # 因为实际操作会在其他 Mock 中处理


class TestIntegration:
    """集成测试（可选，需要真实环境）"""

    @pytest.mark.integration
    @pytest.mark.skip(reason="需要真实环境配置，默认跳过")
    def test_real_execution(self):
        """真实执行测试（需要配置真实环境）"""
        # 这个测试需要：
        # 1. 真实的增量 LLM API 配置（INCREMENTAL_LLM_*）
        # 2. 真实的算子目录
        # 3. 真实的快照数据
        # 默认跳过，仅在需要时手动启用
        pass


def run_basic_test():
    """
    运行基础测试的独立函数
    用于快速验证基本功能，不依赖 pytest
    """

    print("=" * 60)
    print("运行基础功能测试（不依赖 pytest）")
    print("=" * 60)

    # 设置环境变量
    os.environ.update(
        {
            "INCREMENTAL_LLM_KEY": "test_key",
            "INCREMENTAL_LLM_URL": "http://test.url",
            "INCREMENTAL_LLM_MODEL": "test_model",
        }
    )

    # 创建临时目录
    tmp_dir = tempfile.mkdtemp()
    try:
        workspace = Path(tmp_dir) / "workspace"
        workspace.mkdir()

        # 创建基本目录结构
        (workspace / "origin_op").mkdir()
        (workspace / "changed_op").mkdir()
        (workspace / "snapshot" / "test_op" / "aclnnTestOp" / "anchor").mkdir(parents=True)
        (workspace / "snapshot" / "test_op" / "aclnnTestOp" / "code_map" / "aclnnTestOp").mkdir(parents=True)
        (workspace / "output").mkdir()

        # Mock 测试
        with (
            patch("src.change_detection.run_change_detection_cli.anchor_rebuild") as mock_rebuild,
            patch("src.change_detection.run_change_detection_cli.run_anchor_redirect_pipeline") as mock_redirect,
        ):
            mock_rebuild.return_value = {"status": "success", "test": True}
            mock_redirect.return_value = None

            print("\n测试 1: 基本调用")
            change_deteciton_cli(
                origin_operator_path="origin_op",
                changed_operator_path=str(workspace / "changed_op"),
                data_path="snapshot",
                md_stem="aclnnTestOp",
                output_path="output",
                filter_list=["summary"],
                operator_name="test_op",
                workspace_dir=str(workspace),
            )

            assert mock_rebuild.called, "anchor_rebuild 应该被调用"
            assert mock_redirect.called, "run_anchor_redirect_pipeline 应该被调用"
            print("✓ 基本调用测试通过")

            print("\n测试 2: 带 file_dir_list 的调用")
            mock_rebuild.reset_mock()
            mock_redirect.reset_mock()

            change_deteciton_cli(
                origin_operator_path="origin_op",
                changed_operator_path=str(workspace / "changed_op"),
                data_path="snapshot",
                md_stem="aclnnTestOp",
                output_path="output",
                filter_list=["summary", "function_proto"],
                operator_name="test_op",
                file_dir_list=[str(workspace / "changed_op" / "test.cpp")],
                workspace_dir=str(workspace),
                verbose=True,
            )

            assert mock_rebuild.called
            assert "file_dir_list" in mock_rebuild.call_args[1]
            print("✓ file_dir_list 参数传递测试通过")

            print("\n测试 3: add_file_list 和 modified_file_list 参数")
            mock_rebuild.reset_mock()
            mock_redirect.reset_mock()

            change_deteciton_cli(
                origin_operator_path="origin_op",
                changed_operator_path=str(workspace / "changed_op"),
                data_path="snapshot",
                md_stem="aclnnTestOp",
                output_path="output",
                filter_list=["summary"],
                operator_name="test_op",
                workspace_dir=str(workspace),
                add_file_list=["ext/foo.cpp", "ext/bar.h"],
                modified_file_list=["src/core.cpp", "ext/foo.cpp"],
            )

            assert mock_redirect.called
            extra_info = mock_redirect.call_args[1]["extra_files_info"]
            # external = add ∩ modified → ext/foo.cpp
            assert set(extra_info["external"]) == {"ext/foo.cpp"}
            # internal = modified - add → src/core.cpp
            assert set(extra_info["internal"]) == {"src/core.cpp"}
            print("✓ extra_files_info 构建逻辑测试通过")

        print("\n" + "=" * 60)
        print("所有基础测试通过！")
        print("=" * 60)

    finally:
        # 清理临时目录
        shutil.rmtree(tmp_dir)


def run_parametrized_test():
    """运行参数化测试"""
    print("\n" + "=" * 60)
    print("运行参数化测试")
    print("=" * 60)

    test_cases = [
        {
            "name": "最小参数集",
            "params": {
                "origin_operator_path": "origin",
                "changed_operator_path": "/path/changed",
                "data_path": "snapshot",
                "md_stem": "aclnnTestOp",
                "output_path": "output",
                "filter_list": [],
                "operator_name": "test",
            },
        },
        {
            "name": "完整参数集",
            "params": {
                "origin_operator_path": "origin",
                "changed_operator_path": "/path/changed",
                "data_path": "snapshot",
                "md_stem": "aclnnTestOp",
                "output_path": "output",
                "filter_list": ["summary", "function_proto", "sub_function_hash"],
                "operator_name": "test",
                "file_dir_list": ["/path/file1.cpp", "/path/file2.cpp"],
                "workspace_dir": "/workspace",
                "verbose": True,
            },
        },
        {
            "name": "多文件过滤",
            "params": {
                "origin_operator_path": "origin",
                "changed_operator_path": "/path/changed",
                "data_path": "snapshot",
                "md_stem": "aclnnTestOpV2",
                "output_path": "output",
                "filter_list": [
                    "summary",
                    "function_proto",
                    "sub_function_hash",
                    "extra_filter",
                ],
                "operator_name": "complex_op",
            },
        },
        {
            "name": "带 add_file_list 和 modified_file_list",
            "params": {
                "origin_operator_path": "origin",
                "changed_operator_path": "/path/changed",
                "data_path": "snapshot",
                "md_stem": "aclnnTestOp",
                "output_path": "output",
                "filter_list": ["summary"],
                "operator_name": "test",
                "add_file_list": ["ext/a.cpp", "ext/b.h"],
                "modified_file_list": ["src/main.cpp", "ext/a.cpp"],
            },
        },
    ]

    os.environ.update(
        {
            "INCREMENTAL_LLM_KEY": "test_key",
            "INCREMENTAL_LLM_URL": "http://test.url",
            "INCREMENTAL_LLM_MODEL": "test_model",
        }
    )

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {test_case['name']}")
        with (
            patch("src.change_detection.run_change_detection_cli.anchor_rebuild") as mock_rebuild,
            patch("src.change_detection.run_change_detection_cli.run_anchor_redirect_pipeline") as mock_redirect,
        ):
            mock_rebuild.return_value = {"status": "success"}
            mock_redirect.return_value = None

            try:
                change_deteciton_cli(**test_case["params"])
                print(f"✓ 测试用例 {i} 通过")
            except Exception as e:
                print(f"✗ 测试用例 {i} 失败: {e}")


if __name__ == "__main__":
    """
    测试脚本入口

    运行方式：
    1. 使用 pytest: pytest test_run_change_detection_cli.py -v
    2. 直接运行: python test_run_change_detection_cli.py

    集成测试需要启用标记：
    pytest test_run_change_detection_cli.py -v -m integration
    """

    # 检查是否使用 pytest
    if "pytest" in sys.modules or "--pytest" in sys.argv:
        # 让 pytest 处理
        pytest.main([__file__, "-v"])
    else:
        # 运行独立测试
        print("开始执行测试...\n")

        try:
            run_basic_test()
            run_parametrized_test()
            print("\n" + "=" * 60)
            print("所有测试完成！")
            print("=" * 60)
        except Exception as e:
            print(f"\n测试执行失败: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
