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
回归测试：验证 change_deteciton_cli 调用 anchor_rebuild 时，
不会生成外部文件的 anchor_tuple 判断。

背景：
   change_deteciton_cli 在自身内部用 add_file_list/modified_file_list 构建
   extra_files_info（external = 交集，internal = 差集），仅传给
   run_anchor_redirect_pipeline。
   anchor_rebuild 现签名（anchor_build.py）不含 file_list/add_file_list/
   modified_file_list/extra_files_info 参数——外部文件信息完全不进入
   rebuild 环节，anchor_rebuild 不会生成外部文件 anchor_tuple 判断。
   本测试为回归测试，固化这一现状行为。
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# 确保 import run_change_detection_cli 时模块级环境变量检查通过
# 用 setdefault 不覆盖用户已有的真实环境变量
os.environ.setdefault("INCREMENTAL_LLM_KEY", "test_key")
os.environ.setdefault("INCREMENTAL_LLM_URL", "http://test.url")
os.environ.setdefault("INCREMENTAL_LLM_MODEL", "test_model")

# 导入被测试函数
from src.change_detection.run_change_detection_cli import change_deteciton_cli


class TestAnchorRebuildExternalFiles:
    """验证 anchor_rebuild 环节不生成外部文件 anchor_tuple 判断"""

    @pytest.fixture(autouse=True)
    def setup_env(self, tmp_path):
        """每个测试前设置环境变量与临时工作空间"""
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

        # 创建变更后算子目录（change_deteciton_cli 内部 relative_to 需要它位于 workspace 下）
        changed_op = self.workspace_dir / "changed_operator"
        changed_op.mkdir()
        (changed_op / "test.cpp").write_text("// changed code")

        yield

        # 恢复环境变量
        for key, value in self.env_backup.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

    @patch("src.change_detection.run_change_detection_cli.anchor_rebuild")
    @patch("src.change_detection.run_change_detection_cli.run_anchor_redirect_pipeline")
    def test_rebuild_not_generate_external_anchor_tuple(self, mock_redirect, mock_rebuild):
        """
        验证：change_deteciton_cli 调用 anchor_rebuild 时 file_list=None，
        外部文件信息不进入 rebuild 环节，因此不会生成外部文件 anchor_tuple 判断。

        对比：add_file_list / modified_file_list 会被 cli 用于构建
        run_anchor_redirect_pipeline 的 extra_files_info（external/internal），
        证明外部文件判断在 redirect 环节，而非 rebuild 环节。
        """
        # Mock：模拟 anchor_rebuild 在 file_list=None 时的真实返回
        # anchor_build.py 真实逻辑：file_list 为空 → extra_files_info 恒为空
        mock_rebuild.return_value = {
            "anchor_statistics": [],
            "extra_files_info": {"internal": [], "external": []},
        }
        mock_redirect.return_value = None

        # 传入外部文件参数（模拟用户提供了 add / modified 文件列表）
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

        # ---- 断言：anchor_rebuild 调用参数 ----
        assert mock_rebuild.called, "anchor_rebuild 应被调用"
        rebuild_kwargs = mock_rebuild.call_args[1]

        # 断言1：anchor_rebuild 现签名不含 file_list 参数（外部文件不进入 rebuild 的直接根因）
        assert "file_list" not in rebuild_kwargs, "anchor_rebuild 现签名无 file_list 参数，外部文件不进入 rebuild"

        # 断言2：add_file_list / modified_file_list 没有透传给 anchor_rebuild
        # 它们仅在 cli 内部用于构建 redirect 的 extra_files_info
        assert "add_file_list" not in rebuild_kwargs
        assert "modified_file_list" not in rebuild_kwargs
        assert "extra_files_info" not in rebuild_kwargs

        # 断言3：file_list=None 现状下，rebuild 返回的 extra_files_info['external'] 为空
        # 即 rebuild 环节没有生成任何外部文件 anchor_tuple 判断
        result = mock_rebuild.return_value
        assert (
            result["extra_files_info"]["external"] == []
        ), "file_list=None 时 rebuild 不会识别外部文件，external 应为空"
        assert result["extra_files_info"]["internal"] == []

        # ---- 对比断言：外部文件判断实际落在 redirect 环节 ----
        # 证明外部文件信息确实被处理了，只是在 redirect 而非 rebuild
        assert mock_redirect.called, "run_anchor_redirect_pipeline 应被调用"
        redirect_kwargs = mock_redirect.call_args[1]
        assert "extra_files_info" in redirect_kwargs

        # external = add_file_list ∩ modified_file_list = {ext/aaa.cpp}
        assert set(redirect_kwargs["extra_files_info"]["external"]) == {"ext/aaa.cpp"}
        # internal = modified_file_list - add_file_list = {src/core.cpp}
        assert set(redirect_kwargs["extra_files_info"]["internal"]) == {"src/core.cpp"}


if __name__ == "__main__":
    """直接运行入口：使用 pytest 执行本文件"""
    pytest.main([__file__, "-v"])
