# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""src.settings 单点配置测试（Task 10：Settings/env 收敛）。

覆盖:
- import 纯净性: 空 env 子进程导入 4 个入口模块零 raise（import 期零 env 依赖）
- load_settings 校验: KEY/URL 缺失时 RuntimeError 报文含变量名与 WORKFLOW.md 指引
- INCREMENTAL_LLM_MODEL 默认 qwen3.8-max（UD5 单点）与显式覆盖
- 可选变量透传
- apply_settings 注入 PubRefConfig 契约（含恢复，避免污染同进程其它测试）
"""

import subprocess
import sys
from pathlib import Path

import pytest

from src.settings import DEFAULT_MODEL, Settings, apply_settings, load_settings

REPO_ROOT = Path(__file__).resolve().parent.parent

_ENV_FREE_IMPORT_TARGETS = (
    "import src.main_snapshot, "
    "src.change_detection.anchor_build, "
    "src.change_detection.anchor_redirect_file.ancor_match_file_0422, "
    "src.run_all_cli"
)


def test_env_free_imports_exit_zero():
    """空 env 下导入全部入口模块必须零 raise（import 期零校验零副作用）。"""
    proc = subprocess.run(
        [sys.executable, "-c", _ENV_FREE_IMPORT_TARGETS],
        cwd=REPO_ROOT,
        env={},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


class TestLoadSettingsValidation:
    def test_missing_key_names_var_and_hint(self, monkeypatch):
        monkeypatch.setenv("INCREMENTAL_LLM_URL", "https://dummy.example.com")
        monkeypatch.delenv("INCREMENTAL_LLM_KEY", raising=False)
        with pytest.raises(RuntimeError) as excinfo:
            load_settings()
        assert "INCREMENTAL_LLM_KEY" in str(excinfo.value)
        assert "WORKFLOW.md" in str(excinfo.value)

    def test_missing_url_names_var(self, monkeypatch):
        monkeypatch.setenv("INCREMENTAL_LLM_KEY", "dummy-key")
        monkeypatch.delenv("INCREMENTAL_LLM_URL", raising=False)
        with pytest.raises(RuntimeError) as excinfo:
            load_settings()
        assert "INCREMENTAL_LLM_URL" in str(excinfo.value)
        assert "WORKFLOW.md" in str(excinfo.value)

    def test_missing_both_lists_both(self, monkeypatch):
        monkeypatch.delenv("INCREMENTAL_LLM_KEY", raising=False)
        monkeypatch.delenv("INCREMENTAL_LLM_URL", raising=False)
        with pytest.raises(RuntimeError) as excinfo:
            load_settings()
        msg = str(excinfo.value)
        assert "INCREMENTAL_LLM_KEY" in msg
        assert "INCREMENTAL_LLM_URL" in msg

    def test_empty_string_counts_as_missing(self, monkeypatch):
        monkeypatch.setenv("INCREMENTAL_LLM_KEY", "")
        monkeypatch.setenv("INCREMENTAL_LLM_URL", "https://dummy.example.com")
        with pytest.raises(RuntimeError) as excinfo:
            load_settings()
        assert "INCREMENTAL_LLM_KEY" in str(excinfo.value)


class TestModelDefault:
    def test_default_model_is_qwen38_max(self, monkeypatch):
        monkeypatch.setenv("INCREMENTAL_LLM_KEY", "dummy-key")
        monkeypatch.setenv("INCREMENTAL_LLM_URL", "https://dummy.example.com")
        monkeypatch.delenv("INCREMENTAL_LLM_MODEL", raising=False)
        assert load_settings().model == "qwen3.8-max"
        assert DEFAULT_MODEL == "qwen3.8-max"

    def test_explicit_model_overrides_default(self, monkeypatch):
        monkeypatch.setenv("INCREMENTAL_LLM_KEY", "dummy-key")
        monkeypatch.setenv("INCREMENTAL_LLM_URL", "https://dummy.example.com")
        monkeypatch.setenv("INCREMENTAL_LLM_MODEL", "test-model")
        assert load_settings().model == "test-model"


class TestOptionalPassthrough:
    def test_optional_vars_passed_through(self, monkeypatch):
        monkeypatch.setenv("INCREMENTAL_LLM_KEY", "dummy-key")
        monkeypatch.setenv("INCREMENTAL_LLM_URL", "https://dummy.example.com")
        monkeypatch.setenv("CANN_OPS_ROOT", "/tmp/opencode/ops-root")
        monkeypatch.setenv("TEST_OPERATOR_PATH", "/tmp/opencode/op")
        monkeypatch.delenv("CANN_PLUGIN_LOG_DIR", raising=False)
        monkeypatch.delenv("CANN_OPS_WORKSPACE", raising=False)
        monkeypatch.delenv("CANN_EVIDENCE_DIR", raising=False)
        settings = load_settings()
        assert settings.cann_ops_root == "/tmp/opencode/ops-root"
        assert settings.test_operator_path == "/tmp/opencode/op"
        assert settings.cann_plugin_log_dir is None
        assert settings.cann_ops_workspace is None
        assert settings.cann_evidence_dir is None


class TestApplySettings:
    def test_injects_pub_ref_and_restores(self):
        from src.snapshot_construction.pub_ref import PubRefConfig

        orig = (
            PubRefConfig.INCREMENTAL_LLM_KEY,
            PubRefConfig.INCREMENTAL_LLM_URL,
            PubRefConfig.MODEL,
        )
        try:
            apply_settings(Settings(incremental_llm_key="k", incremental_llm_url="u"))
            assert PubRefConfig.INCREMENTAL_LLM_KEY == "k"
            assert PubRefConfig.INCREMENTAL_LLM_URL == "u"
            assert PubRefConfig.MODEL == "qwen3.8-max"
        finally:
            (
                PubRefConfig.INCREMENTAL_LLM_KEY,
                PubRefConfig.INCREMENTAL_LLM_URL,
                PubRefConfig.MODEL,
            ) = orig
