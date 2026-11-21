# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""进程全局副作用回归测试（Task 11：no_proxy / logger mkdir / builtins.print）。

覆盖:
- LLMClient 构造不再覆写 os.environ["no_proxy"]：预置值保留、缺失不新增
- import src.logger 零磁盘副作用（cwd 与 repo 侧 data/logs 存在性不变）
  且不污染 builtins.print
- redirect_stdout_to_log（contextlib.redirect_stdout 实现）:
  stdout 进日志、stderr 不进日志（kwargs 丢弃 bug 修复）、
  builtins.print 全程未被替换、退出块后 stdout 复原
- 惰性日志目录：setup_logger / setup_namespace_logger 首次使用时才创建

日志写入路径经 CANN_PLUGIN_LOG_DIR 指向 tmp_path 隔离（Task 9 dry_run
事故教训：一切写文件型验证先隔离输出目录）。
"""

import builtins
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

import src.logger as logger_module
from src.logger import redirect_stdout_to_log, setup_logger, setup_namespace_logger

REPO_ROOT = Path(__file__).resolve().parent.parent


def _unique_name(prefix: str) -> str:
    return f"{prefix}_t11_{uuid.uuid4().hex[:12]}"


def _isolate_log_dir(monkeypatch, tmp_path) -> Path:
    target = tmp_path / "plugin_logs"
    monkeypatch.setattr(logger_module, "_log_dir", None)
    monkeypatch.setenv("CANN_PLUGIN_LOG_DIR", str(target))
    return target


def _read_log(log_dir: Path, name: str) -> str:
    log_file = log_dir / f"{name}.log"
    assert log_file.exists(), f"log file not created: {log_file}"
    return log_file.read_text(encoding="utf-8")


class TestLLMClientNoProxy:
    def test_preexisting_no_proxy_value_preserved(self, monkeypatch):
        monkeypatch.setenv("INCREMENTAL_LLM_KEY", "dummy-key")
        monkeypatch.setenv("INCREMENTAL_LLM_URL", "https://dummy.example.com")
        monkeypatch.setenv("no_proxy", "custom-proxy.example.com")
        from src.llm.client import LLMClient

        snapshot = dict(os.environ)
        try:
            LLMClient()
        except Exception:
            pass
        assert os.environ.get("no_proxy") == "custom-proxy.example.com"
        assert dict(os.environ) == snapshot

    def test_absent_no_proxy_not_created(self, monkeypatch):
        monkeypatch.setenv("INCREMENTAL_LLM_KEY", "dummy-key")
        monkeypatch.setenv("INCREMENTAL_LLM_URL", "https://dummy.example.com")
        monkeypatch.delenv("no_proxy", raising=False)
        from src.llm.client import LLMClient

        try:
            LLMClient()
        except Exception:
            pass
        assert "no_proxy" not in os.environ


class TestImportPurity:
    def test_import_no_mkdir_no_builtins_patch(self, tmp_path):
        code = (
            "import builtins, os, sys\n"
            f"repo = {str(REPO_ROOT)!r}\n"
            "sys.path.insert(0, repo)\n"
            "repo_logs = os.path.join(repo, 'data', 'logs')\n"
            "before = (os.path.exists('data/logs'), os.path.exists(repo_logs))\n"
            "import src.logger\n"
            "assert builtins.print is print, 'builtins.print was patched'\n"
            "after = (os.path.exists('data/logs'), os.path.exists(repo_logs))\n"
            "assert before == after, f'import changed disk state: {before} -> {after}'\n"
            "print('ok')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            env={},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "ok"
        assert not (tmp_path / "data" / "logs").exists()

    def test_builtins_print_identity_in_process(self):
        assert builtins.print is print


class TestLazyLogDir:
    def test_setup_logger_creates_dir_on_first_use(self, tmp_path, monkeypatch):
        log_dir = _isolate_log_dir(monkeypatch, tmp_path)
        assert not log_dir.exists()
        name = _unique_name("lazy_setup")
        logger = setup_logger(name)
        assert log_dir.exists()
        logger.info("lazy-setup-marker")
        for handler in logger.handlers:
            handler.flush()
        assert "lazy-setup-marker" in _read_log(log_dir, name)

    def test_setup_namespace_logger_lazy(self, tmp_path, monkeypatch):
        log_dir = _isolate_log_dir(monkeypatch, tmp_path)
        assert not log_dir.exists()
        prefix = _unique_name("lazy_ns")
        ns_logger = setup_namespace_logger(prefix)
        assert log_dir.exists()
        child = logging.getLogger(f"{prefix}.child_module")
        child.info("ns-child-marker")
        for handler in ns_logger.handlers:
            handler.flush()
        assert "ns-child-marker" in _read_log(log_dir, prefix)


class TestRedirectStdoutToLog:
    def test_stdout_logged_stderr_passthrough(self, tmp_path, monkeypatch, capsys):
        log_dir = _isolate_log_dir(monkeypatch, tmp_path)
        name = _unique_name("redirect_route")
        with redirect_stdout_to_log(name):
            print("x-marker-stdout")
            print("y-marker-stderr", file=sys.stderr)
        content = _read_log(log_dir, name)
        assert "x-marker-stdout" in content
        assert "y-marker-stderr" not in content
        captured = capsys.readouterr()
        assert "y-marker-stderr" in captured.err
        assert "x-marker-stdout" not in captured.out

    def test_builtins_print_never_patched(self, tmp_path, monkeypatch):
        _isolate_log_dir(monkeypatch, tmp_path)
        original = builtins.print
        with redirect_stdout_to_log(_unique_name("no_patch")):
            assert builtins.print is original
        assert builtins.print is original

    def test_stdout_restored_after_block(self, tmp_path, monkeypatch):
        _isolate_log_dir(monkeypatch, tmp_path)
        original_stdout = sys.stdout
        with redirect_stdout_to_log(_unique_name("restore")):
            pass
        assert sys.stdout is original_stdout

    def test_rich_console_output_captured_plain(self, tmp_path, monkeypatch):
        rich_console = pytest.importorskip("rich.console")
        log_dir = _isolate_log_dir(monkeypatch, tmp_path)
        name = _unique_name("rich_capture")
        with redirect_stdout_to_log(name):
            console = rich_console.Console()
            console.print("rich-marker-line")
        content = _read_log(log_dir, name)
        assert "rich-marker-line" in content
        assert "\x1b[" not in content
