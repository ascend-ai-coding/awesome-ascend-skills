# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Task 12: CLI headless 行为——非 TTY 缺参 exit 2 报错、全参过 argparse 层、TTY 保留 TUI 分派。
Task 13: --output_data_path 非 TTY 条件必填 + gitcode_pr_file_get -o 默认值仓内锚定。
"""

import argparse
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from src import run_all_cli

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ALL_CLI = REPO_ROOT / "src" / "run_all_cli.py"

ALL_PATH_FLAGS = (
    "--original_op_repo",
    "--original_op_path",
    "--modified_op_repo",
    "--modified_op_path",
)
FULL_FLAGS_ARGS = [
    "--original_op_repo",
    "a",
    "--original_op_path",
    "b",
    "--modified_op_repo",
    "c",
    "--modified_op_path",
    "d",
]


class _FakeStdin:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _ns(**kwargs):
    defaults = dict(
        original_op_repo=None,
        original_op_path=None,
        modified_op_repo=None,
        modified_op_path=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_missing_flags_non_tty_subprocess_exits_2():
    proc = subprocess.run(
        [sys.executable, str(RUN_ALL_CLI)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        env={**os.environ},
    )
    assert proc.returncode == 2
    assert "非交互" in proc.stderr
    for flag in ALL_PATH_FLAGS:
        assert flag in proc.stderr


def test_partial_flags_non_tty_reports_only_missing():
    proc = subprocess.run(
        [sys.executable, str(RUN_ALL_CLI), "--original_op_repo", "a"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        env={**os.environ},
    )
    assert proc.returncode == 2
    error_lines = [line for line in proc.stderr.splitlines() if "error:" in line]
    assert error_lines, "parser.error 应输出 error 行"
    listed = set(re.findall(r"--[a-z_]+", error_lines[-1]))
    assert listed == {"--original_op_path", "--modified_op_repo", "--modified_op_path"}


def test_check_headless_required_non_tty_unit(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=False))
    parser = argparse.ArgumentParser(prog="run_all_cli")
    with pytest.raises(SystemExit) as exc:
        run_all_cli._check_headless_required(parser, _ns(), run_all_cli._REQUIRED_PATH_FLAGS)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "非交互" in err
    assert "--original_op_repo" in err


def test_check_headless_required_tty_returns_missing(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=True))
    parser = argparse.ArgumentParser(prog="run_all_cli")
    missing = run_all_cli._check_headless_required(parser, _ns(original_op_path="x"), run_all_cli._REQUIRED_PATH_FLAGS)
    assert missing == ["original_op_repo", "modified_op_repo", "modified_op_path"]


def test_check_headless_required_all_present(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=False))
    parser = argparse.ArgumentParser(prog="run_all_cli")
    args = _ns(
        original_op_repo="a",
        original_op_path="b",
        modified_op_repo="c",
        modified_op_path="d",
    )
    assert run_all_cli._check_headless_required(parser, args, run_all_cli._REQUIRED_PATH_FLAGS) == []


def test_full_flags_non_tty_passes_argparse_layer(monkeypatch, tmp_path, capsys):
    class FakeLLM:
        inits = 0

        def __init__(self, *a, **k):
            type(self).inits += 1
            self.model = "fake-model"

        def test_connection(self):
            return True

    monkeypatch.setenv("INCREMENTAL_LLM_KEY", "test-key")
    monkeypatch.setenv("INCREMENTAL_LLM_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("INCREMENTAL_LLM_MODEL", "test-model")
    monkeypatch.setenv("CANN_PLUGIN_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr("src.llm.client.LLMClient", FakeLLM)
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=False))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_all_cli", *FULL_FLAGS_ARGS, "--output_data_path", str(tmp_path / "out")],
    )
    try:
        runpy.run_module("src.run_all_cli", run_name="__main__")
    except SystemExit as e:
        assert e.code != 2
    captured = capsys.readouterr()
    assert "非交互" not in captured.out + captured.err
    assert FakeLLM.inits >= 1


def test_tty_missing_flags_dispatches_to_tui(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=True))
    parser = argparse.ArgumentParser(prog="run_all_cli")
    missing = run_all_cli._check_headless_required(parser, _ns(), run_all_cli._REQUIRED_PATH_FLAGS)
    assert missing, "TTY 下缺参不报错，分派走 TUI 分支"

    calls = {}

    def fake_pipeline(console, interactive=False, **kwargs):
        calls["interactive"] = interactive

    monkeypatch.setattr(run_all_cli, "_run_pipeline", fake_pipeline)
    run_all_cli.run_cli()
    assert calls["interactive"] is True


def test_full_flags_without_output_non_tty_exits_2():
    """非 TTY + 4 路径 flag 齐全 + 无 --output_data_path → exit 2，报错含 flag 名与"非交互"。

    报错发生在 argparse 层（先于 Settings），故无需隔离 INCREMENTAL_LLM_* env。
    """
    proc = subprocess.run(
        [sys.executable, str(RUN_ALL_CLI), *FULL_FLAGS_ARGS],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        env={**os.environ},
    )
    assert proc.returncode == 2
    assert "--output_data_path" in proc.stderr
    assert "非交互" in proc.stderr
    assert "[settings]" not in proc.stderr


def test_output_data_path_required_non_tty_unit(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=False))
    parser = argparse.ArgumentParser(prog="run_all_cli")
    args = _ns(
        original_op_repo="a",
        original_op_path="b",
        modified_op_repo="c",
        modified_op_path="d",
        output_data_path=None,
    )
    with pytest.raises(SystemExit) as exc:
        run_all_cli._check_headless_required(parser, args, ("output_data_path",))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--output_data_path" in err
    assert "非交互" in err


def test_output_data_path_tty_keeps_default_derivation(monkeypatch):
    """TTY + output 缺失 → 不报错仅返回缺失列表（<workspace>/data/outputs/snapshot 默认推导保留）。"""
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=True))
    parser = argparse.ArgumentParser(prog="run_all_cli")
    args = _ns(
        original_op_repo="a",
        original_op_path="b",
        modified_op_repo="c",
        modified_op_path="d",
        output_data_path=None,
    )
    missing = run_all_cli._check_headless_required(parser, args, ("output_data_path",))
    assert missing == ["output_data_path"]


def test_main_dispatch_output_check_precedes_settings(monkeypatch, capsys):
    """__main__ 分派顺序：output 缺失报错先于 Settings 校验（清空 INCREMENTAL_LLM_* 仍报 output 而非 [settings]）。"""
    for var in ("INCREMENTAL_LLM_KEY", "INCREMENTAL_LLM_URL", "INCREMENTAL_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=False))
    monkeypatch.setattr(sys, "argv", ["run_all_cli", *FULL_FLAGS_ARGS])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("src.run_all_cli", run_name="__main__")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--output_data_path" in err
    assert "[settings]" not in err


def test_gitcode_pr_get_default_output_dir_bound_to_repo(tmp_path, monkeypatch):
    """gitcode_pr_file_get -o 默认值锚定脚本所在目录（__file__ 相对），从任意 cwd 运行不漂移。"""
    import gitcode_pr_file_get as gpg

    monkeypatch.chdir(tmp_path)
    captured = {}

    def spy_parse(self, args=None, namespace=None):
        captured["default"] = self.get_default("output_dir")
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy_parse)
    monkeypatch.setattr(sys, "argv", ["gitcode_pr_file_get.py", "https://gitcode.com/x/y/pull/1"])
    with pytest.raises(SystemExit):
        gpg.main()

    assert Path(captured["default"]).resolve() == (REPO_ROOT / ".temp_git").resolve()
    assert not (tmp_path / ".temp_git").exists()
