#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
"""test_freeze.py — pytest UT for freeze_baseline.py.

Run:  pytest test_freeze.py -v
      (or: python3 -m pytest test_freeze.py -v)

Tests the script via subprocess (black-box) to mirror how Phase 1 invokes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
FREEZE_SCRIPT = SCRIPTS_DIR / "freeze_baseline.py"


def _run_freeze(
    work_dir: Path,
    op_name: str,
    mode: str | None = None,
    source_path: str | None = None,
) -> subprocess.CompletedProcess:
    args = [sys.executable, str(FREEZE_SCRIPT), "--op_name", op_name, "--work_dir", str(work_dir)]
    if mode is not None:
        args.extend(["--mode", mode])
    if source_path is not None:
        args.extend(["--source_path", source_path])
    return subprocess.run(args, capture_output=True, text=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read())
    return h.hexdigest()


def test_first_freeze_writes_anchor(tmp_path: Path) -> None:
    op_name = "MyOp"
    op_file = tmp_path / f"{op_name}.py"
    op_file.write_text("print('hello')\n")

    result = _run_freeze(tmp_path, op_name, mode="auto")

    assert result.returncode == 0, f"expected exit 0, got {result.returncode}; stderr={result.stderr}"
    anchor_path = tmp_path / "output" / ".baseline_anchor.json"
    assert anchor_path.is_file(), "anchor file not created"

    record = json.loads(anchor_path.read_text())
    assert record["op_name"] == op_name
    assert record["torch_sha256"] == _sha256(op_file)
    assert record["torch_path"] == str(op_file.resolve())
    assert "frozen_at" in record


def test_second_freeze_refuses_overwrite(tmp_path: Path) -> None:
    op_name = "MyOp"
    (tmp_path / f"{op_name}.py").write_text("x = 1\n")
    anchor_path = tmp_path / "output" / ".baseline_anchor.json"

    first = _run_freeze(tmp_path, op_name, mode="auto")
    assert first.returncode == 0
    original_content = anchor_path.read_text()

    # Even with unchanged target, second freeze must refuse
    second = _run_freeze(tmp_path, op_name, mode="auto")
    assert second.returncode == 1, f"expected exit 1, got {second.returncode}; stderr={second.stderr}"
    assert anchor_path.read_text() == original_content, "anchor was modified on second freeze"


def test_target_file_missing(tmp_path: Path) -> None:
    result = _run_freeze(tmp_path, "NoSuchOp", mode="auto")
    assert result.returncode == 2
    assert "not found" in result.stderr.lower()


def test_mode_auto_no_source_path_required(tmp_path: Path) -> None:
    op_name = "AutoOp"
    (tmp_path / f"{op_name}.py").write_text("# auto\n")
    result = _run_freeze(tmp_path, op_name, mode="auto")
    assert result.returncode == 0
    record = json.loads((tmp_path / "output" / ".baseline_anchor.json").read_text())
    assert record["source_mode"] == "auto"
    assert "source_sha256" not in record


def test_mode_user_recorded(tmp_path: Path) -> None:
    op_name = "UserOp"
    content = "# user\n"
    src = tmp_path / "source.py"
    src.write_text(content)
    # byte-level cp to work dir
    (tmp_path / f"{op_name}.py").write_text(content)
    result = _run_freeze(tmp_path, op_name, mode="user", source_path=str(src))
    assert result.returncode == 0
    record = json.loads((tmp_path / "output" / ".baseline_anchor.json").read_text())
    assert record["source_mode"] == "user"
    assert record["source_sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert record["source_path"] == str(src.resolve())


def test_mode_user_without_source_path_rejected(tmp_path: Path) -> None:
    op_name = "UserOp"
    (tmp_path / f"{op_name}.py").write_text("# user\n")
    result = _run_freeze(tmp_path, op_name, mode="user")
    assert result.returncode != 0  # exit 3, argparse/our check
    assert "--source_path" in result.stderr


def test_mode_default_is_user(tmp_path: Path) -> None:
    # default mode is user, so still need source_path
    op_name = "DefOp"
    content = "# default\n"
    src = tmp_path / "source.py"
    src.write_text(content)
    (tmp_path / f"{op_name}.py").write_text(content)
    result = _run_freeze(tmp_path, op_name, source_path=str(src))  # no --mode
    assert result.returncode == 0
    record = json.loads((tmp_path / "output" / ".baseline_anchor.json").read_text())
    assert record["source_mode"] == "user"


def test_mode_invalid_rejected(tmp_path: Path) -> None:
    op_name = "BadModeOp"
    (tmp_path / f"{op_name}.py").write_text("# x\n")
    result = _run_freeze(tmp_path, op_name, mode="bogus")
    assert result.returncode != 0  # argparse exits 2 for invalid choice


def test_creates_output_dir_if_missing(tmp_path: Path) -> None:
    op_name = "MkdirOp"
    (tmp_path / f"{op_name}.py").write_text("# x\n")
    # No output/ dir yet
    assert not (tmp_path / "output").exists()
    result = _run_freeze(tmp_path, op_name, mode="auto")
    assert result.returncode == 0
    assert (tmp_path / "output" / ".baseline_anchor.json").is_file()


# ---------------------------------------------------------------------------
# Mode-user source-equality gate (the L3 ConvTranspose2d case)
# ---------------------------------------------------------------------------

def test_mode_user_copy_matches_source_passes(tmp_path: Path) -> None:
    """Scenario B (user benchmark): Agent byte-level cp'd source → freeze OK."""
    op_name = "UserCpOp"
    content = "import torch\nclass Model(torch.nn.Module):\n    pass\n"
    src = tmp_path / "source_dir" / f"{op_name}.py"
    src.parent.mkdir()
    src.write_text(content)
    (tmp_path / f"{op_name}.py").write_text(content)  # exact cp

    result = _run_freeze(tmp_path, op_name, mode="user", source_path=str(src))
    assert result.returncode == 0, f"expected exit 0, got {result.returncode}; stderr={result.stderr}"
    record = json.loads((tmp_path / "output" / ".baseline_anchor.json").read_text())
    assert record["torch_sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert record["source_sha256"] == record["torch_sha256"]


def test_mode_user_rewritten_copy_rejected(tmp_path: Path) -> None:
    """Scenario B: Agent rewrote Model class in work-dir copy → exit 5.

    This is the L3 ConvTranspose2d case: source has a Model.__init__ that
    requires args, Agent rewrites it to be no-arg with constant weights.
    The freeze must refuse to anchor the rewritten version.
    """
    op_name = "RewrittenOp"
    src_content = "import torch\nclass Model:\n    def __init__(self, x, y): pass\n"
    rewritten_content = "import torch\nclass Model:\n    def __init__(self): pass\n"

    src = tmp_path / "source_dir" / f"{op_name}.py"
    src.parent.mkdir()
    src.write_text(src_content)
    (tmp_path / f"{op_name}.py").write_text(rewritten_content)  # NOT a cp

    result = _run_freeze(tmp_path, op_name, mode="user", source_path=str(src))
    assert result.returncode == 5, f"expected exit 5, got {result.returncode}; stderr={result.stderr}"
    assert "rewrote the baseline" in result.stderr.lower() or "!=" in result.stderr
    # Anchor must NOT be written
    assert not (tmp_path / "output" / ".baseline_anchor.json").exists()


def test_mode_user_source_path_missing(tmp_path: Path) -> None:
    op_name = "SrcMissingOp"
    (tmp_path / f"{op_name}.py").write_text("# x\n")
    result = _run_freeze(tmp_path, op_name, mode="user", source_path="/no/such/file.py")
    assert result.returncode == 2


def test_mode_auto_ignores_source_path_mismatch(tmp_path: Path) -> None:
    """Scenario A (Agent-generated): source_path not required, copy can differ."""
    op_name = "AutoGenOp"
    src = tmp_path / "src.py"
    src.write_text("class A: pass\n")
    (tmp_path / f"{op_name}.py").write_text("class B: pass\n")  # different

    # mode=auto: no source check at all; even passing source_path is ignored
    result = _run_freeze(tmp_path, op_name, mode="auto", source_path=str(src))
    assert result.returncode == 0
    record = json.loads((tmp_path / "output" / ".baseline_anchor.json").read_text())
    assert "source_sha256" not in record


if __name__ == "__main__":
    # Allow running without pytest: minimal shim
    sys.exit(pytest.main([__file__, "-v"]))
