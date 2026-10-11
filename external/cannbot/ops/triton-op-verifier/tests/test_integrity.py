#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
"""test_integrity.py — pytest UT for _baseline_integrity._check_baseline_integrity.

Run:  pytest test_integrity.py -v

Covers:
  - anchor missing → exit 3
  - anchor present, hash mismatch → exit 4
  - anchor present, hash matches → no exit
  - work_dir resolution from various verify_dir shapes
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import pytest

# scripts/ is added to sys.path by conftest.py at the tests/ root.
from _baseline_integrity import (
    BaselineGateError,
    _check_baseline_integrity,
    _resolve_work_dir,
)

OP_NAME = "TestOp"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_anchor(
    work_dir: Path,
    digest: str,
    mode: str = "user",
    source_sha256: str | None = None,
    source_path: str | None = None,
) -> Path:
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor = output_dir / ".baseline_anchor.json"
    record = {
        "op_name": OP_NAME,
        "torch_sha256": digest,
        "torch_path": str(work_dir / f"{OP_NAME}.py"),
        "source_mode": mode,
        "frozen_at": "2026-07-08T00:00:00+00:00",
    }
    if source_sha256 is not None:
        record["source_sha256"] = source_sha256
        record["source_path"] = source_path or "/tmp/fake_source.py"
    anchor.write_text(json.dumps(record))
    return anchor


def _write_baseline(work_dir: Path, content: bytes = b"# baseline\n") -> Path:
    p = work_dir / f"{OP_NAME}.py"
    p.write_bytes(content)
    return p


# silence logger.error noise during expected-failure tests
@pytest.fixture(autouse=True)
def _silence_logger():
    logging.getLogger("_baseline_integrity").setLevel(logging.CRITICAL)


# ---------- exit code tests ----------

def test_anchor_missing_exits_3(tmp_path: Path) -> None:
    _write_baseline(tmp_path)
    verify_dir = tmp_path / "output" / "iter_0" / "verify"
    verify_dir.mkdir(parents=True)
    with pytest.raises(BaselineGateError) as exc:
        _check_baseline_integrity(verify_dir, OP_NAME)
    assert exc.value.exit_code == 3


def test_hash_mismatch_exits_4(tmp_path: Path) -> None:
    _write_baseline(tmp_path, b"# original\n")
    _write_anchor(tmp_path, _sha256_bytes(b"# different\n"))
    verify_dir = tmp_path / "output" / "iter_0" / "verify"
    verify_dir.mkdir(parents=True)
    with pytest.raises(BaselineGateError) as exc:
        _check_baseline_integrity(verify_dir, OP_NAME)
    assert exc.value.exit_code == 4


def test_hash_match_no_exit(tmp_path: Path) -> None:
    content = b"# original\n"
    _write_baseline(tmp_path, content)
    _write_anchor(tmp_path, _sha256_bytes(content))
    verify_dir = tmp_path / "output" / "iter_0" / "verify"
    verify_dir.mkdir(parents=True)
    # Should NOT raise
    _check_baseline_integrity(verify_dir, OP_NAME)


def test_baseline_file_missing_exits_4(tmp_path: Path) -> None:
    _write_anchor(tmp_path, _sha256_bytes(b"# whatever\n"))
    # No {op_name}.py created
    verify_dir = tmp_path / "output" / "iter_0" / "verify"
    verify_dir.mkdir(parents=True)
    with pytest.raises(BaselineGateError) as exc:
        _check_baseline_integrity(verify_dir, OP_NAME)
    assert exc.value.exit_code == 4


def test_anchor_corrupt_json_exits_3(tmp_path: Path) -> None:
    _write_baseline(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    (output_dir / ".baseline_anchor.json").write_text("not json {{{")
    verify_dir = output_dir / "iter_0" / "verify"
    verify_dir.mkdir(parents=True)
    with pytest.raises(BaselineGateError) as exc:
        _check_baseline_integrity(verify_dir, OP_NAME)
    assert exc.value.exit_code == 3


def test_anchor_missing_sha256_field_exits_3(tmp_path: Path) -> None:
    _write_baseline(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    (output_dir / ".baseline_anchor.json").write_text(json.dumps({"op_name": OP_NAME}))
    verify_dir = output_dir / "iter_0" / "verify"
    verify_dir.mkdir(parents=True)
    with pytest.raises(BaselineGateError) as exc:
        _check_baseline_integrity(verify_dir, OP_NAME)
    assert exc.value.exit_code == 3


# ---------- work_dir resolution ----------

def test_resolve_work_dir_from_iter_verify(tmp_path: Path) -> None:
    work = tmp_path
    (work / "output" / "iter_3" / "verify").mkdir(parents=True)
    assert _resolve_work_dir(work / "output" / "iter_3" / "verify") == work.resolve()


def test_resolve_work_dir_from_opt_iter_verify(tmp_path: Path) -> None:
    work = tmp_path
    (work / "output" / "opt_iter_2" / "verify").mkdir(parents=True)
    assert _resolve_work_dir(work / "output" / "opt_iter_2" / "verify") == work.resolve()


def test_resolve_work_dir_when_passed_work_dir_directly(tmp_path: Path) -> None:
    # If caller passes work_dir (no output/ inside), function should return it as-is
    work = tmp_path / "no_output"
    work.mkdir()
    assert _resolve_work_dir(work) == work.resolve()


def test_full_flow_with_opt_iter_verify_dir(tmp_path: Path) -> None:
    """Integration: hash matches when verify_dir is opt_iter_N/verify/."""
    content = b"# original\n"
    _write_baseline(tmp_path, content)
    _write_anchor(tmp_path, _sha256_bytes(content))
    verify_dir = tmp_path / "output" / "opt_iter_5" / "verify"
    verify_dir.mkdir(parents=True)
    _check_baseline_integrity(verify_dir, OP_NAME)  # must not raise


def test_tampering_detected_across_iters(tmp_path: Path) -> None:
    """Simulate the original bug: baseline modified after freeze."""
    content = b"# original\n"
    _write_baseline(tmp_path, content)
    _write_anchor(tmp_path, _sha256_bytes(content))

    # Now simulate tampering: modify the baseline file
    (tmp_path / f"{OP_NAME}.py").write_bytes(b"# tampered\n")

    verify_dir = tmp_path / "output" / "iter_0" / "verify"
    verify_dir.mkdir(parents=True)
    with pytest.raises(BaselineGateError) as exc:
        _check_baseline_integrity(verify_dir, OP_NAME)
    assert exc.value.exit_code == 4


# ---------------------------------------------------------------------------
# Source-sha256 gate (mode=user freeze records source_sha256; verify
# re-checks the original source file hasn't drifted)
# ---------------------------------------------------------------------------

def _setup_user_source(tmp_path: Path, content: bytes = b"# user benchmark\n") -> Path:
    """mode=user freeze 场景准备：写 src、work-dir 基线副本、含 source_sha256 的锚。

    返回 src 路径供各用例后续篡改/删除。
    """
    src = tmp_path / "src_dir" / "source.py"
    src.parent.mkdir()
    src.write_bytes(content)
    _write_baseline(tmp_path, content)
    _write_anchor(
        tmp_path,
        digest=_sha256_bytes(content),
        source_sha256=_sha256_bytes(content),
        source_path=str(src),
    )
    return src


def test_source_sha256_match_passes(tmp_path: Path) -> None:
    """mode=user freeze recorded source; both copy and source unchanged."""
    _setup_user_source(tmp_path)
    verify_dir = tmp_path / "output" / "iter_0" / "verify"
    verify_dir.mkdir(parents=True)
    _check_baseline_integrity(verify_dir, OP_NAME)  # must not raise


def test_source_sha256_tampered_exits_4(tmp_path: Path) -> None:
    """User modifies source benchmark after freeze → exit 4."""
    src = _setup_user_source(tmp_path)
    # Tamper source after freeze
    src.write_bytes(b"# tampered source\n")

    verify_dir = tmp_path / "output" / "iter_0" / "verify"
    verify_dir.mkdir(parents=True)
    with pytest.raises(BaselineGateError) as exc:
        _check_baseline_integrity(verify_dir, OP_NAME)
    assert exc.value.exit_code == 4


def test_source_path_missing_exits_4(tmp_path: Path) -> None:
    """Source file deleted between freeze and verify → exit 4."""
    src = _setup_user_source(tmp_path)
    src.unlink()  # source deleted

    verify_dir = tmp_path / "output" / "iter_0" / "verify"
    verify_dir.mkdir(parents=True)
    with pytest.raises(BaselineGateError) as exc:
        _check_baseline_integrity(verify_dir, OP_NAME)
    assert exc.value.exit_code == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
