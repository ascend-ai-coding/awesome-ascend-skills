# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "operator_owner_config.py"
SPEC = importlib.util.spec_from_file_location("operator_owner_config", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
OWNER_CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OWNER_CONFIG)


def _run_owner_config(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )


def test_missing_config_returns_missing_owner(tmp_path: Path):
    matched_key, owner = OWNER_CONFIG.lookup_owner(
        tmp_path / "operator_owners.yaml", "ops.matmul"
    )

    assert matched_key is None
    assert owner is None


def test_lookup_is_case_insensitive_and_normalizes_at_prefix(tmp_path: Path):
    config = tmp_path / "operator_owners.yaml"
    config.write_text("operators:\n  Ops.MatMul: '@owner-a'\n", encoding="utf-8")

    matched_key, owner = OWNER_CONFIG.lookup_owner(config, "ops.matmul")

    assert matched_key == "Ops.MatMul"
    assert owner == "owner-a"


def test_set_owner_creates_config_and_preserves_unrelated_fields(tmp_path: Path):
    config = tmp_path / "operator_owners.yaml"
    config.write_text(
        "schema_version: 1\noperators:\n  ops.add: owner-a\n",
        encoding="utf-8",
    )
    original_mode = config.stat().st_mode & 0o777

    saved = OWNER_CONFIG.set_owner(config, "ops.matmul", "@owner-b")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))

    assert saved == {"operator": "ops.matmul", "owner": "owner-b"}
    assert document == {
        "schema_version": 1,
        "operators": {"ops.add": "owner-a", "ops.matmul": "owner-b"},
    }
    assert config.stat().st_mode & 0o777 == original_mode


def test_set_owner_creates_missing_config_with_private_mode(tmp_path: Path):
    config = tmp_path / "operator_owners.yaml"

    OWNER_CONFIG.set_owner(config, "ops.add", "owner-a")

    assert yaml.safe_load(config.read_text(encoding="utf-8")) == {
        "operators": {"ops.add": "owner-a"}
    }
    assert config.stat().st_mode & 0o777 == 0o600


def test_set_owner_updates_existing_key_without_case_duplicate(tmp_path: Path):
    config = tmp_path / "operator_owners.yaml"
    config.write_text("operators:\n  Ops.Add: old-owner\n", encoding="utf-8")

    OWNER_CONFIG.set_owner(config, "ops.add", "new-owner")
    operators = yaml.safe_load(config.read_text(encoding="utf-8"))["operators"]

    assert operators == {"Ops.Add": "new-owner"}


@pytest.mark.parametrize("owner", ["", "direct", "@direct", "two owners", "/assign"])
def test_invalid_or_direct_decision_is_never_persisted(tmp_path: Path, owner: str):
    with pytest.raises(OWNER_CONFIG.ConfigError):
        OWNER_CONFIG.set_owner(tmp_path / "operator_owners.yaml", "ops.add", owner)


def test_default_set_reads_legacy_but_writes_canonical_without_overwrite(tmp_path: Path):
    legacy = tmp_path / "operator_owners.yaml"
    legacy.write_text(
        "schema_version: 1\noperators:\n  ops.add: owner-a\n",
        encoding="utf-8",
    )
    before = legacy.read_text(encoding="utf-8")

    result = _run_owner_config(
        tmp_path,
        "set",
        "--operator",
        "ops.matmul",
        "--owner",
        "owner-b",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    canonical = (
        tmp_path
        / ".cannbot/gitcode-issue-handler/config/operator_owners.yaml"
    )
    document = yaml.safe_load(canonical.read_text(encoding="utf-8"))
    assert payload["config"] == (
        ".cannbot/gitcode-issue-handler/config/operator_owners.yaml"
    )
    assert document["operators"] == {
        "ops.add": "owner-a",
        "ops.matmul": "owner-b",
    }
    assert legacy.read_text(encoding="utf-8") == before


def test_explicit_owner_config_never_falls_back(tmp_path: Path):
    (tmp_path / "operator_owners.yaml").write_text(
        "operators:\n  ops.add: legacy-owner\n",
        encoding="utf-8",
    )

    result = _run_owner_config(
        tmp_path,
        "--config",
        "custom/owners.yaml",
        "lookup",
        "--operator",
        "ops.add",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "missing"
    assert payload["config"] == "custom/owners.yaml"


def test_explicit_canonical_owner_path_does_not_enable_legacy_fallback(
    tmp_path: Path,
):
    (tmp_path / "operator_owners.yaml").write_text(
        "operators:\n  ops.add: legacy-owner\n",
        encoding="utf-8",
    )
    canonical = ".cannbot/gitcode-issue-handler/config/operator_owners.yaml"

    result = _run_owner_config(
        tmp_path,
        "--config",
        canonical,
        "lookup",
        "--operator",
        "ops.add",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "missing"
    assert payload["config"] == canonical
