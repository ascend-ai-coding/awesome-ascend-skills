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
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "resolve_repository.py"
SPEC = importlib.util.spec_from_file_location("resolve_repository_under_test", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
GIT = shutil.which("git")
assert GIT is not None


def cfg(root: Path, body: str) -> Path:
    path = root / ".cannbot/gitcode-issue-handler/config/classify_config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(body, encoding="utf-8")
    return path


def git(root: Path, *args: str) -> None:
    subprocess.run([GIT, "-C", str(root), *args], check=True, capture_output=True)


def test_unique_remote_is_saved_and_comments_survive(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    git(tmp_path, "remote", "add", "origin", "https://gitcode.com/acme/widget.git")
    path = cfg(tmp_path, "# keep\nrepo: \"\"  # selected\nother: value\n")
    result = MOD.resolve_repository(tmp_path)
    assert result["status"] == "resolved" and result["repo"] == "acme/widget"
    assert path.read_text(encoding="utf-8") == '# keep\nrepo: "acme/widget"  # selected\nother: value\n'


def test_configured_repo_wins_and_conflicting_target_needs_selection(tmp_path: Path) -> None:
    cfg(tmp_path, "repo: old/repo\n")
    assert MOD.resolve_repository(tmp_path)["repo"] == "old/repo"
    result = MOD.resolve_repository(tmp_path, target="https://gitcode.com/new/repo/issues/2")
    assert result["status"] == "needs_selection"
    assert result["candidates"] == ["old/repo", "new/repo"]


def test_selection_changes_existing_value(tmp_path: Path) -> None:
    path = cfg(tmp_path, "repo: old/repo\n")
    result = MOD.resolve_repository(tmp_path, selection="new/repo")
    assert result["status"] == "resolved" and result["repo"] == "new/repo"
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["repo"] == "new/repo"


def test_remotes_are_deduplicated_and_non_gitcode_excluded(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    git(tmp_path, "remote", "add", "a", "git@gitcode.com:acme/widget.git")
    git(tmp_path, "remote", "add", "b", "https://gitcode.com/acme/widget.git")
    git(tmp_path, "remote", "add", "c", "https://github.com/acme/other.git")
    result = MOD.resolve_repository(tmp_path)
    assert result["status"] == "resolved" and result["repo"] == "acme/widget"


def test_multiple_candidates_require_selection(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    git(tmp_path, "remote", "add", "a", "https://gitcode.com/a/one.git")
    git(tmp_path, "remote", "add", "b", "ssh://git@gitcode.com/b/two.git")
    result = MOD.resolve_repository(tmp_path)
    assert result["status"] == "needs_selection"
    assert result["candidates"] == ["a/one", "b/two"]



@pytest.mark.parametrize(
    "body",
    [
        "",
        "# keep\n",
        "other: value\n",
        "repo:\nother: value\n",
        "repo: null\n",
        "{}\n",
        "{other: value}\n",
        "other: value\n...\n",
    ],
)
def test_fill_missing_or_empty_repo_preserves_values(tmp_path, body):
    path = cfg(tmp_path, body)
    before = yaml.safe_load(body) or {}
    result = MOD.resolve_repository(tmp_path, target="team/math")
    assert result["status"] == "resolved"
    assert yaml.safe_load(path.read_text()) == {**before, "repo": "team/math"}
    if "# keep" in body:
        assert "# keep" in path.read_text()


def test_questionnaire_selection_resumes_and_persists(tmp_path, capsys):
    git(tmp_path, "init", "-q")
    assert MOD.main(["--repository-root", str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "needs_selection"
    assert MOD.main(["--repository-root", str(tmp_path), "--select", "chosen/math"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert yaml.safe_load(Path(result["config_path"]).read_text())["repo"] == "chosen/math"
    assert MOD.resolve_repository(tmp_path)["repo"] == "chosen/math"


def test_legacy_is_copied_and_preserved(tmp_path):
    legacy = tmp_path / "classify_config.yaml"
    legacy.write_text('# important\nrepo: ""\nother: value\n')
    before = legacy.read_text()
    result = MOD.resolve_repository(tmp_path, target="team/math")
    assert legacy.read_text() == before
    assert Path(result["config_path"]) != legacy
    assert "# important" in Path(result["config_path"]).read_text()


def test_explicit_missing_config_errors(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        MOD.resolve_repository(tmp_path, target="team/math", config_path="missing.yaml")
