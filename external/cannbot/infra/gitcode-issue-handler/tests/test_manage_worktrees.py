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

import json
import subprocess
import sys
from pathlib import Path

import pytest

HANDLER_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = HANDLER_ROOT / "scripts" / "manage_worktrees.py"


def run(
    *args: str,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [*args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result


class WorktreeHarness:
    def __init__(self, root: Path) -> None:
        self.repo = root / "repo"
        self.worktrees = root / "worktrees"
        self.manifest = root / "manifests" / "run-1.json"
        run("git", "init", "-b", "master", str(self.repo))
        run("git", "config", "user.name", "Test User", cwd=self.repo)
        run("git", "config", "user.email", "test@example.com", cwd=self.repo)
        (self.repo / "README.md").write_text("base\n", encoding="utf-8")
        run("git", "add", "README.md", cwd=self.repo)
        run("git", "commit", "-m", "initial", cwd=self.repo)

    def create(self, group_id: str = "g1", branch: str = "fix/g1-run-1") -> Path:
        result = run(
            sys.executable,
            str(SCRIPT),
            "create",
            "--repo-root",
            str(self.repo),
            "--manifest",
            str(self.manifest),
            "--worktree-root",
            str(self.worktrees),
            "--run-id",
            "run-1",
            "--group-id",
            group_id,
            "--branch",
            branch,
            "--base-ref",
            "master",
            "--wave",
            "1",
            "--planned-path",
            "src/a.py",
        )
        payload = json.loads(result.stdout)["result"]
        return Path(payload["worktree_path"])

    def mark(self, status: str, group_id: str = "g1") -> None:
        run(
            sys.executable,
            str(SCRIPT),
            "mark",
            "--manifest",
            str(self.manifest),
            "--group-id",
            group_id,
            "--status",
            status,
        )

    def cleanup(self, check: bool = True) -> subprocess.CompletedProcess[str]:
        return run(
            sys.executable,
            str(SCRIPT),
            "cleanup",
            "--manifest",
            str(self.manifest),
            check=check,
        )


@pytest.fixture
def worktree(tmp_path: Path) -> WorktreeHarness:
    return WorktreeHarness(tmp_path)


def test_conflicting_paths_and_resources_are_serialized(tmp_path: Path):
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(
        json.dumps(
            {
                "groups": [
                    {"group_id": "g1", "planned_paths": ["src/a.py"]},
                    {"group_id": "g2", "planned_paths": ["src/b.py"]},
                    {"group_id": "g3", "planned_paths": ["src"]},
                    {
                        "group_id": "g4",
                        "planned_paths": ["docs/a.md"],
                        "exclusive_resources": ["npu:0"],
                    },
                    {
                        "group_id": "g5",
                        "planned_paths": ["docs/b.md"],
                        "exclusive_resources": ["npu:0"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    result = run(sys.executable, str(SCRIPT), "plan", "--groups-json", str(groups_path))
    plan = json.loads(result.stdout)["result"]

    assert plan["assignments"]["g1"] == 1
    assert plan["assignments"]["g2"] == 1
    assert plan["assignments"]["g3"] != 1
    assert plan["assignments"]["g4"] != plan["assignments"]["g5"]


def test_unknown_scope_conflicts_with_every_group(tmp_path: Path):
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(
        json.dumps(
            [
                {"group_id": "known", "planned_paths": ["src/a.py"]},
                {"group_id": "unknown", "planned_paths": []},
            ]
        ),
        encoding="utf-8",
    )
    result = run(sys.executable, str(SCRIPT), "plan", "--groups-json", str(groups_path))
    plan = json.loads(result.stdout)["result"]

    assert plan["assignments"]["known"] != plan["assignments"]["unknown"]


def test_clean_terminal_worktree_is_removed_but_branch_is_retained(
    worktree: WorktreeHarness,
):
    path = worktree.create()
    worktree.mark("no_changes")
    result = worktree.cleanup()
    payload = json.loads(result.stdout)["result"]

    assert not path.exists()
    assert payload["cleaned"][0]["group_id"] == "g1"
    branches = run(
        "git", "branch", "--format=%(refname:short)", cwd=worktree.repo
    ).stdout.splitlines()
    assert "fix/g1-run-1" in branches


def test_active_worktree_is_not_removed(worktree: WorktreeHarness):
    path = worktree.create()
    result = worktree.cleanup()
    payload = json.loads(result.stdout)["result"]

    assert path.exists()
    assert payload["skipped"][0]["reason"] == "non_terminal:active"


def test_dirty_terminal_worktree_is_not_removed(worktree: WorktreeHarness):
    path = worktree.create()
    (path / "untracked.txt").write_text("keep me\n", encoding="utf-8")
    worktree.mark("cancelled_clean")
    result = worktree.cleanup()
    payload = json.loads(result.stdout)["result"]

    assert path.exists()
    assert payload["skipped"][0]["reason"] == "dirty_worktree"


def test_multiple_group_worktrees_coexist_and_cleanup_independently(
    worktree: WorktreeHarness,
):
    first = worktree.create()
    second = worktree.create(group_id="g2", branch="fix/g2-run-1")
    assert first.is_dir()
    assert second.is_dir()

    worktree.mark("no_changes", group_id="g1")
    worktree.mark("no_changes", group_id="g2")
    result = worktree.cleanup()
    payload = json.loads(result.stdout)["result"]

    assert {item["group_id"] for item in payload["cleaned"]} == {"g1", "g2"}
    assert not first.exists()
    assert not second.exists()


def test_worktree_root_inside_repository_is_rejected(worktree: WorktreeHarness):
    result = run(
        sys.executable,
        str(SCRIPT),
        "create",
        "--repo-root",
        str(worktree.repo),
        "--manifest",
        str(worktree.manifest),
        "--worktree-root",
        str(worktree.repo / "worktrees"),
        "--run-id",
        "run-1",
        "--group-id",
        "g1",
        "--branch",
        "fix/g1-run-1",
        "--base-ref",
        "master",
        "--wave",
        "1",
        check=False,
    )

    assert result.returncode == 2
    assert "outside" in result.stderr
