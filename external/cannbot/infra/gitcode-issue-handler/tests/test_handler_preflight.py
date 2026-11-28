# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import os
import shutil
import subprocess
from pathlib import Path

HANDLER_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = HANDLER_ROOT / "scripts" / "preflight.sh"
GIT = shutil.which("git")
assert GIT is not None


def run_preflight(repo: Path, *args: str, token: str | None = None):
    env = os.environ.copy()
    env.pop("GITCODE_TOKEN", None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    if token is not None:
        env["GITCODE_TOKEN"] = token
    result = subprocess.run(
        [str(PREFLIGHT), "--work-dir", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return result, json.loads(result.stdout)


def test_preflight_full_compat_aggregates_user_inputs(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([GIT, "init", "-q", str(repo)], check=True)

    result, report = run_preflight(repo)

    assert result.returncode == 1
    assert report["ready"] is False
    assert report["action"] == "request_inputs"
    assert report["needs_user"] == ["token", "git_author"]
    assert report["blockers"] == []


def test_preflight_accepts_session_token_and_local_author(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([GIT, "init", "-q", str(repo)], check=True)
    subprocess.run(
        [GIT, "-C", str(repo), "config", "user.name", 'Issue "Bot"'], check=True
    )
    subprocess.run(
        [GIT, "-C", str(repo), "config", "user.email", "issue-bot@example.com"],
        check=True,
    )

    result, report = run_preflight(repo, "--token-available")

    assert result.returncode == 0
    assert report["ready"] is True
    assert report["action"] == "continue"
    assert report["needs_user"] == []
    assert report["blockers"] == []
    author_result = next(
        item for item in report["results"] if item["item"] == "git_author"
    )
    assert 'Issue "Bot"' in author_result["detail"]


def test_preflight_never_echoes_token_value(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([GIT, "init", "-q", str(repo)], check=True)
    subprocess.run(
        [GIT, "-C", str(repo), "config", "user.name", "Issue Bot"], check=True
    )
    subprocess.run(
        [GIT, "-C", str(repo), "config", "user.email", "issue-bot@example.com"],
        check=True,
    )
    secret = "token-value-must-not-appear"

    result, report = run_preflight(repo, token=secret)

    assert result.returncode == 0
    assert secret not in result.stdout
    token_result = next(item for item in report["results"] if item["item"] == "token")
    assert token_result["source"] == "env"


def test_preflight_api_only_checks_api_dependencies(tmp_path: Path):
    result, report = run_preflight(tmp_path, "--checks", "api", token="api-token")

    assert result.returncode == 0
    assert report["requested_checks"] == "api"
    assert [item["item"] for item in report["results"]] == [
        "token",
        "curl",
        "python3",
    ]
    assert report["summary"] == {"pass": 3, "fail": 0, "total": 3}


def test_preflight_git_and_tmp_only(tmp_path: Path):
    result, report = run_preflight(tmp_path, "--checks", "git,tmp")

    assert result.returncode == 0
    assert [item["item"] for item in report["results"]] == ["git", "tmp"]
    assert report["summary"] == {"pass": 2, "fail": 0, "total": 2}


def test_preflight_author_only_includes_git_dependency(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([GIT, "init", "-q", str(repo)], check=True)
    subprocess.run([GIT, "-C", str(repo), "config", "user.name", "Issue Bot"], check=True)
    subprocess.run(
        [GIT, "-C", str(repo), "config", "user.email", "issue-bot@example.com"],
        check=True,
    )

    result, report = run_preflight(repo, "--checks", "author")

    assert result.returncode == 0
    assert [item["item"] for item in report["results"]] == ["git", "git_author"]
    assert report["summary"] == {"pass": 2, "fail": 0, "total": 2}


def test_preflight_combines_groups_without_duplicate_results(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([GIT, "init", "-q", str(repo)], check=True)
    subprocess.run([GIT, "-C", str(repo), "config", "user.name", "Issue Bot"], check=True)
    subprocess.run(
        [GIT, "-C", str(repo), "config", "user.email", "issue-bot@example.com"],
        check=True,
    )

    result, report = run_preflight(
        repo, "--checks", "api,git,tmp,author", token="api-token"
    )

    assert result.returncode == 0
    assert [item["item"] for item in report["results"]] == [
        "token",
        "git",
        "curl",
        "python3",
        "tmp",
        "git_author",
    ]
    assert report["summary"] == {"pass": 6, "fail": 0, "total": 6}


def test_preflight_summary_counts_only_requested_checks(tmp_path: Path):
    result, report = run_preflight(tmp_path, "--checks", "api,git")

    assert result.returncode == 1
    assert report["needs_user"] == ["token"]
    assert report["blockers"] == []
    assert report["summary"] == {"pass": 3, "fail": 1, "total": 4}


def test_preflight_does_not_probe_unselected_checks(tmp_path: Path):
    # Neither a token nor git author is configured. A git-only gate must still pass.
    result, report = run_preflight(tmp_path, "--checks", "git")

    assert result.returncode == 0
    assert report["ready"] is True
    assert report["action"] == "continue"
    assert report["needs_user"] == []
    assert report["blockers"] == []
    assert [item["item"] for item in report["results"]] == ["git"]
    assert report["summary"] == {"pass": 1, "fail": 0, "total": 1}


def test_preflight_rejects_ambiguous_check_arguments(tmp_path: Path):
    env = os.environ.copy()
    env.pop("GITCODE_TOKEN", None)

    trailing_comma = subprocess.run(
        [str(PREFLIGHT), "--checks", "api,", "--work-dir", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    repeated = subprocess.run(
        [str(PREFLIGHT), "--checks", "api", "--checks", "git"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert trailing_comma.returncode == 2
    assert "empty check group" in trailing_comma.stderr
    assert repeated.returncode == 2
    assert "only be specified once" in repeated.stderr


def test_author_gate_does_not_request_identity_when_git_is_missing(tmp_path: Path):
    env = os.environ.copy()
    env.pop("GITCODE_TOKEN", None)
    env["PATH"] = "/nonexistent"

    result = subprocess.run(
        [str(PREFLIGHT), "--checks", "author", "--work-dir", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    report = json.loads(result.stdout)

    assert result.returncode == 1
    assert report["action"] == "report_blockers"
    assert report["needs_user"] == []
    assert report["blockers"] == ["git"]
    assert report["results"][1]["status"] == "skip"
