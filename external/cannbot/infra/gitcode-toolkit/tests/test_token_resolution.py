# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Credential reuse using mocked helpers and isolated, fake credential stores."""

import json
import os
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_gitcode_client import CLIENT, SCRIPT


def test_explicit_and_environment_tokens_do_not_query_git(monkeypatch):
    monkeypatch.setenv("GITCODE_TOKEN", "environment-token")
    monkeypatch.setattr(
        CLIENT, "git_credential_token", lambda *a, **kw: pytest.fail("Git must not run")
    )
    assert CLIENT.resolve_token("explicit-token") == "explicit-token"
    assert CLIENT.resolve_token() == "environment-token"


def test_target_repository_and_username_select_credential(monkeypatch, tmp_path):
    monkeypatch.delenv("GITCODE_TOKEN", raising=False)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0, stdout="username=alice\npassword=secret=with=equals\n"
        )

    monkeypatch.setattr(CLIENT.subprocess, "run", run)
    assert (
        CLIENT.resolve_token(
            repo_url="https://alice@gitcode.com/cann/repo/issues/42",
            repository_root=tmp_path,
        )
        == "secret=with=equals"
    )
    command, options = calls[0]
    assert command[-2:] == ["credential", "fill"]
    assert (
        options["input"]
        == "protocol=https\nhost=gitcode.com\npath=cann/repo\nusername=alice\n\n"
    )
    assert options["cwd"] == tmp_path
    assert options["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert options["env"]["GIT_ASKPASS"] == "/bin/false"
    assert options["timeout"] == 5


@pytest.mark.parametrize(
    "url",
    [
        "https://gitcode.com.evil.test/cann/repo",
        "http://gitcode.com/cann/repo",
        "file:///cann/repo",
        "https://gitcode.com/cann",
        "https://[broken",
        "https://alice%0Apassword=injected@gitcode.com/cann/repo",
        "https://gitcode.com/cann/repo%0Ausername=injected",
    ],
)
def test_invalid_context_never_queries_helper(monkeypatch, url):
    monkeypatch.setattr(
        CLIENT.subprocess, "run", lambda *a, **kw: pytest.fail("untrusted context")
    )
    assert CLIENT.git_credential_token(url) is None


@pytest.mark.parametrize(
    "response",
    [
        "username=bob\npassword=wrong-account\n",
        "host=other.test\nusername=alice\npassword=wrong-host\n",
        "protocol=http\nusername=alice\npassword=wrong-protocol\n",
        "path=other/repo\nusername=alice\npassword=wrong-repository\n",
        "username=alice\n",
        "password=missing-username\n",
    ],
)
def test_mismatched_or_incomplete_credentials_are_not_used(monkeypatch, response):
    monkeypatch.setattr(
        CLIENT.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=response),
    )
    assert CLIENT.git_credential_token("https://alice@gitcode.com/cann/repo") is None


@pytest.mark.parametrize(
    "error",
    [
        OSError("secret-in-error"),
        subprocess.TimeoutExpired("git", 5, output="secret-in-error"),
    ],
)
def test_helper_errors_fail_without_exposing_secret(monkeypatch, error):
    monkeypatch.delenv("GITCODE_TOKEN", raising=False)

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(CLIENT.subprocess, "run", run)
    with pytest.raises(CLIENT.GitCodeClientError) as caught:
        CLIENT.resolve_token(repo_url="https://gitcode.com/cann/repo")
    assert "secret-in-error" not in str(caught.value)


def test_ssh_origin_only_locates_existing_https_credentials(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-3:] == ["remote", "get-url", "origin"]:
            return SimpleNamespace(
                returncode=0, stdout="git@gitcode.com:cann/repo.git\n"
            )
        return SimpleNamespace(
            returncode=0, stdout="username=alice\npassword=https-token\n"
        )

    monkeypatch.setattr(CLIENT.subprocess, "run", run)
    assert CLIENT.git_credential_token(repository_root=tmp_path) == "https-token"
    assert (
        calls[1][1]["input"]
        == "protocol=https\nhost=gitcode.com\npath=cann/repo.git\n\n"
    )


def test_preflights_reuse_path_scoped_credentials_without_printing_them(tmp_path):
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    env.pop("GITCODE_TOKEN", None)
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], env=env, check=True)
    credentials = tmp_path / "fake-credentials"
    credentials.write_text(
        "https://saved-user:fixture-secret@gitcode.com/cann/repo.git\n"
    )

    def config(key, value):
        subprocess.run(
            ["git", "-C", str(repo), "config", key, value], env=env, check=True
        )

    config("credential.helper", "store --file " + shlex.quote(str(credentials)))
    config("credential.useHttpPath", "true")
    config("remote.origin.url", "https://saved-user@gitcode.com/cann/repo.git")
    before = credentials.read_text()
    commands = [
        [sys.executable, str(SCRIPT), "--check-token", "--repository-root", str(repo)],
        ["bash", str(SCRIPT.with_name("preflight.sh")), "--skip-git-author"],
        [
            "bash",
            str(SCRIPT.parents[2] / "gitcode-issue-handler/scripts/preflight.sh"),
            "--checks",
            "api",
            "--work-dir",
            str(repo),
        ],
    ]
    for command in commands:
        result = subprocess.run(
            command, cwd=repo, env=env, text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr
        assert "fixture-secret" not in result.stdout + result.stderr
        report = json.loads(result.stdout)
        token = next(
            (item for item in report.get("results", []) if item["item"] == "token"),
            report,
        )
        assert token["source"] == "git_credential"
    assert credentials.read_text() == before
    config("remote.origin.url", "https://gitcode.com/other/repo.git")
    result = subprocess.run(
        commands[0], cwd=repo, env=env, text=True, capture_output=True
    )
    assert result.returncode == 1
    assert "fixture-secret" not in result.stdout + result.stderr
