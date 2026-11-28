# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Real scoped reads/searches, bounded UTF-8 pages and host interception."""

import hashlib
import json
import shlex
import subprocess
import sys

import pytest

from _checks import cli, host_hooks, host_io
from _checks.paths import SKILL_ROOT


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "schema": "pypto-context-policy/1",
                "roots": [str(tmp_path)],
                "limits": {"max_output_bytes": 2048, "max_files": 3, "max_matches": 2},
            }
        )
    )
    return path, host_io.policy(path)


def test_selected_cards_allow_continuations_without_overriding_policy_roots(tmp_path):
    workspace = tmp_path / "task"
    workspace.mkdir()
    cards = tmp_path / "external cards"
    cards.mkdir()
    card = cards / "card.md"
    card.write_text("Original card text.\n")
    config = host_io.policy(workspace=workspace, cards_root=cards)
    assert host_io.checked_path(card, config) == card
    explicit = tmp_path / "policy.json"
    explicit.write_text(
        json.dumps({"schema": "pypto-context-policy/1", "roots": [str(workspace)]})
    )
    with pytest.raises(ValueError, match="outside"):
        host_io.checked_path(
            card, host_io.policy(explicit, workspace=workspace, cards_root=cards)
        )


def test_cli_split_pattern_is_rejected_before_io_and_explicit_literal_recovery_reads_source(
    tmp_path, config, monkeypatch, capsys
):
    path = tmp_path / "source.py"
    path.write_text("def example():\n    pass\n")
    observed = []
    search = host_io.search

    def record(*args, **kwargs):
        observed.append(kwargs["pattern"])
        return search(*args, **kwargs)

    monkeypatch.setattr(host_io, "search", record)
    prefix = ["search", "--path", str(path), "--policy", str(config[0]), "--pattern"]
    with pytest.raises(SystemExit) as rejected:
        cli.main([*prefix, "def", "example("])
    error = capsys.readouterr()
    assert rejected.value.code == 2 and not observed and not error.out
    assert (
        "--pattern 'whole phrase'" in error.err and "No source search ran" in error.err
    )
    assert cli.main([*prefix, "def example(", "--fixed"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert observed == ["def example("]
    assert (
        result["status"] == "complete"
        and result["items"][0]["text"] == "def example():\n"
    )


def test_cli_unknown_search_option_is_not_reinterpreted_as_pattern_text(
    tmp_path, config, monkeypatch, capsys
):
    monkeypatch.setattr(
        host_io,
        "search",
        lambda *args, **kwargs: pytest.fail("Rejected arguments must not search"),
    )
    with pytest.raises(SystemExit) as rejected:
        cli.main(
            [
                "search",
                "--path",
                str(tmp_path),
                "--pattern",
                "example",
                "--not-an-option",
                "value",
            ]
        )
    error = capsys.readouterr()
    assert rejected.value.code == 2 and not error.out
    assert "unrecognized arguments: --not-an-option value" in error.err
    assert "whole phrase" not in error.err


def test_utf8_continuation_is_exact_and_hash_bound(tmp_path, config):
    _, policy = config
    path = tmp_path / "source.py"
    path.write_text("# 中文边界\n" * 500)
    result = host_io.read([path], config=policy)
    first = result["items"][0]["text"]
    assert result["status"] == "partial" and result["next"]["offset_bytes"] == len(
        first.encode()
    )
    assert (
        len((host_io.encode(result) + "\n").encode())
        == result["budget"]["result_bytes"]
        <= 2048
    )
    second = host_io.read(
        [path],
        config=policy,
        offset_bytes=result["next"]["offset_bytes"],
        source_sha256=result["next"]["source_sha256"],
    )
    assert (first + second["items"][0]["text"]).encode() == path.read_bytes()[
        : len((first + second["items"][0]["text"]).encode())
    ]
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        host_io.read(
            [path],
            config=policy,
            offset_bytes=result["next"]["offset_bytes"],
            source_sha256=result["next"]["source_sha256"],
        )


def test_long_single_line_is_paged_without_silent_loss(tmp_path, config):
    _, policy = config
    path = tmp_path / "one.txt"
    path.write_text("中文abc" * 200)
    pieces, offset, digest = [], None, None
    for _ in range(30):
        result = host_io.read(
            [path], config=policy, offset_bytes=offset, source_sha256=digest
        )
        pieces.append(result["items"][0]["text"])
        assert result["budget"]["result_bytes"] <= 2048
        if result["status"] == "complete":
            break
        offset, digest = result["next"]["offset_bytes"], result["next"]["source_sha256"]
    assert "".join(pieces) == path.read_text()


def test_numbered_view_binds_blank_crlf_utf8_lines_to_original_bytes(tmp_path):
    path = tmp_path / "original.py"
    data = "\r\n\r\ndef measure(addr):\r\n    # 地址零合法\r\n    return addr >= 0\r\n".encode()
    path.write_bytes(data)
    policy = host_io.policy(workspace=tmp_path)
    item = host_io.read([path], config=policy, start_line=2, max_lines=4)["items"][0]
    assert item["sha256"] == hashlib.sha256(data).hexdigest()
    assert (
        item["text"]
        == "\r\ndef measure(addr):\r\n    # 地址零合法\r\n    return addr >= 0\r\n"
    )
    assert (
        item["numbered_text"]
        == "2|\r\n3|def measure(addr):\r\n4|    # 地址零合法\r\n5|    return addr >= 0\r\n"
    )
    assert item["start_column_bytes"] == 0
    offset = data.index("零".encode())
    part = host_io.read(
        [path], config=policy, offset_bytes=offset, source_sha256=item["sha256"]
    )["items"][0]
    assert part["start_line"] == 4 and part["start_column_bytes"] == len(
        "    # 地址".encode()
    )
    assert part["numbered_text"].startswith("4|零合法\r\n5|")
    with pytest.raises(ValueError, match="UTF-8 character boundary"):
        host_io.read(
            [path], config=policy, offset_bytes=offset + 1, source_sha256=item["sha256"]
        )


def test_heading_fragments_share_slugs_and_keep_byte_continuation_inside_the_section(
    tmp_path, config
):
    _, policy = config
    path = tmp_path / "headings.md"
    text = (
        "# Guide\r\n## Usage\r\nFirst.\r\n```python\r\n## Fake heading\r\n```\r\n## Usage\r\n"
        + ("中文 continuation\r\n" * 80)
        + "## Next\r\nMust not read this.\r\n"
    )
    path.write_bytes(text.encode())
    pieces = []
    request = {"fragment": "usage-1"}
    for _ in range(20):
        result = host_io.read([path], config=policy, **request)
        pieces.append(result["items"][0]["text"])
        assert (
            len((host_io.encode(result) + "\n").encode())
            <= policy["limits"]["max_output_bytes"]
        )
        if result["status"] == "complete":
            break
        request = {key: value for key, value in result["next"].items() if key != "path"}
    combined = "".join(pieces)
    assert combined == "## Usage\r\n" + ("中文 continuation\r\n" * 80)
    assert (
        "Must not read" not in combined and result["items"][0]["file_complete"] is False
    )


def test_fragment_batch_reads_and_explicit_anchors_are_bounded(tmp_path, config):
    policy = host_io.policy(workspace=tmp_path)
    path = tmp_path / "anchors.md"
    path.write_text(
        '# Guide\n<a id="sample"></a>\n## Example\nSelected.\n## Other\nExcluded.\n'
    )
    result = host_io.read_requests(
        [{"path": str(path), "fragment": "sample"}], config=policy
    )
    assert (
        "Selected." in result["items"][0]["text"]
        and "Excluded." not in result["items"][0]["text"]
    )
    with pytest.raises(ValueError, match="fragment_missing"):
        host_io.read([path], config=policy, fragment="missing")
    with pytest.raises(ValueError, match="outside"):
        host_io.read(
            [path],
            config=policy,
            fragment="sample",
            offset_bytes=path.stat().st_size,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_parent_and_symlink_escapes_are_rejected(tmp_path, config):
    _, policy = config
    with pytest.raises(ValueError, match="outside"):
        host_io.read([tmp_path.parent / "outside.py"], config=policy)
    link = tmp_path / "escape"
    link.symlink_to(SKILL_ROOT / "SKILL.md")
    with pytest.raises(ValueError, match="outside"):
        host_io.read([link], config=policy)


def test_search_retains_source_identity_and_reports_match_cap(tmp_path, config):
    _, policy = config
    path = tmp_path / "input.py"
    path.write_text("FP16 line\n" * 20)
    result = host_io.search([path], config=policy, pattern="fp16", ignore_case=True)
    assert result["status"] == "partial" and not result["coverage"]["complete"]
    assert len(result["items"]) <= 2
    assert result["items"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["budget"]["result_bytes"] <= 2048


def test_file_enumeration_is_bounded_and_explicit(tmp_path, config):
    _, policy = config
    for i in range(10):
        (tmp_path / f"item{i}.py").write_text("x=1")
    result = host_io.search([tmp_path], config=policy, files=True, glob="*.py")
    assert result["status"] == "partial" and len(result["items"]) <= 2
    assert result["coverage"]["paths_seen"] <= 4
    with pytest.raises(ValueError, match="explicit"):
        host_io.search([], config=policy, files=True)


def test_missing_search_backend_does_not_establish_absence(
    tmp_path, config, monkeypatch
):
    monkeypatch.setattr(host_io.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="ripgrep"):
        host_io.search([tmp_path], config=config[1], pattern="missing")


def test_invalid_regex_explains_literal_search_without_changing_semantics(
    tmp_path, config
):
    _, policy = config
    path = tmp_path / "kernel.py"
    path.write_text("one = vf.full(1.0, preg)\n")
    with pytest.raises(ValueError, match="search --fixed"):
        host_io.search([path], config=policy, pattern="vf.full(")
    result = host_io.search([path], config=policy, pattern="vf.full(", fixed=True)
    assert result["status"] == "complete" and result["coverage"]["complete"]
    assert len(result["items"]) == 1 and result["items"][0]["line"] == 1
    assert result["items"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def hook(tmp_path, policy, command):
    return host_hooks.pre_tool(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "cwd": str(tmp_path),
            "tool_input": {"command": command},
        },
        policy_path=policy,
    )


def test_hook_rewrites_only_supported_reads_and_preserves_execution(tmp_path, config):
    policy, _ = config
    path = tmp_path / "space file.py"
    path.write_text("print('ok')\n")
    result = hook(tmp_path, policy, "cat 'space file.py'")
    updated = result["hookSpecificOutput"]["updatedInput"]["command"]
    run = subprocess.run(
        shlex.split(updated), cwd=tmp_path, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["items"][0]["text"] == path.read_text()
    assert hook(tmp_path, policy, "python evaluator.py candidate.py") == {}
    assert hook(tmp_path, policy, "python -c 'print(123)'") == {}


@pytest.mark.parametrize(
    "command",
    [
        "rg --files",
        "rg x . | head -20",
        "cat missing.py",
        "cat /etc/passwd",
        "find . -type f",
    ],
)
def test_unscoped_unsupported_and_outside_reads_are_not_executed(
    tmp_path, config, command
):
    result = hook(tmp_path, config[0], command)
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "updatedInput" not in result["hookSpecificOutput"]


def test_pinned_receipt_read_becomes_status_not_a_relaunch(tmp_path):
    source, receipt = tmp_path / "candidate.py", tmp_path / "receipt.json"
    source.write_text("x=1\n")
    receipt.write_text("{}")
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "pypto-context-policy/1",
                "roots": [str(tmp_path)],
                "receipts": [
                    {
                        "path": str(receipt),
                        "source": str(source),
                        "request_id": "original-job",
                    }
                ],
            }
        )
    )
    result = hook(tmp_path, policy, "cat receipt.json")
    command = shlex.split(result["hookSpecificOutput"]["updatedInput"]["command"])
    assert "status" in command and "original-job" in command and "20" in command
    assert "run" not in command
    denied = hook(tmp_path, policy, "cat receipt.json candidate.py")
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_counters_overwrite_bounded_state_without_command_history(tmp_path):
    path = tmp_path / "counters.json"
    payload = {"session_id": "trial"}
    for _ in range(10):
        host_hooks.count(
            path, payload, {"hookSpecificOutput": {"permissionDecision": "deny"}}
        )
    data = json.loads(path.read_text())
    assert data == {
        "schema": "pypto-host-counters/1",
        "session_id": "trial",
        "calls": 10,
        "denied": 10,
        "rewritten": 0,
    }
    assert path.stat().st_size < 200


def test_read_checks_scan_budget_before_consuming_file(tmp_path, config, monkeypatch):
    _, policy = config
    path = tmp_path / "large.txt"
    path.write_text("x" * 1000)
    policy["limits"]["max_scan_bytes"] = 10
    original = type(path).open

    def cannot_read(self, *args, **kwargs):
        if self == path:
            pytest.fail("over-budget file must not be read")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(path), "open", cannot_read)
    with pytest.raises(ValueError, match="budget"):
        host_io.read([path], config=policy)


def test_cli_invalid_read_still_has_a_small_explicit_budget(tmp_path, config):
    run = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts/check.py"),
            "read",
            "--policy",
            str(config[0]),
            "--path",
            str(tmp_path / "missing.py"),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2
    result = json.loads(run.stdout)
    assert result["status"] == "unverified" and not result["coverage"]["complete"]
    assert len(run.stdout.encode()) == result["budget"]["result_bytes"] <= 2048


def test_explicit_line_range_can_finish_before_end_of_file(tmp_path, config):
    path = tmp_path / "many.py"
    path.write_text("x=1\n" * 500)
    result = host_io.read([path], config=config[1], start_line=100, max_lines=10)
    assert result["status"] == "complete"
    assert (
        result["items"][0]["request_complete"]
        and not result["items"][0]["file_complete"]
    )
    assert result["items"][0]["text"] == "x=1\n" * 10 and "next" not in result


def test_continuation_preserves_explicit_range_instead_of_reading_rest(
    tmp_path, config
):
    path = tmp_path / "range.py"
    path.write_text("x=1\n" * 500)
    first = host_io.read([path], config=config[1], start_line=100, max_lines=140)
    assert first["status"] == "partial"
    continuation = first["next"]
    pieces = [first["items"][0]["text"]]
    for _ in range(20):
        page = host_io.read(
            [path],
            config=config[1],
            offset_bytes=continuation["offset_bytes"],
            source_sha256=continuation["source_sha256"],
            max_lines=continuation["max_lines"],
        )
        pieces.append(page["items"][0]["text"])
        if page["status"] == "complete":
            break
        continuation = page["next"]
    assert page["status"] == "complete"
    assert "".join(pieces) == "x=1\n" * 140


def test_host_combines_only_pure_cat_and_chains(tmp_path, config):
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("value=1\n")
    result = hook(tmp_path, config[0], "cat a.py && cat b.py")
    command = shlex.split(result["hookSpecificOutput"]["updatedInput"]["command"])
    assert command.count("--path") == 2 and "--max-lines" not in command
    assert (
        hook(tmp_path, config[0], "cat a.py && python evaluator.py")[
            "hookSpecificOutput"
        ]["permissionDecision"]
        == "deny"
    )


def test_advertised_cli_reads_cannot_select_a_different_host_policy(tmp_path, config):
    command = shlex.join(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts/check.py"),
            "read",
            "--path",
            "candidate.py",
            "--policy",
            "/other/policy.json",
        ]
    )
    result = hook(tmp_path, config[0], command)
    updated = shlex.split(result["hookSpecificOutput"]["updatedInput"]["command"])
    assert updated[updated.index("--policy") + 1] == str(config[0])
    assert "/other/policy.json" not in updated
    query = shlex.join(
        [sys.executable, str(SKILL_ROOT / "scripts/check.py"), "query", "pl.store"]
    )
    assert hook(tmp_path, config[0], query) == {}
    inventory = shlex.join(
        [sys.executable, str(SKILL_ROOT / "scripts/check.py"), "api", "pl"]
    )
    assert (
        hook(tmp_path, config[0], inventory)["hookSpecificOutput"]["permissionDecision"]
        == "deny"
    )


def test_pinned_status_cannot_adopt_a_replacement_request(tmp_path):
    source, receipt, policy = (
        tmp_path / "candidate.py",
        tmp_path / "receipt.json",
        tmp_path / "policy.json",
    )
    source.write_text("x=1\n")
    receipt.write_text(
        json.dumps(
            {
                "request_id": "replacement",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "state": "completed",
                "exit_code": 0,
            }
        )
    )
    policy.write_text(
        json.dumps(
            {
                "schema": "pypto-context-policy/1",
                "roots": [str(tmp_path)],
                "receipts": [
                    {
                        "path": str(receipt),
                        "source": str(source),
                        "request_id": "original",
                    }
                ],
            }
        )
    )
    run = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts/check.py"),
            "status",
            "--receipt",
            str(receipt),
            "--source",
            str(source),
            "--request-id",
            "replacement",
            "--wait-seconds",
            "0",
            "--policy",
            str(policy),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2
    result = json.loads(run.stdout)
    assert result["status"] == "unverified" and "controller-pinned" in result["error"]


def test_policy_validation_refusal_is_counted_before_any_command(tmp_path):
    counters = tmp_path / "counters.json"
    message = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "session_id": "same-session",
        "cwd": str(tmp_path),
        "tool_input": {"command": "cat input.py"},
    }
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts/check.py"),
            "host-hook",
            "--policy",
            str(tmp_path / "missing-policy.json"),
            "--counters",
            str(counters),
        ],
        input=json.dumps(message),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert (
        json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
    )
    data = json.loads(counters.read_text())
    assert data["calls"] == data["denied"] == 1
