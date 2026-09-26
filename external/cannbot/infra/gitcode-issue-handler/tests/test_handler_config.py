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
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "handler_config.py"
SPEC = importlib.util.spec_from_file_location("handler_config_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIG)


def _config_path(root: Path) -> Path:
    return root / ".cannbot/gitcode-issue-handler/config/classify_config.yaml"


def test_load_merges_template_defaults_and_partial_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _config_path(tmp_path).parent.mkdir(parents=True)
    _config_path(tmp_path).write_text("repo: example/project\nfollow_up:\n  poll_hours: 12\n", encoding="utf-8")

    loaded = CONFIG.load_handler_config()

    assert loaded["repo"] == "example/project"
    assert loaded["lookback_days"] == 7
    assert loaded["follow_up"]["poll_hours"] == 12
    assert loaded["follow_up"]["stale_hours"] == 48
    assert loaded["auto-response"] is False
    assert loaded["auto-assign"] is False


def test_automation_policy_normalizes_public_yaml_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    loaded = CONFIG.load_handler_config()
    assert CONFIG.get_automation_policy(loaded) == {
        "auto_response": False,
        "auto_assign": False,
    }


def test_automation_true_values_are_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _config_path(tmp_path).parent.mkdir(parents=True)
    _config_path(tmp_path).write_text("auto-response: true\nauto-assign: true\n", encoding="utf-8")
    loaded = CONFIG.load_handler_config()
    assert CONFIG.get_automation_policy(loaded) == {
        "auto_response": True,
        "auto_assign": True,
    }


@pytest.mark.parametrize("value", ["yes", "no", "true", "false", 0, 1, [], {}])
def test_automation_values_must_be_booleans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    monkeypatch.chdir(tmp_path)
    _config_path(tmp_path).parent.mkdir(parents=True)
    import yaml

    encoded = yaml.safe_dump({"auto-response": value}, allow_unicode=True)
    _config_path(tmp_path).write_text(encoded, encoding="utf-8")
    with pytest.raises(ValueError, match="auto-response.*boolean"):
        CONFIG.load_handler_config()


def test_auto_assign_requires_auto_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _config_path(tmp_path).parent.mkdir(parents=True)
    _config_path(tmp_path).write_text("auto-assign: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="auto-assign.*auto-response"):
        CONFIG.load_handler_config()


def test_operator_owner_config_does_not_require_automation_keys(tmp_path: Path) -> None:
    config = tmp_path / ".cannbot/gitcode-issue-handler/config/operator_owners.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("operators: {}\n", encoding="utf-8")
    loaded = CONFIG.load_named_config("operator_owners", config)
    assert "auto-response" not in loaded
    assert "auto-assign" not in loaded


@pytest.mark.parametrize(
    "body, expected",
    [
        ("responsibility: {}\n", {}),
        ("responsibility:\n  handle: []\n", []),
    ],
)
def test_empty_mapping_or_list_replaces_template_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str,
    expected: object,
) -> None:
    monkeypatch.chdir(tmp_path)
    _config_path(tmp_path).parent.mkdir(parents=True)
    _config_path(tmp_path).write_text(body, encoding="utf-8")

    loaded = CONFIG.load_handler_config()

    if body.startswith("responsibility: {}"):
        assert loaded["responsibility"] == expected
    else:
        assert loaded["responsibility"]["handle"] == expected
        assert loaded["responsibility"]["list-only"] == []
        assert loaded["responsibility"]["ignore"] == CONFIG.load_template("classify_config")["responsibility"]["ignore"]


@pytest.mark.parametrize(
    "body, message",
    [
        ("responsibility:\n  route: [anything]\n", "invalid level"),
        ("responsibility:\n  handle: anything\n", "list of strings"),
        ("responsibility:\n  ignore: [3]\n", "list of strings"),
    ],
)
def test_invalid_responsibility_configuration_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    _config_path(tmp_path).parent.mkdir(parents=True)
    _config_path(tmp_path).write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        CONFIG.load_handler_config()


def test_explicit_missing_config_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="config file does not exist"):
        CONFIG.load_handler_config(tmp_path / "missing.yaml")


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("repo: [unterminated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot read YAML config"):
        CONFIG.load_handler_config(config)


def test_initialize_is_idempotent_and_preserves_legacy_values(tmp_path: Path) -> None:
    legacy = tmp_path / "classify_config.yaml"
    legacy.write_text("repo: legacy/project\nreport_file: custom/report.txt\n", encoding="utf-8")
    first = CONFIG.initialize_config(tmp_path)
    canonical = _config_path(tmp_path)
    before = canonical.read_text(encoding="utf-8")

    assert "classify_config" in first
    assert "legacy/project" in before
    assert "custom/report.txt" in before
    assert legacy.read_text(encoding="utf-8") == ("repo: legacy/project\nreport_file: custom/report.txt\n")
    assert CONFIG.initialize_config(tmp_path) == {}
    assert canonical.read_text(encoding="utf-8") == before


def test_initialize_keeps_existing_canonical_and_dangling_symlink(
    tmp_path: Path,
) -> None:
    canonical = _config_path(tmp_path)
    canonical.parent.mkdir(parents=True)
    canonical.write_text("repo: existing/project\n", encoding="utf-8")
    operator = tmp_path / ".cannbot/gitcode-issue-handler/config/operator_owners.yaml"
    operator.symlink_to(tmp_path / "does-not-exist.yaml")
    before = canonical.read_text(encoding="utf-8")

    result = CONFIG.initialize_config(tmp_path)

    assert result == {}
    assert canonical.read_text(encoding="utf-8") == before
    assert operator.is_symlink()


def test_initialization_contains_effective_defaults_in_one_file(tmp_path: Path) -> None:
    CONFIG.initialize_config(tmp_path)
    path = _config_path(tmp_path)
    written = CONFIG.yaml.safe_load(path.read_text(encoding="utf-8"))
    loaded = CONFIG.load_handler_config(path)
    assert written == loaded
    assert loaded == CONFIG.load_template()
    assert loaded["lookback_days"] == 7
    assert loaded["follow_up"]["fetch_pages"] == 10
    assert loaded["auto_close"]["inactive_hours"] == 48
    assert loaded["pr_linkage_api_budget"] == 3


def test_legacy_migration_keeps_advanced_overrides(tmp_path: Path) -> None:
    (tmp_path / "classify_config.yaml").write_text(
        "repo: team/project\nlookback_days: 14\nfollow_up:\n  poll_hours: 12\n", encoding="utf-8"
    )
    CONFIG.initialize_config(tmp_path)
    path = _config_path(tmp_path)
    loaded = CONFIG.load_handler_config(path)
    assert loaded["lookback_days"] == 14
    assert loaded["follow_up"]["poll_hours"] == 12
    assert loaded["follow_up"]["stale_hours"] == 48


@pytest.mark.parametrize(
    "value",
    [
        "123",
        "true",
        "[]",
        '"@songkai111"',
        '"https://gitcode.com/songkai111"',
        '"two users"',
        '"direct"',
    ],
)
def test_invalid_fallback_config_rejected(tmp_path, value):
    path = tmp_path / "classify_config.yaml"
    path.write_text(f"auto-assign-fallback-user: {value}\n")
    with pytest.raises(CONFIG.ConfigError, match="auto-assign-fallback-user"):
        CONFIG.load_handler_config(path)


def test_fallback_default_and_login_override(tmp_path):
    path = tmp_path / "classify_config.yaml"
    path.write_text("{}\n")
    assert CONFIG.load_handler_config(path)["auto-assign-fallback-user"] == ""
    path.write_text('auto-assign-fallback-user: "songkai111"\n')
    assert CONFIG.load_handler_config(path)["auto-assign-fallback-user"] == "songkai111"


@pytest.mark.parametrize(
    "body, field",
    [
        ("follow_up: null\n", "follow_up"),
        ("follow_up: []\n", "follow_up"),
        ("follow_up: {}\n", "follow_up.state_file"),
        ("auto_close: {}\n", "auto_close.comment"),
        ("auto_close:\n  inactive_hours: .nan\n", "auto_close.inactive_hours"),
        ("follow_up:\n  fetch_pages: true\n", "follow_up.fetch_pages"),
        ('follow_up:\n  poll_hours: "24"\n', "follow_up.poll_hours"),
        ("pr_fetch_pages: 0\n", "pr_fetch_pages"),
        ("lookback_days: 999999999999999999999\n", "lookback_days"),
        ("auto_close:\n  inactive_hours: 999999999999999999999\n", "auto_close.inactive_hours"),
        ("pr_linkage_api_budget: -1\n", "pr_linkage_api_budget"),
        ("pr_linkage_scan_mode: typo\n", "pr_linkage_scan_mode"),
        ("ignored_issues_ids: [296]\n", "ignored_issues_ids"),
        ('ignored_issue_ids: ["296"]\n', "ignored_issue_ids"),
        ("repo: [team, repo]\n", "repo"),
        ("gitcode_api: https://host:invalid/api\n", "gitcode_api"),
        ("report_file: null\n", "report_file"),
        ("cache_dir: ~/cache\n", "cache_dir"),
        ("auto_close:\n  question_labels: question\n", "auto_close.question_labels"),
    ],
)
def test_rejects_downstream_failure_inputs(tmp_path, body, field):
    path = tmp_path / "config.yaml"
    path.write_text(body)
    with pytest.raises(CONFIG.ConfigError) as caught:
        CONFIG.load_handler_config(path)
    assert any(e.get("field") == field for e in caught.value.errors)
    assert all(e["file"] == str(path) for e in caught.value.errors)


def test_reports_multiple_errors_and_locations_without_values(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("pr_fetch_pages: false\ncache_dir: []\n")
    with pytest.raises(CONFIG.ConfigError) as caught:
        CONFIG.load_handler_config(path)
    assert {(e["field"], e["line"]) for e in caught.value.errors} == {("pr_fetch_pages", 1), ("cache_dir", 2)}


@pytest.mark.parametrize(
    "body",
    [
        "repo: team/a\nrepo: team/b\n",
        "follow_up:\n  poll_hours: 12\n  poll_hours: 24\n",
        "repo: [unterminated\n",
        "follow_up: &cycle\n  again: *cycle\n",
        "? [invalid, key]\n: value\n",
    ],
)
def test_yaml_errors_are_located_and_do_not_echo_content(tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(body)
    with pytest.raises(CONFIG.ConfigError) as caught:
        CONFIG.load_handler_config(path)
    assert caught.value.errors[0]["line"] > 0
    assert "team/b" not in str(caught.value)


def test_path_collisions_and_file_ancestors_fail_early(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.yaml"
    (tmp_path / "blocker").write_text("keep")
    path.write_text("cache_dir: blocker/cache\nreport_file: config.yaml\n")
    with pytest.raises(CONFIG.ConfigError) as caught:
        CONFIG.load_handler_config(path)
    assert {e["field"] for e in caught.value.errors} == {"cache_dir", "report_file"}
    assert (tmp_path / "blocker").read_text() == "keep"
    path.write_text("last_check_file: shared.json\nfollow_up:\n  state_file: shared.json\n")
    with pytest.raises(CONFIG.ConfigError, match="路径冲突"):
        CONFIG.load_handler_config(path)


def test_partial_overrides_zero_budget_and_yaml_merge_are_valid(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "follow_up:\n  <<: &defaults {poll_hours: 24}\n  poll_hours: 12\n"
        "pr_linkage_api_budget: 0\nauto_close:\n  inactive_hours: 0.5\n"
    )
    config = CONFIG.load_handler_config(path)
    assert config["follow_up"]["poll_hours"] == 12
    assert config["pr_linkage_api_budget"] == 0
    assert config["auto_close"]["inactive_hours"] == 0.5


def test_validation_cli_and_pipeline_stop_without_creating_state(tmp_path):
    import subprocess
    import sys
    import json

    path = _config_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("repo: team/repo\nfollow_up: {}\n")
    for script, args in [("validate_config.py", []), ("issue_pipeline.py", ["resume", "--offline"])]:
        run = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *args, "--repository-root", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 2, run.stderr
        result = json.loads(run.stdout)
        assert result["next_action"] == "fix_configuration"
        assert result["errors"]
        assert not (tmp_path / ".cannbot/gitcode-issue-handler/data").exists()
