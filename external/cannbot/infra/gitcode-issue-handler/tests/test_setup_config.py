# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cover setup decisions, installer compatibility and configuration preservation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_ERROR_EXIT = 2


@pytest.fixture(name="setup")
def setup_fixture(monkeypatch):
    """Load the CLI module with its sibling imports available only for this test."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("setup_config")


def configuration(root):
    """Locate the repository policy file without creating it."""
    return root / ".cannbot/gitcode-issue-handler/config/classify_config.yaml"


def snapshot(root):
    """Capture YAML bytes to detect unintended file rewrites."""
    return {path.name: path.read_bytes() for path in configuration(root).parent.glob("*.yaml")}


def test_old_installer_without_config_enters_pending_setup(setup, tmp_path):
    """Create missing templates with automation disabled and await a user decision."""
    result = setup.prepare_setup(tmp_path)
    assert result["status"] == "pending"
    assert result["needs_configuration"] is True
    assert set(result["created"]) == {"classify_config", "operator_owners"}
    assert result["settings"]["auto-response"] is False
    assert result["settings"]["auto-assign"] is False
    assert set(snapshot(tmp_path)) == {"classify_config.yaml", "operator_owners.yaml"}


def test_new_scope_and_previous_explicit_scope_survive_setup(setup, tmp_path):
    """Use broad scope for new projects without migrating an existing team's policy."""
    result = setup.prepare_setup(tmp_path)
    scope = result["settings"]["responsibility"]
    assert scope["handle"] == ["所有非 Roadmap、路线图、规划汇总等纯规划类的 open Issue"]
    assert scope["list-only"] == []
    assert scope["ignore"] == ["Roadmap、路线图、规划汇总等纯规划类 Issue"]
    path = configuration(tmp_path)
    path.write_text(
        "# previous scope\nrepo: team/math\n"
        "responsibility:\n  handle: [A5 kernels]\n"
        "  list-only: [other chips]\n  ignore: []\n",
        encoding="utf-8",
    )
    before = snapshot(tmp_path)
    reopened = setup.prepare_setup(tmp_path, reconfigure=True)
    assert reopened["settings"]["responsibility"] == {
        "handle": ["A5 kernels"], "list-only": ["other chips"], "ignore": []
    }
    assert snapshot(tmp_path) == before


def test_new_installer_templates_still_need_user_choice(setup, tmp_path):
    """Preinstalled templates do not prove that the user has chosen settings."""
    setup.initialize_config(tmp_path)
    before = snapshot(tmp_path)
    result = setup.prepare_setup(tmp_path)
    assert result["status"] == "pending"
    assert result["created"] == {}
    assert snapshot(tmp_path) == before


def test_resolved_repo_in_template_is_not_configuration_consent(setup, tmp_path):
    """Automatic repository discovery must not bypass the configuration question."""
    setup.initialize_config(tmp_path)
    path = configuration(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace('repo: ""', 'repo: "team/math"'), encoding="utf-8")
    assert setup.prepare_setup(tmp_path)["needs_configuration"] is True


def test_unanswered_setup_remains_pending_after_config_edit(setup, tmp_path):
    """An interrupted guide stays pending even when configuration values change."""
    setup.prepare_setup(tmp_path)
    path = configuration(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace("lookback_days: 7", "lookback_days: 14"), encoding="utf-8")
    before = snapshot(tmp_path)
    assert setup.prepare_setup(tmp_path)["status"] == "pending"
    assert setup.prepare_setup(tmp_path)["status"] == "pending"
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("decision", ["configured", "kept"])
def test_explicit_decision_survives_repeated_runs(setup, tmp_path, decision):
    """Remember both completion choices without rewriting existing YAML."""
    setup.prepare_setup(tmp_path)
    before = snapshot(tmp_path)
    assert setup.finish_setup(tmp_path, decision)["status"] == decision
    result = setup.prepare_setup(tmp_path)
    assert result["status"] == decision
    assert result["needs_configuration"] is False
    assert snapshot(tmp_path) == before
    state = tmp_path / setup.SETUP_STATE
    assert state.read_text(encoding="utf-8") == json.dumps({"version": 1, "status": decision}) + "\n"
    assert not list(state.parent.glob(".setup-*"))


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_state_save_cleans_up_and_preserves_previous_state(setup, tmp_path, monkeypatch, failure):
    """Close temporary streams and remove failed writes without changing saved state."""
    setup.prepare_setup(tmp_path)
    state = tmp_path / setup.SETUP_STATE
    before = state.read_bytes()
    files_before = set(state.parent.iterdir())
    streams = []
    original_dump = json.dump

    def dump_state(value, stream):
        streams.append(stream)
        original_dump(value, stream)
        if failure == "write":
            raise OSError("simulated state write failure")

    monkeypatch.setattr(setup.json, "dump", dump_state)
    if failure == "replace":
        monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("simulated state replace failure")))
    with pytest.raises(OSError, match="simulated state"):
        setup.finish_setup(tmp_path, "configured")
    assert state.read_bytes() == before
    assert set(state.parent.iterdir()) == files_before
    assert len(streams) == 1
    assert streams[0].closed


def test_existing_custom_settings_are_not_overwritten_or_forced_to_reconfigure(setup, tmp_path):
    """Reuse customized policies without requiring another setup decision."""
    path = configuration(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        "# team settings\nrepo: team/math\nauto-response: true\nreport_file: reports/custom.txt\n", encoding="utf-8"
    )
    before = path.read_bytes()
    result = setup.prepare_setup(tmp_path)
    assert result["status"] == "existing"
    assert result["settings"]["auto-response"] is True
    assert result["needs_configuration"] is False
    assert path.read_bytes() == before
    assert not Path(result["state_path"]).exists()


def test_completion_uses_edited_values_without_rewriting_other_settings(setup, tmp_path):
    """Accept chosen values while preserving comments and unrelated settings."""
    setup.prepare_setup(tmp_path)
    path = configuration(tmp_path)
    path.write_text(
        "# confirmed choices\nrepo: team/math\n"
        "responsibility: {handle: [shared code], list-only: [], ignore: []}\n"
        "auto-response: true\nauto-assign: false\n"
        "lookback_days: 14  # retain our scan window\n",
        encoding="utf-8",
    )
    before = snapshot(tmp_path)
    result = setup.finish_setup(tmp_path, "configured")
    assert result["settings"]["responsibility"]["handle"] == ["shared code"]
    assert result["settings"]["auto-response"] is True
    assert result["settings"]["auto-assign"] is False
    assert setup.prepare_setup(tmp_path)["needs_configuration"] is False
    assert snapshot(tmp_path) == before


def test_legacy_user_settings_survive_initialization(setup, tmp_path):
    """Reuse legacy values while retaining the original configuration file."""
    legacy = tmp_path / "classify_config.yaml"
    legacy.write_text("repo: team/math\nauto-response: true\n", encoding="utf-8")
    before = legacy.read_bytes()
    result = setup.prepare_setup(tmp_path)
    assert result["status"] == "existing"
    assert result["settings"]["auto-response"] is True
    assert legacy.read_bytes() == before


def test_explicit_reconfigure_preserves_settings_and_reopens_question(setup, tmp_path):
    """Let an explicit request reopen the guide without resetting user settings."""
    setup.prepare_setup(tmp_path)
    setup.finish_setup(tmp_path, "kept")
    path = configuration(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "\n# user note\n", encoding="utf-8")
    before = snapshot(tmp_path)
    result = setup.prepare_setup(tmp_path, reconfigure=True)
    assert result["status"] == "pending"
    assert snapshot(tmp_path) == before
    setup.finish_setup(tmp_path, "configured")
    assert setup.prepare_setup(tmp_path)["status"] == "configured"


def test_missing_policy_restarts_setup_but_preserves_owners(setup, tmp_path):
    """A restored default policy needs review even when owner mappings remain."""
    setup.prepare_setup(tmp_path)
    setup.finish_setup(tmp_path, "configured")
    owners = configuration(tmp_path).with_name("operator_owners.yaml")
    owners.write_text("# owner map\noperators: {Abs: maintainer}\n", encoding="utf-8")
    before = owners.read_bytes()
    configuration(tmp_path).unlink()
    result = setup.prepare_setup(tmp_path)
    assert result["status"] == "pending"
    assert owners.read_bytes() == before


def test_missing_optional_owners_does_not_restart_completed_setup(setup, tmp_path):
    """Restoring the optional owner template keeps the prior setup decision."""
    setup.prepare_setup(tmp_path)
    setup.finish_setup(tmp_path, "kept")
    configuration(tmp_path).with_name("operator_owners.yaml").unlink()
    result = setup.prepare_setup(tmp_path)
    assert result["status"] == "kept"
    assert set(result["created"]) == {"operator_owners"}


@pytest.mark.parametrize("body", ["repo: [", "auto-response: false\nauto-assign: true\n"])
def test_bad_configuration_is_preserved_and_not_marked_complete(setup, tmp_path, body):
    """Reject malformed or conflicting settings without replacing their contents."""
    setup.prepare_setup(tmp_path)
    path = configuration(tmp_path)
    path.write_text(body, encoding="utf-8")
    before = (tmp_path / setup.SETUP_STATE).read_bytes()
    with pytest.raises(ValueError):
        setup.finish_setup(tmp_path, "configured")
    assert path.read_text(encoding="utf-8") == body
    assert (tmp_path / setup.SETUP_STATE).read_bytes() == before


@pytest.mark.parametrize("body", ["{", "[]", '{"version": 1, "status": []}'])
def test_invalid_setup_record_is_preserved(setup, tmp_path, body):
    """Leave an invalid state file intact for diagnosis."""
    setup.prepare_setup(tmp_path)
    path = tmp_path / setup.SETUP_STATE
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        setup.prepare_setup(tmp_path)
    assert path.read_text(encoding="utf-8") == body


def test_completion_requires_prepared_guide(setup, tmp_path):
    """Refuse a completion record when the guide has not been opened."""
    setup.initialize_config(tmp_path)
    with pytest.raises(ValueError, match="prepare or reopen"):
        setup.finish_setup(tmp_path, "kept")
    assert not (tmp_path / setup.SETUP_STATE).exists()


def test_dangling_configuration_symlink_is_not_replaced(setup, tmp_path):
    """Report an unavailable symlink target without overwriting the link."""
    setup.initialize_config(tmp_path)
    path = configuration(tmp_path)
    path.unlink()
    path.symlink_to(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="not a file"):
        setup.prepare_setup(tmp_path)
    assert path.is_symlink()


def test_cli_prepare_keep_and_invalid_combination(setup, tmp_path, capsys):
    """Keep CLI status output and failure exit codes consistent across actions."""
    args = ["--repository-root", str(tmp_path)]
    assert setup.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pending"
    assert setup.main(args + ["--keep"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "kept"
    assert setup.main(args + ["--reconfigure"]) == 0
    capsys.readouterr()
    path = configuration(tmp_path)
    path.write_text("auto-response: false\nauto-assign: true\n", encoding="utf-8")
    assert setup.main(args + ["--complete"]) == CONFIGURATION_ERROR_EXIT
    assert json.loads(capsys.readouterr().out)["status"] == "error"
