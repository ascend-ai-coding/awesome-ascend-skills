# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise checkout setup and query dispatch without network or package installation."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import knowledge_query as query
import knowledge_setup as setup


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source repo"
    scripts = root / setup.QUERY_DIR
    scripts.mkdir(parents=True)
    (root / "knowledge").mkdir()
    (root / "knowledge/card.md").write_text("original card\n")
    (root / "governance/schemas").mkdir(parents=True)
    (root / "governance/schemas/registries.yaml").write_text("platforms: {}\n")
    (root / "governance/contracts").mkdir()
    (root / "governance/contracts/__init__.py").touch()
    (root / "requirements.txt").write_text("PyYAML>=6,<7\n")
    (scripts / "knowledge_query.py").write_text(
        "import json, sys\nprint(json.dumps({'argv': sys.argv[1:]}))\n"
    )
    (scripts / "knowledge_index.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "root = Path(sys.argv[sys.argv.index('--knowledge-root') + 1])\n"
        "index = root / 'artifacts/indexes/ready'\n"
        "if 'build' in sys.argv:\n"
        "    index.parent.mkdir(parents=True, exist_ok=True)\n"
        "    index.write_text('built')\n"
        "else:\n"
        "    raise SystemExit(0 if index.is_file() else 2)\n"
    )
    (root / ".gitignore").write_text("artifacts/\n.venv/\n")
    for args in (
        ["init"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "fixture",
        ],
        ["branch", "-M", "master"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    monkeypatch.setattr(setup, "REPOSITORY", str(root))
    monkeypatch.setattr(setup, "prepare_python", lambda root: Path(sys.executable))
    return root


def test_missing_checkout_is_cloned_indexed_and_reused(source, tmp_path, monkeypatch):
    target = tmp_path / "installed knowledge"
    root, _ = setup.prepare(str(target))
    assert root == target
    assert (root / "artifacts/indexes/ready").read_text() == "built"
    card = root / "knowledge/card.md"
    card.write_text("local edits\n")
    monkeypatch.setattr(setup, "REPOSITORY", "/nonexistent/offline")
    setup.prepare(str(target))
    assert card.read_text() == "local edits\n"


def test_invalid_existing_directory_is_preserved(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    marker = root / "user-data"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="incomplete"):
        setup.prepare(str(root))
    assert marker.read_text() == "keep"


def test_clone_failure_does_not_leave_partial_target(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "REPOSITORY", str(tmp_path / "missing-source"))
    target = tmp_path / "destination"
    with pytest.raises(subprocess.CalledProcessError):
        setup.prepare(str(target))
    assert not target.exists()
    assert not list(tmp_path.glob(".knowledge-clone-*"))


def test_index_failure_is_not_reported_as_ready(source, monkeypatch):
    (source / setup.QUERY_DIR / "knowledge_index.py").write_text(
        "raise SystemExit(2)\n"
    )
    with pytest.raises(subprocess.CalledProcessError):
        setup.prepare(str(source))


def test_project_configuration_is_data_and_relative_to_config(tmp_path, monkeypatch):
    monkeypatch.delenv(setup.ROOT_KEY, raising=False)
    config = tmp_path / ".cannbot/knowledge.env"
    config.parent.mkdir()
    config.write_text("export CANNBOT_KNOWLEDGE_ROOT='../kb with spaces'\n")
    project = tmp_path / "nested"
    project.mkdir()
    monkeypatch.chdir(project)
    assert setup.configured_root().resolve() == tmp_path / "kb with spaces"
    config.write_text("export CANNBOT_KNOWLEDGE_ROOT='$(touch unexpected)'\n")
    assert setup.configured_root().name == "$(touch unexpected)"
    assert not (project / "unexpected").exists()
    monkeypatch.setenv(setup.ROOT_KEY, str(tmp_path / "environment"))
    assert setup.configured_root() == tmp_path / "environment"
    assert setup.configured_root(str(tmp_path / "explicit")) == tmp_path / "explicit"


@pytest.mark.parametrize(
    "args",
    [
        ["--knowledge-root"],
        ["--knowledge-root="],
        ["--knowledge-root", "a", "--knowledge-root=b"],
    ],
)
def test_invalid_root_arguments_fail_before_install(args):
    with pytest.raises(ValueError):
        setup.split_root(args)


def test_query_forwards_arguments_and_preserves_json(source, capfd):
    assert (
        query.main(
            [
                "search",
                "--query",
                "a phrase; $(literal)",
                "--knowledge-root",
                str(source),
                "--platform",
                "a2",
            ]
        )
        == 0
    )
    payload = json.loads(capfd.readouterr().out)
    assert payload["argv"] == [
        "--knowledge-root",
        str(source),
        "search",
        "--query",
        "a phrase; $(literal)",
        "--platform",
        "a2",
    ]


def test_query_exit_code_is_preserved(source):
    (source / setup.QUERY_DIR / "knowledge_query.py").write_text(
        "raise SystemExit(2)\n"
    )
    assert query.main(["--knowledge-root", str(source), "search"]) == 2


def test_help_and_invalid_commands_do_not_install(monkeypatch):
    def unexpected(*args):
        pytest.fail("help or invalid command attempted installation")

    monkeypatch.setattr(query, "prepare", unexpected)
    assert query.main(["--help"]) == 0
    assert query.main(["preflight"]) == 2


def test_missing_dependencies_are_installed_only_in_venv(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append([str(value) for value in argv])
        return subprocess.CompletedProcess(argv, 1 if len(calls) == 1 else 0)

    monkeypatch.setattr(setup, "run", fake_run)
    python = setup.prepare_python(tmp_path)
    assert python == tmp_path / ".venv/bin/python"
    assert calls[1] == [sys.executable, "-m", "venv", str(tmp_path / ".venv")]
    assert calls[2][:4] == [str(python), "-m", "pip", "install"]


def test_ingest_setup_uses_shared_installer_from_copied_skills(tmp_path):
    import shutil

    skill_root = Path(__file__).resolve().parents[2]
    installed = tmp_path / "skills"
    shutil.copytree(skill_root / "knowledge-ingest", installed / "knowledge-ingest")
    shared = installed / "knowledge-query/scripts"
    shared.mkdir(parents=True)
    (shared / "knowledge_setup.py").write_text(
        "import json, sys\nprint(json.dumps({'args': sys.argv[1:]}))\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(installed / "knowledge-ingest/scripts/setup_knowledge.py"),
            "--knowledge-root",
            str(tmp_path / "kb"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["args"] == [
        "--knowledge-root",
        str(tmp_path / "kb"),
    ]
    (shared / "knowledge_setup.py").unlink()
    result = subprocess.run(
        [
            sys.executable,
            str(installed / "knowledge-ingest/scripts/setup_knowledge.py"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Install knowledge-query" in result.stderr


def test_empty_setup_root_does_not_fall_back_to_default():
    with pytest.raises(ValueError, match="non-empty"):
        setup.configured_root("")
