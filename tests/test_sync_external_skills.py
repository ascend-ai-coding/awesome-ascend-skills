"""Tests for sync_external_skills module."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

from scripts import sync_external_skills as sync_module
from scripts.sync_external_skills import (
    build_synced_skill_index,
    copy_skill,
    detect_conflicts,
    load_existing_external_skills,
    parse_skill_md,
    prune_removed_source_skills,
    update_marketplace,
)
from scripts.sync_types import ExternalSource, Skill


def write_skill(skill_dir: Path, name: str, commit_sha: str = "abc123") -> None:
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                "description: Test skill description long enough",
                f"synced-commit: {commit_sha}",
                "---",
                "",
                "# Test Skill",
            ]
        ),
        encoding="utf-8",
    )


def write_raw_skill(skill_dir: Path, content: str) -> None:
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def test_load_existing_external_skills_reads_current_disk_state(tmp_path: Path) -> None:
    external_root = tmp_path / "external"
    write_skill(external_root / "source-a" / "skill-one", "external-source-a-skill-one")
    write_skill(
        external_root / "source-b" / "skill-two",
        "external-source-b-skill-two",
        commit_sha="def456",
    )
    write_skill(
        external_root / "ignored-source" / "skill-three",
        "external-ignored-source-skill-three",
        commit_sha="zzz999",
    )

    sources = [
        ExternalSource(name="source-a", url="https://example.com/a.git"),
        ExternalSource(name="source-b", url="https://example.com/b.git"),
    ]

    existing = load_existing_external_skills(sources, external_root=external_root)

    assert set(existing.keys()) == {
        ("source-a", "skill-one"),
        ("source-b", "skill-two"),
    }
    assert existing[("source-a", "skill-one")][1] == "abc123"
    assert existing[("source-b", "skill-two")][1] == "def456"


def test_detect_conflicts_allows_resync_from_same_source() -> None:
    source = ExternalSource(name="source-a", url="https://example.com/a.git")
    skill = Skill(
        name="skill-one", path=Path("/tmp/skill-one"), source=source, has_skill_md=True
    )

    synced_index = build_synced_skill_index(
        {
            ("source-a", "skill-one"): (skill, "abc123"),
        }
    )

    assert (
        detect_conflicts(skill, local_skills=set(), synced_skills=synced_index) is None
    )


def test_detect_conflicts_blocks_other_synced_sources() -> None:
    source = ExternalSource(name="source-a", url="https://example.com/a.git")
    other_source = ExternalSource(name="source-b", url="https://example.com/b.git")
    skill = Skill(
        name="skill-one", path=Path("/tmp/skill-one"), source=source, has_skill_md=True
    )
    other_skill = Skill(
        name="skill-one",
        path=Path("/tmp/other-skill-one"),
        source=other_source,
        has_skill_md=True,
    )

    synced_index = build_synced_skill_index(
        {
            ("source-b", "skill-one"): (other_skill, "def456"),
        }
    )

    conflict = detect_conflicts(skill, local_skills=set(), synced_skills=synced_index)

    assert conflict is not None
    assert conflict.skill_name == "skill-one"
    assert conflict.external_source == "synced (source-b)"


def test_load_existing_external_skills_keeps_recorded_source_url(
    tmp_path: Path,
) -> None:
    external_root = tmp_path / "external"
    skill_dir = external_root / "source-a" / "skill-one"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: external-source-a-skill-one",
                "description: Test skill description long enough",
                "synced-commit: abc123",
                "synced-from: https://old.example.com/a.git",
                "---",
                "",
                "# Test Skill",
            ]
        ),
        encoding="utf-8",
    )

    existing = load_existing_external_skills(
        [ExternalSource(name="source-a", url="https://new.example.com/a.git")],
        external_root=external_root,
    )

    assert (
        existing[("source-a", "skill-one")][0].source.url
        == "https://old.example.com/a.git"
    )


def test_prune_removed_source_skills_removes_stale_same_source_entries(
    tmp_path: Path,
) -> None:
    source = ExternalSource(name="source-a", url="https://example.com/a.git")
    kept_dir = tmp_path / "external" / "source-a" / "kept-skill"
    stale_dir = tmp_path / "external" / "source-a" / "stale-skill"
    write_skill(kept_dir, "external-source-a-kept-skill", commit_sha="abc123")
    write_skill(stale_dir, "external-source-a-stale-skill", commit_sha="def456")

    existing = load_existing_external_skills(
        [source], external_root=tmp_path / "external"
    )

    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        prune_removed_source_skills(existing, source, {"kept-skill"})
    finally:
        os.chdir(original_cwd)

    assert ("source-a", "kept-skill") in existing
    assert ("source-a", "stale-skill") not in existing
    assert kept_dir.exists()
    assert not stale_dir.exists()


def test_parse_skill_md_falls_back_for_colon_rich_description(tmp_path: Path) -> None:
    skill_dir = tmp_path / "ascendc-operator-code-gen"
    write_raw_skill(
        skill_dir,
        "\n".join(
            [
                "---",
                "name: ascendc-operator-code-gen",
                "description: 根据设计文档生成 AscendC 算子完整代码实现。TRIGGER when: 设计文档已完成，需要生成 op_host/op_kernel 代码。",
                "synced-commit: abc123",
                "---",
                "",
                "# Test Skill",
            ]
        ),
    )

    parsed = parse_skill_md(skill_dir)

    assert parsed["name"] == "ascendc-operator-code-gen"
    assert "TRIGGER when:" in parsed["description"]
    assert parsed["synced-commit"] == "abc123"


def test_load_existing_external_skills_tolerates_malformed_frontmatter(
    tmp_path: Path,
) -> None:
    external_root = tmp_path / "external"
    write_skill(
        external_root / "source-a" / "valid-skill",
        "external-source-a-valid-skill",
    )
    write_raw_skill(
        external_root / "source-a" / "broken-skill",
        "\n".join(
            [
                "---",
                "name: external-source-a-broken-skill",
                "description: invalid: colon-rich description",
                "synced-commit: xyz789",
                "---",
                "",
                "# Broken Skill",
            ]
        ),
    )

    existing = load_existing_external_skills(
        [ExternalSource(name="source-a", url="https://example.com/a.git")],
        external_root=external_root,
    )

    assert set(existing.keys()) == {
        ("source-a", "valid-skill"),
        ("source-a", "broken-skill"),
    }
    assert existing[("source-a", "broken-skill")][1] == "xyz789"


def test_parse_skill_md_does_not_salvage_unsupported_malformed_yaml(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "broken-skill"
    write_raw_skill(
        skill_dir,
        "\n".join(
            [
                "---",
                "name: broken-skill",
                "description: broken skill description long enough",
                "keywords: [broken",
                "---",
                "",
                "# Broken Skill",
            ]
        ),
    )

    parsed = parse_skill_md(skill_dir)

    assert parsed == {}


def test_copy_skill_rolls_back_failed_validation(tmp_path: Path, monkeypatch) -> None:
    source = ExternalSource(name="source-a", url="https://example.com/a.git")
    skill_dir = tmp_path / "repo" / "broken-skill"
    write_skill(skill_dir, "broken-skill")
    skill = Skill(name="broken-skill", path=skill_dir, source=source, has_skill_md=True)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="  ❌ ERROR: Missing 'description' field in frontmatter\n",
            stderr="",
        )

    monkeypatch.setattr(sync_module.subprocess, "run", fake_run)

    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        success, reason = copy_skill(skill, "abc123")
    finally:
        os.chdir(original_cwd)

    assert success is False
    assert reason == "Missing 'description' field in frontmatter"
    assert not (tmp_path / "external" / "source-a" / "broken-skill").exists()


def test_copy_skill_restores_existing_target_on_failed_validation(
    tmp_path: Path, monkeypatch
) -> None:
    source = ExternalSource(name="source-a", url="https://example.com/a.git")
    existing_dir = tmp_path / "external" / "source-a" / "broken-skill"
    write_skill(existing_dir, "external-source-a-broken-skill", commit_sha="old123")

    updated_dir = tmp_path / "repo" / "broken-skill"
    write_raw_skill(
        updated_dir,
        "\n".join(
            [
                "---",
                "name: broken-skill",
                "description: invalid: colon-rich description",
                "---",
                "",
                "# Updated Skill",
            ]
        ),
    )
    skill = Skill(
        name="broken-skill", path=updated_dir, source=source, has_skill_md=True
    )

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="  ❌ ERROR: Missing 'description' field in frontmatter\n",
            stderr="",
        )

    monkeypatch.setattr(sync_module.subprocess, "run", fake_run)

    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        success, reason = copy_skill(skill, "abc123")
        restored_content = (existing_dir / "SKILL.md").read_text(encoding="utf-8")
    finally:
        os.chdir(original_cwd)

    assert success is False
    assert reason == "Missing 'description' field in frontmatter"
    assert "synced-commit: old123" in restored_content


def test_copy_skill_keeps_backup_when_restore_fails(
    tmp_path: Path, monkeypatch
) -> None:
    source = ExternalSource(name="source-a", url="https://example.com/a.git")
    existing_dir = tmp_path / "external" / "source-a" / "broken-skill"
    write_skill(existing_dir, "external-source-a-broken-skill", commit_sha="old123")

    updated_dir = tmp_path / "repo" / "broken-skill"
    write_skill(updated_dir, "broken-skill")
    skill = Skill(
        name="broken-skill", path=updated_dir, source=source, has_skill_md=True
    )

    backup_dir = tmp_path / "backup-dir"

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="  ❌ ERROR: Missing 'description' field in frontmatter\n",
            stderr="",
        )

    real_move = sync_module.shutil.move

    def fake_move(src, dst, *args, **kwargs):
        src_path = Path(src)
        if src_path == backup_dir / "broken-skill":
            raise OSError("restore failed")
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(sync_module.subprocess, "run", fake_run)
    monkeypatch.setattr(sync_module.tempfile, "mkdtemp", lambda prefix: str(backup_dir))
    monkeypatch.setattr(sync_module.shutil, "move", fake_move)

    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        success, reason = copy_skill(skill, "abc123")
    finally:
        os.chdir(original_cwd)

    assert success is False
    assert "Validation failed and restore did not complete" in reason
    assert (backup_dir / "broken-skill" / "SKILL.md").exists()


def test_copy_skill_syncs_malformed_frontmatter_when_validation_passes(
    tmp_path: Path, monkeypatch
) -> None:
    source = ExternalSource(name="source-a", url="https://example.com/a.git")
    skill_dir = tmp_path / "repo" / "broken-skill"
    write_raw_skill(
        skill_dir,
        "\n".join(
            [
                "---",
                "name: broken-skill",
                "description: invalid: colon-rich description",
                "---",
                "",
                "# Broken Skill",
                "Enough body text to satisfy downstream validation checks.",
            ]
        ),
    )
    skill = Skill(name="broken-skill", path=skill_dir, source=source, has_skill_md=True)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sync_module.subprocess, "run", fake_run)

    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        success, reason = copy_skill(skill, "abc123")
        synced_content = (
            tmp_path / "external" / "source-a" / "broken-skill" / "SKILL.md"
        ).read_text(encoding="utf-8")
    finally:
        os.chdir(original_cwd)

    assert success is True
    assert reason == ""
    assert "name: external-source-a-broken-skill" in synced_content
    assert "description: 'invalid: colon-rich description'" in synced_content


def test_update_marketplace_preserves_non_external_plugin_order(tmp_path: Path) -> None:
    marketplace_path = tmp_path / "marketplace.json"
    marketplace_path.write_text(
        json.dumps(
            {
                "plugins": [
                    {"name": "local-before", "source": "./local-before"},
                    {"name": "external-old", "external": True, "skills": []},
                    {"name": "local-after", "source": "./local-after"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    source = ExternalSource(name="source-a", url="https://example.com/a.git")
    skill_dir = tmp_path / "external" / "source-a" / "skill-one"
    write_skill(skill_dir, "external-source-a-skill-one")
    synced_skill = Skill(
        name="skill-one", path=skill_dir, source=source, has_skill_md=True
    )

    update_marketplace(
        [(synced_skill, "abc123")], marketplace_path=str(marketplace_path)
    )

    plugins = json.loads(marketplace_path.read_text(encoding="utf-8"))["plugins"]
    assert [plugin["name"] for plugin in plugins] == [
        "local-before",
        "external-source-a-skills",
        "local-after",
    ]
