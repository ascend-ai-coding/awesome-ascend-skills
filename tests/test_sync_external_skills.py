"""Tests for sync_external_skills module."""

import os
from pathlib import Path

from scripts.sync_external_skills import (
    build_synced_skill_index,
    detect_conflicts,
    load_existing_external_skills,
    prune_removed_source_skills,
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

    assert set(existing.keys()) == {("source-a", "skill-one"), ("source-b", "skill-two")}
    assert existing[("source-a", "skill-one")][1] == "abc123"
    assert existing[("source-b", "skill-two")][1] == "def456"


def test_detect_conflicts_allows_resync_from_same_source() -> None:
    source = ExternalSource(name="source-a", url="https://example.com/a.git")
    skill = Skill(name="skill-one", path=Path("/tmp/skill-one"), source=source, has_skill_md=True)

    synced_index = build_synced_skill_index(
        {
            ("source-a", "skill-one"): (skill, "abc123"),
        }
    )

    assert detect_conflicts(skill, local_skills=set(), synced_skills=synced_index) is None


def test_detect_conflicts_blocks_other_synced_sources() -> None:
    source = ExternalSource(name="source-a", url="https://example.com/a.git")
    other_source = ExternalSource(name="source-b", url="https://example.com/b.git")
    skill = Skill(name="skill-one", path=Path("/tmp/skill-one"), source=source, has_skill_md=True)
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


def test_load_existing_external_skills_keeps_recorded_source_url(tmp_path: Path) -> None:
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

    assert existing[("source-a", "skill-one")][0].source.url == "https://old.example.com/a.git"


def test_prune_removed_source_skills_removes_stale_same_source_entries(tmp_path: Path) -> None:
    source = ExternalSource(name="source-a", url="https://example.com/a.git")
    kept_dir = tmp_path / "external" / "source-a" / "kept-skill"
    stale_dir = tmp_path / "external" / "source-a" / "stale-skill"
    write_skill(kept_dir, "external-source-a-kept-skill", commit_sha="abc123")
    write_skill(stale_dir, "external-source-a-stale-skill", commit_sha="def456")

    existing = load_existing_external_skills([source], external_root=tmp_path / "external")

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
