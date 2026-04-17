#!/usr/bin/env python3
"""Sync external skills from configured repositories."""

import shutil
import subprocess
import sys
import tempfile
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from sync_types import ExternalSource, Skill, ConflictInfo, SyncResult


def load_config(config_path: str) -> List[ExternalSource]:
    """Load external sources from YAML config file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        List of ExternalSource configurations.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file contains invalid YAML.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_data = yaml.safe_load(config_file.read_text(encoding="utf-8"))

    sources = []
    for source_data in config_data.get("sources", []):
        source = ExternalSource(
            name=source_data["name"],
            url=source_data["url"],
            branch=source_data.get("branch", "main"),
            enabled=source_data.get("enabled", True),
            skills_path=source_data.get("skills_path", ""),
        )
        sources.append(source)

    return sources


def detect_config_changes(old_config: str, new_config: str) -> List[ExternalSource]:
    """Detect new or changed sources between two config YAML strings.

    Args:
        old_config: Old YAML config string (from git diff).
        new_config: New YAML config string (current config).

    Returns:
        List of ExternalSource objects for new or changed sources.
        Returns empty list if no changes detected.
    """
    try:
        old_data = yaml.safe_load(old_config) or {}
        new_data = yaml.safe_load(new_config) or {}
    except yaml.YAMLError:
        # If parsing fails, return empty list (no changes detected)
        return []

    old_sources = old_data.get("sources", [])
    new_sources = new_data.get("sources", [])

    # Create a mapping of source names to their data
    old_sources_dict = {s["name"]: s for s in old_sources}
    new_sources_dict = {s["name"]: s for s in new_sources}

    changes = []

    # Check for new sources
    for name, new_source_data in new_sources_dict.items():
        if name not in old_sources_dict:
            changes.append(
                ExternalSource(
                    name=name,
                    url=new_source_data["url"],
                    branch=new_source_data.get("branch", "main"),
                    enabled=new_source_data.get("enabled", True),
                    skills_path=new_source_data.get("skills_path", ""),
                )
            )

    # Check for changed sources (by name and url, which uniquely identifies a source)
    for name, new_source_data in new_sources_dict.items():
        old_source_data = old_sources_dict.get(name)
        if old_source_data and old_source_data["url"] != new_source_data["url"]:
            # Source with same name but different URL (changed source)
            changes.append(
                ExternalSource(
                    name=name,
                    url=new_source_data["url"],
                    branch=new_source_data.get("branch", "main"),
                    enabled=new_source_data.get("enabled", True),
                )
            )

    return changes


def should_sync_on_pr() -> bool:
    """Check if this is a PR context and config file was modified.

    Returns:
        True if running in PR context and .github/external-sources.yml was modified.
        False otherwise.
    """
    import os

    # Check if running in GitHub Actions PR context
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    if event_name != "pull_request":
        return False

    # Check if config file was modified in the PR
    changed_files = os.environ.get("GITHUB_CHANGED_FILES", "")
    config_file = ".github/external-sources.yml"
    return config_file in changed_files


def get_commit_sha(repo_path: Path) -> str:
    """Get the most recent commit SHA from a git repository.

    Args:
        repo_path: Path to the git repository.

    Returns:
        The commit SHA string (40 characters).

    Raises:
        subprocess.CalledProcessError: If git log fails.
    """
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def clone_external_repo(source: ExternalSource) -> Tuple[Path, str]:
    """Clone external repo to temp directory with --depth 1.

    Args:
        source: ExternalSource configuration with url, branch, etc.

    Returns:
        Tuple of (Path to the cloned temporary directory, commit SHA).

    Raises:
        subprocess.CalledProcessError: If git clone fails.
    """
    temp_dir = tempfile.mkdtemp(prefix=f"sync-{source.name}-")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "-b",
            source.branch,
            source.url,
            temp_dir,
        ],
        check=True,
        capture_output=True,
    )
    repo_path = Path(temp_dir)
    commit_sha = get_commit_sha(repo_path)
    return repo_path, commit_sha


def parse_skill_md(skill_path: Path) -> Dict:
    """Parse SKILL.md frontmatter and return as dict.

    Args:
        skill_path: Path to the skill directory.

    Returns:
        Dictionary containing parsed YAML frontmatter, or empty dict
        if no frontmatter found.
    """
    skill_md = skill_path / "SKILL.md"
    content = skill_md.read_text(encoding="utf-8")
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return yaml.safe_load(parts[1]) or {}
    return {}


def find_skills(repo_path: Path, source: ExternalSource) -> List[Skill]:
    """Find all skills (dirs with SKILL.md) in repo.

    Args:
        repo_path: Path to the cloned repository root.
        source: ExternalSource this repository comes from.

    Returns:
        List of Skill objects for directories containing SKILL.md.
    """
    skills = []
    search_path = repo_path
    if source.skills_path:
        search_path = repo_path / source.skills_path
    if not search_path.exists():
        return skills
    for item in search_path.iterdir():
        if item.is_dir() and (item / "SKILL.md").exists():
            skills.append(
                Skill(
                    name=item.name,
                    path=item,
                    source=source,
                    has_skill_md=True,
                )
            )
    return skills


def get_local_skills() -> Set[str]:
    """Get skill names in repo root (excluding external/)."""
    skills = set()
    for item in Path(".").iterdir():
        if item.is_dir() and (item / "SKILL.md").exists() and item.name != "external":
            skills.add(item.name)
    return skills


def get_synced_skills() -> Set[str]:
    """Get skill names already synced in external/."""
    skills = set()
    external_dir = Path("external")
    if external_dir.exists():
        for source_dir in external_dir.iterdir():
            if source_dir.is_dir():
                for skill_dir in source_dir.iterdir():
                    if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                        skills.add(skill_dir.name)
    return skills


def load_existing_external_skills(
    sources: Iterable[ExternalSource], external_root: Union[Path, str] = "external"
) -> Dict[Tuple[str, str], Tuple[Skill, str]]:
    external_dir = Path(external_root)
    existing_skills: Dict[Tuple[str, str], Tuple[Skill, str]] = {}
    source_map = {source.name: source for source in sources if source.enabled}

    if not external_dir.exists():
        return existing_skills

    for source_name, source in source_map.items():
        source_dir = external_dir / source_name
        if not source_dir.exists() or not source_dir.is_dir():
            continue

        for skill_dir in source_dir.iterdir():
            if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
                continue

            parsed = parse_skill_md(skill_dir)
            commit_sha = str(parsed.get("synced-commit", ""))
            source_url = str(parsed.get("synced-from", source.url))
            existing_skills[(source_name, skill_dir.name)] = (
                Skill(
                    name=skill_dir.name,
                    path=skill_dir,
                    source=ExternalSource(
                        name=source.name,
                        url=source_url,
                        branch=source.branch,
                        enabled=source.enabled,
                        skills_path=source.skills_path,
                    ),
                    has_skill_md=True,
                ),
                commit_sha,
            )

    return existing_skills


def build_synced_skill_index(
    synced_skills: Dict[Tuple[str, str], Tuple[Skill, str]]
) -> Dict[str, Set[str]]:
    index: Dict[str, Set[str]] = {}
    for source_name, skill_name in synced_skills.keys():
        index.setdefault(skill_name, set()).add(source_name)
    return index


def prune_removed_source_skills(
    existing_skills: Dict[Tuple[str, str], Tuple[Skill, str]],
    source: ExternalSource,
    current_skill_names: Set[str],
) -> None:
    source_root = Path("external") / source.name
    removed_keys = [
        key
        for key in existing_skills
        if key[0] == source.name and key[1] not in current_skill_names
    ]

    for key in removed_keys:
        skill_path = source_root / key[1]
        if skill_path.exists():
            shutil.rmtree(skill_path)
        del existing_skills[key]

    if source_root.exists() and not any(source_root.iterdir()):
        source_root.rmdir()


def detect_conflicts(
    skill: Skill, local_skills: Set[str], synced_skills: Dict[str, Set[str]]
) -> Optional[ConflictInfo]:
    """Check if skill conflicts with local or synced skills."""
    if skill.name in local_skills:
        return ConflictInfo(
            skill_name=skill.name, local_path=f"./{skill.name}", external_source="local"
        )
    conflict_sources = synced_skills.get(skill.name, set()) - {skill.source.name}
    if conflict_sources:
        return ConflictInfo(
            skill_name=skill.name,
            local_path=f"./external/*/{skill.name}",
            external_source=f"synced ({', '.join(sorted(conflict_sources))})",
        )
    return None


def inject_attribution(skill: Skill, commit_sha: str) -> str:
    """Inject source attribution into SKILL.md frontmatter.

    Args:
        skill: The Skill object containing source information
        commit_sha: The Git commit SHA to attribute

    Returns:
        Modified content string with injected attribution fields.
        Does NOT write to file.

    The function:
    - Preserves existing frontmatter fields
    - Adds attribution fields only if they don't exist
    - Does NOT modify the body content
    """
    skill_md_path = skill.path / "SKILL.md"
    content = skill_md_path.read_text(encoding="utf-8")

    # Parse existing frontmatter
    fm = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm = yaml.safe_load(parts[1]) or {}
            body = parts[2]

    # Rename skill to follow nested naming convention: external-{source}-{name}
    original_name = fm.get("name", skill.name)
    new_name = f"external-{skill.source.name}-{skill.name}"
    fm["name"] = new_name
    fm["original-name"] = original_name

    # Inject attribution fields (don't overwrite existing)
    if "synced-from" not in fm:
        fm["synced-from"] = skill.source.url
    if "synced-date" not in fm:
        fm["synced-date"] = datetime.now().strftime("%Y-%m-%d")
    if "synced-commit" not in fm:
        fm["synced-commit"] = commit_sha
    if "license" not in fm:
        fm["license"] = "UNKNOWN"

    # Reassemble
    new_frontmatter = yaml.dump(fm, sort_keys=False, allow_unicode=True)
    return f"---\n{new_frontmatter}---\n{body}"


def copy_skill(skill: Skill, commit_sha: str) -> bool:
    """Copy skill to external/ directory, inject attribution, and validate.

    Args:
        skill: The Skill object to copy.
        commit_sha: The Git commit SHA for attribution.

    Returns:
        True on success, False if validation fails.
    """
    target = Path("external") / skill.source.name / skill.name

    if target.exists():
        shutil.rmtree(target)

    shutil.copytree(skill.path, target, ignore=shutil.ignore_patterns(".git"))

    copied_skill = Skill(
        name=skill.name, path=target, source=skill.source, has_skill_md=True
    )
    attributed_content = inject_attribution(copied_skill, commit_sha)
    (target / "SKILL.md").write_text(attributed_content, encoding="utf-8")

    result = subprocess.run(
        ["python3", "scripts/validate_skills.py"], capture_output=True, text=True
    )

    return result.returncode == 0


def generate_report(
    results: SyncResult, source: ExternalSource, commit_sha: str
) -> str:
    """Generate markdown sync report.

    Args:
        results: SyncResult containing synced, skipped, and errors.
        source: ExternalSource information.
        commit_sha: Git commit SHA for this sync.

    Returns:
        Markdown formatted sync report.
    """
    report = f"""## 同步报告

**来源**: {source.url}
**提交**: {commit_sha}
**时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

"""
    synced = results.synced if isinstance(results.synced, list) else []
    if synced:
        report += f"### ✅ 同步成功 ({len(synced)})\n"
        for name in synced:
            report += f"- {name}\n"
        report += "\n"

    if results.skipped:
        report += f"### ⏭️ 跳过 ({len(results.skipped)})\n"
        for name, reason in results.skipped:
            report += f"- {name}: {reason}\n"
        report += "\n"

    if results.errors:
        report += f"### ❌ 错误 ({len(results.errors)})\n"
        for name, error in results.errors:
            report += f"- {name}: {error}\n"
        report += "\n"

    return report


def create_sync_pr(results: SyncResult, source: ExternalSource, commit_sha: str) -> str:
    """Create PR with sync report.

    Args:
        results: SyncResult containing synced, skipped, and errors.
        source: ExternalSource information.
        commit_sha: Git commit SHA for this sync.

    Returns:
        PR URL if successful, empty string if failed.
    """
    title = f"sync(external): update skills from {source.name}"
    body = generate_report(results, source, commit_sha)

    result = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--label",
            "external-sync",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        return result.stdout.strip()
    return ""


def sync_all_sources(config_path: str = ".github/external-sources.yml") -> Dict:
    """Sync all external sources and return summary.

    This function orchestrates the entire sync process:
    1. Load external sources from config
    2. For each enabled source:
       - Clone the repository
       - Find all skills in the repository
       - Check for conflicts with local or synced skills
       - Copy skills that don't conflict
       - Clean up the cloned repository
    3. Update marketplace.json and README.md with synced skills
    4. Return summary statistics

    Args:
        config_path: Path to YAML configuration file (default: .github/external-sources.yml)

    Returns:
        Dictionary with summary statistics:
            - synced: Number of successfully synced skills
            - skipped: Number of skipped skills (due to conflicts)
            - errors: Number of errors encountered
    """
    sources = load_config(config_path)
    existing_external_skills = load_existing_external_skills(sources)
    local_skills = get_local_skills()

    all_synced_skills = []  # List of (Skill, commit_sha) tuples
    all_skipped = []
    all_errors = []

    for source in sources:
        if not source.enabled:
            print(f"Skipping disabled source: {source.name}")
            continue

        print(f"\nProcessing source: {source.name} ({source.url})")

        try:
            print(f"  Cloning {source.url} (branch: {source.branch})...")
            repo_path, commit_sha = clone_external_repo(source)
            print(f"  ✓ Cloned to {repo_path} (commit: {commit_sha[:7]})")

            skills = find_skills(repo_path, source)
            print(f"  Found {len(skills)} skills")
            current_skill_names = {skill.name for skill in skills}
            prune_removed_source_skills(existing_external_skills, source, current_skill_names)

            synced_skills = build_synced_skill_index(existing_external_skills)
            print(f"  Local skills: {len(local_skills)}")
            print(f"  Already synced: {len(existing_external_skills)}")

            for skill in skills:
                conflict = detect_conflicts(skill, local_skills, synced_skills)
                if conflict:
                    print(
                        f"  ⏭️  Skipping {skill.name}: conflict with {conflict.external_source}"
                    )
                    all_skipped.append(
                        (skill.name, f"Conflict: {conflict.external_source}")
                    )
                    continue

                try:
                    print(f"  Syncing {skill.name}...")
                    success = copy_skill(skill, commit_sha)
                    if success:
                        print(f"  ✓ Synced {skill.name}")
                        synced_skill = Skill(
                            name=skill.name,
                            path=Path("external") / skill.source.name / skill.name,
                            source=skill.source,
                            has_skill_md=True,
                        )
                        all_synced_skills.append((synced_skill, commit_sha))
                        existing_external_skills[(skill.source.name, skill.name)] = (
                            synced_skill,
                            commit_sha,
                        )
                    else:
                        print(f"  ❌ Validation failed for {skill.name}")
                        all_errors.append((skill.name, "Validation failed"))
                except Exception as e:
                    print(f"  ❌ Error syncing {skill.name}: {e}")
                    all_errors.append((skill.name, str(e)))

            shutil.rmtree(repo_path)
            print(f"  ✓ Cleaned up {repo_path}")

        except Exception as e:
            print(f"  ❌ Error processing source {source.name}: {e}")
            all_errors.append((source.name, str(e)))

    # Update marketplace.json and README.md with synced skills
    merged_synced_skills = sorted(
        existing_external_skills.values(),
        key=lambda item: (item[0].source.name, item[0].name),
    )

    print("\nUpdating marketplace.json...")
    update_marketplace(merged_synced_skills)
    print("\nUpdating README.md...")
    update_readme(merged_synced_skills)

    # Return summary statistics
    results = {
        "synced": len(all_synced_skills),
        "skipped": len(all_skipped),
        "errors": len(all_errors),
    }

    print("\n" + "=" * 60)
    print("SYNC SUMMARY")
    print("=" * 60)
    print(f"  Synced: {results['synced']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Errors: {results['errors']}")
    print(f"  Total: {len(all_synced_skills) + len(all_skipped) + len(all_errors)}")
    print("=" * 60)

    return results


def update_readme(
    synced_skills: List[Tuple[Skill, str]], readme_path: str = "README.md"
) -> None:
    """Update README.md with external skills table.

    Args:
        synced_skills: List of tuples (Skill, commit_sha) that were successfully synced
        readme_path: Path to README.md file (default: "README.md")
    """
    import re

    readme_file = Path(readme_path)
    content = readme_file.read_text(encoding="utf-8")

    def get_description(skill_path: Path) -> str:
        """Get description from SKILL.md frontmatter."""
        parsed = parse_skill_md(skill_path)
        return parsed.get("description", "No description available")

    # Build external skills table
    table_lines = ["## 外部 Skills (External Skills)", ""]
    table_lines.append("> 以下 skills 从外部仓库自动同步，请勿手动修改。")
    table_lines.append("")  # Empty line required for markdown table to render correctly
    table_lines.append("| Skill | 来源 | 描述 |")
    table_lines.append("|-------|------|------|")

    for skill, commit_sha in synced_skills:
        skill_path = skill.path
        description = get_description(skill_path)

        # Format source link
        source_name = skill.source.name
        source_url = skill.source.url
        source_link = f"[{source_name}]({source_url})"

        # Format skill link
        skill_link = f"[{skill.name}](external/{source_name}/{skill.name}/SKILL.md)"

        # Clean description for markdown table (escape pipe, remove newlines, truncate)
        clean_description = description.replace("|", "\\|").replace("\n", " ").strip()
        if len(clean_description) > 100:
            clean_description = clean_description[:97] + "..."

        # Add row to table
        table_lines.append(f"| {skill_link} | {source_link} | {clean_description} |")

    table_lines.append("")
    table_lines.append("---")

    # Check if table section already exists
    section_start = content.find("## 外部 Skills (External Skills)")
    section_end = content.find("\n---", section_start) if section_start != -1 else -1

    if section_start != -1 and section_end != -1:
        # Replace existing table section
        content = (
            content[:section_start]
            + "\n".join(table_lines)
            + content[section_end + 4 :]
        )
    else:
        # Insert after "## Skill 列表" (before "## Skill 工作原理")
        insert_marker = "\n---\n\n## Skill 工作原理"
        if insert_marker in content:
            content = content.replace(
                insert_marker, "\n" + "\n".join(table_lines) + "\n\n## Skill 工作原理"
            )
        else:
            # Fallback: insert before "## 外部 Skills 同步" section
            insert_marker = "\n---\n\n## 外部 Skills 同步"
            if insert_marker in content:
                content = content.replace(
                    insert_marker,
                    "\n" + "\n".join(table_lines) + "\n\n## 外部 Skills 同步",
                )
            else:
                content = content.rstrip() + "\n\n" + "\n".join(table_lines) + "\n"

    # Write back to file
    readme_file.write_text(content, encoding="utf-8")


def update_marketplace(
    synced_skills: List[Tuple[Skill, str]],
    marketplace_path: str = ".claude-plugin/marketplace.json",
) -> None:
    """Update marketplace.json with external skills entries grouped by source.

    Args:
        synced_skills: List of tuples (Skill, commit_sha) that were successfully synced
        marketplace_path: Path to marketplace.json file (default: .claude-plugin/marketplace.json)

    Groups external skills by source and creates one entry per source with skills array.
    """
    import json
    from collections import defaultdict

    marketplace_file = Path(marketplace_path)

    if marketplace_file.exists():
        with marketplace_file.open("r", encoding="utf-8") as f:
            marketplace = json.load(f)
    else:
        marketplace = {
            "$schema": "https://anthropic.com/claude-code/marketplace.schema.json",
            "name": "awesome-ascend-skills",
            "version": "1.0.0",
            "description": "A comprehensive knowledge base for Huawei Ascend NPU development, structured as distributed AI Agent Skills.",
            "owner": {
                "name": "Ascend AI Coding",
                "email": "ascend-ai-coding@example.com",
            },
            "plugins": [],
        }

    plugins = marketplace.get("plugins", [])

    # Group skills by source
    skills_by_source: Dict[str, List[Tuple[Skill, str]]] = defaultdict(list)
    for skill, commit_sha in synced_skills:
        skills_by_source[skill.source.name].append((skill, commit_sha))

    plugins = [
        p
        for p in plugins
        if not (isinstance(p, dict) and p.get("external") is True)
    ]

    # Create grouped entries for each source
    for source_name, skills_list in skills_by_source.items():
        source = skills_list[0][0].source
        skill_paths = []
        descriptions = []

        for skill, _ in skills_list:
            skill_paths.append(f"./external/{source_name}/{skill.name}")
            parsed = parse_skill_md(skill.path)
            desc = parsed.get("description", "")
            if desc:
                descriptions.append(f"- {skill.name}: {desc[:100]}")

        description_text = (
            f"从 {source_name} 同步的 Ascend 技能集，包含 {len(skills_list)} 个技能"
        )
        if descriptions:
            description_text += "：\n" + "\n".join(descriptions[:3])
            if len(descriptions) > 3:
                description_text += f"\n- ... 等 {len(descriptions) - 3} 个技能"

        group_entry = {
            "name": f"external-{source_name}-skills",
            "description": description_text,
            "source": "./",
            "strict": False,
            "external": True,
            "source-url": source.url,
            "source-branch": source.branch,
            "skills": skill_paths,
        }

        plugins.append(group_entry)

    marketplace["plugins"] = plugins
    with marketplace_file.open("w", encoding="utf-8") as f:
        json.dump(marketplace, f, indent=2, ensure_ascii=False)

    print(
        f"✅ Updated marketplace.json with {len(skills_by_source)} external skill groups ({len(synced_skills)} skills total)"
    )


def main():
    """Main entry point for external skills sync."""
    config_path = ".github/external-sources.yml"

    try:
        sources = load_config(config_path)
        print(f"Loaded {len(sources)} external sources")
        for source in sources:
            print(f"  - {source.name}: {source.url} (branch: {source.branch})")
        print("\n✅ Configuration loaded successfully")

        print("\n" + "=" * 60)
        print("Starting sync...")
        print("=" * 60)

        results = sync_all_sources(config_path)

        print("\n" + "=" * 60)
        print("SYNC COMPLETE")
        print("=" * 60)
        print(f"  Synced: {results['synced']}")
        print(f"  Skipped: {results['skipped']}")
        print(f"  Errors: {results['errors']}")

        if results["errors"] > 0:
            sys.exit(1)

    except Exception as e:
        print(f"❌ Failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
