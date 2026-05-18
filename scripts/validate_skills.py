#!/usr/bin/env python3
import json
import posixpath
import re
import sys
from pathlib import Path
from typing import Optional


def parse_frontmatter(content: str) -> tuple[dict, str]:
    if not content.startswith("---"):
        return {}, content

    end_match = re.search(r"\n---\n", content[3:])
    if not end_match:
        return {}, content

    frontmatter_str = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]

    frontmatter = {}
    for line in frontmatter_str.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()

    return frontmatter, body


LOCAL_DOMAIN_DIRS = {
    "base",
    "inference",
    "training",
    "profiling",
    "ops",
    "agent-tools",
    "ai-for-science",
}

PREFIX_DOMAIN_DIRS = {"ai-for-science"}
SKIP_SKILL_PREFIXES = {("tests", "fixtures")}


def should_skip_skill_file(rel_parts: tuple[str, ...]) -> bool:
    return ".worktrees" in rel_parts or rel_parts[:2] in SKIP_SKILL_PREFIXES


def validate_skill_name(actual_name: str, rel_parts: tuple[str, ...]) -> list[str]:
    errors = []

    if ".agents" in rel_parts:
        expected_name = rel_parts[-2]
        if actual_name != expected_name:
            errors.append(
                f"Agent skill name '{actual_name}' doesn't match directory '{expected_name}'"
            )
        return errors

    if rel_parts[0] == "external":
        expected_prefix = f"external-{rel_parts[1]}-"
        if not actual_name.startswith(expected_prefix):
            errors.append(
                f"External skill name '{actual_name}' should start with '{expected_prefix}'"
            )
        return errors

    if rel_parts[0] != "skills":
        errors.append(
            "Local skills must live under 'skills/<domain>/...'; found local SKILL.md outside skills/"
        )
        return errors

    if len(rel_parts) < 4:
        errors.append(
            "Invalid local skill path under skills/: expected skills/<domain>/<skill>/SKILL.md"
        )
        return errors

    domain = rel_parts[1]
    if domain not in LOCAL_DOMAIN_DIRS:
        errors.append(
            f"Unknown local skill domain '{domain}' under skills/; expected one of {sorted(LOCAL_DOMAIN_DIRS)}"
        )
        return errors

    expected_name = rel_parts[-2]
    is_leaf = len(rel_parts) == 4

    if is_leaf:
        if domain in PREFIX_DOMAIN_DIRS:
            expected_prefix = f"{domain}-"
            if not actual_name.startswith(expected_prefix):
                errors.append(
                    f"Local skill name '{actual_name}' should start with '{expected_prefix}'"
                )
        elif actual_name != expected_name:
            errors.append(
                f"Local skill name '{actual_name}' doesn't match directory '{expected_name}'"
            )
        return errors

    prefix_folder = domain if domain in PREFIX_DOMAIN_DIRS else rel_parts[2]
    expected_prefix = f"{prefix_folder}-"
    if not actual_name.startswith(expected_prefix):
        errors.append(
            f"Nested skill name '{actual_name}' should start with '{expected_prefix}'"
        )
    return errors


def validate_skill_file(skill_path: Path, repo_root: Path) -> tuple[list, list]:
    errors = []
    warnings = []

    content = skill_path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)

    if "name" not in frontmatter:
        errors.append("Missing 'name' field in frontmatter")
    elif not frontmatter["name"]:
        errors.append("Empty 'name' field in frontmatter")

    if "description" not in frontmatter:
        errors.append("Missing 'description' field in frontmatter")
    elif not frontmatter["description"]:
        errors.append("Empty 'description' field in frontmatter")
    elif len(frontmatter["description"]) < 20:
        warnings.append(
            f"Description is too short ({len(frontmatter['description'])} chars) - may affect agent matching"
        )

    actual_name = frontmatter.get("name", "")

    rel_path = skill_path.relative_to(repo_root)
    rel_parts = rel_path.parts

    errors.extend(validate_skill_name(actual_name, rel_parts))

    if "[TODO:" in body or "[TODO]" in body:
        warnings.append("Contains TODO placeholder - should be completed before merge")

    if len(body.strip()) < 100:
        warnings.append(f"Body content is very short ({len(body.strip())} chars)")

    return errors, warnings


def strip_local_prefix(path: str) -> str:
    if path.startswith("./"):
        return path[2:]
    return path


def is_same_or_child(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def find_named_bundle_root(skill_paths: list[str], bundle_name: str) -> Optional[str]:
    candidates = [bundle_name]
    if bundle_name.endswith("-skills"):
        candidates.append(bundle_name[: -len("-skills")])

    for candidate in candidates:
        roots = []
        for skill_path in skill_paths:
            parts = skill_path.split("/")
            if candidate not in parts:
                break
            index = parts.index(candidate)
            roots.append("/".join(parts[: index + 1]))
        else:
            if len(set(roots)) == 1:
                return roots[0]

    return None


def validate_marketplace(repo_root: Path) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []
    marketplace_path = repo_root / ".claude-plugin" / "marketplace.json"

    if not marketplace_path.exists():
        warnings.append("Missing .claude-plugin/marketplace.json")
        return errors, warnings

    try:
        marketplace = json.loads(marketplace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid marketplace.json: {exc}")
        return errors, warnings

    category_library = marketplace.get("categoryLibrary", {})
    primary_categories = category_library.get("primaryCategories", {})
    allowed_categories = set()
    for group in category_library.values():
        if isinstance(group, dict):
            allowed_categories.update(group.keys())

    for plugin in marketplace.get("plugins", []):
        plugin_name = plugin.get("name", "<missing-name>")
        category = plugin.get("category")
        categories = plugin.get("categories")

        if category not in primary_categories:
            errors.append(
                f"Plugin '{plugin_name}' has invalid primary category '{category}'"
            )

        if not isinstance(categories, list) or not categories:
            errors.append(f"Plugin '{plugin_name}' must define non-empty categories")
            categories = []
        else:
            if category and category not in categories:
                errors.append(
                    f"Plugin '{plugin_name}' categories must include primary category '{category}'"
                )
            for item in categories:
                if item not in allowed_categories:
                    errors.append(
                        f"Plugin '{plugin_name}' uses unknown category tag '{item}'"
                    )

        source = plugin.get("source")
        if isinstance(source, str) and source.startswith("./") and source != "./":
            if not (repo_root / strip_local_prefix(source)).exists():
                errors.append(f"Plugin '{plugin_name}' source does not exist: {source}")

        skill_paths = [
            strip_local_prefix(path)
            for path in plugin.get("skills", [])
            if isinstance(path, str)
        ]
        for skill_path in skill_paths:
            if not (repo_root / skill_path).exists():
                errors.append(
                    f"Plugin '{plugin_name}' skill path does not exist: ./{skill_path}"
                )

        if not skill_paths:
            continue

        if len(skill_paths) != len(plugin.get("skills", [])):
            errors.append(f"Plugin '{plugin_name}' has non-string entries in skills")

        expected_root = None
        if "official-bundle" in categories:
            domain = next(
                (item for item in categories if item in LOCAL_DOMAIN_DIRS),
                None,
            )
            if domain:
                expected_root = f"skills/{domain}"
            else:
                errors.append(
                    f"Plugin '{plugin_name}' official bundle must include one local domain category"
                )
        elif "external-skill-set" in categories:
            first_parts = skill_paths[0].split("/")
            if len(first_parts) >= 2 and first_parts[0] == "external":
                expected_root = "/".join(first_parts[:2])
            else:
                errors.append(
                    f"Plugin '{plugin_name}' external skill set must live under external/<source>/"
                )
        elif "domain-skill-set" in categories or len(skill_paths) > 1:
            expected_root = find_named_bundle_root(skill_paths, plugin_name)
            if expected_root is None:
                expected_root = posixpath.commonpath(skill_paths)

        if expected_root:
            if expected_root in {"", ".", "skills", "external"}:
                errors.append(
                    f"Plugin '{plugin_name}' skills are not scoped to a concrete bundle directory"
                )
            for skill_path in skill_paths:
                if not is_same_or_child(skill_path, expected_root):
                    errors.append(
                        f"Plugin '{plugin_name}' mixes skill path './{skill_path}' outside './{expected_root}'"
                    )

    return errors, warnings


def main():
    repo_root = Path(__file__).parent.parent

    skill_files = [
        f
        for f in repo_root.glob("**/SKILL.md")
        if not should_skip_skill_file(f.relative_to(repo_root).parts)
    ]

    if not skill_files:
        print("❌ No SKILL.md files found!")
        sys.exit(1)

    print(f"Found {len(skill_files)} SKILL.md files\n")

    total_errors = 0
    total_warnings = 0

    for skill_path in sorted(skill_files):
        rel_path = skill_path.relative_to(repo_root)
        errors, warnings = validate_skill_file(skill_path, repo_root)

        if errors or warnings:
            print(f"\n📄 {rel_path}")
            for error in errors:
                print(f"  ❌ ERROR: {error}")
                total_errors += 1
            for warning in warnings:
                print(f"  ⚠️  WARNING: {warning}")
                total_warnings += 1
        else:
            print(f"✅ {rel_path}")

    marketplace_errors, marketplace_warnings = validate_marketplace(repo_root)
    if marketplace_errors or marketplace_warnings:
        print("\n📄 .claude-plugin/marketplace.json")
        for error in marketplace_errors:
            print(f"  ❌ ERROR: {error}")
            total_errors += 1
        for warning in marketplace_warnings:
            print(f"  ⚠️  WARNING: {warning}")
            total_warnings += 1
    else:
        print("✅ .claude-plugin/marketplace.json")

    print(f"\n{'=' * 50}")
    print(f"Summary: {len(skill_files)} files checked")
    print(f"  Errors: {total_errors}")
    print(f"  Warnings: {total_warnings}")

    if total_errors > 0:
        print("\n❌ Validation FAILED!")
        sys.exit(1)
    else:
        print("\n✅ Validation PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
