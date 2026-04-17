#!/usr/bin/env python3
import re
import sys
from pathlib import Path


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

    expected_name = skill_path.parent.name
    actual_name = frontmatter.get("name", "")

    rel_path = skill_path.relative_to(repo_root)
    rel_parts = rel_path.parts

    is_in_agents = ".agents" in rel_parts
    depth = len(rel_parts) - 1  # -1 for SKILL.md itself

    is_nested_skill = depth > 1 and not is_in_agents

    if is_nested_skill:
        folder_name = rel_parts[0]
        if not actual_name.startswith(f"{folder_name}-"):
            errors.append(
                f"Nested skill name '{actual_name}' should start with '{folder_name}-' (format: folder-skillname)"
            )
    else:
        if actual_name != expected_name:
            errors.append(
                f"Skill name '{actual_name}' doesn't match directory '{expected_name}'"
            )

    if "[TODO:" in body or "[TODO]" in body:
        warnings.append("Contains TODO placeholder - should be completed before merge")

    if len(body.strip()) < 100:
        warnings.append(f"Body content is very short ({len(body.strip())} chars)")

    return errors, warnings


def main():
    repo_root = Path(__file__).parent.parent

    skill_files = list(repo_root.glob("**/SKILL.md"))

    # Exclude .worktrees directory (git worktrees should not be validated)
    skill_files = [f for f in skill_files if ".worktrees" not in f.parts]

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
