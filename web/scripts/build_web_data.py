#!/usr/bin/env python3
"""Build static data for the GitHub Pages skill browser."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


WEB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_SLUG = "ascend-ai-coding/awesome-ascend-skills"
DEFAULT_REF = "main"
LOCAL_DOMAINS = {
    "base",
    "inference",
    "training",
    "profiling",
    "ops",
    "agent-tools",
    "ai-for-science",
}


def split_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, flags=re.S)
    if not match:
        return {}, content

    frontmatter_text, body = match.groups()
    data: dict[str, Any] = {}
    for line in frontmatter_text.splitlines():
        if not line or line.startswith(" ") or line.startswith("\t") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        data[key.strip()] = value
    return data, body


def marketplace_maps(source_root: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    marketplace_path = source_root / ".claude-plugin" / "marketplace.json"
    if not marketplace_path.exists():
        return {}, []

    marketplace = json.loads(marketplace_path.read_text(encoding="utf-8"))
    path_to_category: dict[str, str] = {}
    bundle_candidates: list[dict[str, Any]] = []

    for plugin in marketplace.get("plugins", []):
        category = plugin.get("category") or ("external" if plugin.get("external") else "development")
        source = plugin.get("source")
        if source and source not in {".", "./"}:
            path_to_category[normalize_source(source)] = category

        skill_paths = [normalize_source(path) for path in plugin.get("skills", [])]
        for skill_path in skill_paths:
            skill_category = infer_category(skill_path, "") if category == "bundle" else category
            path_to_category.setdefault(skill_path, skill_category)

        if skill_paths:
            bundle_candidates.append(
                {
                    "name": plugin.get("name", "skill-bundle"),
                    "description": plugin.get("description", ""),
                    "category": category,
                    "external": bool(plugin.get("external")),
                    "sourceUrl": plugin.get("source-url", ""),
                    "skillPaths": skill_paths,
                }
            )

    return path_to_category, bundle_candidates


def normalize_source(source: str) -> str:
    return source.strip().removeprefix("./").rstrip("/")


def infer_category(path: str, description: str) -> str:
    text = f"{path} {description}".lower()
    parts = path.split("/")
    if path.startswith("external/"):
        return "external"
    if len(parts) >= 2 and parts[0] == "skills" and parts[1] in LOCAL_DOMAINS:
        return parts[1]
    if any(token in text for token in ["profiling", "analy", "mfu"]):
        return "profiling"
    if any(token in text for token in ["test", "bench", "eval", "verify", "rca", "issue"]):
        return "agent-tools"
    if any(token in text for token in ["docker", "server", "ssh", "dmi", "smi", "vnpu"]):
        return "base"
    return "inference"


def source_label(path: str) -> str:
    parts = path.split("/")
    if parts[0] == "external" and len(parts) > 1:
        return parts[1]
    if len(parts) >= 2 and parts[0] == "skills":
        return parts[1]
    return "local"


def first_heading(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def keywords_for(path: str, name: str, description: str, original_name: str) -> list[str]:
    raw = " ".join([path, name, original_name, description]).lower()
    tokens = sorted(set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", raw)))
    blocked = {"the", "and", "for", "with", "this", "that", "skill", "skills", "when", "use"}
    return [token for token in tokens if token not in blocked][:40]


def collect_skills(
    source_root: Path, path_to_category: dict[str, str], repo_slug: str, ref: str
) -> list[dict[str, Any]]:
    skills: list[dict[str, Any]] = []
    for skill_file in sorted(source_root.glob("**/SKILL.md")):
        rel_file = skill_file.relative_to(source_root).as_posix()
        if rel_file.startswith(("tests/", "web/", ".agents/", ".git/")):
            continue

        rel_dir = skill_file.parent.relative_to(source_root).as_posix()
        content = skill_file.read_text(encoding="utf-8")
        frontmatter, body = split_frontmatter(content)
        name = str(frontmatter.get("name") or skill_file.parent.name)
        description = str(frontmatter.get("description") or "").strip()
        original_name = str(frontmatter.get("original-name") or "").strip()
        synced_from = str(frontmatter.get("synced-from") or "").strip()
        category = path_to_category.get(rel_dir) or infer_category(rel_dir, description)
        reference_count = len(list(skill_file.parent.glob("references/**/*.md"))) + len(
            list(skill_file.parent.glob("reference/**/*.md"))
        )
        script_count = len([p for p in skill_file.parent.glob("scripts/**/*") if p.is_file()])

        skills.append(
            {
                "name": name,
                "displayName": original_name or name,
                "originalName": original_name,
                "title": first_heading(body) or (original_name or name),
                "description": description,
                "category": category,
                "source": source_label(rel_dir),
                "path": rel_dir,
                "skillFile": rel_file,
                "githubUrl": f"https://github.com/{repo_slug}/blob/{ref}/{rel_file}",
                "syncedFrom": synced_from,
                "referenceCount": reference_count,
                "scriptCount": script_count,
                "keywords": keywords_for(rel_dir, name, description, original_name),
                "body": body.strip(),
            }
        )
    return skills


def build_bundles(
    bundle_candidates: list[dict[str, Any]], skills: list[dict[str, Any]], repo_slug: str, ref: str
) -> list[dict[str, Any]]:
    by_path = {skill["path"]: skill for skill in skills}
    bundles: list[dict[str, Any]] = []
    for bundle in bundle_candidates:
        bundle_skills = [by_path[path] for path in bundle["skillPaths"] if path in by_path]
        if len(bundle_skills) < 2:
            continue
        bundle_path = common_skill_parent([skill["path"] for skill in bundle_skills])
        github_url = (
            f"https://github.com/{repo_slug}/tree/{ref}/{bundle_path}"
            if bundle_path
            else f"https://github.com/{repo_slug}"
        )
        bundles.append(
            {
                "name": bundle["name"],
                "description": bundle["description"],
                "category": bundle["category"],
                "external": bundle["external"],
                "path": bundle_path,
                "githubUrl": github_url,
                "sourceUrl": bundle.get("sourceUrl", ""),
                "skills": [skill["name"] for skill in bundle_skills],
                "displaySkills": [skill["displayName"] for skill in bundle_skills],
            }
        )
    return bundles


def common_skill_parent(paths: list[str]) -> str:
    if not paths:
        return ""
    split_paths = [path.split("/") for path in paths]
    common: list[str] = []
    for parts in zip(*split_paths):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    return "/".join(common)


def build_stats(skills: list[dict[str, Any]], bundles: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "skillCount": len(skills),
        "bundleCount": len(bundles),
        "categoryCounts": dict(sorted(Counter(skill["category"] for skill in skills).items())),
        "sourceCounts": dict(sorted(Counter(skill["source"] for skill in skills).items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root that contains SKILL.md files and .claude-plugin/marketplace.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=WEB_ROOT / "skills-data.js",
        help="Output JS data file. Relative paths are resolved from the web app root.",
    )
    parser.add_argument(
        "--repo-slug",
        default=DEFAULT_REPO_SLUG,
        help="GitHub owner/repo used for generated GitHub links and install commands.",
    )
    parser.add_argument(
        "--ref",
        default=DEFAULT_REF,
        help="Git ref used for generated GitHub blob/tree links.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_path = args.out if args.out.is_absolute() else WEB_ROOT / args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)

    path_to_category, bundle_candidates = marketplace_maps(source_root)
    skills = collect_skills(source_root, path_to_category, args.repo_slug, args.ref)
    bundles = build_bundles(bundle_candidates, skills, args.repo_slug, args.ref)
    payload = {
        "repo": args.repo_slug,
        "defaultBranch": args.ref,
        "generatedBy": "web/scripts/build_web_data.py",
        "stats": build_stats(skills, bundles),
        "skills": skills,
        "bundles": bundles,
    }
    output_path.write_text(
        "window.SKILLS_APP_DATA = "
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )
    display_path = output_path.relative_to(REPO_ROOT) if output_path.is_relative_to(REPO_ROOT) else output_path
    print(f"Wrote {display_path} with {len(skills)} skills and {len(bundles)} bundles")


if __name__ == "__main__":
    main()
