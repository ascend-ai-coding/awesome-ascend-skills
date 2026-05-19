#!/usr/bin/env python3
"""Sync external skills from configured repositories."""

import re
import shutil
import subprocess
import sys
import tempfile
import posixpath
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from sync_types import ExternalBundle, ExternalSource, Skill, ConflictInfo, SyncResult
from validate_skills import validate_skill_file


FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
FALLBACK_FRONTMATTER_KEYS = {
    "name",
    "description",
    "original-name",
    "synced-from",
    "synced-date",
    "synced-commit",
    "license",
}


DEFAULT_CATEGORY_LIBRARY = {
    "primaryCategories": {
        "base": "基础环境、服务器连接、设备管理、容器与硬件诊断",
        "inference": "模型转换、量化、推理服务、评测与在线压测",
        "training": "分布式训练、通信测试、训练数据/权重/启动流程与强化学习",
        "profiling": "Profiling 采集、瓶颈定位、性能分析与 MFU 计算",
        "ops": "算子开发、算子接入、Triton/AscendC 迁移与算子级调优",
        "agent-tools": "面向 Agent/工程流程的 issue 分析、社区反馈、开源合入与知识沉淀工具",
        "ai-for-science": "AI for Science 模型迁移、框架路线与专项模型适配",
        "external": "从外部仓库同步的技能集合",
        "bundle": "组合多个 skills 的安装包或领域技能包",
    },
    "roleCategories": {
        "leaf-skill": "可单独触发的独立技能",
        "router-skill": "负责选择和分流到子技能的入口技能",
        "domain-skill-set": "面向一个技术方向的技能集合",
        "official-bundle": "官方推荐安装包",
        "external-skill-set": "外部同步技能集合",
    },
    "capabilityCategories": {
        "device-management": "NPU 状态、健康、功耗、固件、证书等设备管理",
        "hardware-diagnostics": "硬件诊断、带宽/算力/功耗测试、压力测试和复位",
        "virtualization": "AVI/vNPU 等虚拟化管理",
        "remote-access": "SSH、远程执行、文件传输和容器连接",
        "container-runtime": "Docker 容器启动、NPU 设备挂载和运行环境",
        "environment-setup": "CANN、torch_npu、训练/推理框架等环境准备",
        "pytorch-npu": "PyTorch 到 Ascend NPU 的扩展、迁移与运行",
        "distributed-communication": "HCCL 或 torch.distributed 通信链路与集合通信",
        "training-workflow": "训练数据、权重、脚本、任务启动与端到端流程",
        "reinforcement-learning": "VERL、PPO/GRPO/DAPO 等强化学习训练流程",
        "model-conversion": "PyTorch/ONNX/OM 等模型格式转换与导出",
        "quantization": "模型量化、压缩、精度调优和量化部署",
        "model-serving": "vLLM/MindIE/OpenAI-compatible API 等模型服务部署",
        "benchmarking": "性能、通信、算子或服务压测与基准测试",
        "evaluation": "模型精度、benchmark 数据集、Function Call 等评测",
        "operator-development": "AscendC、op-plugin、自定义算子开发和接入",
        "operator-migration": "Triton/CUDA/PyTorch 算子迁移到 Ascend",
        "performance-analysis": "Profiling 数据分析、慢卡/慢 rank、通信/计算/hostbound 瓶颈定位",
        "mfu-analysis": "训练或算子 MFU/FLOPs 利用率分析",
        "issue-analysis": "GitHub issue 总结、RCA、案例沉淀",
        "gitcode-workflow": "GitCode issue、PR、流水线、review 和合入流程",
        "community-feedback": "昇腾社区论坛抓取、反馈筛选和问题分析",
        "ai4s-model-migration": "AI for Science 模型、TensorFlow/PyTorch 路线与专项迁移",
        "external-sync": "外部来源技能同步和来源分组",
    },
}


BASE_EXTERNAL_CATEGORIES = [
    "external",
    "external-skill-set",
    "external-sync",
]


def ensure_category_library(marketplace: Dict) -> None:
    """Ensure marketplace has all known category library groups and tags."""
    category_library = marketplace.setdefault("categoryLibrary", {})
    for group_name, defaults in DEFAULT_CATEGORY_LIBRARY.items():
        group = category_library.setdefault(group_name, {})
        for tag, description in defaults.items():
            group.setdefault(tag, description)


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
            skills_path=source_config_path(
                source_data.get("skills_path", ""), allow_empty=True
            ),
            sync_mode=source_data.get("sync_mode", "flat"),
            marketplace_path=source_config_path(
                source_data.get("marketplace_path", ".claude-plugin/marketplace.json")
            ),
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
                    sync_mode=new_source_data.get("sync_mode", "flat"),
                    marketplace_path=new_source_data.get(
                        "marketplace_path", ".claude-plugin/marketplace.json"
                    ),
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
                    skills_path=new_source_data.get("skills_path", ""),
                    sync_mode=new_source_data.get("sync_mode", "flat"),
                    marketplace_path=new_source_data.get(
                        "marketplace_path", ".claude-plugin/marketplace.json"
                    ),
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


def split_skill_md(content: str) -> Tuple[str, str]:
    """Split frontmatter and body from SKILL.md content."""
    match = FRONTMATTER_RE.match(content)
    if not match:
        return "", content
    return match.group(1), content[match.end() :]


def parse_frontmatter_fallback(frontmatter: str) -> Optional[Dict[str, str]]:
    """Parse a narrow set of scalar frontmatter fields when YAML parsing fails."""
    parsed: Dict[str, str] = {}
    current_key: Optional[str] = None

    for line in frontmatter.splitlines():
        if not line.strip():
            continue

        if line[:1].isspace():
            if not current_key:
                return None

            continuation = line.strip()
            if continuation.startswith("- "):
                return None
            if continuation:
                parsed[current_key] = f"{parsed[current_key]} {continuation}".strip()
            continue

        if ":" not in line:
            return None

        key, value = line.split(":", 1)
        key = key.strip()
        if not key or key not in FALLBACK_FRONTMATTER_KEYS:
            return None

        parsed[key] = value.strip()
        current_key = key

    return parsed


def read_skill_md(
    skill_path: Path, *, tolerate_invalid_frontmatter: bool = False
) -> Tuple[Dict[str, Any], str]:
    """Read SKILL.md metadata with a tolerant frontmatter parser."""
    skill_md = skill_path / "SKILL.md"
    content = skill_md.read_text(encoding="utf-8")
    frontmatter, body = split_skill_md(content)
    if not frontmatter:
        return {}, body

    try:
        parsed = yaml.safe_load(frontmatter) or {}
        if isinstance(parsed, dict):
            return parsed, body
    except yaml.YAMLError:
        fallback = parse_frontmatter_fallback(frontmatter)
        if fallback is not None:
            return fallback, body

        if tolerate_invalid_frontmatter:
            return {}, body

        raise ValueError(f"Unsupported malformed frontmatter in {skill_md}")

    if tolerate_invalid_frontmatter:
        return {}, body

    raise ValueError(f"Unsupported non-dict frontmatter in {skill_md}")


def parse_skill_md(skill_path: Path) -> Dict[str, Any]:
    """Parse SKILL.md frontmatter and return it as a dictionary."""
    parsed, _ = read_skill_md(skill_path, tolerate_invalid_frontmatter=True)
    return parsed


def parse_skill_name_from_file(skill_md: Path) -> str:
    """Read a SKILL.md file and return its frontmatter name."""
    parsed, _ = read_skill_md(skill_md.parent, tolerate_invalid_frontmatter=True)
    return str(parsed.get("name", "")).strip()


def normalize_external_rel_path(path: str) -> str:
    """Normalize a marketplace path and reject paths outside the source root."""
    normalized = posixpath.normpath(path.strip().replace("\\", "/"))
    if normalized in {"", "."}:
        return ""
    if normalized.startswith("../") or normalized.startswith("/") or normalized == "..":
        raise ValueError(f"External skill path escapes source root: {path}")
    return normalized


def source_config_path(value: str, *, allow_empty: bool = False) -> str:
    """Normalize source config paths before joining them to cloned repos."""
    normalized = normalize_external_rel_path(value)
    if not normalized and not allow_empty:
        raise ValueError("Source config path must not be empty")
    return normalized


def external_skill_rel_path(skill: Skill) -> str:
    """Return the external storage path for a skill under external/<source>/."""
    return normalize_external_rel_path(skill.relative_path or skill.name)


def skill_name_slug(skill: Skill) -> str:
    """Return a stable slug for the synced skill frontmatter name."""
    return external_skill_rel_path(skill).replace("/", "-")


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
    for skill_md in sorted(search_path.rglob("SKILL.md")):
        rel_parts = skill_md.relative_to(search_path).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        skill_dir = skill_md.parent
        skills.append(
            Skill(
                name=skill_dir.name,
                path=skill_dir,
                source=source,
                has_skill_md=True,
                relative_path=skill_dir.name,
            )
        )
    return skills


def resolve_marketplace_skill_path(plugin_source: str, skill_source: str) -> str:
    """Resolve plugin source + skill entry using marketplace path semantics."""
    plugin_root = normalize_external_rel_path(plugin_source.removeprefix("./"))
    skill_path = normalize_external_rel_path(skill_source.removeprefix("./"))
    return normalize_external_rel_path(posixpath.join(plugin_root, skill_path))


def discover_marketplace_skills(
    repo_path: Path, source: ExternalSource
) -> Tuple[List[Skill], List[ExternalBundle]]:
    """Discover skills and bundle definitions from an external marketplace.json."""
    source_root = repo_path / source.skills_path if source.skills_path else repo_path
    marketplace_file = repo_path / source.marketplace_path
    if not marketplace_file.exists():
        return [], []

    marketplace = yaml.safe_load(marketplace_file.read_text(encoding="utf-8")) or {}
    skills_by_rel_path: Dict[str, Skill] = {}
    bundles: List[ExternalBundle] = []

    for plugin in marketplace.get("plugins", []):
        if not isinstance(plugin, dict):
            continue

        plugin_skills = plugin.get("skills", [])
        if not isinstance(plugin_skills, list) or not plugin_skills:
            continue

        plugin_source = str(plugin.get("source", "./"))
        bundle_skill_paths: List[str] = []

        for raw_skill_source in plugin_skills:
            if not isinstance(raw_skill_source, str):
                continue

            rel_path = resolve_marketplace_skill_path(plugin_source, raw_skill_source)
            skill_dir = source_root / rel_path
            if not (skill_dir / "SKILL.md").exists():
                continue

            if rel_path not in skills_by_rel_path:
                skills_by_rel_path[rel_path] = Skill(
                    name=skill_dir.name,
                    path=skill_dir,
                    source=source,
                    has_skill_md=True,
                    relative_path=rel_path,
                )
            bundle_skill_paths.append(f"./external/{source.name}/{rel_path}")

        if bundle_skill_paths:
            bundles.append(
                ExternalBundle(
                    name=str(plugin.get("name", f"{source.name}-skills")),
                    source=source,
                    description=str(plugin.get("description", "")),
                    skill_paths=bundle_skill_paths,
                )
            )

    skills = [skills_by_rel_path[key] for key in sorted(skills_by_rel_path.keys())]
    return skills, bundles


def discover_source_skills(
    repo_path: Path, source: ExternalSource
) -> Tuple[List[Skill], List[ExternalBundle]]:
    """Discover skills using the configured sync mode for a source."""
    if source.sync_mode == "marketplace":
        return discover_marketplace_skills(repo_path, source)
    return find_skills(repo_path, source), []


def get_local_skills() -> Set[str]:
    """Get canonical local skill names under skills/ for conflict detection."""
    skills = set()
    for skill_md in Path(".").glob("skills/**/SKILL.md"):
        skill_name = parse_skill_name_from_file(skill_md)
        if skill_name:
            skills.add(skill_name)
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

        for skill_md in sorted(source_dir.rglob("SKILL.md")):
            skill_dir = skill_md.parent
            rel_path = skill_dir.relative_to(source_dir).as_posix()

            parsed = parse_skill_md(skill_dir)
            commit_sha = str(parsed.get("synced-commit", ""))
            source_url = str(parsed.get("synced-from", source.url))
            existing_skills[(source_name, rel_path)] = (
                Skill(
                    name=skill_dir.name,
                    path=skill_dir,
                    source=ExternalSource(
                        name=source.name,
                        url=source_url,
                        branch=source.branch,
                        enabled=source.enabled,
                        skills_path=source.skills_path,
                        sync_mode=source.sync_mode,
                        marketplace_path=source.marketplace_path,
                    ),
                    has_skill_md=True,
                    relative_path=rel_path,
                ),
                commit_sha,
            )

    return existing_skills


def build_synced_skill_index(
    synced_skills: Dict[Tuple[str, str], Tuple[Skill, str]],
) -> Dict[str, Set[str]]:
    index: Dict[str, Set[str]] = {}
    for source_name, rel_path in synced_skills.keys():
        skill, _commit = synced_skills[(source_name, rel_path)]
        index.setdefault(skill.name, set()).add(source_name)
    return index


def prune_removed_source_skills(
    existing_skills: Dict[Tuple[str, str], Tuple[Skill, str]],
    source: ExternalSource,
    current_skill_names: Set[str],
) -> None:
    def is_current_path(rel_path: str) -> bool:
        if source.sync_mode == "marketplace":
            return any(
                rel_path == current_path or rel_path.startswith(f"{current_path}/")
                for current_path in current_skill_names
            )
        return rel_path in current_skill_names

    source_root = Path("external") / source.name
    removed_keys = [
        key
        for key in existing_skills
        if key[0] == source.name and not is_current_path(key[1])
    ]

    for key in removed_keys:
        skill_path = source_root / key[1]
        if skill_path.exists():
            shutil.rmtree(skill_path)
        del existing_skills[key]

    if source_root.exists():
        for path in sorted(source_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        if not any(source_root.iterdir()):
            source_root.rmdir()


def detect_conflicts(
    skill: Skill, local_skills: Set[str], synced_skills: Dict[str, Set[str]]
) -> Optional[ConflictInfo]:
    """Check if skill conflicts with local or synced skills."""
    if skill.name in local_skills:
        return ConflictInfo(
            skill_name=skill.name,
            local_path=f"./skills/... ({skill.name})",
            external_source="local",
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
    fm, body = read_skill_md(skill.path)

    # Rename skill to follow nested naming convention: external-{source}-{path-slug}
    original_name = fm.get("name", skill.name)
    new_name = f"external-{skill.source.name}-{skill_name_slug(skill)}"
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


def restore_backed_up_skill(target: Path, backup_target: Optional[Path]) -> None:
    """Restore the original synced skill after a failed sync attempt."""
    if not backup_target or not backup_target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(backup_target, target)


def cleanup_copied_skill(target: Path) -> None:
    """Remove a copied skill directory after a failed sync attempt."""
    if target.exists():
        shutil.rmtree(target)
    source_dir = target.parent
    if source_dir.exists() and not any(source_dir.iterdir()):
        source_dir.rmdir()


def get_validation_failure_reason(output: str) -> str:
    """Extract a concise validation failure reason from validator output."""
    for line in output.splitlines():
        if "ERROR:" in line:
            return line.split("ERROR:", 1)[1].strip()
    return "Validation failed"


def inject_attribution_tree(target: Path, source: ExternalSource, commit_sha: str) -> None:
    """Inject attribution into every SKILL.md copied under an external skill tree."""
    source_root = Path("external") / source.name
    for skill_md in sorted(target.rglob("SKILL.md")):
        skill_dir = skill_md.parent
        rel_path = skill_dir.relative_to(source_root).as_posix()
        copied_skill = Skill(
            name=skill_dir.name,
            path=skill_dir,
            source=source,
            has_skill_md=True,
            relative_path=rel_path,
        )
        attributed_content = inject_attribution(copied_skill, commit_sha)
        skill_md.write_text(attributed_content, encoding="utf-8")


def validate_copied_skill_tree(target: Path) -> Tuple[bool, str]:
    """Validate copied SKILL.md files without requiring marketplace to be final yet."""
    repo_root = Path.cwd().resolve()
    for skill_md in sorted(target.rglob("SKILL.md")):
        errors, _warnings = validate_skill_file(skill_md.resolve(), repo_root)
        if errors:
            return False, errors[0]
    return True, ""


def ignore_nested_skill_dirs(root_skill_path: Path):
    """Build a copytree ignore function that skips nested skill directories."""

    def ignore(current_dir: str, names: List[str]) -> Set[str]:
        ignored = {name for name in names if name == ".git"}
        current_path = Path(current_dir)
        if current_path != root_skill_path and "SKILL.md" in names:
            ignored.update(names)
        return ignored

    return ignore


def copy_skill(skill: Skill, commit_sha: str) -> Tuple[bool, str]:
    """Copy skill to external/ directory, inject attribution, and validate.

    Args:
        skill: The Skill object to copy.
        commit_sha: The Git commit SHA for attribution.

    Returns:
        Tuple of (success, reason). Reason is empty on success.
    """
    target = Path("external") / skill.source.name / external_skill_rel_path(skill)
    backup_dir: Optional[Path] = None
    backup_target: Optional[Path] = None
    keep_backup = False

    if target.exists():
        backup_dir = Path(tempfile.mkdtemp(prefix=f"sync-backup-{skill.name}-"))
        backup_target = backup_dir / target.name
        shutil.move(target, backup_target)

    try:
        ignore = shutil.ignore_patterns(".git")
        if skill.source.sync_mode != "marketplace":
            ignore = ignore_nested_skill_dirs(skill.path)

        shutil.copytree(skill.path, target, ignore=ignore)

        inject_attribution_tree(target, skill.source, commit_sha)
        is_valid, validation_reason = validate_copied_skill_tree(target)
        if is_valid:
            return True, ""

        cleanup_copied_skill(target)
        try:
            restore_backed_up_skill(target, backup_target)
        except Exception as restore_exc:
            keep_backup = True
            return (
                False,
                f"Validation failed and restore did not complete: {restore_exc}",
            )
        return False, validation_reason
    except Exception as exc:
        cleanup_copied_skill(target)
        try:
            restore_backed_up_skill(target, backup_target)
        except Exception as restore_exc:
            keep_backup = True
            return (
                False,
                f"Failed to restore existing skill after sync error: {restore_exc}",
            )
        return False, str(exc)
    finally:
        if backup_dir and backup_dir.exists() and not keep_backup:
            shutil.rmtree(backup_dir)


def validate_after_sync() -> None:
    """Run full repository validation after marketplace and README are updated."""
    result = subprocess.run(
        ["python3", "scripts/validate_skills.py"], capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError("Validation failed after external skills sync")


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
    all_bundles: List[ExternalBundle] = []

    for source in sources:
        if not source.enabled:
            print(f"Skipping disabled source: {source.name}")
            continue

        print(f"\nProcessing source: {source.name} ({source.url})")

        try:
            print(f"  Cloning {source.url} (branch: {source.branch})...")
            repo_path, commit_sha = clone_external_repo(source)
            print(f"  ✓ Cloned to {repo_path} (commit: {commit_sha[:7]})")

            skills, bundles = discover_source_skills(repo_path, source)
            print(f"  Found {len(skills)} skills")
            current_skill_names = {external_skill_rel_path(skill) for skill in skills}
            prune_removed_source_skills(
                existing_external_skills, source, current_skill_names
            )

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
                    success, reason = copy_skill(skill, commit_sha)
                    if success:
                        print(f"  ✓ Synced {skill.name}")
                        synced_skill = Skill(
                            name=skill.name,
                            path=Path("external")
                            / skill.source.name
                            / external_skill_rel_path(skill),
                            source=skill.source,
                            has_skill_md=True,
                            relative_path=external_skill_rel_path(skill),
                        )
                        all_synced_skills.append((synced_skill, commit_sha))
                        existing_external_skills[
                            (skill.source.name, external_skill_rel_path(skill))
                        ] = (
                            synced_skill,
                            commit_sha,
                        )
                    else:
                        print(f"  ⏭️  Skipping {skill.name}: {reason}")
                        all_skipped.append((skill.name, reason))
                except Exception as e:
                    print(f"  ❌ Error syncing {skill.name}: {e}")
                    all_errors.append((skill.name, str(e)))

            shutil.rmtree(repo_path)
            print(f"  ✓ Cleaned up {repo_path}")
            all_bundles.extend(bundles)

        except Exception as e:
            print(f"  ❌ Error processing source {source.name}: {e}")
            all_errors.append((source.name, str(e)))

    # Update marketplace.json and README.md with synced skills
    merged_synced_skills = sorted(
        existing_external_skills.values(),
        key=lambda item: (item[0].source.name, item[0].name),
    )

    print("\nUpdating marketplace.json...")
    update_marketplace(merged_synced_skills, external_bundles=all_bundles)
    print("\nUpdating README.md...")
    update_readme(merged_synced_skills)
    print("\nValidating synced repository...")
    validate_after_sync()

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
        skill_link = (
            f"[{skill.name}](external/{source_name}/"
            f"{external_skill_rel_path(skill)}/SKILL.md)"
        )

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


def filter_external_bundles(
    external_bundles: List[ExternalBundle], synced_skills: List[Tuple[Skill, str]]
) -> List[ExternalBundle]:
    """Keep only bundle skill paths that exist in the final synced skill set."""
    existing_paths = {
        f"./external/{skill.source.name}/{external_skill_rel_path(skill)}"
        for skill, _ in synced_skills
    }
    filtered_bundles = []
    for bundle in external_bundles:
        skill_paths = [path for path in bundle.skill_paths if path in existing_paths]
        if not skill_paths:
            continue
        filtered_bundles.append(
            ExternalBundle(
                name=bundle.name,
                source=bundle.source,
                description=bundle.description,
                skill_paths=skill_paths,
            )
        )
    return filtered_bundles


def update_marketplace(
    synced_skills: List[Tuple[Skill, str]],
    marketplace_path: str = ".claude-plugin/marketplace.json",
    external_bundles: Optional[List[ExternalBundle]] = None,
) -> None:
    """Update marketplace.json with external skills entries grouped by source.

    Args:
        synced_skills: List of tuples (Skill, commit_sha) that were successfully synced
        marketplace_path: Path to marketplace.json file (default: .claude-plugin/marketplace.json)

    Groups flat external syncs by source. Marketplace-aware syncs can pass
    external_bundles to preserve upstream bundle entries.
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

    ensure_category_library(marketplace)
    plugins = marketplace.get("plugins", [])
    existing_external_categories = {
        p.get("name"): p.get("categories", [])
        for p in plugins
        if isinstance(p, dict) and p.get("external") is True
    }
    external_insert_index = next(
        (
            index
            for index, plugin in enumerate(plugins)
            if isinstance(plugin, dict) and plugin.get("external") is True
        ),
        len(plugins),
    )

    external_bundles = filter_external_bundles(external_bundles or [], synced_skills)
    bundled_sources = {bundle.source.name for bundle in external_bundles}

    # Group skills by source
    skills_by_source: Dict[str, List[Tuple[Skill, str]]] = defaultdict(list)
    for skill, commit_sha in synced_skills:
        skills_by_source[skill.source.name].append((skill, commit_sha))

    non_external_plugins = [
        p for p in plugins if not (isinstance(p, dict) and p.get("external") is True)
    ]

    # Create grouped entries for each source
    external_plugins = []
    for bundle in sorted(
        external_bundles, key=lambda item: (item.source.name, item.name)
    ):
        entry_name = f"external-{bundle.source.name}-{bundle.name}"
        existing_categories = [
            category
            for category in existing_external_categories.get(entry_name, [])
            if isinstance(category, str)
        ]
        categories = list(dict.fromkeys(BASE_EXTERNAL_CATEGORIES + existing_categories))
        description_text = bundle.description or (
            f"从 {bundle.source.name} 同步的 {bundle.name} 技能包，"
            f"包含 {len(bundle.skill_paths)} 个技能"
        )
        external_plugins.append(
            {
                "name": entry_name,
                "description": description_text,
                "source": "./",
                "strict": False,
                "external": True,
                "source-url": bundle.source.url,
                "source-branch": bundle.source.branch,
                "upstream-name": bundle.name,
                "category": "external",
                "categories": categories,
                "skills": bundle.skill_paths,
            }
        )

    for source_name, skills_list in sorted(skills_by_source.items()):
        if source_name in bundled_sources:
            continue

        source = skills_list[0][0].source
        skill_paths = []
        descriptions = []

        for skill, _ in skills_list:
            skill_paths.append(f"./external/{source_name}/{external_skill_rel_path(skill)}")
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

        entry_name = f"external-{source_name}-skills"
        existing_categories = [
            category
            for category in existing_external_categories.get(entry_name, [])
            if isinstance(category, str)
        ]
        categories = list(dict.fromkeys(BASE_EXTERNAL_CATEGORIES + existing_categories))

        group_entry = {
            "name": entry_name,
            "description": description_text,
            "source": "./",
            "strict": False,
            "external": True,
            "source-url": source.url,
            "source-branch": source.branch,
            "category": "external",
            "categories": categories,
            "skills": skill_paths,
        }

        external_plugins.append(group_entry)

    marketplace["plugins"] = (
        non_external_plugins[:external_insert_index]
        + external_plugins
        + non_external_plugins[external_insert_index:]
    )
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
