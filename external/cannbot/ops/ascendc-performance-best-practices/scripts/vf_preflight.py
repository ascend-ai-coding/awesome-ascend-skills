#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Inventory a target project's Reg VF build contract before code changes.

This is intentionally a conservative text/compile-database check, not a C++ parser.
Candidates are evidence for review; a pass never replaces a real compile/precision run.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".asc"}
SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".ruff_cache"}
VF_PATTERNS = {
    "reg_namespace": re.compile(r"\bAscendC::Reg::|(?<!:)\bReg::"),
    "microapi_namespace": re.compile(r"\bAscendC::MicroAPI::|(?<!:)\bMicroAPI::"),
    "cast_trait": re.compile(r"\bCastTrait\b"),
    "vec_scope": re.compile(r"\b__VEC_SCOPE__\b"),
    "simd_vf": re.compile(r"\b__simd_vf__\b"),
    "asc_vf_call": re.compile(r"\basc_vf_call\b"),
}
ALIAS_PATTERN = re.compile(r"\bnamespace\s+(Reg|MicroAPI)\s*=\s*(Reg|MicroAPI)\s*;")
ARCH_GUARD_PATTERN = re.compile(r"__NPU_ARCH__|__ASC_NPU_HOST__")
ARCH_FLAG_PATTERN = re.compile(r"(?:--npu-arch|--cce-aicore-arch)=([^\s\"']+)")
DEVICE_FUNCTION_PATTERN = re.compile(r"\b(?:__aicore__|__simd_callee__|__simd_vf__)\b")
HOST_DEVICE_API_PATTERN = re.compile(
    r"\b(?:AscendC::)?(?:Reg|MicroAPI)::(?:"
    r"CastTrait|CreateMask|LoadDist|MaskReg|RegTensor|StoreDist|UpdateMask"
    r")\b|\b__local_mem__\b"
)


def _is_skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS or part == "build" or part.startswith("cmake-build") for part in path.parts)


def _positive_arch_guard(directive: str) -> bool:
    if not ARCH_GUARD_PATTERN.search(directive):
        return False
    if re.match(r"^#\s*ifndef\b", directive) or re.search(
        r"!\s*defined\s*\(\s*(?:__NPU_ARCH__|__ASC_NPU_HOST__)\s*\)", directive
    ):
        return False
    return True


def _without_comments(line: str, in_block_comment: bool) -> tuple[str, bool]:
    output: list[str] = []
    index = 0
    while index < len(line):
        if in_block_comment:
            end = line.find("*/", index)
            if end < 0:
                return "".join(output), True
            index = end + 2
            in_block_comment = False
            continue
        block = line.find("/*", index)
        single = line.find("//", index)
        if single >= 0 and (block < 0 or single < block):
            output.append(line[index:single])
            return "".join(output), False
        if block < 0:
            output.append(line[index:])
            return "".join(output), False
        output.append(line[index:block])
        index = block + 2
        in_block_comment = True
    return "".join(output), in_block_comment


def _guarded_local_includes(path: Path) -> list[tuple[str, bool]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    includes: list[tuple[str, bool]] = []
    guard_stack: list[bool] = []
    in_block_comment = False
    for line in lines:
        code, in_block_comment = _without_comments(line, in_block_comment)
        stripped = code.strip()
        if re.match(r"^#\s*(if|ifdef|ifndef)\b", stripped):
            inherited = any(guard_stack)
            guard_stack.append(inherited or _positive_arch_guard(stripped))
        elif re.match(r"^#\s*elif\b", stripped) and guard_stack:
            inherited = any(guard_stack[:-1])
            guard_stack[-1] = inherited or _positive_arch_guard(stripped)
        elif re.match(r"^#\s*else\b", stripped) and guard_stack:
            guard_stack[-1] = any(guard_stack[:-1])
        elif re.match(r"^#\s*endif\b", stripped) and guard_stack:
            guard_stack.pop()
        include = re.match(r'^\s*#\s*include\s*"([^"]+)"', code)
        if include:
            includes.append((include.group(1), any(guard_stack)))
    return includes


def _source_files(
    project: Path, compiled_sources: list[Path]
) -> tuple[list[Path], dict[Path, set[Path]], dict[Path, set[Path]]]:
    if project.is_file():
        if project.suffix.lower() in SOURCE_SUFFIXES:
            resolved = project.resolve()
            return [resolved], {resolved: {resolved}}, {}
        return [], {}, {}
    if compiled_sources:
        local_sources = [
            path
            for path in project.rglob("*")
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES and not _is_skipped(path.relative_to(project))
        ]
        sources_by_name: dict[str, list[Path]] = {}
        for local_source in local_sources:
            sources_by_name.setdefault(local_source.name, []).append(local_source)
        owners: dict[Path, set[Path]] = defaultdict(set)
        guarded_owners: dict[Path, set[Path]] = defaultdict(set)
        for translation_unit in compiled_sources:
            if not translation_unit.is_file():
                continue
            translation_unit = translation_unit.resolve()
            pending = [(translation_unit, False)]
            guarded_reach: dict[Path, bool] = {}
            while pending:
                path, guarded = pending.pop()
                path = path.resolve()
                previous = guarded_reach.get(path)
                if previous is False or previous is guarded:
                    continue
                guarded_reach[path] = guarded
                try:
                    path.relative_to(project)
                except ValueError:
                    continue
                owners[path].add(translation_unit)
                if guarded:
                    guarded_owners[path].add(translation_unit)
                else:
                    guarded_owners[path].discard(translation_unit)
                for include, include_guarded in _guarded_local_includes(path):
                    candidate = (path.parent / include).resolve()
                    if candidate.is_file() and candidate.suffix.lower() in SOURCE_SUFFIXES:
                        pending.append((candidate, guarded or include_guarded))
                        continue
                    matches = [
                        source
                        for source in sources_by_name.get(Path(include).name, [])
                        if source.relative_to(project).as_posix().endswith(include)
                    ]
                    if len(matches) == 1:
                        pending.append((matches[0], guarded or include_guarded))
        return sorted(owners), dict(owners), dict(guarded_owners)
    sources = [
        path
        for path in project.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES and not _is_skipped(path.relative_to(project))
    ]
    return sources, {path.resolve(): {path.resolve()} for path in sources}, {}


def _compile_databases(project: Path, explicit: Path | None) -> list[Path]:
    if explicit is not None:
        return [explicit]
    direct = [project / "compile_commands.json", project / "build" / "compile_commands.json"]
    found = [path for path in direct if path.is_file()]
    if found:
        return list(dict.fromkeys(found))
    return sorted(
        path
        for path in project.rglob("compile_commands.json")
        if not any(part in SKIP_DIRS for part in path.relative_to(project).parts)
    )


def _command_text(entry: dict) -> str:
    if isinstance(entry.get("command"), str):
        return entry["command"]
    arguments = entry.get("arguments")
    if isinstance(arguments, list) and all(isinstance(item, str) for item in arguments):
        return shlex.join(arguments)
    return ""


def _classify_command(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if "--cce-aicore-only" in tokens:
        return "device_only"
    if "-xasc" in tokens or any(tokens[index:index + 2] == ["-x", "asc"] for index in range(len(tokens) - 1)):
        return "xasc_combined"
    if any(tokens[index:index + 2] == ["-x", "cce"] for index in range(len(tokens) - 1)):
        return "cce_wrapper"
    return None


def _read_commands(
    databases: list[Path],
) -> tuple[Counter, dict[str, str], list[str], list[str], list[Path], dict[Path, Counter]]:
    counts: Counter = Counter()
    evidence: dict[str, str] = {}
    arches: set[str] = set()
    errors: list[str] = []
    compiled_sources: set[Path] = set()
    source_contracts: dict[Path, Counter] = defaultdict(Counter)
    for database in databases:
        try:
            entries = json.loads(database.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {database}: {exc}")
            continue
        if not isinstance(entries, list):
            errors.append(f"{database}: root must be an array")
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            source_path: Path | None = None
            source = entry.get("file")
            if isinstance(source, str):
                source_path = Path(source)
                if not source_path.is_absolute():
                    directory = Path(str(entry.get("directory") or database.parent))
                    source_path = directory / source_path
                source_path = source_path.resolve()
                if source_path.suffix.lower() in SOURCE_SUFFIXES:
                    compiled_sources.add(source_path)
                else:
                    source_path = None
            command = _command_text(entry)
            arches.update(ARCH_FLAG_PATTERN.findall(command))
            contract = _classify_command(command)
            if contract is None:
                continue
            counts[contract] += 1
            evidence.setdefault(contract, command)
            if source_path is not None:
                source_contracts[source_path][contract] += 1
    return counts, evidence, sorted(arches), errors, sorted(compiled_sources), dict(source_contracts)


def _scan_source(path: Path, project: Path) -> tuple[Counter, list[dict], list[dict], list[dict]]:
    signals: Counter = Counter()
    risky_traits: list[dict] = []
    risky_host_device_apis: list[dict] = []
    aliases: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return signals, risky_traits, risky_host_device_apis, aliases

    depth = 0
    vec_depths: list[int] = []
    simd_depths: list[int] = []
    device_function_depths: list[int] = []
    pending_vec = False
    pending_simd = False
    pending_device_function = False
    guard_stack: list[bool] = []
    in_block_comment = False

    for line_number, line in enumerate(lines, 1):
        code, in_block_comment = _without_comments(line, in_block_comment)
        stripped = code.strip()
        if re.match(r"^#\s*(if|ifdef|ifndef)\b", stripped):
            inherited = any(guard_stack)
            guard_stack.append(inherited or _positive_arch_guard(stripped))
        elif re.match(r"^#\s*(elif)\b", stripped) and guard_stack:
            inherited = any(guard_stack[:-1])
            guard_stack[-1] = inherited or _positive_arch_guard(stripped)
        elif re.match(r"^#\s*else\b", stripped) and guard_stack:
            guard_stack[-1] = any(guard_stack[:-1])
        elif re.match(r"^#\s*endif\b", stripped) and guard_stack:
            guard_stack.pop()

        for name, pattern in VF_PATTERNS.items():
            if pattern.search(code):
                signals[name] += 1

        relative = str(path.relative_to(project)) if path != project else path.name
        if ALIAS_PATTERN.search(code):
            aliases.append({"path": relative, "line": line_number, "text": stripped[:240]})

        opens_device_function = bool(DEVICE_FUNCTION_PATTERN.search(code) and "{" in code)
        in_device_region = bool(
            vec_depths or simd_depths or device_function_depths or any(guard_stack) or opens_device_function
        )
        if VF_PATTERNS["cast_trait"].search(code) and not in_device_region:
            risky_traits.append({"path": relative, "line": line_number, "text": stripped[:240]})
        host_device_match = HOST_DEVICE_API_PATTERN.search(code)
        if host_device_match and not in_device_region:
            risky_host_device_apis.append(
                {
                    "path": relative,
                    "line": line_number,
                    "symbol": host_device_match.group(0),
                    "text": stripped[:240],
                }
            )

        if VF_PATTERNS["vec_scope"].search(code):
            pending_vec = True
        if VF_PATTERNS["simd_vf"].search(code):
            pending_simd = True
        if DEVICE_FUNCTION_PATTERN.search(code):
            pending_device_function = True

        for character in code:
            if character == "{":
                depth += 1
                if pending_vec:
                    vec_depths.append(depth)
                    pending_vec = False
                if pending_simd:
                    simd_depths.append(depth)
                    pending_simd = False
                if pending_device_function:
                    device_function_depths.append(depth)
                    pending_device_function = False
            elif character == "}":
                if vec_depths and depth == vec_depths[-1]:
                    vec_depths.pop()
                if simd_depths and depth == simd_depths[-1]:
                    simd_depths.pop()
                if device_function_depths and depth == device_function_depths[-1]:
                    device_function_depths.pop()
                depth = max(0, depth - 1)

        if pending_device_function and ";" in code and "{" not in code:
            pending_device_function = False

    return signals, risky_traits, risky_host_device_apis, aliases


def inspect(
    project: Path,
    compile_commands: Path | None = None,
    *,
    expect_vf: bool = False,
    expect_cast: bool = False,
) -> dict:
    project = project.resolve()
    databases = _compile_databases(project, compile_commands.resolve() if compile_commands else None)
    contracts, command_evidence, arches, database_errors, compiled_sources, source_contracts = _read_commands(databases)

    signals: Counter = Counter()
    risky_traits: list[dict] = []
    risky_host_device_apis: list[dict] = []
    aliases: list[dict] = []
    source_findings: dict[Path, tuple[Counter, list[dict], list[dict], list[dict]]] = {}
    sources, source_owners, guarded_source_owners = _source_files(project, compiled_sources)
    for source in sources:
        source_signals, source_traits, source_host_device_apis, source_aliases = _scan_source(source, project)
        source_findings[source.resolve()] = (
            source_signals,
            source_traits,
            source_host_device_apis,
            source_aliases,
        )
        signals.update(source_signals)
        risky_traits.extend(source_traits)
        risky_host_device_apis.extend(source_host_device_apis)
        aliases.extend(source_aliases)

    has_vf = any(signals.values())
    blockers: list[dict] = []
    warnings: list[dict] = []
    if expect_vf and not has_vf:
        blockers.append(
            {
                "code": "required_vf_missing",
                "message": "the implementation contract requires Reg/MicroAPI/VF, but the candidate "
                "contains no active VF signal",
            }
        )
    if expect_cast and not signals.get("cast_trait"):
        blockers.append(
            {
                "code": "required_casttrait_missing",
                "message": "the implementation contract requires CastTrait, but the candidate contains "
                "no active CastTrait declaration",
            }
        )
    if aliases:
        blockers.append(
            {
                "code": "manual_namespace_alias",
                "message": "project defines a Reg/MicroAPI namespace alias; "
                "verify version compatibility instead of masking it",
                "locations": aliases[:50],
            }
        )

    translation_units: list[dict] = []
    unresolved_units: list[dict] = []
    wrapper_only_units: list[dict] = []
    xasc_risky_traits: list[dict] = []
    xasc_risky_host_device_apis: list[dict] = []
    multiple_contract_units: list[dict] = []
    for translation_unit in compiled_sources or sources:
        translation_unit = translation_unit.resolve()
        owned_sources = [source for source, owners in source_owners.items() if translation_unit in owners]
        unit_signals: Counter = Counter()
        unit_traits: list[dict] = []
        unit_host_device_apis: list[dict] = []
        for source in owned_sources:
            source_signal, source_trait, source_host_device_api, _ = source_findings.get(
                source, (Counter(), [], [], [])
            )
            unit_signals.update(source_signal)
            unit_traits.extend(source_trait)
            if translation_unit not in guarded_source_owners.get(source, set()):
                unit_host_device_apis.extend(source_host_device_api)
        if not any(unit_signals.values()):
            continue
        unit_contracts = source_contracts.get(translation_unit, Counter())
        keep_contracts = {"device_only", "xasc_combined"}
        final_contracts = {name: count for name, count in unit_contracts.items() if name in keep_contracts}
        try:
            unit_path = str(translation_unit.relative_to(project))
        except ValueError:
            unit_path = str(translation_unit)
        translation_units.append(
            {
                "path": unit_path,
                "contracts": dict(sorted(unit_contracts.items())),
                "signals": dict(sorted(unit_signals.items())),
            }
        )
        if not final_contracts:
            location = {"path": unit_path}
            if unit_contracts.get("cce_wrapper"):
                wrapper_only_units.append(location)
            else:
                unresolved_units.append(location)
        if unit_contracts.get("xasc_combined") and unit_traits:
            xasc_risky_traits.extend(unit_traits)
        if unit_contracts.get("xasc_combined") and unit_host_device_apis:
            xasc_risky_host_device_apis.extend(unit_host_device_apis)
        if len(final_contracts) > 1:
            multiple_contract_units.append({"path": unit_path, "contracts": sorted(final_contracts)})

    if has_vf and unresolved_units:
        blockers.append(
            {
                "code": "compile_contract_unresolved",
                "message": "VF translation unit has no final device compile contract",
                "locations": unresolved_units[:50],
            }
        )
    if wrapper_only_units:
        blockers.append(
            {
                "code": "backend_contract_missing",
                "message": "VF translation unit has only -x cce wrapper evidence; capture its final backend command",
                "locations": wrapper_only_units[:50],
            }
        )
    if xasc_risky_traits:
        blockers.append(
            {
                "code": "xasc_host_visible_casttrait",
                "message": "CastTrait candidates outside __VEC_SCOPE__, __simd_vf__, or an architecture "
                "guard may be parsed before Reg APIs are visible",
                "locations": xasc_risky_traits[:50],
            }
        )
    if xasc_risky_host_device_apis:
        blockers.append(
            {
                "code": "xasc_host_pass_device_api_leakage",
                "message": (
                    "Reg/MicroAPI device-only declarations are visible to an xasc combined host pass; "
                    "adapt the include/declaration boundary with a target-proven architecture or VF guard. "
                    "This finding does not mean that the SDK API is missing."
                ),
                "locations": xasc_risky_host_device_apis[:50],
            }
        )
    if signals.get("microapi_namespace"):
        warnings.append(
            {
                "code": "legacy_microapi_namespace",
                "message": "legacy MicroAPI spelling is present; on CANN 9.x prefer Reg after checking "
                "the actual SDK and target contract",
            }
        )
    if multiple_contract_units:
        warnings.append(
            {
                "code": "multiple_build_contracts",
                "message": "a VF translation unit has multiple final build contracts",
                "locations": multiple_contract_units[:50],
            }
        )

    if blockers:
        status = "blocked"
    elif not has_vf:
        status = "not_applicable"
    else:
        status = "pass"

    return {
        "schema_version": 3,
        "status": status,
        "project": str(project),
        "compile_databases": [str(path.resolve()) for path in databases],
        "compile_database_errors": database_errors,
        "contracts": dict(sorted(contracts.items())),
        "architectures": arches,
        "command_evidence": command_evidence,
        "vf_translation_units": translation_units,
        "source_file_count": len(sources),
        "signals": dict(sorted(signals.items())),
        "blockers": blockers,
        "warnings": warnings,
        "expectations": {"vf": expect_vf, "casttrait": expect_cast},
        "limitations": [
            "source scope detection is conservative text analysis, not a C++ AST",
            "pass means the static preflight found no blocker; it does not prove compile, precision, or performance",
            "a probe failure never proves SDK API absence; resolve declarations from the target "
            "command's include roots",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--compile-commands", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="return nonzero when status is blocked")
    parser.add_argument(
        "--expect-vf",
        action="store_true",
        help="treat a candidate without active Reg/MicroAPI/VF signals as blocked",
    )
    parser.add_argument(
        "--expect-cast",
        action="store_true",
        help="treat a candidate without an active CastTrait declaration as blocked",
    )
    args = parser.parse_args()

    if not args.project.exists():
        parser.error(f"project does not exist: {args.project}")
    if args.compile_commands and not args.compile_commands.is_file():
        parser.error(f"compile database does not exist: {args.compile_commands}")

    report = inspect(
        args.project,
        args.compile_commands,
        expect_vf=args.expect_vf,
        expect_cast=args.expect_cast,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 1 if args.strict and report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
