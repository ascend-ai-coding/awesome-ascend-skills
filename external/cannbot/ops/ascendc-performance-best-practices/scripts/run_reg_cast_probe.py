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

"""Compile the checked-in Reg CastTrait probe with a target project's real xasc command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SKILL_ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = SKILL_ROOT / "reference" / "common" / "arch35" / "regbase" / "probes"
POSITIVE_PROBE = PROBE_ROOT / "xasc_casttrait_static_local_pass.cpp"
CONTROL_PROBE = PROBE_ROOT / "xasc_casttrait_local_constexpr_control.cpp"
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".asc"}
DEPENDENCY_OPTIONS = {"-MF", "-MT", "-MQ", "--dependency-file"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compile_databases(project: Path, explicit: Path | None) -> list[Path]:
    if explicit is not None:
        return [explicit]
    direct = [project / "compile_commands.json", project / "build" / "compile_commands.json"]
    found = [path for path in direct if path.is_file()]
    if found:
        return list(dict.fromkeys(found))
    return sorted(project.rglob("compile_commands.json"))


def command_tokens(entry: dict[str, Any]) -> list[str]:
    command = entry.get("command")
    if isinstance(command, str):
        return shlex.split(command)
    arguments = entry.get("arguments")
    if isinstance(arguments, list) and all(isinstance(item, str) for item in arguments):
        return list(arguments)
    return []


def is_xasc_command(tokens: list[str]) -> bool:
    return "-xasc" in tokens or any(tokens[index:index + 2] == ["-x", "asc"] for index in range(len(tokens) - 1))


def load_xasc_entry(project: Path, explicit: Path | None, translation_unit: str | None) -> tuple[dict[str, Any], Path]:
    errors: list[str] = []
    for database in compile_databases(project, explicit):
        try:
            entries = json.loads(database.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{database}: {error}")
            continue
        if not isinstance(entries, list):
            errors.append(f"{database}: root is not an array")
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            source = str(entry.get("file") or "")
            if translation_unit and not (source == translation_unit or source.endswith(translation_unit)):
                continue
            if is_xasc_command(command_tokens(entry)):
                return entry, database
    detail = "; ".join(errors) if errors else "no xasc_combined entry"
    raise ValueError(f"cannot find a usable xasc compile command: {detail}")


def entry_source(entry: dict[str, Any], directory: Path) -> Path | None:
    source = entry.get("file")
    if not isinstance(source, str) or not source:
        return None
    path = Path(source)
    return (directory / path).resolve() if not path.is_absolute() else path.resolve()


def build_probe_command(entry: dict[str, Any], probe: Path, output: Path, database: Path) -> tuple[list[str], Path]:
    tokens = command_tokens(entry)
    if not tokens:
        raise ValueError("compile database entry has no command or arguments")
    directory = Path(str(entry.get("directory") or database.parent)).resolve()
    original_source = entry_source(entry, directory)
    rewritten: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"-o", *DEPENDENCY_OPTIONS}:
            index += 2
            continue
        if token == "-c":
            index += 1
            continue
        if token.startswith("-o") and token != "-o":
            index += 1
            continue
        if any(token.startswith(f"{option}=") for option in DEPENDENCY_OPTIONS):
            index += 1
            continue
        candidate = Path(token)
        if original_source is not None:
            resolved = (directory / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
            if resolved == original_source:
                index += 1
                continue
        elif candidate.suffix.lower() in SOURCE_SUFFIXES and not token.startswith("-"):
            index += 1
            continue
        rewritten.append(token)
        index += 1
    rewritten.extend(["-c", str(probe.resolve()), "-o", str(output.resolve())])
    return rewritten, directory


def first_error(stderr: str) -> str | None:
    for line in stderr.splitlines():
        if "error:" in line:
            return line.strip()
    return None


def run_one(entry: dict[str, Any], database: Path, probe: Path, output: Path, timeout: int) -> dict[str, Any]:
    command, directory = build_probe_command(entry, probe, output, database)
    completed = subprocess.run(
        command,
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "source": str(probe.resolve()),
        "source_sha256": sha256_file(probe),
        "command": shlex.join(command),
        "cwd": str(directory),
        "returncode": completed.returncode,
        "first_error": first_error(completed.stderr),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--compile-commands", type=Path)
    parser.add_argument("--translation-unit", help="prefer an xasc entry whose file ends with this value")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project.resolve()
    if not project.exists():
        print(f"[ERROR] project does not exist: {project}", file=sys.stderr)
        return 2
    explicit = args.compile_commands.resolve() if args.compile_commands else None
    if explicit is not None and not explicit.is_file():
        print(f"[ERROR] compile database does not exist: {explicit}", file=sys.stderr)
        return 2
    try:
        entry, database = load_xasc_entry(project, explicit, args.translation_unit)
        with tempfile.TemporaryDirectory(prefix="reg-cast-probe-") as temporary:
            temporary_root = Path(temporary)
            positive = run_one(entry, database, POSITIVE_PROBE, temporary_root / "positive.o", args.timeout)
            control = run_one(entry, database, CONTROL_PROBE, temporary_root / "control.o", args.timeout)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2

    status = "pass" if positive["returncode"] == 0 else "fail"
    report = {
        "schema_version": 1,
        "status": status,
        "capability": "xasc_function_scope_static_constexpr_casttrait",
        "project": str(project),
        "compile_database": str(database.resolve()),
        "positive": positive,
        "diagnostic_control": control,
        "interpretation": (
            "CastTrait is available with function-scope static constexpr storage"
            if status == "pass"
            else "the canonical positive probe failed; keep the conclusion local to this exact "
            "command and investigate the first error"
        ),
    }
    atomic_write(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
