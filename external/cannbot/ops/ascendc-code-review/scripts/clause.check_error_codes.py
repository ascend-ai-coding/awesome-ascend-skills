#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR
# A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------
"""Gate new deprecated CANN error-code references; no compiler or NPU needed."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".cu",
    ".cuh",
    ".inc",
    ".inl",
}
EXCLUDES = {
    ".git",
    ".svn",
    "third_party",
    "build",
    "out",
    "output",
    "dist",
    "install",
    "node_modules",
}
# Legacy macro spellings are syntax triggers, not an independently maintained code catalog.
LEGACY_MACROS = {
    "OP_LOGE_WITH_INVALID_INPUT_SHAPE": "EZ0001",
    "OP_LOGE_WITH_INVALID_ATTR": "EZ0002",
    "OP_LOGE_WITH_INVALID_ATTR_SIZE": "EZ0003",
    "OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE": "EZ0005",
    "OP_LOGE_WITH_INVALID_INPUT_FORMAT": "EZ0006",
    "OP_LOGE_WITH_INVALID_INPUT_DTYPE": "EZ0007",
}
CODE_RE = re.compile(r"\bEZ\d{4}\b")
TOKENS = re.compile(
    r"(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)"
    r'|(?P<raw>(?:u8|[uUL])?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\([\s\S]*?\)(?P=delimiter)")'
    r'|(?P<string>(?:u8|[uUL])?"(?:\\[\s\S]|[^"\\])*")'
    r"|(?P<char>(?:u8|[uUL])?'(?:\\[\s\S]|[^'\\])*')"
    r"|(?P<identifier>[A-Za-z_]\w*)|(?P<other>[^\s])"
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    code: str
    entry: str
    revision: str = "working-tree"


def load_policy(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    codes = data.get("governance_deprecated")
    if data.get("schema_version") != 1 or not isinstance(codes, list):
        raise ValueError(f"invalid cann-error-code-standard catalog: {path}")
    # Keep the gate aligned with the six-code governance policy.
    if set(codes) != set(LEGACY_MACROS.values()) or len(codes) != 6:
        raise ValueError(
            "catalog policy changed; review this six-code gate before reinstalling"
        )
    return set(codes)


def scan_text(
    text: str, path: str, banned: set[str], revision: str = "working-tree"
) -> list[Finding]:
    # Apply C/C++ line splicing before recognizing comments, retaining original source lines.
    newlines = [i for i, ch in enumerate(text) if ch == "\n"]
    origins: list[int] = []
    parts: list[str] = []
    position = 0
    for splice in re.finditer(r"\\\r?\n", text):
        parts.append(text[position: splice.start()])
        origins.extend(range(position, splice.start()))
        position = splice.end()
    parts.append(text[position:])
    origins.extend(range(position, len(text)))
    joined = "".join(parts)
    tokens = [m for m in TOKENS.finditer(joined) if m.lastgroup != "comment"]
    aliases = {name: code for name, code in LEGACY_MACROS.items() if code in banned}
    # Resolve simple local #define alias chains, including function-like wrappers.
    definitions: dict[str, list[str]] = {}
    cleaned = TOKENS.sub(
        lambda m: re.sub(r"[^\n]", " ", m[0])
        if m.lastgroup in {"comment", "string", "raw", "char"}
        else m[0],
        joined,
    )
    for define in re.finditer(
        r"^\s*#\s*define\s+(\w+)(?:\([^\n]*?\))?[^\n]*", cleaned, re.M
    ):
        definitions[define[1]] = re.findall(r"\b[A-Za-z_]\w*\b", define[0])[2:]
    for _ in range(len(definitions) + 1):
        before = len(aliases)
        for name, values in definitions.items():
            for value in values:
                if value in aliases:
                    aliases.setdefault(name, aliases[value])
                    break
        if before == len(aliases):
            break
    findings: set[Finding] = set()

    def add(offset: int, code: str, entry: str) -> None:
        line = bisect.bisect_left(newlines, origins[offset]) + 1
        findings.add(Finding(path, line, code, entry, revision))

    i = 0
    while i < len(tokens):
        token = tokens[i]
        kind = token.lastgroup
        if kind == "identifier":
            if token[0] in aliases:
                add(token.start(), aliases[token[0]], token[0])
            elif token[0] in banned:
                add(token.start(), token[0], "code-identifier")
        elif kind in {"string", "raw"}:
            # Adjacent C/C++ string literals concatenate, even with comments between them.
            contents = ""
            locations: list[int] = []
            while i < len(tokens) and tokens[i].lastgroup in {"string", "raw"}:
                current = tokens[i]
                value = current[0]
                start = (
                    value.find("(") + 1
                    if current.lastgroup == "raw"
                    else value.find('"') + 1
                )
                end = value.rfind(")") if current.lastgroup == "raw" else len(value) - 1
                contents += value[start:end]
                locations.extend(range(current.start() + start, current.start() + end))
                i += 1
            for match in CODE_RE.finditer(contents):
                if match[0] in banned:
                    for offset in locations[match.start(): match.end()]:
                        add(offset, match[0], "code-literal")
            continue
        i += 1
    return sorted(findings, key=lambda f: (f.path, f.line, f.code, f.entry))


def git(repo: Path, *args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace").strip() or "git failed"
        )
    return result.stdout


def source_path(path: str) -> bool:
    p = Path(path)
    return p.suffix.lower() in SUFFIXES and not EXCLUDES.intersection(p.parts[:-1])


def scan_diff(
    repo: Path, banned: set[str], base: str | None, head: str | None
) -> list[Finding]:
    compare = ["--cached"] if head is None else [base, head]
    previous = base
    if head is None:
        resolved = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", "HEAD"],
            capture_output=True,
        )
        if resolved.returncode not in (0, 1):
            raise RuntimeError("cannot resolve index base HEAD")
        previous = (
            resolved.stdout.decode().strip() if resolved.returncode == 0 else None
        )
    if head is not None and base is None:
        compare = [
            git(repo, "hash-object", "-t", "tree", "--stdin", data=b"")
            .decode()
            .strip(),
            head,
        ]
    names = git(
        repo,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--name-only",
        "-z",
        "--diff-filter=ACMR",
        *compare,
    )
    findings = []
    for encoded in names.split(b"\0"):
        if not encoded:
            continue
        path = os.fsdecode(encoded)
        if not source_path(path):
            continue
        diff = git(
            repo,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--unified=0",
            *compare,
            "--",
            f":(literal){path}",
        )
        added: set[int] = set()
        hunks = []
        for hunk in re.finditer(
            rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", diff, re.M
        ):
            old_start, old_length = int(hunk[1]), int(hunk[2] or b"1")
            start, length = int(hunk[3]), int(hunk[4] or b"1")
            hunks.append((old_start, old_length, start, length))
            added.update(range(start, start + length))
        revision = head or "index"
        # Read blobs from the index/commit, never the unstaged working-tree copy.
        content = git(repo, "show", f"{head or ''}:{path}").decode(
            "utf-8", errors="strict"
        )
        matches = scan_text(content, path, banned, revision)
        old_matches = set()
        if previous and git(
            repo, "ls-tree", "--name-only", "-z", previous, "--", f":(literal){path}"
        ):
            old_content = git(repo, "show", f"{previous}:{path}").decode(
                "utf-8", errors="strict"
            )
            old_matches = {(f.line, f.code, f.entry) for f in scan_text(old_content, path, banned)}
        for finding in matches:
            # Removed comment delimiters or newly introduced local aliases can activate an
            # unchanged line. Map it back to the old blob before treating it as legacy.
            delta = 0
            for old_start, old_length, start, length in hunks:
                old_boundary = old_start + old_length if old_length else old_start + 1
                boundary = start + length if length else start + 1
                if finding.line >= boundary:
                    delta = old_boundary - boundary
            if (
                finding.line in added
                or (finding.line + delta, finding.code, finding.entry)
                not in old_matches
            ):
                findings.append(finding)
    return findings


def scan_push(repo: Path, banned: set[str], remote: str, stream: str) -> list[Finding]:
    findings = []
    seen: set[str] = set()
    for line in stream.splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                "invalid pre-push stdin: expected local-ref local-sha remote-ref remote-sha"
            )
        _, local, _, old = fields
        if (
            not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", local)
            or len(local) != len(old)
            or not re.fullmatch(r"[0-9a-f]+", old)
        ):
            raise ValueError("invalid pre-push object ID")
        if set(local) == {"0"}:  # deleting a ref introduces no code
            continue
        tip = git(repo, "rev-parse", "--verify", f"{local}^{{commit}}").decode().strip()
        args = ["rev-list", "--reverse", tip]
        if set(old) != {"0"}:
            args.append(f"^{old}")
        args.extend(["--not", f"--remotes={remote}"])
        for commit in git(repo, *args).decode().splitlines():
            if commit in seen:
                continue
            seen.add(commit)
            parents = (
                git(repo, "rev-list", "--parents", "-n", "1", commit)
                .decode()
                .split()[1:]
            )
            findings.extend(
                scan_diff(repo, banned, parents[0] if parents else None, commit)
            )
    return findings


def default_catalog() -> Path:
    installed = Path(__file__).with_name("documented-error-codes.json")
    if installed.is_file():
        return installed
    return (
        Path(__file__).resolve().parents[2]
        / "cann-error-code-standard/references/documented-error-codes.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--catalog", type=Path, default=default_catalog())
    parser.add_argument("--format", choices=["text", "json"], default="text")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--staged", action="store_true")
    modes.add_argument("--base", help="Diff base revision; --head defaults to HEAD")
    modes.add_argument("--pre-push", nargs=2, metavar=("REMOTE", "URL"))
    parser.add_argument("--head", default="HEAD")
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Relative file path for working-tree review",
    )
    args = parser.parse_args()
    try:
        banned = load_policy(args.catalog)
        repo = args.root.resolve(strict=True)
        if args.staged:
            findings = scan_diff(repo, banned, None, None)
        elif args.base:
            base = (
                git(repo, "rev-parse", "--verify", f"{args.base}^{{commit}}")
                .decode()
                .strip()
            )
            head = (
                git(repo, "rev-parse", "--verify", f"{args.head}^{{commit}}")
                .decode()
                .strip()
            )
            findings = scan_diff(repo, banned, base, head)
        elif args.pre_push:
            findings = scan_push(repo, banned, args.pre_push[0], sys.stdin.read())
        else:
            paths = []
            if args.file:
                for name in args.file:
                    path = (repo / name).resolve(strict=True)
                    path.relative_to(repo)
                    paths.append(path)
            else:
                for directory, dirs, files in os.walk(repo, followlinks=False):
                    dirs[:] = [d for d in dirs if d not in EXCLUDES]
                    paths.extend(Path(directory) / f for f in files)
            findings = []
            for path in sorted(paths):
                relative = path.relative_to(repo).as_posix()
                if not path.is_symlink() and source_path(relative):
                    findings.extend(
                        scan_text(path.read_text(encoding="utf-8"), relative, banned)
                    )
        if args.format == "json":
            print(
                json.dumps(
                    {"findings": [asdict(f) for f in findings], "total": len(findings)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            for finding in findings:
                print(
                    f"{finding.path}:{finding.line}: ERR-1 {finding.code} via {finding.entry} [{finding.revision}]"
                )
            if findings:
                print(
                    f"BLOCKED: deprecated error codes {', '.join(sorted(banned))}. "
                    "Use cann-error-code-standard to choose a registered semantic code.",
                    file=sys.stderr,
                )
            else:
                print(
                    "PASS: no new deprecated error-code references in the selected scope."
                )
        return 1 if findings else 0
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: error-code scan incomplete (blocked): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
