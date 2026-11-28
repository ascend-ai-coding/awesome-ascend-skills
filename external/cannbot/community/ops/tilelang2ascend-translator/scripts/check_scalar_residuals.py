#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Conservative C++ token scan; findings require review, not automatic rewriting."""

import argparse
import bisect
import json
from pathlib import Path
import re


LITERALS = re.compile(
    r'R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)"'
    r'|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
    re.DOTALL,
)
TOKENS = re.compile(
    r"[A-Za-z_]\w*|\d+(?:\.\d+)?|\+\+|--|\+=|-=|\*=|/=|==|!=|<=|>=|&&|\|\||\S"
)
REVIEW = re.compile(
    r"//\s*scalar-review:\s*(batch-control|threshold-fallback|small-tile|"
    r"padding|index-init|tail|layout-view)\s*;\s*(\S.*)"
)
SUFFIXES = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".asc"}


def scan_source(source, filename="kernel.cpp"):
    """Return all loops and raw-address sites, including explicitly reviewed ones.

    Tokens retain original offsets. Strings/comments cannot create findings.
    Braced, unbraced, nested and multiline loops are all inventoried. This is
    not a C++ parser: macros, aliasing and called function bodies need review.
    """
    clean = LITERALS.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), source)
    matches = list(TOKENS.finditer(clean))
    tokens = [m.group() for m in matches]
    pairs, stack = {}, []
    for i, token in enumerate(tokens):
        if token in ("(", "[", "{"):
            stack.append(i)
        elif token in (")", "]", "}"):
            if not stack or tokens[stack[-1]] != {")": "(", "]": "[", "}": "{"}[token]:
                raise ValueError(
                    f"{filename}: unbalanced delimiters; review source/preprocessor branches"
                )
            start = stack.pop()
            pairs[start] = i
    if stack:
        raise ValueError(f"{filename}: unclosed delimiter")

    def statement_end(start):
        if start >= len(tokens):
            raise ValueError(f"{filename}: missing loop body")
        if tokens[start] == "{":
            return pairs[start] + 1
        if tokens[start] in ("if", "for", "while", "switch"):
            opening = start + 1
            if tokens[opening] == "constexpr":
                opening += 1
            if tokens[opening] != "(":
                raise ValueError(f"{filename}: unsupported control statement")
            end = statement_end(pairs[opening] + 1)
            if tokens[start] == "if" and end < len(tokens) and tokens[end] == "else":
                end = statement_end(end + 1)
            return end
        if tokens[start] == "do":
            end = statement_end(start + 1)
            condition_end = end + 2
            if tokens[end:condition_end] != ["while", "("]:
                raise ValueError(f"{filename}: malformed do/while")
            return pairs[end + 1] + 2
        cursor = start
        while cursor < len(tokens):
            if tokens[cursor] == ";":
                return cursor + 1
            if tokens[cursor] == "}":
                break
            cursor = pairs.get(cursor, cursor) + 1
        raise ValueError(f"{filename}: cannot locate statement end")

    newlines = [m.start() for m in re.finditer("\n", source)]
    lines = source.splitlines()
    findings = []
    do_conditions = set()
    for i, token in enumerate(tokens):
        if token in ("for", "while", "do") and i not in do_conditions:
            if token == "do":
                body_start = i + 1
                body_end = statement_end(body_start)
                do_conditions.add(body_end)
            else:
                opening = i + 1
                if opening >= len(tokens) or tokens[opening] != "(":
                    raise ValueError(f"{filename}: unsupported loop syntax")
                body_start = pairs[opening] + 1
                body_end = statement_end(body_start)
            body = " ".join(tokens[body_start:body_end])
            rules = [7]
            if re.search(r"\+=|-=|\*=|/=|\b(\w+)\s*=\s*\1\s*[+*\-/]", body):
                rules.append(1)
            if "GetPhyAddr" in body or "reinterpret_cast" in body:
                rules.append(2)
            if re.search(r"\b(for|while|do)\b", body) and re.search(
                r"\bswap\b|[<>]", body
            ):
                rules.append(3)
            if re.search(r"\[[^\];]*\[", body):
                rules.append(4)
            if "GetValue" in body and re.search(r"\b(Muls|Adds)\b", body):
                rules.append(5)
            if "SetValue" in body or "GetValue" in body:
                rules.append(6)
            kind = "loop"
        elif token in ("GetPhyAddr", "reinterpret_cast"):
            rules, kind = [2], "raw-address"
        else:
            continue
        line = bisect.bisect_right(newlines, matches[i].start()) + 1
        # Only real line comments adjacent to this site can acknowledge it.
        # Do not let comments inside a string/raw string suppress a finding.
        review = None
        for candidate in (line - 2, line - 1):
            if 0 <= candidate < len(lines):
                match = REVIEW.search(lines[candidate])
                if match:
                    offset = sum(len(s) + 1 for s in lines[:candidate]) + match.start()
                    lexical = next(
                        (m for m in LITERALS.finditer(source) if m.start() == offset),
                        None,
                    )
                    if lexical and lexical.group().startswith("//"):
                        review = {"category": match[1], "reason": match[2]}
        findings.append(
            {
                "file": str(filename),
                "line": line,
                "kind": kind,
                "rules": sorted(rules),
                "review": review,
            }
        )
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="C++ files or directories (recursive)"
    )
    args = parser.parse_args(argv)
    try:
        files = set()
        for path in args.paths:
            if not path.exists():
                raise ValueError(f"input does not exist: {path}")
            candidates = path.rglob("*") if path.is_dir() else [path]
            selected = {p for p in candidates if p.is_file() and p.suffix in SUFFIXES}
            if not selected:
                raise ValueError(f"no C++ files in input: {path}")
            files.update(selected)
        findings = []
        for path in sorted(files):
            findings.extend(scan_source(path.read_text(encoding="utf-8"), path))
        pending = sum(item["review"] is None for item in findings)
        print(
            json.dumps(
                {
                    "files_scanned": len(files),
                    "pending_review": pending,
                    "findings": findings,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1 if pending else 0
    except (OSError, UnicodeError, ValueError, IndexError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
