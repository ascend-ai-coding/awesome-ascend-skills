# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Print only the requested rule sections with their inclusive line ranges."""

import argparse
import re
from pathlib import Path

HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+)")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)")


def headings(lines):
    fence = None
    for index, line in enumerate(lines):
        marker = FENCE.match(line)
        if fence:
            if marker and marker[1].startswith(fence) and not marker[2].strip():
                fence = None
            continue
        if marker:
            fence = marker[1]
            continue
        heading = HEADING.match(line)
        if heading:
            yield index, len(heading[1]), heading[2].strip()


def section_range(outline, pattern, line_count, label):
    matches = []
    for index, level, title in outline:
        if pattern.match(title):
            matches.append((index, level))
    if len(matches) != 1:
        raise ValueError(f"Expected one heading for {label}, found {len(matches)}")
    start, level = matches[0]
    end = next(
        (index for index, depth, _ in outline if index > start and depth <= level),
        line_count,
    )
    return start, end


def applicability_range(lines):
    starts = [index for index, line in enumerate(lines) if line.strip() == "<适用>"]
    ends = [index for index, line in enumerate(lines) if line.strip() == "</适用>"]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError("Expected one complete <适用> declaration")
    return starts[0], ends[0] + 1


def clause_sections(lines, outline, labels):
    sections = []
    for label in labels:
        pattern = re.compile(rf"(?:规则\s+)?{re.escape(label)}(?=$|[\s:：]|\.(?:\s|$))")
        start, end = section_range(outline, pattern, len(lines), label)
        sections.append((label, start, end))
    return sections


def index_section(lines, outline):
    label = "快速索引"
    start, end = section_range(outline, re.compile(r"快速索引$"), len(lines), label)
    first_clause = re.compile(r"(?:规则\s+)?[A-Z][A-Z0-9]*-\d")
    for index, _, title in outline:
        if start < index < end and first_clause.match(title):
            end = index
            break

    for index in range(end - 1, start, -1):
        if lines[index].lstrip().startswith("|"):
            return label, start, index + 1
    raise ValueError("Quick index table not found")


def select_sections(lines, clauses, index_only):
    if clauses:
        return clause_sections(lines, list(headings(lines)), clauses)
    if index_only:
        return [index_section(lines, list(headings(lines)))]
    start, end = applicability_range(lines)
    return [("<适用>", start, end)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--clause", action="append", help="Exact clause ID; repeat for multiple clauses"
    )
    mode.add_argument("--index", action="store_true", help="Print the quick index only")
    args = parser.parse_args()

    try:
        lines = args.file.read_text(encoding="utf-8-sig").splitlines()
        sections = select_sections(lines, args.clause, args.index)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{args.file}: {exc}\n")

    for label, start, end in sections:
        while end > start and not lines[end - 1].strip():
            end -= 1
        print(f"=== {args.file}:{start + 1}-{end} [{label}] ===")
        print("\n".join(lines[start:end]))


if __name__ == "__main__":
    main()
