# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check local inline Markdown links and heading fragments without network access.

The reader covers the repository's inline links, ATX headings and explicit HTML
anchors. It excludes code blocks/spans and comments. Reference-style links, HTML
hrefs, generated headings and fragments in non-Markdown files are not validated.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote, urlsplit

from .paths import DOC_ROOTS
from .source import Finding

RULE = "doc-link"
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_CODE = re.compile(r"(`+)(.*?)\1(?!`)")
_LINK = re.compile(
    r"\[(?P<label>(?:\\.|[^\[\]\\]|\[[^\]\n]*\])*)\]\(\s*"
    r"(?P<target><[^>\n]*>|(?:\\.|[^()\s]|\([^()\s]*\))+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
_HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$)(.*)")
_ANCHOR = re.compile(r"<(?:a|h[1-6])\b[^>]*?\b(?:id|name)=[\"']([^\"']+)[\"']", re.I)


@dataclass(frozen=True)
class LinkReport:
    documents: tuple[Path, ...]
    local_links: int
    checked_fragments: int
    external_links: int
    unverified_fragments: int
    findings: tuple[Finding, ...]


def prose_lines(text: str):
    """Retain source line numbers while excluding comments and code blocks."""
    text = re.sub(
        r"<!--.*?-->", lambda match: "\n" * match[0].count("\n"), text, flags=re.S
    )
    fence = None
    for lineno, line in enumerate(text.splitlines(), 1):
        marker = _FENCE.match(line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif (
                marker[1].startswith(fence)
                and not line[slice(marker.end(), None)].strip()
            ):
                fence = None
            continue
        if fence is None and not line.startswith(("    ", "\t")):
            yield lineno, line


def inline_targets(line: str):
    """Yield raw inline-link targets outside code spans, without angle brackets."""
    for match in _LINK.finditer(_CODE.sub("", line)):
        yield match["target"].strip("<>")


def heading_locations(text: str):
    """Keep the link checker's heading slugs and their original source lines."""
    found = []
    used: set[str] = set()
    for lineno, line in prose_lines(text):
        heading = _HEADING.match(line)
        if not heading:
            continue
        title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[1]).strip()
        title = _LINK.sub(lambda match: match["label"], title)
        title = html.unescape(re.sub(r"<[^>]*>", "", title))
        slug = "".join(
            char
            for char in title.lower()
            if char in " _-" or unicodedata.category(char)[0] in "LNM"
        )
        slug = slug.replace(" ", "-")
        base, suffix = slug, 0
        while slug in used:
            suffix += 1
            slug = f"{base}-{suffix}"
        used.add(slug)
        found.append(
            {
                "title": title,
                "fragment": slug,
                "line": lineno,
                "level": len(line.lstrip().split()[0]),
            }
        )
    return found


def anchors(text: str) -> set[str]:
    """Collect ATX heading slugs, duplicate suffixes and explicit HTML anchors."""
    found = {heading["fragment"] for heading in heading_locations(text)}
    for _, line in prose_lines(text):
        found.update(
            html.unescape(value) for value in _ANCHOR.findall(_CODE.sub("", line))
        )
    return found


class FragmentError(ValueError):
    def __init__(self, reason, fragment):
        self.reason = reason
        super().__init__(f"{reason}: {fragment}")


def fragment_range(text: str, fragment: str):
    """Locate a heading subtree or explicit anchor without rendering Markdown."""
    if not isinstance(fragment, str) or not fragment or len(fragment) > 2048:
        raise ValueError("Expected a nonempty Markdown fragment up to 2048 characters")
    fragment = unquote(fragment.removeprefix("#"))
    headings = heading_locations(text)
    locations = [dict(heading) for heading in headings if heading["fragment"] == fragment]
    for lineno, line in prose_lines(text):
        if fragment not in {
            html.unescape(value) for value in _ANCHOR.findall(_CODE.sub("", line))
        }:
            continue
        if not any(item["line"] == lineno for item in locations):
            following = next(
                (item for item in headings if item["line"] >= lineno), None
            )
            lines = text.splitlines()
            adjacent = following and not any(
                part.strip() for part in lines[lineno: following["line"] - 1]
            )
            locations.append(
                {
                    "line": lineno,
                    "level": following["level"] if adjacent else 6,
                    "heading_line": following["line"] if adjacent else lineno,
                }
            )
    if not locations:
        raise FragmentError("fragment_missing", fragment)
    if len(locations) != 1:
        raise FragmentError("fragment_ambiguous", fragment)
    location = locations[0]
    end = next(
        (
            item["line"] - 1
            for item in headings
            if item["line"] > location.get("heading_line", location["line"])
            and item["level"] <= location["level"]
        ),
        len(text.splitlines()),
    )
    return {"fragment": fragment, "start_line": location["line"], "end_line": end}


def collect(targets: tuple[Path, ...] | list[Path] = DOC_ROOTS) -> LinkReport:
    documents: set[Path] = set()
    findings: list[Finding] = []
    for target in targets:
        target = target.resolve()
        if target.is_dir():
            documents.update(path for path in target.rglob("*.md") if path.is_file())
        elif target.is_file() and target.suffix.lower() == ".md":
            documents.add(target)
        else:
            findings.append(
                Finding(target, 1, RULE, "Expected a Markdown file or directory")
            )
    if not documents:
        findings.append(
            Finding(
                Path(targets[0]) if targets else Path.cwd(),
                1,
                RULE,
                "No Markdown documents found",
            )
        )
    local = checked = external = unverified = 0
    target_anchors: dict[Path, set[str]] = {}
    for document in sorted(documents):
        try:
            text = document.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            findings.append(Finding(document, 1, RULE, f"Cannot read document: {exc}"))
            continue
        for lineno, line in prose_lines(text):
            for target in inline_targets(line):
                target = html.unescape(target)
                target = re.sub(
                    r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~\\])", r"\1", target
                )
                try:
                    link = urlsplit(target)
                except ValueError as exc:
                    findings.append(
                        Finding(document, lineno, RULE, f"Invalid link {target}: {exc}")
                    )
                    continue
                if link.scheme or link.netloc:
                    external += 1
                    continue
                local += 1
                destination = (
                    (document.parent / unquote(link.path)).resolve()
                    if link.path
                    else document
                )
                if not destination.exists():
                    findings.append(
                        Finding(document, lineno, RULE, f"Missing target: {target}")
                    )
                    continue
                if not link.fragment:
                    continue
                if not destination.is_file() or destination.suffix.lower() != ".md":
                    unverified += 1
                    continue
                try:
                    if destination not in target_anchors:
                        target_anchors[destination] = anchors(
                            destination.read_text(encoding="utf-8")
                        )
                except (OSError, UnicodeError) as exc:
                    findings.append(
                        Finding(
                            document,
                            lineno,
                            RULE,
                            f"Cannot read target {target}: {exc}",
                        )
                    )
                    continue
                checked += 1
                if unquote(link.fragment) not in target_anchors[destination]:
                    findings.append(
                        Finding(
                            document, lineno, RULE, f"Missing Markdown anchor: {target}"
                        )
                    )
    return LinkReport(
        tuple(sorted(documents)), local, checked, external, unverified, tuple(findings)
    )
