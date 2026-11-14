# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reconcile public matmul/algorithm examples with the catalog's selection tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from .paths import CATALOG_DOC, EXAMPLES_ROOT
from .source import Finding

RULE = "example-catalog"
GROUPS = ("algorithms", "matmul")
_PREFIX = "../../examples/"
_LINK = re.compile(r"\[[^\]\n]*\]\(([^)\n]+)\)")
_FENCE = re.compile(r"^(`{3,}|~{3,})")


@dataclass(frozen=True)
class Catalog:
    examples: frozenset[Path]
    indexed: frozenset[Path]
    findings: tuple[Finding, ...]


def table_links(text: str):
    """Read inline table links, excluding fenced examples and ordinary prose."""
    fence = None
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        marker = _FENCE.match(stripped)
        if marker:
            if fence is None:
                fence = marker[1]
            elif (
                marker[1].startswith(fence)
                and not stripped[slice(marker.end(), None)].strip()
            ):
                fence = None
            continue
        if fence is None and stripped.startswith("|"):
            for target in _LINK.findall(stripped):
                yield lineno, target.strip("<>")


def collect(root: Path = EXAMPLES_ROOT, *, document: Path = CATALOG_DOC) -> Catalog:
    root, document = root.resolve(), document.resolve()
    examples: set[Path] = set()
    indexed: set[Path] = set()
    findings: list[Finding] = []
    for group in GROUPS:
        folder = root / group
        if not folder.is_dir():
            findings.append(Finding(folder, 1, RULE, "Missing example group"))
            continue
        for kernel in folder.rglob("kernel.py"):
            relative = kernel.parent.relative_to(root)
            if not any(part.startswith(("_", ".")) for part in relative.parts):
                examples.add(relative)
    if not examples:
        findings.append(Finding(root, 1, RULE, "No public kernel.py examples found"))
    try:
        text = document.read_text(encoding="utf-8")
    except OSError as exc:
        findings.append(Finding(document, 1, RULE, f"Cannot read catalog: {exc}"))
        return Catalog(frozenset(examples), frozenset(), tuple(findings))
    for lineno, target in table_links(text):
        link = urlsplit(target)
        if link.scheme or link.netloc:
            continue
        path = unquote(link.path)
        if not path.startswith(_PREFIX):
            continue
        relative = Path(path[slice(len(_PREFIX), None)])
        if not relative.parts or relative.parts[0] not in GROUPS:
            continue
        destination = (root / relative).resolve()
        if not destination.is_relative_to(root) or not destination.exists():
            findings.append(
                Finding(document, lineno, RULE, f"Broken example link: {target}")
            )
            continue
        owner = destination.relative_to(root)
        if destination.name == "kernel.py":
            owner = owner.parent
        elif not destination.is_dir():
            continue
        if owner in examples:
            indexed.add(owner)
        elif not any(owner in example.parents for example in examples):
            findings.append(
                Finding(
                    document,
                    lineno,
                    RULE,
                    f"Catalog target has no example kernel: {target}",
                )
            )
    for example in sorted(examples - indexed):
        findings.append(
            Finding(
                document,
                1,
                RULE,
                f"Example not indexed: {example.as_posix()}; "
                "add a direct directory or kernel.py link to a selection table",
            )
        )
    return Catalog(frozenset(examples), frozenset(indexed), tuple(findings))
