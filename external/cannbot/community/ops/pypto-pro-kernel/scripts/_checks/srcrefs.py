# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`srcrefs`: check `<file>.py:<line>` references in docs against the source tree and report suspicious ones.

## Why This Is Needed

There are **more than 250** source-line references across the repository (`block_ops.py` alone has 42). They point into
`../pypto/`, which we edit ourselves—**inserting code above a cited line silently makes the line number point
elsewhere**. A measurement on 2026-09-28 found that **all five line numbers** in the helper table in `check-coverage.md`
were stale (`_static_dim` was cited at 3234 but was actually at 308) because our PyPTO stack had inserted code above
them.

A wrong line number is worse than none: it sends the reader to code that **looks relevant enough**.

## This Is a Report, Not a Gate

The criterion is heuristic: take backtick-quoted identifiers on the reference line and search within ±15 lines of the
target. Report the citation if none are found. On 2026-09-28, the first 25 cases were **checked one by one**: 12 were
genuinely stale; 13 were ambiguous or mostly correct, giving **48% precision**. (The initial figure was 56%; when
checking `MAX_MUTEX_ID`, I found that `system_ops.py:38-41` / `:44-72` were correct—the constant is at 39-41 and the
comment at 37-38, and the citation covers “comment + constant.” I had miscounted them as stale.) All false positives had
the same shape: “the reference points precisely to a line inside a function, while the identifier occurs in the prose of
the document, not near that line” (`_call_parser.py:1159` points exactly to the `raise NotSupported(` line; the document
mentions `store_align`, elsewhere).

Therefore, it **does not run under `rules`** (a gate must have zero false positives); invoke `python scripts/check.py
srcrefs`. Like `coverage`, it is a maintenance report.

## Manual Verification Waivers: `VERIFIED`

A report that **always shows 11 known false positives** is equivalent to no report—people learn to ignore it. Therefore,
add manually verified citations to `VERIFIED`, keyed by `(document, target file, target line)`: **if the line moves, the
waiver expires automatically**, and the citation reappears in the next report. This is the same principle as `反例：`
declarations in `rules`: a waiver leaves a reason; it does not loosen the criterion.

Before adding an entry, actually check it: open the target line and confirm that the citation points to it. Waiving an
incorrect reference is worse than leaving it in the report.

## Fixes: Choose One of Three, Ordered by Cost

| Fix | When to use |
| --- | --- |
| Use a symbol name (`_static_dim`, not `block_ops.py:308`). | **Default choice.** Symbol names never go stale. |
| Line number + pinned commit. | For a line **inside** a function (e.g. a `raise`) where the symbol is imprecise. |
| Quote the literal line text. | For error messages or `static_assert` conditions—the exact text is searchable. |

**Do not bulk-renumber automatically.** On 2026-09-28, “replace each number with the definition line of its identifier”
collapsed `_errors.py:437` (the source-window renderer) and `_errors.py:360` (the window constant) onto the `__str__`
definition, collapsing three different citations into one number; it also broke two **originally correct** references,
`system_ops.py:130` / `:44`. Review individually; do not trust a heuristic fix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .paths import PYPTO_ROOT, SKILL_ROOT

RULE = "src-line-drift"

PYPTO = PYPTO_ROOT
"""Default source checkout in the same workspace; override with check.py --pypto-root."""

WINDOW = 15
"""Number of lines to inspect on either side of the target. 15 usually spans a function header through its first
check."""

_REF = re.compile(r"([\w/]+\.(?:py|cpp|hpp|h))[`:]{1,2}(\d{2,5})")
_IDENT = re.compile(r"`([A-Za-z_][\w.]{3,})`")

_SKIP_PREFIXES = ("pl.", "vf.")
"""DSL names such as `pl.load` are not source identifiers; searching for them would match everywhere."""

VERIFIED = {
    # Checked each entry against the source on 2026-09-28; every citation points exactly to its target.
    # The heuristic cannot tell which identifier corresponds to which line, so these would always be reported.
    # Checked against pypto @ 428f3e8d9 (dev/pro-agentkit-build).
    (
        "references/diagnostics/defects/tuple-unpack-collapses-tile-memory-space.md",
        "_assignment_parser.py",
        303,
    ),  # let(f"_tuple_tmp_{...}")
    (
        "references/api/datamove.md",
        "out_memory.cpp",
        66,
    ),  # Reject pad=null with fillpad
    (
        "references/api/support.md",
        "_expr_evaluator.py",
        266,
    ),  # raise in TileType closure
    (
        "references/api/support.md",
        "_call_parser.py",
        1159,
    ),  # raise printed inside VF body
}
"""Manually verified waiver table. **Keys include line numbers**: changing the source line makes the entry stop matching
and return to the report."""


@dataclass(frozen=True)
class Suspect:
    doc: Path
    lineno: int
    target: str
    target_line: int
    idents: tuple[str, ...]
    reason: str

    def __str__(self) -> str:
        names = "、".join(f"`{i}`" for i in self.idents[:2])
        return (
            f"{self.doc}:{self.lineno}: {RULE}: "
            f"`{self.target}:{self.target_line}` 附近找不到 {names}——{self.reason}"
        )


def document_key(doc: Path) -> str:
    """Bundled waivers use skill-relative keys so moving the skill preserves them."""
    try:
        return doc.resolve().relative_to(SKILL_ROOT).as_posix()
    except ValueError:
        return doc.as_posix()


def _source_files(root: Path) -> dict[str, list[str]]:
    """Map each basename to its lines. **Skip duplicate basenames** because the target cannot be identified reliably."""
    seen: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.suffix not in {".py", ".cpp", ".hpp", ".h"} or "build" in path.parts:
            continue
        seen.setdefault(path.name, []).append(path)
    found = {}
    for name, paths in seen.items():
        if len(paths) == 1:
            try:
                found[name] = (
                    paths[0].read_text(encoding="utf-8", errors="replace").splitlines()
                )
            except OSError:
                pass
    return found


def collect(
    docs: list[Path],
    root: Path = PYPTO,
    verified: set[tuple[str, str, int]] = VERIFIED,
) -> tuple[list[Suspect], int, int, int]:
    """Return `(suspects, references_checked, skipped_due_to_duplicate_or_missing_files, manually_verified_waivers)`."""
    if not root.is_dir():
        raise SystemExit(
            f"{root} 不存在——`srcrefs` 要源码树才能核，请用 --pypto-root 指定"
        )
    sources = _source_files(root)
    suspects: list[Suspect] = []
    checked = skipped = waived = 0
    for doc in docs:
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for match in _REF.finditer(line):
                name = match.group(1).split("/")[-1]
                target_line = int(match.group(2))
                lines = sources.get(name)
                if lines is None:
                    skipped += 1
                    continue
                idents = tuple(
                    i
                    for i in _IDENT.findall(line)
                    if not i.startswith(_SKIP_PREFIXES)
                    and not i.endswith((".py", ".cpp", ".hpp", ".h"))
                )
                if target_line > len(lines):
                    checked += 1
                    suspects.append(
                        Suspect(
                            doc,
                            lineno,
                            name,
                            target_line,
                            idents,
                            f"那个文件只有 {len(lines)} 行",
                        )
                    )
                    continue
                if not idents:
                    skipped += 1  # No checkable identifier; do not guess.
                    continue
                checked += 1
                window = "\n".join(
                    lines[slice(max(0, target_line - WINDOW - 1), target_line + WINDOW)]
                )
                if any(i.split(".")[-1] in window for i in idents):
                    continue
                if (document_key(doc), name, target_line) in verified:
                    waived += 1
                    continue
                suspects.append(
                    Suspect(
                        doc,
                        lineno,
                        name,
                        target_line,
                        idents,
                        f"上下 {WINDOW} 行里都没有",
                    )
                )
    return suspects, checked, skipped, waived
