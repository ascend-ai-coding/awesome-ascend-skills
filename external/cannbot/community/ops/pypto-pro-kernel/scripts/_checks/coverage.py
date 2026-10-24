# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`api-coverage`: verify the assignments in the 81-item table from §8.2 against example code.

Read the assignment table from `references/tooling/example-coverage.md` and check that the assigned example source
actually uses each API. Source-call coverage does not establish argument-combination coverage or numerical correctness.

The reverse index (API → examples that use it) is a separate `surface` command. It **is not written to a file**: a
generated index committed to the repository would become stale, so computing it on demand avoids maintaining a second
copy of the facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .source import read, used_names
from .paths import COVERAGE_DOC, EXAMPLES_ROOT

NAMESPACES = ("vf", "pl", "pl.system")
"""Namespaces to index. `pl.system.*` must be listed separately: examples make 56 such calls, and indexing only the
two-level prefix would miss them all."""


def _prefix(source, namespace: str):
    """Replace namespace names with the import aliases actually used in this file."""
    if namespace == "vf":
        return source.vf
    if source.pl is None:
        return None
    return (
        source.pl if namespace == "pl" else f"{source.pl}.{namespace.split('.', 1)[1]}"
    )


RULE = "api-coverage"

_SECTION = re.compile(r"^### 八点二、(\d+) 项逐项\s*$", re.MULTILINE)
_ROW = re.compile(r"^\|\s*\d+\s*\|\s*`([a-z0-9_]+)`\s*\|([^|\n]*)\|", re.MULTILINE)
_EXAMPLE = re.compile(r"(?:^|\+)\s*`([a-z0-9_]+)`")


@dataclass
class Coverage:
    expected: int
    """Item count declared in the §8.2 heading; must match the number of API rows parsed from the table."""
    assigned: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """All examples assigned to each `vf.<api>` in the table."""
    used: dict[str, set[str]] = field(default_factory=dict)
    """Example directory name → the `vf.<api>` calls actually made by its `kernel.py`."""

    @property
    def written(self) -> set[str]:
        return set(self.used)

    def problems(self) -> list[str]:
        """Count as failures: the table row count differs, or an existing example does not call an API assigned to
        it.
        """
        found = []
        if len(self.assigned) != self.expected:
            found.append(
                f"§八点二 标题自报 {self.expected} 项，表里解析到 {len(self.assigned)} 行"
            )
        for api, examples in sorted(self.assigned.items()):
            if not examples:
                found.append(f"表中 `vf.{api}` 的样例列没有可识别的样例名")
            for example in examples:
                if example not in self.written or api in self.used[example]:
                    continue
                others = sorted(name for name, apis in self.used.items() if api in apis)
                where = (
                    f"其他调用样例：{'、'.join(others)}"
                    if others
                    else "而且全仓没有任何样例调它"
                )
                found.append(
                    f"表把 `vf.{api}` 指派给 `{example}`，那一例的 kernel.py 没调它——{where}"
                )
        return found

    def pending(self) -> dict[str, list[str]]:
        """Return APIs assigned to examples that do not exist yet; report them separately from missing calls in existing
        examples.
        """
        waiting: dict[str, list[str]] = {}
        for api, examples in sorted(self.assigned.items()):
            for example in examples:
                if example not in self.written:
                    waiting.setdefault(example, []).append(api)
        return waiting


def kernels(root: Path = EXAMPLES_ROOT) -> dict[str, Path]:
    """Map each example directory name to its `kernel.py`; directory names are unique under `basic_api/`."""
    if not (root / "basic_api").is_dir():
        raise SystemExit(
            f"{root / 'basic_api'} 不存在——请用 --examples-root 指定包含 basic_api 的样例目录"
        )
    found = {}
    for path in sorted((root / "basic_api").glob("*/*/kernel.py")):
        found[path.parent.name] = path
    return found


def collect(root: Path = EXAMPLES_ROOT, *, document: Path = COVERAGE_DOC) -> Coverage:
    text = document.read_text(encoding="utf-8")
    heading = _SECTION.search(text)
    if heading is None:
        raise SystemExit(f"{document}: 找不到 `### 八点二、<n> 项逐项` 这个小节标题")
    body = text[slice(heading.end(), None)]
    end = body.find("\n### ")
    section = body if end < 0 else body[:end]
    result = Coverage(
        expected=int(heading.group(1)),
        assigned={
            api: tuple(_EXAMPLE.findall(cell)) for api, cell in _ROW.findall(section)
        },
    )
    for name, path in kernels(root).items():
        source = read(path)
        result.used[name] = used_names(source.tree, source.vf)
    return result


def surface(root: Path = EXAMPLES_ROOT) -> dict[str, dict[str, list[str]]]:
    """Build a reverse index: `{"vf": {api: [examples...]}, "pl": {...}, "pl.system": {...}}`, sorted by API name.

    This answers the question represented by the `surface` field in ascriptor's `metadata.json`: “I want to use an API;
    which example should I inspect?” The table in the design document is only forward (example → API), while an agent
    needs the reverse direction.
    """
    index: dict[str, dict[str, set[str]]] = {namespace: {} for namespace in NAMESPACES}
    for name, path in kernels(root).items():
        source = read(path)
        for namespace in NAMESPACES:
            prefix = _prefix(source, namespace)
            for api in used_names(source.tree, prefix):
                index[namespace].setdefault(api, set()).add(name)
    return {
        namespace: {api: sorted(examples) for api, examples in sorted(apis.items())}
        for namespace, apis in index.items()
    }
