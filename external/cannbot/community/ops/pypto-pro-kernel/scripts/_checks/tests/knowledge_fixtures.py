# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Small complete knowledge sources shared by admission and transport tests."""

from pathlib import Path
from types import SimpleNamespace

from _checks import api_stub


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_corpus(base, *, references=None):
    base = Path(base)
    framework, examples = base / "framework", base / "kit/examples"
    references = references or base / "kit/references"
    write(
        framework / "python/pypto_pro/language/__init__.py",
        """raise RuntimeError("must not import framework")
__all__ = ["move", "insert"]
from ._api import move, insert
""",
    )
    api_source = write(
        framework / "python/pypto_pro/language/_api.py",
        '''raise RuntimeError("must not import declarations")
from typing import Any, List, Union
Tile = Any
TileGroup = Any
DType = Any
Offset = Union[List[int], int]
def move(dst, src):
    """Move source data."""
    return dst
def insert(dst, src, offset):
    """Insert source data at an offset."""
    return dst
''',
    )
    write(
        framework / "docs/zh/api/pro_api/move.md",
        "# pypto_pro.language.move\n\nMove source data.\n",
    )
    write(
        framework / "docs/zh/guide/programming_guide/pro/start.md",
        "# Guide\n\nCall pl.move for this source example.\n",
    )
    write(
        framework / "docs/zh/guide/quick_start/pro/start.md",
        "# Quick start\n\nRead the original test scope.\n",
    )
    for relative in (
        "basic_api/chapter/copy",
        "matmul/product",
        "algorithms/composition",
    ):
        folder = examples / relative
        write(folder / "__init__.py", "")
        write(
            folder / "kernel.py",
            '''"""写法：原始。A fabricated source-only teaching program."""
import pypto_pro.language as pl
def entry(x):
    return pl.move(x, x)
raise RuntimeError("must not import kernel")
''',
        )
        write(
            folder / "reference.py",
            '''"""Independent mathematical entry; no acceptance is claimed."""
def reference(x):
    return sum(x)
raise RuntimeError("must not import reference")
''',
        )
        write(
            folder / "test.py",
            '''"""The original test entry, not a stored pass result."""
from .kernel import entry
from .reference import reference
raise RuntimeError("must not import tests")
''',
        )
    write(examples / "__init__.py", "")
    write(
        examples / "conftest.py",
        '"""Shared device setup must be checked in the real environment."""\n',
    )
    write(examples / "pytest.ini", "[pytest]\npython_files = test.py\n")
    owner = write(
        references / "owner.md",
        """<!-- kernel-query: {"id":"copy","aliases":["copy route"],"symbols":["pl.move"],"example":"basic_api/chapter/copy/kernel.py","facets":{"source_memory":["Vec"]}} -->
## Copy contract
Read the source layout and independent test before adapting.
""",
    )
    catalog = write(
        references / "tooling/examples.md",
        """# Selection
| Mechanism | Example |
| --- | --- |
| Product | [product](../../examples/matmul/product/) |
| Composition | [composition](../../examples/algorithms/composition/) |
""",
    )
    write(references / "tooling/example-coverage.md", "### 八点二、0 项逐项\n")
    write(references / "api/pypto_pro_language.pyi", api_stub.generate(framework))
    return SimpleNamespace(
        framework=framework,
        examples=examples,
        references=references,
        owner=owner,
        catalog=catalog,
        api_source=api_source,
    )
