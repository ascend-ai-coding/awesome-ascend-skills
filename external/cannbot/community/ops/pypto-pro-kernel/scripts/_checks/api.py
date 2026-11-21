# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The `api` subcommand computes and prints a signature table from declaration stubs.

It parses the three declaration stubs in the PyPTO source repository, `python/pypto_pro/language/` (`_api.py`,
`_vf_api.py`, `_simt_api.py`), and computes two lines per public API: one signature line and one first-docstring-line
entry. The default report writes no files and imports no framework code. The separate `api_stub` generator handles `api
--write-pyi` and `api --check-pyi` for the agent-readable reference.

**The stubs describe the declared surface, not the entire interface surface**: the parser dispatch table is wider than
the stubs (`pl.muls` once had a board-run example but no declaration). This table answers “what exists and what does it
look like”; absence does not prove nonexistence—see the absence note in the skill primer.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .paths import PYPTO_ROOT

STUBS: dict[str, Path] = {
    "pl": PYPTO_ROOT / "python/pypto_pro/language/_api.py",
    "vf": PYPTO_ROOT / "python/pypto_pro/language/_vf_api.py",
    "pl.simt": PYPTO_ROOT / "python/pypto_pro/language/_simt_api.py",
}


def _signature(fn: ast.FunctionDef) -> str:
    """Return one-line `def name(args) -> ret:` text; remove `self` for methods."""
    args = fn.args
    if args.args and args.args[0].arg == "self":
        args = ast.arguments(
            posonlyargs=args.posonlyargs,
            args=args.args[1:],
            vararg=args.vararg,
            kwonlyargs=args.kwonlyargs,
            kw_defaults=args.kw_defaults,
            kwarg=args.kwarg,
            defaults=args.defaults[slice(-len(args.args[1:]), None)]
            if args.defaults
            else [],
        )
    stub = ast.FunctionDef(
        name=fn.name,
        args=args,
        body=[ast.Pass()],
        decorator_list=[],
        returns=fn.returns,
        type_params=[],
    )
    module = ast.Module(body=[stub], type_ignores=[])
    text = ast.unparse(ast.fix_missing_locations(module))
    head = text.split("\n    pass")[0].strip()
    return head.removeprefix("def ").removesuffix(":")


def declarations(path: Path) -> list[tuple[str, str]]:
    """Return `[(signature, first docstring line)]` in source order; skip names prefixed with `_`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rows: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        doc = ast.get_docstring(node, clean=False)
        first = doc.lstrip("\n").split("\n")[0].strip() if doc else ""
        rows.append((_signature(node), first))
    return rows


def report(
    namespaces: list[str] | None = None, *, pypto_root: Path = PYPTO_ROOT
) -> bool:
    chosen = namespaces or list(STUBS)
    unknown = [name for name in chosen if name not in STUBS]
    if unknown:
        print(f"api：未知命名空间 {'、'.join(unknown)}，可选：{'、'.join(STUBS)}")
        return False
    for name in chosen:
        path = pypto_root / "python/pypto_pro/language" / STUBS[name].name
        if not path.exists():
            print(f"api：{path} 不存在——请用 --pypto-root 指定 PyPTO 源码树")
            return False
        rows = declarations(path)
        print(f"# {name}.* —— {len(rows)} declared（{path}）")
        for signature, doc in rows:
            print(f"{name}.{signature}")
            if doc:
                print(f"    {doc}")
        print()
    return True
