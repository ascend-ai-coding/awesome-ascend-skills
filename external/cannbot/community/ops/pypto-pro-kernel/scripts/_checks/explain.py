# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bind runtime symptoms and changed work sites to source without executing it.

The report describes syntax and lexical loop context, not a lowered schedule,
dynamic call count, memory-traffic estimate or proof of a hardware mechanism.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from pathlib import Path

from .source import attribute_path, read


_WORK = {
    "load",
    "store",
    "move",
    "insert",
    "extract",
    "gemm",
    "gemm_mx",
    "load_align",
    "store_align",
    "reduce_sum",
    "reduce_max",
    "sum",
    "cast",
    "astype",
    "sync_all",
    "bar_all",
}
_ALLOCATIONS = {"empty", "zeros", "ones", "full", "empty_like", "zeros_like"}
_RANK = re.compile(
    r"Tensor parameter ['\"]([^'\"]+)['\"] rank mismatch: expected (\d+), got (\d+)"
)


def inspect(path: Path, function: str | None = None) -> dict:
    source = read(path)
    contracts, allocations, work, tiles = [], [], [], []
    torch_aliases = {
        alias.asname or alias.name
        for node in ast.walk(source.tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "torch"
    }

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.functions = []
            self.loops = []
            self.conditions = []
            self.sections = []

        def selected(self):
            return (
                function is None
                or function in self.functions
                or function == ".".join(self.functions)
            )

        def location(self, node):
            return {
                "path": str(path.resolve()),
                "line": node.lineno,
                "function": ".".join(self.functions) or "<module>",
            }

        def visit_FunctionDef(self, node):
            self.functions.append(node.name)
            old_loops, old_conditions, old_sections = (
                self.loops,
                self.conditions,
                self.sections,
            )
            self.loops, self.conditions, self.sections = [], [], []
            if self.selected():
                for arg in (
                    node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                ):
                    ann = arg.annotation
                    if (
                        not isinstance(ann, ast.Subscript)
                        or attribute_path(ann.value) != f"{source.pl}.Tensor"
                    ):
                        continue
                    parts = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else []
                    dims = (
                        parts[0].elts
                        if parts and isinstance(parts[0], (ast.List, ast.Tuple))
                        else None
                    )
                    contracts.append(
                        {
                            **self.location(arg),
                            "parameter": arg.arg,
                            "annotation": ast.unparse(ann),
                            "rank": len(dims) if dims is not None else None,
                            "dimensions": [ast.unparse(d) for d in dims]
                            if dims is not None
                            else None,
                        }
                    )
            for stmt in node.body:
                self.visit(stmt)
            self.loops, self.conditions, self.sections = (
                old_loops,
                old_conditions,
                old_sections,
            )
            self.functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_With(self, node):
            previous = self.sections
            self.sections = previous + [
                ast.unparse(item.context_expr) for item in node.items
            ]
            for stmt in node.body:
                self.visit(stmt)
            self.sections = previous

        def visit_For(self, node):
            self.visit(node.iter)
            self.loops.append(
                {
                    "target": ast.unparse(node.target),
                    "iterator": ast.unparse(node.iter),
                    "line": node.lineno,
                }
            )
            for stmt in node.body:
                self.visit(stmt)
            self.loops.pop()
            for stmt in node.orelse:
                self.visit(stmt)

        def visit_While(self, node):
            self.loops.append(
                {"condition": ast.unparse(node.test), "line": node.lineno}
            )
            for stmt in node.body:
                self.visit(stmt)
            self.loops.pop()
            for stmt in node.orelse:
                self.visit(stmt)

        def visit_If(self, node):
            self.visit(node.test)
            self.conditions.append(ast.unparse(node.test))
            for stmt in node.body:
                self.visit(stmt)
            self.conditions[-1] = "not (" + ast.unparse(node.test) + ")"
            for stmt in node.orelse:
                self.visit(stmt)
            self.conditions.pop()

        def visit_Assign(self, node):
            call = node.value
            name = attribute_path(call.func) if isinstance(call, ast.Call) else None
            if (
                self.selected()
                and name
                and name.rsplit(".", 1)[0] in torch_aliases
                and name.rsplit(".", 1)[-1] in _ALLOCATIONS
            ):
                shape = (
                    call.args[0]
                    if call.args
                    else next(
                        (k.value for k in call.keywords if k.arg in ("size", "shape")),
                        None,
                    )
                )
                dims = shape.elts if isinstance(shape, (ast.List, ast.Tuple)) else None
                allocations.append(
                    {
                        **self.location(node),
                        "targets": [ast.unparse(t) for t in node.targets],
                        "call": ast.unparse(call),
                        "rank": len(dims) if dims is not None else None,
                        "dimensions": [ast.unparse(d) for d in dims]
                        if dims is not None
                        else None,
                    }
                )
            self.generic_visit(node)

        def visit_Call(self, node):
            name = attribute_path(node.func)
            if self.selected() and name:
                prefix, _, leaf = name.rpartition(".")
                native = prefix in (source.pl, source.vf) and leaf in _WORK
                initialize = prefix in torch_aliases and leaf in _ALLOCATIONS
                if native or initialize:
                    work.append(
                        {
                            **self.location(node),
                            "api": name,
                            "call": ast.unparse(node),
                            "loops": list(self.loops),
                            "conditions": list(self.conditions),
                            "sections": list(self.sections),
                            "dynamic_invocations": None,
                            "transferred_bytes": None,
                        }
                    )
                if name == f"{source.pl}.TileType":
                    tiles.append(
                        {**self.location(node), "declaration": ast.unparse(node)}
                    )
            self.generic_visit(node)

    Visitor().visit(source.tree)
    bindings = {}
    for node in source.tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings.setdefault(target.id, []).append(
                        {"line": node.lineno, "value": ast.unparse(node.value)}
                    )
    return {
        "source": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameters": contracts,
        "host_allocations": allocations,
        "tile_types": tiles,
        "work_sites": work,
        "module_bindings": bindings,
    }


def _signature(site):
    loops = tuple(
        tuple((k, v) for k, v in loop.items() if k != "line") for loop in site["loops"]
    )
    return (
        site["function"],
        site["call"],
        loops,
        tuple(site["conditions"]),
        tuple(site["sections"]),
    )


def rank_diagnostics(current, text):
    symptoms = []
    for message in dict.fromkeys(m.group(0) for m in _RANK.finditer(text)):
        match = _RANK.search(message)
        parameter, expected, actual = (
            match.group(1),
            int(match.group(2)),
            int(match.group(3)),
        )
        symptoms.append(
            {
                "message": message,
                "parameter": parameter,
                "expected_rank": expected,
                "actual_rank": actual,
                "declarations": [
                    p for p in current["parameters"] if p["parameter"] == parameter
                ],
                "same_name_allocations": [
                    a for a in current["host_allocations"] if parameter in a["targets"]
                ],
                "next_check": "Compare the selected callable declaration with the actual argument at its launch. "
                "A stale declaration explains a rank mismatch without requiring a cache hypothesis; "
                "a cache claim needs an isolated unchanged-source control.",
            }
        )
    return symptoms


def report(
    path: Path,
    *,
    baseline: Path | None = None,
    error_log: Path | None = None,
    function: str | None = None,
) -> dict:
    current = inspect(path, function)
    result = {
        "current": current,
        "runtime_symptoms": [],
        "comparison": None,
        "limits": [
            "Static sites are not dynamic invocation counts. Resolve workload, loop bounds, tile extents, "
            "padding and worker ownership before counting bytes.",
            "Unknown work is null, never zero. Source order does not establish pipe ordering, cache hits "
            "or bandwidth saturation.",
            "Name-matched allocations are inspection leads, not a proven interprocedural binding. Helpers, "
            "alias shadowing and lowered specialization require native evidence.",
        ],
        "next_evidence": {
            "source_contract": "references/workflows/debugging.md#first-moves-before-any-theory",
            "mechanism": "references/diagnostics/performance.md#test-a-mechanism-with-source-bound-evidence",
            "counters": "references/verification/native-timing.md#collect-pipe-counters-for-diagnosis",
        },
    }
    if error_log:
        data = error_log.read_bytes()
        result["error_log"] = {
            "path": str(error_log.resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        result["runtime_symptoms"] = rank_diagnostics(
            current, data.decode(errors="replace")
        )
        if not result["runtime_symptoms"]:
            result["runtime_symptoms"].append(
                {
                    "status": "unrecognized",
                    "next_check": "Use the exact error and source location in the debugging owner; "
                    "no cause is inferred.",
                }
            )
    if baseline:
        before = inspect(baseline, function)
        old, new = (
            Counter(map(_signature, before["work_sites"])),
            Counter(map(_signature, current["work_sites"])),
        )
        result["baseline"] = {
            k: before[k]
            for k in (
                "source",
                "sha256",
                "parameters",
                "host_allocations",
                "tile_types",
            )
        }
        result["comparison"] = {
            "changed_module_bindings": {
                name: {
                    "before": before["module_bindings"].get(name),
                    "after": current["module_bindings"].get(name),
                }
                for name in sorted(
                    before["module_bindings"].keys() | current["module_bindings"].keys()
                )
                if [v["value"] for v in before["module_bindings"].get(name, [])]
                != [v["value"] for v in current["module_bindings"].get(name, [])]
            },
            "removed_or_changed_work_sites": [
                s
                for s in before["work_sites"]
                if _signature(s) not in new or old[_signature(s)] != new[_signature(s)]
            ],
            "added_or_changed_work_sites": [
                s
                for s in current["work_sites"]
                if _signature(s) not in old or old[_signature(s)] != new[_signature(s)]
            ],
            "unchanged_static_sites": sum((old & new).values()),
            "claim_scope": "These are lexical differences. Equal load sites can still transfer different bytes "
            "when shapes or repetitions change; added initialization and stores must be included in "
            "complete-call measurement.",
        }
    return {
        key: result[key]
        for key in (
            "runtime_symptoms",
            "comparison",
            "limits",
            "next_evidence",
            "error_log",
            "baseline",
            "current",
        )
        if key in result
    }
