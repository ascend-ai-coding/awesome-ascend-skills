# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Advisory reviews of connected conversion/reduction consumers, without importing kernels.

Only straight-line local bindings are propagated. Lane provenance is not a
numerical proof; unknown masks, helpers and control flow retain explicit limits.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from . import api_policy
from .paths import PYPTO_ROOT
from .source import Finding, read, vector_functions

REDUCTION_RULE = "vf-reduction-consumer-lanes"
CHAIN_RULE = "vf-conversion-reduction-chain"
WIDENING_RULE = "vf-unpk-widening-mask"
REDUCTIONS = {"reduce_sum", "reduce_min", "reduce_max"}
BINARY = {"add", "sub", "mul", "div", "max", "min"}


@dataclass(frozen=True)
class Value:
    role: str = "unknown"
    sites: tuple = ()
    conversions: tuple = ()
    widening: tuple = ()


@dataclass(frozen=True)
class PrefixMask:
    bits: int
    count: object


def inspect(path: Path, *, pypto_root=PYPTO_ROOT):
    source = read(path)
    calls = []
    api_policy.inspect(path, pypto_root=pypto_root, resolved_calls=calls)
    operations = {
        (node.lineno, node.col_offset): name[len(api_policy.VF_PREFIX):]
        for node, name in calls
        if name.startswith(api_policy.VF_PREFIX)
    }
    findings, reviews = [], []
    coverage = {
        "checked_consumers": 0,
        "unverified_calls": 0,
        "limits": (
            "Straight-line local VF bindings only; no helper, branch, loop, dtype,"
            " physical-footprint or numerical proof."
        ),
    }

    def keyword(call, name, default=None, position=None):
        for item in call.keywords:
            if item.arg == name:
                return item.value
        return (
            call.args[position]
            if position is not None and len(call.args) > position
            else default
        )

    def spelling(node):
        return ast.unparse(node) if node is not None else None

    # Resolve only unambiguous module dtype aliases; local/helper-derived types
    # remain unknown. Do not infer dtype from a register's variable name.
    dtype_aliases = {}
    writes = {}
    for node in ast.walk(source.tree):
        names = []
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names = [node.id]
        elif isinstance(node, ast.arg):
            names = [node.arg]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.asname or alias.name.split(".")[0] for alias in node.names]
        for name in names:
            writes[name] = writes.get(name, 0) + 1
    for statement in source.tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and writes.get(target.id) == 1:
                dtype_aliases[target.id] = spelling(statement.value)

    def inspect_function(function):
        values, masks, prefixes, versions, counts = {}, {}, {}, {}, {}
        seen_reviews, seen_widening = set(), set()
        locals_ = {
            arg.arg
            for arg in function.args.posonlyargs
            + function.args.args
            + function.args.kwonlyargs
        }
        locals_.update(
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )

        def selector(node, group, member):
            if (
                source.pl is None
                or source.pl in locals_
                or writes.get(source.pl.split(".")[0]) != 1
            ):
                return False
            name = spelling(node)
            if isinstance(node, ast.Name) and node.id not in locals_:
                name = dtype_aliases.get(node.id)
            return name == f"{source.pl}.{group}{member}"

        def count(node):
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                operand = count(node.operand)
                return -operand if type(operand) is int else None
            if isinstance(node, ast.Name):
                return counts.get(node.id, (node.id, versions.get(node.id, 0)))
            return None

        def prefix(node):
            if isinstance(node, ast.Name):
                return prefixes.get(node.id)
            if not isinstance(node, ast.Call):
                return None
            operation = operations.get((node.lineno, node.col_offset))
            bits = next(
                (
                    bits
                    for bits in (16, 32)
                    if selector(keyword(node, "dtype"), "DT_FP", str(bits))
                ),
                None,
            )
            if bits is None or any(item.arg is None for item in node.keywords):
                return None
            if operation == "update_mask" and node.args:
                return PrefixMask(bits, count(node.args[0]))
            if operation == "create_mask":
                pattern = keyword(node, "pattern")
                if selector(pattern, "MaskPattern.", "ALL"):
                    return PrefixMask(bits, 2048 // bits)
                for lanes in (1, 2, 4, 8, 16, 32, 64, 128):
                    if selector(pattern, "MaskPattern.", f"VL{lanes}"):
                        return PrefixMask(bits, lanes)
            return None

        def review_widening(node, operation, args):
            # These consumers read the corresponding FP32 lanes directly.
            if operation in BINARY | {"adds", "muls", "maxs", "mins", "store_align"}:
                predicate = keyword(node, "preg", position=2)
                inputs = args[1:2] if operation == "store_align" else args[:2]
            elif operation in REDUCTIONS | {"abs", "exp", "ln", "sqrt", "rec", "neg"}:
                predicate = keyword(node, "preg", position=1)
                inputs = args[:1]
            else:
                return
            consumer = prefix(predicate)
            if consumer is None or consumer.bits != 32 or consumer.count is None:
                return
            for incoming in inputs:
                if not incoming.widening:
                    continue
                cast_line, conversion = incoming.widening
                if cast_line in seen_widening or conversion.count is None:
                    continue
                a, b = conversion.count, consumer.count
                if type(a) is int and type(b) is int:
                    covered = (min(a % (1 << 32), 128) + 1) // 2
                    needed = min(b % (1 << 32), 64)
                    review = covered < needed
                    bounds = {"converted_lanes": covered, "consumer_lanes": needed}
                else:
                    review = a == b
                    bounds = {"converted_lanes": None, "consumer_lanes": None}
                if not review:
                    continue
                seen_widening.add(cast_line)
                coverage["checked_consumers"] += 1
                findings.append(
                    Finding(
                        source.path,
                        node.lineno,
                        WIDENING_RULE,
                        f"FP16 UNPK → FP32 ZERO conversion at line {cast_line} uses a"
                        " b16 prefix that may cover fewer logical values than this b32"
                        " consumer. The same unsaturated count n>1 converts only"
                        " ceil(n/2) values. If all consumer lanes need converted data,"
                        " reuse its b32 predicate or derive a b16 predicate covering"
                        " the selected even halfwords. Full/saturated predicates and"
                        " intentionally unused lanes are legal; verify actual needed"
                        " lanes and the caller's FP16 source at the widening contract before changing code.",
                        advisory=True,
                    )
                )
                reviews.append(
                    {
                        "function": function.name,
                        "consumer_line": node.lineno,
                        "sites": [
                            {"line": line, "operation": op}
                            for line, op in (*incoming.sites, (node.lineno, operation))
                        ][-12:],
                        "mask": spelling(predicate),
                        **bounds,
                        "rules": [WIDENING_RULE],
                    }
                )

        def value(node):
            return (
                values.get(node.id, Value()) if isinstance(node, ast.Name) else Value()
            )

        def evaluate(node):
            if isinstance(node, ast.Name):
                return value(node)
            if (
                not isinstance(node, ast.Call)
                or (node.lineno, node.col_offset) not in operations
            ):
                return Value()
            operation = operations[(node.lineno, node.col_offset)]
            args = [value(arg) for arg in node.args]
            if not any(item.arg is None for item in node.keywords):
                review_widening(node, operation, args)
            if operation == "load_align":
                dist = spelling(keyword(node, "dist"))
                return Value(
                    "vector",
                    ((node.lineno, operation),),
                    ((node.lineno, "compact_or_default_load"),)
                    if dist is None or dist.endswith(".NORM")
                    else ((node.lineno, "unpk16"),)
                    if selector(keyword(node, "dist"), "LoadDist.", "UNPK")
                    else (),
                )
            if operation in REDUCTIONS:
                if any(item.arg is None for item in node.keywords):
                    coverage["unverified_calls"] += 1
                    return Value()
                mode = keyword(node, "datablock", ast.Constant(False), position=2)
                if not isinstance(mode, ast.Constant) or mode.value is not False:
                    coverage["unverified_calls"] += 1
                    return Value()
                incoming = args[0] if args else Value()
                return Value(
                    "reduced",
                    (*incoming.sites, (node.lineno, operation)),
                    incoming.conversions,
                )
            if operation == "astype" and args:
                dtype = spelling(keyword(node, "dtype"))
                layout = spelling(keyword(node, "layout"))
                current = args[0]
                conversions = current.conversions
                if (
                    dtype
                    and dtype.endswith(".DT_FP32")
                    and (layout is None or layout.endswith(".ZERO"))
                ):
                    conversions = (*conversions, (node.lineno, "to_fp32_zero"))
                cast_mask = prefix(keyword(node, "preg", position=1))
                widening = ()
                if (
                    current.sites
                    and current.sites[-1][1] == "load_align"
                    and any(kind == "unpk16" for _, kind in current.conversions)
                    and selector(keyword(node, "dtype"), "DT_FP", "32")
                    and selector(keyword(node, "layout"), "CastLayout.", "ZERO")
                    and cast_mask is not None
                    and cast_mask.bits == 16
                    and not any(item.arg is None for item in node.keywords)
                ):
                    widening = (node.lineno, cast_mask)
                return Value(
                    current.role,
                    (*current.sites, (node.lineno, operation)),
                    conversions,
                    widening,
                )
            if operation == "full":
                return Value("vector", ((node.lineno, operation),))
            if operation == "move" and args:
                return args[0]
            if operation in BINARY and len(args) >= 2:
                left, right = args[:2]
                reduced = [v for v in (left, right) if v.role == "reduced"]
                if len(reduced) == 2:
                    return Value(
                        "reduced",
                        tuple(dict.fromkeys((*left.sites, *right.sites))),
                        tuple(dict.fromkeys((*left.conversions, *right.conversions))),
                    )
                if reduced:
                    predicate = (
                        node.args[2] if len(node.args) > 2 else keyword(node, "preg")
                    )
                    one_lane = isinstance(predicate, ast.Name) and masks.get(
                        predicate.id
                    ) in (0, 1)
                    coverage["checked_consumers"] += 1
                    if not one_lane:
                        origin = reduced[0]
                        if origin.sites in seen_reviews:
                            return Value("vector", ((node.lineno, operation),))
                        seen_reviews.add(origin.sites)
                        sites = list(
                            dict.fromkeys((*origin.sites, (node.lineno, operation)))
                        )
                        evidence = {
                            "function": function.name,
                            "consumer_line": node.lineno,
                            "sites": [
                                {"line": line, "operation": op} for line, op in sites
                            ][-12:],
                            "mask": spelling(predicate),
                            "mask_lanes": None,
                            "rules": [REDUCTION_RULE],
                        }
                        findings.append(
                            Finding(
                                source.path,
                                node.lineno,
                                REDUCTION_RULE,
                                "A non-datablock reduction reaches lane-wise arithmetic"
                                " without a visible broadcast. If this consumer needs"
                                " the row scalar in multiple lanes, bind the reduction"
                                " result then use vf.full; a lane-0-only consumer is"
                                " legal. Review the complete connected chain before another device request.",
                                advisory=True,
                            )
                        )
                        kinds = {kind for _, kind in origin.conversions}
                        if {"compact_or_default_load", "to_fp32_zero"} <= kinds:
                            evidence["rules"].append(CHAIN_RULE)
                            evidence["conversion_sites"] = [
                                {"line": line, "kind": kind}
                                for line, kind in origin.conversions
                            ][-8:]
                            findings.append(
                                Finding(
                                    source.path,
                                    node.lineno,
                                    CHAIN_RULE,
                                    "The same reduction consumer depends on a"
                                    " default/compact load and FP32 ZERO cast. Verify"
                                    " the caller's source dtype and intended element"
                                    " order: for contiguous FP16 widening, check UNPK"
                                    " and physical predicate coverage as well as"
                                    " reduction broadcasting. FP32 input or intentional"
                                    " half-element selection is not a widening defect.",
                                    advisory=True,
                                )
                            )
                        reviews.append(evidence)
                return Value("vector", ((node.lineno, operation),))
            return Value()

        for statement in function.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                name, node = statement.targets[0].id, statement.value
                result, mask_prefix, scalar_count = (
                    evaluate(node),
                    prefix(node),
                    count(node),
                )
                values[name] = result
                prefixes.pop(name, None)
                if mask_prefix is not None:
                    prefixes[name] = mask_prefix
                versions[name] = versions.get(name, 0) + 1
                counts.pop(name, None)
                if scalar_count is not None:
                    counts[name] = scalar_count
                masks.pop(name, None)
                operation = operations.get(
                    (getattr(node, "lineno", -1), getattr(node, "col_offset", -1))
                )
                if (
                    operation == "update_mask"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    if node.args[0].value in (0, 1):
                        masks[name] = node.args[0].value
                elif operation == "create_mask":
                    pattern = spelling(keyword(node, "pattern"))
                    if pattern and pattern.endswith(".VL1"):
                        masks[name] = 1
                elif isinstance(node, ast.Name) and node.id in masks:
                    masks[name] = masks[node.id]
            elif isinstance(statement, ast.Expr):
                evaluate(statement.value)
            elif not isinstance(statement, (ast.Pass, ast.Return)):
                # A write through control flow can replace any incoming binding.
                assigned = {
                    n.id
                    for n in ast.walk(statement)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                }
                for name in assigned:
                    values.pop(name, None)
                    masks.pop(name, None)
                    prefixes.pop(name, None)
                    counts.pop(name, None)
                    versions[name] = versions.get(name, 0) + 1
                coverage["unverified_calls"] += sum(
                    (getattr(n, "lineno", -1), getattr(n, "col_offset", -1))
                    in operations
                    for n in ast.walk(statement)
                    if isinstance(n, ast.Call)
                )

    for candidate in vector_functions(source):
        inspect_function(candidate)
    return findings, reviews, coverage


def check(path: Path, *, pypto_root=PYPTO_ROOT):
    return inspect(path, pypto_root=pypto_root)[0]
