# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Inspect manual TileGroup aliases without executing a kernel.

Compare overlapping slots within one memory space, using resolved shapes, byte
extents and explicit IDs. Unknown declarations and automatically assigned IDs
remain outside this rule's coverage. A finding establishes disjoint mutex IDs,
not an observed race or a complete lifetime proof.

Direct accesses in explicit Cube/Vector sections allow two bounded reviews:
separate same-core phases divided by unconditional bar_all, and accesses in
different core domains. Both remain advisories requiring native evidence.
Conditional/partial barriers, loops, escaped aliases and unresolved helpers do
not establish this phase separation. Cross-core ownership needs a real handover;
using equal per-core mutex IDs does not provide it.

Within a group, compare slots individually. Whole-group ID intersection can
hide two overlapping slots protected by different IDs. Constant evaluation
supports the source forms below; measure declaration coverage when extending it
and never turn an unresolved layout into a successful alias proof.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .source import (
    Call,
    Finding,
    Source,
    apply_claim,
    attribute_path,
    auto_mutex_off,
    calls,
    counterexample_claims,
    has_decorator,
    read,
)

RULE = "overlapping-groups-need-shared-mutex"

MAKE_TILE_GROUP = "make_tile_group"
TILE_TYPE = "TileType"

DTYPE_BITS = {
    # Mirrored from C++ `DataType::GetBit()` (`framework/include/core/dtype.h`); keys omit the `DT_` prefix.
    # **`BOOL` is 8 bits, not 1.** Four-bit types are rounded by `ceil(bits/8)` to 1 byte/element—the framework uses the
    # same slot spacing, so match it here.
    "BOOL": 8,
    "HF4": 4,
    "FP4": 4,
    "FP4E2M1": 4,
    "FP4E1M2": 4,
    "UINT4": 4,
    "INT4": 4,
    "HF8": 8,
    "FP8E4M3FN": 8,
    "FP8E5M2": 8,
    "FP8E8M0": 8,
    "UINT8": 8,
    "INT8": 8,
    "BF16": 16,
    "FP16": 16,
    "UINT16": 16,
    "INT16": 16,
    "FP32": 32,
    "UINT32": 32,
    "INT32": 32,
    "UINT64": 64,
    "INT64": 64,
    "FP64": 64,
    "INDEX": 64,
}


@dataclass(frozen=True)
class Group:
    """One successfully parsed `make_tile_group`; `slots` and `ids` correspond one-to-one and have equal length."""

    call: Call
    memory: str
    slots: tuple[tuple[int, int], ...]
    ids: tuple[int, ...]


def _kwarg(call: ast.Call, name: str) -> Optional[ast.expr]:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _constants(source: Source) -> dict[str, object]:
    """Return module-level `NAME = <constant>` bindings: expressions reducible to int and aliases such as `FP32 =
    pl.DT_FP32`.

    Inspect module-level assignments only. Do **not** follow a same-named variable rebound inside a function; leave that
    form unresolved and skip it rather than following the wrong value.
    """
    env: dict[str, object] = {}
    for node in source.tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        # `TM, TN = 16, 64` is common in examples; without it, the whole group fails to parse (measured:
        # all six `entry_and_loop` groups depend on these TM/TN values).
        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(node.value, (ast.Tuple, ast.List))
            and len(target.elts) == len(node.value.elts)
        ):
            for name, value_node in zip(target.elts, node.value.elts):
                if isinstance(name, ast.Name):
                    _bind(env, name.id, value_node, source)
            continue
        if isinstance(target, ast.Name):
            _bind(env, target.id, node.value, source)
    return env


def _bind(env: dict[str, object], name: str, node: ast.expr, source: Source) -> None:
    """Add a binding to the constant environment; accept dtype aliases, integers, and **literal dicts**.

    Dicts are needed because examples often use `_ADDR = {"a": 0x0, "b": 0x2000}` with `addrs=[_ADDR["a"]]` (`vf_arith`
    alone has eight such cases). Without dicts, none of those groups can be parsed.
    """
    path = attribute_path(node)
    if path is not None and path.startswith(f"{source.pl}.DT_"):
        env[name] = path.split(".DT_")[1]
        return
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_as_int(item, env) for item in node.elts]
        if items and all(i is not None for i in items):
            env[name] = items
        return
    if isinstance(node, ast.ListComp):
        items = _fold_range_comp(node, env)
        if items is not None:
            env[name] = items
        return
    if isinstance(node, ast.Dict):
        table = {}
        for key, value in zip(node.keys, node.values):
            folded = _as_int(value, env)
            if isinstance(key, ast.Constant) and folded is not None:
                table[key.value] = folded
        if table:
            env[name] = table
        return
    value = _as_int(node, env)
    if value is not None:
        env[name] = value


def _fold_range_comp(node: ast.ListComp, env: dict[str, object]) -> Optional[list[int]]:
    """Fold only this form: `[<expression> for <name> in range(<constant>)]`.

    Example address tables often use this form (`_O = [_A + TILE_BYTES * (i + 1) for i in range(5)]`). **Deliberately do
    not use `eval`**: this is a gate and must not execute expressions from the checked file. An extra `if`, a condition,
    or a nested comprehension always returns None.
    """
    if len(node.generators) != 1:
        return None
    comp = node.generators[0]
    if comp.ifs or comp.is_async or not isinstance(comp.target, ast.Name):
        return None
    if not (
        isinstance(comp.iter, ast.Call)
        and isinstance(comp.iter.func, ast.Name)
        and comp.iter.func.id == "range"
        and len(comp.iter.args) == 1
    ):
        return None
    count = _as_int(comp.iter.args[0], env)
    if count is None or not 0 < count <= 1024:
        return None
    values = []
    for index in range(count):
        value = _as_int(node.elt, {**env, comp.target.id: index})
        if value is None:
            return None
        values.append(value)
    return values


def _as_int(node: ast.expr, env: dict[str, object]) -> Optional[int]:
    """Reduce a node to int; return None if it cannot be reduced. Do not guess."""
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.Name):
        value = env.get(node.id)
        return value if isinstance(value, int) else None
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Constant)
    ):
        table = env.get(node.value.id)
        key = node.slice.value
        if isinstance(table, dict):
            value = table.get(key)
            return value if isinstance(value, int) else None
        if (
            isinstance(table, list)
            and isinstance(key, int)
            and -len(table) <= key < len(table)
        ):
            return table[key]
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _as_int(node.operand, env)
        return None if inner is None else -inner
    if isinstance(node, ast.BinOp):
        left, right = _as_int(node.left, env), _as_int(node.right, env)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv) and right != 0:
            return left // right
        if isinstance(node.op, ast.LShift):
            return left << right
    return None


def _int_list(node: Optional[ast.expr], env: dict[str, object]) -> Optional[list[int]]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values = [_as_int(item, env) for item in node.elts]
    return (
        None if any(v is None for v in values) else [v for v in values if v is not None]
    )


def _dtype_bits(node: Optional[ast.expr], env: dict[str, object]) -> Optional[int]:
    """Map `pl.DT_FP32` or a module-level alias to bit width; return None for unknown dtypes (skip rather than
    guess)."""
    if node is None:
        return None
    path = attribute_path(node)
    if path is not None and ".DT_" in path:
        return DTYPE_BITS.get(path.split(".DT_")[1])
    if isinstance(node, ast.Name):
        alias = env.get(node.id)
        return DTYPE_BITS.get(alias) if isinstance(alias, str) else None
    return None


def _tile_types(function: ast.AST, source: Source) -> dict[str, ast.Call]:
    """Find local bindings of the form `tt = pl.TileType(...)` inside a function.

    **Without this layer, the rule would miss most real cases**: in measured examples, a majority of 351
    `make_tile_group` calls use `type=tt`, binding TileType first (six groups sharing one type in a kernel is common).
    Follow one level only and accept only names that are **not reassigned**; when the same name is rebound, skip the
    group rather than risk using the wrong value.
    """
    seen: dict[str, ast.Call] = {}
    rebound: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in seen or target.id in rebound:
            rebound.add(target.id)
            seen.pop(target.id, None)
            continue
        if isinstance(node.value, ast.Call):
            path = attribute_path(node.value.func)
            is_helper = isinstance(
                node.value.func, ast.Name
            ) and node.value.func.id in _type_helpers(source)
            if path == f"{source.pl}.{TILE_TYPE}" or is_helper:
                seen[target.id] = node.value
    return seen


def _type_helpers(source: Source) -> dict[str, ast.FunctionDef]:
    """Find module-level `def _tile_type(...): return pl.TileType(...)` helpers with a single return; reject other
    forms.

    **This is the dominant form in examples**; without it, the rule would miss many groups. Of 351 measured
    `make_tile_group` calls, 221 `type=` arguments use this kind of helper call (the `init_output` rule hit the same
    form).
    """
    found = {}
    for node in source.tree.body:
        if not isinstance(node, ast.FunctionDef) or len(node.body) != 1:
            continue
        stmt = node.body[0]
        if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.Call):
            continue
        if attribute_path(stmt.value.func) == f"{source.pl}.{TILE_TYPE}":
            found[node.name] = node
    return found


def _resolve_type(
    node: Optional[ast.expr],
    source: Source,
    types: dict[str, ast.Call],
    helpers: dict[str, ast.FunctionDef],
    env: dict[str, object],
) -> tuple[Optional[ast.Call], dict[str, object]]:
    """Return `(TileType call, environment needed to evaluate that call)`.

    Three supported forms: literal `pl.TileType(...)`, local name `tt`, and helper call `_tile_type(FP32)`. Bind helper
    parameters to arguments so `dtype=dtype` inside `_tile_type(dtype)` can be resolved.
    """
    if isinstance(node, ast.Call):
        if attribute_path(node.func) == f"{source.pl}.{TILE_TYPE}":
            return node, env
        if isinstance(node.func, ast.Name) and node.func.id in helpers:
            helper = helpers[node.func.id]
            params = helper.args.args
            if node.keywords or len(node.args) > len(params):
                return None, env  # Keyword arguments or excess arguments: do not guess.
            # Omitted parameters use defaults. `def _ub(dtype=FP32, n=N)` with `_ub()` is common in examples;
            # without default handling, all groups in the `block_grid` / `dtype_carriers` batch fail to parse.
            supplied = list(node.args)
            defaults = helper.args.defaults
            missing = len(params) - len(supplied)
            if missing > len(defaults):
                return (
                    None,
                    env,
                )  # A missing parameter has no default: the call is invalid, so do not guess.
            supplied += (
                list(defaults[slice(len(defaults) - missing, None)]) if missing else []
            )
            bound = dict(env)
            for param, arg in zip(params, supplied):
                path = attribute_path(arg)
                if path is not None and path.startswith(f"{source.pl}.DT_"):
                    bound[param.arg] = path.split(".DT_")[1]
                elif isinstance(arg, ast.Name) and arg.id in env:
                    bound[param.arg] = env[arg.id]
                else:
                    value = _as_int(arg, env)
                    if value is None:
                        return None, env
                    bound[param.arg] = value
            return helper.body[0].value, bound  # type: ignore[union-attr]
        return None, env
    if isinstance(node, ast.Name):
        bound = types.get(node.id)
        if bound is None:
            return None, env
        # For `tt = _tile_type(FP32)`, the local binding is a **helper call** and needs one more resolution step.
        # The cost of omitting this step was measured: 48 `type=tt` cases all stopped here.
        return _resolve_type(bound, source, {}, helpers, env)
    return None, env


def _slot_size(
    type_node: Optional[ast.expr], source: Source, env: dict[str, object]
) -> Optional[int]:
    """Compute `prod(shape) * ceil(bits / 8)`, mirroring `_cross_core_scanner._tile_type_slot_size`."""
    if not isinstance(type_node, ast.Call):
        return None
    if attribute_path(type_node.func) != f"{source.pl}.{TILE_TYPE}":
        return None
    shape = _int_list(_kwarg(type_node, "shape"), env)
    bits = _dtype_bits(_kwarg(type_node, "dtype"), env)
    if shape is None or bits is None or not shape:
        return None
    elems = 1
    for dim in shape:
        elems *= dim
    return elems * ((bits + 7) // 8)


def _memory(type_node: ast.expr, source: Source) -> Optional[str]:
    """Convert `target_memory=pl.MemorySpace.Vec` to `"pl.MemorySpace.Vec"`.

    **Require a literal attribute chain.** Return None for a variable (skip it); guessing the wrong space can report two
    unrelated memory regions as overlapping, one of the worst false positives. Cross-core buffer `target_memory` also
    requires a literal (`references/api/constraints.md`), so this requirement is not unusually strict.
    """
    if not isinstance(type_node, ast.Call):
        return None
    return attribute_path(_kwarg(type_node, "target_memory") or ast.Pass())


def _group(
    call: Call,
    source: Source,
    env: dict[str, object],
    types: dict[str, ast.Call],
    helpers: dict[str, ast.FunctionDef],
) -> Optional[Group]:
    type_node, type_env = _resolve_type(
        _kwarg(call.node, "type"), source, types, helpers, env
    )
    if type_node is None:
        return None
    memory = _memory(type_node, source)
    size = _slot_size(type_node, source, type_env)
    ids = _int_list(_kwarg(call.node, "mutex_ids"), env)
    if memory is None or size is None or not ids:
        return None

    depth = _as_int(_kwarg(call.node, "depth") or ast.Pass(), env)
    count = depth if depth is not None else len(ids)
    if count != len(ids):
        return None  # The framework requires len(mutex_ids) == depth; do not guess when they disagree.

    addrs_node = _kwarg(call.node, "addrs")
    starts = _int_list(addrs_node, env)
    if starts is None:
        base = _as_int(addrs_node or ast.Pass(), env)
        if base is None:
            return None
        starts = [base + i * size for i in range(count)]
    if len(starts) != count:
        return None

    return Group(
        call=call,
        memory=memory,
        slots=tuple((start, start + size) for start in starts),
        ids=tuple(ids),
    )


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _groups(source: Source, function: ast.AST, env: dict[str, object]) -> list[Group]:
    types = _tile_types(function, source)
    helpers = _type_helpers(source)
    found = [
        _group(call, source, env, types, helpers)
        for call in calls(function, source.pl)
        if call.name == MAKE_TILE_GROUP
    ]
    return [group for group in found if group is not None]


def _access_phases(source: Source, function: ast.AST, groups: list[Group]):
    """Track only direct, non-escaping accesses in unconditional native sections."""
    declarations = {group.call.node for group in groups}
    bindings, targets, invalid = {}, set(), set()
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.value in declarations
        ):
            name = node.targets[0].id
            if name in bindings:
                invalid.update((bindings[name], node.value))
            bindings[name] = node.value
            targets.add(node.targets[0])
    accesses = {call: set() for call in declarations}
    for node in ast.walk(source.tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == source.pl:
            invalid.update(declarations)
        if (isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del))
                and (attribute_path(node) or "").startswith(source.pl + ".")):
            invalid.update(declarations)
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if node.id in bindings and node not in targets:
                invalid.add(bindings[node.id])
            if node.id == source.pl:
                invalid.update(declarations)
        if isinstance(node, ast.arg) and node.arg == source.pl:
            invalid.update(declarations)

    def referenced(node):
        return {
            bindings[item.id]
            for item in ast.walk(node)
            if isinstance(item, ast.Name) and item.id in bindings
        }

    operations = {
        source.pl + "." + name
        for name in ("load", "store", "move", "insert", "set_validshape", "gemm",
                     "gemm_mx", "matmul", "matmul_acc", "matmul_mx", "matmul_mx_acc")
    }
    epochs = {"cube": 0, "vector": 0}

    def visit(body, section=None):
        for statement in body:
            if isinstance(statement, ast.Assign) and statement.value in declarations:
                continue
            refs = referenced(statement)
            if isinstance(statement, ast.With) and len(statement.items) == 1:
                item = statement.items[0]
                expr = item.context_expr
                name = attribute_path(expr.func) if isinstance(expr, ast.Call) else None
                if (name in {source.pl + ".section_cube", source.pl + ".section_vector"}
                        and not expr.args and not expr.keywords and item.optional_vars is None):
                    visit(statement.body, name.rsplit("_", 1)[-1])
                    continue
            call = statement.value if isinstance(statement, ast.Expr) else None
            if isinstance(call, ast.Call):
                name = attribute_path(call.func)
                if (section and name == source.pl + ".system.bar_all"
                        and not call.args and not call.keywords):
                    epochs[section] += 1
                    continue
                if section and name in operations:
                    escaped = any(
                        isinstance(child, ast.Call) and child is not call and referenced(child)
                        and not (
                            isinstance(child.func, ast.Attribute)
                            and isinstance(child.func.value, ast.Name)
                            and child.func.value.id in bindings
                            and child.func.attr in {"next", "current", "previous"}
                        )
                        for child in ast.walk(call)
                    )
                    if not escaped:
                        for ref in refs:
                            accesses[ref].add((section, epochs[section]))
                        continue
            invalid.update(refs)
            if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                break

    visit(function.body)
    return {call: uses for call, uses in accesses.items() if uses and call not in invalid}


def _phase_review(a: Group, b: Group, accesses) -> str:
    left, right = accesses.get(a.call.node), accesses.get(b.call.node)
    if not left or not right:
        return ""
    sections = {section for section, _ in left | right}
    if len({section for section, _ in left}) == 1 and len({section for section, _ in right}) == 1 and len(sections) > 1:
        return "直接访问涉及不同核心段；相同 mutex ID 也不能跨核心互斥，须核对实际物理归属与跨核交接。"
    if len(sections) != 1:
        return ""
    before, after = {phase for _, phase in left}, {phase for _, phase in right}
    if max(before) < min(after) or max(after) < min(before):
        return (
            "直接访问分属同一核心段内由无条件 bar_all 分隔的阶段；"
            "须核对生成代码的 PIPE_ALL、预取完成、数据初始化和所有消费者，不能仅凭该提示认定竞态。"
        )
    return ""


def _conflicts(
    source: Source, function: ast.AST, env: dict[str, object]
) -> list[Finding]:
    groups = _groups(source, function, env)
    accesses = _access_phases(source, function, groups)
    findings = []
    for outer in range(len(groups)):
        for inner in range(outer + 1, len(groups)):
            a, b = groups[outer], groups[inner]
            if a.memory != b.memory:
                continue
            review = _phase_review(a, b, accesses)
            for i, slot_a in enumerate(a.slots):
                for j, slot_b in enumerate(b.slots):
                    if not _overlaps(slot_a, slot_b) or a.ids[i] == b.ids[j]:
                        continue
                    findings.append(
                        Finding(
                            source.path,
                            b.call.lineno,
                            RULE,
                            f"这个 group 的第 {j} 槽 `[{slot_b[0]:#x}, {slot_b[1]:#x})` "
                            f"与 {a.call.lineno} 行那个 group 的第 {i} 槽 "
                            f"`[{slot_a[0]:#x}, {slot_a[1]:#x})` 重叠（都在 `{a.memory}`），"
                            f"但 mutex_id 分别是 {b.ids[j]} 和 {a.ids[i]}。"
                            + (review or "auto_mutex 不会仅凭地址重叠建立依赖；需统一同期别名的锁所有权，或证明独立的完整交接。本扫描未证明实际并发访问。"),
                            advisory=bool(review),
                        )
                    )
    return findings


def check(path: Path) -> list[Finding]:
    source = read(path)
    if source.pl is None:
        return []
    env = _constants(source)
    findings: list[Finding] = []
    for node in ast.walk(source.tree):
        if not has_decorator(node, source.pl, "jit"):
            continue
        # Skip only explicit `auto_mutex=False`. `auto_mutex` defaults to True
        # (see the signature in `runtime/jit.py`), so bare `@pl.jit()` still inserts locks automatically.
        if auto_mutex_off(node, source.pl):
            continue
        found = _conflicts(source, node, env)
        findings += apply_claim(source, node, RULE, found, counterexample_claims(node))
    return findings
