# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check a narrow source proof of an unprotected bare-Vec-Tile handover.

Declarations and known single-pipe accesses do not establish a race. Helpers,
aliases, control flow and manual synchronization remain advisory. Only adjacent
load/store statements on one unrebound local Vec Tile are a hard finding, with
roles and pipe registrations checked against the selected framework sources.
This is not a general synchronization or physical-alias analysis.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from .paths import PYPTO_ROOT
from .source import (
    Finding,
    apply_claim,
    attribute_path,
    auto_mutex_off,
    counterexample_claims,
    has_decorator,
    read,
)
from .stage_handover import body_nodes

RULE = "auto-mutex-needs-tile-group"


def _contract(root):
    """Read the original role table and guard the two Vec transfer pipes."""
    names = (
        "python/pypto_pro/language/parser/_op_pipeline.py",
        "framework/src/interface/pypto_pro/backend/backend_cce_block_out_ops.cpp",
    )
    identity = {}
    try:
        payloads = [(Path(root) / name).read_bytes() for name in names]
        identity = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in zip(names, payloads)
        }
        tree = ast.parse(payloads[0])
        tables = {
            node.target.id: node.value
            for node in tree.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        roles = ast.literal_eval(tables["_BLOCK_OP_TILE_ROLES"])
        aliases = ast.literal_eval(tables["_BLOCK_OP_ALIASES"])
        if not isinstance(roles, dict) or not isinstance(aliases, dict):
            return None, identity
        reader = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_op_tile_roles"
        )
        if not any(
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and attribute_path(node.value.func) == "_BLOCK_OP_TILE_ROLES.get"
            for node in ast.walk(reader)
        ):
            return None, identity
        cpp = payloads[1].decode()
        for name, pipe in [("load", "MTE2"), ("store", "MTE3")]:
            pattern = (
                r'REGISTER_BACKEND_OP\(BackendCCE,\s*"block\.'
                + name
                + r'"\)\s*\.set_pipe\(ir::PipeType::'
                + pipe
                + r"\)"
            )
            if (
                roles.get(name) != ["W", "R"]
                or aliases.get(name, name) != name
                or len(re.findall(pattern, cpp)) != 1
            ):
                return None, identity
        return {name: roles[name] for name in ("load", "store")}, identity
    except (
        OSError,
        SyntaxError,
        ValueError,
        KeyError,
        StopIteration,
    ):
        return None, identity


def _enabled(fn, alias):
    if len(fn.decorator_list) != 1:
        return False
    for decorator in fn.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if attribute_path(target) != f"{alias}.jit":
            continue
        if not isinstance(decorator, ast.Call):
            return True
        if any(k.arg not in {"auto_mutex", "name", "arch"} for k in decorator.keywords):
            return False
        option = next(
            (k.value for k in decorator.keywords if k.arg == "auto_mutex"), None
        )
        return (
            option is None or isinstance(option, ast.Constant) and option.value is True
        )
    return False


def _vec_declaration(call, alias):
    type_node = (
        call.args[0]
        if call.args
        else next((k.value for k in call.keywords if k.arg == "type"), None)
    )
    return (
        isinstance(type_node, ast.Call)
        and attribute_path(type_node.func) == f"{alias}.TileType"
        and any(
            k.arg == "target_memory"
            and attribute_path(k.value) == f"{alias}.MemorySpace.Vec"
            for k in type_node.keywords
        )
    )


def _adjacent_handover(fn, alias, name):
    sections = [n for n in fn.body if isinstance(n, ast.With)]
    if len(sections) != 1:
        return False
    for node in fn.body:
        if node is sections[0]:
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if not isinstance(node, ast.Assign):
            return False
        if any(
            isinstance(n, ast.Call)
            and attribute_path(n.func)
            not in {f"{alias}.make_tile", f"{alias}.TileType"}
            for n in ast.walk(node.value)
        ):
            return False
    for section in fn.body:
        if not isinstance(section, ast.With) or len(section.items) != 1:
            continue
        context = section.items[0].context_expr
        if (
            not isinstance(context, ast.Call)
            or attribute_path(context.func) != f"{alias}.section_vector"
            or context.args
            or context.keywords
        ):
            continue
        for statement in section.body:
            if not isinstance(statement, ast.Expr) or not isinstance(
                statement.value, ast.Call
            ):
                return False
            call = statement.value
            if attribute_path(call.func) not in {
                f"{alias}.load",
                f"{alias}.store",
                f"{alias}.set_validshape",
            }:
                return False
            if any(isinstance(n, ast.Call) and n is not call for n in ast.walk(call)):
                return False
        for first, second in zip(section.body, section.body[1:]):
            if not all(
                isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                for s in (first, second)
            ):
                continue
            load, store = first.value, second.value
            if (
                attribute_path(load.func) == f"{alias}.load"
                and attribute_path(store.func) == f"{alias}.store"
                and len(load.args) >= 2
                and len(store.args) >= 2
                and isinstance(load.args[0], ast.Name)
                and load.args[0].id == name
                and isinstance(store.args[1], ast.Name)
                and store.args[1].id == name
                and not any(k.arg is None for c in (load, store) for k in c.keywords)
                and not any(
                    isinstance(a, ast.Starred) for c in (load, store) for a in c.args
                )
            ):
                return True
    return False


def inspect(path: Path, *, pypto_root: Path = PYPTO_ROOT):
    source = read(path)
    roles, identity = _contract(pypto_root)
    coverage = {
        "guard_verified": roles is not None,
        "framework_sources": identity,
        "unused": 0,
        "single_pipe": 0,
        "proven_handover": 0,
        "unverified": 0,
        "limits": "Direct local bindings only; no physical-alias, helper, lifetime or general control-flow proof.",
    }
    if not source.pl:
        return [], coverage
    alias = source.pl
    namespace = alias.split(".")[0]
    shadowed = any(
        isinstance(n, ast.Name)
        and n.id == namespace
        and isinstance(n.ctx, (ast.Store, ast.Del))
        or isinstance(n, ast.arg)
        and n.arg == namespace
        or isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and n.name == namespace
        or isinstance(n, ast.alias)
        and (n.asname or n.name.split(".")[0]) == namespace
        and n.name != "pypto_pro.language"
        or isinstance(n, ast.Attribute)
        and isinstance(n.ctx, (ast.Store, ast.Del))
        and (attribute_path(n) or "").startswith(alias + ".")
        or isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in {"setattr", "delattr", "exec", "eval"}
        for n in ast.walk(source.tree)
    )
    findings = []
    for fn in ast.walk(source.tree):
        if not has_decorator(fn, alias, "jit") or auto_mutex_off(fn, alias):
            continue
        nodes = list(body_nodes(fn))
        parents = {
            child: node for node in nodes for child in ast.iter_child_nodes(node)
        }
        bare = [
            n
            for n in nodes
            if isinstance(n, ast.Call)
            and attribute_path(n.func) == f"{alias}.make_tile"
        ]
        manual = any(
            isinstance(n, ast.Call)
            and (attribute_path(n.func) or "").startswith(
                tuple(
                    f"{alias}.system.{prefix}" for prefix in ("sync_", "bar_", "mutex_")
                )
            )
            for n in nodes
        )
        found = []
        for call in bare:
            binding = parents.get(call)
            name = (
                binding.targets[0].id
                if isinstance(binding, ast.Assign)
                and len(binding.targets) == 1
                and isinstance(binding.targets[0], ast.Name)
                else None
            )
            if name is None:
                uses = [None]
            else:
                uses = [
                    n
                    for n in nodes
                    if isinstance(n, ast.Name)
                    and n.id == name
                    and isinstance(n.ctx, ast.Load)
                ]
                if any(
                    isinstance(
                        n,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                            ast.ClassDef,
                            ast.Lambda,
                        ),
                    )
                    and any(
                        isinstance(part, ast.Name)
                        and part.id == name
                        and isinstance(part.ctx, ast.Load)
                        for part in ast.walk(n)
                    )
                    for n in ast.iter_child_nodes(fn)
                ):
                    uses.append(None)
            if not uses:
                coverage["unused"] += 1
                continue
            accesses = []
            unknown = roles is None or shadowed
            for use in uses:
                parent = parents.get(use)
                if isinstance(parent, ast.Call) and use in parent.args:
                    api = attribute_path(parent.func)
                    if api == f"{alias}.set_validshape":
                        continue
                    operation = (
                        api.removeprefix(f"{alias}.")
                        if api and api.startswith(f"{alias}.")
                        else ""
                    )
                    index = parent.args.index(use)
                    if (
                        roles is not None
                        and operation in roles
                        and index < len(roles[operation])
                    ):
                        accesses.append((operation, roles[operation][index]))
                        continue
                unknown = True
            if not accesses and not unknown:
                coverage["unused"] += 1
                continue
            if not unknown and len({op for op, _ in accesses}) <= 1:
                coverage["single_pipe"] += 1
                continue
            rebound = (
                name is None
                or sum(
                    isinstance(n, ast.Name)
                    and n.id == name
                    and isinstance(n.ctx, ast.Store)
                    for n in nodes
                )
                != 1
            )
            proven = (
                not unknown
                and not manual
                and not rebound
                and _enabled(fn, alias)
                and _vec_declaration(call, alias)
                and _adjacent_handover(fn, alias, name)
            )
            if proven:
                coverage["proven_handover"] += 1
                message = (
                    f"{name}: adjacent MTE2 load and MTE3 store access the same bare"
                    " Vec Tile without a handover. Automatic mutex insertion has no"
                    " tile-group ID for it; use matched ordering or a protected Tile group."
                )
            else:
                coverage["unverified"] += 1
                message = (
                    f"{name or 'Bare Tile'}: buffer ordering is unverified. Inspect"
                    " actual read/write pipes, aliases, manual event pairs and reuse"
                    " in the generated IR; a declaration or auto_mutex option alone does not establish a race."
                )
            found.append(
                Finding(source.path, call.lineno, RULE, message, advisory=not proven)
            )
        findings += apply_claim(source, fn, RULE, found, counterexample_claims(fn))
    return findings, coverage


def check(path: Path, *, pypto_root: Path = PYPTO_ROOT) -> list[Finding]:
    return inspect(path, pypto_root=pypto_root)[0]
