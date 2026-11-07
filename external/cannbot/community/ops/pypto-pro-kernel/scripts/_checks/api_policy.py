# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Enforce the Vector authoring policy and verify VF namespace membership.

The policy is independent of whether a tile operation can compile. Source-based
membership is only a spelling check, not argument, dtype, or numerical validation.
"""

import ast
import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .paths import PYPTO_ROOT
from .source import Finding, PL_MODULE, read

TILE_RULE = "tile-vector-compute-forbidden"
VF_RULE = "vf-api-not-in-source"
VF_UNAVAILABLE = "vf-api-source-unavailable"
VF_PREFIX = PL_MODULE + ".Vf."

# Workspace policy, covering declared and parser-supported block computations at
# behavior 872df763f. Keep scalar min/max/astype, memory operations and Cube APIs
# out of this set. Source inspection below also discovers newer Tile signatures.
VECTOR_COMPUTE = frozenset(
    """
abs add add_relu add_relu_cast adds and_ ands argmax argmin axpy cast
col_expand col_expand_add col_expand_sub col_expand_mul col_expand_div
col_expand_max col_expand_min col_expand_expdif col_expand_binop
col_prod col_sum col_max col_min dequant div divs
eq exp expand_div expand_max expand_min expand_mul expand_sub expands fill_index
full fused_mul_add fused_mul_add_relu gather gatherb gathermask ge gt histogram
le log lrelu lt maximum maxs minimum mins mul mul_add_dst mul_cast muls ne neg
not_ or_ ors partadd partmax partmin partmul prelu quant recip relu rem rems
row_expand row_expand_add row_expand_sub row_expand_mul row_expand_div
row_expand_max row_expand_min row_expand_expdif row_expand_binop
row_prod row_sum row_max row_min
rsqrt scatter select shl shls shr shrs sqrt sub sub_relu sub_relu_cast subs sum
transpose xor xors
""".split()
)
SORT_OPS = frozenset({"sort32", "mrgsort", "mrgsort2"})
NON_VECTOR = (
    frozenset(
        """
load load_tile store store_tile init_output move insert ssbuf_load ssbuf_store
getval setval set_validshape reinterpret subview reshape fillpad fillpad_expand
fillpad_inplace ub_copy make_tile make_tile_group allocate_tile_group gemm gemm_mx
matmul matmul_acc matmul_bias matmul_mx matmul_mx_acc gemv gemv_acc gemv_bias
dump_data bind_crosscore_sync
""".split()
    )
    | SORT_OPS
)


@dataclass(frozen=True)
class Surface:
    tile_names: frozenset
    vf_names: frozenset
    vf_complete: bool
    identity: str


@lru_cache(maxsize=8)
def surface(pypto_root: Path) -> Surface:
    """Read source text only; missing or changed source formats remain explicit."""
    root = Path(pypto_root).resolve()
    tile_names = set(VECTOR_COMPUTE)
    api_path = root / "python/pypto_pro/language/_api.py"
    try:
        api_tree = ast.parse(api_path.read_text(encoding="utf-8"))
        for fn in api_tree.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
            if any(
                arg.annotation and re.search(r"\bTile\b", ast.unparse(arg.annotation))
                for arg in args
            ):
                tile_names.add(fn.name)
    except (OSError, SyntaxError):
        pass
    tile_names -= NON_VECTOR

    stub = root / "python/pypto_pro/language/_vf_api.py"
    registry = root / "framework/src/interface/ir/op/vf_ops.cpp"
    names = set()
    identities = []
    complete = True
    try:
        payload = stub.read_bytes()
        tree = ast.parse(payload, filename=str(stub))
        declared = {
            fn.name
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name == "Vf"
            for fn in cls.body
            if isinstance(fn, ast.FunctionDef) and not fn.name.startswith("_")
        }
        complete &= {"full", "load_align"} <= declared
        names.update(declared)
        identities.append("Python=" + hashlib.sha256(payload).hexdigest()[:12])
    except (OSError, SyntaxError):
        complete = False
    try:
        payload = registry.read_bytes()
        registered = set(
            re.findall(
                r'REGISTER_OP\(\s*"vf\.([A-Za-z_][A-Za-z_0-9]*)"\s*\)',
                payload.decode("utf-8"),
            )
        )
        complete &= {"full", "load_align"} <= registered
        names.update(registered)
        identities.append("C++=" + hashlib.sha256(payload).hexdigest()[:12])
    except (OSError, UnicodeError):
        complete = False
    return Surface(
        frozenset(tile_names),
        frozenset(names),
        bool(complete),
        str(root) + " (" + ", ".join(identities) + ")",
    )


def inspect(path: Path, *, pypto_root: Path = PYPTO_ROOT):
    source = read(path)
    catalog = surface(Path(pypto_root))
    findings = []
    counts = {"tile": 0, "vf": 0, "unverified": 0}
    scopes = [{}]

    def resolve(node):
        if isinstance(node, ast.Name):
            for scope in reversed(scopes):
                if node.id in scope:
                    return scope[node.id]
        elif isinstance(node, ast.Attribute):
            parent = resolve(node.value)
            return parent + "." + node.attr if parent else None
        return None

    def bind(target, value=None):
        if isinstance(target, ast.Name):
            scopes[-1][target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                bind(item)

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                scopes[-1][alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )

        def visit_ImportFrom(self, node):
            for alias in node.names:
                scopes[-1][alias.asname or alias.name] = (
                    node.module + "." + alias.name if node.module else None
                )

        def visit_Assign(self, node):
            self.visit(node.value)
            value = resolve(node.value)
            for target in node.targets:
                bind(target, value)

        def visit_AnnAssign(self, node):
            if node.value:
                self.visit(node.value)
            bind(node.target, resolve(node.value))

        def visit_AugAssign(self, node):
            self.visit(node.value)
            bind(node.target)

        def visit_FunctionDef(self, node):
            parameters = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            annotated = {arg.arg: resolve(arg.annotation) for arg in parameters}
            local = {
                arg.arg: (
                    PL_MODULE + ".Vf"
                    if annotated[arg.arg] == PL_MODULE + ".Vf"
                    else None
                )
                for arg in parameters
            }
            if node.args.vararg:
                local[node.args.vararg.arg] = None
            if node.args.kwarg:
                local[node.args.kwarg.arg] = None
            # Python locals shadow outer imports for the whole function.
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    local.setdefault(child.id, None)
            scopes.append(local)
            for statement in node.body:
                self.visit(statement)
            scopes.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            name = resolve(node.func)
            if (
                name
                and name.startswith(VF_PREFIX)
                and "." not in name[slice(len(VF_PREFIX), None)]
            ):
                counts["vf"] += 1
                operation = name[slice(len(VF_PREFIX), None)]
                if catalog.vf_complete and operation not in catalog.vf_names:
                    findings.append(
                        Finding(
                            source.path,
                            node.lineno,
                            VF_RULE,
                            f"vf.{operation} 未在所选 Python 声明与 C++ VF 注册表中找到："
                            f"{catalog.identity}。核对当前安装的命名和能力；不能把 tile API 仅改前缀。"
                            "用 --pypto-root 指定匹配的源码，并查询 check.py api vf。",
                        )
                    )
                elif not catalog.vf_complete:
                    counts["unverified"] += 1
                    if not any(item.rule == VF_UNAVAILABLE for item in findings):
                        findings.append(
                            Finding(
                                source.path,
                                node.lineno,
                                VF_UNAVAILABLE,
                                f"VF 名称未完整核对：{catalog.identity} 缺少可识别的 Python 声明或 C++ 注册表。"
                                "请用 --pypto-root 指向当前安装对应的完整源码；已有静态检查不能代替本项。",
                                advisory=True,
                            )
                        )
            elif name and name.startswith(PL_MODULE + "."):
                operation = name[slice(len(PL_MODULE) + 1, None)]
                if "." not in operation and operation in catalog.tile_names:
                    counts["tile"] += 1
                    findings.append(
                        Finding(
                            source.path,
                            node.lineno,
                            TILE_RULE,
                            f"pl.{operation} 是被工作区策略禁止的 Vector tile 计算。"
                            "除 sort32/mrgsort/mrgsort2 外，Vector 计算应在 @pl.vector_function 中使用实际 VF API；"
                            "不要仅替换前缀。分配、搬运、同步、标量控制和 Cube GEMM 不受此禁令影响。",
                        )
                    )
            self.generic_visit(node)

    Visitor().visit(source.tree)
    return findings, counts


def check(path: Path, *, pypto_root: Path = PYPTO_ROOT):
    return inspect(path, pypto_root=pypto_root)[0]
