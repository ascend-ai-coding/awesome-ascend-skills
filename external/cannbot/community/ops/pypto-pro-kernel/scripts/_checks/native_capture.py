# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Inspect native PyPTO target programs without code generation or execution."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module, metadata
import inspect
import json
import os
from pathlib import Path
import platform
import tempfile
import threading

from . import feedback

_LOCK = threading.RLock()
_CHILDREN = {
    "Program": ("functions",),
    "Function": ("body",),
    "SeqStmts": ("stmts",),
    "AssignStmt": ("value",),
    "EvalStmt": ("expr",),
    "IfStmt": ("condition", "then_body", "else_body"),
    "ForStmt": ("start", "stop", "step", "iter_args", "body"),
    "WhileStmt": ("condition", "iter_args", "body"),
    "SectionStmt": ("body",),
    "YieldStmt": ("value",),
    "ReturnStmt": ("value",),
    "BreakStmt": ("value",),
    "ContinueStmt": ("value",),
    "IterArg": ("initValue",),
    "Call": ("args",),
    "MakeTuple": ("elements",),
    "GetItemExpr": ("value", "slice"),
    "Var": (),
    "ConstInt": (),
    "ConstFloat": (),
    "ConstBool": (),
    "MemRef": ("addr",),
    "ScalarExpr": (),
}
for _kind in (
    "Add Sub Mul FloorDiv FloorMod FloatDiv Min Max Pow Eq Ne Lt Le Gt Ge And Or Xor "
    "BitAnd BitOr BitXor BitShiftLeft BitShiftRight"
).split():
    _CHILDREN[_kind] = ("left", "right")
for _kind in "Abs Neg Not BitNot Cast".split():
    _CHILDREN[_kind] = ("operand",)


@dataclass(frozen=True)
class CaptureConfig:
    block_dim: int | None = None
    tiling_key: dict | None = None


@dataclass(frozen=True)
class Capture:
    digest: str
    sources: dict
    transfers: list
    unknown_nodes: tuple[str, ...]


def _ir_text(ir, program):
    for name in ("text_dump", "python_print"):
        printer = getattr(ir, name, None)
        if callable(printer):
            return printer(program)
    raise RuntimeError("Native preflight requires an IR text printer")


def _runtime():
    """Guard the private preparation interfaces before importing a user entry."""
    import pypto
    import pypto_pro
    from pypto_pro import ir
    from pypto_pro.runtime.kernel import KernelDef

    jit = import_module("pypto_pro.runtime.jit")
    parser = getattr(KernelDef, "parse_target_program", None)
    prepare = getattr(jit, "_prepare_codegen_inputs", None)
    required = (
        getattr(ir, "attributes_api_version", None) == 1
        and callable(getattr(ir, "get_attributes", None))
        and any(callable(getattr(ir, name, None)) for name in ("text_dump", "python_print"))
        and callable(parser)
        and "observer" in inspect.signature(parser).parameters
        and callable(prepare)
        and "bound_signature" in inspect.signature(prepare).parameters
        and callable(getattr(jit, "_extract_param_specs", None))
        and callable(getattr(jit, "_validate_scalar_arg", None))
    )
    if not required:
        raise RuntimeError(
            "Native preflight requires IR attributes v1, a per-call parse observer and bound-signature preparation"
        )
    probe = ir.Call("preflight_probe", [], {"ids": [2, 7]}, ir.Span.unknown())
    if ir.get_attributes(probe) != {"ids": [2, 7]}:
        raise RuntimeError("Native IR attribute probe did not preserve its metadata")
    root = Path(pypto_pro.__file__).resolve().parent
    native_root = Path(pypto.pypto_impl.__file__).resolve().parent
    producer = {
        "package": "pypto",
        "version": metadata.version("pypto"),
        "python": platform.python_version(),
        "attributes_api_version": 1,
        "observer": True,
        "python_sources": {
            str(p.relative_to(root)): feedback.file_identity(p)["sha256"]
            for p in sorted(root.rglob("*.py"))
        },
        "native_libraries": {
            str(p.relative_to(native_root)): feedback.file_identity(p)["sha256"]
            for p in sorted(native_root.rglob("*.so"))
        },
    }
    return ir, jit, producer


@contextmanager
def _architecture(arch):
    name = str(getattr(arch, "name", arch)).strip().lower()
    if name not in {"a5", "3510", "dav_3510", "npuarch.dav_3510"}:
        raise RuntimeError(
            f"Native preflight supports the current a5 frontend, got {arch!r}"
        )
    selected = {"PYPTOPRO_JIT_ARCH": "a5" if name == "a5" else "3510", "PYPTOPRO_NPU_ARCH": "dav-c310"}
    previous = {key: os.environ.get(key) for key in selected}
    os.environ.update(selected)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _source(filename, sources):
    path = Path(filename)
    try:
        sha = feedback.file_identity(path)["sha256"]
    except OSError:
        sha = None
    sources[f"{sha or 'unavailable'}:{path.name}"] = {
        "name": path.name,
        "sha256": sha,
        "available": sha is not None,
    }


def _tile(value):
    tile = getattr(value, "type", None)
    if type(tile).__name__ != "TileType":
        return {}

    def dimension(value):
        return value.value if type(value).__name__ == "ConstInt" else None

    memory = getattr(tile, "memref", None)
    hardware = getattr(tile, "hardware_info", None)
    view = getattr(tile, "tile_view", None)
    layout = None
    if hardware is not None:
        layout = {
            ("row_major", "none_box"): "ND",
            ("col_major", "none_box"): "DN",
            ("col_major", "row_major"): "NZ",
            ("row_major", "col_major"): "ZN",
        }.get((hardware.blayout.name, hardware.slayout.name))
    return {
        "space": memory.memory_space.name if memory is not None else None,
        "layout": layout,
        "dtype": tile.dtype.to_string(),
        "shape": [dimension(d) for d in tile.shape],
        "valid_shape": [dimension(d) for d in view.valid_shape]
        if view is not None
        else None,
    }


def inspect_program(program, target, sources):
    """Read transfer descriptors; unrecognized IR shapes leave coverage incomplete."""
    transfers, unknown, retained = [], set(), {}

    def visit(node, enclosing=None):
        if node is None or type(node) in (str, int, float, bool):
            return
        if isinstance(node, dict):
            for value in node.values():
                visit(value, enclosing)
            return
        if isinstance(node, (tuple, list)):
            for value in node:
                visit(value, enclosing)
            return
        if id(node) in retained:
            return
        retained[id(node)] = node
        kind = type(node).__name__
        fields = _CHILDREN.get(kind)
        if fields is None:
            unknown.add(kind)
            return
        span = getattr(node, "span", None)
        location = enclosing
        if span is not None and span.filename and span.begin_line > 0:
            _source(span.filename, sources)
            location = {
                "file": span.filename,
                "line": span.begin_line,
                "column": max(1, span.begin_column),
            }
        if kind == "Call" and node.name in {"block.move", "block.insert"}:
            dst, src = map(_tile, node.args[:2])
            if src.get("space") == "Vec" and dst.get("space") in {"Vec", "Mat"}:
                transfers.append(
                    {
                        "api": node.name.split(".")[-1],
                        "source": src,
                        "destination": dst,
                        "location": location,
                        "target": target,
                        "scope": (
                            "native descriptors; runtime valid-shape mutations,"
                            " producer bytes and synchronization remain unproved"
                        ),
                    }
                )
        for field in fields:
            if not hasattr(node, field):
                unknown.add(kind + "." + field)
            else:
                visit(getattr(node, field), location)

    visit(program)
    return transfers, tuple(sorted(unknown))


def _program_observer(target, sources):
    observations = []

    def observe(program, parser):
        observations.append(inspect_program(program, target, sources))

    return observations, observe


def capture_ir(kernel, *args, config):
    """Transform and parse both native targets; never invoke codegen or a launcher."""
    with _LOCK:
        ir, jit, _ = _runtime()
        required = (
            "to_kernel_def",
            "_bind_runtime_args",
            "_normalize_launch_args",
            "_resolve_spec_identity",
        )
        if not all(callable(getattr(kernel, name, None)) for name in required):
            raise ValueError("Expected a native @pl.jit Pro kernel")
        arch = kernel.arch or jit.get_current_arch()
        normalize = getattr(jit, "normalize_arch", None)
        if callable(normalize):
            arch = normalize(arch)
        elif isinstance(arch, str):
            arch = arch.strip().lower()
        with _architecture(arch):
            key = dict(config.tiling_key) if config.tiling_key is not None else None
            if kernel.tilingkey_schema is not None:
                if key is None:
                    raise ValueError("This kernel requires a concrete tiling_key")
                kernel.tilingkey_schema.validate_concrete(key)
            elif key is not None:
                raise ValueError("The kernel does not declare a tiling_key")
            specialization = kernel._resolve_spec_identity(key, None)
            arguments = kernel._normalize_launch_args(tuple(args), {})
            bound = kernel._bind_runtime_args(arguments)
            definition = kernel.to_kernel_def(key, datatype_consts=specialization[1])
            sources, transfers, unknown, programs, parsed = {}, [], set(), [], []
            _source(definition._source_file, sources)
            with tempfile.TemporaryDirectory(prefix="pypto-preflight-") as scratch:
                jit._prepare_codegen_inputs(
                    definition,
                    arch,
                    tilingkey_packed=specialization[0],
                    out_dir=scratch,
                    bound_signature=bound,
                )
                for target, section in (
                    ("cube", ir.SectionKind.Cube),
                    ("vector", ir.SectionKind.Vector),
                ):
                    observations, observe = _program_observer(target, sources)

                    program, active = definition.parse_target_program(
                        section, bound_signature=bound, observer=observe
                    )
                    if len(observations) != 1:
                        raise RuntimeError(
                            "Native parser observer did not run exactly once"
                        )
                    facts, missing = observations[0]
                    parsed.append((bool(active), facts, missing))
                    programs.append(
                        {
                            "target": target,
                            "active": bool(active),
                            "ir": _ir_text(ir, program).replace(
                                scratch, "<native-preflight>"
                            ),
                        }
                    )
                specs = jit._extract_param_specs(
                    program, definition.last_param_directions
                )
                for index, parameter in enumerate(specs):
                    if parameter.kind == jit.ParamKind.SCALAR:
                        jit._validate_scalar_arg(index, arguments[index], parameter)
            any_active = any(active for active, _, _ in parsed)
            for active, facts, missing in parsed:
                if active or not any_active:
                    transfers.extend(facts)
                    unknown.update(missing)
            digest = feedback.digest(
                json.dumps(
                    {
                        "programs": programs,
                        "sources": sources,
                        "block_dim": config.block_dim,
                        "tiling_key": key,
                    },
                    sort_keys=True,
                ).encode()
            )
            return Capture(digest, sources, transfers, tuple(sorted(unknown)))
