# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Explicit metadata-only native parsing through the installed PyPTO runtime.

Importing an entry executes that Python module. Entries must keep launch code
behind a main guard; this tool invokes only native capture, never the launcher.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import linecache
import sys
from pathlib import Path

from . import feedback, query, runtime_diagnostics, ub_l1
from .paths import EXAMPLES_ROOT, PYPTO_ROOT


def read_spec(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) - {
        "entry",
        "args",
        "block_dim",
        "tiling_key",
    }:
        raise ValueError(
            "Native spec accepts entry, args, block_dim and tiling_key only"
        )
    entry = document.get("entry", "")
    if not isinstance(entry, str) or ":" not in entry:
        raise ValueError("Native entry must be path.py:callable")
    filename, name = entry.rsplit(":", 1)
    if not all(
        part.isidentifier() and not part.startswith("__") for part in name.split(".")
    ):
        raise ValueError(
            "Native entry must name a callable attribute, not an expression"
        )
    source = (path.parent / filename).resolve()
    if not source.is_file() or source.suffix != ".py":
        raise ValueError("Native entry must be an existing Python source file")
    if not isinstance(document.get("args"), list):
        raise ValueError("Native spec args must be a positional-argument list")
    for arg in document["args"]:
        if isinstance(arg, dict):
            if (
                set(arg) != {"shape", "dtype"}
                or not isinstance(arg["shape"], list)
                or not isinstance(arg["dtype"], str)
            ):
                raise ValueError("Tensor arguments require shape and torch dtype name")
            if not all(type(d) is int and d >= 0 for d in arg["shape"]):
                raise ValueError("Tensor dimensions must be nonnegative integers")
        elif type(arg) not in (int, float, bool):
            raise ValueError(
                "Native arguments must be Tensor metadata or numeric scalars"
            )
    if "block_dim" in document and (
        type(document["block_dim"]) is not int or document["block_dim"] <= 0
    ):
        raise ValueError("block_dim must be a positive integer")
    if document.get("tiling_key") is not None and not isinstance(
        document["tiling_key"], dict
    ):
        raise ValueError("tiling_key must be an object")
    return document, source, name


def _load(path, name):
    module_name = "_kernel_preflight_" + feedback.file_identity(path)["sha256"][:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))
    try:
        # A same-size edit can retain the timestamp used by Python's .pyc cache.
        # Capture must parse the bytes whose hash is reported, not stale bytecode.
        linecache.cache.pop(str(path), None)
        exec(
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True),
            module.__dict__,
        )
        target = module
        for part in name.split("."):
            target = getattr(target, part)
        return target
    finally:
        sys.path.pop(0)


def _entry_declaration(function, path):
    code = getattr(function, "__code__", None)
    if code and Path(code.co_filename).resolve() == path.resolve():
        return {
            "file": str(path.resolve()),
            "line": code.co_firstlineno,
            "sha256": feedback.file_identity(path)["sha256"],
        }
    return None


def entry_metadata(kernel, path):
    """Describe the imported callable without treating annotations as a diagnosis."""
    result = {"callable_type": type(kernel).__module__ + "." + type(kernel).__name__}
    try:
        result["tile_interface"] = callable(getattr(kernel, "to_kernel_def", None))
        original = (
            getattr(kernel, "_func", kernel) if result["tile_interface"] else kernel
        )
        hints = getattr(original, "__annotations__", {})
        result["string_annotations"] = {str(k): v[:400] for k, v in list(hints.items())[:8] if isinstance(v, str)}
        declaration = _entry_declaration(original, path)
        if declaration:
            result["declaration"] = declaration
        wrapped = inspect.unwrap(original)
        if callable(getattr(wrapped, "to_kernel_def", None)):
            wrapped = getattr(wrapped, "_func", wrapped)
        if wrapped is not original:
            declaration = _entry_declaration(wrapped, path)
            if declaration:
                result["wrapped_declaration"] = declaration
    except (AttributeError, TypeError, ValueError, NameError, OSError):
        pass  # Optional metadata must not replace the original capture failure.
    return result


def report(spec_path, *, pypto_root=PYPTO_ROOT, examples_root=EXAMPLES_ROOT):
    document, path, name = read_spec(spec_path)
    result = feedback.envelope("native_parse", pypto_root, [path, spec_path])
    result["specialization"] = {k: v for k, v in document.items() if k != "entry"}
    source_identity = feedback.file_identity(path)
    try:
        import torch
        from .native_capture import CaptureConfig, capture_ir, _runtime

        _, _, producer = _runtime()
    except (ImportError, OSError, RuntimeError) as exc:
        result["status"] = "unverified"
        result["diagnostics"] = [
            feedback.diagnostic(
                "native-capability-unavailable",
                "unverified",
                str(exc),
                next_action=(
                    "Run native preflight in the matching installed PyPTO environment;"
                    " source-only checks remain available."
                ),
            )
        ]
        return result
    result["identity"]["installed_runtime"] = {
        **{
            key: producer[key]
            for key in (
                "package",
                "version",
                "python",
                "attributes_api_version",
                "observer",
            )
        },
        "producer_sha256": feedback.digest(
            json.dumps(producer, sort_keys=True).encode()
        ),
        "interface_sha256": producer["native_libraries"].get(
            "lib/libtile_fwk_interface.so"
        ),
        "python_sources": {
            key.removeprefix("python/pypto_pro/"): producer["python_sources"].get(
                key.removeprefix("python/pypto_pro/")
            )
            for key in feedback.FRAMEWORK_FILES
        },
    }
    expected = result["identity"]["selected_source"]
    mismatch = [
        key
        for key, digest in expected["files"].items()
        if producer["python_sources"].get(key.removeprefix("python/pypto_pro/"))
        != digest
    ]
    if not expected["complete"] or mismatch:
        result["status"] = "unverified"
        result["diagnostics"] = [
            feedback.diagnostic(
                "native-source-mismatch",
                "unverified",
                "The selected source is missing or differs from the installed frontend.",
                evidence={"mismatched_files": mismatch},
                next_action="Select the installed version's source with --pypto-root.",
            )
        ]
        return result
    source_transfers = []
    kernel = None
    try:
        result["stage"] = "argument_binding"
        args = []
        for arg in document["args"]:
            if isinstance(arg, dict):
                dtype = getattr(torch, arg["dtype"], None)
                if not isinstance(dtype, torch.dtype):
                    raise ValueError(f"Unknown torch dtype: {arg['dtype']}")
                args.append(torch.empty(arg["shape"], dtype=dtype, device="cpu"))
            else:
                args.append(arg)
        result["stage"] = "entry_import"
        _, source_transfers, _ = ub_l1.inspect(path, pypto_root=pypto_root)
        kernel = _load(path, name)
        result["stage"] = "native_parse"
        snapshot = capture_ir(
            kernel,
            *args,
            config=CaptureConfig(
                block_dim=document.get("block_dim"),
                tiling_key=document.get("tiling_key"),
            ),
        )
        result["snapshot_sha256"] = snapshot.digest
        result["captured_sources"] = snapshot.sources
        result["validation"]["native_parse"] = "passed"
        transfers = snapshot.transfers
        result["transfers"] = transfers
        result["coverage"]["ub_l1"] = {
            "native_transfer_sites": len(transfers),
            "uninspected_node_kinds": list(snapshot.unknown_nodes),
            "limits": (
                "Descriptor checks only; no device codegen, execution, numerical validation or full lifetime analysis."
            ),
        }
        if snapshot.unknown_nodes:
            result["diagnostics"].append(
                feedback.diagnostic(
                    "native-inspection-incomplete",
                    "advisory",
                    "Transfer inspection does not cover every parsed IR node.",
                    evidence={"node_kinds": list(snapshot.unknown_nodes)},
                    next_action="Inspect the generated IR before using descriptor coverage as evidence.",
                )
            )
        for transfer in transfers:
            for rule, advisory, message in ub_l1.transfer_issues(
                transfer["api"],
                transfer["source"],
                transfer["destination"],
                ub_l1.pitch_guard(pypto_root),
            ):
                result["diagnostics"].append(
                    feedback.diagnostic(
                        rule,
                        "advisory" if advisory else "error",
                        message,
                        location=transfer["location"],
                        evidence=transfer,
                        next_action="Inspect the complete UB→L1 recipe and producer address map.",
                    )
                )
    except Exception as exc:
        if result["stage"] == "native_parse":
            result["validation"]["native_parse"] = "failed"
        result["diagnostics"], result["context"] = runtime_diagnostics.from_exception(
            exc,
            stage=result["stage"],
            examples_root=examples_root,
            transfers=source_transfers,
        )
        if kernel is not None:
            for item in result["diagnostics"]:
                if item["rule"] == "native-entry-kind":
                    result["stage"] = item["evidence"]["stage"] = "entry_classification"
                    result["validation"]["native_parse"] = "not_established"
                    item["evidence"]["entry"] = entry_metadata(kernel, path)
                    item["context_ids"] = query.ids_for_rules([item["rule"]])
                    result["context"] = query.context(
                        item["context_ids"], examples_root=examples_root
                    )
    if feedback.file_identity(path) != source_identity:
        result["diagnostics"].append(
            feedback.diagnostic(
                "source-changed", "error", "Entry source changed during native capture."
            )
        )
        result["validation"]["native_parse"] = "invalidated"
    if (
        any(d["rule"].startswith("ub-") for d in result["diagnostics"])
        and not result["context"]
    ):
        result["context"] = query.context(["ub-l1"], examples_root=examples_root)
    return feedback.finish(result)
