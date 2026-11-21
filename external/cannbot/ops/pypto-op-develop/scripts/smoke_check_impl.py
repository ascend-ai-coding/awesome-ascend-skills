#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""smoke_check_impl.py — host-side smoke check for a pypto impl file.

Runs ONLY what runs on the host, so coder can self-check before
``submit_for_verify`` without occupying a device:

  1. AST parse      — syntax errors;
  2. module import  — import-time errors;
  3. frontend trace — for every ``@pypto.frontend.jit`` kernel, build CPU
     example inputs from the cached signature and call
     ``JitCallableWrapper.compile()``. NOTE: ``compile()`` also runs the
     host-side kernel build (AICore/AICPU codegen + make): seconds for
     small kernels, minutes for large ones. No NPU is occupied and
     nothing is launched — but do not run large checks in a foreground
     tool call (the 120 s tool timeout kills the build); use ``--bg``.

Usage:
    python smoke_check_impl.py <impl.py> [--bg] [--strict] [--json]
                                         [--shapes '<json>'|@file.json]
                                         [--scalars '<json>'|@file.json]
    python smoke_check_impl.py --self-test

With ``--bg`` the script re-launches itself detached, logs to
``_debug/smoke_check/smoke.log`` and, on completion, writes the exit code
and conclusion line to ``_debug/smoke_check/smoke.result`` — poll that
file. It prints both paths before returning immediately.

Conclusion line (always the LAST stdout line, machine-greppable)::
    SMOKE RESULT: PASS ... | FAIL ...
With ``--json``, a machine-readable JSON object (per-kernel status, skip
reasons, counts) is printed right before the conclusion line.

SKIP means the script could not build inputs for a kernel (a tool
capability boundary), NOT that the kernel failed. Every SKIP is followed
by a ``[SMOKE HINT]`` line with a ready-to-paste retry command carrying
the ``--shapes``/``--scalars`` snippet to fill in. ``--strict`` turns
skips into failure. Smoke passing never implies verify passing.

Exit codes: 0 pass; 1 check failed (or skips under ``--strict``);
2 usage error (missing file, bad JSON, ...).
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Named logger bound to stdout (never the root logger — basicConfig would
# raise the root level and unleash third-party INFO noise into the output).
# The emitted text is the machine-readable protocol agents consume (SMOKE
# RESULT stays the LAST stdout line), byte-identical to plain prints.
_log = logging.getLogger("smoke_check_impl")
_log.setLevel(logging.INFO)
_log.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
_log.addHandler(_handler)

_DTYPES = None  # lazily built {pypto dtype name suffix: torch.dtype}

# --bg runtime artifacts, always under the caller's cwd (writes outside
# cwd trigger sandbox prompts).
_BG_DIR = Path("_debug") / "smoke_check"
_BG_LOG = "smoke.log"
_BG_RESULT = "smoke.result"


class SkipError(ValueError):
    """Inputs cannot be auto-built for a kernel.

    ``suggestion`` carries a ready-to-paste CLI snippet (``--shapes ...``
    or ``--scalars ...``) that would resolve the skip.
    """

    def __init__(self, msg: str, suggestion: str | None = None):
        super().__init__(msg)
        self.suggestion = suggestion


def _dtype_map():
    global _DTYPES
    if _DTYPES is None:
        import torch

        _DTYPES = {
            "FP32": torch.float32,
            "FLOAT32": torch.float32,
            "FP16": torch.float16,
            "FLOAT16": torch.float16,
            "BF16": torch.bfloat16,
            "BFLOAT16": torch.bfloat16,
            "INT32": torch.int32,
            "INT64": torch.int64,
            "INT16": torch.int16,
            "INT8": torch.int8,
            "UINT8": torch.uint8,
            "BOOL": torch.bool,
        }
    return _DTYPES


def _pto_dtype_name(pto_dtype) -> str:
    """Normalize a pypto DataType to its bare name (e.g. 'FP32').

    Prefers the enum's ``.name`` ('DT_FP32'): ``str()`` is not stable
    across pypto versions — older ones render 'DataType.DT_FP32' while
    newer ones render the numeric value ('7').
    """
    name = getattr(pto_dtype, "name", None) or str(pto_dtype)
    name = name.split(".")[-1].upper()
    return name[3:] if name.startswith("DT_") else name


def _torch_dtype(pto_dtype):
    """Map a pypto DataType (e.g. DataType.DT_FP32) to a torch dtype."""
    dt = _dtype_map().get(_pto_dtype_name(pto_dtype))
    if dt is None:
        raise ValueError(f"unsupported pypto dtype for example input: {pto_dtype}")
    return dt


def _dtype_name(pto_dtype) -> str:
    """Best-effort lowercase dtype name for --shapes skeletons."""
    return _pto_dtype_name(pto_dtype).lower()


def find_jit_kernels(module) -> list:
    """Module-level JitCallableWrapper objects (duck-typed, version-robust)."""
    kernels = []
    for name in dir(module):
        obj = getattr(module, name)
        if name.startswith("__"):
            continue
        if (
            hasattr(obj, "compile")
            and hasattr(obj, "_cached_signature")
            and hasattr(obj, "_create_parser")
        ):
            kernels.append((name, obj))
    return kernels


# Type-based fallback literals for scalar params that have neither a
# --scalars entry nor a declared default. Keys are the annotation type (or
# its name, for impls using `from __future__ import annotations`).
_FALLBACK_LITERALS = {
    float: 1.0,
    int: 1,
    bool: True,
    "float": 1.0,
    "int": 1,
    "bool": True,
}


def _resolve_scalars(name: str, wrapper, scalars_cfg: dict) -> dict:
    """Resolve concrete values for the kernel's non-tensor (scalar) params.

    Priority per param: --scalars entry > declared default (the wrapper's
    cached non-tensor defaults) > type-based literal from the annotation.
    Returns {param: value} ({} for tensor-only kernels) or raises
    SkipError listing the params that remain unresolved.
    """
    import inspect

    _, non_tensor = getattr(wrapper, "_cached_signature")
    if not non_tensor:
        return {}
    # --scalars accepts a flat form {"eps": 1e-6} (applies to every kernel)
    # and/or a nested form {"*": {...}, "<kernel>": {...}}. Priority:
    # kernel-specific entry > "*" entry > flat entry.
    cli_values = {k: v for k, v in scalars_cfg.items() if not isinstance(v, dict)}
    cli_values.update(scalars_cfg.get("*", {}))
    cli_values.update(scalars_cfg.get(name, {}))
    unknown = set(cli_values) - set(non_tensor)
    if unknown:
        raise ValueError(
            f"--scalars entry for {name!r} names unknown non-tensor params "
            f"{sorted(unknown)}; signature expects {non_tensor}"
        )
    defaults = getattr(wrapper, "_cached_non_tensor_defaults", None) or {}
    try:
        sig_params = inspect.signature(getattr(wrapper, "_original_func")).parameters
    except (TypeError, ValueError, AttributeError):
        sig_params = {}
    values, missing = {}, []
    for pname in non_tensor:
        if pname in cli_values:
            values[pname] = cli_values[pname]
            continue
        if pname in defaults:
            values[pname] = defaults[pname]
            continue
        ann = sig_params[pname].annotation if pname in sig_params else None
        literal = _FALLBACK_LITERALS.get(ann)
        if literal is None:
            missing.append(pname)
        else:
            values[pname] = literal
    if missing:
        snippet = json.dumps({p: "<value>" for p in missing})
        raise SkipError(
            f"kernel takes non-tensor params {missing} with no declared "
            f"default and no type-based fallback; fill the placeholders in "
            f"the --scalars snippet below to trace this kernel "
            f"(skipping trace, import+AST only)",
            suggestion=f"--scalars '{snippet}'",
        )
    return values


def _shape_skeleton(tensor_defs) -> list:
    """A --shapes spec skeleton: static dims kept, dynamic dims as <N>."""
    spec = []
    for d in tensor_defs:
        shape = getattr(d, "shape", None) or []
        dims = [dim if isinstance(dim, int) else "<N>" for dim in shape]
        spec.append({"shape": dims, "dtype": _dtype_name(getattr(d, "dtype", "fp32"))})
    return spec


def build_inputs(name: str, wrapper, shapes_cfg: dict, scalars_cfg: dict):
    """Build CPU example inputs for one kernel.

    Returns (torch_tensors, tensor_defs, scalar_values) or raises
    ValueError (SkipError when the caller can fix it via CLI options) with
    the reason inputs cannot be constructed. Error paths raise BEFORE
    importing torch, so SKIP reporting works without torch installed.
    """
    scalar_values = _resolve_scalars(name, wrapper, scalars_cfg)
    tensor_defs, _ = getattr(wrapper, "_cached_signature")
    spec = shapes_cfg.get(name, shapes_cfg.get("*"))
    if spec is not None:
        if len(spec) != len(tensor_defs):
            raise ValueError(
                f"--shapes entry for {name!r} has {len(spec)} tensors, "
                f"signature expects {len(tensor_defs)}"
            )
        import torch

        tensors = []
        for t in spec:
            dt = _dtype_map().get(str(t["dtype"]).upper())
            if dt is None:
                raise ValueError(
                    f"--shapes entry for {name!r}: unknown dtype {t['dtype']!r}"
                )
            tensors.append(torch.zeros(list(t["shape"]), dtype=dt))
        return tensors, tensor_defs, scalar_values
    # no --shapes: require fully static shapes before touching torch
    for d in tensor_defs:
        shape = getattr(d, "shape", None)
        if not shape or not all(isinstance(dim, int) for dim in shape):
            snippet = json.dumps({"*": _shape_skeleton(tensor_defs)})
            raise SkipError(
                f"shape {shape!r} is not fully static; fill the <N> "
                f"placeholders in the --shapes snippet below to trace "
                f"this kernel (skipping trace, import+AST only)",
                suggestion=f"--shapes '{snippet}'",
            )
    import torch

    tensors = []
    for d in tensor_defs:
        tensors.append(torch.zeros(list(d.shape), dtype=_torch_dtype(d.dtype)))
    return tensors, tensor_defs, scalar_values


def trace_one(name: str, wrapper, shapes_cfg: dict, scalars_cfg: dict) -> dict:
    """Run the frontend trace for one kernel. Returns a result dict."""
    try:
        tensors, tensor_defs, scalar_values = build_inputs(
            name, wrapper, shapes_cfg, scalars_cfg
        )
    except ValueError as e:
        res = {"kernel": name, "status": "SKIP", "detail": str(e)}
        if isinstance(e, SkipError) and e.suggestion:
            res["suggestion"] = e.suggestion
        return res
    import pypto.pypto_impl as pypto_impl

    try:
        pypto_impl.Reset()  # isolate repeated traces in the same process
        if scalar_values:
            # the channel JitCallableWrapper.__call__ uses for scalar params
            wrapper.kwargs = scalar_values
        wrapper.compile(tensors, tensor_defs)
    except Exception as e:  # noqa: BLE001 — report every trace-time failure
        return {
            "kernel": name,
            "status": "ERROR",
            "detail": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
    return {"kernel": name, "status": "OK"}


def import_impl(impl_path, module_name="smoke_impl_under_check"):
    """Import the impl file as a module; sibling imports resolvable.

    Bytecode caching is disabled so the check never leaves __pycache__
    next to the impl file.
    """
    sys.path.insert(0, str(impl_path.parent))
    old_dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, impl_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dwb
        try:
            sys.path.remove(str(impl_path.parent))
        except ValueError:
            pass
    return module


_CLI_EPILOG = """\
input-spec JSON (only needed when a kernel is SKIPPED — the SKIP notice
prints a ready-to-paste retry command with the snippet filled in):
  --shapes   {"<kernel>": [{"shape": [64], "dtype": "fp32"}, ...],
              "*": [...]}     one entry per tensor param (incl. out)
  --scalars  {"eps": 1e-6}  or  {"<kernel>": {"eps": 1e-6}, "*": {...}}
             priority per param: kernel-specific > "*" > flat;
             resolution order: --scalars > declared default >
             type literal (float->1.0, int->1, bool->True)
  both options also accept @file.json to read the JSON from a file

long compiles (large kernels take minutes in compile()):
  python smoke_check_impl.py <impl.py> --bg
  re-launches detached; poll the printed smoke.result file (or
  `tail -f` the smoke.log) instead of blocking a foreground tool call.
"""


def _add_input_args(ap) -> None:
    """Arguments describing the kernel's inputs and scratch space."""
    ap.add_argument("impl", nargs="?", help="path to the impl .py file to smoke-check")
    ap.add_argument(
        "--shapes",
        default=None,
        help="JSON input specs for kernels with non-static shapes, or @file.json",
    )
    ap.add_argument(
        "--scalars",
        default=None,
        help="JSON values for non-tensor (scalar) params, or @file.json",
    )
    ap.add_argument(
        "--workdir",
        default=None,
        help="scratch dir for pypto's output/ scratch (default: "
        "./_debug/smoke_trace under the current working "
        "directory); cleaned up after the run. NOTE: "
        "tempfile.TemporaryDirectory() is forbidden here — "
        "it lands in the system temporary directory and "
        "triggers sandbox prompts.",
    )


def _add_mode_args(ap) -> None:
    """Arguments selecting how the check runs and reports."""
    ap.add_argument(
        "--bg",
        action="store_true",
        help="re-launch detached: log to "
        "_debug/smoke_check/smoke.log, final verdict to "
        "smoke.result; print both paths and return "
        "immediately",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="treat SKIP (inputs could not be auto-built) as failure (exit 1)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="print a machine-readable JSON object before the SMOKE RESULT line",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="run the script's built-in self-checks and exit",
    )
    # internal: set when we are the detached child of a --bg run
    ap.add_argument(
        "--_fg", dest="fg_child", action="store_true", help=argparse.SUPPRESS
    )


def _parse_cli(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Host-side smoke check for a pypto impl file "
        "(AST + import + frontend trace/codegen; no NPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EPILOG,
    )
    _add_input_args(ap)
    _add_mode_args(ap)
    return ap.parse_args(argv)


def _load_json_cfg(raw, opt_name: str) -> dict:
    """Parse a --shapes/--scalars option: inline JSON or @file.json.

    Raises ValueError (incl. JSONDecodeError) on unreadable file or bad
    JSON; the caller maps that to exit code 2.
    """
    if not raw:
        return {}
    if raw.startswith("@"):
        path = Path(raw[1:])
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError(f"--{opt_name}: cannot read {path}: {e}") from e
    return json.loads(raw)


def _parse_input_cfgs(args) -> bool:
    """Parse --shapes/--scalars into args.<opt>_cfg. False on bad input."""
    for opt_name in ("shapes", "scalars"):
        try:
            cfg = _load_json_cfg(getattr(args, opt_name), opt_name)
        except ValueError as e:  # JSONDecodeError included
            _log.error(f"[SMOKE ERROR] --{opt_name}: {e}")
            return False
        if not isinstance(cfg, dict):
            _log.error(f"[SMOKE ERROR] --{opt_name} must be a JSON object")
            return False
        setattr(args, opt_name + "_cfg", cfg)
    return True


def _ast_precheck(impl: Path) -> bool:
    """Syntax precheck. False (error already printed) on SyntaxError."""
    try:
        ast.parse(impl.read_text(encoding="utf-8"), filename=str(impl))
    except SyntaxError as e:
        _log.error(f"[SMOKE ERROR] stage=ast: SyntaxError: {e}")
        return False
    _log.info("[SMOKE] stage=ast: OK")
    return True


def _prepare_workdir(workdir_arg) -> Path:
    """Create a fresh CWD-local scratch directory to avoid sandbox prompts."""
    workdir = (
        Path(workdir_arg).resolve()
        if workdir_arg
        else (Path.cwd() / "_debug" / "smoke_trace")
    )
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def _cleanup_workdir(workdir: Path, old_cwd: str) -> None:
    os.chdir(old_cwd)
    shutil.rmtree(workdir, ignore_errors=True)
    try:
        workdir.parent.rmdir()  # remove _debug too, but only if empty
    except OSError:
        pass


def _write_result_file(result_path: Path, exit_code: int, result_line: str) -> None:
    """Write the --bg verdict atomically (tmp + rename) for pollers."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = result_path.with_name(result_path.name + ".tmp")
    tmp.write_text(f"exit_code={exit_code}\n{result_line}\n", encoding="utf-8")
    os.replace(tmp, result_path)


def _relaunch_detached(args, impl: Path) -> int:
    """--bg entry: spawn the real check detached, print poll info, return."""
    run_dir = Path.cwd() / _BG_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / _BG_LOG
    result_path = run_dir / _BG_RESULT
    if result_path.exists():
        result_path.unlink()  # drop a stale verdict from a previous run
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_fg", str(impl)]
    for opt in ("shapes", "scalars", "workdir"):
        val = getattr(args, opt)
        if val:
            cmd += [f"--{opt}", val]
    if args.strict:
        cmd.append("--strict")
    if args.json:
        cmd.append("--json")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            start_new_session=True,
        )
    _log.info(f"[SMOKE] --bg: detached pid={proc.pid}")
    _log.info(f"[SMOKE] --bg: log={log_path}")
    _log.info(
        f"[SMOKE] --bg: result={result_path} — appears on completion "
        "with 'exit_code=N' + the SMOKE RESULT line; poll for it"
    )
    _log.info(f"[SMOKE] --bg: follow live output with: tail -f {log_path}")
    return 0


def _trace_kernels(
    module, shapes_cfg: dict, scalars_cfg: dict, retry_base: str
) -> tuple:
    """Trace every module-level jit kernel.

    Returns (results, n_errors, n_skips); every SKIP is followed by a
    [SMOKE HINT] line with a ready-to-paste retry command when the kernel
    reported how to fix its inputs.
    """
    results = []
    n_errors = 0
    n_skips = 0
    kernels = find_jit_kernels(module)
    if not kernels:
        _log.info(
            "[SMOKE] no @pypto.frontend.jit kernel found at module "
            "level; import+AST only"
        )
    for name, wrapper in kernels:
        res = trace_one(name, wrapper, shapes_cfg, scalars_cfg)
        results.append(res)
        if res["status"] == "OK":
            _log.info(f"[SMOKE] stage=trace kernel={name}: TRACE OK")
        elif res["status"] == "SKIP":
            n_skips += 1
            _log.info(f"[SMOKE SKIP] kernel={name}: {res['detail']}")
            if res.get("suggestion"):
                _log.info(
                    f"[SMOKE HINT] kernel={name} retry: "
                    f"{retry_base} {res['suggestion']}"
                )
        else:
            n_errors += 1
            _log.error(f"[SMOKE ERROR] stage=trace kernel={name}: {res['detail']}")
            if res.get("traceback"):
                _log.info(res["traceback"].rstrip())
    return results, n_errors, n_skips


def _finish(
    args,
    exit_code: int,
    result_line: str,
    kernels: list | None = None,
    extra: dict | None = None,
) -> int:
    """Emit --json payload + conclusion line; write the --bg result file.

    Must be called with the caller's cwd already restored (the result
    file lives under the original cwd, not the trace workdir).
    """
    if args.json:
        payload = {
            "file": str(args.impl),
            "exit_code": exit_code,
            "result": result_line,
            "kernels": kernels or [],
        }
        if extra:
            payload.update(extra)
        _log.info(json.dumps(payload, ensure_ascii=False))
    if exit_code:
        _log.error(result_line)
    else:
        _log.info(result_line)
    if args.fg_child:
        _write_result_file(Path.cwd() / _BG_DIR / _BG_RESULT, exit_code, result_line)
    return exit_code


def _conclude(args, impl: Path, results: list, n_errors: int, n_skips: int) -> int:
    """Turn trace tallies into the verdict line + exit code."""
    extra = {"errors": n_errors, "skips": n_skips, "strict": args.strict}
    if n_errors or (args.strict and n_skips):
        line = f"SMOKE RESULT: FAIL errors={n_errors} skips={n_skips} file={impl}"
        if args.strict and n_skips and not n_errors:
            line += " (--strict: skips count as failure)"
        return _finish(args, 1, line, results, extra)
    note = " (trace skipped for some kernels: import+AST only)" if n_skips else ""
    return _finish(
        args, 0, f"SMOKE RESULT: PASS skips={n_skips}{note} file={impl}", results, extra
    )


def _self_test_cli_basic(root: Path, script: str, check) -> None:
    """CLI-level self-checks on throwaway fixtures (no kernels needed)."""
    fixtures = {
        "bad_syntax.py": "def broken(:\n",
        "import_error.py": "raise RuntimeError('boom at import')\n",
        "no_kernels.py": "X = 1\n",
        "cfgs.json": json.dumps({"*": [{"shape": [4], "dtype": "fp32"}]}),
    }
    for fname, content in fixtures.items():
        (root / fname).write_text(content, encoding="utf-8")

    def run_cli(*cli_args) -> tuple:
        proc = subprocess.run(
            [sys.executable, script, *cli_args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.returncode, proc.stdout

    rc, out = run_cli("bad_syntax.py")
    check(
        "syntax error -> exit 1 + FAIL",
        rc == 1 and "SMOKE RESULT: FAIL" in out,
        out[-200:],
    )
    rc, out = run_cli("import_error.py")
    check(
        "import error -> exit 1 + FAIL", rc == 1 and "stage=import" in out, out[-200:]
    )
    rc, out = run_cli("no_kernels.py")
    check(
        "no jit kernels -> exit 0 + PASS",
        rc == 0 and "SMOKE RESULT: PASS" in out,
        out[-200:],
    )
    rc, out = run_cli("no_kernels.py", "--shapes", "not json")
    check("invalid JSON -> exit 2", rc == 2, out[-200:])
    rc, out = run_cli("no_kernels.py", "--scalars", "[1]")
    check("non-object JSON -> exit 2", rc == 2, out[-200:])
    rc, out = run_cli("no_kernels.py", "--shapes", "@cfgs.json")
    check("@file.json accepted", rc == 0, out[-200:])
    rc, out = run_cli("missing.py")
    check("missing file -> exit 2", rc == 2, out[-200:])


def _self_test_cli_bg(root: Path, script: str, check) -> None:
    """--json and --bg self-checks (subprocess on the no-kernel fixture)."""

    def run_cli(*cli_args) -> tuple:
        proc = subprocess.run(
            [sys.executable, script, *cli_args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.returncode, proc.stdout

    rc, out = run_cli("no_kernels.py", "--json")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    try:
        payload = json.loads(lines[-2])
        json_ok = payload["exit_code"] == 0 and lines[-1].startswith("SMOKE RESULT")
    except (IndexError, json.JSONDecodeError, KeyError):
        json_ok = False
    check("--json: JSON object before conclusion line", json_ok, out[-200:])

    # --bg: foreground returns at once; result file appears shortly
    rc, out = run_cli("no_kernels.py", "--bg")
    result_path = root / _BG_DIR / _BG_RESULT
    deadline = time.time() + 60
    while not result_path.exists() and time.time() < deadline:
        time.sleep(0.5)
    bg_ok = rc == 0 and result_path.exists()
    if bg_ok:
        content = result_path.read_text(encoding="utf-8")
        bg_ok = "exit_code=0" in content and "SMOKE RESULT: PASS" in content
    check("--bg: result file with exit code + PASS", bg_ok, out[-200:])


def _self_test_resolution(check) -> None:
    """In-process checks of scalar/shape resolution with fake wrappers."""

    class _FakeWrapper:
        _cached_signature = ([], ["eps", "ratio", "debug"])
        _cached_non_tensor_defaults = {"ratio": 0.5}
        _create_parser = None

        @staticmethod
        def _original_func(x, eps, ratio=0.5, debug: bool = False):
            # NOTE: declared defaults reach the resolver via the cached
            # non-tensor defaults (as real wrappers cache them); ``debug``
            # falls through to the type literal — the fake cache omits it
            # on purpose.
            pass

        def compile(self, *a):
            pass

    vals = _resolve_scalars("k", _FakeWrapper(), {"eps": 1e-6})
    check(
        "scalar priority: cli > default > type literal",
        vals == {"eps": 1e-6, "ratio": 0.5, "debug": True},
        repr(vals),
    )
    try:
        _resolve_scalars("k", _FakeWrapper(), {"nope": 1})
        check("unknown --scalars key rejected", False)
    except ValueError as e:
        check("unknown --scalars key rejected", "unknown" in str(e), str(e))

    class _FakeMissing(_FakeWrapper):
        _cached_signature = ([], ["eps"])
        _cached_non_tensor_defaults = {}

        @staticmethod
        def _original_func(x, eps):
            pass

    try:
        _resolve_scalars("k", _FakeMissing(), {})
        check("unresolvable scalar -> SkipError with snippet", False)
    except SkipError as e:
        check(
            "unresolvable scalar -> SkipError with snippet",
            bool(e.suggestion) and "--scalars" in e.suggestion,
            str(e),
        )

    class _FakeTensorDef:
        shape = [8, "N"]  # one dynamic dim
        dtype = "DT_FP32"

    class _FakeDyn(_FakeWrapper):
        _cached_signature = ([_FakeTensorDef()], [])

    try:
        build_inputs("k", _FakeDyn(), {}, {})
        check("dynamic shape -> SkipError with --shapes skeleton", False)
    except SkipError as e:
        check(
            "dynamic shape -> SkipError with --shapes skeleton",
            bool(e.suggestion)
            and "--shapes" in e.suggestion
            and '"<N>"' in e.suggestion,
            str(e),
        )


def _self_test() -> int:
    """Built-in self-checks; no pypto / NPU / external op required.

    Prints [SELF-TEST] lines; exit 0 iff all checks pass.
    """
    results = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        results.append(bool(ok))
        (_log.error if not ok else _log.info)(
            f"[SELF-TEST] {'PASS' if ok else 'FAIL'} {label}"
            + (f" — {detail}" if detail and not ok else "")
        )

    script = str(Path(__file__).resolve())
    root = Path.cwd() / _BG_DIR / "selftest"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        _self_test_cli_basic(root, script, check)
        _self_test_cli_bg(root, script, check)
        _self_test_resolution(check)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        for parent in (root.parent, root.parent.parent):  # _debug if empty
            try:
                parent.rmdir()
            except OSError:
                pass

    n_fail = results.count(False)
    (_log.error if n_fail else _log.info)(
        f"SELF-TEST RESULT: {'PASS' if not n_fail else 'FAIL'} "
        f"checks={len(results)} failures={n_fail}"
    )
    return 1 if n_fail else 0


def main(argv=None) -> int:
    args = _parse_cli(argv)
    if args.self_test:
        return _self_test()
    if not args.impl:
        _log.error("[SMOKE ERROR] missing impl file argument (or run with --self-test)")
        return 2
    impl = Path(args.impl).resolve()
    if not impl.is_file():
        _log.error(f"[SMOKE ERROR] file not found: {impl}")
        return 2
    if not _parse_input_cfgs(args):
        return 2
    if args.bg and not args.fg_child:
        return _relaunch_detached(args, impl)

    # ---- 1. AST-level precheck (syntax) ----
    if not _ast_precheck(impl):
        return _finish(
            args,
            1,
            f"SMOKE RESULT: FAIL errors=1 file={impl}",
            extra={"errors": 1, "skips": 0},
        )

    # ---- 2+3. import + trace, inside a throwaway cwd so pypto's output/
    # scratch dir lands there instead of polluting the caller's cwd ----
    workdir = _prepare_workdir(args.workdir)
    old_cwd = os.getcwd()
    os.chdir(workdir)
    results, n_errors, n_skips, fatal = [], 0, 0, False
    try:
        try:
            module = import_impl(impl)
        except Exception:  # noqa: BLE001
            _log.error(f"[SMOKE ERROR] stage=import: {traceback.format_exc(limit=3)}")
            n_errors, fatal = 1, True
        else:
            _log.info("[SMOKE] stage=import: OK")
            retry_base = f"python {Path(__file__).resolve()} {impl} --bg"
            results, n_errors, n_skips = _trace_kernels(
                module, args.shapes_cfg, args.scalars_cfg, retry_base
            )
    finally:
        _cleanup_workdir(workdir, old_cwd)

    if fatal:
        return _finish(
            args,
            1,
            f"SMOKE RESULT: FAIL errors=1 file={impl}",
            extra={"errors": 1, "skips": 0},
        )
    return _conclude(args, impl, results, n_errors, n_skips)


if __name__ == "__main__":
    sys.exit(main())
