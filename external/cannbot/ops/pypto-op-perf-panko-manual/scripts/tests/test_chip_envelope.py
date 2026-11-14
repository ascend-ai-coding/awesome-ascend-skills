# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""One chip envelope, consumed by every static rule.

There were two. `feasibility.HW` and `space.HW` each carried their own defaults
and disagreed on L1 -- 192 against 512, where space.py's own comment says the 192
"was an old conservative estimate that has no basis now that the real buffer is
checked". Only the feasibility side was reachable from the CLI, so `--l1-kb`
moved one gate while the tile search kept using the other.

Also pins a defect class rather than one defect: a module in the package being
shadowed by a parameter of the same name. The `bo_` prefix used to keep
`bo_domain` and a `domain` argument apart by accident, and dropping it removed
the accident.
"""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import SCRIPTS_DIR, harness  # noqa: E402

import feasibility  # noqa: E402
import bayesian_optimization as bayesian  # noqa: E402


KERNEL = """import pypto


@pypto.frontend.jit()
def k(a, b):
    pypto.set_cube_tile_shapes([128, 64], [64, 64], [128, 64])
    pypto.set_vec_tile_shapes(64, 128)
    return a + b


def w(x):
    return k(x, x)
"""


class OneEnvelopeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.op_file = Path(self._tmp.name) / "k.py"
        self.op_file.write_text(KERNEL, encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def test_the_two_gates_agree_on_l1(self):
        """The regression: 192 here against 512 there, a factor of 2.7."""
        self.assertEqual(
            feasibility.HW().l1_budget_kb, bayesian.space.HW().l1_budget_kb
        )

    def test_the_envelope_reaches_the_tile_search(self):
        """`hw_for` passed nothing, so the search used dataclass defaults and the
        CLI could not move it.
        """
        env = dict(
            harness.CHIP_ENVELOPE,
            ub_kb=248,
            l1_kb=1024,
            l0c_kb=256,
            cube_cores=28,
            vector_cores=56,
        )
        hw, _ = bayesian.block.hw_for(str(self.op_file), None, env)
        self.assertEqual(hw.ub_budget_kb, 248)
        self.assertEqual(hw.l1_budget_kb, 1024)
        self.assertEqual(hw.l0c_kb, 256)
        self.assertEqual(hw.cube_cores, 28)
        self.assertEqual(hw.vector_cores, 56)

    def test_the_envelope_reaches_the_derived_domain(self):
        env = dict(harness.CHIP_ENVELOPE, ub_kb=248)
        _, domain = bayesian.block.hw_for(str(self.op_file), None, env)
        self.assertEqual(domain.get("ub_bytes"), 248 * 1024)

    def test_a_run_carries_its_own_envelope(self):
        """A resumed campaign keeps validating against what it started with."""
        env = harness.chip_envelope({"chip_envelope": {"ub_kb": 248, "l0c_kb": 256}})
        self.assertEqual(env["ub_kb"], 248)
        self.assertEqual(env["l0c_kb"], 256)
        self.assertEqual(env["l1_kb"], harness.CHIP_ENVELOPE["l1_kb"])

    def test_no_envelope_falls_back_to_the_defaults(self):
        self.assertEqual(harness.chip_envelope(None), dict(harness.CHIP_ENVELOPE))


def _sibling_imports(tree, modules):
    """Names this module binds by importing a sibling module of the package."""
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and (n.module is None or n.level):
            imported |= {
                (a.asname or a.name) for a in n.names if (a.asname or a.name) in modules
            }
    return imported


def _bound_names(fn):
    """Every name a function binds: its parameters and its assignment targets."""
    bound = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    bound |= {
        n.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    return bound


def _functions(tree):
    """Every function defined in this module, nested ones included."""
    return [
        fn
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _module_reads(fn, clash):
    """`mod.attr` reads inside one function, for the modules it shadows."""
    out = []
    for n in ast.walk(fn):
        if not isinstance(n, ast.Attribute) or not isinstance(n.value, ast.Name):
            continue
        if n.value.id in clash:
            out.append((n.lineno, n.value.id, n.attr))
    return out


def _shadowed_in(name, tree, imported):
    """Every `mod.attr` read inside a function that also binds `mod` itself."""
    offenders = []
    for fn in _functions(tree):
        clash = _bound_names(fn) & imported
        for lineno, mod, attr in _module_reads(fn, clash):
            offenders.append(
                f"{name}:{lineno} {fn.name}() uses {mod}.{attr} but binds {mod}"
            )
    return offenders


def _called_names(fn):
    """The plain names this function calls."""
    out = set()
    for c in ast.walk(fn):
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name):
            out.add(c.func.id)
    return out


def _self_shadowed_in(name, tree):
    """Every function that calls a module-level function whose name it also binds."""
    top = {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    offenders = []
    for fn in _functions(tree):
        for shadowed in sorted(_bound_names(fn) & top & _called_names(fn)):
            offenders.append(
                f"{name}:{fn.lineno} {fn.name}() calls "
                f"{shadowed}() but also binds {shadowed}"
            )
    return offenders


class NoShadowedModuleTest(unittest.TestCase):
    """A sibling module must not be shadowed by a name the function binds.

    A parameter shadows a module for the whole function body, so
    `domain.derive(...)` inside a function taking `domain` is an
    UnboundLocalError -- at call time, not at import, which is why the suite did
    not catch it. `block.run` and `driver.run_bo` were both in that state.
    """

    def test_no_function_shadows_a_module_it_uses(self):
        pkg = SCRIPTS_DIR / "bayesian_optimization"
        modules = {p.stem for p in pkg.glob("*.py")} - {"__init__"}
        offenders = []
        for f in sorted(pkg.glob("*.py")):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            offenders += _shadowed_in(f.name, tree, _sibling_imports(tree, modules))
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_no_function_shadows_a_function_it_calls(self):
        """The same defect one level down, and it has bitten this branch twice.

        A local named after a module-level function makes every later call in
        that body a call on the local: `dom = GLOBAL_DOMAINS[...]` followed by
        `dom(dom, val)` raises TypeError on a list. It does not fail at import,
        and it only runs on the branch that binds the local -- which in both
        recorded cases needed a device. Renaming a private helper to a public
        name is how it got introduced both times, so the check is over the whole
        skill rather than one package.
        """
        offenders = []
        for f in sorted(SCRIPTS_DIR.rglob("*.py")):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            offenders += _self_shadowed_in(f.name, tree)
        self.assertEqual(offenders, [], "\n".join(offenders))


class ShadowedCallsRunTest(unittest.TestCase):
    """The functions that were broken, actually called."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.op_file = Path(self._tmp.name) / "k.py"
        self.op_file.write_text(KERNEL, encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def test_hw_for_runs(self):
        hw, domain = bayesian.block.hw_for(str(self.op_file))
        self.assertTrue(hw.ub_budget_kb)
        self.assertIsInstance(domain, dict)

    def test_memory_key_runs(self):
        _, domain = bayesian.block.hw_for(str(self.op_file))
        key = bayesian.block.memory_key("u1", domain)
        self.assertTrue(key.startswith("u1|"))

    def test_vec_domain_runs_and_keeps_its_argument(self):
        """It rebound the parameter to the module and then passed the module in
        its place.
        """
        _, domain = bayesian.block.hw_for(str(self.op_file))
        vals = bayesian.space.vec_domain("vec#0", "d0", ["d0", "d1"], {}, domain)
        self.assertTrue(vals)
        self.assertTrue(all(isinstance(v, int) for v in vals))


if __name__ == "__main__":
    unittest.main()
