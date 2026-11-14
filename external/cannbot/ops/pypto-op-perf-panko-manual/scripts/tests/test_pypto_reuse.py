# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""What is taken from pypto, and what deliberately is not.

Two figures the skill used to keep its own copy of now come from the installed
package: the dtype width table, and the cube / vector core counts. Both have to
keep working on a box with no pypto -- the offline test suite is such a box --
so each has a fallback, and what is under test is that the pypto answer WINS
where both exist and that the fallback is reached where it does not.

pypto is stubbed rather than imported. The real package loads the runtime's
shared objects and answers for whatever silicon is in the box, neither of which
a test can assert against.
"""

from __future__ import annotations

import enum
import itertools
import os
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import write_platform_ini  # noqa: E402

import chip_profile  # noqa: E402
import feasibility  # noqa: E402
import swimlane  # noqa: E402
from bayesian_optimization import space  # noqa: E402

KB = 1024
SOC = "AscendTestA"


class _FakeDataType(enum.Enum):
    """The members the hand-written table was missing, plus one it had."""

    DT_FP32 = 1
    DT_FP8 = 2
    DT_INT4 = 3


_FAKE_WIDTHS = {"DT_FP32": 4, "DT_FP8": 1, "DT_INT4": 0}


def _fake_bytes_of(member):
    return _FAKE_WIDTHS.get(member.name, 0)


def _fake_pypto():
    mod = types.ModuleType("pypto")
    mod.DataType = _FakeDataType
    mod.bytes_of = _fake_bytes_of
    return mod


class _StubbedPypto:
    """`import pypto` resolves to `module` inside the block, and to nothing when
    `module` is None -- so the absent case is tested on a box that has it.
    """

    def __init__(self, module):
        self.module = module
        self._saved = None
        self._had = False

    def __enter__(self):
        self._had = "pypto" in sys.modules
        self._saved = sys.modules.get("pypto")
        if self.module is None:
            sys.modules["pypto"] = None
        else:
            sys.modules["pypto"] = self.module
        feasibility._DTYPE_TABLE = None
        return self

    def __exit__(self, *exc):
        if self._had:
            sys.modules["pypto"] = self._saved
        else:
            sys.modules.pop("pypto", None)
        feasibility._DTYPE_TABLE = None
        return False


class DtypeTableTest(unittest.TestCase):
    def test_widths_come_from_pypto_when_it_is_importable(self):
        with _StubbedPypto(_fake_pypto()):
            table = feasibility.dtype_bytes_table()
        self.assertEqual(table.get("DT_FP8"), 1)

    def test_a_sub_byte_width_is_dropped_rather_than_stored_as_zero(self):
        """A 0 in this table would make a tile of that dtype look free."""
        with _StubbedPypto(_fake_pypto()):
            table = feasibility.dtype_bytes_table()
        self.assertIsNone(table.get("DT_INT4"))

    def test_the_offline_table_is_reached_when_pypto_is_absent(self):
        with _StubbedPypto(None):
            table = feasibility.dtype_bytes_table()
        self.assertEqual(table.get("DT_FP32"), 4)
        self.assertIsNone(table.get("DT_FP8"))

    def test_a_name_only_pypto_knows_changes_the_footprint_width(self):
        """The bug this replaced: `DT_FP8` was not in the table, so it was not
        read, and the width fell through to the 2-byte default -- twice the
        truth, which rejects legal tiles statically.
        """
        src = "x: pypto.Tensor[DT_FP8]\npypto.set_vec_tile_shapes(8, 512)\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "k.py"
            path.write_text(src, encoding="utf-8")
            with _StubbedPypto(None):
                self.assertEqual(feasibility.extract(str(path))["dtype_bytes"], 2)
            with _StubbedPypto(_fake_pypto()):
                self.assertEqual(feasibility.extract(str(path))["dtype_bytes"], 1)


def _fake_bindings(cube, vector):
    mod = types.ModuleType("pypto")
    impl = types.SimpleNamespace(
        GetAICCoreNum=lambda: cube, GetAIVCoreNum=lambda: vector
    )
    mod.pypto_impl = impl
    return mod


class CoreCountTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dirs = [str(self._tmp.name)]
        write_platform_ini(
            Path(self._tmp.name), SOC, cube=4, vector=8, ub=32 * KB, l1=64 * KB
        )
        # `resolve` needs a name before it can find an ini, and no driver here
        # will give it one. Named explicitly rather than inherited, so the run
        # does not depend on the caller's environment.
        patch = unittest.mock.patch.dict(os.environ, {"PANKO_SOC_NAME": SOC})
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_the_live_count_wins_over_the_ini(self):
        """The ini answers for the SKU the name resolved to; the bindings answer
        for the device that is up. Two SKUs can differ here and in nothing else.
        """
        with _StubbedPypto(_fake_bindings(20, 40)):
            env, why = chip_profile.resolve(dirs=self.dirs, device=None)
        self.assertIsNotNone(env, why)
        self.assertEqual((env.get("cube_cores"), env.get("vector_cores")), (20, 40))
        self.assertEqual(env.get("cores_via"), "pypto_impl")

    def test_the_ini_is_used_when_the_bindings_are_absent(self):
        with _StubbedPypto(None):
            env, why = chip_profile.resolve(dirs=self.dirs, device=None)
        self.assertIsNotNone(env, why)
        self.assertEqual((env.get("cube_cores"), env.get("vector_cores")), (4, 8))
        self.assertEqual(env.get("cores_via"), "ini")

    def test_a_refusing_binding_is_not_a_count(self):
        """A driver that declines is not evidence; the ini still answers."""
        mod = types.ModuleType("pypto")
        mod.pypto_impl = types.SimpleNamespace(
            GetAICCoreNum=lambda: (_ for _ in ()).throw(RuntimeError("no device")),
            GetAIVCoreNum=lambda: 40,
        )
        with _StubbedPypto(mod):
            env, _ = chip_profile.resolve(dirs=self.dirs, device=None)
        self.assertEqual(env.get("cube_cores"), 4)
        self.assertEqual(env.get("cores_via"), "ini")

    def test_buffer_sizes_are_never_taken_from_the_bindings(self):
        """`GetMemoryLimitForArch` is keyed on the compilation arch, which maps
        910B and 910C to one value -- the granularity this module exists to stop
        using. Only the counts are taken live.
        """
        with _StubbedPypto(_fake_bindings(20, 40)):
            env, _ = chip_profile.resolve(dirs=self.dirs, device=None)
        self.assertEqual(env.get("ub_kb"), 32)
        self.assertTrue(env.get("source", "").endswith(f"{SOC}.ini"))


# Exactly what pypto's `calculate_pipe_usage` writes: five rows of four columns
# under one header, then a per-core block of twelve. Built the way the writer
# builds it, because the format is the contract -- a reader that only ever saw
# its own output would pass while disagreeing with the writer.
_CORE_HEADER = "Core, TotalTime, " + ", ".join(
    f"{p}_Time, {p}_Usage" for p in swimlane.PIPES
)

PIPE_USAGE_CSV = "\n".join(
    [
        "Total Core Num:3",
        "AIC: 1",
        "AIV: 2",
        "Total Pipe Usage",
        "Pipe, AverageTime, TotalExecuteTime, AverageUsage",
        "CUBE, 305.2, 412.5, 73.99%",
        "VECTOR_ALU, 160.9, 412.5, 39.01%",
        "MTE_IN, 363.0, 412.5, 88.0%",
        "MTE1, 49.5, 412.5, 12.0%",
        "MTE_OUT, 86.6, 412.5, 21.0%",
        "",
        "",
        "AICore Pipe Usage",
        _CORE_HEADER,
        "AIC_0, 412.5, 363.0, 88.0%, 49.5, 12.0%, 86.6, 21.0%, 305.2, 73.99%, 0.0, 0.0%",
        "",
    ]
)


class PipeUsageTest(unittest.TestCase):
    """`pipe_usage.csv` is pypto's own per-pipe breakdown, read rather than
    re-derived. It is not a substitute for `busy_frac`: that says the cores were
    occupied, this says by what.
    """

    def test_every_pipe_comes_back_as_a_fraction(self):
        got = swimlane.pipe_usage(str(self._write(PIPE_USAGE_CSV)))
        self.assertEqual(
            got,
            {
                "pipe_cube_frac": 0.7399,
                "pipe_vector_alu_frac": 0.3901,
                "pipe_mte_in_frac": 0.88,
                "pipe_mte1_frac": 0.12,
                "pipe_mte_out_frac": 0.21,
            },
        )

    def test_the_per_core_block_and_the_header_are_not_read_as_pipes(self):
        """Both sit in the same file; one has twelve columns, the other a name
        this does not know.
        """
        got = swimlane.pipe_usage(str(self._write(PIPE_USAGE_CSV)))
        self.assertEqual(len(got), len(swimlane.PIPES))

    def test_a_non_finite_usage_is_dropped_rather_than_carried(self):
        """The writer divides by a core count and by a span. A symptom must stay
        quiet rather than fire on `inf`.
        """
        text = PIPE_USAGE_CSV.replace(
            "CUBE, 305.2, 412.5, 73.99%", "CUBE, 305.2, 0.0, inf%"
        )
        got = swimlane.pipe_usage(str(self._write(text)))
        self.assertIsNone(got.get("pipe_cube_frac"))
        self.assertEqual(got.get("pipe_mte_in_frac"), 0.88)

    def test_a_missing_file_yields_nothing_rather_than_zeros(self):
        """`--gen_exe_topo_json` is not on by default, so absence is the norm."""
        self.assertEqual(swimlane.pipe_usage("/nonexistent/pipe_usage.csv"), {})
        self.assertEqual(swimlane.find_pipe_usage(""), "")

    def test_it_is_found_beside_the_trace(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        (out / "pipe_usage.csv").write_text(PIPE_USAGE_CSV, encoding="utf-8")
        trace = out / "merged_swimlane.json"
        trace.write_text("{}", encoding="utf-8")
        self.assertTrue(swimlane.find_pipe_usage(str(trace)).endswith("pipe_usage.csv"))

    def _write(self, text):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "pipe_usage.csv"
        path.write_text(text, encoding="utf-8")
        return path


# The four cube rules as `tools/scripts/tuner/tuner.py: HeuristicTile.is_good_tiling`
# states them, transcribed here ONLY because `tools/` is not part of the installed
# package -- the wheel ships `pypto/lib/scripts/` and no tuner -- so there is
# nothing to import. Keeping them written out makes the agreement testable
# instead of asserted in a comment.
#
# The capacities are passed in rather than defaulted: `HeuristicTile.__init__`
# hardcodes `l1_size=131072` / `l0c_size=524288`, which is the transcription this
# PR removed in favour of the per-SoC ini.
def _heuristic_tile_is_good(tile, hw, out_bytes=4):
    m, _m_big, k, k_big, n, n_big = tile
    in_bytes = hw.dtype_bytes
    r1 = k_big % k == 0 and n_big >= n
    r2 = m * n * out_bytes <= hw.l0c_kb * 1024
    r3 = n * k * in_bytes <= hw.l0b_kb * 1024
    r4 = m * k * in_bytes <= hw.l0a_kb * 1024
    return r1 and r2 and r3 and r4


def _our_gate(cfg, hw):
    """`static_feasible` routes a config carrying `mL0` to the cube rules, so the
    public entry reaches them and the private one need not be touched.
    """
    ok, _why = space.static_feasible({"cube#0": cfg}, hw)
    return ok


class CubeRuleAgreementTest(unittest.TestCase):
    """Anything pypto's tuner calls infeasible, our gate must also reject.

    One direction only. Our gate is deliberately stricter -- it requires `kL0`
    16-aligned, ceil-aligns every dimension before measuring a footprint, and
    checks the L1/L0 relation on all three axes rather than on k alone -- so the
    converse does not hold and is not asserted.
    """

    HW = space.HW(
        calibrated=False,
        dtype_bytes=2,
        l0a_kb=64,
        l0b_kb=64,
        l0c_kb=128,
        l1_budget_kb=512,
    )

    def test_no_config_the_tuner_rejects_passes_our_gate(self):
        hw = self.HW
        checked = 0
        for cfg in self._grid():
            tile = (
                cfg["mL0"],
                cfg["mL1"],
                cfg["kL0"],
                cfg["kL1"],
                cfg["nL0"],
                cfg["nL1"],
            )
            good = _heuristic_tile_is_good(tile, hw)
            if good:
                continue
            checked += 1
            self.assertFalse(_our_gate(cfg, hw), cfg)
        self.assertGreater(checked, 0, "the grid exercised no rejection")

    def test_the_grid_also_contains_configs_both_accept(self):
        """Otherwise the implication above is vacuous."""
        hw = self.HW
        agreed = 0
        for cfg in self._grid():
            tile = (
                cfg["mL0"],
                cfg["mL1"],
                cfg["kL0"],
                cfg["kL1"],
                cfg["nL0"],
                cfg["nL1"],
            )
            if _heuristic_tile_is_good(tile, hw) and _our_gate(cfg, hw):
                agreed += 1
        self.assertGreater(agreed, 0)

    def _grid(self):
        axes = itertools.product(
            (16, 64, 128, 256), (16, 32, 128), (16, 64, 256), (1, 3)
        )
        for m, k, n, mult in axes:
            yield {
                "mL0": m,
                "mL1": m * mult,
                "kL0": k,
                "kL1": k * mult,
                "nL0": n,
                "nL1": n * mult,
            }


if __name__ == "__main__":
    unittest.main()
