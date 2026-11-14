# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`infeasible_K` must be reachable from an action's FIRST candidate.

Both counters it bounds are moved by `feasible`, which runs before any device
measurement, while the record they live in used to be created only by `record`.
An action whose whole opening neighbourhood is rejected therefore never got a
record, both counters read 0 forever, and the bound could not fire -- so the
optimizer went on selecting an action that had never produced a measurable
candidate.

These drive the real CLI rather than calling internals, because the defect was
in which command owns the record.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import CHIP_ARGS, file_hash, run_cli  # noqa: E402

FEASIBLE_KERNEL = """import pypto


@pypto.frontend.jit(runtime_options={"device_sched_mode": 1})
def demo(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""

# `cube tile dim 17 is not 16-aligned` -- a certain violation, which is the only
# kind the static gate is allowed to reject on.
INFEASIBLE_KERNEL = """import pypto


@pypto.frontend.jit(runtime_options={"device_sched_mode": 1})
def demo(a, b):
    pypto.set_cube_tile_shapes([128, 64], [17, 64], [128, 64])
    return a + b
"""

INFEASIBLE_K = 2


def _run(*args):
    """The payload only: these tests read counters, not exit codes."""
    return run_cli(*args)[0]


class RefineCounterTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.op_dir = Path(self._tmp.name) / "demo"
        self.op_dir.mkdir(parents=True)
        self.op_file = self.op_dir / "demo_impl.py"
        self.addCleanup(self._tmp.cleanup)

    def test_all_infeasible_from_the_first_candidate_reaches_the_bound(self):
        """No measurement ever happens, so `record` never runs. The bound must
        still fire.
        """
        self._init(INFEASIBLE_KERNEL)
        seen = [self._feasible() for _ in range(3)]
        self.assertEqual([r["feasible"] for r in seen], [False, False, False])
        self.assertEqual([r["n_static"] for r in seen], [1, 2, 3])
        self.assertEqual([r["stop_refine"] for r in seen], [False, True, True])
        self.assertEqual(seen[1]["stop_cause"], "infeasible_K")
        # `n` is untouched: nothing was measured, so nothing was learned about
        # the action. That is what makes this a separate, looser bound.
        self.assertEqual([r["n"] for r in seen], [0, 0, 0])

    def test_all_noop_from_the_first_candidate_reaches_the_bound(self):
        """Same shape, the other pre-device rejection: every candidate is the
        incumbent program written differently.
        """
        self._init(FEASIBLE_KERNEL)
        seen = []
        for i in range(3):
            with self.op_file.open("a", encoding="utf-8") as f:
                f.write(f"\n# comment {i}\n")  # different bytes, same program
            seen.append(self._feasible())
        self.assertTrue(all(r["semantic_noop"] for r in seen))
        self.assertEqual([r["n_noop"] for r in seen], [1, 2, 3])
        self.assertEqual([r["stop_refine"] for r in seen], [False, True, True])
        self.assertEqual(seen[1]["stop_cause"], "semantic_noop_K")

    def test_the_two_bounds_are_separate(self):
        """A no-op is not a hardware infeasibility; sharing one counter made the
        two indistinguishable in the state.
        """
        self._init(FEASIBLE_KERNEL)
        with self.op_file.open("a", encoding="utf-8") as f:
            f.write("\n# comment\n")
        res = self._feasible()
        self.assertEqual(res["n_noop"], 1)
        self.assertEqual(res["n_static"], 0)

    def test_parentless_call_does_not_create_or_clobber_a_refinement(self):
        """`--parent` defaults to empty. Such a call belongs to no action, so it
        must not open a record, nor overwrite the one in flight.
        """
        self._init(INFEASIBLE_KERNEL)
        self.assertEqual(self._feasible()["n_static"], 1)
        stray = self._feasible(parent=None)
        self.assertEqual(stray["n_static"], 0)  # nothing attributed
        state = json.loads(
            (self.op_dir / "optimization" / "search_state.json").read_text(
                encoding="utf-8"
            )
        )
        refine = state["progress"]["refine"]
        self.assertEqual(refine["action"], "u1")  # still u1's
        self.assertEqual(refine["n_static"], 1)  # and still 1
        self.assertEqual(self._feasible()["n_static"], 2)

    def _init(self, source):
        self.op_file.write_text(source, encoding="utf-8")
        digest = file_hash(self.op_file)
        res = _run(
            "init",
            "--op-dir",
            str(self.op_dir),
            "--op",
            "demo",
            "--p-ref",
            "100",
            "--preopt-p",
            "100",
            "--preopt-hash",
            digest,
            "--op-file",
            str(self.op_file),
            "--allow-unversioned",
            *CHIP_ARGS,
        )
        self.assertTrue(res["initialized"])
        state = self.op_dir / "optimization" / "search_state.json"
        st = json.loads(state.read_text(encoding="utf-8"))
        st["config"]["infeasible_K"] = INFEASIBLE_K
        state.write_text(json.dumps(st), encoding="utf-8")
        return res

    def _feasible(self, parent="u1"):
        args = [
            "feasible",
            "--op-dir",
            str(self.op_dir),
            "--op-file",
            str(self.op_file),
            "--ub-kb",
            "192",
        ]
        if parent is not None:
            args += ["--parent", parent]
        return _run(*args)


if __name__ == "__main__":
    unittest.main()
