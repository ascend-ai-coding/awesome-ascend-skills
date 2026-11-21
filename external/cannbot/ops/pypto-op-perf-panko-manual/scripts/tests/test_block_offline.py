# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The tile block, driven with a stub objective and no device.

`block.run` and `driver.run_bo` were the two functions in this skill that
nothing exercised: they need optuna, and the reading of them was that they also
need an NPU. They do not. The objective is a callable the caller supplies, the
rest is AST work and TPE, and both are importable and deterministic here.

That matters because this is where the branch's two worst self-inflicted bugs
landed -- the `bo_domain` shadowing and the `space.dom` collision -- each of
which raised on the first real call and passed every test. These fix the
behaviour in place so that a change to the plumbing has to keep it: what is
proposed, how many trials are charged, which one is kept, and when it stops.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import SCRIPTS_DIR  # noqa: F401, E402

from bayesian_optimization import block, space  # noqa: E402

KERNEL = """import pypto


@pypto.frontend.jit()
def k(a, b):
    pypto.set_cube_tile_shapes([128, 256], [64, 128], [128, 256])
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""


def _tile_total(cfg):
    total = 0
    for site in cfg.values():
        for v in site.values():
            if isinstance(v, int):
                total += v
    return total


class BlockOfflineTest(unittest.TestCase):
    """One block over a two-site kernel, with a pure objective."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.op_file = Path(self._tmp.name) / "k.py"
        self.op_file.write_text(KERNEL, encoding="utf-8")
        self.hw = space.HW(calibrated=False, dtype_bytes=4)
        self.addCleanup(self._tmp.cleanup)

    def test_the_same_seed_proposes_the_same_sequence(self):
        """The search is deterministic given the objective. Every claim this
        skill makes about a recorded run rests on that, and it is also what
        makes a refactor of the plumbing checkable: thread an argument wrong and
        the sequence moves.
        """
        first = self._run()[1]
        second = self._run()[1]
        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)

    def test_every_charged_trial_is_in_the_history(self):
        """A trial the device paid for and the record does not show is how a
        budget silently disappears. The counts have to agree.
        """
        result, seen = self._run()
        measured = [h for h in result["history"] if h.get("status") == "ok"]
        self.assertEqual(len(measured), len(seen))

    def test_the_kept_config_is_the_cheapest_thing_the_block_saw(self):
        """Including the incumbent, which costs no trial. A block that improves
        on nothing keeps the program it was given, and says so.
        """
        result, _ = self._run()
        measured = [
            h["latency"]
            for h in result["history"]
            if h.get("status") == "ok" and h.get("latency") is not None
        ]
        self.assertTrue(measured)
        self.assertEqual(result["best_latency"], min(measured + [1000.0]))
        self.assertEqual(result["best_was_incumbent"], result["best_latency"] == 1000.0)

    def test_it_stops_on_stagnation_rather_than_on_the_budget(self):
        """A flat objective improves on nothing, so the block closes at
        `stagnation_k` with budget left. Reading `max_trials` as the stop would
        spend the whole allowance on a space that said nothing.
        """
        result, seen = self._run(
            objective=lambda cfg: (1, 500.0), stagnation_k=2, max_trials=40
        )
        self.assertTrue(result["stopped_early"])
        self.assertLess(len(seen), 40)

    def test_it_does_not_exceed_the_budget_it_was_given(self):
        result, seen = self._run(stagnation_k=99, max_trials=5)
        self.assertLessEqual(len(seen), 5)
        self.assertLessEqual(result["n_trials"], 5)

    def test_a_kernel_with_no_tile_call_is_named_rather_than_empty(self):
        flat = Path(self._tmp.name) / "flat.py"
        flat.write_text("import pypto\n\n\ndef k(a):\n    return a\n", encoding="utf-8")
        result = block.run(str(flat), lambda cfg: (1, 1.0), self.hw)
        self.assertIn("no tunable tile site", result["reason"])

    def _run(self, objective=None, stagnation_k=3, max_trials=8):
        seen = []

        def recording(cfg):
            seen.append(cfg)
            if objective is not None:
                return objective(cfg)
            return 1, float(1000 + (_tile_total(cfg) % 97))

        result = block.run(
            str(self.op_file),
            recording,
            self.hw,
            block.BlockOptions(
                stagnation_k=stagnation_k,
                max_trials=max_trials,
                current_latency=1000.0,
                seed=0,
            ),
        )
        return result, seen


if __name__ == "__main__":
    unittest.main()
