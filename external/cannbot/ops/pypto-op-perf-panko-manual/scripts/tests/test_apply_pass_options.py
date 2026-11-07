# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`pass_options` discovery and rewriting.

Two defects, and the second is why every assertion here ends in `compile()`:
counting sites proves the knob was found, not that what was written is a
program. A rewrite that emits a second `pass_options=` kwarg produces the right
number of sites and a file that does not parse.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import SCRIPTS_DIR  # noqa: F401, E402

import bayesian_optimization as bayesian  # noqa: E402

NBUF = "pass:cube_nbuffer_setting#0"
L1RE = "pass:cube_l1_reuse_setting#0"

NO_KWARG = """import pypto


@pypto.frontend.jit(runtime_options={"device_sched_mode": 1})
def k(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""

EXISTING = """import pypto


@pypto.frontend.jit(runtime_options={"device_sched_mode": 1}, pass_options={"cube_nbuffer_setting": {-1: 4}})
def k(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""


def _pass_site_ids(src):
    return sorted(s.id for s in bayesian.apply.discover_sites(src) if s.kind == "pass")


class PassOptionsRewriteTest(unittest.TestCase):
    # ── the knob is absent from the decorator entirely ────────────────────────

    def test_single_knob_writes_the_kwarg(self):
        out = self._apply(NO_KWARG, {NBUF: {"value": 4}}, 1)
        self.assertIn('pass_options={"cube_nbuffer_setting": {-1: 4}}', out)

    def test_both_knobs_share_one_kwarg(self):
        """Regression: these used to be two edits at one offset.

        Each wrote a complete `pass_options=`, so selecting both produced
        `keyword argument repeated: pass_options` and the candidate never
        reached a compiler.
        """
        out = self._apply(NO_KWARG, {NBUF: {"value": 4}, L1RE: {"value": 2}}, 2)
        self.assertIn('"cube_nbuffer_setting": {-1: 4}', out)
        self.assertIn('"cube_l1_reuse_setting": {-1: 2}', out)

    def test_unset_knob_is_not_written(self):
        """UNSET is the original kernel, which must stay a reachable point."""
        out, n = bayesian.apply.apply(
            NO_KWARG,
            {
                NBUF: {"value": bayesian.apply.UNSET},
                L1RE: {"value": bayesian.apply.UNSET},
            },
        )
        self.assertEqual(n, 0)
        self.assertEqual(out, NO_KWARG)

    # ── the decorator already carries `{-1: N}` ───────────────────────────────

    def test_existing_literal_is_discovered(self):
        """Regression: `-1` is a UnaryOp, not a Constant.

        The configured knob matched neither the tunable form nor the "missing"
        form, so it was offered as no site at all and the search could not see
        a pass configuration the kernel already had.
        """
        self.assertEqual(_pass_site_ids(EXISTING), [L1RE, NBUF])

    def test_existing_literal_warm_starts_at_its_value(self):
        cfg = bayesian.apply.current_config(EXISTING)
        self.assertEqual(cfg[NBUF], {"value": 4})  # the value, not the -1 key
        self.assertEqual(cfg[L1RE], {"value": bayesian.apply.UNSET})

    def test_existing_literal_is_retunable(self):
        out = self._apply(EXISTING, {NBUF: {"value": 16}}, 1)
        self.assertIn('"cube_nbuffer_setting": {-1: 16}', out)
        self.assertNotIn("{-1: 4}", out)

    def test_second_knob_joins_the_existing_dict(self):
        out = self._apply(EXISTING, {NBUF: {"value": 16}, L1RE: {"value": 8}}, 2)
        self.assertIn('"cube_nbuffer_setting": {-1: 16}', out)
        self.assertIn('"cube_l1_reuse_setting": {-1: 8}', out)

    # ── determinism: the code hash is the search's identity for a candidate ──

    def test_rewrite_is_byte_stable(self):
        cfg = {NBUF: {"value": 4}, L1RE: {"value": 2}}
        first, _ = bayesian.apply.apply(NO_KWARG, cfg)
        for _ in range(5):
            again, _ = bayesian.apply.apply(NO_KWARG, dict(reversed(list(cfg.items()))))
            self.assertEqual(again, first)

    def _apply(self, src, cfg, expect_sites):
        out, n = bayesian.apply.apply(src, cfg)
        self.assertEqual(n, expect_sites)
        # The whole point: the candidate must be a program.
        compile(out, "<candidate>", "exec")
        # One decorator carries at most one `pass_options=`.
        self.assertEqual(out.count("pass_options="), 1, out)
        return out


if __name__ == "__main__":
    unittest.main()
