# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A Cube action moves Cube tiles, and nothing else.

The four tile actions are defined over one site kind each -- the catalogue says
"Cube TileShape" and "Vector tile shapes" -- but the lever tuned every cube AND
vec site whichever one was selected. Selecting F-9 could change a Vector tile,
and whatever it gained was booked to the Cube action.

The union search is still available and still wanted; it is `block`, which the
optimizer asks for deliberately and which is recorded as a block. What the four
actions must not do is perform it silently.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import SCRIPTS_DIR  # noqa: F401, E402

from bayesian_optimization import lever, space  # noqa: E402

BOTH = """import pypto


@pypto.frontend.jit()
def k(a, b):
    pypto.set_cube_tile_shapes([128, 64], [64, 64], [128, 64])
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""

VEC_ONLY = """import pypto


@pypto.frontend.jit()
def k(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""

HW = space.HW(calibrated=False, dtype_bytes=4)
_OPTS = lever.AskOptions(dtype_bytes=4, current_latency=100.0)


def _kinds(cfg):
    return {k.split("#", 1)[0] for k in cfg}


class ScopeTest(unittest.TestCase):
    def test_each_action_declares_the_kind_it_is_defined_over(self):
        self.assertEqual(lever.scope_of("F-9"), "cube")
        self.assertEqual(lever.scope_of("S-11"), "cube")
        self.assertEqual(lever.scope_of("F-10"), "vec")
        self.assertEqual(lever.scope_of("S-12"), "vec")

    def test_an_unknown_action_pools_into_nothing(self):
        """None means "every tile site", which is what `block` does on purpose.
        An id the table does not know must not silently acquire a scope, nor
        share a study with one.
        """
        self.assertIsNone(lever.scope_of("F-7"))
        self.assertNotEqual(lever.lever_key("F-7", BOTH), lever.lever_key("F-9", BOTH))


class StudySharingTest(unittest.TestCase):
    def test_the_two_cube_actions_share_one_study(self):
        """F-9 and S-11 are both "tune the Cube tile". Keyed by action they kept
        separate studies of the same space and paid device evaluations twice to
        rebuild one surrogate.
        """
        self.assertEqual(lever.lever_key("F-9", BOTH), lever.lever_key("S-11", BOTH))

    def test_the_two_vector_actions_share_one_study(self):
        self.assertEqual(lever.lever_key("F-10", BOTH), lever.lever_key("S-12", BOTH))

    def test_cube_and_vector_do_not(self):
        self.assertNotEqual(lever.lever_key("F-9", BOTH), lever.lever_key("F-10", BOTH))

    def test_the_structural_signature_still_separates_programs(self):
        """Sharing is by scope only. The per-structure isolation is the
        separation that has evidence behind it and it is untouched.
        """
        self.assertNotEqual(
            lever.lever_key("F-9", BOTH), lever.lever_key("F-9", VEC_ONLY)
        )


class SeedScopeTest(unittest.TestCase):
    def test_a_cube_action_seeds_only_cube_sites(self):
        pts = lever.seed_points(BOTH, HW, 4, 100.0, scope="cube")
        self.assertTrue(pts)
        for cfg, _, _ in pts:
            self.assertEqual(_kinds(cfg), {"cube"})

    def test_a_vector_action_seeds_only_vec_sites(self):
        pts = lever.seed_points(BOTH, HW, 4, 100.0, scope="vec")
        self.assertTrue(pts)
        for cfg, _, _ in pts:
            self.assertEqual(_kinds(cfg), {"vec"})

    def test_no_scope_still_seeds_everything(self):
        """`block` asks for the union deliberately; scoping the actions must not
        take that away.
        """
        pts = lever.seed_points(BOTH, HW, 4, 100.0)
        self.assertEqual(_kinds(pts[0][0]), {"cube", "vec"})


class AskScopeTest(unittest.TestCase):
    def test_the_cube_action_hands_back_a_cube_only_config(self):
        cfg, meta = lever.ask({}, "F-9", BOTH, HW, _OPTS)
        self.assertIsNotNone(cfg)
        self.assertEqual(_kinds(cfg), {"cube"})
        self.assertEqual(meta["scope"], "cube")

    def test_the_vector_action_hands_back_a_vec_only_config(self):
        cfg, meta = lever.ask({}, "F-10", BOTH, HW, _OPTS)
        self.assertIsNotNone(cfg)
        self.assertEqual(_kinds(cfg), {"vec"})
        self.assertEqual(meta["scope"], "vec")

    def test_a_cube_action_on_a_kernel_with_no_cube_work_declines(self):
        """And says so in those words. "No tunable tile site" on a kernel that
        has vec sites reads as a broken lever; "no cube tile site" is the Cube
        action correctly declining, and the optimizer should retire it.
        """
        cfg, meta = lever.ask({}, "F-9", VEC_ONLY, HW, _OPTS)
        self.assertIsNone(cfg)
        self.assertEqual(meta["reason"], "no cube tile site")

    def test_the_vector_action_still_works_on_that_kernel(self):
        cfg, _ = lever.ask({}, "F-10", VEC_ONLY, HW, _OPTS)
        self.assertIsNotNone(cfg)


class BufferCeilingTest(unittest.TestCase):
    """The tile ceiling is UB/2, and UB is the RUN'S."""

    def test_it_is_half_the_envelope_s_unified_buffer(self):
        self.assertEqual(
            lever.ceil_kb(space.HW(calibrated=False, ub_budget_kb=192)), 96
        )
        self.assertEqual(
            lever.ceil_kb(space.HW(calibrated=False, ub_budget_kb=248)), 124
        )

    def test_a5_tiles_are_no_longer_refused_by_an_a3_constant(self):
        """A 100 KB tile fits a 950's 248 KB UB and not a 910B3's 192 KB. The
        hardcoded 96 refused it on both -- the defect the chip envelope exists
        to remove, surviving in the one module that kept its own copy.
        """
        self.assertGreater(
            100, lever.ceil_kb(space.HW(calibrated=False, ub_budget_kb=192))
        )
        self.assertLess(
            100, lever.ceil_kb(space.HW(calibrated=False, ub_budget_kb=248))
        )

    def test_no_envelope_falls_back_to_the_documented_a3_figure(self):
        self.assertEqual(lever.ceil_kb(None), lever.CEIL_KB)


if __name__ == "__main__":
    unittest.main()
