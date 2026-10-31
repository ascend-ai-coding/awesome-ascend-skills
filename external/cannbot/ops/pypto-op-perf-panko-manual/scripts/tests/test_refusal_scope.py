# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A compiler refusal is evidence about the structure it was measured on.

`error_signature` keeps ErrCode + Enum + PassName and nothing else, and `F4FFFF`
is the generic FeError code that this repo's experience table maps to several
causes. So a signature being stable across occurrences of the same refusal --
which is what its docstring claims -- does not establish that two refusals
sharing one share a cause. Carrying a refusal to a structure it was never
measured on can therefore delete a legal tile from that program's space,
permanently and without a measurement.

Structure-scoped by default, promoted on evidence rather than on assumption.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import harness  # noqa: E402


PARAMS = {"vec#0": {"d0": 16384}}
SIG = "F4FFFF|InternalError::PASS_INNER_ERROR|PadLocalBuffer.Tensor"
OTHER = {"vec#0": {"d0": 8192}}


def _observed():
    return [{"params": PARAMS, "sig": SIG}]


class RefusalScopeTest(unittest.TestCase):
    def test_a_first_sighting_is_scoped_to_its_structure(self):
        rows = harness.merge_refusals([], _observed(), "structA")
        self.assertEqual(rows[0]["scope"], "structure")
        self.assertEqual(rows[0]["structures"], ["structA"])

    def test_it_applies_on_the_structure_it_was_measured_on(self):
        rows = harness.merge_refusals([], _observed(), "structA")
        self.assertEqual(len(harness.refusals_for_structure(rows, "structA")), 1)

    def test_it_does_not_apply_to_a_structure_it_was_never_seen_on(self):
        """The regression: this used to reject the tile without measuring it."""
        rows = harness.merge_refusals([], _observed(), "structA")
        self.assertEqual(harness.refusals_for_structure(rows, "structB"), [])

    def test_a_second_distinct_structure_promotes_it(self):
        """Seen surviving a restructure, so now it is carried rather than assumed."""
        rows = harness.merge_refusals([], _observed(), "structA")
        rows = harness.merge_refusals(rows, _observed(), "structB")
        self.assertEqual(rows[0]["scope"], "global")
        self.assertEqual(rows[0]["structures"], ["structA", "structB"])
        self.assertEqual(len(harness.refusals_for_structure(rows, "structC")), 1)

    def test_the_same_structure_twice_does_not_promote(self):
        """A retry after a busy device is one sighting, not two."""
        rows = harness.merge_refusals([], _observed(), "structA")
        rows = harness.merge_refusals(rows, _observed(), "structA")
        self.assertEqual(rows[0]["scope"], "structure")
        self.assertEqual(rows[0]["structures"], ["structA"])

    def test_promote_after_one_restores_the_old_behaviour(self):
        rows = harness.merge_refusals([], _observed(), "structA", promote_after=1)
        self.assertEqual(rows[0]["scope"], "global")

    def test_refusals_on_other_structures_are_not_lost_by_a_merge(self):
        """The driver is handed only this structure's subset and folds its new
        rows into THAT. Assigning its result back would drop the rest.
        """
        rows = harness.merge_refusals([], [{"params": OTHER, "sig": SIG}], "structA")
        rows = harness.merge_refusals(rows, _observed(), "structB")
        self.assertEqual(len(rows), 2)
        keys = {tuple(sorted(r["params"]["vec#0"].items())) for r in rows}
        self.assertEqual(len(keys), 2)

    def test_a_legacy_row_matches_nothing_until_re_earned(self):
        """Rows written before the field existed carry no structure. Trusting
        them would delete a tile from a program they were never observed on; the
        cost of distrusting them is one repeated device trial.
        """
        st = {"bayesian_optimization_refusals": [{"params": PARAMS, "sig": SIG}]}
        rows = harness.stored_refusals(st)
        self.assertEqual(rows[0]["scope"], "structure")
        self.assertEqual(rows[0]["structures"], [])
        self.assertEqual(harness.refusals_for_structure(rows, "structA"), [])

    def test_malformed_rows_are_dropped(self):
        st = {
            "bayesian_optimization_refusals": [
                {"params": PARAMS},
                {"sig": SIG},
                {},
                {"params": PARAMS, "sig": SIG},
            ]
        }
        self.assertEqual(len(harness.stored_refusals(st)), 1)

    def test_the_state_distinguishes_the_two_scopes(self):
        """`scope` is what a reader of search_state.json needs to tell a
        structure-scoped refusal from a globally invariant one.
        """
        rows = harness.merge_refusals([], _observed(), "structA")
        rows = harness.merge_refusals(rows, [{"params": OTHER, "sig": SIG}], "structA")
        rows = harness.merge_refusals(rows, [{"params": OTHER, "sig": SIG}], "structB")
        scopes = {r["params"]["vec#0"]["d0"]: r["scope"] for r in rows}
        self.assertEqual(scopes, {16384: "structure", 8192: "global"})


if __name__ == "__main__":
    unittest.main()
