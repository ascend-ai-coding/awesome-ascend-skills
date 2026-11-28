# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A search in progress must survive the `bo` -> `bayesian_optimization` rename.

State files outlive a rename: a run is resumed by the next `select`, reading a
search_state.json written by the build before it. The tile studies, the tile
memory and the compiler refusals all live there, and each of them cost device
time to produce.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import harness  # noqa: E402


def _old_state():
    return {
        "op": "demo",
        "config": {
            "tile_bo": True,
            "bo_free_trials": 3,
            "bo_startup_min_feasible": 5,
            "stagnation_K": 7,
        },
        "bo_studies": {"u1": {"trials": [{"value": 51.9}]}},
        "bo_memory": {"u1|abc123": {"trials": [{"value": 47.0}]}},
        "bo_refusals": [{"sig": "F4FFFF|ALLOC_FAILED|TilePass"}],
    }


class StateKeyMigrationTest(unittest.TestCase):
    def test_old_keys_are_renamed_and_values_kept(self):
        st = harness.migrate_keys(_old_state())
        self.assertEqual(st["config"]["bayesian_optimization_tile_search"], True)
        self.assertEqual(st["config"]["bayesian_optimization_free_trials"], 3)
        self.assertEqual(st["config"]["bayesian_optimization_startup_min_feasible"], 5)
        self.assertEqual(
            st["bayesian_optimization_studies"]["u1"]["trials"][0]["value"], 51.9
        )
        self.assertEqual(
            st["bayesian_optimization_memory"]["u1|abc123"]["trials"][0]["value"], 47.0
        )
        self.assertEqual(len(st["bayesian_optimization_refusals"]), 1)

    def test_old_keys_are_dropped_so_save_cannot_write_them_back(self):
        st = harness.migrate_keys(_old_state())
        self.assertNotIn("tile_bo", st["config"])
        self.assertNotIn("bo_free_trials", st["config"])
        self.assertNotIn("bo_studies", st)
        self.assertNotIn("bo_memory", st)
        self.assertNotIn("bo_refusals", st)

    def test_unrelated_keys_are_untouched(self):
        st = harness.migrate_keys(_old_state())
        self.assertEqual(st["config"]["stagnation_K"], 7)
        self.assertEqual(st["op"], "demo")

    def test_a_new_state_passes_through_unchanged(self):
        new = {
            "config": {"bayesian_optimization_tile_search": False},
            "bayesian_optimization_studies": {"u2": {}},
        }
        self.assertEqual(
            harness.migrate_keys(dict(new, config=dict(new["config"]))), new
        )

    def test_the_new_spelling_wins_if_a_file_carries_both(self):
        """Only reachable by hand-editing, but it must not be the old value."""
        st = harness.migrate_keys(
            {"config": {"tile_bo": True, "bayesian_optimization_tile_search": False}}
        )
        self.assertIs(st["config"]["bayesian_optimization_tile_search"], False)
        self.assertNotIn("tile_bo", st["config"])

    def test_a_state_with_no_config_does_not_raise(self):
        self.assertEqual(
            harness.migrate_keys({"bo_studies": {}}),
            {"bayesian_optimization_studies": {}},
        )


if __name__ == "__main__":
    unittest.main()
