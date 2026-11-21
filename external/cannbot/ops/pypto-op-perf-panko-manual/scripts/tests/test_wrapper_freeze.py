# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The host wrapper is frozen for the run, and the harness enforces it.

`p` counts AICore time inside the @jit kernel. That is a sound objective exactly
while the uncounted part is identical between the baseline and the candidate,
because then it cancels in the comparison. Move work across the boundary and the
kernel gets cheaper, the wrapper gets more expensive, `p` improves, and the
ratchet keeps a candidate that is slower to run.

Documenting the rule is not enough: the coder is handed free-form intents, so the
rule has to be checkable. These pin both directions -- a kernel rewrite must pass,
a wrapper edit must not.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import (
    CATALOG,
    CHIP_ARGS,
    HARNESS,
    feasible_argv,
    file_hash,
    harness,
    init_argv,
    read_state,
    record_argv,
    run_cli,
)


KERNEL = """import torch
import pypto

S2_TILE = 128


@pypto.frontend.jit(runtime_options={"device_sched_mode": 1})
def demo_kernel(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b


def demo_wrapper(x):
    y = torch.empty_like(x)
    demo_kernel(x, y)
    return y
"""


class WrapperSignatureTest(unittest.TestCase):
    """What counts as the wrapper, and what does not."""

    def setUp(self):
        self.base = harness.wrapper_signature(KERNEL)

    # ── kernel edits must be invisible: they are the whole point of the search ──

    def test_a_tile_rewrite_is_not_wrapper_drift(self):
        self.assertTrue(
            self._same(
                KERNEL.replace(
                    "set_vec_tile_shapes(64, 128)", "set_vec_tile_shapes(32, 256)"
                )
            )
        )

    def test_a_decorator_option_is_not_wrapper_drift(self):
        """pass_options / runtime_options live on the kernel's decorator."""
        self.assertTrue(
            self._same(
                KERNEL.replace('"device_sched_mode": 1', '"device_sched_mode": 2')
            )
        )

    def test_comments_and_formatting_are_invisible(self):
        """Hashed over the AST, so a reformat is not a violation."""
        self.assertTrue(self._same(KERNEL + "\n# trailing note\n"))
        self.assertTrue(
            self._same(
                KERNEL.replace("    return a + b", "    # note\n    return a + b")
            )
        )

    # ── host edits must be caught ────────────────────────────────────────────

    def test_a_wrapper_body_change_is_drift(self):
        self.assertFalse(
            self._same(
                KERNEL.replace("y = torch.empty_like(x)", "y = torch.zeros_like(x)")
            )
        )

    def test_work_moved_into_the_wrapper_is_drift(self):
        """The shape the review describes: the kernel gets cheaper because the
        host now does the work.
        """
        self.assertFalse(
            self._same(
                KERNEL.replace(
                    "    y = torch.empty_like(x)",
                    "    x = torch.cat([x, x], dim=0)\n    y = torch.empty_like(x)",
                )
            )
        )

    def test_a_module_constant_is_drift(self):
        """S2_TILE-style constants feed the wrapper's padding: that is host work."""
        self.assertFalse(self._same(KERNEL.replace("S2_TILE = 128", "S2_TILE = 256")))

    def test_an_added_import_is_drift(self):
        """Deliberate erring. Deciding which statements are 'work' is the
        judgement the check exists to avoid.
        """
        self.assertFalse(
            self._same(KERNEL.replace("import torch", "import torch\nimport numpy"))
        )

    def test_an_unparseable_candidate_is_not_silently_passed(self):
        self.assertIsNone(harness.wrapper_signature("def ("))

    def _same(self, src):
        return harness.wrapper_signature(src) == self.base


class WrapperFreezeGateTest(unittest.TestCase):
    """The gates that act on it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.op_dir = Path(self._tmp.name) / "demo"
        self.op_dir.mkdir(parents=True)
        self.op_file = self.op_dir / "demo_impl.py"
        self.op_file.write_text(KERNEL, encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)
        res, rc = self._run(*init_argv(self.op_dir, self.op_file, *CHIP_ARGS))
        self.assertTrue(res["initialized"], res)

    def test_init_records_the_baseline_wrapper(self):
        self.assertTrue(self._state()["wrapper_signature"])

    def test_feasible_rejects_a_drifted_candidate_before_the_device(self):
        self.op_file.write_text(
            KERNEL.replace("y = torch.empty_like(x)", "y = torch.zeros_like(x)"),
            encoding="utf-8",
        )
        res, _ = self._run(*feasible_argv(self.op_dir, self.op_file))
        self.assertFalse(res["feasible"])
        self.assertFalse(res["wrapper_frozen"])
        self.assertEqual(self._state()["progress"]["wrapper_drift_rejected"], 1)

    def test_feasible_passes_a_kernel_only_candidate(self):
        self.op_file.write_text(
            KERNEL.replace(
                "set_vec_tile_shapes(64, 128)", "set_vec_tile_shapes(32, 256)"
            ),
            encoding="utf-8",
        )
        res, _ = self._run(*feasible_argv(self.op_dir, self.op_file))
        self.assertTrue(res["feasible"], res)

    def test_record_discards_a_measurement_taken_on_a_drifted_wrapper(self):
        """`feasible` is the cheap gate and is not mandatory, so the ratchet
        checks too -- such a measurement is not comparable with the campaign's.
        """
        drifted = KERNEL.replace("y = torch.empty_like(x)", "y = torch.zeros_like(x)")
        self.op_file.write_text(drifted, encoding="utf-8")
        digest = file_hash(self.op_file)
        res, rc = self._run(*record_argv(self.op_dir, digest, "10"))
        self.assertEqual(rc, 3)
        self.assertEqual(res["error"], "wrapper_drift")
        self.assertFalse(res["improved"])
        # A spectacular p must not have become the best.
        self.assertEqual(self._state()["progress"]["best_latency_us"], 500.0)

    def _run(self, *args):
        return run_cli(*args)

    def _state(self):
        return read_state(self.op_dir)


class CatalogueScopeTest(unittest.TestCase):
    def test_the_boundary_crossing_actions_are_gone(self):
        """F-16 / F-21 / F-22 moved work from the kernel to the host. With the
        wrapper frozen they are unreachable, and offering them would be offering
        a candidate the gates reject.
        """
        ids = {a["id"] for a in json.loads(CATALOG.read_text(encoding="utf-8"))}
        for gone in ("F-16", "F-21", "F-22"):
            self.assertNotIn(gone, ids)

    def test_no_symptom_still_prescribes_one(self):
        src = HARNESS.read_text(encoding="utf-8")
        for gone in ('"F-16"', '"F-21"', '"F-22"'):
            self.assertNotIn(gone, src)


if __name__ == "__main__":
    unittest.main()
