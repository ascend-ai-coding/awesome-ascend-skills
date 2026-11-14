# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The snapshot chain is a precondition, not a best effort.

Every cycle is anchored to `nodes/<code_hash>.py`: `select` restores the global
best into the working file and tells the coder to apply the delta on top of it,
`record` rolls a loser back to it, `stop` delivers it. When a link in that chain
is missing the harness used to carry on and report the failure in a field
nothing reads, so which program a candidate was actually built on depended on
the caller's path and on residue from earlier runs.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import CHIP_ARGS, file_hash, init_argv, record_argv, run_cli

KERNEL = """import pypto


@pypto.frontend.jit(runtime_options={"device_sched_mode": 1})
def demo(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""


def _run(*args):
    """(payload, returncode). Refusals exit non-zero but still emit JSON."""
    return run_cli(*args)


class SnapshotContractTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.op_dir = Path(self._tmp.name) / "demo"
        self.op_dir.mkdir(parents=True)
        self.op_file = self.op_dir / "demo_impl.py"
        self.op_file.write_text(KERNEL, encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    # ── init ──────────────────────────────────────────────────────────────────

    def test_init_derives_the_baseline_hash_from_the_file(self):
        """`--preopt-hash` is optional. Without it the run used to start with an
        empty root hash, an empty eval cache and no snapshot at all.
        """
        res, rc = self._init()
        self.assertEqual(rc, 0)
        self.assertTrue(res["initialized"])
        self.assertTrue(res["preopt_snapshot"])
        digest = file_hash(self.op_file)
        self.assertEqual(res["preopt_hash"], digest)
        self.assertEqual([p.stem for p in self._nodes()], [digest])

    def test_init_refuses_a_hash_that_does_not_match_the_file(self):
        """The caller believes it measured one program; the file is another. The
        preopt latency every speedup is normalised against is then not the
        baseline's.
        """
        res, rc = self._init("--preopt-hash", "deadbeef")
        self.assertEqual(rc, 3)
        self.assertFalse(res["initialized"])
        self.assertEqual(res["error"], "preopt_hash_mismatch")

    def test_init_accepts_a_hash_that_does_match(self):
        digest = file_hash(self.op_file)
        res, rc = self._init("--preopt-hash", digest)
        self.assertEqual(rc, 0)
        self.assertTrue(res["initialized"])

    def test_init_refuses_an_unreadable_op_file(self):
        res, rc = self._init(op_file=self.op_dir / "missing.py")
        self.assertEqual(rc, 3)
        self.assertEqual(res["error"], "op_file_unreadable")

    def test_a_refused_init_leaves_no_state_behind(self):
        """A half-built run whose baseline is missing is worse than no run."""
        self._init("--preopt-hash", "deadbeef")
        self.assertFalse((self.op_dir / "optimization" / "search_state.json").exists())

    # ── select ────────────────────────────────────────────────────────────────

    def test_select_refuses_when_the_best_cannot_be_restored(self):
        self._init()
        for snap in self._nodes():
            snap.unlink()
        with self.op_file.open("a", encoding="utf-8") as f:
            f.write("\n# drifted\n")
        res, rc = _run("select", "--op-dir", str(self.op_dir))
        self.assertEqual(rc, 3)
        self.assertIsNone(res["selected"])
        self.assertEqual(res["error"], "restore_failed")

    def test_select_still_works_when_the_chain_is_intact(self):
        self._init()
        res, rc = _run("select", "--op-dir", str(self.op_dir))
        self.assertEqual(rc, 0)
        self.assertIsNotNone(res["selected"])
        self.assertTrue(res["selected"]["restored"])
        self.assertTrue(res["selected"]["restore_code_hash"])

    # ── stop ──────────────────────────────────────────────────────────────────

    def test_stop_refuses_to_record_a_stop_it_cannot_deliver(self):
        """`stop_reason` is sticky and the completion gate reads it, so recording
        it over some other program would complete Stage 7 on a kernel the search
        never chose.
        """
        self._init()
        for snap in self._nodes():
            snap.unlink()
        with self.op_file.open("a", encoding="utf-8") as f:
            f.write("\n# drifted\n")
        res, rc = _run("stop", "--op-dir", str(self.op_dir), "--force", "manual")
        self.assertEqual(rc, 3)
        self.assertFalse(res["stop"])
        self.assertEqual(res["error"], "restore_failed")
        state = json.loads(
            (self.op_dir / "optimization" / "search_state.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIsNone(state["progress"]["stop_reason"])  # not recorded

    def test_stop_does_not_refuse_when_the_file_already_is_the_best(self):
        """No false refusal: a missing snapshot does not matter if the working
        file is already the program that would have been restored.
        """
        self._init()
        for snap in self._nodes():
            snap.unlink()
        res, rc = _run("stop", "--op-dir", str(self.op_dir), "--force", "manual")
        self.assertEqual(rc, 0)
        self.assertTrue(res["stop"])

    # ── record ────────────────────────────────────────────────────────────────

    def test_record_halts_when_a_loser_cannot_be_rolled_back(self):
        """A refinement submits candidate after candidate for one action without
        passing through `select`, so a break here would go unnoticed until the
        next select -- with everything in between built on a rejected program.
        """
        self._init()
        _run("select", "--op-dir", str(self.op_dir))
        for snap in self._nodes():
            snap.unlink()
        with self.op_file.open("a", encoding="utf-8") as f:
            f.write("\n# a losing candidate\n")
        digest = file_hash(self.op_file)
        res, rc = _run(*record_argv(self.op_dir, digest, "900"))
        self.assertEqual(rc, 3)
        self.assertFalse(res["improved"])
        self.assertEqual(res["error"], "restore_failed")
        # The measurement was paid for, so it is still recorded.
        state = json.loads(
            (self.op_dir / "optimization" / "search_state.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn(digest, state["eval_cache"])

    def test_record_is_quiet_when_the_chain_is_intact(self):
        self._init()
        _run("select", "--op-dir", str(self.op_dir))
        with self.op_file.open("a", encoding="utf-8") as f:
            f.write("\n# a losing candidate\n")
        digest = file_hash(self.op_file)
        res, rc = _run(*record_argv(self.op_dir, digest, "900"))
        self.assertEqual(rc, 0)
        self.assertNotIn("error", res)
        # and the loser really was rolled back
        self.assertEqual(self.op_file.read_text(encoding="utf-8"), KERNEL)

    def _init(self, *extra, op_file=None):
        return _run(
            *init_argv(self.op_dir, op_file or self.op_file, *CHIP_ARGS, *extra)
        )

    def _nodes(self):
        return sorted((self.op_dir / "optimization" / "nodes").glob("*.py"))


if __name__ == "__main__":
    unittest.main()
