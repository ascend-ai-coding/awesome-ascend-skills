# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`optuna` is a real dependency of the tile search, and of nothing else here.

Two things must never happen. A run configured for the tile search must not
proceed with the package missing -- the tile values would come back from the
model instead of TPE while the config, the log and the report all still said
the search was on. And an import failure that is NOT the optional dependency --
a circular import, a syntax error in this package, an optuna API change -- must
not be turned into "the tile search is off".
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import (
    CHIP_ARGS,
    HARNESS,
    file_hash,  # noqa: E402
    harness,
    init_argv,
    read_state,
    run_cli,
)


KERNEL = """import pypto


@pypto.frontend.jit()
def demo(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""


class MissingDependencyClassificationTest(unittest.TestCase):
    """Which import failures may be treated as "the optional dep is absent"."""

    def test_missing_optuna_is_the_optional_dependency(self):
        self.assertTrue(
            harness.is_missing_optuna(
                ModuleNotFoundError("No module named 'optuna'", name="optuna")
            )
        )

    def test_a_missing_optuna_submodule_counts_too(self):
        self.assertTrue(
            harness.is_missing_optuna(
                ModuleNotFoundError(
                    "No module named 'optuna.samplers'", name="optuna.samplers"
                )
            )
        )

    def test_another_missing_module_is_a_bug_in_this_package(self):
        self.assertFalse(
            harness.is_missing_optuna(
                ModuleNotFoundError("No module named 'scipy'", name="scipy")
            )
        )

    def test_an_api_change_is_a_bug_in_this_package(self):
        """`from optuna.trial import create_trial` failing is ImportError, not
        ModuleNotFoundError, and must reach a traceback.
        """
        self.assertFalse(
            harness.is_missing_optuna(
                ImportError("cannot import name 'create_trial' from 'optuna.trial'")
            )
        )

    def test_a_circular_import_is_a_bug_in_this_package(self):
        self.assertFalse(
            harness.is_missing_optuna(
                ImportError(
                    "cannot import name 'space' from partially initialized module"
                )
            )
        )


class CapabilityReportTest(unittest.TestCase):
    def test_capability_names_the_installed_version(self):
        cap = harness.bayesian_capability()
        self.assertIn("available", cap)
        self.assertIn("optuna_version", cap)
        if cap["available"]:
            self.assertIsNone(cap["reason"])
            self.assertTrue(cap["optuna_version"])
        else:
            self.assertTrue(cap["reason"])


class InitPreflightTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.op_dir = Path(self._tmp.name) / "demo"
        self.op_dir.mkdir(parents=True)
        self.op_file = self.op_dir / "demo_impl.py"
        self.op_file.write_text(KERNEL, encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def test_init_reports_the_capability(self):
        """The report has to name the algorithm that ran, not the one configured."""
        res, rc = self._init()
        self.assertEqual(rc, 0)
        self.assertIn("bayesian_optimization", res)
        self.assertIn("available", res["bayesian_optimization"])

    def test_the_capability_is_persisted_in_the_state(self):
        self._init()
        state = read_state(self.op_dir)
        self.assertIn("bayesian_optimization", state)

    def test_the_refusal_carries_a_question_for_the_user(self):
        """Refusing is right; deciding for the user is not.

        Simulated rather than driven, because hiding optuna from a subprocess
        cannot be done through PYTHONPATH: the harness puts its own scripts
        directory first on sys.path.
        """
        cap = {
            "available": False,
            "optuna_version": None,
            "reason": "optuna is not importable (No module named 'optuna')",
        }
        self.assertFalse(cap["available"])
        # The shape the skill keys off, asserted against the source so the
        # contract and the payload cannot drift apart silently.
        src = HARNESS.read_text(encoding="utf-8")
        self.assertIn('"ask_user"', src)
        self.assertIn('"relay the question to the user; do not choose for them"', src)
        for option in ("install", "ablation", "stepwise"):
            self.assertIn(f'"id": "{option}"', src)

    def test_the_ablation_arm_runs_without_the_tile_search(self):
        """`bayesian_optimization_tile_search: false` is the supported way to run
        without optuna, and it has to keep working.
        """
        res, rc = self._init({"bayesian_optimization_tile_search": False})
        self.assertEqual(rc, 0)
        self.assertTrue(res["initialized"])

    def _init(self, config=None):
        args = init_argv(
            self.op_dir,
            self.op_file,
            "--preopt-hash",
            file_hash(self.op_file),
            *CHIP_ARGS,
        )
        if config:
            args += ["--config", json.dumps(config)]
        return run_cli(*args)


if __name__ == "__main__":
    unittest.main()
