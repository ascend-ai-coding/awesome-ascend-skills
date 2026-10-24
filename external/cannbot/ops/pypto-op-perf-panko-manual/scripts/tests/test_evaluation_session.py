# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Worker lifetime and complete test execution without requiring an NPU."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
evaluation_session = importlib.import_module("evaluation_session")
worker = importlib.import_module("session_worker")
measure_latency = importlib.import_module("measure_latency")
RealEvaluator = importlib.import_module("bayesian_optimization.evaluator").RealEvaluator


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.impl = self.root / "demo_impl.py"
        self.impl.write_text("def demo_wrapper(): return 42\n")
        self.test = self.root / "test_demo.py"
        self.test.write_text("raise SystemExit(2)\n")
        self.evaluator = RealEvaluator(
            "demo",
            str(self.root),
            str(self.impl),
            "0",
            f"{sys.executable} {self.test}",
            eval_timeout_s=10,
        )
        self.addCleanup(evaluation_session.close_session, self.root)

    def session(self):
        return Path(
            json.loads(
                (self.root / "optimization/.evaluation/active.json").read_text()
            )["session"]
        )

    def config(self):
        return {k: getattr(self.evaluator, k) for k in evaluation_session.CONFIG_FIELDS}

    def test_worker_reuse_dependency_change_and_cleanup(self):
        self.assertEqual(evaluation_session.request(self.evaluator)["test_rc"], 2)
        first = self.session()
        self.impl.write_text("def demo_wrapper(): return 43\n")
        self.assertEqual(evaluation_session.request(self.evaluator)["test_rc"], 2)
        self.assertEqual(first, self.session())
        self.test.write_text(
            'import os; os.write(2, b"native failure"); raise SystemExit("exit detail")\n'
        )
        response = evaluation_session.request(self.evaluator)
        self.assertEqual(response["test_rc"], 1)
        self.assertIn("native failure", response["test_output"])
        self.assertIn("exit detail", response["test_output"])
        self.assertNotEqual(first, self.session())
        self.assertFalse(worker.worker_alive(first))
        evaluation_session.close_session(self.root)
        self.assertFalse(worker.worker_alive(self.session()))

    def test_timeout_releases_worker_and_next_request_recovers(self):
        self.test.write_text("import time; time.sleep(30)\n")
        self.evaluator.eval_timeout_s = 0.3
        self.assertEqual(evaluation_session.request(self.evaluator)["test_rc"], 124)
        self.assertFalse(worker.worker_alive(self.session()))
        self.test.write_text("raise SystemExit(2)\n")
        self.evaluator.eval_timeout_s = 10
        self.assertEqual(evaluation_session.request(self.evaluator)["test_rc"], 2)

    def test_correctness_and_measurement_share_the_module(self):
        self.test.write_text("import demo_impl; demo_impl.tested = True\n")

        def build(*args, **kwargs):
            self.assertTrue(args[2].tested)
            self.assertEqual(args[2].demo_wrapper(), 42)
            return args[2].demo_wrapper

        with (
            patch.object(measure_latency, "build_run_once", side_effect=build),
            patch.object(
                measure_latency, "measure", return_value={"p": 1, "raw_count": 4}
            ),
        ):
            result = evaluation_session.evaluate_candidate(self.config())
        self.assertEqual(result["metrics"]["raw_count"], 4)

    def test_failed_step_skips_following_steps_and_measurement(self):
        command = f'{sys.executable} {self.test} && {sys.executable} -c "raise RuntimeError()"'
        self.assertEqual(evaluation_session.run_python_tests(command, "0"), 2)
        with patch.object(measure_latency, "build_run_once") as build:
            self.assertEqual(
                evaluation_session.evaluate_candidate(self.config())["test_rc"], 2
            )
            self.evaluator.test_command = f'{sys.executable} -c "pass"'
            with patch.object(
                RealEvaluator, "lint_result", return_value=(False, "lint failure")
            ):
                self.assertEqual(
                    evaluation_session.evaluate_candidate(self.config())["lint_rc"], 1
                )
            build.assert_not_called()
        with self.assertRaises(ValueError):
            evaluation_session.run_python_tests("sh test_demo.py", "0")
