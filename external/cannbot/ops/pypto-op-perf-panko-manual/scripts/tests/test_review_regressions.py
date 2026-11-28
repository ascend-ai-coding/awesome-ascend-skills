# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regressions for fixture invalidation, retryable lint and command imports."""

import importlib
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
session = importlib.import_module("evaluation_session")
worker = importlib.import_module("session_worker")
measure_latency = importlib.import_module("measure_latency")
evaluator_module = importlib.import_module("bayesian_optimization.evaluator")


class ReviewRegressionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.operator = self.root / "operator"
        self.operator.mkdir()
        self.source = self.operator / "demo_impl.py"
        self.source.write_text("def demo_wrapper(): return 42\n")
        self.test = self.operator / "test_demo.py"
        self.test.write_text(
            "from demo_impl import demo_wrapper\nassert demo_wrapper() == 42\n"
        )
        self.evaluator = evaluator_module.RealEvaluator(
            "demo",
            str(self.operator),
            str(self.source),
            "0",
            f'cd "{self.operator}" && "{sys.executable}" "{self.test}"',
            evaluation_mode="cached",
        )
        self.addCleanup(session.close_session, self.operator)

    def evaluate_in_process(self, root, config, fingerprint, payload, timeout):
        return session.evaluate_candidate(config)

    def test_changed_fixture_reruns_original_correctness(self):
        fixture = self.operator / "fixture.txt"
        fixture.write_text("42")
        self.test.write_text(
            "from pathlib import Path\nfrom demo_impl import demo_wrapper\n"
            'assert demo_wrapper() == int(Path("fixture.txt").read_text())\n'
        )
        with (
            patch.object(
                worker, "dispatch", side_effect=self.evaluate_in_process
            ) as dispatch,
            patch.object(measure_latency, "build_run_once", return_value=lambda: None),
            patch.object(
                measure_latency, "measure", return_value=dict(p=3, raw_count=4)
            ),
        ):
            self.assertEqual(session.request(self.evaluator)["test_rc"], 0)
            self.assertTrue(session.request(self.evaluator)["cache_hit"])
            fixture.write_text("43")
            result = session.request(self.evaluator)
        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["test_rc"], 1)
        self.assertIn("AssertionError", result["test_output"])
        self.assertEqual(dispatch.call_count, 2)

    def test_binary_extensionless_and_symlinked_fixtures_invalidate(self):
        config = dict(op_dir=str(self.operator), op_file=str(self.source))
        for name in ("fixture.npy", "fixture.pt", "fixture", "fixture.md"):
            fixture = self.operator / name
            fixture.write_bytes(b"original")
            before = session.context_fingerprint(config)
            fixture.write_bytes(b"changed")
            self.assertNotEqual(before, session.context_fingerprint(config), name)
        external = self.root / "external-fixture.bin"
        external.write_bytes(b"original")
        (self.operator / "linked-fixture.bin").symlink_to(external)
        before = session.context_fingerprint(config)
        external.write_bytes(b"changed")
        self.assertNotEqual(before, session.context_fingerprint(config))

    def test_fixture_change_during_evaluation_is_not_certified(self):
        fixture = self.operator / "fixture.txt"
        fixture.write_text("42")

        def change_fixture(*args):
            fixture.write_text("43")
            return dict(test_rc=0, lint_rc=0, metrics=dict(p=3, raw_count=4))

        with patch.object(worker, "dispatch", side_effect=change_fixture):
            result = session.request(self.evaluator)
        self.assertEqual(result["test_rc"], 1)
        self.assertIn("context changed", result["test_output"])
        self.assertFalse(
            list((self.operator / "optimization/.evaluation/results").glob("*.json"))
        )

    def test_missing_lint_package_is_retried_by_real_worker(self):
        self.evaluator.lint_root = str(self.root / "missing-lint-package")
        with patch.object(worker, "dispatch", wraps=worker.dispatch) as dispatch:
            first = session.request(self.evaluator)
            second = session.request(self.evaluator)
        self.assertEqual(first["test_rc"], 0)
        self.assertEqual(first["lint_rc"], 1)
        self.assertIn("No module named pypto_op_lint", first["lint_output"])
        self.assertFalse(first["cache_hit"])
        self.assertFalse(second["cache_hit"])
        self.assertEqual(dispatch.call_count, 2)

    def test_lint_failures_never_become_permanent_source_verdicts(self):
        for code, text in (
            (1, "PermissionError"),
            (124, "timeout"),
            (1, "No module named pypto_op_lint"),
            (1, "SyntaxError: broken lint installation"),
            (2, "S1 candidate violation"),
        ):
            result = dict(test_rc=0, lint_rc=code, lint_output=text)
            self.assertFalse(session.static_failure(result), text)
            self.assertFalse(
                session.reusable_result(result, dict(warmup=2, runs=2)), text
            )

    def test_pythonpath_matches_subprocess_for_absolute_relative_and_empty_entries(
        self,
    ):
        dependencies = self.root / "dependencies"
        dependencies.mkdir()
        for index, value in enumerate(
            (str(dependencies), "../dependencies", "../dependencies:")
        ):
            module = "panko_review_dependency_" + str(index)
            (dependencies / (module + ".py")).write_text("VALUE = 42\n")
            self.addCleanup(sys.modules.pop, module, None)
            self.test.write_text(f"from {module} import VALUE\nassert VALUE == 42\n")
            command = (
                f'cd "{self.operator}" && PYTHONPATH={shlex.quote(value)} '
                f'"{sys.executable}" "{self.test}"'
            )
            original_rc, output = evaluator_module.default_run(command, 20)
            self.assertEqual(original_rc, 0, output)
            self.assertEqual(session.run_python_tests(command, "0"), 0)

    def test_pythonpath_override_and_failure_restore_process_state(self):
        inherited = self.root / "inherited"
        inherited.mkdir()
        dependencies = self.root / "dependencies"
        dependencies.mkdir()
        previous_path, previous_cwd, previous_argv = (
            sys.path[:],
            os.getcwd(),
            sys.argv[:],
        )
        with (
            patch.dict(os.environ, PYTHONPATH=str(inherited)),
            patch.object(sys, "path", [str(inherited), *previous_path]),
        ):
            environment, paths = os.environ.copy(), sys.path[:]
            self.test.write_text(
                "import sys\n"
                f"assert {str(inherited)!r} not in sys.path\n"
                f"assert sys.path.index({str(dependencies)!r}) < len(sys.path) - 1\n"
                "raise SystemExit(3)\n"
            )
            command = (
                f'cd "{self.operator}" && PYTHONPATH="{dependencies}" '
                f'"{sys.executable}" "{self.test}"'
            )
            original_rc, output = evaluator_module.default_run(command, 20)
            self.assertEqual(original_rc, 3, output)
            self.assertEqual(session.run_python_tests(command, "0"), 3)
            self.assertEqual(sys.path, paths)
            self.assertEqual(dict(os.environ), environment)
        self.assertEqual(sys.path, previous_path)
        self.assertEqual(os.getcwd(), previous_cwd)
        self.assertEqual(sys.argv, previous_argv)


if __name__ == "__main__":
    unittest.main()
