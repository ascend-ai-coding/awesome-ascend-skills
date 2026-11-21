# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Prove cache invalidation, content guards, harness reuse and real stop gating."""

import importlib
import builtins
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
session = importlib.import_module("evaluation_session")
harness_session = importlib.import_module("harness_session")
prepared_inputs = importlib.import_module("prepared_inputs")
RealEvaluator = importlib.import_module("bayesian_optimization.evaluator").RealEvaluator
worker = importlib.import_module("session_worker")
measure_latency = importlib.import_module("measure_latency")


class Tensor:
    def __init__(self, value=1):
        self.value = value
        self._version = 0

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.value)

    def mutate(self):
        self._version += 1


class AdditionalReuseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "demo_impl.py"
        self.source.write_text("def demo_wrapper(): return 42\n")
        self.evaluator = RealEvaluator(
            "demo",
            str(self.root),
            str(self.source),
            "0",
            f'{sys.executable} -c "pass"',
            evaluation_mode="cached",
        )
        self.addCleanup(session.close_session, self.root)

    def success(self):
        return dict(
            test_rc=0,
            test_output="original correctness passed",
            lint_rc=0,
            lint_output="original lint passed",
            metrics=dict(p=3.0, raw_count=4),
        )

    def test_identical_success_reuses_evidence_and_dependencies_invalidate(self):
        with patch.object(
            worker, "dispatch", side_effect=lambda *args: self.success()
        ) as dispatch:
            self.assertFalse(session.request(self.evaluator)["cache_hit"])
            result = session.request(self.evaluator)
            self.assertTrue(result["cache_hit"])
            self.assertEqual(result["test_output"], "original correctness passed")
            self.assertEqual(dispatch.call_count, 1)
            for filename in ("demo_golden.py", "test_demo.py", "SPEC.md"):
                (self.root / filename).write_text("changed dependency")
                self.assertFalse(session.request(self.evaluator)["cache_hit"])
            with patch.dict("os.environ", PANKO_REUSE_TEST_CONTEXT="changed"):
                self.assertFalse(session.request(self.evaluator)["cache_hit"])
            self.evaluator.device = "1"
            self.assertFalse(session.request(self.evaluator)["cache_hit"])
            self.evaluator.runs = 4
            self.assertFalse(session.request(self.evaluator)["cache_hit"])
            self.source.write_text("def demo_wrapper(): return 43\n")
            self.assertFalse(session.request(self.evaluator)["cache_hit"])

    def test_static_failure_reused_but_transient_faults_and_timeouts_are_not(self):
        static = dict(
            test_rc=1,
            test_output="TensorErr::TENSOR_MEMORY_ALLOCATION exceeds MEM_UB size",
        )
        with (
            patch.object(worker, "dispatch", return_value=static) as dispatch,
            patch.object(session, "close_session") as close,
        ):
            self.assertFalse(session.request(self.evaluator)["cache_hit"])
            self.assertTrue(session.request(self.evaluator)["cache_hit"])
            self.assertEqual(dispatch.call_count, 1)
            close.assert_called_once_with(self.root)
        for code, message in (
            (1, "MAP_REG_ADDR_FAILED SyntaxError:"),
            (124, "SyntaxError: timeout"),
        ):
            self.source.write_text(f"def demo_wrapper(): return {code}\n")
            with patch.object(
                worker, "dispatch", return_value=dict(test_rc=code, test_output=message)
            ) as dispatch:
                self.assertFalse(session.request(self.evaluator)["cache_hit"])
                self.assertFalse(session.request(self.evaluator)["cache_hit"])
                self.assertEqual(dispatch.call_count, 2)

    def test_new_search_invalidates_evidence_and_terminal_but_preserves_old_files(self):
        marker = self.root / "optimization" / "controller-terminal.json"
        with patch.object(
            worker, "dispatch", side_effect=lambda *args: self.success()
        ) as dispatch:
            session.request(self.evaluator)
            cache = self.root / "optimization" / ".evaluation" / "results"
            old_files = set(cache.glob("*.json"))
            marker.write_text('{"directive": "STOP"}')
            (self.root / "tuning_notes.md").write_text("new tuning notes")
            self.assertTrue(session.request(self.evaluator)["cache_hit"])
            session.reset_session(self.root)
            self.assertFalse(marker.exists())
            self.assertFalse(session.request(self.evaluator)["cache_hit"])
            self.assertEqual(dispatch.call_count, 2)
            self.assertTrue(all(path.exists() for path in old_files))
            self.assertEqual(len(list(cache.glob("*.json"))), 2)

    def test_invalid_syntax_never_starts_device_worker(self):
        self.source.write_text("def broken(\n")
        with patch.object(worker, "dispatch") as dispatch:
            first = session.request(self.evaluator)
            second = session.request(self.evaluator)
        self.assertEqual(first["test_rc"], 1)
        self.assertIn("SyntaxError:", first["test_output"])
        self.assertTrue(second["cache_hit"])
        dispatch.assert_not_called()

    def test_worker_runs_original_lint_in_every_resident_mode(self):
        self.evaluator.lint_root = str(self.root)
        with (
            patch.object(RealEvaluator, "_default_run", return_value=0) as command,
            patch.object(measure_latency, "build_run_once", return_value=lambda: None),
            patch.object(
                measure_latency, "measure", return_value=dict(p=1, raw_count=4)
            ),
        ):
            for mode in ("resident", "cached", "prepared"):
                self.evaluator.evaluation_mode = mode
                config = {
                    key: getattr(self.evaluator, key) for key in session.CONFIG_FIELDS
                }
                result = session.evaluate_candidate(config)
                self.assertEqual(result["lint_rc"], 0)
                self.assertEqual(result["metrics"]["raw_count"], 4)
            self.assertEqual(command.call_count, 3)
            self.assertTrue(
                all(
                    "-m pypto_op_lint" in call.args[0]
                    for call in command.call_args_list
                )
            )

    def test_source_race_and_incomplete_profile_are_not_cached(self):
        def change(*args):
            self.source.write_text("def demo_wrapper(): return 99\n")
            return self.success()

        with patch.object(worker, "dispatch", side_effect=change):
            self.assertEqual(session.request(self.evaluator)["test_rc"], 1)
        for count in (0, 3):
            value = self.success()
            value["metrics"]["raw_count"] = count
            with patch.object(worker, "dispatch", return_value=value) as dispatch:
                session.request(self.evaluator)
                session.request(self.evaluator)
                self.assertEqual(dispatch.call_count, 2)

    def test_tensor_versions_content_and_mutated_references_are_guarded(self):
        torch = types.SimpleNamespace(equal=lambda a, b: a.value == b.value)
        calls = []

        def reference(value):
            calls.append(value.value)
            return Tensor(value.value * 2)

        memo = prepared_inputs.PreparedCall(reference, torch)
        value = Tensor()
        first = memo(value)
        memo.reset()
        self.assertIs(memo(value), first)
        value.value = 2  # Simulate .data mutation without a version increment.
        memo.reset()
        second = memo(value)
        self.assertIsNot(second, first)
        second.value = 100  # A changed cached reference must also be recomputed.
        memo.reset()
        self.assertEqual(memo(value).value, 4)
        value.mutate()
        memo.reset()
        self.assertEqual(memo(value).value, 4)
        self.assertEqual(len(calls), 4)

    def test_preparation_restores_original_functions_after_test_failure(self):
        (self.root / "demo_golden.py").write_text("original reference")
        reference = types.ModuleType("demo_golden")

        def golden(value):
            return Tensor(value.value * 2)

        reference.demo_golden = golden
        torch = types.SimpleNamespace(
            randn=lambda *args, **kwargs: Tensor(),
            equal=lambda a, b: a.value == b.value,
        )
        original_random, original_reference = torch.randn, reference.demo_golden
        prepared = prepared_inputs.PreparedInputs()
        config = dict(op="demo", op_dir=str(self.root))
        with patch.dict(sys.modules, torch=torch, demo_golden=reference):
            with self.assertRaises(RuntimeError), prepared.correctness(config):
                reference.demo_golden(torch.randn(2))
                raise RuntimeError("test failed")
            self.assertIs(torch.randn, original_random)
            self.assertIs(reference.demo_golden, original_reference)

    def test_real_harness_worker_reused_dependency_restart_and_cleanup(self):
        root = self.root / "optimization" / ".harness"
        self.addCleanup(worker.close_worker, root)
        config = dict(
            worker_module="harness_session",
            op_dir=str(self.root),
            cwd=str(Path.cwd()),
            device="0",
        )
        fingerprints = []
        outputs = []
        for changed in (False, False, True):
            if changed:
                config["device"] = "1"
            fingerprint = harness_session.context_fingerprint(config)
            result = worker.dispatch(root, config, fingerprint, ["--help"], 30)
            self.assertEqual(result["returncode"], 0, result["stderr"])
            outputs.append(result["stdout"])
            fingerprints.append(
                Path(json.loads((root / "active.json").read_text())["session"])
            )
        self.assertEqual(outputs[0], outputs[1])
        self.assertIn("panko_harness", outputs[0])
        self.assertEqual(fingerprints[0], fingerprints[1])
        self.assertNotEqual(fingerprints[1], fingerprints[2])
        self.assertFalse(worker.worker_alive(fingerprints[0]))
        worker.close_worker(root)
        self.assertFalse(worker.worker_alive(fingerprints[2]))

    def test_harness_and_evaluator_have_independent_workers_and_shared_cleanup(self):
        self.evaluator.test_command = f'{sys.executable} -c "raise SystemExit(2)"'
        self.assertEqual(session.request(self.evaluator)["test_rc"], 2)
        evaluation_root = self.root / "optimization" / ".evaluation"
        evaluation_worker = Path(
            json.loads((evaluation_root / "active.json").read_text())["session"]
        )
        harness_root = self.root / "optimization" / ".harness"
        self.addCleanup(worker.close_worker, harness_root)
        config = dict(
            worker_module="harness_session",
            op_dir=str(self.root),
            cwd=str(Path.cwd()),
            device="0",
        )
        fingerprint = harness_session.context_fingerprint(config)
        result = worker.dispatch(harness_root, config, fingerprint, ["--help"], 30)
        self.assertEqual(result["returncode"], 0, result["stderr"])
        harness_worker = Path(
            json.loads((harness_root / "active.json").read_text())["session"]
        )
        self.assertNotEqual(evaluation_worker, harness_worker)
        worker.close_worker(harness_root)
        self.assertFalse(worker.worker_alive(harness_worker))
        self.assertTrue(worker.worker_alive(evaluation_worker))
        self.assertEqual(session.request(self.evaluator)["test_rc"], 2)
        self.assertEqual(
            evaluation_worker,
            Path(json.loads((evaluation_root / "active.json").read_text())["session"]),
        )
        worker.dispatch(harness_root, config, fingerprint, ["--help"], 30)
        harness_worker = Path(
            json.loads((harness_root / "active.json").read_text())["session"]
        )
        with patch.object(
            sys, "argv", ["harness_session.py", "--op-dir", str(self.root), "--close"]
        ):
            self.assertEqual(harness_session.main(), 0)
        self.assertFalse(worker.worker_alive(evaluation_worker))
        self.assertFalse(worker.worker_alive(harness_worker))

    def test_harness_exit_codes_do_not_terminate_the_resident_worker(self):
        for value, expected in ((None, 0), (0, 0), (3, 3), ("exit detail", 1)):
            main = Mock(side_effect=builtins.SystemExit(value))
            with patch.object(
                harness_session.traceback, "print_exception"
            ) as diagnostic:
                self.assertEqual(harness_session.run_main(main), expected)
            self.assertEqual(diagnostic.call_count, int(isinstance(value, str)))
        self.assertEqual(harness_session.run_main(lambda: None), 0)

    def test_terminal_marker_requires_real_stop_restored_best_and_report(self):
        reports = []

        def report(args):
            reports.append(True)

        harness = types.SimpleNamespace(
            cmd_report=report,
            load=lambda path: dict(
                progress=dict(stop_reason="eval_budget", _start_ts=123)
            ),
        )
        (self.root / "optimization").mkdir()
        for stop, delivered, forced in (
            (False, True, False),
            (True, False, False),
            (True, True, True),
            (True, True, False),
        ):

            def stop_command(args, stop=stop, delivered=delivered, forced=forced):
                json.dump(
                    dict(
                        stop=stop,
                        delivered_is_best=delivered,
                        forced=forced,
                        delivered_code_hash="best-source",
                    ),
                    sys.stdout,
                )

            harness.cmd_stop = stop_command
            with patch.object(
                harness_session.importlib, "import_module", return_value=harness
            ):
                terminal = harness_session.complete_stop(dict(op_dir=str(self.root)))
            self.assertEqual(bool(terminal), stop and delivered and not forced)
        self.assertEqual(len(reports), 1)
        marker = self.root / "optimization" / "controller-terminal.json"
        self.assertEqual(
            json.loads(marker.read_text()),
            dict(
                directive="STOP",
                reason="eval_budget",
                run_start=123,
                delivered_code_hash="best-source",
            ),
        )
