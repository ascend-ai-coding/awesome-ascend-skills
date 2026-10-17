# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_perf_application.py"
SPEC = importlib.util.spec_from_file_location("collect_perf_application", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_op_summary(path: Path, launches: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Op Name", "Task Start Time(us)", "Task Duration(us)", "Block Num"])
        for index, (name, duration) in enumerate(launches):
            writer.writerow([name, index * 10, duration, 56])


def execution_manifest(case_ids: list[str], warmup: dict | None = None) -> dict:
    selected = warmup or MODULE.select_warmup("Exp", {"status": "success"}, {"status": "not_needed"})
    warmup_sequence = [selected["logical_op"]]
    return {
        "schema_version": MODULE.EXECUTION_MANIFEST_SCHEMA_VERSION,
        "samples_per_case": 10,
        "target_operator": "Exp",
        "warmup": selected,
        "case_runs": [
            {
                "case_id": case_id,
                "warmup_kernel_sequence": warmup_sequence,
                "target_kernel_sequence": ["target_a", "target_b"],
                "iteration_kernel_sequence": warmup_sequence + ["target_a", "target_b"],
            }
            for case_id in case_ids
        ],
    }


class WarmupPolicyTest(unittest.TestCase):
    def test_ops_warmup_is_argmax_except_for_argmax_family(self) -> None:
        self.assertEqual(MODULE.ops_warmup_op("Exp"), "ArgMax")
        self.assertEqual(MODULE.ops_warmup_op("ArgMax"), "Max")
        self.assertEqual(MODULE.ops_warmup_op("torch_npu::arg_max"), "Max")
        self.assertEqual(MODULE.ops_warmup_op("ArgMaxV2"), "Max")
        self.assertEqual(MODULE.ops_warmup_op("ReduceMax"), "ArgMax")

    def test_successful_ops_probe_selects_exact_256_mib_argmax(self) -> None:
        selected = MODULE.select_warmup("Exp", {"status": "success"}, {"status": "not_needed"})
        self.assertEqual(selected["mode"], "framework_ops")
        self.assertEqual(selected["logical_op"], "ArgMax")
        self.assertEqual(selected["implementation"], "torch_argmax_256mib_fp16_v1")
        self.assertEqual(selected["tensor_bytes"], 256 * 1024 * 1024)
        self.assertEqual(selected["tensor_allocation"]["shape"], [128, 1024, 1024])
        self.assertEqual(selected["tensor_allocation"]["dtype"], "float16")

    def test_probe_declared_warmup_implementation_is_adopted(self) -> None:
        probe = {
            "status": "success",
            "warmup_implementation": "aclnn_argmax_256mib_fp16_v1",
        }
        selected = MODULE.select_warmup("Exp", probe, {"status": "not_needed"})
        self.assertEqual(selected["mode"], "framework_ops")
        self.assertEqual(selected["logical_op"], "ArgMax")
        self.assertEqual(selected["implementation"], "aclnn_argmax_256mib_fp16_v1")

    def test_parse_probe_declaration_reads_last_result_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "probe.log"
            log.write_text(
                "[runner] probe ok: ArgMax on (128,1024,1024) fp16\n"
                'OPS_WARMUP_PROBE_RESULT: {"logical_op": "ArgMax", '
                '"implementation": "aclnn_argmax_256mib_fp16_v1"}\n',
                encoding="utf-8",
            )
            declaration = MODULE.parse_probe_declaration(log)
        self.assertEqual(declaration["implementation"], "aclnn_argmax_256mib_fp16_v1")
        self.assertEqual(MODULE.parse_probe_declaration(Path("/nonexistent/log")), None)

    def test_failed_ops_probe_selects_successfully_built_package(self) -> None:
        selected = MODULE.select_warmup("ArgMax", {"status": "failed"}, {"status": "success"})
        self.assertEqual(
            selected,
            {
                "mode": "compiled",
                "logical_op": "OpsProfilingWarmup",
                "implementation": "ops_profiling_warmup_v1",
                "fallback_reason": "ops_warmup_probe_failed",
                "tensor_bytes": 256 * 1024 * 1024,
                "tensor_allocation": {
                    "shape": [128, 1024, 1024],
                    "dtype": "float16",
                    "elements": 128 * 1024 * 1024,
                },
            },
        )

    def test_failed_ops_probe_and_build_fail_collection(self) -> None:
        with self.assertRaisesRegex(ValueError, "ops warmup unavailable"):
            MODULE.select_warmup("Exp", {"status": "failed"}, {"status": "failed"})

    def test_successful_ops_probe_skips_package_build(self) -> None:
        args = MODULE.argparse.Namespace(
            ops_warmup_probe_cmd="python probe.py --op {warmup_op}",
            target_op="Exp",
            ops_warmup_probe_timeout=10,
            warmup_build_cmd="bash build.sh",
            warmup_build_timeout=20,
        )
        with (
            mock.patch.object(MODULE, "run_ops_warmup_probe", return_value={"status": "success"}),
            mock.patch.object(MODULE, "run_warmup_build") as build,
        ):
            probe, build_result, selected = MODULE.prepare_warmup(
                args, Path("probe.log"), Path("build.log"), Path("missing.so")
            )
        build.assert_not_called()
        self.assertEqual(probe["status"], "success")
        self.assertEqual(build_result["status"], "not_needed")
        self.assertEqual(selected["mode"], "framework_ops")

    def test_failed_ops_probe_triggers_package_build(self) -> None:
        args = MODULE.argparse.Namespace(
            ops_warmup_probe_cmd="python probe.py --op {warmup_op}",
            target_op="Exp",
            ops_warmup_probe_timeout=10,
            warmup_build_cmd="bash build.sh",
            warmup_build_timeout=20,
        )
        with (
            mock.patch.object(MODULE, "run_ops_warmup_probe", return_value={"status": "failed"}),
            mock.patch.object(MODULE, "run_warmup_build", return_value={"status": "success"}) as build,
            mock.patch.object(MODULE, "attach_compiled_warmup_binary", side_effect=lambda warmup, _path: warmup),
        ):
            _probe, build_result, selected = MODULE.prepare_warmup(
                args, Path("probe.log"), Path("build.log"), Path("warmup.so")
            )
        build.assert_called_once_with("bash build.sh", Path("build.log"), 20)
        self.assertEqual(build_result["status"], "success")
        self.assertEqual(selected["mode"], "compiled")

    def test_build_nonzero_is_audited_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "build.log"
            with mock.patch.object(MODULE.subprocess, "run", return_value=mock.Mock(returncode=7)):
                result = MODULE.run_warmup_build("bash build.sh", log, 10)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["returncode"], 7)

    def test_ops_probe_receives_selected_max_for_argmax_target(self) -> None:
        with mock.patch.object(MODULE, "run_warmup_build", return_value={"status": "success"}) as run:
            result = MODULE.run_ops_warmup_probe(
                "python probe.py --op {warmup_op}",
                "ArgMax",
                Path("probe.log"),
                9,
            )
        run.assert_called_once_with(
            "python probe.py --op Max",
            Path("probe.log"),
            9,
            action="ops warmup probe",
        )
        self.assertEqual(result["warmup_op"], "Max")

    def test_compiled_binary_is_independently_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "_C.so"
            binary.write_bytes(b"compiled-warmup")
            selected = MODULE.attach_compiled_warmup_binary(
                MODULE.select_warmup("Exp", {"status": "failed"}, {"status": "success"}), binary
            )
        self.assertEqual(
            selected["binary_sha256"],
            MODULE.hashlib.sha256(b"compiled-warmup").hexdigest(),
        )

    def test_missing_compiled_binary_is_not_silently_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.so"
            with self.assertRaisesRegex(ValueError, "did not produce binary"):
                MODULE.attach_compiled_warmup_binary(
                    MODULE.select_warmup("Exp", {"status": "failed"}, {"status": "success"}), missing
                )

    def test_identity_manifest_requires_real_comparison_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "identity.json"
            path.write_text(
                json.dumps(
                    {
                        "source_sha256": "a" * 64,
                        "binary_sha256": "b" * 64,
                        "device_id": 0,
                        "cann_version": "9.0.0",
                        "frequency": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "frequency"):
                MODULE.load_identity_manifest(path)

            path.write_text(
                json.dumps(
                    {
                        "source_sha256": "not-a-sha",
                        "binary_sha256": "b" * 64,
                        "device_id": 0,
                        "cann_version": "9.0.0",
                        "frequency": {"policy": "observed"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                MODULE.load_identity_manifest(path)


class ApplicationParserTest(unittest.TestCase):
    def test_sums_all_target_kernels_and_takes_ten_sample_median(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = execution_manifest(["1", "2"])
            launches: list[tuple[str, float]] = [("unrelated_init", 999.0)]
            expected_medians = []
            for case_offset in (0.0, 100.0):
                totals = []
                for sample in range(10):
                    launches.extend(
                        [
                            ("ArgMax", 500.0),
                            ("target_a", case_offset + sample + 1.0),
                            ("target_b", 2.0),
                        ]
                    )
                    totals.append(case_offset + sample + 3.0)
                expected_medians.append((totals[4] + totals[5]) / 2)
            op_summary = root / "op_summary.csv"
            write_op_summary(op_summary, launches)

            rows, validation = MODULE.parse_profile(op_summary, manifest)

            self.assertEqual([row["total_us"] for row in rows], expected_medians)
            self.assertEqual([row["sample_count"] for row in rows], [10, 10])
            self.assertTrue(all(row["aggregation"] == "median" for row in rows))
            self.assertEqual(validation["expected_relevant_launches"], 60)
            self.assertEqual(validation["mapped_relevant_launches"], 60)
            self.assertFalse(validation["warmup_time_included"])
            self.assertEqual(
                [item["role"] for item in validation["mapped_launches"][:3]],
                ["warmup", "target", "target"],
            )

    def test_rejects_reordered_relevant_launches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = execution_manifest(["1"])
            launches = []
            for sample in range(10):
                pair = [("ArgMax", 1.0), ("target_a", 2.0), ("target_b", 3.0)]
                if sample == 4:
                    pair[0], pair[1] = pair[1], pair[0]
                launches.extend(pair)
            op_summary = root / "op_summary.csv"
            write_op_summary(op_summary, launches)
            with self.assertRaisesRegex(ValueError, "topology mismatch"):
                MODULE.parse_profile(op_summary, manifest)

    def test_ops_manifest_requires_exact_bytes_and_op(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            ops_warmup = {
                "mode": "framework_ops",
                "logical_op": "ArgMax",
                "implementation": "torch_argmax_256mib_fp16_v1",
                "tensor_bytes": 255 * 1024 * 1024,
                "tensor_allocation": {
                    "shape": [128, 1024, 1024],
                    "dtype": "float16",
                    "elements": 128 * 1024 * 1024,
                },
            }
            path.write_text(json.dumps(execution_manifest(["1"], ops_warmup)), encoding="utf-8")
            selected = MODULE.select_warmup("Exp", {"status": "success"}, {"status": "not_needed"})
            with self.assertRaisesRegex(ValueError, "exactly"):
                MODULE.load_execution_manifest(path, ["1"], "Exp", selected)

    def test_argmax_target_requires_max_ops_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            manifest = execution_manifest(
                ["1"],
                {
                    "mode": "framework_ops",
                    "logical_op": "ArgMax",
                    "implementation": "torch_argmax_256mib_fp16_v1",
                    "tensor_bytes": MODULE.FALLBACK_TENSOR_BYTES,
                    "tensor_allocation": {
                        "shape": [128, 1024, 1024],
                        "dtype": "float16",
                        "elements": 128 * 1024 * 1024,
                    },
                },
            )
            manifest["target_operator"] = "ArgMax"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            selected = MODULE.select_warmup("ArgMax", {"status": "success"}, {"status": "not_needed"})
            with self.assertRaisesRegex(ValueError, "logical_op mismatch"):
                MODULE.load_execution_manifest(path, ["1"], "ArgMax", selected)

    def test_compiled_manifest_requires_bundled_kernel_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            selected = MODULE.select_warmup("Exp", {"status": "failed"}, {"status": "success"})
            manifest = execution_manifest(["1"], selected)
            manifest["case_runs"][0]["warmup_kernel_sequence"] = ["different_warmup"]
            manifest["case_runs"][0]["iteration_kernel_sequence"] = [
                "different_warmup",
                "target_a",
                "target_b",
            ]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "compiled warmup sequence"):
                MODULE.load_execution_manifest(path, ["1"], "Exp", selected)


class ApplicationCommandTest(unittest.TestCase):
    def test_runner_command_receives_fixed_protocol(self) -> None:
        rendered = MODULE.render_application_command(
            "python run.py --ids {case_ids} --samples {samples} --manifest {execution_manifest} "
            "--warmup-mode {warmup_mode} --warmup-op {warmup_op}",
            ["1", "2"],
            Path("/tmp/manifest.json"),
            {"mode": "framework_ops", "logical_op": "ArgMax"},
        )
        self.assertIn("--ids 1,2", rendered)
        self.assertIn("--samples 10", rendered)
        self.assertIn("--warmup-mode framework_ops", rendered)
        self.assertIn("--warmup-op ArgMax", rendered)

    def test_main_invokes_one_application_session_and_writes_protocol_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = root / "cases.csv"
            cases.write_text("case_id\n1\n", encoding="utf-8")
            runner_file = root / "run.py"
            runner_file.write_text("# synthetic runner\n", encoding="utf-8")
            identity_manifest = root / "perf-identity.json"
            identity_manifest.write_text(
                json.dumps(
                    {
                        "source_sha256": "a" * 64,
                        "binary_sha256": "b" * 64,
                        "device_id": 0,
                        "cann_version": "9.0.0",
                        "frequency": {"policy": "observed", "before_mhz": 1650},
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            args = MODULE.argparse.Namespace(
                cases=str(cases),
                runner_file=str(runner_file),
                identity_manifest=str(identity_manifest),
                application_cmd=(
                    "python run.py --ids {case_ids} --samples {samples} "
                    "--manifest {execution_manifest} --warmup-mode {warmup_mode} "
                    "--warmup-op {warmup_op}"
                ),
                target_op="Exp",
                ops_warmup_probe_cmd="python probe.py --op {warmup_op}",
                warmup_build_cmd="bash build.sh",
                compiled_warmup_binary=str(root / "warmup" / "_C.so"),
                out_dir=str(out),
                msprof_bin="msprof",
                aic_metrics="PipeUtilization",
                timeout=30,
                warmup_build_timeout=10,
                ops_warmup_probe_timeout=5,
                max_launches_per_session=720,
                parse_only=False,
            )

            def fake_msprof(command, **_kwargs):
                manifest = execution_manifest(["1"])
                (out / "execution-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                launches = []
                for sample in range(10):
                    launches.extend(
                        [
                            ("ArgMax", 100.0),
                            ("target_a", float(sample + 1)),
                            ("target_b", 2.0),
                        ]
                    )
                write_op_summary(out / "PROF_1" / "op_summary_1.csv", launches)
                return mock.Mock(returncode=0)

            with (
                mock.patch.object(MODULE, "inspect_cann_environment", return_value={}),
                mock.patch.object(MODULE, "parse_args", return_value=args),
                mock.patch.object(
                    MODULE,
                    "run_ops_warmup_probe",
                    return_value={"status": "success", "returncode": 0, "warmup_op": "ArgMax"},
                ),
                mock.patch.object(MODULE, "run_warmup_build") as build,
                mock.patch.object(MODULE.subprocess, "run", side_effect=fake_msprof) as run,
            ):
                result = MODULE.main()

            self.assertEqual(result, 0)
            build.assert_not_called()
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertEqual(command[0], "msprof")
            self.assertTrue(any(value.startswith("--application=") for value in command))
            self.assertIn("--aic-mode=task-based", command)
            self.assertIn("--aic-metrics=PipeUtilization", command)
            collection = json.loads((out / "collection_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(collection["mode"], "msprof_application_single_session")
            self.assertEqual(collection["schema_version"], 5)
            self.assertEqual(collection["samples_per_case"], 10)
            self.assertEqual(collection["aggregation"], "median_target_total_us")
            self.assertEqual(collection["warmup"]["mode"], "framework_ops")
            self.assertEqual(collection["warmup"]["logical_op"], "ArgMax")
            self.assertEqual(collection["warmup_build"]["status"], "not_needed")
            self.assertEqual(collection["warmup_ops_probe"]["status"], "success")
            self.assertEqual(collection["identity"]["source_sha256"], "a" * 64)
            with (out / "performance.csv").open(newline="", encoding="utf-8") as stream:
                performance = list(csv.DictReader(stream))
            self.assertEqual(float(performance[0]["total_us"]), 7.5)


class SessionPlanningTest(unittest.TestCase):
    def test_chunks_respect_launch_cap(self) -> None:
        warmup = {"mode": "framework_ops", "logical_op": "ArgMax"}
        # ArgMax warmup = 2 kernels + >=1 target kernel -> 30 launches per case.
        case_ids = [f"case_{index:03d}" for index in range(60)]
        template = "run --ids {case_ids} --samples {samples} --manifest {execution_manifest}"
        paths = [Path(f"/out/session-{index:03d}/execution-manifest.json") for index in range(1, 61)]
        chunks = MODULE.plan_session_chunks(case_ids, warmup, template, paths, 720)
        for chunk in chunks:
            launches = len(chunk) * MODULE.SAMPLES_PER_CASE * 3
            self.assertLessEqual(launches, 720)
        self.assertEqual(sorted(c for chunk in chunks for c in chunk), sorted(case_ids))

    def test_chunks_split_on_application_arg_length(self) -> None:
        warmup = {"mode": "framework_ops", "logical_op": "ArgMax"}
        case_ids = [f"very_long_case_identifier_{index:04d}_suffix" for index in range(40)]
        template = (
            "python runner --ids {case_ids} --samples {samples} "
            "--manifest {execution_manifest} --warmup-mode {warmup_mode} --warmup-op {warmup_op}"
        )
        paths = [Path("/out/execution-manifest.json")]
        chunks = MODULE.plan_session_chunks(case_ids, warmup, template, paths, 100000)
        budget = MODULE.MSPROF_APPLICATION_ARG_LIMIT - MODULE.MSPROF_APPLICATION_ARG_HEADROOM
        for chunk in chunks:
            rendered = MODULE.render_application_command(template, chunk, paths[0], warmup)
            self.assertLess(len(rendered), budget)
        self.assertEqual(sorted(c for chunk in chunks for c in chunk), sorted(case_ids))

    def test_warmup_kernel_estimate_per_mode(self) -> None:
        self.assertEqual(MODULE.warmup_kernel_estimate({"mode": "framework_ops", "logical_op": "ArgMax"}), 2)
        self.assertEqual(MODULE.warmup_kernel_estimate({"mode": "framework_ops", "logical_op": "Max"}), 1)
        self.assertEqual(MODULE.warmup_kernel_estimate({"mode": "compiled", "logical_op": "OpsProfilingWarmup"}), 1)


class CannEnvironmentTest(unittest.TestCase):
    def test_incomplete_install_reports_missing_warmup_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "cann-9.1.0"
            lib64 = home / "x86_64-linux" / "lib64"
            lib64.mkdir(parents=True)
            (home / "version.info").write_text("Version=9.1.0\ntimestamp=20260610\n", encoding="utf-8")
            # only the trimmed library exists, without warmup symbols
            (lib64 / "libopapi_oam.so").write_bytes(b"\x7fELF-fake")
            with mock.patch.dict(MODULE.os.environ, {"ASCEND_HOME_PATH": str(home)}):
                environment = MODULE.inspect_cann_environment()
            self.assertEqual(environment["version"], "9.1.0")
            self.assertFalse(environment["framework_ops_warmup_available"])
            self.assertIn("incomplete", " ".join(environment["notes"]))
            self.assertEqual(environment["opapi_libraries"]["libopapi.so"], "")

    def test_complete_install_reports_available_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "cann-9.0.0"
            lib64 = home / "x86_64-linux" / "lib64"
            lib64.mkdir(parents=True)
            (home / "version.info").write_text("Version=9.0.0\ntimestamp=20260422\n", encoding="utf-8")
            (lib64 / "libopapi.so").write_bytes(b"\x7fELF-fake")
            with (
                mock.patch.dict(MODULE.os.environ, {"ASCEND_HOME_PATH": str(home)}),
                mock.patch.object(MODULE, "_nm_defines_symbol", return_value=True),
            ):
                environment = MODULE.inspect_cann_environment()
            self.assertTrue(environment["framework_ops_warmup_available"])
            self.assertTrue(environment["warmup_symbol_probe"]["aclnnArgMaxGetWorkspaceSize"])

    def test_unset_home_path_is_recorded(self) -> None:
        with mock.patch.dict(MODULE.os.environ, {}, clear=True):
            environment = MODULE.inspect_cann_environment()
        self.assertEqual(environment["ascend_home_path"], "")
        self.assertTrue(environment["notes"])


class MultiSessionMainTest(unittest.TestCase):
    def test_main_splits_sessions_and_merges_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = root / "cases.csv"
            cases.write_text("case_id\n1\n2\n", encoding="utf-8")
            runner_file = root / "run.py"
            runner_file.write_text("# synthetic runner\n", encoding="utf-8")
            identity_manifest = root / "perf-identity.json"
            identity_manifest.write_text(
                json.dumps(
                    {
                        "source_sha256": "a" * 64,
                        "binary_sha256": "b" * 64,
                        "device_id": 0,
                        "cann_version": "9.0.0",
                        "frequency": {"policy": "observed", "before_mhz": 1650},
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            args = MODULE.argparse.Namespace(
                cases=str(cases),
                runner_file=str(runner_file),
                identity_manifest=str(identity_manifest),
                application_cmd=(
                    "python run.py --ids {case_ids} --samples {samples} "
                    "--manifest {execution_manifest} --warmup-mode {warmup_mode} "
                    "--warmup-op {warmup_op}"
                ),
                target_op="Exp",
                ops_warmup_probe_cmd="python probe.py --op {warmup_op}",
                warmup_build_cmd="bash build.sh",
                compiled_warmup_binary=str(root / "warmup" / "_C.so"),
                out_dir=str(out),
                msprof_bin="msprof",
                aic_metrics="PipeUtilization",
                timeout=30,
                warmup_build_timeout=10,
                ops_warmup_probe_timeout=5,
                max_launches_per_session=30,  # one case per session
                parse_only=False,
            )

            def fake_msprof(command, **_kwargs):
                application = command[1]
                if "--ids 1 " in application:
                    chunk = ["1"]
                else:
                    chunk = ["2"]
                session_dir = Path(command[2].split("=", 1)[1])
                session_dir.mkdir(parents=True, exist_ok=True)
                manifest = execution_manifest(chunk)
                (session_dir / "execution-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                launches = []
                for sample in range(10):
                    launches.extend(
                        [
                            ("ArgMax", 100.0),
                            ("target_a", float(sample + 1)),
                            ("target_b", 2.0),
                        ]
                    )
                write_op_summary(session_dir / "PROF_1" / "op_summary_1.csv", launches)
                return mock.Mock(returncode=0)

            with (
                mock.patch.object(MODULE, "parse_args", return_value=args),
                mock.patch.object(
                    MODULE,
                    "run_ops_warmup_probe",
                    return_value={"status": "success", "returncode": 0, "warmup_op": "ArgMax"},
                ),
                mock.patch.object(MODULE, "run_warmup_build"),
                mock.patch.object(MODULE, "inspect_cann_environment", return_value={"ascend_home_path": "/cann"}),
                mock.patch.object(MODULE.subprocess, "run", side_effect=fake_msprof) as run,
            ):
                result = MODULE.main()

            self.assertEqual(result, 0)
            self.assertEqual(run.call_count, 2)
            collection = json.loads((out / "collection_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(collection["mode"], "msprof_application_session_series")
            self.assertEqual(collection["session_count"], 2)
            self.assertEqual(collection["max_launches_per_session"], 30)
            self.assertEqual(collection["sessions"][0]["case_ids"], ["1"])
            self.assertEqual(collection["sessions"][1]["case_ids"], ["2"])
            validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
            self.assertEqual(validation["session_count"], 2)
            self.assertEqual(validation["expected_relevant_launches"], 60)
            with (out / "performance.csv").open(newline="", encoding="utf-8") as stream:
                performance = list(csv.DictReader(stream))
            self.assertEqual([row["case_id"] for row in performance], ["1", "2"])

    def test_incomplete_install_with_failed_fallback_fails_with_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = root / "cases.csv"
            cases.write_text("case_id\n1\n", encoding="utf-8")
            runner_file = root / "run.py"
            runner_file.write_text("# synthetic runner\n", encoding="utf-8")
            identity_manifest = root / "perf-identity.json"
            identity_manifest.write_text(
                json.dumps(
                    {
                        "source_sha256": "a" * 64,
                        "binary_sha256": "b" * 64,
                        "device_id": 0,
                        "cann_version": "9.1.0",
                        "frequency": {"policy": "observed", "before_mhz": 1650},
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            args = MODULE.argparse.Namespace(
                cases=str(cases),
                runner_file=str(runner_file),
                identity_manifest=str(identity_manifest),
                application_cmd=(
                    "python run.py --ids {case_ids} --samples {samples} "
                    "--manifest {execution_manifest} --warmup-mode {warmup_mode} "
                    "--warmup-op {warmup_op}"
                ),
                target_op="Exp",
                ops_warmup_probe_cmd="python probe.py --op {warmup_op}",
                warmup_build_cmd="bash build.sh",
                compiled_warmup_binary=str(root / "warmup" / "_C.so"),
                out_dir=str(out),
                msprof_bin="msprof",
                aic_metrics="PipeUtilization",
                timeout=30,
                warmup_build_timeout=10,
                ops_warmup_probe_timeout=5,
                max_launches_per_session=720,
                parse_only=False,
            )
            incomplete_environment = {
                "ascend_home_path": "/usr/local/Ascend/cann-9.1.0",
                "framework_ops_warmup_available": False,
                "notes": ["incomplete ops install"],
            }
            with (
                mock.patch.object(MODULE, "parse_args", return_value=args),
                mock.patch.object(
                    MODULE,
                    "run_ops_warmup_probe",
                    return_value={"status": "failed", "returncode": 1, "error": "561103"},
                ),
                mock.patch.object(
                    MODULE,
                    "run_warmup_build",
                    return_value={"status": "failed", "returncode": 1, "error": "bad_alloc"},
                ),
                mock.patch.object(MODULE, "inspect_cann_environment", return_value=incomplete_environment),
            ):
                result = MODULE.main()

            self.assertEqual(result, 2)
            collection = json.loads((out / "collection_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(collection["status"], "failed")
            self.assertIn("incomplete ops install", collection["cann_environment"]["notes"])
            self.assertIn("aclnnArgMax/aclnnMax host symbols", collection["error"])

    def test_empty_op_summary_session_is_retried_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = root / "cases.csv"
            cases.write_text("case_id\n1\n", encoding="utf-8")
            runner_file = root / "run.py"
            runner_file.write_text("# synthetic runner\n", encoding="utf-8")
            identity_manifest = root / "perf-identity.json"
            identity_manifest.write_text(
                json.dumps(
                    {
                        "source_sha256": "a" * 64,
                        "binary_sha256": "b" * 64,
                        "device_id": 0,
                        "cann_version": "9.0.0",
                        "frequency": {"policy": "observed", "before_mhz": 1650},
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            args = MODULE.argparse.Namespace(
                cases=str(cases),
                runner_file=str(runner_file),
                identity_manifest=str(identity_manifest),
                application_cmd=(
                    "python run.py --ids {case_ids} --samples {samples} "
                    "--manifest {execution_manifest} --warmup-mode {warmup_mode} "
                    "--warmup-op {warmup_op}"
                ),
                target_op="Exp",
                ops_warmup_probe_cmd="python probe.py --op {warmup_op}",
                warmup_build_cmd="bash build.sh",
                compiled_warmup_binary=str(root / "warmup" / "_C.so"),
                out_dir=str(out),
                msprof_bin="msprof",
                aic_metrics="PipeUtilization",
                timeout=30,
                warmup_build_timeout=10,
                ops_warmup_probe_timeout=5,
                max_launches_per_session=720,
                parse_only=False,
            )

            def fake_msprof(command, **_kwargs):
                session_dir = Path(command[2].split("=", 1)[1])
                session_dir.mkdir(parents=True, exist_ok=True)
                manifest = execution_manifest(["1"])
                (session_dir / "execution-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                launches = []
                for sample in range(10):
                    launches.extend(
                        [
                            ("ArgMax", 100.0),
                            ("target_a", float(sample + 1)),
                            ("target_b", 2.0),
                        ]
                    )
                write_op_summary(session_dir / "PROF_1" / "op_summary_1.csv", launches)
                return mock.Mock(returncode=0)

            summary_calls = {"count": 0}

            def flaky_wait(session_dir, timeout_seconds=60):
                summary_calls["count"] += 1
                if summary_calls["count"] == 1:
                    raise ValueError("expected exactly one op_summary CSV under session dir, found 0")
                return next(session_dir.rglob("op_summary_1.csv"))

            with (
                mock.patch.object(MODULE, "parse_args", return_value=args),
                mock.patch.object(
                    MODULE,
                    "run_ops_warmup_probe",
                    return_value={"status": "success", "returncode": 0, "warmup_op": "ArgMax"},
                ),
                mock.patch.object(MODULE, "run_warmup_build"),
                mock.patch.object(MODULE, "inspect_cann_environment", return_value={"ascend_home_path": "/cann"}),
                mock.patch.object(MODULE.subprocess, "run", side_effect=fake_msprof) as run,
                mock.patch.object(MODULE, "wait_for_op_summary", side_effect=flaky_wait),
            ):
                result = MODULE.main()

            self.assertEqual(result, 0)
            self.assertEqual(run.call_count, 2)  # one retry after the empty summary
            collection = json.loads((out / "collection_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(collection["sessions"][0]["empty_summary_retries"], 1)
            with (out / "performance.csv").open(newline="", encoding="utf-8") as stream:
                performance = list(csv.DictReader(stream))
            self.assertEqual(float(performance[0]["total_us"]), 7.5)


if __name__ == "__main__":
    unittest.main()
