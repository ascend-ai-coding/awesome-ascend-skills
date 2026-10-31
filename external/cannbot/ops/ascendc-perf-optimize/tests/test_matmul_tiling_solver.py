# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import csv
import importlib.util
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SKILL = (
    Path(__file__).resolve().parents[1] / "references" / "tiling" / "matmul" / "solver"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "matmul_solver_test_runner", SKILL / "scripts" / "ttk_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_worktree():
    spec = importlib.util.spec_from_file_location(
        "matmul_worktree_test_runner", SKILL / "scripts" / "worktree_setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SolverTest(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()
        self.profile = {
            "identity": {"op_name": "demo"},
            "fold_rule": {},
            "input_rules": {},
            "quant_modes": {},
            "case_classes": {},
        }
        self.runner._ACTIVE_OP_NAME = "demo"
        self.runner._PROFILE_CACHE["demo"] = self.profile

    def write_csv(self, path, rows):
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_timeouts_use_milliseconds_for_npu_and_seconds_for_process(self):
        for shape, expected in (
            ((3, 7, 5), (30000, 600)),
            ((3, 7, 12001), (60000, 1200)),
            ((1024, 1024, 1024), (60000, 1200)),
            ((3, 7, 20001), (90000, 1800)),
            ((1024, 1024, 512, 2), (60000, 1200)),
        ):
            with self.subTest(shape=shape):
                timeouts = self.runner.calc_timeout(*shape)
                self.assertEqual(timeouts, expected)
                with mock.patch.object(self.runner, "_require_runtime_paths"):
                    command = self.runner.build_ttk_command("case.csv", *timeouts)
                self.assertEqual(command[0:2], ["timeout", str(expected[1])])
                self.assertEqual(
                    command[command.index("--npu-timeout") + 1], str(expected[0])
                )
                self.assertEqual(command[command.index("-i") + 1], "case.csv")
                self.assertIn("--compare=close", command)

    def test_new_profile_requires_twenty_generated_candidates_per_template(self):
        spec = importlib.util.spec_from_file_location(
            "matmul_profile_scaffold_test",
            SKILL / "scripts" / "create_profile_scaffold.py",
        )
        scaffold = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scaffold)
        with tempfile.TemporaryDirectory() as tmp:
            target = scaffold.create_scaffold("demo", "matmul/demo", "abcdef1", tmp)
            profile = json.loads((target / "profile.json").read_text())
        self.profile["constraints"] = profile["constraints"]
        with self.assertRaisesRegex(ValueError, "second: 19/20"):
            self.runner.validate_candidate_count(
                {"first": 20, "second": 19}, {"first": 12, "second": 8}
            )
        # The minimum applies to generated candidates; static constraints may reject some.
        self.runner.validate_candidate_count(
            {"first": 20, "second": 20}, {"first": 12, "second": 8}
        )

    def test_operator_adapter_controls_input_order_and_attributes(self):
        # B, bias, A: deliberately unlike QBMM; no dtype attribute or MX scales.
        row = dict.fromkeys(self.runner.REQUIRED_CSV_COLUMNS, "None")
        row.update(
            testcase_name="case",
            op_name="demo",
            is_enabled="1",
            input_shapes="((5, 7), (7,), (3, 5))",
            input_dtypes="('int8', 'float32', 'float16')",
            input_formats="('NZ', 'ND', 'ND')",
            attributes="{'adj_a': False, 'adj_b': True}",
            output_dtypes="'float16'",
        )
        seen = []

        def derive(ctx):
            return {
                "M": ctx["input_shapes"][2][0],
                "N": 7,
                "K": 5,
                "x1_dtype": ctx["input_dtypes"][2],
                "x2_dtype": ctx["input_dtypes"][0],
                "scale_dtype": None,
                "bias": ctx["input_dtypes"][1],
                "trans_a": ctx["attributes"]["adj_a"],
                "trans_b": ctx["attributes"]["adj_b"],
                "x1_format": ctx["input_formats"][2],
                "x2_format": ctx["input_formats"][0],
                "case_class": "plain",
                "batch_count": 1,
                "quant_mode": "plain",
                "custom_fact": 42,
            }

        adapter = types.SimpleNamespace(
            derive_facts=derive, validate_case=lambda ctx, rules: seen.append(ctx)
        )
        self.runner._RECOGNITION_CACHE["demo"] = adapter
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.csv"
            with mock.patch.object(
                self.runner, "load_profile", return_value=self.profile
            ):
                for enabled in ("1", "1.0", "True", "TRUE", "true", "tRuE", " True "):
                    row["is_enabled"] = enabled
                    self.write_csv(path, [row])
                    original = path.read_text()
                    self.assertTrue(self.runner.validate_case_csv(str(path)), enabled)
                    self.assertEqual(path.read_text(), original)
                for disabled in ("0", "false", "FALSE", "", "yes"):
                    row["is_enabled"] = disabled
                    self.write_csv(path, [row])
                    self.assertFalse(self.runner.validate_case_csv(str(path)), disabled)
            row["is_enabled"] = "1"
        params = self.runner.extract_shape_params(row)
        self.assertEqual((params["M"], params["N"], params["K"]), (3, 7, 5))
        self.assertEqual(params["x1_dtype"], "float16")
        self.assertEqual(params["bias"], "float32")
        self.assertTrue(params["trans_b"])
        self.assertEqual(params["custom_fact"], 42)
        self.assertEqual(seen[0]["row"], row)
        adapter.validate_case = lambda ctx, rules: "unsupported bias"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.csv"
            self.write_csv(path, [row])
            with mock.patch.object(
                self.runner, "load_profile", return_value=self.profile
            ):
                self.assertFalse(self.runner.validate_case_csv(str(path)))
        mode_missing = derive(self.runner._case_context(row))
        del mode_missing["quant_mode"]
        adapter.derive_facts = lambda ctx: mode_missing
        with self.assertRaisesRegex(ValueError, "quant_mode"):
            self.runner.extract_shape_params(row)

    def test_two_and_six_input_cases_use_native_interfaces_through_adapter(self):
        # Port the input scenarios from skills-dev 094fcab3 through the current adapter contract.
        cases = (
            (
                "gemm_syrk",
                ("a", "c"),
                ((1024, 256), (1024, 1024)),
                ("float16", "float16"),
                {"alpha": 1.0, "beta": 1.0, "transpose_x": True, "fill_mode": "full"},
                {
                    "M": 1024,
                    "N": 1024,
                    "K": 256,
                    "x1_dtype": "float16",
                    "x2_dtype": "float16",
                    "scale_dtype": None,
                    "bias": "none",
                    "trans_a": True,
                    "trans_b": False,
                },
            ),
            (
                "quant_matmul",
                ("x1", "x2", "scale", "offset", "bias", "pertoken_scale"),
                ((128, 64), (64, 128), (1,), (1,), (128,), (1,)),
                ("float16", "int8", "float32", "int32", "float16", "float32"),
                {"transpose_x1": False, "transpose_x2": True},
                {
                    "M": 128,
                    "N": 128,
                    "K": 64,
                    "x1_dtype": "float16",
                    "x2_dtype": "int8",
                    "scale_dtype": "float32",
                    "bias": "float16",
                    "trans_a": False,
                    "trans_b": True,
                },
            ),
        )
        for op, slots, shapes, dtypes, attributes, scenario_facts in cases:
            with self.subTest(op=op), tempfile.TemporaryDirectory() as tmp:
                self.profile["identity"]["op_name"] = op
                self.profile["input_rules"] = {"slot_order": list(slots)}
                self.runner._ACTIVE_OP_NAME = op
                self.runner._PROFILE_CACHE[op] = self.profile
                facts = {
                    **scenario_facts,
                    "x1_format": "ND",
                    "x2_format": "ND",
                    "case_class": "plain",
                    "batch_count": 1,
                    "quant_mode": "plain",
                }
                seen = []

                def validate(ctx, rules):
                    seen.append(ctx)
                    if any(
                        len(ctx[key]) != len(rules["slot_order"])
                        for key in ("input_shapes", "input_dtypes", "input_formats")
                    ):
                        return "input slot count mismatch"

                adapter = types.SimpleNamespace(
                    derive_facts=mock.Mock(return_value=facts), validate_case=validate
                )
                self.runner._RECOGNITION_CACHE[op] = adapter
                row = dict.fromkeys(self.runner.REQUIRED_CSV_COLUMNS, "None")
                row.update(
                    testcase_name="case",
                    op_name=op,
                    is_enabled="1",
                    input_shapes=repr(shapes),
                    input_dtypes=repr(dtypes),
                    input_formats=repr(tuple("ND" for _ in slots)),
                    attributes=repr(attributes),
                    output_dtypes="'float16'",
                )
                path = Path(tmp) / "case.csv"
                self.write_csv(path, [row])
                with mock.patch.object(
                    self.runner, "load_profile", return_value=self.profile
                ):
                    self.assertTrue(self.runner.validate_case_csv(str(path)))
                    params = self.runner.extract_shape_params(row)
                    for key, value in facts.items():
                        self.assertEqual(params[key], value)
                    context = adapter.derive_facts.call_args.args[0]
                    self.assertEqual(context["input_shapes"], shapes)
                    self.assertEqual(context["input_dtypes"], dtypes)
                    self.assertEqual(context["attributes"], attributes)
                    self.assertEqual(seen[0]["batch_count"], 1)
                    self.assertNotIn("dtype", attributes)
                    row["input_dtypes"] = repr(dtypes[:-1])
                    self.write_csv(path, [row])
                    self.assertFalse(self.runner.validate_case_csv(str(path)))

    def test_incomplete_facts_are_rejected_instead_of_inferred_from_slots(self):
        self.runner._RECOGNITION_CACHE["demo"] = types.SimpleNamespace(
            derive_facts=lambda ctx: {"M": 3}
        )
        row = {
            "op_name": "demo",
            "input_shapes": "()",
            "input_dtypes": "()",
            "input_formats": "()",
            "attributes": "{}",
            "output_dtypes": "'float16'",
        }
        with self.assertRaisesRegex(ValueError, "缺少执行接口事实"):
            self.runner.extract_shape_params(row)

    def test_same_length_layouts_require_unique_source_derived_context(self):
        self.profile["struct_layouts"] = {
            "plain": {
                "u32_slots": 2,
                "when": {"quant_mode": "plain"},
                "fields": {"baseM": {"offset": 0, "fmt": "I"}},
            },
            "packed": {
                "u32_slots": 2,
                "when": {"quant_mode": ["packed"]},
                "fields": {"baseM": {"offset": 4, "fmt": "I"}},
            },
        }
        plain, _ = self.runner._td_from_values(
            [64, 128], context={"quant_mode": "plain"}
        )
        packed, _ = self.runner._td_from_values(
            [64, 128], context={"quant_mode": "packed"}
        )
        self.assertEqual((plain.baseM, packed.baseM), (64, 128))
        with self.assertRaisesRegex(ValueError, "无法唯一确定"):
            self.runner._td_from_values([64, 128])
        self.profile["struct_layouts"]["packed"]["when"] = {}
        with self.assertRaisesRegex(ValueError, "无法唯一确定"):
            self.runner._td_from_values([64, 128], context={"quant_mode": "plain"})

    def test_coverage_detects_ambiguous_layout_instead_of_counting_slots_only(self):
        layouts = {name: {"u32_slots": 2, "fields": {}} for name in ("first", "second")}
        self.profile["struct_layouts"] = layouts
        self.profile["tiling_key"] = {"kernel_type_shift": 0, "kernel_type_bits": 8}
        self.profile["quant_modes"] = {"mode": {"kernel_type_map": {"plain": 0}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.csv"
            self.write_csv(
                path,
                [
                    {
                        "testcase_name": "case",
                        "dyn_tiling_data": "(1, 2)",
                        "dyn_tiling_key": "0x0",
                    }
                ],
            )
            result = self.runner.verify_layout_coverage(str(path), verbose=False)
        self.assertEqual(result["parseable"], {})
        self.assertEqual(result["undeclared"], {2: ["case"]})

    def test_template_lookup_is_scoped_to_mode_and_case_class(self):
        self.profile["tiling_key"] = {"kernel_type_shift": 4, "kernel_type_bits": 4}
        self.profile["quant_modes"] = {
            "a": {"kernel_type_map": {"first": 2}},
            "b": {
                "kernel_type_map": {"second": 2},
                "kernel_type_map_by_case_class": {"batched": {"third": 2}},
            },
        }
        detect = self.runner._detect_template_from_key_str
        self.assertEqual(detect("0x20", {"quant_mode": "a"}), "first")
        self.assertEqual(
            detect("0x20", {"quant_mode": "b", "case_class": "plain"}), "second"
        )
        self.assertEqual(
            detect("0x20", {"quant_mode": "b", "case_class": "batched"}), "third"
        )
        self.assertEqual(detect("0x20"), "unknown")
        self.assertEqual(detect("0x20", {"quant_mode": "unsupported"}), "unknown")

    def test_missing_key_is_not_guessed_from_42_slots_or_k_split(self):
        self.profile["struct_layouts"] = {
            "split": {
                "u32_slots": 42,
                "fields": {
                    "k": {"offset": 0, "fmt": "I"},
                    "singleCoreK": {"offset": 4, "fmt": "I"},
                },
            },
        }
        self.profile["quant_modes"] = {"mode": {"kernel_type_map": {"streamk": 11}}}
        self.profile["tiling_key"] = {"kernel_type_shift": 0, "kernel_type_bits": 8}
        _, template = self.runner._td_from_values([128, 64] + [0] * 40)
        self.assertEqual(template, "unknown")
        del self.profile["tiling_key"]
        self.assertEqual(self.runner._detect_template_from_key_str("0xb00"), "unknown")

    def test_actual_fields_survive_baseline_anchor_and_ranking(self):
        self.profile["tiling_key"] = {"kernel_type_shift": 0, "kernel_type_bits": 8}
        self.profile["quant_modes"] = {"mode": {"kernel_type_map": {"custom": 2}}}
        self.profile["struct_layouts"] = {
            "custom": {
                "u32_slots": 2,
                "fields": {
                    "chunkWidth": {"offset": 0, "fmt": "I"},
                    "tailLength": {"offset": 4, "fmt": "I"},
                },
            }
        }
        params = {
            "M": 3,
            "N": 7,
            "K": 5,
            "x1_dtype": "float16",
            "x2_dtype": "int8",
            "scale_dtype": None,
            "y_dtype": "float16",
            "bias": "none",
            "trans_a": False,
            "trans_b": False,
            "x2_format": "ND",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "result.csv"
            self.write_csv(
                csv_path,
                [
                    {
                        "testcase_name": "case",
                        "dyn_tiling_data": "(32, 0)",
                        "dyn_tiling_key": "0x2",
                    }
                ],
            )
            td, template = self.runner.extract_baseline_tiling(str(csv_path))
            self.assertEqual(td.to_dict(), {"chunkWidth": 32, "tailLength": 0})
            with self.assertRaises(AttributeError):
                _ = td.weightMustHitL2
            log_path = root / "anchor.log"
            log_path.write_text(
                "===== SCAN_START custom 0 =====\nTiling Key: 2 (0x2)\n"
                "Tiling Data Parsed Dict: (32, 0)\nBlockDim: 4\n"
                "===== SCAN_END custom 0 =====\n"
            )
            anchor = self.runner.extract_anchor(str(log_path), ["custom"])["custom"]
            self.assertEqual(anchor["chunkWidth"], 32)
            self.assertEqual(anchor["tailLength"], 0)
            self.assertEqual(anchor["blockDim"], 4)
            baseline = self.runner.BaselineResult(
                template=template, tiling_data=td, perf_us=10, precision_status="PASS"
            )
            output = root / "baseline.json"
            self.runner.output_baseline_info(
                baseline, params, "case", str(output), aic_num=4
            )
            stored = json.loads(output.read_text())["baseline"]
            self.assertEqual(stored["tiling_data"], td.to_dict())
            self.assertNotIn("health", stored)
            self.assertIn("chunkWidth:32", self.runner.generate_ranking(baseline, []))

    def test_ranking_accepts_dictionary_tiling_data_in_result_objects(self):
        raw = {"chunkWidth": 32, "tailLength": 0}
        baseline = self.runner.BaselineResult(
            template="custom", perf_us=10, precision_status="PASS", tiling_data=raw
        )
        candidate = self.runner.CandidateResult(
            template="custom", perf_us=8, tiling_data={"chunkWidth": 64}
        )
        self.assertIsInstance(baseline.tiling_data, self.runner.TilingData)
        self.assertEqual(baseline.tiling_data.to_dict(), raw)
        raw["chunkWidth"] = 99
        ranking = self.runner.generate_ranking(baseline, [candidate])
        self.assertIn("chunkWidth:32", ranking)
        self.assertIn("tailLength:0", ranking)
        self.assertIn("chunkWidth:64", ranking)
        self.assertIn("1.25x", ranking)
        existing = self.runner.TilingData(chunkWidth=16)
        self.assertIs(
            self.runner.BaselineResult(tiling_data=existing).tiling_data, existing
        )
        empty = self.runner.BaselineResult(
            template="custom",
            perf_us=10,
            precision_status="PASS",
            tiling_data={},
            tiling_parse_error="layout unavailable",
        )
        self.assertIn("layout unavailable", self.runner.generate_ranking(empty, []))

    def test_scan_results_keep_nonstandard_candidate_fields(self):
        self.profile["tiling_key"] = {"kernel_type_shift": 0, "kernel_type_bits": 8}
        self.profile["quant_modes"] = {"mode": {"kernel_type_map": {"custom": 2}}}
        self.runner.MACRO_MAP = {"chunkWidth": "TUNE_WIDTH"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.log"
            path.write_text(
                "===== SCAN_START custom 0 =====\nTiling Key: 2 (0x2)\n"
                "DYN_PERF: 9\nPRECISION_STATUS: PASS\n"
                "===== SCAN_END custom 0 =====\n"
            )
            with mock.patch.object(
                self.runner, "scan_paths", return_value={"output": str(path)}
            ):
                results = self.runner.collect_ttk_scan(
                    [{"template": "custom", "chunkWidth": 32}],
                    log_tag="case",
                    save_path="",
                    params={"quant_mode": "mode"},
                )
        self.assertEqual(results[0].status, "ok")
        self.assertEqual(results[0].tiling_data.to_dict(), {"chunkWidth": 32})

    def test_dead_fields_accept_lists_and_mappings_in_updates_and_scans(self):
        self.runner.MACRO_MAP = {"width": "TUNE_WIDTH", "unused": "TUNE_UNUSED"}
        self.runner.VENDOR_PACKAGES = {"custom": "test_custom"}
        candidate = {"template": "custom", "width": 32, "unused": 99}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kernel = root / "kernel.h"
            path_names = ("script", "output", "progress", "done")
            paths = {key: str(root / key) for key in path_names}
            for dead_fields in (
                ["unused"],
                {"unused": "kernel does not read this field"},
            ):
                with self.subTest(dead_fields=dead_fields):
                    self.profile["quant_modes"] = {"mode": {"dead_fields": dead_fields}}
                    kernel.write_text("#define TUNE_WIDTH 1\n#define TUNE_UNUSED 7\n")
                    with (
                        mock.patch.object(
                            self.runner,
                            "get_kernel_file_path",
                            return_value=str(kernel),
                        ),
                        mock.patch.object(self.runner, "_require_runtime_paths"),
                        mock.patch.object(
                            self.runner, "injectable_templates", return_value=["custom"]
                        ),
                        mock.patch.object(
                            self.runner, "scan_paths", return_value=paths
                        ),
                    ):
                        self.runner.update_kernel_macros(
                            "custom", candidate, quant_mode="mode"
                        )
                        script = self.runner.generate_scan_script(
                            [candidate], "case.csv", 3, 7, 5, quant_mode="mode"
                        )
                    self.assertEqual(
                        kernel.read_text(),
                        "#define TUNE_WIDTH 32\n#define TUNE_UNUSED 7\n",
                    )
                    text = Path(script).read_text()
                    self.assertIn("#define TUNE_WIDTH 32", text)
                    self.assertNotIn("TUNE_UNUSED", text)
                    self.assertIn("--npu-timeout 30000", text)
                    self.assertEqual(candidate["unused"], 99)

    def test_package_gate_accepts_formatted_code_and_still_rejects_missing_content(
        self,
    ):
        needle = "GET_TILING_DATA_WITH_STRUCT(Data, tilingData, tilingGM)"
        rule = {"path": "kernel.cpp", "contains": [needle]}
        self.profile["package_gate"] = {"required_files": [rule]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "normal"
            package.mkdir()
            kernel = package / "kernel.cpp"
            self.runner.VENDOR_CONFIG = str(root / "config.ini")
            self.runner.VENDOR_PACKAGES = {"normal": "normal"}
            with mock.patch.object(self.runner, "_require_runtime_paths"):
                for newline in ("\n", "\\\n", "\\\r\n"):
                    kernel.write_text(
                        "GET_TILING_DATA_WITH_STRUCT(Data,"
                        + newline
                        + "\t tilingData,"
                        + newline
                        + "    tilingGM);\n"
                    )
                    report = self.runner.validate_vendor_package_gate(templates=[])
                    self.assertTrue(report["ok"])
                rule["contains_match"] = "literal"
                with self.assertRaisesRegex(RuntimeError, "missing content"):
                    self.runner.validate_vendor_package_gate(templates=[])
                rule["contains_match"] = "whitespace"
                kernel.write_text(
                    "GET_TILING_DATA_WITH_STRUCT(WrongData, tilingData, tilingGM);\n"
                )
                saved = root / "package_gate.json"
                with self.assertRaisesRegex(RuntimeError, "missing content"):
                    self.runner.validate_vendor_package_gate(
                        templates=[], output_path=str(saved)
                    )
                self.assertFalse(json.loads(saved.read_text())["ok"])
                kernel.unlink()
                report = self.runner.validate_vendor_package_gate(
                    templates=[], strict=False
                )
                self.assertEqual(report["checks"][0]["error"], "missing")

    def test_kernel_install_paths_use_vendor_packages_independent_of_cwd(self):
        self.runner.MACRO_MAP = {"width": "TUNE_WIDTH"}
        self.runner.VENDOR_PACKAGES = {
            "custom": "test_forced",
            "normal": "custom_normal_nn",
        }
        worktree = load_worktree()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor_base = root / "vendors"
            self.runner.VENDOR_CONFIG = str(vendor_base / "config.ini")
            worktree.VENDOR_BASE = str(vendor_base)
            relative = "op_impl/ai_core/tbe/kernel.h"
            entry = {
                "install": relative,
                "insert_mode": "before",
                "sed_anchor": "anchor",
            }
            self.profile["quant_modes"] = {
                "forced": {"kernel_files": {"plain": {"custom": relative}}},
                "stub": {
                    "kernel_files": {"plain": {"custom": entry}},
                    "tune_macro_define_block": "defines",
                    "tuning_block_template": "block",
                },
            }
            for package in self.runner.VENDOR_PACKAGES.values():
                target = vendor_base / package / relative
                target.parent.mkdir(parents=True)
                target.write_text("#define TUNE_WIDTH 1\n")
            previous = os.getcwd()
            try:
                for cwd in (root, vendor_base):
                    os.chdir(cwd)
                    for mode, package in (
                        ("forced", "test_forced"),
                        ("stub", "custom_normal_nn"),
                    ):
                        expected = str(vendor_base / package / relative)
                        self.assertEqual(
                            self.runner.get_kernel_file_path("custom", "plain", mode),
                            expected,
                        )
                        self.runner.update_kernel_macros(
                            "custom", {"width": 32}, "plain", mode
                        )
                        self.assertEqual(
                            Path(expected).read_text().strip(), "#define TUNE_WIDTH 32"
                        )
                    self.assertEqual(
                        self.runner._kernel_entry_to_path(
                            str(root / "absolute.h"), "test_forced"
                        ),
                        str(root / "absolute.h"),
                    )
                    self.assertEqual(
                        worktree._stub_kernel_path(
                            {"install": str(root / "absolute.h")}
                        ),
                        str(root / "absolute.h"),
                    )
                    with (
                        mock.patch.object(
                            worktree, "_load_stub_profile", return_value=self.profile
                        ),
                        mock.patch.object(
                            worktree, "install_stub_file", return_value=True
                        ) as install,
                        mock.patch.object(
                            worktree, "restore_stub_file", return_value=True
                        ) as restore,
                        mock.patch.object(
                            worktree, "stub_file_status", return_value="clean"
                        ) as status,
                    ):
                        self.assertTrue(worktree.install_stub("stub", op="demo"))
                        self.assertTrue(worktree.restore_stub("stub", op="demo"))
                        worktree.stub_status("stub", op="demo")
                    expected = str(vendor_base / "custom_normal_nn" / relative)
                    self.assertEqual(install.call_args.args[0], expected)
                    self.assertEqual(restore.call_args.args[0], expected)
                    self.assertEqual(status.call_args.args[0], expected)
            finally:
                os.chdir(previous)

    def test_stub_macros_are_visible_outside_conditional_include_and_restore_exactly(
        self,
    ):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest(
                "C++ compiler required to verify preprocessor branch visibility"
            )
        worktree = load_worktree()
        definitions = "// TUNE PARAMS\n#define TUNE_WIDTH 32\n"
        block = "// TILING TUNING BLOCK\nconstexpr int width = TUNE_WIDTH;\n"
        preambles = (
            '#pragma once\n#include "common.h"\n',
            '#ifndef TEST_KERNEL_H\n#define TEST_KERNEL_H\n#include "common.h"\n',
            "",
        )
        for preamble in preambles:
            with self.subTest(preamble=preamble), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                header = root / "kernel.h"
                (root / "common.h").write_text("// common declarations\n")
                (root / "blaze.h").write_text(
                    "// conditionally included declarations\n"
                )
                source = (
                    preamble
                    + "#if defined(WEIGHT_ND) && (defined(WQBMMV2_S8) || defined(WEIGHT_F8_INPUT))\n"
                    '#include "blaze.h"\n#define BLAZE_AVAILABLE\n#endif\n'
                    "// 原始代码锚点\n"
                )
                if "#ifndef TEST_KERNEL_H" in preamble:
                    source += "#endif\n"
                original = source.encode()
                header.write_bytes(original)
                args = (
                    str(header),
                    "test",
                    "// 原始代码锚点",
                    definitions,
                    block,
                    "after",
                )
                self.assertTrue(worktree.install_stub_file(*args, dry_run=True))
                self.assertEqual(header.read_bytes(), original)
                self.assertFalse(Path(str(header) + ".orig").exists())
                self.assertTrue(worktree.install_stub_file(*args))
                patched = header.read_bytes()
                self.assertTrue(worktree.install_stub_file(*args))
                self.assertEqual(header.read_bytes(), patched)
                translation_unit = root / "check.cpp"
                translation_unit.write_text(
                    '#include "kernel.h"\n'
                    'static_assert(width == 32, "macro must be visible");\n'
                )
                for flags in (
                    [],
                    ["-DWEIGHT_ND", "-DWQBMMV2_S8"],
                    ["-DWEIGHT_ND", "-DWEIGHT_F8_INPUT"],
                    ["-DWEIGHT_ND", "-DWQBMMV2_S4"],
                    ["-DWEIGHT_ND", "-DWEIGHT_F4_INPUT"],
                ):
                    result = subprocess.run(
                        [
                            compiler,
                            "-std=c++11",
                            "-fsyntax-only",
                            *flags,
                            str(translation_unit),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                # An old manifest must not let a definition inside #if pass the idempotence check.
                unsafe = original.replace(
                    b'#include "blaze.h"\n',
                    b'#include "blaze.h"\n' + definitions.encode(),
                )
                unsafe = unsafe.replace(
                    "// 原始代码锚点".encode(),
                    "// 原始代码锚点\n".encode() + block.encode(),
                )
                header.write_bytes(unsafe)
                self.assertFalse(worktree.install_stub_file(*args))
                self.assertEqual(header.read_bytes(), unsafe)
                self.assertTrue(worktree.restore_stub_file(str(header)))
                self.assertEqual(header.read_bytes(), original)
                self.assertFalse(Path(str(header) + ".orig").exists())
                self.assertFalse(Path(str(header) + ".stub.json").exists())

    def test_stub_status_cli_accepts_mx_profiles_with_string_kernel_paths(self):
        worktree = load_worktree()
        with tempfile.TemporaryDirectory() as tmp:
            kernel = Path(tmp) / "mx.h"
            kernel.write_text("// TUNE PARAMS\n#define TUNE_WIDTH 32\n")
            profile = {
                "quant_modes": {
                    "mx": {
                        "templates_by_case_class": {"plain": ["custom"]},
                        "kernel_files": {"plain": {"custom": str(kernel)}},
                    }
                }
            }
            self.assertEqual(worktree._stub_modes(profile), [])
            output = io.StringIO()
            with (
                mock.patch.object(worktree, "_available_ops", return_value=["demo"]),
                mock.patch.object(worktree, "load_profile", return_value=profile),
                mock.patch("sys.stdout", output),
            ):
                worktree._run_stub_cli(["stub-status", "--op", "demo"])
            self.assertIn("无需检查", output.getvalue())
            self.assertNotIn("inconsistent", output.getvalue())
            self.assertEqual(
                kernel.read_text(), "// TUNE PARAMS\n#define TUNE_WIDTH 32\n"
            )

    def test_stub_status_and_restore_only_process_declared_stub_modes(self):
        worktree = load_worktree()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forced, normal = root / "forced.h", root / "normal.h"
            forced.write_text("// TUNE PARAMS\nworktree content\n")
            normal.write_text("// TUNE PARAMS\nstubbed content\n")
            Path(str(forced) + ".orig").write_text("unrelated backup\n")
            Path(str(normal) + ".orig").write_text("original content\n")
            Path(str(normal) + ".stub.json").write_text(
                json.dumps({"mode": "pertoken"})
            )
            entry = {
                "install": str(normal),
                "sed_anchor": "anchor",
                "insert_mode": "after",
            }
            profile = {
                "quant_modes": {
                    "mx": {"kernel_files": {"plain": {"custom": str(forced)}}},
                    "other_worktree": {
                        "kernel_files": {"plain": {"custom": str(forced)}}
                    },
                    "pertoken": {
                        "kernel_files": {
                            "plain": {"custom": entry},
                            "batch": {"custom": entry},
                        }
                    },
                }
            }
            self.assertEqual(worktree._stub_modes(profile), ["pertoken"])
            with (
                mock.patch.object(worktree, "_available_ops", return_value=["demo"]),
                mock.patch.object(worktree, "load_profile", return_value=profile),
            ):
                for args in ([], ["--mode", "pertoken"]):
                    output = io.StringIO()
                    with mock.patch("sys.stdout", output):
                        worktree._run_stub_cli(["stub-status", "--op", "demo", *args])
                    self.assertEqual(output.getvalue().count("normal.h"), 1)
                    self.assertIn("stubbed", output.getvalue())
                    self.assertNotIn("forced.h", output.getvalue())
                with mock.patch("sys.stdout", io.StringIO()):
                    self.assertTrue(worktree.restore_stub(op="demo"))
                self.assertEqual(normal.read_text(), "original content\n")
                self.assertFalse(Path(str(normal) + ".orig").exists())
                self.assertFalse(Path(str(normal) + ".stub.json").exists())
                output = io.StringIO()
                with mock.patch("sys.stdout", output):
                    worktree._run_stub_cli(["stub-status", "--op", "demo"])
                self.assertIn("clean", output.getvalue())
                self.assertNotIn("stubbed", output.getvalue())
            self.assertEqual(forced.read_text(), "// TUNE PARAMS\nworktree content\n")
            self.assertEqual(
                Path(str(forced) + ".orig").read_text(), "unrelated backup\n"
            )

    def test_anchor_self_check_distinguishes_warnings_from_errors(self):
        self.profile["soc"] = {}
        warning = "afull 不满足全载条件，跳过相关规则"
        result = {"errors": [], "warnings": [warning]}
        self.runner._CONSTRAINTS_CACHE["demo"] = types.SimpleNamespace(
            validate_rules_against_anchors=lambda anchors, params, ctx: result
        )
        anchor = {"afull": {"width": 32, "blockDim": 4, "layout": "custom"}}
        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp) / "anchors.json"
            with (
                mock.patch.object(self.runner, "extract_anchor", return_value=anchor),
                mock.patch("builtins.print") as output,
            ):
                anchors, errors = self.runner.collect_anchor_probe(
                    ["afull"], {}, log_tag="case", save_path=str(saved), strict=True
                )
                self.assertEqual(anchors, anchor)
                self.assertEqual(errors, [])
                self.assertTrue(saved.exists())
                output.assert_any_call(f"[anchor warning] {warning}")
                for result in (
                    {"errors": ["afull: capacity exceeded"], "warnings": []},
                    ["afull: capacity exceeded"],
                ):
                    with self.subTest(result=result):
                        with self.assertRaisesRegex(RuntimeError, "capacity exceeded"):
                            self.runner.collect_anchor_probe(
                                ["afull"],
                                {},
                                log_tag="case",
                                save_path=str(saved),
                                strict=True,
                            )
                        _, errors = self.runner.collect_anchor_probe(
                            ["afull"],
                            {},
                            log_tag="case",
                            save_path=str(saved),
                            strict=False,
                        )
                        self.assertEqual(errors, ["afull: capacity exceeded"])
                result = {"errors": [], "warnings": [warning]}
                with self.assertRaisesRegex(RuntimeError, "缺少锚点模板"):
                    self.runner.collect_anchor_probe(
                        ["afull", "missing"], {}, log_tag="case", save_path=str(saved)
                    )
            with mock.patch.object(
                self.runner, "extract_anchor", return_value={"afull": {"width": 32}}
            ):
                with self.assertRaisesRegex(RuntimeError, "blockDim/layout"):
                    self.runner.collect_anchor_probe(
                        ["afull"], {}, log_tag="case", save_path=str(saved)
                    )

    def test_operator_schedule_can_model_k_split_without_standard_tile_fields(self):
        self.profile["soc"] = {}
        self.profile["scheduling"] = {
            "model": "K split tasks",
            "evidence": "host.cc:20",
        }
        self.runner._RECOGNITION_CACHE["demo"] = types.SimpleNamespace(
            derive_schedule=lambda ctx: {
                "tasks": ctx["tiling_data"]["splitCount"],
                "multi_core_util": min(ctx["tiling_data"]["splitCount"], ctx["aic_num"])
                / ctx["aic_num"],
                "k_loop_count": 2,
            }
        )
        params = {
            "M": 1,
            "N": 1,
            "K": 1024,
            "x1_dtype": "float16",
            "x2_dtype": "float16",
            "scale_dtype": None,
            "y_dtype": "float16",
            "bias": "none",
            "trans_a": False,
            "trans_b": False,
            "x2_format": "ND",
        }
        baseline = self.runner.BaselineResult(
            tiling_data=self.runner.TilingData(splitCount=8)
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "info.json"
            self.runner.output_baseline_info(
                baseline, params, "case", str(path), aic_num=8
            )
            stored = json.loads(path.read_text())["baseline"]
        self.assertEqual(stored["health"]["tasks"], 8)
        self.assertEqual(stored["health"]["multi_core_util"], 1)
        self.assertNotIn("tail_load_balance", stored["health"])
        self.assertEqual(stored["scheduling_model"]["evidence"], "host.cc:20")

    def test_capacity_model_is_preserved_without_inventing_a_schedule(self):
        self.profile["soc"] = dict(
            l1_size=65536,
            l0a_size=16384,
            l0b_size=16384,
            l0c_size=16384,
            data_size_l0c=4,
            mx_group_size=32,
        )
        td = self.runner.TilingData(
            baseM=16,
            baseN=32,
            baseK=64,
            kAL1=64,
            kBL1=64,
            scaleKL1=64,
            nBufferNum=2,
            dbL0C=1,
        )
        params = dict(
            M=128,
            N=256,
            K=512,
            x1_dtype="float16",
            x2_dtype="float16",
            scale_dtype="float8_e8m0",
            y_dtype="float16",
            bias="none",
            trans_a=False,
            trans_b=False,
            x2_format="ND",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "info.json"
            self.runner.output_baseline_info(
                self.runner.BaselineResult(tiling_data=td),
                params,
                "case",
                str(path),
                aic_num=8,
            )
            stored = json.loads(path.read_text())["baseline"]
        self.assertEqual(stored["health"]["l0a_util"], "25%")
        self.assertEqual(stored["health"]["l0c_util"], "12%")
        self.assertEqual(stored["health"]["l1_remaining_bytes"], 65536 - 12480)
        self.assertNotIn("tiles", stored["health"])
        self.assertNotIn("k_loop_count", stored["health"])
        self.assertIn("scheduling_unavailable", stored)

    def test_unknown_dtype_requires_declared_size(self):
        self.profile["soc"] = {}
        with self.assertRaisesRegex(ValueError, "未知 dtype"):
            self.runner._dtype_size("new_fp")
        self.profile["dtype_sizeof"] = {"new_fp": 0.5}
        self.profile["dtype_evidence"] = {"new_fp": "types.h:12"}
        self.assertEqual(self.runner._dtype_size("new_fp"), 0.5)
        self.assertEqual(self.runner._constraints_ctx()["dtype_sizeof"]["new_fp"], 0.5)
        self.profile["dtype_sizeof"]["new_fp"] = 0
        with self.assertRaisesRegex(ValueError, "非法"):
            self.runner._dtype_size("new_fp")

    def test_zero_scale_depth_does_not_require_mx_constants(self):
        self.profile["soc"] = dict(
            l1_size=65536,
            l0a_size=16384,
            l0b_size=16384,
            l0c_size=16384,
            data_size_l0c=4,
        )
        td = self.runner.TilingData(
            baseM=16,
            baseN=32,
            baseK=64,
            kAL1=64,
            kBL1=64,
            scaleKL1=0,
            nBufferNum=2,
            dbL0C=1,
        )
        params = dict(
            M=128,
            N=256,
            K=512,
            x1_dtype="float16",
            x2_dtype="float16",
            scale_dtype=None,
            y_dtype="float16",
            bias="none",
            trans_a=False,
            trans_b=False,
            x2_format="ND",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "info.json"
            self.runner.output_baseline_info(
                self.runner.BaselineResult(tiling_data=td),
                params,
                "case",
                str(path),
                aic_num=8,
            )
            stored = json.loads(path.read_text())["baseline"]
        self.assertEqual(stored["health"]["l1_remaining_bytes"], 65536 - 12288)
        td._values["scaleKL1"] = 64
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "mx_group_size"):
                self.runner.output_baseline_info(
                    self.runner.BaselineResult(tiling_data=td),
                    params,
                    "case",
                    str(Path(tmp) / "info.json"),
                    aic_num=8,
                )

    def test_profile_validation_accepts_non_mx_and_contextual_kernel_types(self):
        self.profile.update(
            schema="v2",
            support={},
            version_anchor={},
            templates={},
            soc=dict(
                l1_size=65536,
                l0a_size=16384,
                l0b_size=16384,
                l0c_size=16384,
                data_size_l0c=4,
            ),
            case_classes={"plain": {}},
            struct_layouts={
                "a": {"u32_slots": 2, "when": {"quant_mode": "a"}},
                "b": {"u32_slots": 2, "when": {"quant_mode": "b"}},
            },
            quant_modes={
                "a": {"kernel_type_map": {"first": 0}},
                "b": {"kernel_type_map": {"second": 0}},
            },
            recognition={},
            constraints={"required_candidates_per_template": 1},
            package_gate={
                "required_files": [{"path": "dynamic/demo.py", "contains": ["demo"]}]
            },
        )
        with self.assertRaisesRegex(RuntimeError, "tiling_key"):
            self.runner._validate_profile(self.profile, "demo")
        self.profile["tiling_key"] = {"kernel_type_shift": 0, "kernel_type_bits": 8}
        self.runner._validate_profile(self.profile, "demo")
        self.profile["dtype_sizeof"] = {"custom": 2}
        with self.assertRaisesRegex(RuntimeError, "dtype_evidence"):
            self.runner._validate_profile(self.profile, "demo")
        self.profile["dtype_evidence"] = {"custom": "types.h:3"}
        self.runner._validate_profile(self.profile, "demo")

    def test_cann_resolution_follows_active_environment_or_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "cann-9.3", Path(tmp) / "cann-9.10"
            for root in (first, second):
                (root / "python/site-packages/tbe").mkdir(parents=True)
            with mock.patch.dict(
                self.runner.os.environ,
                {"ASCEND_HOME_PATH": str(second), "ASCEND_TOOLKIT_HOME": str(second)},
                clear=True,
            ):
                self.assertEqual(
                    self.runner._tbe_site_packages(),
                    str(second / "python/site-packages"),
                )
                self.assertEqual(
                    self.runner._tbe_site_packages(str(first)),
                    str(first / "python/site-packages"),
                )
            with mock.patch.dict(
                self.runner.os.environ,
                {"ASCEND_HOME_PATH": str(first), "ASCEND_TOOLKIT_HOME": str(second)},
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "不同安装"):
                    self.runner._tbe_site_packages()
            with (
                mock.patch.dict(self.runner.os.environ, {}, clear=True),
                mock.patch.object(
                    self.runner.importlib.util, "find_spec", return_value=None
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "无法唯一确定"):
                    self.runner._tbe_site_packages()

    def test_core_probe_uses_declared_semantics_and_rejects_loaded_version_mismatch(
        self,
    ):
        self.profile["soc"] = {
            "core_query": {
                "core_type": "VectorCore",
                "spec_key": "VECTOR_CORE_CNT",
                "evidence": "platform.py:155",
            }
        }
        platform = types.SimpleNamespace(
            __file__="/test/cann/python/site-packages/tbe/common/platform/__init__.py",
            set_current_compile_soc_info=mock.Mock(),
            get_soc_spec=mock.Mock(return_value="24"),
        )
        with (
            mock.patch.object(
                self.runner,
                "_tbe_site_packages",
                return_value="/test/cann/python/site-packages",
            ),
            mock.patch.object(
                self.runner.importlib, "import_module", return_value=platform
            ),
            mock.patch.object(self.runner.sys, "path", list(sys.path)),
        ):
            self.assertEqual(self.runner.get_aic_num("test-soc"), 24)
            platform.set_current_compile_soc_info.assert_called_once_with(
                "test-soc", "VectorCore"
            )
            platform.get_soc_spec.assert_called_once_with("VECTOR_CORE_CNT")
            platform.__file__ = "/other/cann/tbe/common/platform/__init__.py"
            with self.assertRaisesRegex(RuntimeError, "其他 CANN"):
                self.runner.get_aic_num("test-soc")
        self.profile["soc"] = {}
        with self.assertRaisesRegex(RuntimeError, "core_query"):
            self.runner.get_aic_num("test-soc")

    def test_forced_return_is_idempotent_when_marker_is_inside_return_statement(self):
        worktree = load_worktree()
        for statement, replacement in (
            (
                "return true; /* FORCED_MARKER trailing comment */",
                "bool Demo::IsCapable()\n{\n    <RETURN>\n",
            ),
            (
                "return true;",
                "bool Demo::IsCapable()\n{\n    // FORCED_MARKER\n"
                + " " * 1200
                + "<RETURN>\n",
            ),
        ):
            with (
                self.subTest(statement=statement),
                tempfile.TemporaryDirectory() as tmp,
            ):
                path = Path(tmp) / "host.cpp"
                path.write_text("bool Demo::IsCapable()\n{\n    return false;\n}\n")
                decl = {
                    "marker": "FORCED_MARKER",
                    "return_statement": statement,
                    "replacement_template": replacement,
                    "anchor_regex": r"bool Demo::IsCapable\(\)\s*\{",
                }
                self.assertTrue(
                    worktree.patch_forced_return(str(path), decl, "IsCapable")
                )
                patched = path.read_text()
                self.assertTrue(
                    worktree.patch_forced_return(str(path), decl, "IsCapable")
                )
                self.assertEqual(path.read_text(), patched)
                # A changed return at the target must fail even if the expected return exists elsewhere.
                path.write_text(
                    patched.replace(statement, statement.replace("true", "false"), 1)
                    + "\n// unrelated: "
                    + statement
                    + "\n"
                )
                changed = path.read_text()
                self.assertFalse(
                    worktree.patch_forced_return(str(path), decl, "IsCapable")
                )
                self.assertEqual(path.read_text(), changed)

    def generate_local_build_script(self, worktree, root, repo, run_dir=None):
        script = root / "run_build.sh"
        worktree.BUILD_DONE_FLAG = str(root / "done")
        worktree.BUILD_FAIL_FLAG = str(root / "fail")
        real_open = open

        def local_open(path, *args, **kwargs):
            return real_open(
                script if path == "/tmp/run_build.sh" else path, *args, **kwargs
            )

        with (
            mock.patch.object(worktree, "open", side_effect=local_open, create=True),
            mock.patch.object(worktree.os, "chmod"),
        ):
            worktree.generate_build_script(
                str(repo),
                jobs=2,
                op="demo",
                progress_log=str(root / "progress"),
                done_flag=worktree.BUILD_DONE_FLAG,
                fail_flag=worktree.BUILD_FAIL_FLAG,
                output_log=str(root / "output"),
                run_dir=run_dir,
            )
        return script

    def test_build_recompiles_existing_packages_for_current_operator_without_stamps(
        self,
    ):
        worktree = load_worktree()
        profile = {"identity": {"op_name": "demo"}, "soc": {"name": "test-soc"}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normal, forced = root / "repo", root / "repo_forced"
            config = types.SimpleNamespace(
                name="forced",
                path=str(forced),
                vendor_name="custom_forced",
                template_decl={"vendor_install_dir": "custom_forced_nn"},
            )
            worktree.VENDOR_BASE = str(root / "vendors")
            packages = (
                (normal, "custom_normal", "custom_normal_nn"),
                (forced, "custom_forced", "custom_forced_nn"),
            )
            for repo, vendor, install_dir in packages:
                repo.mkdir()
                (repo / "build_out").mkdir()
                (repo / f"build_out/cann-ops-nn-{vendor}_linux-x86_64.run").write_text(
                    "exit 99\n"
                )
                installer = (
                    f"mkdir -p {shlex.quote(str(root / 'vendors' / install_dir))}\n"
                )
                (repo / "build.sh").write_text(
                    'echo "$*" >> calls\nmkdir -p build_out\n'
                    f"cat > build_out/cann-ops-nn-{vendor}_linux-x86_64.run <<'INSTALLER'\n"
                    + installer
                    + "INSTALLER\n"
                )
            with (
                mock.patch.object(worktree, "load_profile", return_value=profile),
                mock.patch.object(
                    worktree, "resolve_worktree_paths", return_value=[config]
                ),
            ):
                self.assertEqual(worktree._count_need_build(str(normal), "demo"), 2)
                self.assertFalse(worktree.check_vendors_exist("demo", str(normal)))
                script = self.generate_local_build_script(worktree, root, normal)
                result = subprocess.run(
                    ["bash", str(script)], capture_output=True, text=True, timeout=10
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(worktree.collect_build_result("demo", str(normal)))
                self.assertTrue(worktree.check_vendors_exist("demo", str(normal)))
                self.assertEqual(worktree._count_need_build(str(normal), "demo"), 2)
                profile["identity"]["op_name"] = "other_operator"
                script = self.generate_local_build_script(worktree, root, normal)
                result = subprocess.run(
                    ["bash", str(script)], capture_output=True, text=True, timeout=10
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                for repo, _, _ in packages:
                    calls = (repo / "calls").read_text().splitlines()
                    self.assertEqual(len(calls), 2)
                    self.assertIn("--soc=test-soc", calls[0])
                    self.assertIn("--ops=demo", calls[0])
                    self.assertIn("--ops=other_operator", calls[1])
                with (
                    mock.patch.object(worktree, "_verify_repo_version"),
                    mock.patch.object(worktree, "create_worktree", return_value=True),
                    mock.patch.object(
                        worktree, "run_build_with_retry", return_value=True
                    ) as build,
                ):
                    self.assertTrue(worktree.run_build_all(str(normal), op="demo"))
                    build.assert_called_once_with(
                        str(normal), jobs=32, op="demo", run_dir=None
                    )
                self.assertFalse(list(root.rglob("*.solver-stamp")))
                self.assertFalse(list(root.rglob(".matmul-tiling-solver-build.json")))
                # Existing installed directories must not turn a failed/currently unfinished build into success.
                Path(worktree.BUILD_FAIL_FLAG).touch()
                self.assertFalse(worktree.collect_build_result("demo", str(normal)))
                Path(worktree.BUILD_FAIL_FLAG).unlink()
                Path(worktree.BUILD_DONE_FLAG).unlink()
                self.assertFalse(worktree.collect_build_result("demo", str(normal)))
                (root / "vendors" / "custom_forced_nn").rmdir()
                self.assertFalse(worktree.check_vendors_exist("demo", str(normal)))

    def make_cleanup_repo(self, root, worktree):
        repo = root / "repo"
        repo.mkdir()
        subprocess.run(
            ["git", "init", "-q", str(repo)], check=True, capture_output=True
        )
        (repo / "host.cpp").write_text("int value = 0;\n")
        worktree.VENDOR_BASE = str(root / "vendors")
        Path(worktree.VENDOR_BASE).mkdir()
        (repo / "build.sh").write_text(
            'for arg in "$@"; do\n'
            '  case "$arg" in --vendor_name=*) vendor=${arg#--vendor_name=} ;; esac\n'
            "done\nmkdir -p build_out\n"
            'cat > "build_out/cann-ops-nn-${vendor}_linux-x86_64.run" <<INSTALLER\n'
            f"mkdir -p {shlex.quote(worktree.VENDOR_BASE)}/" + "${vendor}_nn\n"
            f"echo new > {shlex.quote(worktree.VENDOR_BASE)}/"
            + "${vendor}_nn/new.txt\nINSTALLER\n"
        )
        subprocess.run(
            ["git", "-C", str(repo), "add", "."], check=True, capture_output=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=Solver Test",
                "-c",
                "user.email=solver@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "-qm",
                "initial",
            ],
            check=True,
            capture_output=True,
        )
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        profile = {
            "identity": {"op_name": "demo"},
            "soc": {"name": "test-soc"},
            "version_anchor": {"op_repo_commit": head},
            "templates": {
                "forced": {
                    "worktree": {"path_suffix": "_forced", "branch": "user_branch"},
                    "vendor_name": "custom_forced",
                    "vendor_install_dir": "custom_forced_nn",
                    "forced_get_kernel_type": {"return_statement": "return 1;"},
                    "forced_is_capable": {"return_statement": "return true;"},
                    "kernel_targets": [{"repo_rel": "host.cpp", "variants": ["plain"]}],
                }
            },
        }
        return repo, profile

    def test_run_cleanup_restores_environment_and_preserves_existing_user_worktree(
        self,
    ):
        for forced_existed in (False, True):
            with (
                self.subTest(forced_existed=forced_existed),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                worktree = load_worktree()
                repo, profile = self.make_cleanup_repo(root, worktree)
                legacy = root / "repo_forced"
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repo),
                        "worktree",
                        "add",
                        "-qb",
                        "user_branch",
                        str(legacy),
                    ],
                    check=True,
                    capture_output=True,
                )
                (legacy / "host.cpp").write_text("int user_change = 17;\n")
                (legacy / "user.txt").write_text("keep user file\n")
                vendors = Path(worktree.VENDOR_BASE)
                normal = vendors / "custom_normal_nn"
                normal.mkdir()
                (normal / "original.bin").write_bytes(b"user package\x00\xff")
                (normal / "link").symlink_to("original.bin")
                forced = vendors / "custom_forced_nn"
                if forced_existed:
                    forced.mkdir()
                    (forced / "original.txt").write_text("original forced package")
                config = vendors / "config.ini"
                old_config = "# 用户配置\r\nload_priority=original_vendor\r\n".encode()
                config.write_bytes(old_config)
                config.chmod(0o640)
                (repo / "build_out").mkdir()
                (repo / "build_out/user.run").write_bytes(b"old build output")
                run_dir = root / "run"
                (run_dir / "reports").mkdir(parents=True)
                report = run_dir / "reports/final_report.md"
                report.write_text("results to keep\n")

                def patch(config, dry_run=False, op=""):
                    Path(config.path, "host.cpp").write_text("int value = 1;\n")
                    return True

                def build(repo_dir, jobs, op, run_dir):
                    script = self.generate_local_build_script(
                        worktree, root, repo, run_dir=run_dir
                    )
                    result = subprocess.run(
                        ["bash", str(script)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    return True

                with (
                    mock.patch.object(worktree, "load_profile", return_value=profile),
                    mock.patch.object(
                        worktree, "_patch_worktree_files", side_effect=patch
                    ),
                    mock.patch.object(
                        worktree, "run_build_with_retry", side_effect=build
                    ),
                ):
                    self.assertTrue(
                        worktree.run_build_all(
                            str(repo), op="demo", run_dir=str(run_dir)
                        )
                    )
                    state = worktree._read_environment(run_dir)
                    temporary = Path(state["worktrees"][0]["path"])
                    self.assertNotEqual(temporary, legacy)
                    self.assertTrue(temporary.exists())
                    self.assertTrue(state["worktrees"][0]["created"])
                    self.assertNotEqual(config.read_bytes(), old_config)
                    # A repeated prepare must retain the original snapshot.
                    worktree.prepare_run_environment(str(repo), str(run_dir), "demo")
                    self.assertTrue(worktree.cleanup_run_environment(str(run_dir)))
                    self.assertTrue(worktree.cleanup_run_environment(str(run_dir)))
                self.assertFalse(temporary.exists())
                self.assertEqual(
                    (legacy / "host.cpp").read_text(), "int user_change = 17;\n"
                )
                self.assertEqual((legacy / "user.txt").read_text(), "keep user file\n")
                registered = subprocess.run(
                    ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                self.assertIn(str(legacy), registered)
                self.assertNotIn(str(temporary), registered)
                self.assertEqual(config.read_bytes(), old_config)
                self.assertEqual(config.stat().st_mode & 0o777, 0o640)
                self.assertEqual(
                    (normal / "original.bin").read_bytes(), b"user package\x00\xff"
                )
                self.assertEqual(os.readlink(normal / "link"), "original.bin")
                self.assertFalse((normal / "new.txt").exists())
                if forced_existed:
                    self.assertEqual(
                        (forced / "original.txt").read_text(), "original forced package"
                    )
                    self.assertFalse((forced / "new.txt").exists())
                else:
                    self.assertFalse(forced.exists())
                self.assertEqual(
                    list((repo / "build_out").iterdir()), [repo / "build_out/user.run"]
                )
                self.assertEqual(report.read_text(), "results to keep\n")
                patches = list((run_dir / "build/worktree_patches").glob("*.patch"))
                self.assertEqual(len(patches), 1)
                self.assertIn("+int value = 1;", patches[0].read_text())
                self.assertFalse((run_dir / "build/environment_backup").exists())
                self.assertTrue(worktree._read_environment(run_dir)["cleaned"])

    def test_run_cleanup_handles_partial_preparation_and_failed_cleanup_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = load_worktree()
            repo, profile = self.make_cleanup_repo(root, worktree)
            run_dir = root / "run"
            with (
                mock.patch.object(worktree, "load_profile", return_value=profile),
                mock.patch.object(
                    worktree, "_patch_worktree_files", return_value=False
                ),
            ):
                self.assertFalse(
                    worktree.run_build_all(str(repo), op="demo", run_dir=str(run_dir))
                )
                state = worktree._read_environment(run_dir)
                path = state["worktrees"][0]["path"]
                subprocess.run(
                    ["git", "-C", str(repo), "worktree", "lock", path],
                    check=True,
                    capture_output=True,
                )
                vendors = Path(worktree.VENDOR_BASE)
                (vendors / "config.ini").write_text("load_priority=temporary\n")
                (vendors / "custom_normal_nn").mkdir()
                (vendors / "custom_normal_nn/new.txt").write_text("temporary package")
                (repo / "build_out").mkdir()
                (repo / "build_out/temp.run").touch()
                self.assertFalse(worktree.cleanup_run_environment(str(run_dir)))
                self.assertTrue(Path(path).exists())
                self.assertFalse((vendors / "config.ini").exists())
                self.assertFalse((vendors / "custom_normal_nn").exists())
                self.assertFalse((repo / "build_out").exists())
                self.assertTrue((run_dir / "build/environment_backup").exists())
                subprocess.run(
                    ["git", "-C", str(repo), "worktree", "unlock", path],
                    check=True,
                    capture_output=True,
                )
                # Also tolerate a manual Git removal before cleanup progress was saved.
                subprocess.run(
                    ["git", "-C", str(repo), "worktree", "remove", "--force", path],
                    check=True,
                    capture_output=True,
                )
                self.assertTrue(worktree.cleanup_run_environment(str(run_dir)))
                self.assertFalse(Path(path).exists())

    def test_run_cleanup_refuses_to_guess_without_snapshot(self):
        worktree = load_worktree()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_file = root / "keep.txt"
            user_file.write_text("keep")
            with self.assertRaises(FileNotFoundError):
                worktree.cleanup_run_environment(str(root))
            self.assertEqual(user_file.read_text(), "keep")

    def test_build_reports_compile_install_and_missing_output_failures(self):
        profile = {"identity": {"op_name": "demo"}, "soc": {"name": "test-soc"}}
        for scenario, diagnostic in (
            ("compile", "compile rc=9"),
            ("no_run", "produced no run file"),
            ("install", "install rc=8"),
            ("no_install_dir", "install directory missing"),
        ):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                repo = root / "repo"
                repo.mkdir()
                vendor_base = root / "vendors"
                vendor_base.mkdir()
                worktree = load_worktree()
                worktree.VENDOR_BASE = str(vendor_base)
                if scenario != "no_install_dir":
                    (vendor_base / "custom_normal_nn").mkdir()
                builder = "mkdir -p build_out\n"
                if scenario != "no_run":
                    installer = "exit 8\n" if scenario == "install" else "exit 0\n"
                    builder += "cat > build_out/cann-ops-nn-custom_normal_linux-x86_64.run <<'INSTALLER'\n"
                    builder += installer + "INSTALLER\n"
                builder += "exit 9\n" if scenario == "compile" else "exit 0\n"
                (repo / "build.sh").write_text(builder)
                with (
                    mock.patch.object(worktree, "load_profile", return_value=profile),
                    mock.patch.object(
                        worktree, "resolve_worktree_paths", return_value=[]
                    ),
                ):
                    script = self.generate_local_build_script(worktree, root, repo)
                    result = subprocess.run(
                        ["bash", str(script)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertTrue(Path(worktree.BUILD_FAIL_FLAG).exists())
                    self.assertFalse(Path(worktree.BUILD_DONE_FLAG).exists())
                    self.assertIn(diagnostic, (root / "progress").read_text())
                    self.assertFalse(worktree.collect_build_result("demo", str(repo)))
                    if scenario == "compile":
                        self.assertFalse(
                            (
                                repo
                                / "build_out/cann-ops-nn-custom_normal_linux-x86_64.run"
                            ).exists()
                        )

    def test_baseline_batch_budgets_the_entire_csv_process(self):
        params = [{"M": 3, "N": 7, "K": 5}]
        params += [{"M": 1024, "N": 1024, "K": 1024} for _ in range(27)]
        params += [{"M": 1, "N": 1, "K": 20001}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path_names = ("script", "output", "progress", "done")
            paths = {key: str(root / key) for key in path_names}
            self.runner.BASELINE_LAST_TAG_FILE = str(root / "last_tag")
            self.runner.VENDOR_PACKAGES = {"normal": "test_normal"}
            exists = self.runner.os.path.exists
            with (
                mock.patch.object(self.runner, "_require_runtime_paths"),
                mock.patch.object(self.runner, "baseline_paths", return_value=paths),
                mock.patch.object(self.runner.subprocess, "Popen"),
                mock.patch.object(
                    self.runner.os.path,
                    "exists",
                    side_effect=lambda p: False
                    if p == "/tmp/baseline_done.flag"
                    else exists(p),
                ),
            ):
                eta = self.runner.start_baseline_batch(str(root / "cases.csv"), params)
            script = Path(paths["script"]).read_text()
            self.assertIn("timeout 34800 python3", script)
            self.assertIn("--npu-timeout 90000", script)
            self.assertEqual(eta, 2580)

    def test_baseline_launch_monitor_and_collection_share_the_same_tag(self):
        self.profile["tiling_key"] = {"kernel_type_shift": 0, "kernel_type_bits": 8}
        self.profile["quant_modes"] = {"mode": {"kernel_type_map": {"custom": 2}}}
        self.profile["struct_layouts"] = {
            "custom": {
                "u32_slots": 1,
                "fields": {"chunkWidth": {"offset": 0, "fmt": "I"}},
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.runner.BASELINE_LAST_TAG_FILE = str(root / "last_tag")
            self.runner.VENDOR_PACKAGES = {"normal": "test_normal"}
            self.runner.VENDOR_CONFIG = str(root / "config.ini")
            self.runner.KERNEL_META_DIR = str(root / "kernel_meta")
            self.runner.TTK_DIR = str(root / "ttk")
            self.runner.PLUGIN_DIR = str(root / "plugin")
            exists = self.runner.os.path.exists
            for supplied_tag in ("", root.name + "_explicit"):
                case_csv = str(root / (root.name + "_case.csv"))
                tag = supplied_tag or self.runner._tag_from_case_csv(case_csv)
                paths = self.runner.baseline_paths(tag)
                try:
                    with (
                        mock.patch.object(
                            self.runner, "_require_runtime_paths", return_value=None
                        ),
                        mock.patch.object(self.runner.subprocess, "Popen") as launch,
                        mock.patch.object(
                            self.runner.os.path,
                            "exists",
                            side_effect=lambda p: False
                            if p == "/tmp/baseline_done.flag"
                            else exists(p),
                        ),
                    ):
                        self.runner.start_baseline(
                            case_csv, 3, 7, 5, log_tag=supplied_tag
                        )
                    script = launch.call_args.args[0][1]
                    self.assertEqual(script, paths["script"])
                    self.assertTrue(Path(script).exists())
                    with mock.patch.object(
                        self.runner,
                        "_runner_script_alive",
                        side_effect=lambda p: p == script,
                    ) as alive:
                        self.assertFalse(self.runner.check_task_done(paths["done"]))
                        self.assertIn(mock.call(script), alive.call_args_list)
                    Path(paths["done"]).touch()
                    self.assertTrue(self.runner.check_task_done(paths["done"]))
                    Path(paths["output"]).write_text(
                        "DYN_PERF: 8\nPRECISION_STATUS: PASS\n"
                    )
                    self.write_csv(
                        Path(case_csv.replace(".csv", "_result.csv")),
                        [
                            {
                                "testcase_name": "case",
                                "dyn_tiling_data": "(32,)",
                                "dyn_tiling_key": "0x2",
                            }
                        ],
                    )
                    result = self.runner.collect_baseline(
                        case_csv, log_tag=supplied_tag
                    )
                    self.assertEqual(result.perf_us, 8)
                    self.assertEqual(result.precision_status, "PASS")
                    self.assertEqual(result.tiling_data.to_dict(), {"chunkWidth": 32})
                finally:
                    for path in paths.values():
                        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
