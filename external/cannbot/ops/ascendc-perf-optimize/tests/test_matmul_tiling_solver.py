# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "references" / "tiling" / "matmul" / "solver"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MatmulTilingSolverTest(unittest.TestCase):
    def test_moved_engines_resolve_profiles_from_references(self):
        runner = _load("matmul_ttk_runner_paths", SKILL / "scripts" / "ttk_runner.py")
        worktree = _load("matmul_worktree_setup_paths", SKILL / "scripts" / "worktree_setup.py")
        expected = SKILL / "references" / "profiles"

        self.assertEqual(Path(runner._PROFILE_ROOT), expected)
        self.assertEqual(Path(runner._PATHS_SIDECAR), SKILL / "scripts" / "paths.json")
        with self.assertRaises(FileNotFoundError) as caught:
            worktree.load_profile("definitely_missing_profile")
        self.assertIn(str(expected), str(caught.exception))

    def test_profile_scaffold_is_safe_and_cannot_be_loaded(self):
        scaffold = _load("profile_scaffold", SKILL / "scripts" / "create_profile_scaffold.py")
        runner = _load("matmul_ttk_runner_scaffold", SKILL / "scripts" / "ttk_runner.py")

        with tempfile.TemporaryDirectory() as tmp:
            target = scaffold.create_scaffold("demo_matmul", "matmul/demo_matmul", "abcdef1", tmp)
            profile = json.loads((target / "profile.json").read_text())

            self.assertEqual(profile["identity"]["op_name"], "demo_matmul")
            self.assertEqual(profile["version_anchor"]["op_repo_commit"], "abcdef1")
            self.assertEqual(
                {p.name for p in target.iterdir() if p.is_file()},
                {"profile.json", "recognition.py", "constraints.py", "constraints.md"},
            )
            for path in target.iterdir():
                if path.is_file():
                    text = path.read_text()
                    self.assertNotIn("TO" + "DO", text)
                    self.assertNotIn("replace" + "_me", text)
            with self.assertRaisesRegex(RuntimeError, "仍是草稿"):
                runner._validate_profile(profile, "demo_matmul")

    def test_anchor_probe_script_has_deterministic_markers_and_no_macro_edit(self):
        runner = _load("matmul_ttk_runner_anchor", SKILL / "scripts" / "ttk_runner.py")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = {
                "script": str(tmp_path / "run.sh"),
                "output": str(tmp_path / "output.log"),
                "progress": str(tmp_path / "progress.log"),
                "done": str(tmp_path / "done.flag"),
            }
            patches = (
                mock.patch.object(runner, "anchor_paths", lambda _tag="": paths),
                mock.patch.object(runner, "_require_runtime_paths", lambda: None),
                mock.patch.object(
                    runner, "_require_profile", lambda: {"quant_modes": {"mode": {"tune_macro_define_block": ""}}}
                ),
                mock.patch.object(
                    runner,
                    "injectable_templates",
                    lambda _case, _mode, natural_template="": ["swat", "afull"],
                ),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                runner.VENDOR_PACKAGES = {"normal": "normal_pkg", "swat": "swat_pkg", "afull": "afull_pkg"}
                runner.VENDOR_CONFIG = str(tmp_path / "config.ini")
                runner.KERNEL_META_DIR = str(tmp_path / "kernel_meta")
                runner.TTK_DIR = str(tmp_path / "ttk")
                runner.PLUGIN_DIR = str(tmp_path / "plugin")
                script_path = runner.generate_anchor_probe_script(
                    "case.csv",
                    ["swat", "afull"],
                    128,
                    256,
                    512,
                    log_tag="c1",
                    quant_mode="mode",
                )

            text = Path(script_path).read_text()
            self.assertIn("SCAN_START swat 0", text)
            self.assertIn("SCAN_END swat 0", text)
            self.assertIn("SCAN_START afull 0", text)
            self.assertIn("SCAN_END afull 0", text)
            self.assertIn("load_priority=swat_pkg", text)
            self.assertIn("load_priority=afull_pkg", text)
            self.assertNotIn("sed ", text)
            self.assertNotIn("TUNE_", text)
            self.assertIn("--clean-artifacts", text)
            self.assertNotIn("find ", text)
            self.assertNotIn("-exec", text)
            self.assertNotIn("rm -", text)

    def test_package_gate_checks_discovery_file_and_persists_report(self):
        runner = _load("matmul_ttk_runner_package", SKILL / "scripts" / "ttk_runner.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendors = root / "vendors"
            package = "normal_pkg"
            dynamic = vendors / package / "op_impl" / "ai_core" / "tbe" / f"{package}_impl" / "dynamic" / "demo.py"
            dynamic.parent.mkdir(parents=True)
            dynamic.write_text("def demo():\n    return 'demo'\n")
            report_path = root / "package_gate.json"
            profile = {
                "package_gate": {
                    "required_files": [
                        {
                            "path": "op_impl/ai_core/tbe/{package}_impl/dynamic/demo.py",
                            "package_scope": "normal",
                            "syntax": "python",
                            "contains": ["def demo"],
                        }
                    ]
                }
            }
            with (
                mock.patch.object(runner, "_require_runtime_paths", lambda: None),
                mock.patch.object(runner, "_require_profile", lambda: profile),
            ):
                runner._ACTIVE_OP_NAME = "demo"
                runner.VENDOR_CONFIG = str(vendors / "config.ini")
                runner.VENDOR_PACKAGES = {"normal": package}
                report = runner.validate_vendor_package_gate(templates=[], output_path=str(report_path))

            self.assertTrue(report["ok"])
            self.assertEqual(json.loads(report_path.read_text())["ok"], True)


if __name__ == "__main__":
    unittest.main()
