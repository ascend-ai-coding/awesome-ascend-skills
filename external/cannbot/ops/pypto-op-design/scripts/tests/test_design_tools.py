#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Behavior checks for the syntax-only design helpers (CPU, standard unittest)."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "estimate", SCRIPTS / "estimate_decomposition.py"
)
estimate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimate)
spec = importlib.util.spec_from_file_location(
    "validate", SCRIPTS / "validate_artifacts.py"
)
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)


class DesignToolsTest(unittest.TestCase):
    def test_complete_design_requires_valid_interfaces(self):
        import yaml
        from validate_yaml import _VALID

        with tempfile.TemporaryDirectory() as folder:
            op = Path(folder)
            (op / "DESIGN.md").write_text(
                (SCRIPTS.parent / "templates/DESIGN.md.tmpl").read_text()
            )
            command = [
                sys.executable,
                str(SCRIPTS / "validate_artifacts.py"),
                "--op-dir",
                str(op),
            ]
            missing = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            (op / "eval").mkdir()
            data = dict(
                _VALID, schema_version=1, op="demo", composition_verification={}
            )
            path = op / "eval/module_interfaces.yaml"
            path.write_text(yaml.safe_dump(data))
            valid = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
            data["modules"] = []
            path.write_text(yaml.safe_dump(data))
            empty = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(empty.returncode, 0)

    def test_function_scope_and_no_execution(self):
        source = '''raise RuntimeError("must never execute")
def unrelated():
    return x @ y
def demo_golden(x):
    """Not an effective statement."""
    def unused():
        return x @ x
    for i in range(2):
        x = x @ x
    return x.sum()
'''
        result = estimate.estimate(source)
        self.assertEqual(
            result["signals"],
            dict(
                statements=3, loops=1, matmul_calls=1, einsum_calls=0, reduction_calls=1
            ),
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "golden.py"
            path.write_text(source)
            run = subprocess.run(
                [sys.executable, str(SCRIPTS / "estimate_decomposition.py"), str(path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_ambiguous_entry_requires_selection(self):
        source = "def a_golden(): return 1\ndef b_golden(): return 2\n"
        with self.assertRaises(ValueError):
            estimate.estimate(source)
        self.assertEqual(estimate.estimate(source, "b_golden")["function"], "b_golden")

    def test_code_block_is_not_document_structure(self):
        self.assertEqual(
            validate.headings("```python\n## example\n```\n## Actual\n"), {"actual"}
        )

    def test_frontmatter_requires_dynamic_axes(self):
        contract = {
            "artifacts": {
                "DESIGN": {"frontmatter": {"required": ["op_name", "dynamic_axes"]}}
            }
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "DESIGN.md"
            path.write_text("---\nop_name: demo\ndynamic_axes: [B]\n---\n")
            self.assertEqual(validate.validate_design(path, contract), [])
            for value in ("[]", "B", "null", "{}"):
                path.write_text(f"---\nop_name: demo\ndynamic_axes: {value}\n---\n")
                self.assertTrue(validate.validate_design(path, contract))


if __name__ == "__main__":
    unittest.main()
