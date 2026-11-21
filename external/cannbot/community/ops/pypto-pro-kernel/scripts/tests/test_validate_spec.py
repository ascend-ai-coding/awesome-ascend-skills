#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regression tests for the canonical SPEC JSON contract."""

from __future__ import annotations

import tempfile
import importlib.util
import json
import unittest
from pathlib import Path

validator_spec = importlib.util.spec_from_file_location(
    "kernel_spec_contract", Path(__file__).resolve().parents[1] / "validate_spec.py"
)
validator = importlib.util.module_from_spec(validator_spec)
validator_spec.loader.exec_module(validator)
_extract = validator._extract
load_spec_contract = validator.load_spec_contract
load_spec_contract_text = validator.load_spec_contract_text
validate_text = validator.validate_text


VALID = '''```json machine-contract
{
  "schema_version": 1,
  "op_name": "demo_op",
  "formula": "y = x",
  "supported_dtypes": ["float32"],
  "inputs": [{"name": "x", "shape": ["N", 16], "dtype": "float32", "value_range": [-4, 4]}],
  "outputs": [{"name": "y", "shape": ["N", 16], "dtype": "float32", "value_range": [-4, 4]}],
  "default_params": {},
  "tolerance": {"atol": 0.001, "rtol": 0.002},
  "dynamic_axes_ranges": {"N": [1, 128]},
  "shape_constraints": [],
  "perf_target": null,
  "p0_cases": [
    {"name": "small", "params": {}, "input_shapes": {"x": [8, 16]}, "output_shapes": {"y": [8, 16]}},
    {"name": "large", "params": {}, "input_shapes": {"x": [32, 16]}, "output_shapes": {"y": [32, 16]}}
  ]
}
```

## Semantic explanation
y = x
'''


class SpecValidationTests(unittest.TestCase):
    def test_explicit_integer_atol_contract(self):
        contract = _extract(VALID)
        for amount in (0, 1, 3):
            contract["tolerance"]["integer_atol"] = amount
            self.assertEqual(
                load_spec_contract_text(
                    "```json machine-contract\n" + json.dumps(contract) + "\n```"
                )["tolerance"]["integer_atol"],
                amount,
            )
        for amount in (-1, True, 1.5, "1"):
            contract["tolerance"]["integer_atol"] = amount
            self.assertIn(
                "integer_atol",
                validate_text(
                    "```json machine-contract\n" + json.dumps(contract) + "\n```"
                )[0],
            )

    def test_p0_special_inputs_are_explicit_and_typed(self) -> None:
        contract = _extract(VALID)
        contract["p0_cases"][0]["input_special_values"] = {"x": ["-inf", "+inf"]}
        render = lambda value: "```json machine-contract\n" + json.dumps(value) + "\n```"
        self.assertEqual(load_spec_contract_text(render(contract))["p0_cases"][0]
                         ["input_special_values"]["x"], ["-inf", "+inf"])
        for values, error in (({"wrong": ["nan"]}, "declared input names"),
                              ({"x": ["inf"]}, "unique -inf, +inf or nan"),
                              ({"x": ["nan", "nan"]}, "unique -inf, +inf or nan")):
            changed = json.loads(json.dumps(contract))
            changed["p0_cases"][0]["input_special_values"] = values
            self.assertIn(error, validate_text(render(changed))[0])
        contract["inputs"][0]["dtype"] = "int32"
        contract["outputs"][0]["dtype"] = "int32"
        contract["supported_dtypes"] = ["int32"]
        self.assertIn("floating input dtype", validate_text(render(contract))[0])

    def test_rank_polymorphic_interface_keeps_exact_p0_shapes(self) -> None:
        contract = _extract(VALID)
        contract["inputs"][0]["shape"] = ["..."]
        contract["outputs"][0]["shape"] = ["..."]
        contract["dynamic_axes_ranges"] = {}
        contract["p0_cases"][1]["input_shapes"] = {"x": [2, 4, 16]}
        contract["p0_cases"][1]["output_shapes"] = {"y": [2, 4, 16]}
        spec = "```json machine-contract\n" + json.dumps(contract) + "\n```"
        parsed = load_spec_contract_text(spec)
        self.assertEqual(parsed["p0_cases"][1]["input_shapes"]["x"], [2, 4, 16])
        changed = json.loads(json.dumps(contract))
        changed["inputs"][0]["shape"] = ["...", 16]
        self.assertIn("rank wildcard", validate_text("```json machine-contract\n" + json.dumps(changed) + "\n```")[0])

    def test_per_case_dtypes_cover_one_public_interface(self) -> None:
        contract = _extract(VALID)
        contract["supported_dtypes"] = ["float32", "float16", "bfloat16"]
        for case, dtype in zip(contract["p0_cases"], ("float16", "bfloat16")):
            case["input_dtypes"] = {"x": dtype}
            case["output_dtypes"] = {"y": dtype}
        # The first case is the canonical tensor default.
        contract["p0_cases"][0]["input_dtypes"] = {"x": "float32"}
        contract["p0_cases"][0]["output_dtypes"] = {"y": "float32"}
        contract["p0_cases"].append({"name": "bf16", "params": {},
            "input_shapes": {"x": [64, 16]}, "output_shapes": {"y": [64, 16]},
            "input_dtypes": {"x": "bfloat16"}, "output_dtypes": {"y": "bfloat16"}})
        contract["p0_cases"][1]["input_dtypes"] = {"x": "float16"}
        contract["p0_cases"][1]["output_dtypes"] = {"y": "float16"}
        spec = "```json machine-contract\n" + json.dumps(contract) + "\n```"
        self.assertEqual(load_spec_contract_text(spec)["p0_cases"][2]["output_dtypes"]["y"], "bfloat16")
        for mutate, message in (
            (lambda c: c["p0_cases"][1].pop("output_dtypes"), "together"),
            (lambda c: c["p0_cases"][1]["input_dtypes"].update(z="float16"), "names/order"),
            (lambda c: c["p0_cases"][1]["input_dtypes"].update(x="bf16"), "canonical"),
            (lambda c: c.update(supported_dtypes=["float32", "float16"]), "first-appearance"),
        ):
            with self.subTest(message=message):
                changed = json.loads(json.dumps(contract))
                mutate(changed)
                self.assertIn(message, validate_text("```json machine-contract\n" + json.dumps(changed) + "\n```")[0])

    def test_explicit_workflow_precision_policy(self) -> None:
        spec = VALID.replace('{"atol": 0.001, "rtol": 0.002}', '{"policy": "pro_scheme_a"}')
        self.assertFalse(validate_text(spec))
        for policy in ('{"policy": "unknown"}', '{"policy": "pro_scheme_a", "atol": 1, "rtol": 1}'):
            self.assertTrue(validate_text(spec.replace('{"policy": "pro_scheme_a"}', policy)))

    def test_valid_contract_and_compatibility_field(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "SPEC.md"
            path.write_text(VALID, encoding="utf-8")
            contract = load_spec_contract(path)
        self.assertEqual(contract["p0_shapes"], [[8, 16]])
        self.assertEqual(len(contract["p0_cases"]), 2)

    def test_rejects_duplicate_key_and_placeholder(self) -> None:
        duplicate = VALID.replace(
            '"schema_version": 1,',
            '"schema_version": 1,\n  "schema_version": 1,',
        )
        self.assertIn("duplicate JSON key", validate_text(duplicate)[0])
        self.assertIn("placeholder", validate_text(VALID.replace("demo_op", "{{OP_NAME}}"))[0])

    def test_rejects_schema_and_p0_shape_errors(self) -> None:
        self.assertTrue(validate_text(
            VALID.replace('"supported_dtypes": ["float32"]', '"supported_dtypes": []')
        ))
        bad = VALID.replace(
            '"output_shapes": {"y": [32, 16]}',
            '"output_shapes": {"y": [31, 16]}',
        )
        self.assertIn("does not equal", validate_text(bad)[0])

    def test_requires_auditable_formula(self) -> None:
        missing = VALID.replace('  "formula": "y = x",\n', "")
        self.assertIn("missing fields: formula", validate_text(missing)[0])

        empty = VALID.replace('"formula": "y = x"', '"formula": ""')
        self.assertIn("non-empty string", validate_text(empty)[0])

        non_string = VALID.replace('"formula": "y = x"', '"formula": ["y = x"]')
        self.assertIn("non-empty string", validate_text(non_string)[0])

    def test_rejects_second_contract_and_unknown_fields(self) -> None:
        self.assertIn("exactly one", validate_text(VALID + VALID)[0])
        bad = VALID.replace('"schema_version": 1,', '"schema_version": 1,\n  "extra": true,')
        self.assertIn("unknown fields", validate_text(bad)[0])

    def test_perf_target_is_null_or_positive_finite_number(self) -> None:
        self.assertEqual(validate_text(VALID), [])
        self.assertEqual(validate_text(VALID.replace('"perf_target": null',
                                                     '"perf_target": 1.25')), [])
        for invalid in ('"未指定"', "true", "0", "-1", "{}"):
            with self.subTest(invalid=invalid):
                errors = validate_text(
                    VALID.replace('"perf_target": null', f'"perf_target": {invalid}'))
                self.assertIn("perf_target", errors[0])
        non_finite = validate_text(
            VALID.replace('"perf_target": null', '"perf_target": NaN'))
        self.assertIn("non-finite JSON number", non_finite[0])

    def test_rejects_unknown_dtype_and_dynamic_symbol_drift(self) -> None:
        unknown_dtype = VALID.replace('"float32"', '"banana"')
        self.assertIn("canonical dtypes", validate_text(unknown_dtype)[0])

        missing_range = VALID.replace('"dynamic_axes_ranges": {"N": [1, 128]}',
                                      '"dynamic_axes_ranges": {}')
        self.assertIn("exactly match shape symbols", validate_text(missing_range)[0])

        unused_range = VALID.replace('"dynamic_axes_ranges": {"N": [1, 128]}',
                                     '"dynamic_axes_ranges": {"N": [1, 128], "Z": [1, 8]}')
        self.assertIn("exactly match shape symbols", validate_text(unused_range)[0])

    def test_preserves_mixed_public_tensor_dtypes(self) -> None:
        mixed = VALID.replace(
            '"supported_dtypes": ["float32"]',
            '"supported_dtypes": ["int32", "float32"]',
        ).replace(
            '"name": "x", "shape": ["N", 16], "dtype": "float32"',
            '"name": "x", "shape": ["N", 16], "dtype": "int32"',
        )
        contract = load_spec_contract_text(mixed)
        self.assertEqual(contract["supported_dtypes"], ["int32", "float32"])
        self.assertEqual(contract["inputs"][0]["dtype"], "int32")

    def test_requires_each_shape_symbol_to_have_an_input_anchor(self) -> None:
        unanchored = VALID.replace('["N", 16]', '["2*N", 16]')
        self.assertIn("standalone dimension", validate_text(unanchored)[0])


if __name__ == "__main__":
    unittest.main()
