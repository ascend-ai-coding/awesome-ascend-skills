# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the shared explicit exit-criteria evaluator."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "criteria_test", Path(__file__).resolve().parents[1] / "exit_criteria.py"
)
criteria = importlib.util.module_from_spec(spec)
spec.loader.exec_module(criteria)


def condition(identifier="v", *, basis="model", aggregation="each"):
    return {
        "id": identifier,
        "metric": "vector_pipe_utilization_pct",
        "operator": ">=",
        "threshold": 70,
        "unit": "%",
        "basis": basis,
        "cases": ["p0", "p1"],
        "aggregation": aggregation,
        "source": "test_model",
    }


def observation(case, value, **kwargs):
    return {
        "metric": "vector_pipe_utilization_pct",
        "case": case,
        "value": value,
        "unit": "%",
        "basis": "model",
        "source": "test_model",
        **kwargs,
    }


class CriteriaTests(unittest.TestCase):
    def test_missing_case_is_unknown(self):
        report = criteria.evaluate_exit_criteria(condition(), [observation("p0", 90)])
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertFalse(report["met"])

    def test_incompatible_basis_and_unit_do_not_pass(self):
        for changes in ({"basis": "hardware"}, {"unit": "us"}):
            values = [observation(case, 100, **changes) for case in ("p0", "p1")]
            self.assertEqual(
                criteria.evaluate_exit_criteria(condition(), values)["status"],
                "UNKNOWN",
            )

    def test_all_any_and_aggregation(self):
        a, b = condition("a"), condition("b", aggregation="mean")
        values = [observation("p0", 60), observation("p1", 90)]
        self.assertFalse(
            criteria.evaluate_exit_criteria({"all": [a, b]}, values)["met"]
        )
        self.assertTrue(criteria.evaluate_exit_criteria({"any": [a, b]}, values)["met"])

    def test_empty_duplicate_and_nonfinite_criteria_rejected(self):
        bad = [
            {"all": []},
            {"any": []},
            {"all": [condition(), condition()]},
            {**condition(), "threshold": float("nan")},
            {**condition(), "cases": ["p0", "p0"]},
        ]
        for tree in bad:
            with self.subTest(tree=tree), self.assertRaises(ValueError):
                criteria.validate_exit_criteria(tree)

    def test_absent_condition_is_not_vacuously_met(self):
        self.assertEqual(
            criteria.evaluate_exit_criteria(None, []),
            {"status": "NOT_REQUESTED", "met": False, "conditions": []},
        )

    def test_conflicting_observations_are_unknown(self):
        values = [observation("p0", 90), observation("p0", 40), observation("p1", 90)]
        self.assertFalse(criteria.evaluate_exit_criteria(condition(), values)["met"])


if __name__ == "__main__":
    unittest.main()
