# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import json
import unittest
from pathlib import Path


EVALS_PATH = Path(__file__).resolve().parents[1] / "evals" / "evals.json"

REQUIRED_CASE_KEYS = {
    "id",
    "title",
    "config",
    "prompt",
    "expected_output",
    "files",
}
PROMPT_FACTS = {
    1: ("3.640416",),
    2: ("aiv_mte2_active_bw", "aiv_mte3_active_bw"),
    4: ("3.430269", "3.891671"),
    5: ("Ascend950PR_9599 V100", "62061.89"),
    6: ("62063.16", "62061.89"),
    8: ("0.097392", "0.096824"),
    9: ("0.002268", "0.730649"),
    10: ("displayTimeUnit", "SCALAR", "VECTOR", "MTE2"),
    11: ("displayTimeUnit", '"ts":2.5', '"dur":0.25'),
    12: ("process2.result3.device0.replay1.group4.veccore1",),
    13: ("traceEvents", '"ts":1.0', '"dur":2.0'),
    14: ("./add_custom", "Memory"),
    15: ("run.sh", "Memory", "L2Cache", "--input input.bin", "--repeat 2"),
    16: ("场景 A", "场景 B", "examples/add", "./add_custom"),
    17: ("reports/input.npu-rep", "unpacked-results"),
    18: (
        "unpacked-results/npu-compute-import-123",
        "collection-p42-0001",
        "unknown.bin",
    ),
    19: ("summary.jsonl", '"category":"Memory"', '"category":"OpInfoSummary"'),
    20: ("group0.veccore0", "group5.cubecore"),
    21: ("./add_custom",),
    22: ("./add_custom",),
    23: ("./add_custom",),
    24: ("./add_custom",),
    25: ("./add_custom",),
    27: ("./add_custom",),
    29: ("./add_custom", "--input input.bin"),
}


class SkillEvalsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(EVALS_PATH.read_text(encoding="utf-8"))
        cls.data = cls.document
        cls.evals = cls.document["evals"]
        cls.evals_by_id = {item["id"]: item for item in cls.evals}

    def test_top_level_uses_text_evaluation_mode(self):
        self.assertEqual(self.document["skill_name"], "tool-npu-compute")
        self.assertEqual(self.document.get("eval_mode"), "text")

    def test_case_ids_are_unique_and_sequential(self):
        ids = [case["id"] for case in self.evals]
        self.assertEqual(ids, list(range(1, len(ids) + 1)))

    def test_cases_follow_cannbot_schema(self):
        for case in self.evals:
            with self.subTest(case_id=case["id"]):
                self.assertTrue(REQUIRED_CASE_KEYS.issubset(case))
                self.assertNotIn("name", case)
                self.assertIsInstance(case["title"], str)
                self.assertTrue(case["title"].strip())
                self.assertIsInstance(case["config"], dict)
                self.assertEqual(case["config"]["eval_mode"], "text")
                self.assertGreater(case["config"]["max_tokens"], 0)

    def test_deepseek_token_budgets_match_eval_complexity(self):
        for case in self.evals:
            expected_budget = 100000 if case["id"] <= 20 else 140000
            with self.subTest(case_id=case["id"]):
                self.assertEqual(
                    case["config"].get("max_tokens_by_model", {}).get(
                        "deepseek-v4-flash"
                    ),
                    expected_budget,
                )

    def test_evaluation_inputs_are_self_contained(self):
        for case in self.evals:
            with self.subTest(case_id=case["id"]):
                self.assertEqual(case["files"], [])
                for fact in PROMPT_FACTS.get(case["id"], ()):
                    self.assertIn(fact, case["prompt"])


TRIGGER_SCENARIOS = {
    "positive_tool_name": (True, "tool_name"),
    "positive_case_variant": (True, "tool_name"),
    "positive_report": (True, "artifact"),
    "positive_trace": (True, "artifact"),
    "positive_section": (True, "metric"),
    "negative_directory_analysis": (False, "directory"),
    "negative_memory_analysis": (False, "domain_intent"),
    "positive_cache": (True, "domain_intent"),
    "negative_ascend_context": (False, "context"),
    "positive_directory_tool": (True, "directory"),
    "positive_memory_collection": (True, "domain_intent"),
    "negative_performance_only": (False, "intent_boundary"),
    "negative_ascend_optimization": (False, "intent_boundary"),
    "positive_existing_csv": (True, "context"),
    "negative_python_set": (False, "generic_term"),
    "negative_ci_pipeline": (False, "generic_term"),
    "negative_network": (False, "other_domain"),
    "negative_csv": (False, "generic_term"),
    "negative_web": (False, "other_domain"),
    "negative_compile": (False, "intent_boundary"),
    "negative_transcript": (False, "incidental_mention"),
    "negative_section": (False, "generic_term"),
    "negative_memory": (False, "generic_term"),
    "negative_context": (False, "context"),
}


class SkillTriggerEvalsTest(unittest.TestCase):
    """Validate evaluation data, not the host's skill-selection behavior."""

    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(
            EVALS_PATH.with_name("trigger-evals.json").read_text(encoding="utf-8")
        )

    def validate_document(self, document):
        self.assertIsInstance(document, dict)
        self.assertEqual(document.get("skill_name"), "tool-npu-compute")
        self.assertEqual(document.get("eval_mode"), "skill_trigger")
        cases = document.get("cases")
        self.assertIsInstance(cases, list)
        categories = {category for _, category in TRIGGER_SCENARIOS.values()}
        by_id = {}
        for case in cases:
            self.assertIsInstance(case, dict)
            for key in ("id", "prompt", "reason", "category"):
                self.assertIsInstance(case.get(key), str, key)
                self.assertTrue(case.get(key).strip(), key)
            self.assertIsInstance(case.get("context"), str)
            self.assertIs(type(case.get("should_trigger")), bool)
            self.assertIn(case.get("category"), categories)
            self.assertNotIn(case.get("id"), by_id, "duplicate ID")
            by_id[case.get("id")] = case
        self.assertTrue(TRIGGER_SCENARIOS.keys() <= by_id.keys(), "missing scenarios")
        for case_id, (expected, category) in TRIGGER_SCENARIOS.items():
            case = by_id.get(case_id)
            self.assertIsNotNone(case, case_id)
            self.assertIs(case.get("should_trigger"), expected, case_id)
            self.assertEqual(case.get("category"), category, case_id)
            if category == "context":
                self.assertTrue(case.get("context", "").strip(), case_id)

    def test_trigger_data_schema_and_required_coverage(self):
        self.validate_document(self.document)


if __name__ == "__main__":
    unittest.main()
