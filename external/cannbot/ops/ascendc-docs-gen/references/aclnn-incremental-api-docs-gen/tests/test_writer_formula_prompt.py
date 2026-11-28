# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Tests locking the narrowed formula-update prompt fix.

Scope (per review): formula scenarios are rare, so the fix must stay narrow —
only the summary doc_part is touched, and only when the evidence actually
carries formula content (evidences_str contains "公式"). Global knobs
(temperature, rule-10 rewording) were deliberately reverted.
"""

import inspect


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature=None, **kwargs):
        self.calls.append({"messages": messages, "temperature": temperature})
        return (
            '<analysis>{"need_change": true, "reason": "r", "evidence_conflict": false}</analysis>'
            "\n<changes>[]</changes>\n<document>generated</document>"
        )


class _FakeTemplateLoader:
    def get_label(self, *args, **kwargs):
        return "功能说明"

    def get_requirement(self, *args, **kwargs):
        return "requirement"

    def get_template_example(self, *args, **kwargs):
        return "template"


def _capture(doc_part, change_reason=None, evidences=None):
    from src.generation_validation.generator.writer import TextWriter

    llm = _FakeLLM()
    writer = TextWriter(llm_client=llm, template_loader=_FakeTemplateLoader())
    writer.generate(
        original="## 功能说明\n原文",
        evidences=evidences,
        style_summary="summary",
        style_extraction="extraction",
        doc_part=doc_part,
        change_reason=change_reason,
    )
    assert llm.calls, "LLM 未被调用"
    return llm.calls[0]


_FORMULA_EVIDENCES = {"计算公式": [{"content": "bias 在反量化之后累加"}]}


class TestFormulaCarveOutIsNarrow:
    def test_summary_with_formula_evidence_gets_carve_out(self):
        call = _capture(
            "summary",
            change_reason={"keep": True, "reason": "r"},
            evidences=_FORMULA_EVIDENCES,
        )
        prompt = call["messages"][1]["content"]
        assert "计算公式注意" in prompt
        assert "公式必须按证据同步更新" in prompt

    def test_summary_without_formula_evidence_no_carve_out(self):
        call = _capture(
            "summary",
            change_reason={"keep": True, "reason": "r"},
            evidences={"算子接口功能": [{"content": "功能描述"}]},
        )
        assert "计算公式注意" not in call["messages"][1]["content"]

    def test_non_summary_no_carve_out_even_with_formula_evidence(self):
        call = _capture(
            "constraint",
            change_reason={"keep": True, "reason": "r"},
            evidences=_FORMULA_EVIDENCES,
        )
        assert "计算公式注意" not in call["messages"][1]["content"]

    def test_no_change_reason_no_carve_out(self):
        call = _capture("summary", change_reason=None, evidences=_FORMULA_EVIDENCES)
        assert "计算公式注意" not in call["messages"][1]["content"]


class TestGlobalKnobsUntouched:
    def test_generation_keeps_default_temperature(self):
        call = _capture(
            "summary",
            change_reason={"keep": True, "reason": "r"},
            evidences=_FORMULA_EVIDENCES,
        )
        assert call["temperature"] is None

    def test_rule10_keeps_original_wording(self):
        call = _capture("constraint")
        prompt = call["messages"][1]["content"]
        assert "公式类约束的最小增量修改" in prompt
        assert "不约束内容更新" not in prompt


class TestUpstreamPromptRules:
    def test_evaluator_reason_must_name_formula_changes(self):
        from src.change_detection.doc_part_change_detection.change_evaluator import (
            SYSTEM_PROMPT,
        )

        assert "评估 reason 必须显式点名计算公式相关变化" in SYSTEM_PROMPT
        assert "reason 未提公式时生成环节会默认保留原文档公式" in SYSTEM_PROMPT

    def test_extractor_summary_formula_task_extracts_deltas(self):
        from src.generation_validation.evidence import extractor

        source = inspect.getsource(extractor)
        assert "公式差异项" in source
        assert "计算逻辑未变时不得臆造差异" in source
