# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Tests locking the change_evaluator fix for the pr_10517-class false negative.

The generic evaluate_changes prompt must carry the rule that runtime input
dtype reinterpretation (e.g. UINT8 stored FP8 view, SetDataType/TensorWrapper)
changes the accepted-input contract even when the function signature is
unchanged. This rule is what flips the pr_10517 evaluation from 4/4 SKIP to
KEEP constraint + sub_function_hash (verified against the real PR 10517 diff,
2/2 stable with temperature=0).
"""

from src.change_detection.doc_part_change_detection.change_evaluator import (
    SYSTEM_PROMPT,
)


class TestSystemPromptInputContractRule:
    def test_rule_present_in_keep_true_section(self):
        head, sep, tail = SYSTEM_PROMPT.partition("keep: true 的情况")
        assert sep, "SYSTEM_PROMPT 缺少 keep: true 段落"
        assert "算子对输入的接受范围变化" in tail

    def test_rule_covers_signature_unchanged_case(self):
        assert "即使函数签名与原校验代码未变" in SYSTEM_PROMPT

    def test_rule_names_dtype_reinterpretation_example(self):
        assert "重解释" in SYSTEM_PROMPT
        assert "UINT8" in SYSTEM_PROMPT

    def test_rule_links_constraint_and_sub_function_hash(self):
        head, sep, tail = SYSTEM_PROMPT.partition("算子对输入的接受范围变化")
        assert sep
        assert "constraint" in tail and "sub_function_hash" in tail
