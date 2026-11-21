# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""一致性检查器 prompt 的版本/命名错配豁免规则回归测试（P1 修复）。

背景：昇腾算子仓库用同一目录共享代码实现多个 API 版本（V2 命名文件实现 V3 API）。
旧 prompt 会把"证据文件名/结构体/注册宏的版本号与文档 API 版本不一致"误判为
"算子名称矛盾"类事实错误——该类 issue writer 无法修复，且曾诱导 writer 把旧版本
算子名写进文档（pr_5992 dispatch_v3 attempt2 实锤）。

修复（consistency.py prompt 两处）：
1. 核心裁决原则新增"版本与命名错配豁免"，并附收紧句（豁免仅限名称层面，
   语义写反判定标准不变）防止 halo 外推；
2. "代码块精确校验"的"任何不一致均视为事实错误"为命名错配开出例外。

测试以 mock LLM 捕获实际发出的 prompt，断言豁免规则在位（机械防回归；
行为有效性已由真实 LLM A/B 实验验证：命名误报 1/3→0/3，BS 写反检出 2/3→3/3）。
"""

from unittest.mock import MagicMock

from src.generation_validation.validator.consistency import ConsistencyChecker

_MOCK_LLM_RESULT = {
    "batch_scores": {
        "factual_accuracy": 1.0,
        "formula_correctness": 1.0,
        "numerical_accuracy": 1.0,
        "description_consistency": 1.0,
        "logical_coherence": 1.0,
    },
    "issues": [],
    "cross_source_findings": [],
}


def _capture_prompt() -> str:
    llm = MagicMock()
    llm.chat_with_json.return_value = _MOCK_LLM_RESULT
    checker = ConsistencyChecker(llm)
    checker.check(
        "## 约束说明\n",
        [{"evidence_id": "ev_001", "target": [["constraint"]], "content": "int x;"}],
        doc_part="constraint",
    )
    assert llm.chat_with_json.called, "应至少调用一次 LLM"
    return llm.chat_with_json.call_args.args[0][0]["content"]


class TestNamingMismatchExemptionInPrompt:
    def test_exemption_rule_present(self):
        prompt = _capture_prompt()
        assert "版本与命名错配豁免" in prompt
        assert "共享实现常态" in prompt
        # 具体示例须保留（真实误判形态：IMPL_OP(V2) 实现 V3 API）
        assert "IMPL_OP(MoeDistributeCombineV2)" in prompt
        # kernel 内部变量名 vs 文档参数名示例
        assert "epSendCountsOut" in prompt and "epRecvCountsOut" in prompt
        # 禁止诱导 writer 改名（防 attempt2 式文档污染）
        assert "禁止建议将文档中的算子名/参数名改为证据中的旧版本名" in prompt

    def test_exemption_scoped_to_names_only(self):
        """收紧句在位：豁免不得外推到语义写反（BS 上限对调类）。"""
        prompt = _capture_prompt()
        assert "该豁免仅适用于名称层面" in prompt
        assert "不得因该豁免放宽" in prompt

    def test_code_block_check_carves_out_exemption(self):
        """代码块精确校验的"任何不一致均视为事实错误"须为命名错配开例外。"""
        prompt = _capture_prompt()
        assert '但"版本与命名错配"除外' in prompt
