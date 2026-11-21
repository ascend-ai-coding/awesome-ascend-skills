# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""writer prompt 的共享代码命名铁律回归测试（规则 11）。

背景：昇腾算子仓库用同一目录共享代码实现多个 API 版本。writer 侧此前对共享文件
命名问题零覆盖（旧规则 10 仅覆盖公式场景的 kernel 中间量），存在三个缺口：
writer 自发把证据中的旧版本算子名/结构体名写进文档、kernel 内部变量名进入非公式
场景、共享文件中其他算子变体（RMSNorm 融合系）的参数被引入。与验证器侧
（consistency.py 命名错配豁免）和确定性巡检（param_ref_out_of_scope）构成三层防御
中的"前置预防"层。

测试以 mock LLM 捕获 writer 实际发出的 prompt，断言规则在位（机械防回归）。
"""

from unittest.mock import MagicMock

from src.generation_validation.generator.writer import TextWriter

_MOCK_LLM_RESPONSE = """
<analysis>
{"need_change": false, "reason": "无需修改", "evidence_conflict": false, "conflict_details": []}
</analysis>
<changes>[]</changes>
<document>## 约束说明

- 原文内容保持不变。
</document>
"""


def _capture_prompt() -> str:
    llm = MagicMock()
    llm.chat.return_value = _MOCK_LLM_RESPONSE
    template_loader = MagicMock()
    template_loader.get_label.return_value = "约束说明"
    template_loader.get_requirement.return_value = "按要求写作"
    template_loader.get_template_example.return_value = "## 模板"

    writer = TextWriter(llm, template_loader)
    writer.generate(
        original="## 约束说明\n\n- 原文。\n",
        evidences={"constraint": []},
        doc_part="constraint",
    )
    assert llm.chat.called, "writer 应调用一次 LLM"
    return llm.chat.call_args.args[0][1]["content"]


class TestSharedNamingRuleInWriterPrompt:
    def test_rule_present_with_real_case_example(self):
        prompt = _capture_prompt()
        assert "共享代码命名铁律" in prompt
        # 真实误判形态示例必须保留（IMPL_OP(V2) 实现 V3 API）
        assert "IMPL_OP(MoeDistributeCombineV2)" in prompt
        assert "MoeDistributeCombineV2TilingFunc" in prompt

    def test_doc_api_name_is_authoritative(self):
        """算子名权威性：禁止把证据旧版本名写入文档或据此改写文档算子名。"""
        prompt = _capture_prompt()
        assert "文档的算子名/API 版本是权威" in prompt
        assert (
            "禁止把证据中的旧版本算子名、结构体名、注册宏名、文件名写入文档" in prompt
        )

    def test_kernel_var_ban_generalizes_beyond_formula(self):
        """kernel 内部变量名禁令不限于公式场景（泛化规则 10）。"""
        prompt = _capture_prompt()
        assert "kernel 内部变量名不得作为文档符号" in prompt
        assert "不限于公式" in prompt
        assert "maxSharedGroupNum" in prompt

    def test_variant_isolation_rule(self):
        """共享文件中其他算子变体（RMSNorm 融合系）的内容禁止引入。"""
        prompt = _capture_prompt()
        assert "其他算子变体的内容禁止引入" in prompt
        assert "RMSNorm 融合系" in prompt
