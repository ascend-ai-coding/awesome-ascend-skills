# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""确定性巡检 deterministic_checks 的纯单元测试（无网络无 LLM，tmp_path 构造布局）。

覆盖四项已确诊缺陷的修复：
1. param_ref_out_of_scope 忽略 dtype/format/dim 等非参数反引号词（constraint 分支）；
2. 真外部参数（不在任何允许集）仍触发拦截；
3. build_sig_corpus 对 evidences_changed_only 回补兄弟全量证据（签名形参放行）；
4. format_assert_introduced 方向感知（forbid/require 一致放行，读反/无证据拦截）；
5. resolve_orig_doc_full 经证据路径回溯 <算子输出目录>/doc/<doc_name>；
6. 回归：md_link_lost / param_row_out_of_scope 仍工作。
"""

import json
from pathlib import Path

from src.generation_validation.pipeline.deterministic_checks import (
    build_sig_corpus,
    collect_issues,
    resolve_orig_doc_full,
)


def _issue_ids(issues: list[dict]) -> list[str]:
    return [i["id"] for i in issues]


class TestParamRefIgnoresNonParamTokens:
    """1. dtype/format/dim 等反引号词不再触发 param_ref_out_of_scope（constraint 分支）。"""

    def test_dtype_format_dim_words_not_flagged(self):
        orig_section = "## 约束说明\n\n- `x` 的取值应满足公式约束。\n"
        generated = (
            "## 约束说明\n\n"
            "- `x` 数据类型支持 `bfloat16`、`int32`、`bool`、`float16`。\n"
            "- `x` 数据格式不能为 `fractal_nz`，只支持 `nd`。\n"
            "- `x` 的 `dim0` 与 `dim1` 需相等。\n"
            "- `shape` 满足广播约束，`dtype` 由输入决定。\n"
        )
        issues = collect_issues(orig_section, generated, "constraint", "", "")
        assert "param_ref_out_of_scope" not in _issue_ids(issues), (
            f"dtype/format/dim 非参数词不应触发 param_ref: {_issue_ids(issues)}"
        )

    def test_param_row_branch_unaffected_by_ignore_set(self):
        """忽略集只作用于 constraint 分支：sub_function_hash 表行首列词仍按原逻辑比对。"""
        orig_section = (
            "## aclnnTestGetWorkspaceSize\n\n| a | 说明 |\n|---|---|\n| a | int32 |\n"
        )
        generated = (
            "## aclnnTestGetWorkspaceSize\n\n| a | 说明 |\n|---|---|\n| int32 | abc |\n"
        )
        issues = collect_issues(orig_section, generated, "sub_function_hash", "", "")
        # `int32` 是表格首列内容（不是反引号词），param_row 分支应照常命中
        assert "param_row_out_of_scope" in _issue_ids(issues)


class TestParamRefStillFlagsExternalParam:
    """2. 真外部参数（不在允许集、不在忽略集）仍触发拦截。"""

    def test_external_param_flagged(self):
        orig_section = "## 约束说明\n\n- `x` 的取值应满足公式约束。\n"
        generated = "## 约束说明\n\n- `rmsNormAlpha` 的取值应大于 0。\n"
        issues = collect_issues(orig_section, generated, "constraint", "", "")
        ids = _issue_ids(issues)
        assert "param_ref_out_of_scope" in ids, "真外部参数 rmsNormAlpha 应触发拦截"
        flagged = [i for i in issues if i["id"] == "param_ref_out_of_scope"]
        assert any("`rmsNormAlpha`" in i["reason"] for i in flagged)


class TestBuildSigCorpusChangedOnlyBackfill:
    """3. changed_only 证据 → 回补兄弟全量证据，签名形参重新可放行。"""

    @staticmethod
    def _make_layout(tmp_path: Path) -> Path:
        """构造 ver/re_detection/evidences（全量，含签名）+ evidences_changed_only（无签名）。"""
        full_dir = tmp_path / "ver" / "re_detection" / "evidences"
        full_dir.mkdir(parents=True)
        (full_dir / "constraint.json").write_text(
            json.dumps(
                {"anchor_tuple": [{"content": "aclnnOpName(const aclTensor* b)"}]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        changed_dir = tmp_path / "ver" / "re_detection" / "evidences_changed_only"
        changed_dir.mkdir(parents=True)
        (changed_dir / "constraint.json").write_text(
            json.dumps(
                {"anchor_tuple": [{"content": "// 无签名"}]}, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        return changed_dir / "constraint.json"

    def test_backfill_lets_signature_param_pass(self, tmp_path):
        changed_path = self._make_layout(tmp_path)
        sig_corpus = build_sig_corpus(["## 约束说明\n"], {}, {}, changed_path)
        orig_section = "## 约束说明\n\n- `x` 的取值应满足公式约束。\n"
        generated = "## 约束说明\n\n- `b` 为必填输入。\n"
        issues = collect_issues(orig_section, generated, "constraint", sig_corpus, "")
        assert "param_ref_out_of_scope" not in _issue_ids(issues), (
            "回补全量证据后，签名形参 b 应被放行"
        )

    def test_without_evidence_path_param_flagged(self, tmp_path):
        """不给 evidence_path 时无回补，b 不在允许集 → 触发（与旧行为一致）。"""
        sig_corpus = build_sig_corpus(["## 约束说明\n"], {}, {})
        orig_section = "## 约束说明\n\n- `x` 的取值应满足公式约束。\n"
        generated = "## 约束说明\n\n- `b` 为必填输入。\n"
        issues = collect_issues(orig_section, generated, "constraint", sig_corpus, "")
        assert "param_ref_out_of_scope" in _issue_ids(issues)

    def test_full_evidence_dir_not_appended(self, tmp_path):
        """evidence_path 指向全量 evidences/ 时不做回补（语料保持基础构造）。"""
        full_path = tmp_path / "ver" / "re_detection" / "evidences" / "constraint.json"
        full_path.parent.mkdir(parents=True)
        full_path.write_text(
            json.dumps(
                {"anchor_tuple": [{"content": "aclnnOpName(const aclTensor* b)"}]}
            ),
            encoding="utf-8",
        )
        # 直接传全量路径：基础语料本身不含该文件内容（raw_input/extracted 为空）
        corpus = build_sig_corpus(["## 约束说明\n"], {}, {}, full_path)
        orig_section = "## 约束说明\n\n- `x` 的取值应满足公式约束。\n"
        generated = "## 约束说明\n\n- `b` 为必填输入。\n"
        issues = collect_issues(orig_section, generated, "constraint", corpus, "")
        assert "param_ref_out_of_scope" in _issue_ids(issues), (
            "evidence_path 不是 changed_only 目录时不应追加文件内容"
        )


class TestFractalNzDirection:
    """4. format_assert_introduced 方向感知。"""

    ORIG_SECTION = "## 约束说明\n\n- `x` 非零。\n"
    FORBID_CORPUS = 'OP_TILING_CHECK(format == FORMAT_FRACTAL_NZ, "nz not supported");'
    # 实际证据中常带 ge:: 命名空间前缀（如 if (xDescFormat == ge::FORMAT_FRACTAL_NZ)）
    GE_FORBID_CORPUS = 'if (xDescFormat == ge::FORMAT_FRACTAL_NZ) { OP_LOGE(nodeName, "nz"); return false; }'

    def test_direction_consistent_not_flagged(self):
        generated = "## 约束说明\n\n- `x` 的数据格式不能为`FRACTAL_NZ`。\n"
        issues = collect_issues(
            self.ORIG_SECTION, generated, "constraint", self.FORBID_CORPUS, ""
        )
        assert "format_assert_introduced" not in _issue_ids(issues), (
            "writer 写'不能为 FRACTAL_NZ'与证据 '== FORMAT_FRACTAL_NZ'（报错条件=禁止）方向一致，应放行"
        )

    def test_ge_namespace_prefix_direction_consistent_not_flagged(self):
        """证据写法带 ge:: 命名空间前缀（真实批次形态）时方向判定同样生效。"""
        generated = "## 约束说明\n\n- `x` 的数据格式：不支持FRACTAL_NZ。\n"
        issues = collect_issues(
            self.ORIG_SECTION, generated, "constraint", self.GE_FORBID_CORPUS, ""
        )
        assert "format_assert_introduced" not in _issue_ids(issues), (
            "证据 '== ge::FORMAT_FRACTAL_NZ'（报错条件=禁止）+ 生成'不支持FRACTAL_NZ' 方向一致，应放行"
        )

    def test_ge_namespace_prefix_direction_reversed_flagged(self):
        generated = "## 约束说明\n\n- `x` 的格式必须为`FRACTAL_NZ`。\n"
        issues = collect_issues(
            self.ORIG_SECTION, generated, "constraint", self.GE_FORBID_CORPUS, ""
        )
        assert "format_assert_introduced" in _issue_ids(issues), (
            "证据 '== ge::FORMAT_FRACTAL_NZ'（禁止）+ 生成'必须为FRACTAL_NZ'（要求）方向读反，应拦截"
        )

    def test_direction_reversed_flagged(self):
        generated = "## 约束说明\n\n- `x` 的格式必须为`FRACTAL_NZ`。\n"
        issues = collect_issues(
            self.ORIG_SECTION, generated, "constraint", self.FORBID_CORPUS, ""
        )
        assert "format_assert_introduced" in _issue_ids(issues), "方向读反应触发拦截"

    def test_no_nz_evidence_flagged(self):
        """证据无 NZ 信息 → 保留新颖性守卫，仍然拦截。"""
        generated = "## 约束说明\n\n- `x` 的数据格式不能为`FRACTAL_NZ`。\n"
        issues = collect_issues(
            self.ORIG_SECTION, generated, "constraint", "int32 someKernel(...);", ""
        )
        assert "format_assert_introduced" in _issue_ids(issues)

    def test_direction_unknown_wording_flagged(self):
        """生成措辞无方向词 → 方向未知，仍拦截。"""
        generated = "## 约束说明\n\n- `x` 的格式为`FRACTAL_NZ`。\n"
        issues = collect_issues(
            self.ORIG_SECTION, generated, "constraint", self.FORBID_CORPUS, ""
        )
        assert "format_assert_introduced" in _issue_ids(issues)

    def test_orig_section_mention_not_flagged(self):
        """原文已出现 FRACTAL_NZ 时不触发（原有前置条件回归）。"""
        orig = "## 约束说明\n\n- 支持 FRACTAL_NZ 排布。\n"
        generated = "## 约束说明\n\n- 支持 FRACTAL_NZ 排布。\n"
        issues = collect_issues(orig, generated, "constraint", "", "")
        assert "format_assert_introduced" not in _issue_ids(issues)


class TestResolveOrigDocFull:
    """5. resolve_orig_doc_full 布局回溯与旧行为兼容。"""

    @staticmethod
    def _make_layout(tmp_path: Path) -> tuple[Path, Path]:
        """构造 <op>/doc/foo.md 与证据 <op>/<ver>/re_detection/evidences/constraint.json。"""
        op_dir = tmp_path / "op"
        doc_file = op_dir / "doc" / "foo.md"
        doc_file.parent.mkdir(parents=True)
        doc_file.write_text("# foo 原文档全文", encoding="utf-8")
        evidence = op_dir / "ver" / "re_detection" / "evidences" / "constraint.json"
        evidence.parent.mkdir(parents=True)
        evidence.write_text("{}", encoding="utf-8")
        return doc_file, evidence

    def test_relative_doc_path_resolved_via_evidence(self, tmp_path):
        doc_file, evidence = self._make_layout(tmp_path)
        # doc_path="."（流水线实况）：CWD 下 foo.md 不存在，经证据祖先回溯读到
        content = resolve_orig_doc_full(".", "foo.md", evidence)
        assert content == "# foo 原文档全文"

    def test_absolute_doc_dir_backward_compat(self, tmp_path):
        doc_file, _ = self._make_layout(tmp_path)
        # 绝对目录路径（旧行为）：目录拼 doc_name 直读
        content = resolve_orig_doc_full(str(doc_file.parent), "foo.md", None)
        assert content == "# foo 原文档全文"

    def test_absolute_file_path_backward_compat(self, tmp_path):
        doc_file, _ = self._make_layout(tmp_path)
        content = resolve_orig_doc_full(str(doc_file), "foo.md", None)
        assert content == "# foo 原文档全文"

    def test_missing_everywhere_returns_empty(self, tmp_path):
        _, evidence = self._make_layout(tmp_path)
        assert resolve_orig_doc_full(".", "bar.md", evidence) == ""
        assert resolve_orig_doc_full(".", "bar.md", None) == ""

    def test_accepts_path_and_str_evidence(self, tmp_path):
        doc_file, evidence = self._make_layout(tmp_path)
        assert resolve_orig_doc_full(".", "foo.md", str(evidence)) == "# foo 原文档全文"
        assert (
            resolve_orig_doc_full(".", "foo.md", Path(evidence)) == "# foo 原文档全文"
        )


class TestRegressionExistingChecks:
    """6. 回归：md_link_lost 与 param_row_out_of_scope 仍工作。"""

    def test_md_link_lost_still_works(self):
        orig_section = "## 约束说明\n\n- 详见 [a](b.md)。\n"
        generated = "## 约束说明\n\n- 详见 a。\n"
        issues = collect_issues(orig_section, generated, "constraint", "", "")
        assert "md_link_lost" in _issue_ids(issues)

    def test_md_link_kept_not_flagged(self):
        orig_section = "## 约束说明\n\n- 详见 [a](b.md)。\n"
        generated = "## 约束说明\n\n- 详见 [a](b.md) 原文。\n"
        issues = collect_issues(orig_section, generated, "constraint", "", "")
        assert "md_link_lost" not in _issue_ids(issues)

    def test_param_row_out_of_scope_still_works(self):
        orig_section = (
            "## aclnnTestGetWorkspaceSize\n\n"
            "| a | 说明 |\n|---|---|\n| a | int32 输入。 |\n"
        )
        generated = (
            "## aclnnTestGetWorkspaceSize\n\n"
            "| a | 说明 |\n|---|---|\n| a | int32 输入。 |\n| rogueParam | 越界行。 |\n"
        )
        issues = collect_issues(orig_section, generated, "sub_function_hash", "", "")
        assert "param_row_out_of_scope" in _issue_ids(issues)
        flagged = [i for i in issues if i["id"] == "param_row_out_of_scope"]
        assert any("`rogueParam`" in i["reason"] for i in flagged)
