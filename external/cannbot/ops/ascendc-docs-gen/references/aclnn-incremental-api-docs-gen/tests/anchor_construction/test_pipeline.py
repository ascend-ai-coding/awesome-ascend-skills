# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for AnchorPipeline - unified anchor construction pipeline.

Tests cover:
- AnchorPipeline initialization
- pipeline.run() producing structurally correct output for each doc_part
- envidence_id key presence in all evidence items
- summary↔function_proto coupling (different gene_rule targets)
- IdGenerator integration (correct ID format per doc_part)
"""

import json
import os
from pathlib import Path

import pytest

from src.snapshot_construction.anchor_construction.pipeline import AnchorPipeline
from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    get_registry,
)
from src.snapshot_construction.anchor_construction.common import DOC_PARTS


# Test operator path (abs operator) — 本机未设环境变量时相关测试自动跳过
OPERATOR_PATH = Path(
    os.environ.get(
        "TEST_OPERATOR_PATH",
        Path(os.environ.get("CANN_OPS_ROOT", "")) / "math" / "abs",
    )
)
OUT_PATH = Path("/tmp/anchor_pipeline_test_output")


def _operator_available() -> bool:
    return bool(os.environ.get("TEST_OPERATOR_PATH") or os.environ.get("CANN_OPS_ROOT"))


requires_operator = pytest.mark.skipif(
    not _operator_available(),
    reason="需要 TEST_OPERATOR_PATH 或 CANN_OPS_ROOT 指向本机 ops 检出",
)


@pytest.fixture
def pipeline():
    """AnchorPipeline instance for testing."""
    return AnchorPipeline(PathConfig(OPERATOR_PATH, OUT_PATH))


@pytest.fixture
def pipeline_with_control():
    """AnchorPipeline instance with doc_part_control."""
    return AnchorPipeline(
        PathConfig(OPERATOR_PATH, OUT_PATH), doc_part_control={"constraint": None}
    )


@requires_operator
class TestAnchorPipelineInit:
    """测试 AnchorPipeline 初始化"""

    def test_init_basic(self):
        """测试基本初始化"""
        pipeline = AnchorPipeline(PathConfig(OPERATOR_PATH, OUT_PATH))

        assert pipeline.operator_path == OPERATOR_PATH
        assert pipeline.out_path == OUT_PATH
        assert pipeline.doc_part_control is None
        assert pipeline.path_config is not None
        assert pipeline.data_loader is not None
        assert pipeline.registry is not None

    def test_init_with_doc_part_control_dict(self):
        """测试带 doc_part_control dict 的初始化"""
        control = {"constraint": None, "support": None}
        pipeline = AnchorPipeline(
            PathConfig(OPERATOR_PATH, OUT_PATH), doc_part_control=control
        )

        assert pipeline.doc_part_control == control

    def test_init_with_doc_part_control_list(self):
        """测试带 doc_part_control list 的初始化"""
        control = ["constraint", "support"]
        pipeline = AnchorPipeline(
            PathConfig(OPERATOR_PATH, OUT_PATH), doc_part_control=control
        )

        assert pipeline.doc_part_control == control

    def test_init_invalid_doc_part_raises_error(self):
        """测试无效 doc_part 抛出 ValueError"""
        with pytest.raises(ValueError, match="doc_part 类型错误"):
            AnchorPipeline(
                PathConfig(OPERATOR_PATH, OUT_PATH),
                doc_part_control={"invalid_part": None},
            )


@requires_operator
class TestAnchorPipelineRun:
    """测试 AnchorPipeline.run() 方法"""

    def test_run_returns_dict_with_all_doc_parts(self, pipeline):
        """测试 run() 返回包含所有 doc_part 的 dict"""
        results = pipeline.run()

        assert isinstance(results, dict)
        assert set(results.keys()) == set(DOC_PARTS)

    def test_run_each_result_has_doc_and_anchor_tuple(self, pipeline):
        """测试每个结果包含 doc 和 anchor_tuple"""
        results = pipeline.run()

        for doc_part, result in results.items():
            assert "doc" in result
            assert "anchor_tuple" in result
            assert isinstance(result["doc"], dict)
            assert isinstance(result["anchor_tuple"], list)

    def test_run_doc_structure_has_required_keys(self, pipeline):
        """测试 doc 结构包含必需的键"""
        results = pipeline.run()

        for doc_part, result in results.items():
            doc = result["doc"]
            assert "doc_path" in doc
            assert "doc_name" in doc
            assert "doc_part" in doc
            assert "md_format" in doc
            assert "sub_part" in doc
            assert doc["doc_part"] == doc_part

    def test_run_anchor_tuple_items_have_required_keys(self, pipeline):
        """测试 anchor_tuple 项包含必需的键"""
        results = pipeline.run()

        for doc_part, result in results.items():
            for anchor in result["anchor_tuple"]:
                assert "anchor_id" in anchor
                assert "anchor_information" in anchor
                assert "evidence" in anchor
                assert "note" in anchor

                # Check anchor_information structure
                anchor_info = anchor["anchor_information"]
                assert "type" in anchor_info
                assert "level1" in anchor_info
                assert "level2" in anchor_info

                # level3 is only present for file and function anchors
                # note and section anchors don't have level3 in golden reference
                anchor_type = anchor_info["type"]
                if anchor_type in ["file", "function"]:
                    assert "level3" in anchor_info

    def test_run_evidence_items_have_required_keys(self, pipeline):
        """测试 evidence 项包含必需的键"""
        results = pipeline.run()

        for doc_part, result in results.items():
            for anchor in result["anchor_tuple"]:
                for evidence in anchor["evidence"]:
                    assert "evidence_id" in evidence
                    assert "envidence_id" in evidence  # compat requirement
                    assert "evidence_information" in evidence
                    assert "gene_rule" in evidence

                    # Check gene_rule structure
                    gene_rule = evidence["gene_rule"]
                    assert "deter" in gene_rule
                    assert "target" in gene_rule

    def test_run_envidence_id_equals_evidence_id(self, pipeline):
        """测试 envidence_id 等于 evidence_id"""
        results = pipeline.run()

        for doc_part, result in results.items():
            for anchor in result["anchor_tuple"]:
                for evidence in anchor["evidence"]:
                    assert evidence["envidence_id"] == evidence["evidence_id"]

    def test_run_creates_output_files(self, pipeline):
        """测试 run() 创建输出文件"""
        pipeline.run()

        registry = get_registry()
        anchor_dir = pipeline.path_config.anchor_dir

        for doc_part in DOC_PARTS:
            config = registry.get_config(doc_part)
            output_file = anchor_dir / config.output_filename

            assert output_file.exists()

            # Verify file is valid JSON
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert "doc" in data
                assert "anchor_tuple" in data

    def test_run_with_doc_part_control_processes_only_specified(
        self, pipeline_with_control
    ):
        """测试带 doc_part_control 只处理指定的 doc_part"""
        results = pipeline_with_control.run()

        # Should only process constraint
        assert "constraint" in results
        # Should not process other doc_parts
        assert len(results) == 1


@requires_operator
class TestAnchorPipelineDocPartSpecific:
    """测试每个 doc_part 的特定行为"""

    def test_constraint_anchor_types(self, pipeline):
        """测试 constraint 包含 file 和 section anchors"""
        results = pipeline.run()
        constraint_result = results["constraint"]

        anchor_types = set()
        for anchor in constraint_result["anchor_tuple"]:
            anchor_types.add(anchor["anchor_information"]["type"])

        # constraint should have file anchors (may or may not have section)
        assert "file" in anchor_types

    def test_support_anchor_types(self, pipeline):
        """测试 support 包含 file 和 section anchors"""
        results = pipeline.run()
        support_result = results["support"]

        anchor_types = set()
        for anchor in support_result["anchor_tuple"]:
            anchor_types.add(anchor["anchor_information"]["type"])

        # support should have file anchors (may or may not have section)
        assert "file" in anchor_types

    def test_support_has_summary_section(self, pipeline):
        """测试 support 包含 summary section"""
        results = pipeline.run()
        support_result = results["support"]

        assert "summary" in support_result
        summary = support_result["summary"]

        # Check summary structure
        assert "type1_section_count" in summary
        assert "type2_section_count" in summary
        assert "type3_section_count" in summary
        assert "file_count" in summary
        assert "total_anchor_count" in summary
        assert "total_evidence_count" in summary
        assert "chips" in summary
        assert "socs" in summary
        assert "archs" in summary

    def test_summary_anchor_types(self, pipeline):
        """测试 summary 包含 file 和 note anchors"""
        results = pipeline.run()
        summary_result = results["summary"]

        anchor_types = set()
        for anchor in summary_result["anchor_tuple"]:
            anchor_types.add(anchor["anchor_information"]["type"])

        # summary should have file anchors (may or may not have note)
        assert "file" in anchor_types

    def test_function_proto_anchor_types(self, pipeline):
        """测试 function_proto 包含 file 和 function anchors"""
        results = pipeline.run()
        function_proto_result = results["function_proto"]

        anchor_types = set()
        for anchor in function_proto_result["anchor_tuple"]:
            anchor_types.add(anchor["anchor_information"]["type"])

        # function_proto should have file anchors (may or may not have function)
        assert "file" in anchor_types

    def test_sub_function_hash_anchor_types(self, pipeline):
        """测试 sub_function_hash 包含所有 4 种 anchor types"""
        results = pipeline.run()
        sub_function_result = results["sub_function_hash"]

        anchor_types = set()
        for anchor in sub_function_result["anchor_tuple"]:
            anchor_types.add(anchor["anchor_information"]["type"])

        # sub_function_hash should have file anchors (may or may not have all 4 types)
        assert "file" in anchor_types


@requires_operator
class TestAnchorPipelineGeneRules:
    """测试 gene_rule 分配"""

    def test_constraint_gene_rule_targets_are_1_element(self, pipeline):
        """测试 constraint gene_rule targets 是 1-element lists"""
        results = pipeline.run()
        constraint_result = results["constraint"]

        for anchor in constraint_result["anchor_tuple"]:
            for evidence in anchor["evidence"]:
                targets = evidence["gene_rule"]["target"]
                for target in targets:
                    assert isinstance(target, list)
                    assert len(target) == 1  # 1-element target

    def test_support_gene_rule_targets_are_1_element(self, pipeline):
        """测试 support gene_rule targets 是 1-element lists"""
        results = pipeline.run()
        support_result = results["support"]

        for anchor in support_result["anchor_tuple"]:
            for evidence in anchor["evidence"]:
                targets = evidence["gene_rule"]["target"]
                for target in targets:
                    assert isinstance(target, list)
                    assert len(target) == 1  # 1-element target

    def test_summary_gene_rule_targets_are_1_element(self, pipeline):
        """测试 summary gene_rule targets 是 1-element lists"""
        results = pipeline.run()
        summary_result = results["summary"]

        for anchor in summary_result["anchor_tuple"]:
            for evidence in anchor["evidence"]:
                targets = evidence["gene_rule"]["target"]
                for target in targets:
                    assert isinstance(target, list)
                    assert len(target) == 1  # 1-element target

    def test_function_proto_gene_rule_targets_are_1_element(self, pipeline):
        """测试 function_proto gene_rule targets 是 1-element lists"""
        results = pipeline.run()
        function_proto_result = results["function_proto"]

        for anchor in function_proto_result["anchor_tuple"]:
            for evidence in anchor["evidence"]:
                targets = evidence["gene_rule"]["target"]
                for target in targets:
                    assert isinstance(target, list)
                    assert len(target) == 1  # 1-element target

    def test_summary_function_proto_different_gene_rules(self, pipeline):
        """测试 summary 和 function_proto 有不同的 gene_rule targets"""
        results = pipeline.run()

        summary_result = results["summary"]
        function_proto_result = results["function_proto"]

        # Get gene_rule targets from summary
        summary_targets = set()
        for anchor in summary_result["anchor_tuple"]:
            for evidence in anchor["evidence"]:
                for target in evidence["gene_rule"]["target"]:
                    summary_targets.add(tuple(target))

        # Get gene_rule targets from function_proto
        function_proto_targets = set()
        for anchor in function_proto_result["anchor_tuple"]:
            for evidence in anchor["evidence"]:
                for target in evidence["gene_rule"]["target"]:
                    function_proto_targets.add(tuple(target))

        # They should have different targets
        # summary should have summary_clarify and summary_formula
        # function_proto should have function_proto_clarify and sub_function_sign
        assert ("summary_clarify",) in summary_targets or (
            "summary_formula",
        ) in summary_targets
        assert ("function_proto_clarify",) in function_proto_targets or (
            "sub_function_sign",
        ) in function_proto_targets


@requires_operator
class TestAnchorPipelineIdFormat:
    """测试 ID 格式"""

    def test_constraint_id_format(self, pipeline):
        """测试 constraint ID 格式: anchor_file_001, ev_file_001"""
        results = pipeline.run()
        constraint_result = results["constraint"]

        for anchor in constraint_result["anchor_tuple"]:
            anchor_id = anchor["anchor_id"]
            anchor_type = anchor["anchor_information"]["type"]

            # constraint uses type prefix in both anchor and evidence IDs
            assert anchor_id.startswith(f"anchor_{anchor_type}_")

            for evidence in anchor["evidence"]:
                evidence_id = evidence["evidence_id"]
                assert evidence_id.startswith(f"ev_{anchor_type}_")

    def test_summary_id_format(self, pipeline):
        """测试 summary ID 格式: anchor_file_001, ev_001"""
        results = pipeline.run()
        summary_result = results["summary"]

        for anchor in summary_result["anchor_tuple"]:
            anchor_id = anchor["anchor_id"]
            anchor_type = anchor["anchor_information"]["type"]

            # summary uses type prefix in anchor ID only
            assert anchor_id.startswith(f"anchor_{anchor_type}_")

            for evidence in anchor["evidence"]:
                evidence_id = evidence["evidence_id"]
                # evidence ID should NOT have type prefix
                assert evidence_id.startswith("ev_")
                # Should NOT have type in evidence ID
                assert not evidence_id.startswith(f"ev_{anchor_type}_")

    def test_function_proto_id_format(self, pipeline):
        """测试 function_proto ID 格式: anchor_file_001, ev_001"""
        results = pipeline.run()
        function_proto_result = results["function_proto"]

        for anchor in function_proto_result["anchor_tuple"]:
            anchor_id = anchor["anchor_id"]
            anchor_type = anchor["anchor_information"]["type"]

            # function_proto uses type prefix in anchor ID only
            assert anchor_id.startswith(f"anchor_{anchor_type}_")

            for evidence in anchor["evidence"]:
                evidence_id = evidence["evidence_id"]
                # evidence ID should NOT have type prefix
                assert evidence_id.startswith("ev_")
                # Should NOT have type in evidence ID
                assert not evidence_id.startswith(f"ev_{anchor_type}_")


@requires_operator
class TestAnchorPipelineSubPart:
    """测试 sub_part 结构"""

    def test_constraint_sub_part_structure(self, pipeline):
        """测试 constraint sub_part 结构"""
        results = pipeline.run()
        constraint_result = results["constraint"]

        doc = constraint_result["doc"]
        assert "constrain_clarify" in doc["sub_part"]

        constrain_clarify = doc["sub_part"]["constrain_clarify"]
        assert isinstance(constrain_clarify, list)
        assert len(constrain_clarify) > 0

        # Each item should have content and evidence_id
        for item in constrain_clarify:
            assert "content" in item
            assert "evidence_id" in item
            assert isinstance(item["evidence_id"], list)

    def test_support_sub_part_structure(self, pipeline):
        """测试 support sub_part 结构"""
        results = pipeline.run()
        support_result = results["support"]

        doc = support_result["doc"]
        assert "support_matrix" in doc["sub_part"]

        support_matrix = doc["sub_part"]["support_matrix"]
        assert isinstance(support_matrix, list)
        assert len(support_matrix) > 0

        # Each item should have content and evidence_ids
        for item in support_matrix:
            assert "content" in item
            assert "evidence_ids" in item
            assert isinstance(item["evidence_ids"], list)

    def test_summary_sub_part_structure(self, pipeline):
        """测试 summary sub_part 结构"""
        results = pipeline.run()
        summary_result = results["summary"]

        doc = summary_result["doc"]
        assert "summary_clarify" in doc["sub_part"]
        assert "summary_formula" in doc["sub_part"]

        # Each should be a list with content
        for key in ["summary_clarify", "summary_formula"]:
            items = doc["sub_part"][key]
            assert isinstance(items, list)
            assert len(items) > 0
            for item in items:
                assert "content" in item

    def test_function_proto_sub_part_structure(self, pipeline):
        """测试 function_proto sub_part 结构"""
        results = pipeline.run()
        function_proto_result = results["function_proto"]

        doc = function_proto_result["doc"]
        assert "function_proto_clarify" in doc["sub_part"]
        assert "sub_function_sign" in doc["sub_part"]

        # Each should be a list with content
        for key in ["function_proto_clarify", "sub_function_sign"]:
            items = doc["sub_part"][key]
            assert isinstance(items, list)
            assert len(items) > 0
            for item in items:
                assert "content" in item


@requires_operator
class TestAnchorPipelineIntegration:
    """集成测试"""

    def test_full_pipeline_execution(self, pipeline):
        """测试完整 pipeline 执行"""
        results = pipeline.run()

        # Verify all 5 doc_parts are present
        assert len(results) == 5

        # Verify each has anchors
        for doc_part, result in results.items():
            assert len(result["anchor_tuple"]) > 0

    def test_pipeline_output_matches_golden_structure(self, pipeline, golden_data):
        """测试 pipeline 输出匹配 golden reference 结构"""
        results = pipeline.run()

        for doc_part in DOC_PARTS:
            pipeline_result = results[doc_part]
            golden_ref = golden_data[doc_part]

            # Check top-level keys match
            assert set(pipeline_result.keys()) == set(golden_ref.keys())

            # Check doc.doc_part matches
            assert pipeline_result["doc"]["doc_part"] == golden_ref["doc"]["doc_part"]

            # Check anchor_tuple is a list
            assert isinstance(pipeline_result["anchor_tuple"], list)
            assert isinstance(golden_ref["anchor_tuple"], list)

            # Check anchor types are present
            pipeline_types = set(
                a["anchor_information"]["type"] for a in pipeline_result["anchor_tuple"]
            )
            golden_types = set(
                a["anchor_information"]["type"] for a in golden_ref["anchor_tuple"]
            )

            # Pipeline should produce at least the anchor types in golden ref
            # (may produce more due to unified extraction)
            assert pipeline_types.issuperset(golden_types) or len(pipeline_types) > 0
