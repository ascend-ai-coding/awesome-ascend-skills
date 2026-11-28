# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for summary DocPartConfig factory function.

Validates:
- make_summary_config() returns correct DocPartConfig
- Summary config matches golden reference structure
- gene_rule targets per file group
- doc.sub_part structure
- IdConfig settings
"""

import json
from pathlib import Path

import pytest

from src.snapshot_construction.anchor_construction.common.id_generator import IdConfig
from src.snapshot_construction.anchor_construction.doc_parts.summary_config import (
    make_summary_config,
)
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    DocPartConfig,
)


class TestSummaryConfigCreation:
    """Tests for make_summary_config() factory function."""

    def test_make_summary_config_returns_doc_part_config(self):
        """测试 make_summary_config() 返回 DocPartConfig 实例"""
        config = make_summary_config()

        assert isinstance(config, DocPartConfig)
        assert config.doc_part == "summary"

    def test_summary_config_anchor_types(self):
        """测试 summary config anchor_types 为 ['file', 'note']"""
        config = make_summary_config()

        assert config.anchor_types == ["file", "note"]
        assert len(config.anchor_types) == 2

    def test_summary_config_output_filename(self):
        """测试 summary config output_filename 为 'summary.json'"""
        config = make_summary_config()

        assert config.output_filename == "summary.json"

    def test_summary_config_gene_rule_mapping_keys(self):
        """测试 summary config gene_rule_mapping 包含 file 和 note"""
        config = make_summary_config()

        assert "file" in config.gene_rule_mapping
        assert "note" in config.gene_rule_mapping
        assert len(config.gene_rule_mapping) == 2


class TestSummaryGeneRuleTargets:
    """Tests for gene_rule targets per file group."""

    def test_op_api_aclnn_h_targets(self):
        """测试 op_api_aclnn_h gene_rule targets"""
        config = make_summary_config()

        file_mapping = config.gene_rule_mapping["file"]
        assert "op_api_aclnn_h" in file_mapping

        rule = file_mapping["op_api_aclnn_h"]
        assert rule["deter"] == ""
        assert rule["target"] == [["summary_clarify"], ["summary_formula"]]

    def test_op_api_aclnn_cpp_targets(self):
        """测试 op_api_aclnn_cpp gene_rule targets"""
        config = make_summary_config()

        file_mapping = config.gene_rule_mapping["file"]
        assert "op_api_aclnn_cpp" in file_mapping

        rule = file_mapping["op_api_aclnn_cpp"]
        assert rule["deter"] == ""
        assert rule["target"] == [["summary_clarify"]]

    def test_op_kernel_cpp_h_targets(self):
        """测试 op_kernel_cpp_h gene_rule targets"""
        config = make_summary_config()

        file_mapping = config.gene_rule_mapping["file"]
        assert "op_kernel_cpp_h" in file_mapping

        rule = file_mapping["op_kernel_cpp_h"]
        assert rule["deter"] == ""
        assert rule["target"] == [["summary_formula"]]

    def test_note_default_targets(self):
        """测试 note default gene_rule targets"""
        config = make_summary_config()

        note_mapping = config.gene_rule_mapping["note"]
        assert "default" in note_mapping

        rule = note_mapping["default"]
        assert rule["deter"] == ""
        assert rule["target"] == [["summary_clarify"], ["summary_formula"]]

    def test_all_targets_are_1_element_lists(self):
        """测试所有 gene_rule targets 都是 1-element lists"""
        config = make_summary_config()

        # Check file anchors
        for file_type, rule in config.gene_rule_mapping["file"].items():
            for target in rule["target"]:
                assert len(target) == 1, f"{file_type} target should be 1-element list"

        # Check note anchors
        for note_type, rule in config.gene_rule_mapping["note"].items():
            for target in rule["target"]:
                assert len(target) == 1, f"{note_type} target should be 1-element list"


class TestSummarySubPartTemplate:
    """Tests for doc.sub_part structure."""

    def test_sub_part_template_keys(self):
        """测试 sub_part_template 包含 summary_clarify 和 summary_formula"""
        config = make_summary_config()

        assert "summary_clarify" in config.sub_part_template
        assert "summary_formula" in config.sub_part_template
        assert len(config.sub_part_template) == 2

    def test_summary_clarify_structure(self):
        """测试 summary_clarify 结构"""
        config = make_summary_config()

        clarify = config.sub_part_template["summary_clarify"]
        assert isinstance(clarify, list)
        assert len(clarify) == 1
        assert "content" in clarify[0]
        assert clarify[0]["content"] == ""

    def test_summary_formula_structure(self):
        """测试 summary_formula 结构"""
        config = make_summary_config()

        formula = config.sub_part_template["summary_formula"]
        assert isinstance(formula, list)
        assert len(formula) == 1
        assert "content" in formula[0]
        assert formula[0]["content"] == ""

    def test_sub_part_template_matches_golden_ref(self):
        """测试 sub_part_template 结构匹配 golden reference"""
        config = make_summary_config()

        # Load golden reference
        golden_ref_path = Path(__file__).parent / "golden_ref" / "summary.json"
        with open(golden_ref_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)

        golden_sub_part = golden_data["doc"]["sub_part"]

        # Verify keys match
        assert set(config.sub_part_template.keys()) == set(golden_sub_part.keys())

        # Verify structure matches (both have list with content field)
        for key in config.sub_part_template:
            assert isinstance(config.sub_part_template[key], list)
            assert isinstance(golden_sub_part[key], list)
            assert len(config.sub_part_template[key]) == len(golden_sub_part[key])
            assert "content" in config.sub_part_template[key][0]
            assert "content" in golden_sub_part[key][0]


class TestSummaryIdConfig:
    """Tests for IdConfig settings."""

    def test_id_config_type_prefix_in_anchor(self):
        """测试 IdConfig type_prefix_in_anchor=True"""
        config = make_summary_config()

        assert isinstance(config.id_config, IdConfig)
        assert config.id_config.type_prefix_in_anchor is True

    def test_id_config_type_prefix_in_evidence(self):
        """测试 IdConfig type_prefix_in_evidence=False"""
        config = make_summary_config()

        assert config.id_config.type_prefix_in_evidence is False

    def test_id_config_shared_counters(self):
        """测试 IdConfig shared_counters=False"""
        config = make_summary_config()

        assert config.id_config.shared_counters is False

    def test_id_config_category_offsets(self):
        """测试 IdConfig category_offsets=None"""
        config = make_summary_config()

        assert config.id_config.category_offsets is None


class TestSummaryConfigMatchesRegistry:
    """Tests that summary config matches GeneRuleRegistry inline definition."""

    def test_summary_config_matches_registry(self):
        """测试 summary config 匹配 GeneRuleRegistry 内联定义"""
        from src.snapshot_construction.anchor_construction.gene_rule_registry import (
            GeneRuleRegistry,
        )

        # Get config from factory function
        factory_config = make_summary_config()

        # Get config from registry
        registry = GeneRuleRegistry()
        registry_config = registry.get_config("summary")

        # Compare all fields
        assert factory_config.doc_part == registry_config.doc_part
        assert factory_config.anchor_types == registry_config.anchor_types
        assert factory_config.output_filename == registry_config.output_filename

        # Compare gene_rule_mapping
        assert factory_config.gene_rule_mapping == registry_config.gene_rule_mapping

        # Compare id_config
        assert factory_config.id_config.type_prefix_in_anchor == registry_config.id_config.type_prefix_in_anchor
        assert factory_config.id_config.type_prefix_in_evidence == registry_config.id_config.type_prefix_in_evidence
        assert factory_config.id_config.shared_counters == registry_config.id_config.shared_counters
        assert factory_config.id_config.category_offsets == registry_config.id_config.category_offsets

        # Compare sub_part_template
        assert factory_config.sub_part_template == registry_config.sub_part_template


class TestSummaryConfigIntegration:
    """Integration tests for summary config with golden reference."""

    def test_summary_config_anchor_types_match_golden(self):
        """测试 anchor_types 匹配 golden reference 中的 anchor 类型"""
        config = make_summary_config()

        # Load golden reference
        golden_ref_path = Path(__file__).parent / "golden_ref" / "summary.json"
        with open(golden_ref_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)

        # Extract anchor types from golden reference
        anchor_types_in_golden = set()
        for anchor in golden_data["anchor_tuple"]:
            anchor_type = anchor["anchor_information"]["type"]
            anchor_types_in_golden.add(anchor_type)

        # Verify config anchor_types match golden reference
        assert set(config.anchor_types) == anchor_types_in_golden

    def test_summary_config_gene_rule_targets_match_golden(self):
        """测试 gene_rule targets 匹配 golden reference"""
        config = make_summary_config()

        # Load golden reference
        golden_ref_path = Path(__file__).parent / "golden_ref" / "summary.json"
        with open(golden_ref_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)

        # Extract gene_rule targets from golden reference
        golden_targets = {}
        for anchor in golden_data["anchor_tuple"]:
            anchor_type = anchor["anchor_information"]["type"]

            # Skip note anchors (they have different structure)
            if anchor_type == "note":
                continue

            # Get file_name for file anchors
            if anchor_type == "file":
                file_name = anchor["anchor_information"]["level1"]["file_name"]

                # Get gene_rule target from evidence
                if anchor["evidence"]:
                    target = anchor["evidence"][0]["gene_rule"]["target"]

                    # Map file_name to file_type
                    if file_name.endswith(".h") and "aclnn" in file_name:
                        file_type = "op_api_aclnn_h"
                    elif file_name.endswith(".cpp") and "aclnn" in file_name:
                        file_type = "op_api_aclnn_cpp"
                    elif (
                        file_name.endswith(".h") and "op_kernel" in anchor["anchor_information"]["level1"]["file_path"]
                    ):
                        file_type = "op_kernel_cpp_h"
                    else:
                        continue

                    if file_type not in golden_targets:
                        golden_targets[file_type] = target

        # Verify config targets match golden reference
        for file_type, golden_target in golden_targets.items():
            config_target = config.gene_rule_mapping["file"][file_type]["target"]
            assert config_target == golden_target, f"{file_type} targets mismatch"


if __name__ == "__main__":
    """Run tests directly."""
    pytest.main([__file__, "-v"])
