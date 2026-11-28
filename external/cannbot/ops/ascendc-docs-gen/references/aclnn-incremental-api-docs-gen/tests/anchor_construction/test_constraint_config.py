# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for constraint DocPartConfig factory function.

Verifies that make_constraint_config() produces the same config as defined
in gene_rule_registry.py lines 58-106, and that the config matches the
golden reference constraint.json structure.
"""

import json
from pathlib import Path

import pytest

from src.snapshot_construction.anchor_construction.common.id_generator import IdConfig
from src.snapshot_construction.anchor_construction.doc_parts.constraint_config import (
    make_constraint_config,
)
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    DocPartConfig,
    GeneRuleRegistry,
)


class TestConstraintConfigFactory:
    """Test make_constraint_config() factory function."""

    def test_factory_returns_doc_part_config(self):
        """Factory should return DocPartConfig instance."""
        config = make_constraint_config()
        assert isinstance(config, DocPartConfig)

    def test_doc_part_name(self):
        """Config should have doc_part='constraint'."""
        config = make_constraint_config()
        assert config.doc_part == "constraint"

    def test_anchor_types(self):
        """Config should have anchor_types=['file', 'section']."""
        config = make_constraint_config()
        assert config.anchor_types == ["file", "section"]
        assert len(config.anchor_types) == 2

    def test_output_filename(self):
        """Config should have output_filename='constraint.json'."""
        config = make_constraint_config()
        assert config.output_filename == "constraint.json"

    def test_gene_rule_mapping_has_file_and_section(self):
        """Gene rule mapping should have 'file' and 'section' keys."""
        config = make_constraint_config()
        assert "file" in config.gene_rule_mapping
        assert "section" in config.gene_rule_mapping
        assert len(config.gene_rule_mapping) == 2

    def test_file_anchor_types(self):
        """File anchors should have 5 types: op_api_aclnn_h, op_api_aclnn_cpp, op_host_def_cpp,
        op_host_tiling_cpp, op_kernel_cpp_h.
        """
        config = make_constraint_config()
        file_types = config.gene_rule_mapping["file"]
        expected_types = [
            "op_api_aclnn_h",
            "op_api_aclnn_cpp",
            "op_host_def_cpp",
            "op_host_tiling_cpp",
            "op_kernel_cpp_h",
        ]
        assert set(file_types.keys()) == set(expected_types)

    def test_section_anchor_types(self):
        """Section anchors should have 2 types: op_host_tiling_cpp, op_kernel_cpp_h."""
        config = make_constraint_config()
        section_types = config.gene_rule_mapping["section"]
        expected_types = ["op_host_tiling_cpp", "op_kernel_cpp_h"]
        assert set(section_types.keys()) == set(expected_types)


class TestConstraintGeneRuleTargets:
    """Test gene_rule.target format for constraint config."""

    def test_op_api_aclnn_h_targets(self):
        """op_api_aclnn_h should have 3 targets: deter_describe, constrain_model_scene,
        constrain_model_scene_clarify.
        """
        config = make_constraint_config()
        targets = config.gene_rule_mapping["file"]["op_api_aclnn_h"]["target"]
        assert targets == [
            ["deter_describe"],
            ["constrain_model_scene"],
            ["constrain_model_scene_clarify"],
        ]
        assert len(targets) == 3

    def test_op_host_def_cpp_targets(self):
        """op_host_def_cpp should have 2 targets: constrain_model_scene, constrain_model_scene_clarify."""
        config = make_constraint_config()
        targets = config.gene_rule_mapping["file"]["op_host_def_cpp"]["target"]
        assert targets == [["constrain_model_scene"], ["constrain_model_scene_clarify"]]
        assert len(targets) == 2

    def test_op_host_tiling_cpp_file_targets(self):
        """op_host_tiling_cpp (file) should have 2 targets."""
        config = make_constraint_config()
        targets = config.gene_rule_mapping["file"]["op_host_tiling_cpp"]["target"]
        assert targets == [["constrain_model_scene"], ["constrain_model_scene_clarify"]]
        assert len(targets) == 2

    def test_op_kernel_cpp_h_file_targets(self):
        """op_kernel_cpp_h (file) should have 2 targets."""
        config = make_constraint_config()
        targets = config.gene_rule_mapping["file"]["op_kernel_cpp_h"]["target"]
        assert targets == [["constrain_model_scene"], ["constrain_model_scene_clarify"]]
        assert len(targets) == 2

    def test_op_host_tiling_cpp_section_targets(self):
        """op_host_tiling_cpp (section) should have 2 targets."""
        config = make_constraint_config()
        targets = config.gene_rule_mapping["section"]["op_host_tiling_cpp"]["target"]
        assert targets == [["constrain_model_scene"], ["constrain_model_scene_clarify"]]
        assert len(targets) == 2

    def test_op_kernel_cpp_h_section_targets(self):
        """op_kernel_cpp_h (section) should have 2 targets."""
        config = make_constraint_config()
        targets = config.gene_rule_mapping["section"]["op_kernel_cpp_h"]["target"]
        assert targets == [["constrain_model_scene"], ["constrain_model_scene_clarify"]]
        assert len(targets) == 2

    def test_all_targets_are_one_element_lists(self):
        """All gene_rule targets should be 1-element lists."""
        config = make_constraint_config()

        # Check file anchors
        for file_type, rule in config.gene_rule_mapping["file"].items():
            for target in rule["target"]:
                assert isinstance(target, list)
                assert len(target) == 1
                assert isinstance(target[0], str)

        # Check section anchors
        for section_type, rule in config.gene_rule_mapping["section"].items():
            for target in rule["target"]:
                assert isinstance(target, list)
                assert len(target) == 1
                assert isinstance(target[0], str)

    def test_all_deter_values(self):
        """All gene_rule deter values should be '暂空'."""
        config = make_constraint_config()

        # Check file anchors
        for file_type, rule in config.gene_rule_mapping["file"].items():
            assert rule["deter"] == "暂空"

        # Check section anchors
        for section_type, rule in config.gene_rule_mapping["section"].items():
            assert rule["deter"] == "暂空"


class TestConstraintIdConfig:
    """Test IdConfig settings for constraint config."""

    def test_id_config_instance(self):
        """id_config should be IdConfig instance."""
        config = make_constraint_config()
        assert isinstance(config.id_config, IdConfig)

    def test_type_prefix_in_anchor(self):
        """type_prefix_in_anchor should be True."""
        config = make_constraint_config()
        assert config.id_config.type_prefix_in_anchor is True

    def test_type_prefix_in_evidence(self):
        """type_prefix_in_evidence should be True."""
        config = make_constraint_config()
        assert config.id_config.type_prefix_in_evidence is True

    def test_shared_counters(self):
        """shared_counters should be True."""
        config = make_constraint_config()
        assert config.id_config.shared_counters is True

    def test_category_offsets_none(self):
        """category_offsets should be None."""
        config = make_constraint_config()
        assert config.id_config.category_offsets is None


class TestConstraintSubPartTemplate:
    """Test sub_part_template structure for constraint config."""

    def test_sub_part_has_constrain_clarify(self):
        """sub_part_template should have 'constrain_clarify' key."""
        config = make_constraint_config()
        assert "constrain_clarify" in config.sub_part_template
        assert len(config.sub_part_template) == 1

    def test_constrain_clarify_is_list(self):
        """constrain_clarify should be a list."""
        config = make_constraint_config()
        assert isinstance(config.sub_part_template["constrain_clarify"], list)
        assert len(config.sub_part_template["constrain_clarify"]) == 1

    def test_constrain_clarify_item_structure(self):
        """constrain_clarify item should have 'content' and 'evidence_id' keys."""
        config = make_constraint_config()
        item = config.sub_part_template["constrain_clarify"][0]
        assert "content" in item
        assert "evidence_id" in item
        assert item["content"] == ""
        assert isinstance(item["evidence_id"], list)
        assert len(item["evidence_id"]) == 0


class TestConstraintConfigMatchesRegistry:
    """Test that factory config matches GeneRuleRegistry inline definition."""

    def test_matches_registry_constraint_config(self):
        """Factory config should match registry's constraint config exactly."""
        factory_config = make_constraint_config()
        registry = GeneRuleRegistry()
        registry_config = registry.get_config("constraint")

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


class TestConstraintConfigMatchesGoldenRef:
    """Test that config matches golden reference constraint.json structure."""

    @pytest.fixture
    def golden_constraint(self):
        """Load golden reference constraint.json."""
        golden_path = Path(__file__).parent / "golden_ref" / "constraint.json"
        with open(golden_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_doc_part_matches_golden(self, golden_constraint):
        """Config doc_part should match golden doc.doc_part."""
        config = make_constraint_config()
        assert config.doc_part == golden_constraint["doc"]["doc_part"]

    def test_output_filename_matches_golden_file(self, golden_constraint):
        """Config output_filename should match golden file name."""
        config = make_constraint_config()
        # Golden file is constraint.json
        assert config.output_filename == "constraint.json"

    def test_anchor_types_match_golden_anchor_tuple(self, golden_constraint):
        """Config anchor_types should match types found in golden anchor_tuple."""
        config = make_constraint_config()

        # Extract anchor types from golden reference
        anchor_types_in_golden = set()
        for anchor in golden_constraint["anchor_tuple"]:
            anchor_types_in_golden.add(anchor["anchor_information"]["type"])

        # Constraint golden ref has file anchors only (no section anchors in this sample)
        # But config should support both file and section
        assert "file" in config.anchor_types
        assert "section" in config.anchor_types

    def test_sub_part_template_matches_golden_structure(self, golden_constraint):
        """Config sub_part_template should match golden doc.sub_part structure."""
        config = make_constraint_config()

        # Golden has constrain_clarify key
        assert "constrain_clarify" in config.sub_part_template

        # Golden constrain_clarify is a list with one item
        golden_sub_part = golden_constraint["doc"]["sub_part"]["constrain_clarify"]
        assert isinstance(golden_sub_part, list)

        # Template should have same structure
        template_item = config.sub_part_template["constrain_clarify"][0]
        golden_item = golden_sub_part[0]

        # Both should have content and evidence_id
        assert "content" in template_item
        assert "evidence_id" in template_item
        assert "content" in golden_item
        assert "evidence_id" in golden_item

    def test_gene_rule_targets_match_golden_format(self, golden_constraint):
        """Config gene_rule targets should match golden format (1-element lists)."""
        config = make_constraint_config()

        # Check golden reference gene_rule targets
        for anchor in golden_constraint["anchor_tuple"]:
            for evidence in anchor["evidence"]:
                gene_rule_target = evidence["gene_rule"]["target"]

                # All targets should be 1-element lists
                for target in gene_rule_target:
                    assert isinstance(target, list)
                    assert len(target) == 1

        # Config targets should also be 1-element lists
        for anchor_type, rules in config.gene_rule_mapping.items():
            for rule_key, rule in rules.items():
                for target in rule["target"]:
                    assert isinstance(target, list)
                    assert len(target) == 1

    def test_file_anchor_gene_rules_present_in_golden(self, golden_constraint):
        """Config file anchor gene_rules should cover patterns found in golden."""
        config = make_constraint_config()

        # Golden has file anchors with specific patterns
        file_anchor_paths = [
            anchor["anchor_information"]["level1"]["file_path"]
            for anchor in golden_constraint["anchor_tuple"]
            if anchor["anchor_information"]["type"] == "file"
        ]

        # Check that config covers these patterns
        # op_api/aclnn_*.h -> op_api_aclnn_h
        # op_host/*_def.cpp -> op_host_def_cpp
        # op_kernel/*.cpp or *.h -> op_kernel_cpp_h

        has_aclnn_h = any("aclnn_" in path and path.endswith(".h") for path in file_anchor_paths)
        has_def_cpp = any("_def.cpp" in path for path in file_anchor_paths)
        has_kernel_cpp_h = any(
            "op_kernel" in path and (path.endswith(".cpp") or path.endswith(".h")) for path in file_anchor_paths
        )

        if has_aclnn_h:
            assert "op_api_aclnn_h" in config.gene_rule_mapping["file"]
        if has_def_cpp:
            assert "op_host_def_cpp" in config.gene_rule_mapping["file"]
        if has_kernel_cpp_h:
            assert "op_kernel_cpp_h" in config.gene_rule_mapping["file"]


class TestConstraintConfigIntegration:
    """Integration tests for constraint config."""

    def test_config_can_be_imported_from_doc_parts(self):
        """Config should be importable from doc_parts package."""
        from src.snapshot_construction.anchor_construction.doc_parts import (
            make_constraint_config,
        )

        config = make_constraint_config()
        assert config.doc_part == "constraint"

    def test_config_is_in_all_exports(self):
        """make_constraint_config should be in __all__."""
        from src.snapshot_construction.anchor_construction.doc_parts import __all__

        assert "make_constraint_config" in __all__

    def test_config_logger_name(self):
        """Config should use correct logger name."""
        import logging

        from src.snapshot_construction.anchor_construction.doc_parts.constraint_config import (
            logger,
        )

        assert logger.name == "src.snapshot_construction.anchor_construction.doc_parts.constraint_config"


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v"])
