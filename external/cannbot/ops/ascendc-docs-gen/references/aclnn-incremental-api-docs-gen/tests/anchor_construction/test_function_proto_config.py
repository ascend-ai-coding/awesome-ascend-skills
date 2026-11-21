# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for function_proto DocPartConfig factory function.

Validates:
- make_function_proto_config() returns correct DocPartConfig
- Config matches gene_rule_registry.py inline definition
- Gene_rule targets match golden reference structure
- IdConfig settings are correct
- sub_part_template structure matches golden reference
"""

import json
from pathlib import Path

import pytest

from src.snapshot_construction.anchor_construction.common.id_generator import IdConfig
from src.snapshot_construction.anchor_construction.doc_parts.function_proto_config import (
    make_function_proto_config,
)
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    DocPartConfig,
    GeneRuleRegistry,
)


class TestFunctionProtoConfigFactory:
    """Tests for make_function_proto_config() factory function."""

    def test_factory_returns_doc_part_config(self):
        """测试 make_function_proto_config() 返回 DocPartConfig 实例"""
        config = make_function_proto_config()

        assert isinstance(config, DocPartConfig)
        assert config.doc_part == "function_proto"

    def test_factory_matches_registry_config(self):
        """测试工厂函数返回的配置与 GeneRuleRegistry 中的配置一致"""
        factory_config = make_function_proto_config()
        registry = GeneRuleRegistry()
        registry_config = registry.get_config("function_proto")

        # Check all fields match
        assert factory_config.doc_part == registry_config.doc_part
        assert factory_config.anchor_types == registry_config.anchor_types
        assert factory_config.output_filename == registry_config.output_filename

        # Check gene_rule_mapping structure
        assert factory_config.gene_rule_mapping.keys() == registry_config.gene_rule_mapping.keys()
        assert factory_config.gene_rule_mapping["file"] == registry_config.gene_rule_mapping["file"]
        assert factory_config.gene_rule_mapping["function"] == registry_config.gene_rule_mapping["function"]

        # Check id_config
        assert factory_config.id_config.type_prefix_in_anchor == registry_config.id_config.type_prefix_in_anchor
        assert factory_config.id_config.type_prefix_in_evidence == registry_config.id_config.type_prefix_in_evidence
        assert factory_config.id_config.shared_counters == registry_config.id_config.shared_counters

        # Check sub_part_template
        assert factory_config.sub_part_template == registry_config.sub_part_template

    def test_anchor_types_file_and_function(self):
        """测试 anchor_types 包含 file 和 function"""
        config = make_function_proto_config()

        assert config.anchor_types == ["file", "function"]
        assert len(config.anchor_types) == 2

    def test_output_filename(self):
        """测试 output_filename 为 function_proto.json"""
        config = make_function_proto_config()

        assert config.output_filename == "function_proto.json"


class TestFunctionProtoGeneRuleMapping:
    """Tests for function_proto gene_rule_mapping structure."""

    def test_gene_rule_mapping_has_file_and_function(self):
        """测试 gene_rule_mapping 包含 file 和 function 键"""
        config = make_function_proto_config()

        assert "file" in config.gene_rule_mapping
        assert "function" in config.gene_rule_mapping
        assert len(config.gene_rule_mapping) == 2

    def test_file_gene_rule_op_api_aclnn_h(self):
        """测试 file anchor 的 op_api_aclnn_h gene_rule"""
        config = make_function_proto_config()

        file_mapping = config.gene_rule_mapping["file"]

        assert "op_api_aclnn_h" in file_mapping
        op_api_rule = file_mapping["op_api_aclnn_h"]

        # Check deter is empty string
        assert op_api_rule["deter"] == ""

        # Check target is 1-element lists
        target = op_api_rule["target"]
        assert len(target) == 2
        assert target[0] == ["function_proto_clarify"]
        assert target[1] == ["sub_function_sign"]

    def test_function_gene_rule_default(self):
        """测试 function anchor 的 default gene_rule"""
        config = make_function_proto_config()

        function_mapping = config.gene_rule_mapping["function"]

        assert "default" in function_mapping
        default_rule = function_mapping["default"]

        # Check deter is empty string
        assert default_rule["deter"] == ""

        # Check target is 1-element list
        target = default_rule["target"]
        assert len(target) == 1
        assert target[0] == ["sub_function_sign"]

    def test_gene_rule_targets_are_one_element_lists(self):
        """测试所有 gene_rule.target 都是1元素列表"""
        config = make_function_proto_config()

        for anchor_type, mappings in config.gene_rule_mapping.items():
            for pattern, rule in mappings.items():
                for target_item in rule["target"]:
                    assert len(target_item) == 1
                    assert isinstance(target_item[0], str)


class TestFunctionProtoIdConfig:
    """Tests for function_proto IdConfig settings."""

    def test_type_prefix_in_anchor_true(self):
        """测试 type_prefix_in_anchor 为 True"""
        config = make_function_proto_config()

        assert config.id_config.type_prefix_in_anchor

    def test_type_prefix_in_evidence_false(self):
        """测试 type_prefix_in_evidence 为 False"""
        config = make_function_proto_config()

        assert not config.id_config.type_prefix_in_evidence

    def test_shared_counters_false(self):
        """测试 shared_counters 为 False"""
        config = make_function_proto_config()

        assert not config.id_config.shared_counters

    def test_category_offsets_none(self):
        """测试 category_offsets 为 None"""
        config = make_function_proto_config()

        assert config.id_config.category_offsets is None

    def test_id_config_matches_summary_pattern(self):
        """测试 IdConfig 与 summary 配置模式相同"""
        # function_proto and summary share the same IdConfig pattern
        function_proto_config = make_function_proto_config()
        registry = GeneRuleRegistry()
        summary_config = registry.get_config("summary")

        assert function_proto_config.id_config.type_prefix_in_anchor == summary_config.id_config.type_prefix_in_anchor
        assert (
            function_proto_config.id_config.type_prefix_in_evidence == summary_config.id_config.type_prefix_in_evidence
        )
        assert function_proto_config.id_config.shared_counters == summary_config.id_config.shared_counters


class TestFunctionProtoSubPartTemplate:
    """Tests for function_proto sub_part_template structure."""

    def test_sub_part_template_has_two_keys(self):
        """测试 sub_part_template 包含两个键"""
        config = make_function_proto_config()

        assert "function_proto_clarify" in config.sub_part_template
        assert "sub_function_sign" in config.sub_part_template
        assert len(config.sub_part_template) == 2

    def test_function_proto_clarify_structure(self):
        """测试 function_proto_clarify 结构"""
        config = make_function_proto_config()

        clarify_list = config.sub_part_template["function_proto_clarify"]

        assert isinstance(clarify_list, list)
        assert len(clarify_list) == 1

        # Check template item has content field
        template_item = clarify_list[0]
        assert "content" in template_item
        assert template_item["content"] == ""

    def test_sub_function_sign_structure(self):
        """测试 sub_function_sign 结构"""
        config = make_function_proto_config()

        sign_list = config.sub_part_template["sub_function_sign"]

        assert isinstance(sign_list, list)
        assert len(sign_list) == 1

        # Check template item has content field
        template_item = sign_list[0]
        assert "content" in template_item
        assert template_item["content"] == ""

    def test_sub_part_template_matches_golden_reference(self):
        """测试 sub_part_template 与 golden reference 结构一致"""
        config = make_function_proto_config()

        # Load golden reference
        golden_path = Path(__file__).parent / "golden_ref" / "function_proto.json"
        with open(golden_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)

        golden_sub_part = golden_data["doc"]["sub_part"]

        # Check keys match
        assert set(config.sub_part_template.keys()) == set(golden_sub_part.keys())

        # Check structure: both should have list with content field
        for key in config.sub_part_template.keys():
            assert isinstance(config.sub_part_template[key], list)
            assert isinstance(golden_sub_part[key], list)

            # Template has 1 item, golden may have more
            assert len(config.sub_part_template[key]) >= 1

            # Check content field exists
            assert "content" in config.sub_part_template[key][0]
            assert "content" in golden_sub_part[key][0]


class TestFunctionProtoGoldenReferenceIntegration:
    """Tests verifying function_proto config matches golden reference output."""

    def test_gene_rule_targets_match_golden_file_anchor(self):
        """测试 file anchor gene_rule.target 与 golden reference 一致"""
        config = make_function_proto_config()

        # Load golden reference
        golden_path = Path(__file__).parent / "golden_ref" / "function_proto.json"
        with open(golden_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)

        # Find file anchor in golden reference
        file_anchor = None
        for anchor in golden_data["anchor_tuple"]:
            if anchor["anchor_information"]["type"] == "file":
                file_anchor = anchor
                break

        assert file_anchor is not None

        # Check gene_rule.target matches
        golden_target = file_anchor["evidence"][0]["gene_rule"]["target"]
        expected_target = [["function_proto_clarify"], ["sub_function_sign"]]

        assert golden_target == expected_target
        assert config.gene_rule_mapping["file"]["op_api_aclnn_h"]["target"] == expected_target

    def test_gene_rule_targets_match_golden_function_anchor(self):
        """测试 function anchor gene_rule.target 与 golden reference 一致"""
        config = make_function_proto_config()

        # Load golden reference
        golden_path = Path(__file__).parent / "golden_ref" / "function_proto.json"
        with open(golden_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)

        # Find function anchors in golden reference
        # fmt: off
        function_anchors = [
            anchor
            for anchor in golden_data["anchor_tuple"]
            if anchor["anchor_information"]["type"] == "function"
        ]
        # fmt: on

        assert len(function_anchors) > 0

        # All function anchors should have same gene_rule.target
        expected_target = [["sub_function_sign"]]

        for anchor in function_anchors:
            golden_target = anchor["evidence"][0]["gene_rule"]["target"]
            assert golden_target == expected_target

        # Check config matches
        assert config.gene_rule_mapping["function"]["default"]["target"] == expected_target

    def test_anchor_types_match_golden_reference(self):
        """测试 anchor_types 与 golden reference 一致"""
        config = make_function_proto_config()

        # Load golden reference
        golden_path = Path(__file__).parent / "golden_ref" / "function_proto.json"
        with open(golden_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)

        # Extract anchor types from golden reference
        golden_anchor_types = set()
        for anchor in golden_data["anchor_tuple"]:
            golden_anchor_types.add(anchor["anchor_information"]["type"])

        # Check config anchor_types match golden
        assert set(config.anchor_types) == golden_anchor_types
        assert golden_anchor_types == {"file", "function"}

    def test_doc_part_matches_golden_reference(self):
        """测试 doc_part 与 golden reference 一致"""
        config = make_function_proto_config()

        # Load golden reference
        golden_path = Path(__file__).parent / "golden_ref" / "function_proto.json"
        with open(golden_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)

        assert config.doc_part == golden_data["doc"]["doc_part"]


class TestFunctionProtoConfigSelfTest:
    """Tests for function_proto_config.py self-test block."""

    def test_self_test_executable(self):
        """测试 self-test block 可执行"""
        # Import and run the module's self-test
        import subprocess
        import sys

        module_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "snapshot_construction"
            / "anchor_construction"
            / "doc_parts"
            / "function_proto_config.py"
        )

        # Run from the correct directory with PYTHONPATH set
        result = subprocess.run(
            [sys.executable, str(module_path)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(Path(__file__).parent.parent.parent),  # Run from scripts/cann-document-generation
            env={"PYTHONPATH": str(Path(__file__).parent.parent.parent)},
        )

        # Should execute successfully
        assert result.returncode == 0
        assert "Self-test passed" in result.stdout
        assert "function_proto" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
