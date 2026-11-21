# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for support DocPartConfig factory.

Validates:
- make_support_config() returns correct DocPartConfig
- Config matches gene_rule_registry.py inline definition
- TYPE1/TYPE2/TYPE3 section patterns
- category_offsets (TYPE1→1, TYPE2→101, TYPE3→201)
- doc.sub_part structure: support_matrix with content and evidence_ids
- gene_rule.target format: 1-element lists
"""

import json
from pathlib import Path

import pytest

from src.snapshot_construction.anchor_construction.common.id_generator import IdConfig
from src.snapshot_construction.anchor_construction.doc_parts.support_config import (
    make_support_config,
)
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    GeneRuleRegistry,
)


class TestMakeSupportConfig:
    """Tests for make_support_config() factory function."""

    def test_make_support_config_returns_doc_part_config(self):
        """测试 make_support_config() 返回 DocPartConfig 实例"""
        config = make_support_config()

        assert config.doc_part == "support"
        assert isinstance(config.id_config, IdConfig)

    def test_make_support_config_doc_part_name(self):
        """测试 support config doc_part 名称"""
        config = make_support_config()

        assert config.doc_part == "support"

    def test_make_support_config_anchor_types(self):
        """测试 support anchor_types 包含 file 和 section"""
        config = make_support_config()

        assert config.anchor_types == ["file", "section"]
        assert "file" in config.anchor_types
        assert "section" in config.anchor_types
        assert len(config.anchor_types) == 2

    def test_make_support_config_output_filename(self):
        """测试 support output_filename"""
        config = make_support_config()

        assert config.output_filename == "support.json"


class TestSupportGeneRuleMapping:
    """Tests for support gene_rule_mapping structure."""

    def test_gene_rule_mapping_has_file_and_section(self):
        """测试 gene_rule_mapping 包含 file 和 section"""
        config = make_support_config()

        assert "file" in config.gene_rule_mapping
        assert "section" in config.gene_rule_mapping

    def test_file_mapping_has_default(self):
        """测试 file mapping 包含 default 模式"""
        config = make_support_config()

        file_mapping = config.gene_rule_mapping["file"]
        assert "default" in file_mapping

        # Check default pattern
        default_rule = file_mapping["default"]
        assert default_rule["deter"] == ""
        assert default_rule["target"] == [["support_matrix"]]

    def test_section_mapping_has_three_types(self):
        """测试 section mapping 包含 TYPE1/TYPE2/TYPE3"""
        config = make_support_config()

        section_mapping = config.gene_rule_mapping["section"]

        assert "TYPE1" in section_mapping
        assert "TYPE2" in section_mapping
        assert "TYPE3" in section_mapping

    def test_type1_deter_string(self):
        """测试 TYPE1 deter 字符串包含芯片信息"""
        config = make_support_config()

        type1_rule = config.gene_rule_mapping["section"]["TYPE1"]

        assert "AICore().AddConfig" in type1_rule["deter"]
        assert "芯片" in type1_rule["deter"]
        assert "{chips}" in type1_rule["deter"]

    def test_type2_deter_string(self):
        """测试 TYPE2 deter 字符串包含 SOC 信息"""
        config = make_support_config()

        type2_rule = config.gene_rule_mapping["section"]["TYPE2"]

        assert "curShortSocName" in type2_rule["deter"]
        assert "SocVersion" in type2_rule["deter"]
        assert "SOC" in type2_rule["deter"]
        assert "{socs}" in type2_rule["deter"]

    def test_type3_deter_string(self):
        """测试 TYPE3 deter 字符串包含架构宏信息"""
        config = make_support_config()

        type3_rule = config.gene_rule_mapping["section"]["TYPE3"]

        assert "{arch_macro}" in type3_rule["deter"]
        assert "{arch_name}" in type3_rule["deter"]

    def test_all_section_types_target_support_matrix(self):
        """测试所有 section TYPE 都 target support_matrix"""
        config = make_support_config()

        section_mapping = config.gene_rule_mapping["section"]

        for type_name in ["TYPE1", "TYPE2", "TYPE3"]:
            target = section_mapping[type_name]["target"]
            assert target == [["support_matrix"]]

    def test_gene_rule_target_format_one_element(self):
        """测试 gene_rule.target 为1元素列表"""
        config = make_support_config()

        # All targets should be 1-element lists
        for anchor_type, mappings in config.gene_rule_mapping.items():
            for pattern, rule in mappings.items():
                for target_item in rule["target"]:
                    assert len(target_item) == 1
                    assert isinstance(target_item[0], str)


class TestSupportIdConfig:
    """Tests for support IdConfig."""

    def test_type_prefix_in_anchor_false(self):
        """测试 type_prefix_in_anchor 为 False"""
        config = make_support_config()

        assert not config.id_config.type_prefix_in_anchor

    def test_type_prefix_in_evidence_false(self):
        """测试 type_prefix_in_evidence 为 False"""
        config = make_support_config()

        assert not config.id_config.type_prefix_in_evidence

    def test_shared_counters_false(self):
        """测试 shared_counters 为 False"""
        config = make_support_config()

        assert not config.id_config.shared_counters

    def test_category_offsets_exist(self):
        """测试 category_offsets 存在"""
        config = make_support_config()

        assert config.id_config.category_offsets is not None

    def test_category_offsets_type1_starts_at_1(self):
        """测试 TYPE1 category_offset 为 1"""
        config = make_support_config()

        offsets = config.id_config.category_offsets
        assert offsets["TYPE1"] == 1

    def test_category_offsets_type2_starts_at_101(self):
        """测试 TYPE2 category_offset 为 101"""
        config = make_support_config()

        offsets = config.id_config.category_offsets
        assert offsets["TYPE2"] == 101

    def test_category_offsets_type3_starts_at_201(self):
        """测试 TYPE3 category_offset 为 201"""
        config = make_support_config()

        offsets = config.id_config.category_offsets
        assert offsets["TYPE3"] == 201

    def test_category_offsets_all_three_types(self):
        """测试 category_offsets 包含全部3种 TYPE"""
        config = make_support_config()

        offsets = config.id_config.category_offsets
        assert len(offsets) == 3
        assert "TYPE1" in offsets
        assert "TYPE2" in offsets
        assert "TYPE3" in offsets


class TestSupportSubPartTemplate:
    """Tests for support sub_part_template structure."""

    def test_sub_part_template_has_support_matrix(self):
        """测试 sub_part_template 包含 support_matrix"""
        config = make_support_config()

        assert "support_matrix" in config.sub_part_template

    def test_support_matrix_is_list(self):
        """测试 support_matrix 为列表"""
        config = make_support_config()

        assert isinstance(config.sub_part_template["support_matrix"], list)

    def test_support_matrix_item_has_content(self):
        """测试 support_matrix item 包含 content"""
        config = make_support_config()

        matrix_item = config.sub_part_template["support_matrix"][0]
        assert "content" in matrix_item

    def test_support_matrix_item_has_evidence_ids(self):
        """测试 support_matrix item 包含 evidence_ids (plural)"""
        config = make_support_config()

        matrix_item = config.sub_part_template["support_matrix"][0]
        assert "evidence_ids" in matrix_item
        assert isinstance(matrix_item["evidence_ids"], list)

    def test_support_matrix_item_not_evidence_id(self):
        """测试 support_matrix item 不使用 evidence_id (singular)"""
        config = make_support_config()

        matrix_item = config.sub_part_template["support_matrix"][0]
        # Should NOT have evidence_id (singular)
        assert "evidence_id" not in matrix_item


class TestSupportConfigMatchesRegistry:
    """Tests that factory config matches registry inline definition."""

    def test_matches_registry_support_config(self):
        """测试 factory config 与 registry inline 定义一致"""
        factory_config = make_support_config()
        registry = GeneRuleRegistry()
        registry_config = registry.get_config("support")

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

    def test_matches_registry_section_type1(self):
        """测试 TYPE1 与 registry 一致"""
        factory_config = make_support_config()
        registry_config = GeneRuleRegistry().get_config("support")

        factory_type1 = factory_config.gene_rule_mapping["section"]["TYPE1"]
        registry_type1 = registry_config.gene_rule_mapping["section"]["TYPE1"]

        assert factory_type1 == registry_type1

    def test_matches_registry_section_type2(self):
        """测试 TYPE2 与 registry 一致"""
        factory_config = make_support_config()
        registry_config = GeneRuleRegistry().get_config("support")

        factory_type2 = factory_config.gene_rule_mapping["section"]["TYPE2"]
        registry_type2 = registry_config.gene_rule_mapping["section"]["TYPE2"]

        assert factory_type2 == registry_type2

    def test_matches_registry_section_type3(self):
        """测试 TYPE3 与 registry 一致"""
        factory_config = make_support_config()
        registry_config = GeneRuleRegistry().get_config("support")

        factory_type3 = factory_config.gene_rule_mapping["section"]["TYPE3"]
        registry_type3 = registry_config.gene_rule_mapping["section"]["TYPE3"]

        assert factory_type3 == registry_type3


class TestSupportConfigMatchesGoldenRef:
    """Tests that config matches golden reference output structure."""

    def test_matches_golden_ref_doc_part(self):
        """测试 config doc_part 与 golden ref 一致"""
        config = make_support_config()

        golden_ref_path = Path(__file__).parent / "golden_ref" / "support.json"
        with open(golden_ref_path, "r", encoding="utf-8") as f:
            golden = json.load(f)

        assert config.doc_part == golden["doc"]["doc_part"]

    def test_matches_golden_ref_sub_part_structure(self):
        """测试 config sub_part_template 与 golden ref 结构一致"""
        config = make_support_config()

        golden_ref_path = Path(__file__).parent / "golden_ref" / "support.json"
        with open(golden_ref_path, "r", encoding="utf-8") as f:
            golden = json.load(f)

        # Golden ref has support_matrix in sub_part
        assert "support_matrix" in golden["doc"]["sub_part"]

        # Config template should match structure
        assert "support_matrix" in config.sub_part_template

        # Golden ref support_matrix is a list
        assert isinstance(golden["doc"]["sub_part"]["support_matrix"], list)

        # Config template support_matrix is also a list
        assert isinstance(config.sub_part_template["support_matrix"], list)

    def test_matches_golden_ref_evidence_ids_field(self):
        """测试 config 使用 evidence_ids 与 golden ref 一致"""
        config = make_support_config()

        golden_ref_path = Path(__file__).parent / "golden_ref" / "support.json"
        with open(golden_ref_path, "r", encoding="utf-8") as f:
            golden = json.load(f)

        # Golden ref uses evidence_ids (plural)
        golden_item = golden["doc"]["sub_part"]["support_matrix"][0]
        assert "evidence_ids" in golden_item

        # Config template also uses evidence_ids
        config_item = config.sub_part_template["support_matrix"][0]
        assert "evidence_ids" in config_item

    def test_matches_golden_ref_anchor_types(self):
        """测试 config anchor_types 与 golden ref 一致"""
        config = make_support_config()

        golden_ref_path = Path(__file__).parent / "golden_ref" / "support.json"
        with open(golden_ref_path, "r", encoding="utf-8") as f:
            golden = json.load(f)

        # Extract anchor types from golden ref
        anchor_types_in_golden = set()
        for anchor in golden["anchor_tuple"]:
            anchor_types_in_golden.add(anchor["anchor_information"]["type"])

        # Config should have file and section
        assert "file" in anchor_types_in_golden
        assert "section" in anchor_types_in_golden

        # Config anchor_types should match
        assert set(config.anchor_types) == anchor_types_in_golden

    def test_matches_golden_ref_gene_rule_target_format(self):
        """测试 config gene_rule.target 格式与 golden ref 一致"""
        config = make_support_config()

        golden_ref_path = Path(__file__).parent / "golden_ref" / "support.json"
        with open(golden_ref_path, "r", encoding="utf-8") as f:
            golden = json.load(f)

        # Check golden ref gene_rule.target format
        for anchor in golden["anchor_tuple"]:
            for evidence in anchor["evidence"]:
                target = evidence["gene_rule"]["target"]
                # Each target item should be a 1-element list
                for target_item in target:
                    assert len(target_item) == 1
                    assert target_item[0] == "support_matrix"

        # Config targets should also be 1-element lists
        for anchor_type, mappings in config.gene_rule_mapping.items():
            for pattern, rule in mappings.items():
                for target_item in rule["target"]:
                    assert len(target_item) == 1


class TestSupportConfigSummaryKey:
    """Tests that support output preserves summary top-level key."""

    def test_golden_ref_has_summary_key(self):
        """测试 golden ref 包含 summary 顶层键"""
        golden_ref_path = Path(__file__).parent / "golden_ref" / "support.json"
        with open(golden_ref_path, "r", encoding="utf-8") as f:
            golden = json.load(f)

        # Golden ref has summary key (for backward compatibility)
        assert "summary" in golden

        # Summary should contain type counts
        assert "type1_section_count" in golden["summary"]
        assert "type2_section_count" in golden["summary"]
        assert "type3_section_count" in golden["summary"]

    def test_golden_ref_summary_counts_match_section_patterns(self):
        """测试 golden ref summary counts 与 section patterns 对应"""
        golden_ref_path = Path(__file__).parent / "golden_ref" / "support.json"
        with open(golden_ref_path, "r", encoding="utf-8") as f:
            golden = json.load(f)

        # Config has TYPE1/TYPE2/TYPE3 patterns
        config = make_support_config()
        section_patterns = list(config.gene_rule_mapping["section"].keys())

        # Golden ref summary should have counts for these types
        summary = golden["summary"]
        for pattern in section_patterns:
            count_key = f"{pattern.lower()}_section_count"
            assert count_key in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
