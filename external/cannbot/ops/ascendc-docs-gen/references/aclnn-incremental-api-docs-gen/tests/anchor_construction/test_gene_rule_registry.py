# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for GeneRuleRegistry and DocPartConfig.

Validates:
- DocPartConfig dataclass structure
- GeneRuleRegistry initialization with 5 default configs
- get_config() and get_all_configs() methods
- gene_rule_mapping structure per doc_part
- IdConfig integration
- Singleton pattern
"""

import pytest
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    DocPartConfig,
    GeneRuleRegistry,
    get_registry,
)
from src.snapshot_construction.anchor_construction.common.id_generator import IdConfig


class TestDocPartConfig:
    """Tests for DocPartConfig dataclass."""

    def test_doc_part_config_creation(self):
        """测试 DocPartConfig 基本创建"""
        config = DocPartConfig(
            doc_part="test",
            anchor_types=["file", "section"],
            gene_rule_mapping={
                "file": {"default": {"deter": "", "target": [["test_target"]]}}
            },
            id_config=IdConfig(),
            output_filename="test.json",
            sub_part_template={"test_part": []},
        )

        assert config.doc_part == "test"
        assert config.anchor_types == ["file", "section"]
        assert "file" in config.gene_rule_mapping
        assert config.output_filename == "test.json"
        assert "test_part" in config.sub_part_template

    def test_doc_part_config_required_fields(self):
        """测试 DocPartConfig 必需字段"""
        # Should raise error if missing required fields
        with pytest.raises(TypeError):
            DocPartConfig()  # Missing all required fields

        with pytest.raises(TypeError):
            DocPartConfig(doc_part="test")  # Missing other required fields


class TestGeneRuleRegistry:
    """Tests for GeneRuleRegistry class."""

    def test_registry_initialization(self):
        """测试 GeneRuleRegistry 初始化包含5个默认配置"""
        registry = GeneRuleRegistry()
        configs = registry.get_all_configs()

        assert len(configs) == 5
        assert "constraint" in configs
        assert "support" in configs
        assert "summary" in configs
        assert "function_proto" in configs
        assert "sub_function_hash" in configs

    def test_get_config_valid_doc_part(self):
        """测试 get_config() 获取有效 doc_part"""
        registry = GeneRuleRegistry()

        constraint_config = registry.get_config("constraint")
        assert constraint_config.doc_part == "constraint"
        assert constraint_config.anchor_types == ["file", "section"]
        assert constraint_config.output_filename == "constraint.json"

    def test_get_config_invalid_doc_part(self):
        """测试 get_config() 无效 doc_part 抛出 KeyError"""
        registry = GeneRuleRegistry()

        with pytest.raises(KeyError) as exc_info:
            registry.get_config("invalid_doc_part")

        assert "invalid_doc_part" in str(exc_info.value)

    def test_get_all_configs_returns_copy(self):
        """测试 get_all_configs() 返回副本而非原字典"""
        registry = GeneRuleRegistry()
        configs1 = registry.get_all_configs()
        configs2 = registry.get_all_configs()

        # Should be equal but not the same object
        assert configs1 == configs2
        assert configs1 is not configs2

    def test_register_custom_config(self):
        """测试 register_config() 注册自定义配置"""
        registry = GeneRuleRegistry()

        custom_config = DocPartConfig(
            doc_part="custom",
            anchor_types=["file"],
            gene_rule_mapping={
                "file": {"default": {"deter": "", "target": [["custom_target"]]}}
            },
            id_config=IdConfig(),
            output_filename="custom.json",
            sub_part_template={"custom_part": []},
        )

        registry.register_config(custom_config)

        # Should now have 6 configs
        configs = registry.get_all_configs()
        assert len(configs) == 6
        assert "custom" in configs

        # Retrieved config should match registered
        retrieved = registry.get_config("custom")
        assert retrieved.doc_part == "custom"
        assert retrieved.output_filename == "custom.json"


class TestConstraintConfig:
    """Tests for constraint doc_part configuration."""

    def test_constraint_anchor_types(self):
        """测试 constraint anchor_types 包含 file 和 section"""
        registry = GeneRuleRegistry()
        config = registry.get_config("constraint")

        assert "file" in config.anchor_types
        assert "section" in config.anchor_types
        assert len(config.anchor_types) == 2

    def test_constraint_gene_rule_mapping_file(self):
        """测试 constraint file gene_rule_mapping 结构"""
        registry = GeneRuleRegistry()
        config = registry.get_config("constraint")

        file_mapping = config.gene_rule_mapping["file"]

        # Should have 4 file patterns
        assert "op_api_aclnn_h" in file_mapping
        assert "op_host_def_cpp" in file_mapping
        assert "op_host_tiling_cpp" in file_mapping
        assert "op_kernel_cpp_h" in file_mapping

        # Check target format (1-element lists)
        op_api_target = file_mapping["op_api_aclnn_h"]["target"]
        assert len(op_api_target) == 3
        assert op_api_target[0] == ["deter_describe"]
        assert op_api_target[1] == ["constrain_model_scene"]
        assert op_api_target[2] == ["constrain_model_scene_clarify"]

    def test_constraint_gene_rule_mapping_section(self):
        """测试 constraint section gene_rule_mapping 结构"""
        registry = GeneRuleRegistry()
        config = registry.get_config("constraint")

        section_mapping = config.gene_rule_mapping["section"]

        # Should have 2 section patterns
        assert "op_host_tiling_cpp" in section_mapping
        assert "op_kernel_cpp_h" in section_mapping

        # Check target format
        target = section_mapping["op_host_tiling_cpp"]["target"]
        assert len(target) == 2
        assert target[0] == ["constrain_model_scene"]
        assert target[1] == ["constrain_model_scene_clarify"]

    def test_constraint_id_config(self):
        """测试 constraint IdConfig 配置"""
        registry = GeneRuleRegistry()
        config = registry.get_config("constraint")

        assert config.id_config.type_prefix_in_anchor
        assert config.id_config.type_prefix_in_evidence
        assert config.id_config.shared_counters
        assert config.id_config.category_offsets is None

    def test_constraint_sub_part_template(self):
        """测试 constraint sub_part_template 结构"""
        registry = GeneRuleRegistry()
        config = registry.get_config("constraint")

        assert "constrain_clarify" in config.sub_part_template
        assert isinstance(config.sub_part_template["constrain_clarify"], list)
        assert len(config.sub_part_template["constrain_clarify"]) == 1

        # Check template structure
        template_item = config.sub_part_template["constrain_clarify"][0]
        assert "content" in template_item
        assert "evidence_id" in template_item


class TestSupportConfig:
    """Tests for support doc_part configuration."""

    def test_support_anchor_types(self):
        """测试 support anchor_types 包含 file 和 section"""
        registry = GeneRuleRegistry()
        config = registry.get_config("support")

        assert "file" in config.anchor_types
        assert "section" in config.anchor_types
        assert len(config.anchor_types) == 2

    def test_support_gene_rule_mapping_section_types(self):
        """测试 support section gene_rule_mapping 包含 TYPE1/TYPE2/TYPE3"""
        registry = GeneRuleRegistry()
        config = registry.get_config("support")

        section_mapping = config.gene_rule_mapping["section"]

        assert "TYPE1" in section_mapping
        assert "TYPE2" in section_mapping
        assert "TYPE3" in section_mapping

        # All should target support_matrix
        assert section_mapping["TYPE1"]["target"] == [["support_matrix"]]
        assert section_mapping["TYPE2"]["target"] == [["support_matrix"]]
        assert section_mapping["TYPE3"]["target"] == [["support_matrix"]]

    def test_support_id_config_category_offsets(self):
        """测试 support IdConfig category_offsets"""
        registry = GeneRuleRegistry()
        config = registry.get_config("support")

        assert not config.id_config.type_prefix_in_anchor
        assert not config.id_config.type_prefix_in_evidence
        assert not config.id_config.shared_counters

        # Check category offsets
        offsets = config.id_config.category_offsets
        assert offsets is not None
        assert offsets["TYPE1"] == 1
        assert offsets["TYPE2"] == 101
        assert offsets["TYPE3"] == 201

    def test_support_sub_part_template(self):
        """测试 support sub_part_template 结构"""
        registry = GeneRuleRegistry()
        config = registry.get_config("support")

        assert "support_matrix" in config.sub_part_template
        assert isinstance(config.sub_part_template["support_matrix"], list)

        # Check template structure (uses evidence_ids not evidence_id)
        template_item = config.sub_part_template["support_matrix"][0]
        assert "content" in template_item
        assert "evidence_ids" in template_item


class TestSummaryConfig:
    """Tests for summary doc_part configuration."""

    def test_summary_anchor_types(self):
        """测试 summary anchor_types 包含 file 和 note"""
        registry = GeneRuleRegistry()
        config = registry.get_config("summary")

        assert "file" in config.anchor_types
        assert "note" in config.anchor_types
        assert len(config.anchor_types) == 2

    def test_summary_gene_rule_mapping_file_patterns(self):
        """测试 summary file gene_rule_mapping 包含3种文件模式"""
        registry = GeneRuleRegistry()
        config = registry.get_config("summary")

        file_mapping = config.gene_rule_mapping["file"]

        assert "op_api_aclnn_h" in file_mapping
        assert "op_api_aclnn_cpp" in file_mapping
        assert "op_kernel_cpp_h" in file_mapping

        # Check targets
        assert file_mapping["op_api_aclnn_h"]["target"] == [
            ["summary_clarify"],
            ["summary_formula"],
        ]
        assert file_mapping["op_api_aclnn_cpp"]["target"] == [["summary_clarify"]]
        assert file_mapping["op_kernel_cpp_h"]["target"] == [["summary_formula"]]

    def test_summary_id_config(self):
        """测试 summary IdConfig 配置"""
        registry = GeneRuleRegistry()
        config = registry.get_config("summary")

        assert config.id_config.type_prefix_in_anchor
        assert not config.id_config.type_prefix_in_evidence
        assert not config.id_config.shared_counters

    def test_summary_sub_part_template(self):
        """测试 summary sub_part_template 包含两个子部分"""
        registry = GeneRuleRegistry()
        config = registry.get_config("summary")

        assert "summary_clarify" in config.sub_part_template
        assert "summary_formula" in config.sub_part_template

        # Both should be lists with content field
        for key in ["summary_clarify", "summary_formula"]:
            assert isinstance(config.sub_part_template[key], list)
            assert "content" in config.sub_part_template[key][0]


class TestFunctionProtoConfig:
    """Tests for function_proto doc_part configuration."""

    def test_function_proto_anchor_types(self):
        """测试 function_proto anchor_types 包含 file 和 function"""
        registry = GeneRuleRegistry()
        config = registry.get_config("function_proto")

        assert "file" in config.anchor_types
        assert "function" in config.anchor_types
        assert len(config.anchor_types) == 2

    def test_function_proto_gene_rule_mapping(self):
        """测试 function_proto gene_rule_mapping 结构"""
        registry = GeneRuleRegistry()
        config = registry.get_config("function_proto")

        file_mapping = config.gene_rule_mapping["file"]
        function_mapping = config.gene_rule_mapping["function"]

        # File should target both clarify and sign
        assert file_mapping["op_api_aclnn_h"]["target"] == [
            ["function_proto_clarify"],
            ["sub_function_sign"],
        ]

        # Function should target only sign
        assert function_mapping["default"]["target"] == [["sub_function_sign"]]

    def test_function_proto_sub_part_template(self):
        """测试 function_proto sub_part_template 包含两个子部分"""
        registry = GeneRuleRegistry()
        config = registry.get_config("function_proto")

        assert "function_proto_clarify" in config.sub_part_template
        assert "sub_function_sign" in config.sub_part_template


class TestSubFunctionConfig:
    """Tests for sub_function_hash doc_part configuration."""

    def test_sub_function_anchor_types_all_four(self):
        """测试 sub_function_hash anchor_types 包含全部4种类型"""
        registry = GeneRuleRegistry()
        config = registry.get_config("sub_function_hash")

        assert "file" in config.anchor_types
        assert "function" in config.anchor_types
        assert "note" in config.anchor_types
        assert "section" in config.anchor_types
        assert len(config.anchor_types) == 4

    def test_sub_function_gene_rule_mapping_file_patterns(self):
        """测试 sub_function_hash file gene_rule_mapping 包含5种文件模式"""
        registry = GeneRuleRegistry()
        config = registry.get_config("sub_function_hash")

        file_mapping = config.gene_rule_mapping["file"]

        assert "aclnn_cpp" in file_mapping
        assert "aclnn_h" in file_mapping
        assert "proto_h" in file_mapping
        assert "tiling_h" in file_mapping
        assert "def_h" in file_mapping

    def test_sub_function_gene_rule_target_two_elements(self):
        """测试 sub_function_hash gene_rule.target 为2元素列表"""
        registry = GeneRuleRegistry()
        config = registry.get_config("sub_function_hash")

        # File targets should be 2-element lists
        file_mapping = config.gene_rule_mapping["file"]
        aclnn_cpp_target = file_mapping["aclnn_cpp"]["target"]

        # Each target item should be a 2-element list
        for target_item in aclnn_cpp_target:
            assert len(target_item) == 2
            assert isinstance(target_item[0], str)
            assert isinstance(target_item[1], str)

    def test_sub_function_section_mapping(self):
        """测试 sub_function_hash section gene_rule_mapping 包含 SECTION_MAPPING"""
        registry = GeneRuleRegistry()
        config = registry.get_config("sub_function_hash")

        section_mapping = config.gene_rule_mapping["section"]

        # Should match SECTION_MAPPING from merge_section_extract.py
        assert "SUPPORT_LIST" in section_mapping
        assert "check_dtype" in section_mapping
        assert "CheckFormat" in section_mapping
        assert "CheckShape" in section_mapping
        assert "continuous" in section_mapping
        assert "CheckParams" in section_mapping

        # Check dtype mapping
        assert section_mapping["SUPPORT_LIST"]["target"] == [
            ["{sub_function_hash}", "dtype"]
        ]

        # Check errorcode special case (2 targets)
        check_params_target = section_mapping["CheckParams"]["target"]
        assert len(check_params_target) == 2
        assert check_params_target[0] == ["sub_function_1_errorcode", "errorcode"]
        assert check_params_target[1] == ["sub_function_2_errorcode", "errorcode"]

    def test_sub_function_id_config(self):
        """测试 sub_function_hash IdConfig 配置"""
        registry = GeneRuleRegistry()
        config = registry.get_config("sub_function_hash")

        assert config.id_config.type_prefix_in_anchor
        assert config.id_config.type_prefix_in_evidence
        assert config.id_config.shared_counters

    def test_sub_function_sub_part_template(self):
        """测试 sub_function_hash sub_part_template 结构"""
        registry = GeneRuleRegistry()
        config = registry.get_config("sub_function_hash")

        assert "param_matrix" in config.sub_part_template
        assert isinstance(config.sub_part_template["param_matrix"], dict)


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_get_registry_returns_singleton(self):
        """测试 get_registry() 返回单例"""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_singleton_has_all_configs(self):
        """测试单例包含所有5个配置"""
        registry = get_registry()
        configs = registry.get_all_configs()

        assert len(configs) == 5


class TestGeneRuleTargetFormat:
    """Tests for gene_rule.target format validation."""

    def test_constraint_target_one_element(self):
        """测试 constraint target 为1元素列表"""
        registry = GeneRuleRegistry()
        config = registry.get_config("constraint")

        # All targets should be 1-element lists
        for anchor_type, mappings in config.gene_rule_mapping.items():
            for pattern, rule in mappings.items():
                for target_item in rule["target"]:
                    assert len(target_item) == 1
                    assert isinstance(target_item[0], str)

    def test_support_target_one_element(self):
        """测试 support target 为1元素列表"""
        registry = GeneRuleRegistry()
        config = registry.get_config("support")

        # All targets should be 1-element lists
        for anchor_type, mappings in config.gene_rule_mapping.items():
            for pattern, rule in mappings.items():
                for target_item in rule["target"]:
                    assert len(target_item) == 1
                    assert isinstance(target_item[0], str)

    def test_summary_target_one_element(self):
        """测试 summary target 为1元素列表"""
        registry = GeneRuleRegistry()
        config = registry.get_config("summary")

        # All targets should be 1-element lists
        for anchor_type, mappings in config.gene_rule_mapping.items():
            for pattern, rule in mappings.items():
                for target_item in rule["target"]:
                    assert len(target_item) == 1
                    assert isinstance(target_item[0], str)

    def test_function_proto_target_one_element(self):
        """测试 function_proto target 为1元素列表"""
        registry = GeneRuleRegistry()
        config = registry.get_config("function_proto")

        # All targets should be 1-element lists
        for anchor_type, mappings in config.gene_rule_mapping.items():
            for pattern, rule in mappings.items():
                for target_item in rule["target"]:
                    assert len(target_item) == 1
                    assert isinstance(target_item[0], str)

    def test_sub_function_target_two_elements(self):
        """测试 sub_function_hash target 为2元素列表"""
        registry = GeneRuleRegistry()
        config = registry.get_config("sub_function_hash")

        # Most targets should be 2-element lists (except CheckParams which has 2 targets)
        file_mapping = config.gene_rule_mapping["file"]
        for pattern, rule in file_mapping.items():
            for target_item in rule["target"]:
                assert len(target_item) == 2
                assert isinstance(target_item[0], str)
                assert isinstance(target_item[1], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
