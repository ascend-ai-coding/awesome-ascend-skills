# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for sub_function_hash DocPartConfig factory function.

Verifies:
- Factory function returns correct DocPartConfig
- ALL 4 anchor types present (file, function, note, section)
- 2-element gene_rule target format [[func_id, field]]
- SECTION_MAPPING entries correct
- Dynamic errorcode generation based on sub_function_count
- IdConfig settings correct
- sub_part_template structure correct
"""

import pytest
from src.snapshot_construction.anchor_construction.doc_parts.sub_function_hash_config import (
    make_sub_function_hash_config,
    SECTION_MAPPING,
)
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    DocPartConfig,
)
from src.snapshot_construction.anchor_construction.common.id_generator import IdConfig


class TestSubFunctionHashConfigFactory:
    """测试 make_sub_function_hash_config 工厂函数"""

    def test_factory_returns_docpartconfig(self):
        """验证工厂函数返回 DocPartConfig 实例"""
        config = make_sub_function_hash_config()
        assert isinstance(config, DocPartConfig)

    def test_doc_part_name(self):
        """验证 doc_part 名称正确"""
        config = make_sub_function_hash_config()
        assert config.doc_part == "sub_function_hash"

    def test_all_4_anchor_types_present(self):
        """验证包含全部 4 种锚点类型"""
        config = make_sub_function_hash_config()
        assert config.anchor_types == ["file", "function", "note", "section"]
        assert len(config.anchor_types) == 4

    def test_output_filename(self):
        """验证输出文件名正确"""
        config = make_sub_function_hash_config()
        assert config.output_filename == "sub_function_hash.json"

    def test_id_config_settings(self):
        """验证 IdConfig 设置正确"""
        config = make_sub_function_hash_config()
        assert isinstance(config.id_config, IdConfig)
        assert config.id_config.type_prefix_in_anchor
        assert config.id_config.type_prefix_in_evidence
        assert config.id_config.category_offsets is None
        assert config.id_config.shared_counters

    def test_sub_part_template_structure(self):
        """验证 sub_part_template 结构正确"""
        config = make_sub_function_hash_config()
        assert config.sub_part_template == {
            "param_matrix": {},
            "fixconfig_1": {},
            "fixconfig_2": {},
            "changeconfig": {},
            "errorcode_matrix": {},
        }
        assert "param_matrix" in config.sub_part_template
        assert "fixconfig_1" in config.sub_part_template
        assert "fixconfig_2" in config.sub_part_template
        assert "changeconfig" in config.sub_part_template
        assert "errorcode_matrix" in config.sub_part_template

    def test_gene_rule_mapping_has_all_anchor_types(self):
        """验证 gene_rule_mapping 包含所有锚点类型"""
        config = make_sub_function_hash_config()
        assert "file" in config.gene_rule_mapping
        assert "function" in config.gene_rule_mapping
        assert "note" in config.gene_rule_mapping
        assert "section" in config.gene_rule_mapping


class TestFileAnchorTargets:
    """测试 file 锚点的 gene_rule targets"""

    def test_file_anchor_has_5_patterns(self):
        """验证 file 锚点有 5 个文件模式"""
        config = make_sub_function_hash_config()
        file_patterns = config.gene_rule_mapping["file"]
        assert len(file_patterns) == 5
        assert "aclnn_cpp" in file_patterns
        assert "aclnn_h" in file_patterns
        assert "proto_h" in file_patterns
        assert "tiling_h" in file_patterns
        assert "def_h" in file_patterns

    def test_aclnn_cpp_targets_2_element_format(self):
        """验证 aclnn_cpp targets 为 2 元素格式"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["file"]["aclnn_cpp"]["target"]
        assert len(targets) == 8
        for target in targets:
            assert len(target) == 2
            assert isinstance(target[0], str)
            assert isinstance(target[1], str)

    def test_aclnn_cpp_targets_fields(self):
        """验证 aclnn_cpp targets 包含所有字段"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["file"]["aclnn_cpp"]["target"]
        fields = [target[1] for target in targets]
        expected_fields = [
            "des",
            "inout",
            "usage",
            "dtype",
            "dmat",
            "shape",
            "tensor",
            "errorcode",
        ]
        assert fields == expected_fields

    def test_aclnn_h_targets_2_element_format(self):
        """验证 aclnn_h targets 为 2 元素格式"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["file"]["aclnn_h"]["target"]
        assert len(targets) == 6
        for target in targets:
            assert len(target) == 2

    def test_aclnn_h_targets_fields(self):
        """验证 aclnn_h targets 包含正确字段"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["file"]["aclnn_h"]["target"]
        fields = [target[1] for target in targets]
        expected_fields = ["des", "inout", "usage", "dtype", "dmat", "shape"]
        assert fields == expected_fields

    def test_proto_h_targets_2_element_format(self):
        """验证 proto_h targets 为 2 元素格式"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["file"]["proto_h"]["target"]
        assert len(targets) == 5
        for target in targets:
            assert len(target) == 2

    def test_proto_h_targets_fields(self):
        """验证 proto_h targets 包含正确字段"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["file"]["proto_h"]["target"]
        fields = [target[1] for target in targets]
        expected_fields = ["des", "inout", "usage", "dtype", "dmat"]
        assert fields == expected_fields

    def test_tiling_h_targets_2_element_format(self):
        """验证 tiling_h targets 为 2 元素格式"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["file"]["tiling_h"]["target"]
        assert len(targets) == 3
        for target in targets:
            assert len(target) == 2

    def test_tiling_h_targets_fields(self):
        """验证 tiling_h targets 包含正确字段"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["file"]["tiling_h"]["target"]
        fields = [target[1] for target in targets]
        expected_fields = ["dtype", "dmat", "shape"]
        assert fields == expected_fields

    def test_def_h_targets_2_element_format(self):
        """验证 def_h targets 为 2 元素格式"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["file"]["def_h"]["target"]
        assert len(targets) == 3
        for target in targets:
            assert len(target) == 2

    def test_def_h_targets_fields(self):
        """验证 def_h targets 包含正确字段"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["file"]["def_h"]["target"]
        fields = [target[1] for target in targets]
        expected_fields = ["dtype", "dmat", "tensor"]
        assert fields == expected_fields

    def test_file_targets_use_placeholders(self):
        """验证 file targets 使用占位符 {sub_function_hash} 和 {sub_error_code}"""
        config = make_sub_function_hash_config()
        aclnn_cpp_targets = config.gene_rule_mapping["file"]["aclnn_cpp"]["target"]
        # 前7个使用 {sub_function_hash}
        for i in range(7):
            assert aclnn_cpp_targets[i][0] == "{sub_function_hash}"
        # 最后一个使用 {sub_error_code}
        assert aclnn_cpp_targets[7][0] == "{sub_error_code}"


class TestFunctionAnchorTargets:
    """测试 function 锚点的 gene_rule targets"""

    def test_function_anchor_has_default_pattern(self):
        """验证 function 锚点有 default 模式"""
        config = make_sub_function_hash_config()
        assert "default" in config.gene_rule_mapping["function"]

    def test_function_targets_2_element_format(self):
        """验证 function targets 为 2 元素格式"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["function"]["default"]["target"]
        assert len(targets) == 1
        assert len(targets[0]) == 2

    def test_function_targets_use_placeholders(self):
        """验证 function targets 使用占位符"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["function"]["default"]["target"]
        assert targets[0][0] == "{sub_function_hash}"
        assert targets[0][1] == "{source_field}"


class TestNoteAnchorTargets:
    """测试 note 锚点的 gene_rule targets"""

    def test_note_anchor_has_default_pattern(self):
        """验证 note 锚点有 default 模式"""
        config = make_sub_function_hash_config()
        assert "default" in config.gene_rule_mapping["note"]

    def test_note_targets_2_element_format(self):
        """验证 note targets 为 2 元素格式"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["note"]["default"]["target"]
        assert len(targets) == 1
        assert len(targets[0]) == 2

    def test_note_targets_use_placeholders(self):
        """验证 note targets 使用占位符"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["note"]["default"]["target"]
        assert targets[0][0] == "{sub_function_hash}"
        assert targets[0][1] == "{source_field}"


class TestSectionAnchorTargets:
    """测试 section 锚点的 gene_rule targets"""

    def test_section_anchor_has_6_patterns(self):
        """验证 section 锚点有 6 个模式"""
        config = make_sub_function_hash_config()
        section_patterns = config.gene_rule_mapping["section"]
        assert len(section_patterns) == 6
        assert "SUPPORT_LIST" in section_patterns
        assert "check_dtype" in section_patterns
        assert "CheckFormat" in section_patterns
        assert "CheckShape" in section_patterns
        assert "continuous" in section_patterns
        assert "CheckParams" in section_patterns

    def test_section_targets_2_element_format(self):
        """验证 section targets 为 2 元素格式"""
        config = make_sub_function_hash_config()
        for pattern_name, pattern_config in config.gene_rule_mapping["section"].items():
            targets = pattern_config["target"]
            for target in targets:
                assert len(target) == 2

    def test_support_list_targets_dtype(self):
        """验证 SUPPORT_LIST targets 映射到 dtype"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["section"]["SUPPORT_LIST"]["target"]
        assert len(targets) == 1
        assert targets[0][1] == "dtype"

    def test_check_dtype_targets_dtype(self):
        """验证 check_dtype targets 映射到 dtype"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["section"]["check_dtype"]["target"]
        assert len(targets) == 1
        assert targets[0][1] == "dtype"

    def test_checkformat_targets_dmat(self):
        """验证 CheckFormat targets 映射到 dmat"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["section"]["CheckFormat"]["target"]
        assert len(targets) == 1
        assert targets[0][1] == "dmat"

    def test_checkshape_targets_shape(self):
        """验证 CheckShape targets 映射到 shape"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["section"]["CheckShape"]["target"]
        assert len(targets) == 1
        assert targets[0][1] == "shape"

    def test_continuous_targets_tensor(self):
        """验证 continuous targets 映射到 tensor"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["section"]["continuous"]["target"]
        assert len(targets) == 1
        assert targets[0][1] == "tensor"

    def test_section_targets_use_placeholder(self):
        """验证 section targets 使用占位符 {sub_function_hash}"""
        config = make_sub_function_hash_config()
        for pattern_name in [
            "SUPPORT_LIST",
            "check_dtype",
            "CheckFormat",
            "CheckShape",
            "continuous",
        ]:
            targets = config.gene_rule_mapping["section"][pattern_name]["target"]
            assert targets[0][0] == "{sub_function_hash}"


class TestDynamicErrorcodeGeneration:
    """测试动态 errorcode 生成"""

    def test_default_sub_function_count_2(self):
        """验证默认 sub_function_count=2 生成 2 个 errorcode targets"""
        config = make_sub_function_hash_config()
        targets = config.gene_rule_mapping["section"]["CheckParams"]["target"]
        assert len(targets) == 2
        assert targets[0] == ["sub_function_1_errorcode", "errorcode"]
        assert targets[1] == ["sub_function_2_errorcode", "errorcode"]

    def test_sub_function_count_3(self):
        """验证 sub_function_count=3 生成 3 个 errorcode targets"""
        config = make_sub_function_hash_config(sub_function_count=3)
        targets = config.gene_rule_mapping["section"]["CheckParams"]["target"]
        assert len(targets) == 3
        assert targets[0] == ["sub_function_1_errorcode", "errorcode"]
        assert targets[1] == ["sub_function_2_errorcode", "errorcode"]
        assert targets[2] == ["sub_function_3_errorcode", "errorcode"]

    def test_sub_function_count_1(self):
        """验证 sub_function_count=1 生成 1 个 errorcode target"""
        config = make_sub_function_hash_config(sub_function_count=1)
        targets = config.gene_rule_mapping["section"]["CheckParams"]["target"]
        assert len(targets) == 1
        assert targets[0] == ["sub_function_1_errorcode", "errorcode"]

    def test_sub_function_count_5(self):
        """验证 sub_function_count=5 生成 5 个 errorcode targets"""
        config = make_sub_function_hash_config(sub_function_count=5)
        targets = config.gene_rule_mapping["section"]["CheckParams"]["target"]
        assert len(targets) == 5
        for i in range(1, 6):
            assert targets[i - 1] == [f"sub_function_{i}_errorcode", "errorcode"]

    def test_errorcode_targets_2_element_format(self):
        """验证 errorcode targets 为 2 元素格式"""
        config = make_sub_function_hash_config(sub_function_count=4)
        targets = config.gene_rule_mapping["section"]["CheckParams"]["target"]
        for target in targets:
            assert len(target) == 2
            assert target[1] == "errorcode"


class TestSectionMapping:
    """测试 SECTION_MAPPING 常量"""

    def test_section_mapping_has_6_entries(self):
        """验证 SECTION_MAPPING 有 6 个条目"""
        assert len(SECTION_MAPPING) == 6

    def test_support_list_mapping(self):
        """验证 SUPPORT_LIST 映射到 dtype"""
        assert SECTION_MAPPING["SUPPORT_LIST"] == "dtype"

    def test_check_dtype_mapping(self):
        """验证 check_dtype 映射到 dtype"""
        assert SECTION_MAPPING["check_dtype"] == "dtype"

    def test_checkformat_mapping(self):
        """验证 CheckFormat 映射到 dmat"""
        assert SECTION_MAPPING["CheckFormat"] == "dmat"

    def test_checkshape_mapping(self):
        """验证 CheckShape 映射到 shape"""
        assert SECTION_MAPPING["CheckShape"] == "shape"

    def test_continuous_mapping(self):
        """验证 continuous 映射到 tensor"""
        assert SECTION_MAPPING["continuous"] == "tensor"

    def test_checkparams_mapping(self):
        """验证 CheckParams 映射到 errorcode"""
        assert SECTION_MAPPING["CheckParams"] == "errorcode"

    def test_section_mapping_matches_config(self):
        """验证 SECTION_MAPPING 与 config 中的 section targets 一致"""
        config = make_sub_function_hash_config()
        for section_key, field in SECTION_MAPPING.items():
            if section_key != "CheckParams":  # CheckParams 是动态生成的
                targets = config.gene_rule_mapping["section"][section_key]["target"]
                assert targets[0][1] == field


class TestDeterValues:
    """测试 deter 值"""

    def test_file_deter_values_empty(self):
        """验证 file 锚点的 deter 值为空字符串"""
        config = make_sub_function_hash_config()
        for pattern_name, pattern_config in config.gene_rule_mapping["file"].items():
            assert pattern_config["deter"] == ""

    def test_function_deter_value_empty(self):
        """验证 function 锚点的 deter 值为空字符串"""
        config = make_sub_function_hash_config()
        assert config.gene_rule_mapping["function"]["default"]["deter"] == ""

    def test_note_deter_value_empty(self):
        """验证 note 锚点的 deter 值为空字符串"""
        config = make_sub_function_hash_config()
        assert config.gene_rule_mapping["note"]["default"]["deter"] == ""

    def test_section_deter_values_empty(self):
        """验证 section 锚点的 deter 值为空字符串"""
        config = make_sub_function_hash_config()
        for pattern_name, pattern_config in config.gene_rule_mapping["section"].items():
            assert pattern_config["deter"] == ""


class TestBackwardCompatibility:
    """测试向后兼容性"""

    def test_default_config_matches_registry(self):
        """验证默认配置与 gene_rule_registry.py 中的配置一致（除了动态 errorcode）"""
        from src.snapshot_construction.anchor_construction.gene_rule_registry import (
            GeneRuleRegistry,
        )

        registry = GeneRuleRegistry()
        registry_config = registry.get_config("sub_function_hash")
        factory_config = make_sub_function_hash_config()

        # 验证基本属性一致
        assert factory_config.doc_part == registry_config.doc_part
        assert factory_config.anchor_types == registry_config.anchor_types
        assert factory_config.output_filename == registry_config.output_filename
        assert factory_config.sub_part_template == registry_config.sub_part_template

        # 验证 IdConfig 一致
        assert (
            factory_config.id_config.type_prefix_in_anchor
            == registry_config.id_config.type_prefix_in_anchor
        )
        assert (
            factory_config.id_config.type_prefix_in_evidence
            == registry_config.id_config.type_prefix_in_evidence
        )
        assert (
            factory_config.id_config.category_offsets
            == registry_config.id_config.category_offsets
        )
        assert (
            factory_config.id_config.shared_counters
            == registry_config.id_config.shared_counters
        )

        # 验证 gene_rule_mapping 结构一致（除了 CheckParams）
        assert set(factory_config.gene_rule_mapping.keys()) == set(
            registry_config.gene_rule_mapping.keys()
        )

        # 验证 file anchors 一致
        assert set(factory_config.gene_rule_mapping["file"].keys()) == set(
            registry_config.gene_rule_mapping["file"].keys()
        )

        # 验证 function anchors 一致
        assert (
            factory_config.gene_rule_mapping["function"]["default"]["target"]
            == registry_config.gene_rule_mapping["function"]["default"]["target"]
        )

        # 验证 note anchors 一致
        assert (
            factory_config.gene_rule_mapping["note"]["default"]["target"]
            == registry_config.gene_rule_mapping["note"]["default"]["target"]
        )

        # 验证 section anchors（除了 CheckParams）一致
        for section_key in [
            "SUPPORT_LIST",
            "check_dtype",
            "CheckFormat",
            "CheckShape",
            "continuous",
        ]:
            assert (
                factory_config.gene_rule_mapping["section"][section_key]
                == registry_config.gene_rule_mapping["section"][section_key]
            )

    def test_default_errorcode_matches_registry(self):
        """验证默认 sub_function_count=2 的 errorcode targets 与 registry 一致"""
        from src.snapshot_construction.anchor_construction.gene_rule_registry import (
            GeneRuleRegistry,
        )

        registry = GeneRuleRegistry()
        registry_config = registry.get_config("sub_function_hash")
        factory_config = make_sub_function_hash_config()

        # registry 中硬编码了 2 个 errorcode targets
        registry_errorcode = registry_config.gene_rule_mapping["section"][
            "CheckParams"
        ]["target"]
        factory_errorcode = factory_config.gene_rule_mapping["section"]["CheckParams"][
            "target"
        ]

        assert factory_errorcode == registry_errorcode
        assert len(factory_errorcode) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
