# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for IdGenerator and IdConfig classes.

Tests all 4 existing ID formats:
- constraint/sub_function: type_prefix_in_anchor=True, type_prefix_in_evidence=True
- support: category_offsets with TYPE1→1, TYPE2→101, TYPE3→201
- summary/function_proto: type_prefix_in_anchor=True, type_prefix_in_evidence=False
"""

import pytest
from src.snapshot_construction.anchor_construction.common.id_generator import (
    IdGenerator,
    IdConfig,
)


class TestIdConfig:
    """测试 IdConfig 配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = IdConfig()
        assert config.type_prefix_in_anchor is True
        assert config.type_prefix_in_evidence is True
        assert config.category_offsets is None
        assert config.shared_counters is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            category_offsets={"TYPE1": 1, "TYPE2": 101},
            shared_counters=False,
        )
        assert config.type_prefix_in_anchor is True
        assert config.type_prefix_in_evidence is False
        assert config.category_offsets == {"TYPE1": 1, "TYPE2": 101}
        assert config.shared_counters is False


class TestSubFunctionFormat:
    """测试 sub_function 格式：type_prefix_in_anchor=True, type_prefix_in_evidence=True, shared_counters=True

    预期格式：
    - anchor_file_001, ev_file_001
    - anchor_function_001, ev_function_001
    - anchor_note_001, ev_note_001
    - anchor_section_001, ev_section_001
    """

    def test_file_anchor_format(self):
        """测试 file 类型锚点 ID 格式"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("file")
        assert anchor_id == "anchor_file_001"
        assert evidence_id == "ev_file_001"

        anchor_id, evidence_id = gen.next_pair("file")
        assert anchor_id == "anchor_file_002"
        assert evidence_id == "ev_file_002"

    def test_function_anchor_format(self):
        """测试 function 类型锚点 ID 格式"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("function")
        assert anchor_id == "anchor_function_001"
        assert evidence_id == "ev_function_001"

    def test_note_anchor_format(self):
        """测试 note 类型锚点 ID 格式"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("note")
        assert anchor_id == "anchor_note_001"
        assert evidence_id == "ev_note_001"

    def test_section_anchor_format(self):
        """测试 section 类型锚点 ID 格式"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("section")
        assert anchor_id == "anchor_section_001"
        assert evidence_id == "ev_section_001"

    def test_sequential_increment(self):
        """测试 ID 序列递增"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        gen = IdGenerator(config)

        for i in range(1, 11):
            anchor_id, evidence_id = gen.next_pair("file")
            assert anchor_id == f"anchor_file_{i:03d}"
            assert evidence_id == f"ev_file_{i:03d}"


class TestSupportFormat:
    """测试 support 格式：category_offsets, shared_counters=False

    预期格式：
    - TYPE1: anchor_section_001, ev_001 (offset=1)
    - TYPE2: anchor_section_101, ev_101 (offset=101)
    - TYPE3: anchor_section_201, ev_201 (offset=201)
    """

    def test_type1_offset(self):
        """测试 TYPE1 类别偏移（起始值 1）"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            category_offsets={"TYPE1": 1, "TYPE2": 101, "TYPE3": 201},
            shared_counters=False,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("section", "TYPE1")
        assert anchor_id == "anchor_section_001"
        assert evidence_id == "ev_001"

        anchor_id, evidence_id = gen.next_pair("section", "TYPE1")
        assert anchor_id == "anchor_section_002"
        assert evidence_id == "ev_002"

    def test_type2_offset(self):
        """测试 TYPE2 类别偏移（起始值 101）"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            category_offsets={"TYPE1": 1, "TYPE2": 101, "TYPE3": 201},
            shared_counters=False,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("section", "TYPE2")
        assert anchor_id == "anchor_section_101"
        assert evidence_id == "ev_101"

        anchor_id, evidence_id = gen.next_pair("section", "TYPE2")
        assert anchor_id == "anchor_section_102"
        assert evidence_id == "ev_102"

    def test_type3_offset(self):
        """测试 TYPE3 类别偏移（起始值 201）"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            category_offsets={"TYPE1": 1, "TYPE2": 101, "TYPE3": 201},
            shared_counters=False,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("section", "TYPE3")
        assert anchor_id == "anchor_section_201"
        assert evidence_id == "ev_201"

        anchor_id, evidence_id = gen.next_pair("section", "TYPE3")
        assert anchor_id == "anchor_section_202"
        assert evidence_id == "ev_202"

    def test_mixed_categories(self):
        """测试混合类别使用"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            category_offsets={"TYPE1": 1, "TYPE2": 101, "TYPE3": 201},
            shared_counters=False,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("section", "TYPE1")
        assert anchor_id == "anchor_section_001"
        assert evidence_id == "ev_001"

        anchor_id, evidence_id = gen.next_pair("section", "TYPE2")
        assert anchor_id == "anchor_section_101"
        assert evidence_id == "ev_101"

        anchor_id, evidence_id = gen.next_pair("section", "TYPE3")
        assert anchor_id == "anchor_section_201"
        assert evidence_id == "ev_201"

        anchor_id, evidence_id = gen.next_pair("section", "TYPE1")
        assert anchor_id == "anchor_section_002"
        assert evidence_id == "ev_002"

        anchor_id, evidence_id = gen.next_pair("section", "TYPE2")
        assert anchor_id == "anchor_section_102"
        assert evidence_id == "ev_102"

    def test_no_type_prefix_in_evidence(self):
        """测试 evidence ID 不包含类型前缀"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            category_offsets={"TYPE1": 1},
            shared_counters=False,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("section", "TYPE1")
        assert anchor_id == "anchor_section_001"
        assert evidence_id == "ev_001"

        anchor_id, evidence_id = gen.next_pair("file", "TYPE1")
        assert anchor_id == "anchor_file_002"
        assert evidence_id == "ev_002"


class TestSummaryFunctionProtoFormat:
    """测试 summary/function_proto 格式：type_prefix_in_anchor=True, type_prefix_in_evidence=False, shared_counters=False

    预期格式：
    - anchor_file_001, ev_001
    - anchor_function_001, ev_001
    """

    def test_file_anchor_with_type_prefix(self):
        """测试 file 类型锚点 ID 包含类型前缀"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            shared_counters=False,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("file")
        assert anchor_id == "anchor_file_001"
        assert evidence_id == "ev_001"

    def test_function_anchor_with_type_prefix(self):
        """测试 function 类型锚点 ID 包含类型前缀"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            shared_counters=False,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("function")
        assert anchor_id == "anchor_function_001"
        assert evidence_id == "ev_001"

    def test_sequential_ids(self):
        """测试序列 ID 递增"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            shared_counters=False,
        )
        gen = IdGenerator(config)

        for i in range(1, 6):
            anchor_id, evidence_id = gen.next_pair("file")
            assert anchor_id == f"anchor_file_{i:03d}"
            assert evidence_id == f"ev_{i:03d}"


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_anchor_type(self):
        """测试空类型参数"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("")
        assert anchor_id == "anchor_001"
        assert evidence_id == "ev_001"

    def test_no_type_prefix_in_anchor(self):
        """测试 anchor ID 不包含类型前缀"""
        config = IdConfig(
            type_prefix_in_anchor=False,
            type_prefix_in_evidence=False,
            shared_counters=True,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("file")
        assert anchor_id == "anchor_001"
        assert evidence_id == "ev_001"

    def test_reset_counter(self):
        """测试重置计数器"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        gen = IdGenerator(config)

        gen.next_pair("file")
        gen.next_pair("file")

        gen.reset()

        anchor_id, evidence_id = gen.next_pair("file")
        assert anchor_id == "anchor_file_001"
        assert evidence_id == "ev_file_001"

    def test_reset_counter_with_category(self):
        """测试重置类别计数器"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            category_offsets={"TYPE1": 1, "TYPE2": 101},
            shared_counters=False,
        )
        gen = IdGenerator(config)

        gen.next_pair("section", "TYPE1")
        gen.next_pair("section", "TYPE1")

        gen.reset("TYPE1")

        anchor_id, evidence_id = gen.next_pair("section", "TYPE1")
        assert anchor_id == "anchor_section_001"
        assert evidence_id == "ev_001"

    def test_large_sequence_numbers(self):
        """测试大序列号格式化"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        gen = IdGenerator(config)

        gen._counters["_default_"] = 999

        anchor_id, evidence_id = gen.next_pair("file")
        assert anchor_id == "anchor_file_1000"
        assert evidence_id == "ev_file_1000"

    def test_category_without_offset(self):
        """测试未配置偏移的类别"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            category_offsets={"TYPE1": 1},
            shared_counters=False,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("section", "TYPE2")
        assert anchor_id == "anchor_section_001"
        assert evidence_id == "ev_001"


class TestIntegration:
    """集成测试：验证所有格式可以正确重现"""

    def test_reproduce_constraint_format(self):
        """重现 constraint 格式"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("file")
        assert anchor_id == "anchor_file_001"
        assert evidence_id == "ev_file_001"

        anchor_id, evidence_id = gen.next_pair("section")
        assert anchor_id == "anchor_section_002"
        assert evidence_id == "ev_section_002"

    def test_reproduce_support_format(self):
        """重现 support 格式"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            category_offsets={"TYPE1": 1, "TYPE2": 101, "TYPE3": 201},
            shared_counters=False,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("section", "TYPE1")
        assert anchor_id == "anchor_section_001"
        assert evidence_id == "ev_001"

        anchor_id, evidence_id = gen.next_pair("section", "TYPE2")
        assert anchor_id == "anchor_section_101"
        assert evidence_id == "ev_101"

        anchor_id, evidence_id = gen.next_pair("section", "TYPE3")
        assert anchor_id == "anchor_section_201"
        assert evidence_id == "ev_201"

    def test_reproduce_summary_function_proto_format(self):
        """重现 summary/function_proto 格式"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            shared_counters=False,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("file")
        assert anchor_id == "anchor_file_001"
        assert evidence_id == "ev_001"

        anchor_id, evidence_id = gen.next_pair("function")
        assert anchor_id == "anchor_function_002"
        assert evidence_id == "ev_002"

    def test_reproduce_sub_function_format(self):
        """重现 sub_function 格式"""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        gen = IdGenerator(config)

        anchor_id, evidence_id = gen.next_pair("file")
        assert anchor_id == "anchor_file_001"
        assert evidence_id == "ev_file_001"

        anchor_id, evidence_id = gen.next_pair("function")
        assert anchor_id == "anchor_function_002"
        assert evidence_id == "ev_function_002"

        anchor_id, evidence_id = gen.next_pair("note")
        assert anchor_id == "anchor_note_003"
        assert evidence_id == "ev_note_003"
