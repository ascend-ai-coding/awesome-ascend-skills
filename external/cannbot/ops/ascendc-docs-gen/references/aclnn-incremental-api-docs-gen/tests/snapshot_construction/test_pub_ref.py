# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for PubRefConfig configuration class.

Tests cover:
- DOC_PARTS list structure, ordering, and contents
- Individual DOC_PARTS_* constants matching DOC_PARTS indices
- DOC_PART_MAX_WORKERS setting
- INCREMENTAL_LLM_KEY, INCREMENTAL_LLM_URL, MODEL class attributes
"""

import pytest

from src.snapshot_construction.pub_ref import PubRefConfig


class TestDocParts:
    """测试 DOC_PARTS 列表"""

    def test_doc_parts_is_list(self):
        """测试 DOC_PARTS 是列表类型"""
        assert isinstance(PubRefConfig.DOC_PARTS, list)

    def test_doc_parts_has_five_elements(self):
        """测试 DOC_PARTS 包含 5 个元素"""
        assert len(PubRefConfig.DOC_PARTS) == 5

    def test_doc_parts_contains_expected_values_in_order(self):
        """测试 DOC_PARTS 按顺序包含预期值"""
        expected = [
            "support",
            "summary",
            "function_proto",
            "sub_function_hash",
            "constraint",
        ]
        assert PubRefConfig.DOC_PARTS == expected

    def test_doc_parts_support_is_first(self):
        """测试 DOC_PARTS[0] 是 support"""
        assert PubRefConfig.DOC_PARTS[0] == "support"

    def test_doc_parts_summary_is_second(self):
        """测试 DOC_PARTS[1] 是 summary"""
        assert PubRefConfig.DOC_PARTS[1] == "summary"

    def test_doc_parts_function_proto_is_third(self):
        """测试 DOC_PARTS[2] 是 function_proto"""
        assert PubRefConfig.DOC_PARTS[2] == "function_proto"

    def test_doc_parts_sub_function_hash_is_fourth(self):
        """测试 DOC_PARTS[3] 是 sub_function_hash"""
        assert PubRefConfig.DOC_PARTS[3] == "sub_function_hash"

    def test_doc_parts_constraint_is_fifth(self):
        """测试 DOC_PARTS[4] 是 constraint"""
        assert PubRefConfig.DOC_PARTS[4] == "constraint"


class TestDocPartsConstants:
    """测试 DOC_PARTS_* 常量"""

    def test_doc_parts_support_constant(self):
        """测试 DOC_PARTS_SUPPORT 常量"""
        assert PubRefConfig.DOC_PARTS_SUPPORT == "support"
        assert PubRefConfig.DOC_PARTS_SUPPORT == PubRefConfig.DOC_PARTS[0]

    def test_doc_parts_summary_constant(self):
        """测试 DOC_PARTS_SUMMARY 常量"""
        assert PubRefConfig.DOC_PARTS_SUMMARY == "summary"
        assert PubRefConfig.DOC_PARTS_SUMMARY == PubRefConfig.DOC_PARTS[1]

    def test_doc_parts_function_proto_constant(self):
        """测试 DOC_PARTS_FUNCTION_PROTO 常量"""
        assert PubRefConfig.DOC_PARTS_FUNCTION_PROTO == "function_proto"
        assert PubRefConfig.DOC_PARTS_FUNCTION_PROTO == PubRefConfig.DOC_PARTS[2]

    def test_doc_parts_sub_function_hash_constant(self):
        """测试 DOC_PARTS_SUB_FUNCTION_HASH 常量"""
        assert PubRefConfig.DOC_PARTS_SUB_FUNCTION_HASH == "sub_function_hash"
        assert PubRefConfig.DOC_PARTS_SUB_FUNCTION_HASH == PubRefConfig.DOC_PARTS[3]

    def test_doc_parts_constraint_constant(self):
        """测试 DOC_PARTS_CONSTRAINT 常量"""
        assert PubRefConfig.DOC_PARTS_CONSTRAINT == "constraint"
        assert PubRefConfig.DOC_PARTS_CONSTRAINT == PubRefConfig.DOC_PARTS[4]


class TestDocPartMaxWorkers:
    """测试 DOC_PART_MAX_WORKERS"""

    def test_doc_part_max_workers_is_int(self):
        """测试 DOC_PART_MAX_WORKERS 是整数类型"""
        assert isinstance(PubRefConfig.DOC_PART_MAX_WORKERS, int)

    def test_doc_part_max_workers_value(self):
        """测试 DOC_PART_MAX_WORKERS 值为 2"""
        assert PubRefConfig.DOC_PART_MAX_WORKERS == 2

    def test_doc_part_max_workers_positive(self):
        """测试 DOC_PART_MAX_WORKERS 是正数"""
        assert PubRefConfig.DOC_PART_MAX_WORKERS > 0


class TestIncrementalLLMConfig:
    """测试增量 LLM 配置属性。

    INCREMENTAL_LLM_KEY / INCREMENTAL_LLM_URL / MODEL 在 import 期为 str 占位（''），运行期由
    各入口经 src.settings.apply_settings() 注入；MODEL 的有效默认值在
    src.settings 单点定义（DEFAULT_MODEL == "qwen3.8-max"，UD5）。
    """

    def test_incremental_llm_key_exists(self):
        """测试 INCREMENTAL_LLM_KEY 存在且为字符串"""
        assert hasattr(PubRefConfig, "INCREMENTAL_LLM_KEY")
        assert isinstance(PubRefConfig.INCREMENTAL_LLM_KEY, str)

    def test_incremental_llm_url_exists(self):
        """测试 INCREMENTAL_LLM_URL 存在且为字符串"""
        assert hasattr(PubRefConfig, "INCREMENTAL_LLM_URL")
        assert isinstance(PubRefConfig.INCREMENTAL_LLM_URL, str)

    def test_model_exists(self):
        """测试 MODEL 存在且为字符串"""
        assert hasattr(PubRefConfig, "MODEL")
        assert isinstance(PubRefConfig.MODEL, str)

    def test_model_default_matches_settings(self):
        """MODEL 默认与 Settings 一致：qwen3.8-max（UD5，单点定义于 src/settings.py）"""
        from src.settings import DEFAULT_MODEL, Settings

        assert DEFAULT_MODEL == "qwen3.8-max"
        assert (
            Settings(incremental_llm_key="k", incremental_llm_url="u").model
            == DEFAULT_MODEL
        )
