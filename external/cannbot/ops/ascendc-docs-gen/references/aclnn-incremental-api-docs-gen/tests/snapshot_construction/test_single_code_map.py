# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for build_functions_summary() in single_code_map module.

Uses a mock graph object with .nodes and .edges attributes.
Does NOT test build_file_maps or main (those require ClangParser).
"""

import json
import os

import pytest
from src.snapshot_construction.code_graph.single_code_map import build_functions_summary


# =============================================================================
# Helper: Mock graph object
# =============================================================================


class MockGraph:
    """模拟 CodeMap 的 graph 对象，只包含 nodes 和 edges"""

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges


# =============================================================================
# Tests for build_functions_summary()
# =============================================================================


class TestBuildFunctionsSummary:
    """测试 build_functions_summary() 方法"""

    def test_creates_functions_json(self, tmp_path):
        """测试生成 functions.json 文件"""
        graph = MockGraph(nodes=[], edges=[])
        build_functions_summary(graph, str(tmp_path))
        out_path = os.path.join(str(tmp_path), "functions.json")
        assert os.path.exists(out_path)

    def test_empty_graph_returns_empty_list(self, tmp_path):
        """测试空 graph 生成空结果列表"""
        graph = MockGraph(nodes=[], edges=[])
        build_functions_summary(graph, str(tmp_path))
        out_path = os.path.join(str(tmp_path), "functions.json")
        with open(out_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        assert result == []

    def test_result_structure(self, tmp_path):
        """测试结果条目包含 id, path, calls, called_by, returns 字段"""
        nodes = [
            {"id": "funcA", "type": "Function", "path": "/src/a.cpp"},
        ]
        edges = []
        graph = MockGraph(nodes=nodes, edges=edges)
        build_functions_summary(graph, str(tmp_path))
        out_path = os.path.join(str(tmp_path), "functions.json")
        with open(out_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        assert len(result) == 1
        entry = result[0]
        assert entry["id"] == "funcA"
        assert entry["path"] == "/src/a.cpp"
        assert entry["calls"] == []
        assert entry["called_by"] == []
        assert entry["returns"] == []

    def test_calls_and_called_by_bidirectional(self, tmp_path):
        """测试 calls 和 called_by 的双向关系"""
        nodes = [
            {"id": "funcA", "type": "Function", "path": "/src/a.cpp"},
            {"id": "funcB", "type": "Function", "path": "/src/b.cpp"},
        ]
        edges = [
            {"type": "calls", "source": "funcA", "target": "funcB"},
        ]
        graph = MockGraph(nodes=nodes, edges=edges)
        build_functions_summary(graph, str(tmp_path))
        out_path = os.path.join(str(tmp_path), "functions.json")
        with open(out_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        # result is sorted by function id
        entry_a = result[0]
        entry_b = result[1]
        assert entry_a["id"] == "funcA"
        assert entry_b["id"] == "funcB"
        # A calls B
        assert entry_a["calls"] == ["funcB"]
        # B is called by A
        assert entry_b["called_by"] == ["funcA"]

    def test_returns_relationship(self, tmp_path):
        """测试 returns 关系"""
        nodes = [
            {"id": "funcA", "type": "Function", "path": "/src/a.cpp"},
            {"id": "ACLNN_ERR_PARAM", "type": "errorcode"},
        ]
        edges = [
            {"type": "returns", "source": "funcA", "target": "ACLNN_ERR_PARAM"},
        ]
        graph = MockGraph(nodes=nodes, edges=edges)
        build_functions_summary(graph, str(tmp_path))
        out_path = os.path.join(str(tmp_path), "functions.json")
        with open(out_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        # Only Function nodes appear in result
        assert len(result) == 1
        entry = result[0]
        assert entry["id"] == "funcA"
        assert entry["returns"] == ["ACLNN_ERR_PARAM"]

    def test_non_function_nodes_excluded(self, tmp_path):
        """测试非 Function 类型的节点不出现在结果中"""
        nodes = [
            {"id": "funcA", "type": "Function", "path": "/src/a.cpp"},
            {"id": "MyClass", "type": "class", "path": "/src/a.cpp"},
            {"id": "MyStruct", "type": "struct", "path": "/src/a.cpp"},
            {"id": "myns", "type": "namespace", "path": "/src/a.cpp"},
        ]
        edges = []
        graph = MockGraph(nodes=nodes, edges=edges)
        build_functions_summary(graph, str(tmp_path))
        out_path = os.path.join(str(tmp_path), "functions.json")
        with open(out_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        assert len(result) == 1
        assert result[0]["id"] == "funcA"

    def test_result_sorted_by_function_id(self, tmp_path):
        """测试结果按函数 ID 排序"""
        nodes = [
            {"id": "funcC", "type": "Function", "path": "/src/c.cpp"},
            {"id": "funcA", "type": "Function", "path": "/src/a.cpp"},
            {"id": "funcB", "type": "Function", "path": "/src/b.cpp"},
        ]
        edges = []
        graph = MockGraph(nodes=nodes, edges=edges)
        build_functions_summary(graph, str(tmp_path))
        out_path = os.path.join(str(tmp_path), "functions.json")
        with open(out_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        ids = [entry["id"] for entry in result]
        assert ids == ["funcA", "funcB", "funcC"]
