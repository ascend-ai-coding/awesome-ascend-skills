# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Tests for CodeMap class — pure methods only.

Tests add_node(), add_edge(), should_skip(), detect_component(), and build_call_flow_edges().
Does NOT test build() or run() (those require ClangParser to parse real C++ files).
"""

from unittest.mock import patch

import pytest

from src.snapshot_construction.code_graph.code_map_0309 import CodeMap

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cm():
    """创建 CodeMap 实例，mock 掉 ClangParser 避免依赖 libclang"""
    with patch("src.snapshot_construction.code_graph.code_map_0309.ClangParser"):
        m = CodeMap("/tmp/fake_root")
        yield m


# =============================================================================
# Tests for add_node()
# =============================================================================


class TestAddNode:
    """测试 add_node() 方法"""

    def test_add_single_node(self, cm):
        """测试添加单个节点后出现在 nodes 中"""
        cm.add_node("funcA", "Function", "/path/a.cpp")
        assert len(cm.nodes) == 1
        assert cm.nodes[0]["id"] == "funcA"
        assert cm.nodes[0]["type"] == "Function"

    def test_add_duplicate_not_added(self, cm):
        """测试添加重复节点（相同 id+path）不会重复添加"""
        cm.add_node("funcA", "Function", "/path/a.cpp")
        cm.add_node("funcA", "Function", "/path/a.cpp")
        assert len(cm.nodes) == 1

    def test_add_node_with_path_has_dir(self, cm):
        """测试带 path 的节点包含 path 和 dir 字段"""
        cm.add_node("funcA", "Function", "/path/to/a.cpp")
        node = cm.nodes[0]
        assert "path" in node
        assert node["path"] == "/path/to/a.cpp"
        assert "dir" in node
        assert node["dir"] == "/path/to"

    def test_add_node_with_component(self, cm):
        """测试带 component 的节点包含 component 字段"""
        cm.add_node("funcA", "Function", "/path/a.cpp", "API")
        node = cm.nodes[0]
        assert "component" in node
        assert node["component"] == "API"

    def test_add_node_without_path_no_path_key(self, cm):
        """测试不带 path 的节点没有 path 键"""
        cm.add_node("errCode", "errorcode")
        node = cm.nodes[0]
        assert "path" not in node
        assert "dir" not in node


# =============================================================================
# Tests for add_edge()
# =============================================================================


class TestAddEdge:
    """测试 add_edge() 方法"""

    def test_add_single_edge(self, cm):
        """测试添加单条边后出现在 edges 中"""
        edge = {"type": "calls", "source": "A", "target": "B"}
        cm.add_edge(edge)
        assert len(cm.edges) == 1
        assert cm.edges[0] is edge

    def test_add_multiple_edges(self, cm):
        """测试添加多条边全部出现在 edges 中"""
        e1 = {"type": "calls", "source": "A", "target": "B"}
        e2 = {"type": "calls", "source": "A", "target": "C"}
        e3 = {"type": "returns", "source": "A", "target": "ERR"}
        cm.add_edge(e1)
        cm.add_edge(e2)
        cm.add_edge(e3)
        assert len(cm.edges) == 3
        assert cm.edges[0] is e1
        assert cm.edges[1] is e2
        assert cm.edges[2] is e3


# =============================================================================
# Tests for should_skip()
# =============================================================================


class TestShouldSkip:
    """测试 should_skip() 方法"""

    def test_path_with_tests(self, cm):
        """测试包含 /tests/ 的路径返回 True"""
        assert cm.should_skip("/home/user/tests/file.cpp") is True

    def test_path_with_examples(self, cm):
        """测试包含 /examples/ 的路径返回 True"""
        assert cm.should_skip("/home/user/examples/file.cpp") is True

    def test_normal_path(self, cm):
        """测试普通路径返回 False"""
        assert cm.should_skip("/home/user/src/file.cpp") is False


# =============================================================================
# Tests for detect_component()
# =============================================================================


class TestDetectComponent:
    """测试 detect_component() 方法"""

    def test_op_api(self, cm):
        """测试 op_api 路径返回 API"""
        assert cm.detect_component("/project/op_api/aclnn_abs.cpp") == "API"

    def test_op_host(self, cm):
        """测试 op_host 路径返回 Host"""
        assert cm.detect_component("/project/op_host/aclnn_abs.cpp") == "Host"

    def test_op_kernel(self, cm):
        """测试 op_kernel 路径返回 Kernel"""
        assert cm.detect_component("/project/op_kernel/aclnn_abs.cpp") == "Kernel"

    def test_op_graph(self, cm):
        """测试 op_graph 路径返回 Proto"""
        assert cm.detect_component("/project/op_graph/aclnn_abs.cpp") == "Proto"

    def test_other_path(self, cm):
        """测试其他路径返回 Other"""
        assert cm.detect_component("/project/random/file.cpp") == "Other"


# =============================================================================
# Tests for build_call_flow_edges()
# =============================================================================


class TestBuildCallFlowEdges:
    """测试 build_call_flow_edges() 方法"""

    def test_empty_edges_no_next_call(self, cm):
        """测试空 edges 不生成 next_call 边"""
        cm.build_call_flow_edges()
        next_call_edges = [e for e in cm.edges if e["type"] == "next_call"]
        assert len(next_call_edges) == 0

    def test_single_call_no_next_call(self, cm):
        """测试单条 calls 边不生成 next_call（需要 2 条以上才能串链）"""
        cm.add_edge(
            {
                "type": "calls",
                "source": "A",
                "target": "B",
                "order": 1,
                "line": 10,
                "column": 5,
            }
        )
        cm.build_call_flow_edges()
        next_call_edges = [e for e in cm.edges if e["type"] == "next_call"]
        assert len(next_call_edges) == 0

    def test_multiple_calls_same_caller_creates_chain(self, cm):
        """测试同一 caller 的多条 calls 生成 next_call 链"""
        cm.add_edge(
            {
                "type": "calls",
                "source": "A",
                "target": "B",
                "order": 1,
                "line": 10,
                "column": 5,
            }
        )
        cm.add_edge(
            {
                "type": "calls",
                "source": "A",
                "target": "C",
                "order": 2,
                "line": 20,
                "column": 5,
            }
        )
        cm.add_edge(
            {
                "type": "calls",
                "source": "A",
                "target": "D",
                "order": 3,
                "line": 30,
                "column": 5,
            }
        )
        cm.build_call_flow_edges()
        next_call_edges = [e for e in cm.edges if e["type"] == "next_call"]
        # B->C, C->D
        assert len(next_call_edges) == 2
        assert next_call_edges[0]["source"] == "B"
        assert next_call_edges[0]["target"] == "C"
        assert next_call_edges[1]["source"] == "C"
        assert next_call_edges[1]["target"] == "D"

    def test_different_callers_separate_chains(self, cm):
        """测试不同 caller 的 calls 生成各自独立的链"""
        cm.add_edge(
            {
                "type": "calls",
                "source": "A",
                "target": "B",
                "order": 1,
                "line": 10,
                "column": 5,
            }
        )
        cm.add_edge(
            {
                "type": "calls",
                "source": "A",
                "target": "C",
                "order": 2,
                "line": 20,
                "column": 5,
            }
        )
        cm.add_edge(
            {
                "type": "calls",
                "source": "X",
                "target": "Y",
                "order": 1,
                "line": 10,
                "column": 5,
            }
        )
        cm.add_edge(
            {
                "type": "calls",
                "source": "X",
                "target": "Z",
                "order": 2,
                "line": 20,
                "column": 5,
            }
        )
        cm.build_call_flow_edges()
        next_call_edges = [e for e in cm.edges if e["type"] == "next_call"]
        # A chain: B->C, X chain: Y->Z
        assert len(next_call_edges) == 2
        sources = {e["source"] for e in next_call_edges}
        assert sources == {"B", "Y"}

    def test_order_field_sorting(self, cm):
        """测试按 order 字段排序后再串链"""
        # 故意乱序添加
        cm.add_edge(
            {
                "type": "calls",
                "source": "A",
                "target": "C",
                "order": 3,
                "line": 30,
                "column": 5,
            }
        )
        cm.add_edge(
            {
                "type": "calls",
                "source": "A",
                "target": "B",
                "order": 1,
                "line": 10,
                "column": 5,
            }
        )
        cm.add_edge(
            {
                "type": "calls",
                "source": "A",
                "target": "D",
                "order": 2,
                "line": 20,
                "column": 5,
            }
        )
        cm.build_call_flow_edges()
        next_call_edges = [e for e in cm.edges if e["type"] == "next_call"]
        # 排序后: B(1) -> D(2) -> C(3)
        # 链: B->D, D->C
        assert len(next_call_edges) == 2
        assert next_call_edges[0]["source"] == "B"
        assert next_call_edges[0]["target"] == "D"
        assert next_call_edges[1]["source"] == "D"
        assert next_call_edges[1]["target"] == "C"
