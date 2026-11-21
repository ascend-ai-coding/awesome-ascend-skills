# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
import json
from pathlib import Path
from src.snapshot_construction.code_graph.clang_parser import ClangParser
from src.logger import info


class CodeMap:
    def __init__(self, root):
        self.root = Path(root)

        self.operator = self.root.name

        self.nodes = []
        self.edges = []

        self.node_set = set()

        self.parser = ClangParser(str(self.root))

    def add_node(self, node_id, node_type, path=None, component=None):
        key = (node_id, path)

        if key in self.node_set:
            return

        node = {"id": node_id, "type": node_type}

        if path:
            node["path"] = path
            node["dir"] = str(Path(path).parent)

        if component:
            node["component"] = component

        self.nodes.append(node)

        self.node_set.add(key)

    def add_edge(self, edge):
        self.edges.append(edge)

    def should_skip(self, path):
        p = str(path).replace("\\", "/")

        if "/tests/" in p or "/examples/" in p:
            return True

        return False

    def detect_component(self, filepath):
        p = str(filepath)

        if "op_api" in p:
            return "API"

        if "op_host" in p:
            return "Host"

        if "op_kernel" in p:
            return "Kernel"

        if "op_graph" in p:
            return "Proto"

        return "Other"

    def build_call_flow_edges(self):
        """
        把同一个父函数下的 calls 按顺序串成链

        例如：
        parent -> A (order=1)
        parent -> B (order=2)
        parent -> C (order=3)

        生成：
        A -> B   type=next_call, parent=parent
        B -> C   type=next_call, parent=parent
        """

        call_groups = {}

        for e in self.edges:
            if e.get("type") != "calls":
                continue

            caller = e.get("source")

            if not caller:
                continue

            if caller not in call_groups:
                call_groups[caller] = []

            call_groups[caller].append(e)

        for caller, call_list in call_groups.items():
            call_list = sorted(
                call_list,
                key=lambda x: (
                    x.get("order", 10**9),
                    x.get("line", 10**9),
                    x.get("column", 10**9),
                ),
            )

            for i in range(len(call_list) - 1):
                curr_call = call_list[i]
                next_call = call_list[i + 1]

                self.edges.append(
                    {
                        "type": "next_call",
                        "source": curr_call["target"],
                        "target": next_call["target"],
                        "parent": caller,
                        "sourcefile": curr_call.get("targetfile"),
                        "targetfile": next_call.get("targetfile"),
                        "file": curr_call.get("file"),
                        "from_order": curr_call.get("order"),
                        "to_order": next_call.get("order"),
                        # "from_line": curr_call.get("line"),
                        # "from_column": curr_call.get("column"),
                        # "to_line": next_call.get("line"),
                        # "to_column": next_call.get("column")
                    }
                )

    def build(self):
        self.add_node(self.operator, "Operator")

        components = ["API", "Host", "Kernel", "Proto", "Other"]

        for comp in components:
            self.add_node(comp, "Component")

            self.edges.append(
                {
                    "type": "contains",
                    "source": self.operator,
                    "target": comp,
                    "sourcefile": None,
                    "targetfile": None,
                    "file": None,
                }
            )

        entity_types = {"Function", "class", "struct", "namespace"}

        for file in self.root.rglob("*"):
            if self.should_skip(file):
                continue

            if file.suffix not in [".cpp", ".cc", ".c", ".h", ".hpp"]:
                continue

            path = str(file)

            component = self.detect_component(path)

            self.add_node(path, "file", path, component)

            self.edges.append(
                {
                    "type": "contains",
                    "source": component,
                    "target": path,
                    "sourcefile": None,
                    "targetfile": path,
                    "file": path,
                }
            )

            ast_nodes, ast_edges = self.parser.parse_file(path)

            for n in ast_nodes:
                node_type = n["type"]

                node_path = n.get("path")

                node_component = component if node_type in entity_types else None

                self.add_node(n["id"], node_type, node_path, node_component)

            for e in ast_edges:
                self.add_edge(e)

        # 新增：把同一个父函数的子调用串成链式流程图
        self.build_call_flow_edges()

    def export(self, output_file):
        data = {
            "nodes": self.nodes,
            "edges": self.edges,
            "metadata": {"operator": self.operator, "root": str(self.root)},
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def run(self):
        self.build()

        output_file = "code_graph.json"

        self.export(output_file)

        info(f"Graph saved to {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        info("Usage: python code_map_0309.py <operator_dir>")
        sys.exit(1)

    root = sys.argv[1]

    graph = CodeMap(root)

    graph.run()
