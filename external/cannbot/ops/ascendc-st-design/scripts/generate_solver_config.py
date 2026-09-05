#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""
求解配置生成器

功能：
1. 解析约束定义 YAML 文件
2. 构建因子依赖图
3. 拓扑排序确定求解层级
4. 识别锚点因子（入度为0）
5. 输出求解配置到 YAML 文件

使用方法：
    python generate_solver_config.py <约束定义.yaml> [输出配置.yaml]

示例：
    python generate_solver_config.py 05_约束定义.yaml 07_求解配置.yaml
"""

import sys
import yaml
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque
from pathlib import Path


class DependencyGraph:
    """因子依赖图"""
    
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_edges: Dict[str, Set[str]] = defaultdict(set)
        self.in_degree: Dict[str, int] = defaultdict(int)
        self.out_degree: Dict[str, int] = defaultdict(int)
        self.constraint_map: Dict[str, List[str]] = defaultdict(list)
        self.bidirectional_groups: List[Tuple[str, str, str]] = []

    def get_anchors(self) -> List[str]:
        return sorted([n for n in self.nodes if self.in_degree[n] == 0])
    
    def get_sinks(self) -> List[str]:
        return sorted([n for n in self.nodes if self.out_degree[n] == 0])
    
    def get_dependencies(self, node: str) -> Set[str]:
        return self.reverse_edges.get(node, set())
    
    def get_dependents(self, node: str) -> Set[str]:
        return self.edges.get(node, set())
    
    def topological_sort(self) -> List[List[str]]:
        in_deg = {n: self.in_degree[n] for n in self.nodes}
        levels = []
        remaining = set(self.nodes)
        
        while remaining:
            current_level = [n for n in remaining if in_deg[n] == 0]
            
            if not current_level:
                current_level = list(remaining)
                print(f"警告：检测到循环依赖，剩余节点: {current_level}")
            
            current_level.sort()
            levels.append(current_level)
            
            for node in current_level:
                remaining.remove(node)
                for target in self.edges[node]:
                    if target in remaining:
                        in_deg[target] -= 1
        
        return levels
    
    def print_graph(self):
        print("\n" + "=" * 60)
        print("依赖图结构")
        print("=" * 60)
        
        print(f"\n节点总数: {len(self.nodes)}")
        print(f"边总数: {sum(len(t) for t in self.edges.values())}")
        
        print("\n锚点因子（入度=0）:")
        for a in self.get_anchors():
            print(f"  - {a}")
        
        print("\n双向约束组:")
        for f1, f2, cid in self.bidirectional_groups:
            print(f"  - {f1} <-> {f2} ({cid})")
        
        print("\n拓扑层级:")
        for i, level in enumerate(self.topological_sort()):
            print(f"  level_{i}: {level}")

    def add_node(self, node: str):
        if node not in self.nodes:
            self.nodes.add(node)
            self.in_degree[node] = 0
            self.out_degree[node] = 0
    
    def add_edge(self, source: str, target: str, constraint_id: str = ""):
        self.add_node(source)
        self.add_node(target)
        
        if source == target:
            if constraint_id and constraint_id not in self.constraint_map[target]:
                self.constraint_map[target].append(constraint_id)
            return
        
        if target not in self.edges[source]:
            self.edges[source].add(target)
            self.reverse_edges[target].add(source)
            self.in_degree[target] += 1
            self.out_degree[source] += 1
            
        if constraint_id and constraint_id not in self.constraint_map[target]:
            self.constraint_map[target].append(constraint_id)
    
    def add_bidirectional(self, factor1: str, factor2: str, constraint_id: str):
        self.add_node(factor1)
        self.add_node(factor2)
        self.bidirectional_groups.append((factor1, factor2, constraint_id))
        
        if constraint_id:
            if constraint_id not in self.constraint_map[factor1]:
                self.constraint_map[factor1].append(constraint_id)
            if constraint_id not in self.constraint_map[factor2]:
                self.constraint_map[factor2].append(constraint_id)


class ConstraintParser:
    """约束解析器"""
    
    UNIDIRECTIONAL_CONSTRAINTS = {'calculate', 'derive', 'range', 'inferable_filter'}
    BIDIRECTIONAL_CONSTRAINTS = {'inferable', 'match', 'broadcast_dim'}
    CONVERTIBLE_CONSTRAINTS = {'convertible'}
    BROADCAST_CONSTRAINTS = {'broadcast_shape'}
    
    IMPLICIT_FACTORS = {
        'length': {'source_attr': 'length_ranges', 'target_attr': 'length'},
        'shape_list': {'source_attrs': ['length', 'dimensions'], 'target_attr': 'shape_list'},
        'value_list': {'source_attrs': ['length', 'value_range'], 'target_attr': 'value'},
        'value': {'source_attr': 'value_range', 'target_attr': 'value'},
    }
    
    def __init__(self, yaml_data: Dict[str, Any]):
        self.data = yaml_data
        self.metadata = yaml_data.get('metadata', {})
        self.factors = yaml_data.get('factors', {})
        self.constraints = yaml_data.get('constraints', []) or []
        self.graph = DependencyGraph()

    def parse(self) -> DependencyGraph:
        self._add_implicit_factors()
        
        for factor_name in self.factors:
            self.graph.add_node(factor_name)
        
        for constraint in self.constraints:
            self._parse_constraint(constraint)
        
        return self.graph
    
    def _is_valid_factor(self, factor_name: str) -> bool:
        return factor_name in self.factors

    def _infer_factor_info(self, factor_name: str) -> Dict[str, Any]:
        parts = factor_name.rsplit('.', 1)
        if len(parts) != 2:
            return {}
        
        param_name, factor_type = parts
        
        io_type = 'input'
        for existing_factor, info in self.factors.items():
            if existing_factor.startswith(f"{param_name}."):
                io_type = info.get('io_type', 'input')
                break
        
        return {
            'type': factor_type,
            'param': param_name,
            'io_type': io_type
        }
    
    def _add_implicit_factors(self):
        implicit_count = 0
        for constraint in self.constraints:
            target = constraint.get('target', '')
            sources = constraint.get('sources', [])
            is_implicit = constraint.get('implicit', False)
            
            if target and target not in self.factors:
                factor_info = self._infer_factor_info(target)
                if factor_info:
                    self.factors[target] = factor_info
                    print(f"  自动添加因子: {target}")
                    if is_implicit:
                        implicit_count += 1
            
            for source in sources:
                if source and source not in self.factors:
                    factor_info = self._infer_factor_info(source)
                    if factor_info:
                        self.factors[source] = factor_info
                        print(f"  自动添加因子: {source}")
        
        if implicit_count > 0:
            print(f"  隐式依赖引入 {implicit_count} 个推导因子")
    
    def _parse_constraint(self, constraint: Dict[str, Any]):
        constraint_type = constraint.get('type', '')
        
        if constraint_type in self.UNIDIRECTIONAL_CONSTRAINTS:
            self._parse_unidirectional(constraint)
        elif constraint_type in self.BIDIRECTIONAL_CONSTRAINTS:
            self._parse_bidirectional(constraint)
        elif constraint_type in self.CONVERTIBLE_CONSTRAINTS:
            self._parse_convertible(constraint)
        elif constraint_type in self.BROADCAST_CONSTRAINTS:
            self._parse_broadcast_shape(constraint)
        elif constraint_type == 'conditional':
            self._parse_conditional(constraint)
    
    def _parse_unidirectional(self, constraint: Dict[str, Any]):
        constraint_id = constraint.get('id', '')
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        
        if not target or target not in self.factors:
            return
        
        if constraint_id and constraint_id not in self.graph.constraint_map[target]:
            self.graph.constraint_map[target].append(constraint_id)
        
        for source in sources:
            if source in self.factors:
                self.graph.add_edge(source, target, constraint_id)
    
    def _parse_bidirectional(self, constraint: Dict[str, Any]):
        constraint_id = constraint.get('id', '')
        constraint_type = constraint.get('type', '')
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        
        if constraint_type == 'inferable':
            valid_sources = [s for s in sources if s in self.factors]
            
            if len(valid_sources) >= 2:
                for i in range(len(valid_sources) - 1):
                    self.graph.add_bidirectional(
                        valid_sources[i], valid_sources[i + 1], constraint_id
                    )
                
                anchor = valid_sources[0]
                for factor in valid_sources[1:]:
                    self.graph.add_edge(anchor, factor, constraint_id)
        
        elif constraint_type in ('match', 'broadcast_dim'):
            if sources and target and target in self.factors:
                source = sources[0]
                if source in self.factors:
                    self.graph.add_bidirectional(source, target, constraint_id)
                    self.graph.add_edge(source, target, constraint_id)
    
    def _parse_convertible(self, constraint: Dict[str, Any]):
        constraint_id = constraint.get('id', '')
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        
        if not target or target not in self.factors:
            return
        
        for source in sources:
            if source in self.factors:
                self.graph.add_edge(source, target, constraint_id)
    
    def _parse_broadcast_shape(self, constraint: Dict[str, Any]):
        constraint_id = constraint.get('id', '')
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        mode = constraint.get('mode', 'unidirectional')
        
        if not target or target not in self.factors:
            return
        
        if mode == 'unidirectional':
            for source in sources:
                if source in self.factors:
                    self.graph.add_edge(source, target, constraint_id)
        else:
            for source in sources:
                if source in self.factors:
                    self.graph.add_bidirectional(source, target, constraint_id)
                    self.graph.add_edge(source, target, constraint_id)
    
    def _parse_conditional(self, constraint: Dict[str, Any]):
        constraint_id = constraint.get('id', '')
        condition = constraint.get('condition', {})
        condition_factor = condition.get('factor', '')
        
        then_clause = constraint.get('then', {})
        else_clause = constraint.get('else', {})
        
        then_target = then_clause.get('target', '')
        else_target = else_clause.get('target', '')
        
        if condition_factor and condition_factor in self.factors:
            if then_target and then_target in self.factors:
                self.graph.add_edge(condition_factor, then_target, constraint_id)
            if else_target and else_target in self.factors:
                self.graph.add_edge(condition_factor, else_target, constraint_id)


class SolverConfigGenerator:
    """求解配置生成器"""
    
    def __init__(self, graph: DependencyGraph, parser: ConstraintParser):
        self.graph = graph
        self.parser = parser

    @staticmethod
    def _format_constraint_comment(constraint: Dict) -> str:
        cid = constraint.get('id', '')
        ctype = constraint.get('type', '')
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        is_implicit = constraint.get('implicit', False)

        if ctype == 'inferable':
            return f"# - {cid}: {', '.join(sources)} (互推导)"
        if ctype == 'match':
            si = constraint.get('source_index', '')
            ti = constraint.get('target_index', '')
            if sources and target:
                return f"# - {cid}: {sources[0]}[{si}] <-> {target}[{ti}] (匹配)"
            return ""
        if ctype == 'broadcast_dim':
            si = constraint.get('source_index', '')
            ti = constraint.get('target_index', '')
            if sources and target:
                return f"# - {cid}: {sources[0]}[{si}] <-> {target}[{ti}] (广播)"
            return ""
        if ctype == 'broadcast_shape':
            mode = constraint.get('mode', '')
            if sources and target:
                return f"# - {cid}: {sources[0]} -> {target} ({mode} 广播)"
            return ""
        if ctype == 'calculate':
            if sources and target:
                tag = ' (隐式)' if is_implicit else ''
                return f"# - {cid}: {target} <- {', '.join(sources)}{tag}"
            return ""
        if ctype == 'convertible':
            if sources and target:
                return f"# - {cid}: {target} <- {', '.join(sources)} (可转换)"
            return ""
        return ""

    @staticmethod
    def _format_type_dep_line(constraint: Dict) -> str:
        cid = constraint.get('id', '')
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        ctype = constraint.get('type', '')

        if ctype == 'inferable' and sources:
            return f"#   {' <-> '.join(sources)}  (互推导，{cid})"
        if ctype == 'calculate' and target and sources:
            return f"#   {target} <- {', '.join(sources)}  ({cid})"
        if ctype == 'convertible' and target and sources:
            return f"#   {target} <- {', '.join(sources)}  (可转换，{cid})"
        return ""

    @staticmethod
    def _format_shape_dep_line(constraint: Dict) -> str:
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        ctype = constraint.get('type', '')
        cid = constraint.get('id', '')

        has_shape = any('.shape' in s for s in sources) if sources else False
        if not (has_shape or (target and '.shape' in target)):
            return ""

        if ctype == 'match':
            si = constraint.get('source_index', '')
            ti = constraint.get('target_index', '')
            if sources:
                return f"#   {sources[0]}[{si}] <-> {target}[{ti}]  (匹配，{cid})"
        elif ctype == 'broadcast_dim':
            si = constraint.get('source_index', '')
            ti = constraint.get('target_index', '')
            if sources:
                return f"#   {sources[0]}[{si}] <-> {target}[{ti}]  (广播，{cid})"
        elif ctype == 'broadcast_shape' and sources:
            return f"#   {sources[0]} <- {target}  (广播约束，{cid})"
        elif ctype == 'calculate' and target and sources:
            return f"#   {target} <- {', '.join(sources)}  ({cid})"
        return ""

    @staticmethod
    def _add_anchor_section(lines, anchors):
        lines.append("  ")
        lines.append("  # ========== 锚点因子 ==========")
        lines.append("  # 定义：入度为0的因子，无依赖，可独立随机采样")
        lines.append("  anchors:")

        dtype_anchors = [a for a in anchors if '.dtype' in a or '.value' in a]
        shape_anchors = [a for a in anchors if '.shape' in a]
        other_anchors = [a for a in anchors if a not in dtype_anchors and a not in shape_anchors]

        if dtype_anchors:
            lines.append("    # 类型锚点")
            for a in dtype_anchors:
                lines.append(f"    - {a}")

        if shape_anchors:
            lines.append("    # 形状锚点")
            for a in shape_anchors:
                lines.append(f"    - {a}")

        if other_anchors:
            lines.append("    # 固定值因子（无依赖）")
            for a in other_anchors:
                lines.append(f"    - {a}")

    def generate(self) -> str:
        levels = self.graph.topological_sort()
        anchors = self.graph.get_anchors()
        lines = []

        self._add_header_comments(lines)

        lines.append("")
        lines.append("solver:")
        lines.append("  strategy: topological")

        self._add_anchor_section(lines, anchors)
        self._add_derivation_order(lines, levels)
        self._add_implicit_dependency_docs(lines)
        self._add_type_dependency_graph(lines)
        self._add_shape_dependency_graph(lines)

        return '\n'.join(lines)

    def _add_header_comments(self, lines):
        lines.append("# 求解配置")
        lines.append("# 基于约束定义自动生成")
        lines.append("# ")
        lines.append("# 依赖关系分析：")

        for constraint in self.parser.constraints:
            comment = self._format_constraint_comment(constraint)
            if comment:
                lines.append(comment)

    def _format_level_deps_comment(self, level, level_index):
        if level_index == 0:
            return "    # Level 0: 锚点因子（无依赖，可独立采样）"

        deps_info = []
        for factor in level:
            deps = self.graph.get_dependencies(factor)
            if deps:
                deps_list = sorted(list(deps))
                deps_info.append(f"{factor} <- [{', '.join(deps_list)}]")

        if not deps_info:
            return f"    # Level {level_index}:"

        parts = [f"    # Level {level_index}: 从前面层级推导"]
        for info in deps_info[:5]:
            parts.append(f"    #   - {info}")
        if len(deps_info) > 5:
            parts.append(f"    #   ... (+{len(deps_info) - 5} more)")
        return "\n".join(parts)

    def _add_derivation_order(self, lines, levels):
        lines.append("  ")
        lines.append("  # ========== 推导顺序 ==========")
        lines.append("  derivation_order:")

        for i, level in enumerate(levels):
            comment = self._format_level_deps_comment(level, i)
            for comment_line in comment.split("\n"):
                lines.append(comment_line)

            lines.append(f"    level_{i}:")
            for factor in level:
                lines.append(f"      - {factor}")

    def _add_implicit_dependency_docs(self, lines):
        implicit_constraints = [c for c in self.parser.constraints if c.get('implicit', False)]
        if not implicit_constraints:
            return
        lines.append("")
        lines.append("# ========== 隐式依赖说明 ==========")
        lines.append("# ")
        lines.append("# 隐式约束由 generate_implicit_constraints.py 自动生成，")
        lines.append("# 描述参数内部属性的推导关系。")
        lines.append("# ")
        for ic in implicit_constraints:
            cid = ic.get('id', '')
            sources = ic.get('sources', [])
            target = ic.get('target', '')
            desc = ic.get('description', '')
            lines.append(f"#   {target} <- {', '.join(sources)}  ({cid}: {desc})")
        lines.append("# ")

    def _add_type_dependency_graph(self, lines):
        lines.append("# ========== 依赖图说明 ==========")
        lines.append("# ")
        lines.append("# 类型依赖图：")

        for constraint in self.parser.constraints:
            if constraint.get('type') in ['inferable', 'calculate', 'convertible']:
                dep_line = self._format_type_dep_line(constraint)
                if dep_line:
                    lines.append(dep_line)

    def _add_shape_dependency_graph(self, lines):
        lines.append("# ")
        lines.append("# 形状依赖图：")

        for constraint in self.parser.constraints:
            if constraint.get('type') in ['match', 'broadcast_dim', 'broadcast_shape', 'calculate']:
                dep_line = self._format_shape_dep_line(constraint)
                if dep_line:
                    lines.append(dep_line)


def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_text(content: str, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def print_summary(graph: DependencyGraph, levels: List[List[str]]):
    print("\n" + "=" * 70)
    print("求解配置生成摘要")
    print("=" * 70)
    
    print(f"\n因子总数: {len(graph.nodes)}")
    print(f"层级总数: {len(levels)}")
    
    anchors = graph.get_anchors()
    print(f"\n锚点因子（入度=0）: {len(anchors)}个")
    for a in anchors:
        print(f"  - {a}")
    
    print(f"\n双向约束组: {len(graph.bidirectional_groups)}组")
    for f1, f2, cid in graph.bidirectional_groups:
        print(f"  - {f1} <-> {f2} ({cid})")
    
    print(f"\n推导层级:")
    for i, level in enumerate(levels):
        level_preview = ', '.join(level[:5])
        suffix = '...' if len(level) > 5 else ''
        print(f"  level_{i} ({len(level)}个): [{level_preview}{suffix}]")
    
    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        print("使用方法: python generate_solver_config.py <约束定义.yaml> [输出配置.yaml]")
        print("示例: python generate_solver_config.py 05_约束定义.yaml 07_求解配置.yaml")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_file).exists():
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)
    
    print(f"正在读取: {input_file}")
    
    yaml_data = load_yaml(input_file)
    
    print("正在解析约束关系...")
    parser = ConstraintParser(yaml_data)
    graph = parser.parse()
    
    if '--verbose' in sys.argv or '-v' in sys.argv:
        graph.print_graph()
    
    print("正在生成求解配置...")
    generator = SolverConfigGenerator(graph, parser)
    config_text = generator.generate()
    
    levels = graph.topological_sort()
    print_summary(graph, levels)
    
    if output_file:
        save_text(config_text, output_file)
        print(f"\n求解配置已保存到: {output_file}")
    else:
        print("\n" + "=" * 70)
        print("生成的配置:")
        print("=" * 70)
        print(config_text)


if __name__ == "__main__":
    main()
