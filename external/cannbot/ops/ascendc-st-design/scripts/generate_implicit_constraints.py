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
隐式约束生成脚本

功能：
1. 从04_测试因子.yaml中识别需要生成隐式依赖的因子
2. 生成隐式约束并追加到05_约束定义.yaml
3. 保持约束ID的唯一性（避免重复）
4. 保持原始文件格式不变

隐式依赖规则：
1. Tensor输入：{param}.shape 依赖于 {param}.dimensions
   - 当参数有 dimensions 因子时，生成 shape 约束
2. 所有输入：{param}.value_range 依赖于 {param}.dtype
   - 当参数有 dtype 和 value_range 因子时，生成 value_range 约束
3. 非Tensor且非枚举类型输入：{param}.value 依赖于 {param}.value_range
   - 当参数有 value_range 因子且非 Tensor 非枚举时，生成 value 约束

使用方法:
    python scripts/generate_implicit_constraints.py <测试因子.yaml> <约束定义.yaml>
    
示例:
    python scripts/generate_implicit_constraints.py \
        result/04_测试因子.yaml \
        result/05_约束定义.yaml
"""

import sys
import yaml
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import OrderedDict
from datetime import datetime


class ImplicitConstraintGenerator:
    """隐式约束生成器"""

    def __init__(self, factors_path: str, constraints_path: str):
        self.factors_path = factors_path
        self.constraints_path = constraints_path
        self.factors_data = {}
        self.constraints_data = {}
        self.original_content = ""
        self.existing_constraint_ids = set()
        self.new_constraints = []

        print(f"正在加载测试因子: {factors_path}")
        try:
            with open(self.factors_path, "r", encoding="utf-8") as f:
                self.factors_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"测试因子文件不存在: {factors_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"测试因子文件格式错误: {e}")

        print(f"正在加载约束定义: {constraints_path}")
        try:
            with open(self.constraints_path, "r", encoding="utf-8") as f:
                self.original_content = f.read()
                f.seek(0)
                self.constraints_data = yaml.safe_load(f)
                if self.constraints_data is None:
                    self.constraints_data = {}
        except FileNotFoundError:
            print(f"约束定义文件不存在，将创建新文件: {constraints_path}")
            self.constraints_data = {}
            self.original_content = ""
        except yaml.YAMLError as e:
            raise ValueError(f"约束定义文件格式错误: {e}")

        if "constraints" not in self.constraints_data:
            self.constraints_data["constraints"] = []

        self.existing_constraint_ids = set(
            c.get("id") for c in self.constraints_data.get("constraints", []) if c.get("id")
        )
        print(f"已存在 {len(self.existing_constraint_ids)} 个约束")

    def _get_param_type(self, param_name: str) -> str:
        if param_name in self.factors_data:
            return self.factors_data[param_name].get("type", "unknown")
        return "unknown"

    def _is_tensor_param(self, param_name: str) -> bool:
        param_type = self._get_param_type(param_name)
        return param_type in ("aclTensor", "aclTensorList")

    def _is_enum_param(self, param_name: str, factors: Dict) -> bool:
        has_value = f"{param_name}.value" in factors
        has_value_range = any(key.startswith(f"{param_name}.value_range") for key in factors.keys())
        return has_value and not has_value_range

    def _should_generate_value_range_constraint(self, param_name: str, factors: Dict) -> bool:
        has_dtype = f"{param_name}.dtype" in factors
        has_value_range = any(key.startswith(f"{param_name}.value_range") for key in factors.keys())
        return has_dtype and has_value_range

    def _identify_implicit_dependencies(self) -> List[Dict]:
        dependencies = []

        if not self.factors_data:
            print("警告: 测试因子数据为空")
            return dependencies
        print('++self.factors_data: ', self.factors_data)

        for param_name, param_data in self.factors_data.items():
            if not isinstance(param_data, dict):
                continue

            factors = param_data.get("factors", {})
            if not factors:
                continue

            self._check_shape_from_dimensions(param_name, factors, dependencies)
            self._check_value_range_from_dtype(param_name, factors, dependencies)
            self._check_length_from_length_ranges(param_name, factors, dependencies)
            self._check_shape_list_from_length_dims(param_name, factors, dependencies)
            self._check_array_value_from_range(param_name, factors, dependencies)
            self._check_value_from_value_range(param_name, factors, dependencies)

        return dependencies

    def _generate_shape_constraint(self, param_name: str) -> Dict:
        return OrderedDict([
            ("id", f"IMPLICIT-SHAPE-{param_name}"),
            ("type", "calculate"),
            ("sources", [f"{param_name}.dimensions"]),
            ("target", f"{param_name}.shape"),
            ("expression", "derive_shape_from_dimensions(sources[0])"),
            ("description", f"根据dimensions生成{param_name}.shape"),
            ("implicit", True),
        ])

    def _generate_value_range_constraint(self, param_name: str) -> Dict:
        return OrderedDict([
            ("id", f"IMPLICIT-RANGE-{param_name}"),
            ("type", "calculate"),
            ("sources", [f"{param_name}.dtype"]),
            ("target", f"{param_name}.value_range"),
            ("expression", "derive_value_range_from_dtype(sources[0])"),
            ("description", f"根据dtype选择{param_name}.value_range"),
            ("implicit", True),
        ])

    def _generate_value_constraint(self, param_name: str) -> Dict:
        return OrderedDict([
            ("id", f"IMPLICIT-VALUE-{param_name}"),
            ("type", "calculate"),
            ("sources", [f"{param_name}.value_range"]),
            ("target", f"{param_name}.value"),
            ("expression", "derive_value_from_range(sources[0])"),
            ("description", f"根据value_range生成{param_name}.value"),
            ("implicit", True),
        ])

    @staticmethod
    def _generate_length_constraint(param_name: str) -> Dict:
        return OrderedDict([
            ("id", f"IMPLICIT-LENGTH-{param_name}"),
            ("type", "calculate"),
            ("sources", [f"{param_name}.length_ranges"]),
            ("target", f"{param_name}.length"),
            ("expression", "derive_length_from_length_ranges(sources[0])"),
            ("description", f"根据length_ranges生成{param_name}.length"),
            ("implicit", True),
        ])

    @staticmethod
    def _generate_shape_list_constraint(param_name: str) -> Dict:
        return OrderedDict([
            ("id", f"IMPLICIT-SHAPE-LIST-{param_name}"),
            ("type", "calculate"),
            ("sources", [f"{param_name}.length", f"{param_name}.dimensions"]),
            ("target", f"{param_name}.shape_list"),
            ("expression", "derive_shape_list_from_length_and_dimensions(sources[0], sources[1])"),
            ("description", f"根据length和dimensions生成{param_name}.shape_list"),
            ("implicit", True),
        ])

    @staticmethod
    def _generate_array_value_constraint(param_name: str) -> Dict:
        return OrderedDict([
            ("id", f"IMPLICIT-VALUE-LIST-{param_name}"),
            ("type", "calculate"),
            ("sources", [f"{param_name}.length", f"{param_name}.value_range"]),
            ("target", f"{param_name}.value"),
            ("expression", "derive_array_value_from_length_and_value_range(sources[0], sources[1])"),
            ("description", f"根据length和value_range生成{param_name}.value"),
            ("implicit", True),
        ])

    def generate(self) -> List[Dict]:
        print("\n开始识别隐式依赖...")
        dependencies = self._identify_implicit_dependencies()

        if not dependencies:
            print("未发现需要生成的隐式约束")
            return []

        print(f"\n生成 {len(dependencies)} 个隐式约束...")
        for dep in dependencies:
            rule = dep["rule"]
            param_name = dep["param_name"]

            if rule == "shape_from_dimensions":
                constraint = self._generate_shape_constraint(param_name)
            elif rule == "value_range_from_dtype":
                constraint = self._generate_value_range_constraint(param_name)
            elif rule == "value_from_value_range":
                constraint = self._generate_value_constraint(param_name)
            elif rule == "length_from_length_ranges":
                constraint = self._generate_length_constraint(param_name)
            elif rule == "shape_list_from_length_and_dimensions":
                constraint = self._generate_shape_list_constraint(param_name)
            elif rule == "array_value_from_length_and_value_range":
                constraint = self._generate_array_value_constraint(param_name)
            else:
                print(f"警告: 未知的规则类型 {rule}")
                continue

            self.new_constraints.append(constraint)

        return self.new_constraints

    def save(self):
        if not self.new_constraints:
            print("无需保存，没有新生成的约束")
            return

        backup_path = Path(self.constraints_path).with_suffix(".yaml.backup")
        backup_created = False
        try:
            with open(self.constraints_path, "r", encoding="utf-8") as f:
                original_content = f.read()
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(original_content)
            print(f"  备份已保存: {backup_path}")
            backup_created = True
        except FileNotFoundError:
            print("  原文件不存在，跳过备份")
            original_content = ""

        if not original_content.strip():
            content = self._create_new_constraints_file()
        else:
            content = self._append_constraints_to_file(original_content)

        try:
            with open(self.constraints_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"  隐式约束已追加到: {self.constraints_path}")
            print(f"  新增 {len(self.new_constraints)} 个约束")
            print(f"  总约束数: {len(self.constraints_data['constraints']) + len(self.new_constraints)}")

            if backup_created and backup_path.exists():
                backup_path.unlink()
                print(f"  备份文件已删除: {backup_path}")

        except Exception as e:
            print(f"  保存失败，备份文件保留: {backup_path}")
            print(f"  错误: {e}")
            raise

    def _is_tensor_list_param(self, param_name: str) -> bool:
        return self._get_param_type(param_name) == "aclTensorList"

    def _is_array_param(self, param_name: str) -> bool:
        return self._get_param_type(param_name) in (
            "aclIntArray", "aclFloatArray", "aclBoolArray", "aclScalarList"
        )

    def _is_scalar_type_param(self, param_name: str) -> bool:
        param_type = self._get_param_type(param_name)
        scalar_types = {
            "int4_t", "int8_t", "int16_t", "int32_t", "int64_t",
            "uint1_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
            "bool", "float", "float16", "bfloat16", "float32", "double",
            "char", "string",
        }
        return param_type in scalar_types

    def _check_shape_from_dimensions(self, param_name, factors, dependencies):
        if not self._is_tensor_param(param_name):
            return
        if f"{param_name}.dimensions" not in factors:
            return
        constraint_id = f"IMPLICIT-SHAPE-{param_name}"
        if constraint_id not in self.existing_constraint_ids:
            dependencies.append({
                "rule": "shape_from_dimensions",
                "param_name": param_name,
                "constraint_id": constraint_id,
            })
            print(f"  发现隐式依赖: {param_name}.shape <- {param_name}.dimensions")

    def _check_value_range_from_dtype(self, param_name, factors, dependencies):
        if not self._should_generate_value_range_constraint(param_name, factors):
            return
        constraint_id = f"IMPLICIT-RANGE-{param_name}"
        if constraint_id not in self.existing_constraint_ids:
            dependencies.append({
                "rule": "value_range_from_dtype",
                "param_name": param_name,
                "constraint_id": constraint_id,
            })
            print(f"  发现隐式依赖: {param_name}.value_range <- {param_name}.dtype")

    def _check_length_from_length_ranges(self, param_name, factors, dependencies):
        param_type = self._get_param_type(param_name)
        is_list_or_array = (
            param_type == "aclTensorList"
            or param_type in ("aclIntArray", "aclFloatArray", "aclBoolArray", "aclScalarList")
        )
        if not is_list_or_array:
            return
        if f"{param_name}.length_ranges" not in factors:
            return
        constraint_id = f"IMPLICIT-LENGTH-{param_name}"
        if constraint_id not in self.existing_constraint_ids:
            dependencies.append({
                "rule": "length_from_length_ranges",
                "param_name": param_name,
                "constraint_id": constraint_id,
            })
            print(f"  发现隐式依赖: {param_name}.length <- {param_name}.length_ranges")

    def _check_shape_list_from_length_dims(self, param_name, factors, dependencies):
        if self._get_param_type(param_name) != "aclTensorList":
            return
        if f"{param_name}.dimensions" not in factors:
            return
        constraint_id = f"IMPLICIT-SHAPE-LIST-{param_name}"
        if constraint_id not in self.existing_constraint_ids:
            dependencies.append({
                "rule": "shape_list_from_length_and_dimensions",
                "param_name": param_name,
                "constraint_id": constraint_id,
            })
            print(f"  发现隐式依赖: {param_name}.shape_list <- {param_name}.length, {param_name}.dimensions")

    def _check_array_value_from_range(self, param_name, factors, dependencies):
        param_type = self._get_param_type(param_name)
        if param_type not in ("aclIntArray", "aclFloatArray", "aclBoolArray", "aclScalarList"):
            return
        has_value_range = any(key.startswith(f"{param_name}.value_range") for key in factors.keys())
        if not has_value_range:
            return
        constraint_id = f"IMPLICIT-VALUE-LIST-{param_name}"
        if constraint_id not in self.existing_constraint_ids:
            dependencies.append({
                "rule": "array_value_from_length_and_value_range",
                "param_name": param_name,
                "constraint_id": constraint_id,
            })
            print(f"  发现隐式依赖: {param_name}.value <- {param_name}.length, {param_name}.value_range")

    def _check_value_from_value_range(self, param_name, factors, dependencies):
        if self._is_tensor_param(param_name):
            return
        if self._is_enum_param(param_name, factors):
            return
        if self._get_param_type(param_name) in (
            "aclIntArray", "aclFloatArray", "aclBoolArray", "aclScalarList"
        ):
            return
        has_value_range = any(key.startswith(f"{param_name}.value_range") for key in factors.keys())
        if not has_value_range:
            return
        constraint_id = f"IMPLICIT-VALUE-{param_name}"
        if constraint_id not in self.existing_constraint_ids:
            dependencies.append({
                "rule": "value_from_value_range",
                "param_name": param_name,
                "constraint_id": constraint_id,
            })
            print(f"  发现隐式依赖: {param_name}.value <- {param_name}.value_range")

    def _format_constraint_yaml(self, constraint: Dict) -> str:
        lines = [f"  - id: \"{constraint['id']}\""]
        lines.append(f"    type: {constraint['type']}")
        lines.append(f"    sources: {yaml.dump(constraint['sources'], default_flow_style=True).strip()}")
        lines.append(f"    target: \"{constraint['target']}\"")
        lines.append(f"    expression: \"{constraint['expression']}\"")
        lines.append(f"    description: \"{constraint['description']}\"")
        lines.append(f"    implicit: {constraint['implicit']}")
        return "\n".join(lines)

    def _create_new_constraints_file(self) -> str:
        lines = [
            "# 隐式约束定义",
            f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "constraints:",
        ]

        for constraint in self.new_constraints:
            lines.append(self._format_constraint_yaml(constraint))

        lines.append("")
        return "\n".join(lines)

    def _append_constraints_to_file(self, original_content: str) -> str:
        lines = original_content.split("\n")

        constraints_index = -1
        for i, line in enumerate(lines):
            if line.strip() == "constraints:":
                constraints_index = i
                break

        if constraints_index == -1:
            result = original_content.rstrip() + "\n\nconstraints:\n"
        else:
            result_lines = []

            result_lines.extend(lines[:constraints_index + 1])

            result_lines.append("")

            result_lines.append("  # ========================================")
            result_lines.append("  # 隐式约束（自动生成）")
            result_lines.append("  # ========================================")
            result_lines.append("")

            for constraint in self.new_constraints:
                constraint_yaml = self._format_constraint_yaml(constraint)
                result_lines.append(constraint_yaml)

            result_lines.append("")

            i = constraints_index + 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1

            while i < len(lines):
                result_lines.append(lines[i])
                i += 1

            if result_lines and result_lines[-1].strip() != "":
                result_lines.append("")

            result = "\n".join(result_lines)

        return result


def main():
    parser = argparse.ArgumentParser(
        description="从测试因子定义生成隐式约束并追加到约束定义文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    python generate_implicit_constraints.py result/04_测试因子.yaml result/05_约束定义.yaml
    python generate_implicit_constraints.py factors.yaml constraints.yaml --verbose
        """,
    )
    parser.add_argument("factors_file", help="测试因子YAML文件（04_测试因子.yaml）")
    parser.add_argument("constraints_file", help="约束定义YAML文件（05_约束定义.yaml）")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出模式")

    args = parser.parse_args()

    factors_path = Path(args.factors_file)
    if not factors_path.exists():
        print(f"错误: 测试因子文件不存在: {factors_path}")
        sys.exit(1)

    constraints_path = Path(args.constraints_file)

    try:
        generator = ImplicitConstraintGenerator(str(factors_path), str(constraints_path))
        implicit_constraints = generator.generate()

        if args.verbose and implicit_constraints:
            print("\n生成的隐式约束详情:")
            for constraint in implicit_constraints:
                print(f"  - {constraint['id']}: {constraint['type']}")
                print(f"    {constraint['sources']} -> {constraint['target']}")
                print(f"    {constraint['description']}")

        if implicit_constraints:
            print(f"\n正在保存 {len(implicit_constraints)} 个隐式约束...")
            generator.save()
            print("\n完成")
        else:
            print("\n无需生成新的隐式约束")

    except Exception as e:
        print(f"\n错误: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
