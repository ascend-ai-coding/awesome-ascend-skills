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
空tensor测试用例生成脚本

核心理念：从已生成的正常用例派生空tensor用例，确保约束自然满足。

工作流程：
1. 读取已生成的L0/L1测试用例CSV
2. 解析约束定义，提取shape相关约束
3. 分析空tensor场景
4. 选择代表性用例作为模板
5. 按场景修改维度值为0
6. 追加到用例文件

使用方法：
    python generate_empty_tensor_cases.py <test_cases.csv> <constraints.yaml> [options]
    
示例：
    python generate_empty_tensor_cases.py \
        ops/addbmm/tests/st/testcases/L0_test_cases.csv \
        ops/addbmm/tests/st/design/05_约束定义.yaml \
        --output ops/addbmm/tests/st/testcases/L0_test_cases_with_empty.csv \
        --verbose
"""

import argparse
import sys
import re
import json
import ast
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import yaml
import pandas as pd


# 基础配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class ProcessInputParams:
    """封装 _process_input_shapes 的输入参数，减少参数数量"""
    input_shapes: list
    input_dtypes: list
    input_formats: list
    input_types: list
    input_indices: list
    dim_scenario: any
    affected_params: list
    zero_positions: dict
    original_dim: list


@dataclass
class OutputShapeParams:
    """计算输出形状所需参数封装"""
    dim_scenario: str
    self_shape: list
    batch1_shape: list
    batch2_shape: list
    keep_dim_value: bool
    new_row: pd.Series
    default_output_shape: list


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='从正常用例派生空tensor用例',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('test_cases_csv', help='已生成的测试用例CSV文件路径')
    parser.add_argument('constraints_yaml', help='约束定义YAML文件路径')
    parser.add_argument('--output', '-o', help='输出CSV文件路径（默认覆盖原文件）')
    parser.add_argument('--param-def', help='参数定义YAML文件路径（用于获取参数信息）')
    parser.add_argument('--num-empty', type=int, default=5, help='生成的空tensor用例数量')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    return parser.parse_args()


def load_constraints(constraints_path: str) -> Dict:
    """加载约束定义"""
    with open(constraints_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_param_def(param_def_path: str) -> Dict:
    """加载参数定义"""
    if not param_def_path:
        return {}
    with open(param_def_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_test_cases(csv_path: str) -> pd.DataFrame:
    """加载测试用例"""
    return pd.read_csv(csv_path)


def parse_shape_from_str(shape_str: str) -> List[int]:
    """从字符串解析shape列表"""
    if pd.isna(shape_str) or shape_str == '':
        return []
    
    shape_str = str(shape_str).strip()
    
    if shape_str.startswith('[') and shape_str.endswith(']'):
        try:
            return ast.literal_eval(shape_str)
        except (ValueError, SyntaxError):
            return []
    
    return []


def parse_shapes_list(shapes_str: str) -> List[List[int]]:
    """解析input_tensor_shape字符串为shape列表"""
    if pd.isna(shapes_str) or shapes_str == '':
        return []
    
    shapes_str = str(shapes_str).strip()
    
    if shapes_str.startswith('[') and shapes_str.endswith(']'):
        parsed = ast.literal_eval(shapes_str)
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], list):
                return parsed
            else:
                return [parsed]
        return []
    
    return []


def analyze_operator_scenario(constraints: Dict) -> List[Dict]:
    """
    分析算子的空tensor场景
    
    根据约束定义中的shape约束，识别可以置0的维度位置
    
    Returns:
        空tensor场景列表，每个场景包含：
        - name: 场景名称
        - description: 场景描述
        - zero_positions: 各参数需要置0的维度位置
        - scenario_type: 'empty_in_empty_out' 或 'empty_in_non_empty_out'
    """
    scenarios = []
    
    operator_name = constraints.get('metadata', {}).get('operator', 'unknown')
    
    constraints_list = constraints.get('constraints', [])
    factors = constraints.get('factors', {})
    
    tensor_params = set()
    for _, factor_def in factors.items():
        param_name = factor_def.get('param', '')
        if param_name:
            io_type = factor_def.get('io_type', '')
            if io_type in ['input', 'output']:
                tensor_params.add(param_name)
    
    shape_constraints = []
    for constraint in constraints_list:
        target = constraint.get('target', '')
        if 'shape' in target:
            shape_constraints.append(constraint)
    
    if operator_name.lower().find('addbmm') != -1 or operator_name.lower().find('bmm') != -1:
        scenarios = analyze_matmul_like_operator(constraints, factors)
    elif operator_name.lower().find('matmul') != -1:
        scenarios = analyze_matmul_like_operator(constraints, factors)
    elif (operator_name.lower().find('mean') != -1
        or operator_name.lower().find('reduce') != -1
        or operator_name.lower().find('sum') != -1):
        scenarios = analyze_reduce_like_operator(constraints, factors)
    else:
        scenarios = analyze_general_operator(constraints, factors)
    
    return scenarios


def analyze_matmul_like_operator(constraints: Dict, factors: Dict) -> List[Dict]:
    """
    分析矩阵乘类算子的空tensor场景
    
    Addbmm/Baddbmm/Matmul 等算子的shape约束：
    - batch1.shape = [b, m, k]
    - batch2.shape = [b, k, n]
    - out.shape = [m, n]
    - self.shape 可广播到 out.shape
    
    空tensor场景：
    (1) b=0: batch维度为空 → 入空出非空（bmm求和结果为0但shape不变）
    (2) m=0: M维度为空 → 入空出空
    (3) k=0: K维度为空 → 入空出非空（matmul结果仍为[m,n]）
    (4) n=0: N维度为空 → 入空出空
    (5) 全空: 所有轴为空
    """
    scenarios = []
    
    tensor_inputs = []
    tensor_output = None
    
    for factor_id, factor_def in factors.items():
        param = factor_def.get('param', '')
        io_type = factor_def.get('io_type', '')
        if io_type == 'input' and 'shape' in factor_id:
            tensor_inputs.append(param)
        elif io_type == 'output' and 'shape' in factor_id:
            tensor_output = param
    
    batch1_param = None
    batch2_param = None
    self_param = None
    out_param = tensor_output
    
    for param in tensor_inputs:
        if 'batch1' in param.lower():
            batch1_param = param
        elif 'batch2' in param.lower():
            batch2_param = param
        elif 'self' in param.lower():
            self_param = param
    
    scenarios = check_param(scenarios, batch1_param, batch2_param, self_param, out_param)

    return scenarios


def check_param(scenarios, batch1_param, batch2_param, self_param, out_param):
    if batch1_param and batch2_param:
        scenarios = [
            {
                'name': 'b=0',
                'description': 'batch维度为空，bmm求和结果为0但shape不变',
                'zero_positions': {
                    batch1_param: [0],
                    batch2_param: [0],
                },
                'scenario_type': 'empty_in_non_empty_out',
                'affected_params': [batch1_param, batch2_param],
            },
            {
                'name': 'm=0',
                'description': 'M维度为空，输出对应轴也为空',
                'zero_positions': {
                    batch1_param: [1],
                    self_param: [0] if self_param else None,
                    out_param: [0] if out_param else None,
                },
                'scenario_type': 'empty_in_empty_out',
                'affected_params': [batch1_param, self_param, out_param],
            },
            {
                'name': 'k=0',
                'description': 'K维度为空，matmul结果仍为[m,n]',
                'zero_positions': {
                    batch1_param: [2],
                    batch2_param: [1],
                },
                'scenario_type': 'empty_in_non_empty_out',
                'affected_params': [batch1_param, batch2_param],
            },
            {
                'name': 'n=0',
                'description': 'N维度为空，输出对应轴也为空',
                'zero_positions': {
                    batch2_param: [2],
                    self_param: [1] if self_param else None,
                    out_param: [1] if out_param else None,
                },
                'scenario_type': 'empty_in_empty_out',
                'affected_params': [batch2_param, self_param, out_param],
            },
        ]
    return scenarios


def analyze_reduce_like_operator(constraints: Dict, factors: Dict) -> List[Dict]:
    """
    分析reduce类算子的空tensor场景
    Mean/ReduceSum/ReduceMax 等算子的特性：
    - 对指定维度进行reduce操作
    - keepDim决定是否保留reduce维度
    - 空tensor场景需考虑reduce维度是否包含空轴
    """
    # 1. 提取输入输出参数
    tensor_input, tensor_output = _extract_reduce_tensor_params(factors)
    if not tensor_input:
        return []

    # 2. 构建场景
    return _build_reduce_empty_scenarios(tensor_input, tensor_output)


def _extract_reduce_tensor_params(factors: Dict) -> tuple:
    """从 factors 中提取 input / output tensor 参数名"""
    tensor_input = None
    tensor_output = None

    for factor_id, factor_def in factors.items():
        param = factor_def.get('param', '')
        io_type = factor_def.get('io_type', '')
        if io_type == 'input' and 'shape' in factor_id and not tensor_input:
            tensor_input = param
        elif io_type == 'output' and 'shape' in factor_id:
            tensor_output = param

    return tensor_input, tensor_output


def _build_reduce_empty_scenarios(tensor_input: str, tensor_output: str) -> List[Dict]:
    """构建空tensor场景列表"""
    return [
        {
            'name': 'non_reduce_dim_empty',
            'description': '非reduce维度为空，输出对应轴仍为空',
            'zero_positions': {tensor_input: [0]},
            'dim_scenario': 'non_reduce',
            'scenario_type': 'empty_in_empty_out',
            'affected_params': [tensor_input, tensor_output],
        },
        {
            'name': 'reduce_dim_empty',
            'description': 'reduce维度为空，reduce结果需特殊处理',
            'zero_positions': {tensor_input: [1]},
            'dim_scenario': 'reduce',
            'scenario_type': 'empty_in_non_empty_out',
            'affected_params': [tensor_input],
        },
        {
            'name': 'all_reduce_output_empty',
            'description': 'reduce所有维度，输出为scalar',
            'zero_positions': {},
            'dim_scenario': 'all_reduce',
            'scenario_type': 'empty_in_empty_out',
            'affected_params': [tensor_output],
        },
    ]


def analyze_general_operator(constraints: Dict, factors: Dict) -> List[Dict]:
    """
    分析通用算子的空tensor场景
    
    对于非矩阵乘类算子，采用通用策略：
    - 找出所有输入tensor参数
    - 为每个参数生成单轴为空的场景
    - 生成全空场景
    """
    scenarios = []
    
    tensor_inputs = []
    tensor_output = None
    
    tensor_inputs, tensor_output = _extract_reduce_tensor_params(factors)
    
    if not tensor_inputs:
        return scenarios
    
    first_tensor = tensor_inputs[0]
    
    scenarios.append({
        'name': 'single_axis_zero',
        'description': '第一个输入tensor的某个维度为空',
        'zero_positions': {
            first_tensor: [0],
        },
        'scenario_type': 'empty_in_empty_out',
        'affected_params': [first_tensor],
    })
    
    return scenarios


def select_template_cases(df: pd.DataFrame, num_templates: int = 3) -> List[int]:
    """
    选择代表性用例作为模板
    
    选择标准：
    - 优先选择fp32/fp16 dtype（不使用complex/float64等）
    - 优先选择多维tensor（维度>=2）
    - dim值合理（在shape维度范围内）
    - 避免已经是空tensor的用例
    """
    # 过滤空tensor用例
    non_empty_df = df[~df['case_name'].str.contains('EmptyTensor', na=False)]
    non_empty_df = non_empty_df if len(non_empty_df) > 0 else df

    # 打分排序
    scored_indices = _score_candidate_cases(non_empty_df)
    scored_indices.sort(key=lambda x: x[1], reverse=True)

    # 选取前N个模板
    template_indices = [idx for idx, _ in scored_indices[:num_templates]]
    return template_indices


def _score_candidate_cases(non_empty_df: pd.DataFrame) -> List[tuple]:
    """对候选用例进行打分，返回 (idx, score) 列表"""
    scored_indices = []
    for idx, row in non_empty_df.iterrows():
        score = 0
        score = _calculate_dtype_score(row, score)
        score = _calculate_shape_score(row, score)
        score = _calculate_dim_valid_score(row, score)
        scored_indices.append((idx, score))
    return scored_indices


def _calculate_dtype_score(row, score):
    """根据数据类型计算分数"""
    input_dtypes = ast.literal_eval(str(row['input_tensor_dtype'])) if not pd.isna(row['input_tensor_dtype']) else []
    dtype_str = str(input_dtypes[0]) if input_dtypes else ''

    if 'fp32' in dtype_str or 'float32' in dtype_str:
        return score + 20
    elif 'fp16' in dtype_str or 'float16' in dtype_str:
        return score + 18
    elif 'bf16' in dtype_str or 'bfloat16' in dtype_str:
        return score + 10
    elif 'complex' in dtype_str:
        return score - 50
    elif 'fp64' in dtype_str or 'float64' in dtype_str:
        return score - 30
    return score


def _calculate_shape_score(row, score):
    """根据张量维度计算分数"""
    input_shapes = parse_shapes_list(row['input_tensor_shape'])
    if input_shapes and len(input_shapes[0]) >= 2:
        return score + 10
    return score


def _calculate_dim_valid_score(row, score):
    """根据dim合法性计算分数"""
    original_dim = get_original_dim([], row)
    input_shapes = parse_shapes_list(row['input_tensor_shape'])

    if input_shapes and original_dim:
        shape_dims = len(input_shapes[0])
        valid_dim = all(-shape_dims <= d < shape_dims for d in original_dim if isinstance(d, int))
        if valid_dim:
            return score + 5
    return score


def get_original_dim(original_dim, row):
    for col in row.index:
        # 不是 dim 属性，直接跳过
        if not (col.startswith('attr_name') and str(row[col]).lower() == 'dim'):
            continue

        # 找不到对应 value 列，跳过
        value_col = col.replace('attr_name', 'attr_value')
        if value_col not in row.index:
            continue

        dim_str = str(row[value_col])
        if dim_str.startswith('['):
            original_dim = ast.literal_eval(dim_str)
    
    return original_dim


def apply_zero_positions(shape: List[int], zero_positions: List[int] or str) -> List[int]:
    """
    应用空轴位置到shape
    
    Args:
        shape: 原始shape
        zero_positions: 需要置0的维度索引列表，或 'all' 表示全部置0
    
    Returns:
        修改后的shape
    """
    if zero_positions == 'all':
        return [0] * len(shape)
    
    new_shape = deepcopy(shape)
    
    for pos in zero_positions:
        if 0 <= pos < len(new_shape):
            new_shape[pos] = 0
    
    return new_shape


def generate_empty_case(row: pd.Series, scenario: Dict, case_idx: int) -> pd.Series:
    """从模板用例生成空tensor用例"""
    new_row = deepcopy(row)
    # 1. 基础信息解析
    scenario_name, dim_scenario = _parse_basic_scenario(scenario)
    level, aclnn_name = _parse_case_level_and_name(row)
    new_row['case_name'] = f"{aclnn_name}-{level}-EmptyTensor-{case_idx:03d}"
    # 2. 解析输入张量信息
    input_shapes, input_dtypes, input_formats, input_types, input_indices = _parse_input_tensors(row)
    # 3. 获取场景配置
    affected_params, zero_positions = _parse_scenario_params(scenario)
    original_dim = get_original_dim([], row)
    # 4. 处理输入形状（核心逻辑拆分）
    # 封装成参数对象
    process_params = ProcessInputParams(
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        input_formats=input_formats,
        input_types=input_types,
        input_indices=input_indices,
        dim_scenario=dim_scenario,
        affected_params=affected_params,
        zero_positions=zero_positions,
        original_dim=original_dim
    )
    # 调用函数
    new_input_shapes, batch1_shape, batch2_shape, self_shape = _process_input_shapes(process_params)
    # 5. 解析属性 keepDim  这里改名
    keep_dim_value = _parse_keepdim_attr(new_row)
    # 6. 计算输出形状      这里也改名
    # 封装参数
    output_shape_params = OutputShapeParams(
        dim_scenario=dim_scenario,
        self_shape=self_shape,
        batch1_shape=batch1_shape,
        batch2_shape=batch2_shape,
        keep_dim_value=keep_dim_value,
        new_row=new_row,
        default_output_shape=parse_shape_from_str(row['output_tensor_shape'])
    )
    # 调用函数
    output_shape = _calculate_output_shape(output_shape_params)
    # 7. 赋值并返回
    new_row['input_tensor_shape'] = str(new_input_shapes)
    new_row['output_tensor_shape'] = str(output_shape) if output_shape else ''
    return new_row


def _parse_basic_scenario(scenario: Dict) -> tuple:
    """解析场景基础信息"""
    return scenario.get('name', 'unknown'), scenario.get('dim_scenario', None)


def _parse_case_level_and_name(row: pd.Series) -> tuple:
    """解析用例等级和算子名称"""
    level = 'L0' if 'L0-' in row['case_name'] else 'L1'
    aclnn_name_parts = row['case_name'].split('-')
    aclnn_name = aclnn_name_parts[0] if aclnn_name_parts else 'OP'
    return level, aclnn_name


def _parse_input_tensors(row: pd.Series) -> tuple:
    """解析输入张量的形状、类型、格式等信息"""
    input_shapes = parse_shapes_list(row['input_tensor_shape'])
    input_dtypes = ast.literal_eval(str(row['input_tensor_dtype'])) if not pd.isna(row['input_tensor_dtype']) else []
    input_formats = ast.literal_eval(str(row['input_tensor_format'])) if not pd.isna(row['input_tensor_format']) else []
    input_types = ast.literal_eval(str(row['input_tensor_type'])) if not pd.isna(row['input_tensor_type']) else []
    input_indices = ast.literal_eval(str(row['input_tensor_index'])) if not pd.isna(row['input_tensor_index']) else []
    return input_shapes, input_dtypes, input_formats, input_types, input_indices


def _parse_scenario_params(scenario: Dict) -> tuple:
    """解析场景中的影响参数和零位置配置"""
    return scenario.get('affected_params', []), scenario.get('zero_positions', {})


def _get_param_name(i: int, dtype: str, format_: str, affected_params: list) -> str:
    """根据索引和类型获取参数名称"""
    dtype_lower = str(dtype).lower()
    format_lower = str(format_).lower()
    
    if 'batch' in dtype_lower or 'batch' in format_lower:
        if i == 0 and 'batch1' in affected_params:
            return 'batch1'
        elif i == 1 and 'batch2' in affected_params:
            return 'batch2'
    
    if i == 0:
        return 'self'
    elif i == 1:
        return 'batch1'
    elif i == 2:
        return 'batch2'
    return ''


def _modify_shape_by_dim_scenario(shape: list, dim_scenario: str, original_dim: list) -> list:
    """根据 dim_scenario 修改形状"""
    new_shape = deepcopy(shape)
    if dim_scenario == 'non_reduce':
        non_reduce_axes = [i for i in range(len(shape)) if i not in original_dim]
        if non_reduce_axes and non_reduce_axes[0] < len(shape):
            new_shape[non_reduce_axes[0]] = 0
    elif dim_scenario == 'reduce':
        if original_dim and original_dim[0] < len(shape):
            new_shape[original_dim[0]] = 0
    return new_shape


def _process_input_shapes(params: ProcessInputParams):
    """处理所有输入形状，返回新形状 + 关键形状变量"""
    new_input_shapes = []
    batch1_shape = batch2_shape = self_shape = None

    for i, (shape, dtype, format_, type_, idx_) in enumerate(
        zip(params.input_shapes, params.input_dtypes, params.input_formats, params.input_types, params.input_indices)
    ):
        param_name = _get_param_name(i, dtype, format_, params.affected_params)
        new_shape = shape

        if params.dim_scenario and param_name == 'self' and len(shape) > 0:
            new_shape = _modify_shape_by_dim_scenario(shape, params.dim_scenario, params.original_dim)
        elif param_name in params.zero_positions:
            new_shape = apply_zero_positions(shape, params.zero_positions[param_name])

        new_input_shapes.append(new_shape)

        if param_name == 'batch1':
            batch1_shape = new_shape
        elif param_name == 'batch2':
            batch2_shape = new_shape
        elif param_name == 'self':
            self_shape = new_shape

    return new_input_shapes, batch1_shape, batch2_shape, self_shape


def _parse_keepdim_attr(new_row: pd.Series) -> bool:
    """解析 keepdim 属性值"""
    keep_dim_value = False
    for col in new_row.index:
        # 提前过滤不符合条件的列，减少缩进
        if not (col.startswith('attr_name') and str(new_row[col]).lower() == 'keepdim'):
            continue

        value_col = col.replace('attr_name', 'attr_value')
        if value_col not in new_row.index:
            continue

        keep_dim_str = str(new_row[value_col])
        if 'True' in keep_dim_str or '1' in keep_dim_str:
            keep_dim_value = True
            
    return keep_dim_value


def _update_dim_attr(new_row: pd.Series, new_dim: list):
    """更新 dim 属性值"""
    for col in new_row.index:
        if col.startswith('attr_name') and str(new_row[col]).lower() == 'dim':
            dtype_col = col.replace('attr_name', 'attr_dtype')
            value_col = col.replace('attr_name', 'attr_value')
            if dtype_col in new_row.index and value_col in new_row.index:
                new_row[value_col] = str(new_dim)


def _get_reduce_new_dim(dim_scenario: str, self_shape: list, original_dim: list) -> list:
    """获取规约场景的 new_dim"""
    num_dims = len(self_shape)
    if dim_scenario == 'non_reduce':
        return original_dim if original_dim else [1] if num_dims > 1 else [0]
    elif dim_scenario == 'reduce':
        return original_dim if original_dim else [0]
    elif dim_scenario == 'all_reduce':
        return list(range(num_dims)) if num_dims > 0 else []
    return []


def _build_reduce_output_shape(self_shape: list, new_dim: list, keep_dim_value: bool) -> list:
    """构建规约操作的输出形状"""
    new_out_shape = []
    for i, s in enumerate(self_shape):
        if i in new_dim:
            new_out_shape.append(1 if keep_dim_value else s)
        else:
            new_out_shape.append(s)
    return new_out_shape


def _calculate_output_shape(params: OutputShapeParams):
    """计算最终输出形状（核心逻辑）"""
    is_matmul_like = (params.batch1_shape and params.batch2_shape and
                        len(params.batch1_shape) == 3 and
                        len(params.batch2_shape) == 3)
    is_reduce_like = params.dim_scenario and params.self_shape

    if is_matmul_like:
        return [params.batch1_shape[1], params.batch2_shape[2]]
    elif is_reduce_like:
        new_dim = _get_reduce_new_dim(params.dim_scenario, params.self_shape, [])
        _update_dim_attr(params.new_row, new_dim)
        
        if params.dim_scenario in ['non_reduce', 'reduce']:
            return _build_reduce_output_shape(params.self_shape, new_dim, params.keep_dim_value)
        elif params.dim_scenario == 'all_reduce':
            return [1] * len(params.self_shape) if params.keep_dim_value else []
    return params.self_shape if params.self_shape else params.default_output_shape


def generate_empty_tensor_cases(
    df: pd.DataFrame,
    constraints: Dict,
    param_def: Dict,
    num_cases: int = 5,
    verbose: bool = False
) -> pd.DataFrame:
    """
    生成空tensor测试用例
    
    设计原则：
    - 每个空tensor场景生成1个用例，覆盖场景本身
    - 不与dtype、attr等因子组合
    - 空tensor用例数 = 算子支持的空tensor场景数
    """
    if verbose:
        logging.info("分析算子空tensor场景...")
    
    scenarios = analyze_operator_scenario(constraints)
    
    if not scenarios:
        if verbose:
            logging.info("无空tensor场景，跳过生成")
        return pd.DataFrame()
    
    if verbose:
        logging.info(f"发现 {len(scenarios)} 个空tensor场景:")
        for scenario in scenarios:
            logging.info(f"  - {scenario['name']}: {scenario['description']}")
    
    template_idx = select_template_cases(df, num_templates=1)[0]
    template_row = df.iloc[template_idx]
    
    if verbose:
        logging.info(f"选择1个模板用例")
    
    empty_cases = []
    
    for i, scenario in enumerate(scenarios):
        empty_case = generate_empty_case(template_row, scenario, i + 1)
        empty_cases.append(empty_case)
    
    empty_df = pd.DataFrame(empty_cases)
    
    if verbose:
        logging.info(f"生成 {len(empty_cases)} 个空tensor用例")
    
    return empty_df


def save_test_cases(df: pd.DataFrame, output_path: str):
    """保存测试用例"""
    df.to_csv(output_path, index=False)
    logging.info(f"已保存到: {output_path}")


def main():
    """主函数"""
    args = parse_arguments()
    
    if args.verbose:
        logging.info(f"加载测试用例: {args.test_cases_csv}")
    
    df = load_test_cases(args.test_cases_csv)
    
    if args.verbose:
        logging.info(f"加载约束定义: {args.constraints_yaml}")
    
    constraints = load_constraints(args.constraints_yaml)
    
    param_def = {}
    if args.param_def:
        if args.verbose:
            logging.info(f"加载参数定义: {args.param_def}")
        param_def = load_param_def(args.param_def)
    
    empty_df = generate_empty_tensor_cases(
        df, constraints, param_def,
        num_cases=args.num_empty,
        verbose=args.verbose
    )
    
    non_empty_df = df[~df['case_name'].str.contains('EmptyTensor', na=False)]
    result_df = pd.concat([empty_df, non_empty_df], ignore_index=True)
    
    output_path = args.output or args.test_cases_csv
    save_test_cases(result_df, output_path)
    
    if args.verbose:
        logging.info(f"原有用例: {len(non_empty_df)}")
        logging.info(f"空tensor用例: {len(empty_df)}")
        logging.info(f"合计: {len(result_df)}")


if __name__ == '__main__':
    main()