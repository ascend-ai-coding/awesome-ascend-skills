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
测试用例生成脚本
支持 L0（单因子覆盖）、L1（两两组合覆盖）和 L2（异常场景用例）
"""

import argparse
import sys
import re
import random
import json
from pathlib import Path
from dataclasses import dataclass
import logging
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import numpy as np
import yaml
from copy import deepcopy
import pandas as pd
from ast import literal_eval
from utils import (
    normalize_dtype,
)
from generate_empty_tensor_cases_derive import (
    load_constraints as load_constraints_derive,
    analyze_operator_scenario,
    generate_empty_tensor_cases as generate_empty_tensor_derive_cases,
)


@dataclass
class CreateCsvParams:
    args: any
    levels: list
    constraints_def: any
    param_def: any
    factors: dict
    df: any
    all_factor_values: dict
    md_content: str


@dataclass
class SaveCaseParams:
    level: str
    args: any
    report: any
    case_df_ttk: any
    report_file: str
    case_file: str
    aclnn_name: str


@dataclass
class SaveResultsParams:
    report: dict
    case_df_ttk: any
    output_dir: str
    report_file: str
    case_file: str
    aclnn_name: str
    verbose: bool


@dataclass
class TensorProcessData:
    """用于封装 _process_acl_tensor 的输入输出集合，减少参数"""
    input_shapes: list
    input_dtypes: list
    input_ranges: list
    output_shapes: list
    output_dtypes: list
    output_indexes: list
    tensor_list_lengths: list


@dataclass
class TtkCaseData:
    """用于封装构建 TTK Case 所需的所有数据"""
    input_shapes: list
    input_dtypes: list
    input_ranges: list
    output_shapes: list
    output_dtypes: list
    output_indexes: list = None
    scalar_dtypes_list: list = None
    attrs_dict: dict = None
    tensor_list_lengths: list = None

    def __post_init__(self):
        if self.output_indexes is None:
            self.output_indexes = []
        if self.scalar_dtypes_list is None:
            self.scalar_dtypes_list = []
        if self.attrs_dict is None:
            self.attrs_dict = {}
        if self.tensor_list_lengths is None:
            self.tensor_list_lengths = []


@dataclass
class ErrorCaseSummary:
    dtype_error_cases: list
    dtype_combo_cases: list
    dim_error_cases: list
    format_error_cases: list
    shape_error_cases: list
    empty_tensor_cases: list
    attr_error_cases: list


@dataclass
class InputTensorData:
    input_shapes: list
    input_dtypes: list
    input_ranges: list


@dataclass
class TtkProcessContext:
    """封装 _process_single_ttk_case 所需的上下文参数"""
    row: any
    idx: int
    param_def: dict
    param_names: list
    aclnn_name: str
    case_level: str


@dataclass
class ErrorCaseConfig:
    """封装创建错误用例所需的所有配置参数"""
    param_def: dict
    aclnn_name: str
    param_name: str
    supported_dtype: str
    error_type: str
    error_param: str
    case_name: str
    shape_values: dict = None


# 全局常量：各 dtype 边界值定义
DTYPE_BOUNDARIES = {
    'float32': {'min': -3.4028235e+38, 'max': 3.4028235e+38, 'special': ['inf', '-inf', 'nan']},
    'float16': {'min': -65504.0, 'max': 65504.0, 'special': ['inf', '-inf', 'nan']},
    'float64': {'min': -1.7976931348623157e+308, 'max': 1.7976931348623157e+308, 'special': ['inf', '-inf', 'nan']},
    'bfloat16': {'min': -3.38e+38, 'max': 3.38e+38, 'special': ['inf', '-inf', 'nan']},
    'int32': {'min': -2147483648, 'max': 2147483647, 'special': []},
    'int64': {'min': -9223372036854775808, 'max': 9223372036854775807, 'special': []},
    'int16': {'min': -32768, 'max': 32767, 'special': []},
    'int8': {'min': -128, 'max': 127, 'special': []},
    'uint8': {'min': 0, 'max': 255, 'special': []},
    'bool': {'min': 0, 'max': 1, 'special': []},
    'complex64': {'min': None, 'max': None, 'special': ['nan']},
    'complex128': {'min': None, 'max': None, 'special': ['nan']},
}

# ============ 全量数据类型和格式定义（用于异常用例生成）============

ALL_DTYPES = [
    'float32', 'float16', 'float64', 'bfloat16',
    'int32', 'int64', 'int16', 'int8', 'uint8',
    'uint16', 'uint32', 'uint64',
    'bool', 'complex64', 'complex128',
    'float4_e2m1', 'float4_e1m2', 'float8_e4m3fn', 'float8_e5m2'
]

ALL_FORMATS = ['ND', 'NCHW', 'NHWC', 'NCDHW', 'NDHWC', 'HWCN', 'NCW', 'NWC', 'NC1HWC0']

OVERFLOW_DIMENSIONS = [9, 10, 11, 12]

INCOMPATIBLE_DTYPE_COMBOS = [
    ('int64', 'float16'),
    ('uint64', 'int32'),
    ('complex64', 'float32'),
    ('complex128', 'float16'),
    ('uint64', 'int64'),
    ('uint32', 'int32'),
]


def main():
    """主函数"""
    # 1. 解析参数
    args = parse_arguments()
    
    # 2. 解析级别
    levels = parse_levels(args.level)
    
    # 3. 参数校验（冲突检测）
    validate_parameters(args, levels)

    # 4. 加载约束定义（L2需要）
    constraints_def = None
    if 'L2' in levels:
        constraints_def = load_yaml(args.constraints_file)
        if args.verbose:
            print(f"[INFO] 加载约束定义: {args.constraints_file}")
    
    # 5. 加载数据（只加载一次，批量生成时共享）
    param_def, factors, df = load_data(args)
    
    # 6. 提取因子值（只提取一次，批量生成时共享）
    all_factor_values = extract_all_factor_values(factors, df)
    
    # 7. 获取接口文档内容（用于判断搬运类算子）
    md_content = load_md_content(args, param_def)
    
    if args.verbose:
        print(f"[INFO] 将生成级别: {', '.join(levels)}\n")
    
    # 8. 按级别生成用例
    # 封装参数
    csv_params = CreateCsvParams(
        args=args,
        levels=levels,
        constraints_def=constraints_def,
        param_def=param_def,
        factors=factors,
        df=df,
        all_factor_values=all_factor_values,
        md_content=md_content
    )

    # 调用函数
    create_csv(csv_params)

    if args.verbose:
        print(f"[INFO] 完成! 共生成 {len(levels)} 个级别的用例")


def create_csv(params: CreateCsvParams):
    for level in params.levels:
        if params.args.verbose:
            logging.info(f"{'='*10} 开始生成 {level} 用例 {'='*10}")

        # L2 异常用例
        if level == 'L2':
            _handle_l2_cases(params.args, params.param_def, params.factors, params.constraints_def)
            continue

        # L0 / L1 用例筛选
        selected_indices, report = _generate_l0_l1_cases(
            level, params.df, params.all_factor_values, params.args
        )

        # 生成输出文件
        aclnn_name = extract_aclnn_name(params.args)
        report_file, case_file = get_output_filenames(
            params.args.report_output, params.args.case_output, level, len(params.levels) > 1, aclnn_name
        )

        # 转换格式
        selected_df = params.df.iloc[selected_indices].reset_index(drop=True)
        case_df_ttk = convert_to_ttk_format(selected_df, params.param_def, aclnn_name, level)

        # 空 tensor 用例
        if not params.args.skip_empty and params.args.empty_count > 0 and level in ('L0', 'L1'):
            constraints_path = infer_constraints_path(params.args)
            case_df_ttk = empty_tensor_create(constraints_path, case_df_ttk, params.param_def, params.args)

        # 保存结果
        # 封装参数
        save_params = SaveCaseParams(
            level=level,
            args=params.args,
            report=report,
            case_df_ttk=case_df_ttk,
            report_file=report_file,
            case_file=case_file,
            aclnn_name=aclnn_name
        )
        # 调用
        _save_case_results(save_params)

        if params.args.verbose:
            logging.info(f"{'='*10} {level} 用例生成完成 {'='*10}\n")


def _handle_l2_cases(args, param_def, factors, constraints_def):
    """处理 L2 异常用例生成与保存"""
    error_cases = generate_error_cases(param_def, factors, constraints_def, args.verbose)
    case_df_ttk = create_error_ttk_dataframe(error_cases, param_def)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aclnn_name = extract_aclnn_name(args)

    case_file = args.case_output if args.case_output else "l2_test_cases.csv"
    base_name = Path(case_file).stem
    ttk_csv = f"{aclnn_name}_{base_name}.csv"
    case_path = output_dir / ttk_csv

    case_df_ttk.to_csv(case_path, index=False)
    if args.verbose:
        print(f"[INFO] 生成异常用例: {case_path} ({len(case_df_ttk)}条)")


def _generate_l0_l1_cases(level, df, all_factor_values, args):
    """生成 L0/L1 用例并返回选中索引 + 报告"""
    if level == 'L0':
        selected_indices, coverage_info = select_cases_L0(df, all_factor_values, args.verbose)
        report = generate_L0_report(all_factor_values, coverage_info)
    else:
        pairwise_combinations = generate_pairwise_combinations(all_factor_values)
        if args.verbose:
            print(f"[INFO] 两两组合数: {len(pairwise_combinations)}")

        selected_indices, coverage_info = select_cases_L1(
            df, pairwise_combinations, all_factor_values, args.verbose, args.sample_size
        )
        selected_indices = complete_case(selected_indices, args)
        report = generate_L1_report(
            all_factor_values, pairwise_combinations, coverage_info, args.target_count
        )
    return selected_indices, report


def _save_case_results(params: SaveCaseParams):
    """保存 L0/L1 用例与报告"""
    output_dir = Path(params.args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if params.level == 'L0':
        # 封装参数
        save_params = SaveResultsParams(
            report=params.report,
            case_df_ttk=params.case_df_ttk,
            output_dir=params.args.output_dir,
            report_file=params.report_file,
            case_file=params.case_file,
            aclnn_name=params.aclnn_name,
            verbose=params.args.verbose
        )

        # 调用函数
        save_results(save_params)
    else:
        ttk_csv = f"{params.aclnn_name}_{params.case_file.split('.')[0]}.csv"
        case_path = output_dir / ttk_csv
        params.case_df_ttk.to_csv(case_path, index=False)
        if params.args.verbose:
            logging.info(f"生成用例: {case_path} ({len(params.case_df_ttk)}条)")


def complete_case(selected_indices, args):
    if len(selected_indices) < args.target_count:
        if args.verbose:
            print(f"[INFO] 补齐用例: {len(selected_indices)} -> {args.target_count}")
        selected_indices = pad_cases(
            selected_indices, args.target_count, args.seed
        )
    return selected_indices


def empty_tensor_create(constraints_path, case_df_ttk, param_def, args):
    if constraints_path:
        empty_df = generate_empty_cases_inline(
            case_df_ttk, constraints_path, param_def, args.empty_count, args.verbose
        )
        if len(empty_df) > 0:
            case_df_ttk = pd.concat([empty_df, case_df_ttk], ignore_index=True)
            if args.verbose:
                print(f"[INFO] 合计用例: {len(case_df_ttk)}条（含{len(empty_df)}条空tensor）")
    return case_df_ttk


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='测试用例生成脚本（支持L0和L1）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成L0用例
  python generate_test_cases.py param.yaml factors.yaml values.csv output/ --level L0
  
  # 生成L1用例（500条）
  python generate_test_cases.py param.yaml factors.yaml values.csv output/ --level L1 --target-count 500
  
  # 批量生成L0和L1
  python generate_test_cases.py param.yaml factors.yaml values.csv output/ --level L0 L1
  
  # 自定义输出文件名
  python generate_test_cases.py param.yaml factors.yaml values.csv output/ --level L0 --report-output my_report.yaml
        """
    )
    
    parser.add_argument('param_def', help='参数定义YAML文件（03_参数定义.yaml）')
    parser.add_argument('factors', help='测试因子YAML文件（04_测试因子.yaml）')
    parser.add_argument('values', help='因子值CSV文件（07_因子值.csv）')
    parser.add_argument('output_dir', help='输出目录')
    
    parser.add_argument('--level', nargs='+', required=True,
                       help='用例级别（必填，支持多个）: L0=单因子覆盖，L1=两两组合覆盖，L2=异常场景用例')
    parser.add_argument('--aclnn-name', help='算子名称（默认从文件路径中提取）')
    parser.add_argument('--target-count', type=int, default=500,
                       help='L1目标用例数量（默认500，仅L1有效）')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='L1每次迭代采样的候选用例数量（默认1000，仅L1有效，用于加速大数据集）')
    parser.add_argument('--seed', type=int, help='随机数种子（用于复现L1补齐，仅L1有效）')
    parser.add_argument('--report-output', help='覆盖度报告文件名（默认: {level}_coverage_report.yaml）')
    parser.add_argument('--case-output', help='测试用例文件名（默认: {level}_test_cases.csv）')
    parser.add_argument('--md-file', help='接口文档路径（用于判断搬运类算子，可选）')
    parser.add_argument('--constraints-file', help='约束定义YAML文件（05_约束定义.yaml，L2级别必填）')
    parser.add_argument('--empty-count', type=int, default=5, help='空tensor用例数量（默认5，L0和L1有效）')
    parser.add_argument('--skip-empty', action='store_true', help='跳过空tensor用例生成')
    parser.add_argument('--verbose', action='store_true', help='详细输出模式')
    
    return parser.parse_args()


def parse_levels(level_arg):
    """解析 --level 参数"""
    levels = []
    
    for item in level_arg:
        # 支持逗号分隔
        if ',' in item:
            levels.extend([l.strip() for l in item.split(',')])
        else:
            levels.append(item.strip())
    
    # 去重并排序（L0在前，L2最后）
    levels = sorted(set(levels), key=lambda x: (x == 'L2', x != 'L0', x))
    
    # 验证
    valid_levels = {'L0', 'L1', 'L2'}
    invalid = set(levels) - valid_levels
    if invalid:
        print(f"[ERROR] 无效的级别: {invalid}，仅支持 L0、L1 和 L2")
        sys.exit(1)
    
    return levels


def validate_parameters(args, levels):
    """验证参数冲突"""
    errors = []

    # L2 级别必须提供 constraints_file
    _validate_l2_requirements(levels, args, errors)

    # 仅 L0 时，检查 L1 专用参数
    _validate_l0_exclusive_params(levels, args, errors)

    # 如果有错误，打印并退出
    if errors:
        _print_parameter_errors(errors)
        sys.exit(1)


def _validate_l2_requirements(levels, args, errors):
    """检查 L2 级别必须的参数"""
    if 'L2' in levels and not args.constraints_file:
        errors.append({
            'param': '--constraints-file',
            'value': 'None',
            'reason': 'L2 级别必须提供 --constraints-file 参数',
            'detail': 'L2 需要从约束定义文件提取支持的dtype/format等信息',
            'solution': ['添加 --constraints-file ops/aclnnAdds/tests/st/design/05_约束定义.yaml']
        })


def _validate_l0_exclusive_params(levels, args, errors):
    """仅生成 L0 时，检查 L1 专用参数是否被误用"""
    if levels == ['L0']:
        if args.target_count != 500:
            errors.append({
                'param': '--target-count',
                'value': args.target_count,
                'reason': 'L0 不支持 --target-count 参数',
                'detail': 'L0 的用例数量由算法自动确定（覆盖所有因子值的最小集合）',
                'solution': ['移除 --target-count 参数', '或使用 --level L1 生成 L1 用例']
            })

        if args.seed is not None:
            errors.append({
                'param': '--seed',
                'value': args.seed,
                'reason': 'L0 不支持 --seed 参数',
                'detail': 'L0 不需要补齐，因此无需随机数种子',
                'solution': ['移除 --seed 参数', '或使用 --level L1 生成 L1 用例']
            })


def _print_parameter_errors(errors):
    """统一打印参数错误信息并退出"""
    logging.info("\n" + "=" * 80)
    print("参数冲突错误")
    print("=" * 80 + "\n")

    for error in errors:
        print(f"[ERROR] {error['param']}={error['value']} 无效")
        print(f"        {error['reason']}")
        print(f"        {error['detail']}")
        print("        解决方法：")
        for i, sol in enumerate(error['solution'], 1):
            print(f"        {i}. {sol}")
        print()

    print("=" * 80)
    print("脚本已退出（错误码：1）")
    print("=" * 80 + "\n")


def infer_constraints_path(args):
    """从参数定义路径推断约束定义文件路径"""
    if args.constraints_file:
        return args.constraints_file
    param_path = Path(args.param_def)
    design_dir = param_path.parent

    constraints_file = design_dir / "05_约束定义.yaml"
    if constraints_file.exists():
        return str(constraints_file)
    
    return None


def generate_empty_cases_inline(case_df, constraints_path, param_def, num_empty, verbose):
    """内联生成空tensor用例（L0和L1都生成）"""
    try:
        constraints = load_constraints_derive(constraints_path)
        scenarios = analyze_operator_scenario(constraints)
        if not scenarios:
            if verbose:
                print("[INFO] 无空tensor场景，跳过生成")
                return pd.DataFrame()
        if verbose:
            print(f"[INFO] 分析到 {len(scenarios)} 个空tensor场景")
        empty_df = generate_empty_tensor_derive_cases(
            case_df, constraints, param_def, num_cases=num_empty, verbose=verbose
        )
        return empty_df
    except Exception as e:
        if verbose:
            print(f"[WARN] 空tensor用例生成失败: {e}")
        return pd.DataFrame()


def load_data(args):
    """加载数据文件"""
    param_def = load_yaml(args.param_def)
    factors = load_yaml(args.factors)
    df = pd.read_csv(args.values)
    
    if args.verbose:
        print(f"[INFO] 加载参数定义: {args.param_def}")
        print(f"[INFO] 加载测试因子: {args.factors}")
        print(f"[INFO] 加载因子值: {args.values} ({len(df)}个用例)\n")
    
    return param_def, factors, df


def load_md_content(args, param_def):
    """
    加载接口文档内容（用于判断搬运类算子）
    
    Args:
        args: 命令行参数
        param_def: 参数定义
    
    Returns:
        str: md 文件内容（如果找不到则返回 None）
    """
    md_file = _resolve_md_file_path(args, param_def)
    
    if md_file is None:
        return None
    
    return _read_md_file_content(md_file, args.verbose)


def _resolve_md_file_path(args, param_def):
    """确定 md 文件路径"""
    if args.md_file:
        return Path(args.md_file)
    
    return _infer_md_file_from_param_def(args.param_def, param_def)


def _infer_md_file_from_param_def(param_def_path, param_def):
    """从参数定义文件路径推断 md 文件路径"""
    param_def_path = Path(param_def_path)
    
    ops_idx = _find_ops_index(param_def_path.parts)
    if ops_idx is None or ops_idx + 1 >= len(param_def_path.parts):
        return None
    
    operator_name = param_def_path.parts[ops_idx + 1]
    docs_dir = param_def_path.parents[3] / operator_name / 'docs'
    
    return _find_md_file_in_docs(docs_dir, param_def)


def _find_ops_index(parts):
    """在路径中找到 ops 目录的索引"""
    for i, part in enumerate(parts):
        if part == 'ops':
            return i
    return None


def _find_md_file_in_docs(docs_dir, param_def):
    """在 docs 目录中查找 md 文件"""
    aclnn_name = param_def.get('aclnn_name', '')
    
    if aclnn_name:
        return docs_dir / f"{aclnn_name}.md"
    
    if not docs_dir.exists():
        return None
    
    md_files = list(docs_dir.glob('aclnn*.md'))
    return md_files[0] if md_files else None


def _read_md_file_content(md_file, verbose):
    """读取 md 文件内容"""
    if not md_file.exists():
        return None
    
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        if verbose:
            print(f"[INFO] 加载接口文档: {md_file}")
        return content
    except Exception as e:
        if verbose:
            print(f"[WARN] 无法加载接口文档 {md_file}: {e}")
        return None


# ============ L0 相关函数 ============

def select_cases_L0(df, all_factor_values, verbose=False):
    """
    L0: 使用贪心算法选择覆盖所有单个因子值的最小用例集
    
    Args:
        df: 因子值 DataFrame
        all_factor_values: 所有因子及其值
        verbose: 是否输出详细信息
    
    Returns:
        Tuple[List[int], Dict]: (选中的用例索引列表, 覆盖度信息)
    """
    if verbose:
        print(f"[INFO] 提取因子: {len(all_factor_values)}个因子, {sum(len(v) for v in all_factor_values.values())}个因子值")
    
    selected_indices = []
    uncovered = deepcopy(all_factor_values)
    covered = {k: set() for k in all_factor_values.keys()}
    
    iteration = 0
    while any(uncovered.values()):
        iteration += 1
        best_idx = None
        best_new_coverage = 0
        best_covered = {}
        
        for idx, row in df.iterrows():
            if idx in selected_indices:
                continue
            
            new_coverage = 0
            current_covered = {}
            
            for factor_name, required_values in uncovered.items():
                if factor_name not in row.index:
                    continue
                
                value = row[factor_name]
                value_key = make_hashable(value)
                
                if value_key in required_values:
                    new_coverage += 1
                    current_covered[factor_name] = value_key
            
            if new_coverage > best_new_coverage:
                best_new_coverage = new_coverage
                best_idx = idx
                best_covered = current_covered
        
        if best_idx is None or best_new_coverage == 0:
            break
        
        selected_indices.append(best_idx)
        
        for factor_name, value in best_covered.items():
            covered[factor_name].add(value)
            uncovered[factor_name].discard(value)
        
        if verbose and iteration % 5 == 0:
            total_uncovered = sum(len(v) for v in uncovered.values())
            print(f"[INFO] L0迭代{iteration}: 选择用例{best_idx}, 新覆盖{best_new_coverage}个因子值, 剩余{total_uncovered}")
    
    if verbose:
        total_values = sum(len(v) for v in all_factor_values.values())
        covered_count = sum(len(v) for v in covered.values())
        print(f"[INFO] 完成用例选择: {len(selected_indices)}个用例")
        print(f"[INFO] 因子值覆盖: {covered_count}/{total_values} ({covered_count/total_values*100:.2f}%)")
    
    coverage_info = {
        'covered_values': covered,
        'uncovered_values': uncovered,
        'total_factors': len(all_factor_values),
        'total_values': sum(len(v) for v in all_factor_values.values()),
        'covered_count': sum(len(v) for v in covered.values())
    }
    
    return selected_indices, coverage_info


def generate_L0_report(all_factor_values, coverage_info):
    """生成L0覆盖度报告"""
    total_values = sum(len(v) for v in all_factor_values.values())
    covered_count = coverage_info['covered_count']
    uncovered_count = total_values - covered_count
    
    report = {
        'summary': {
            'level': 'L0',
            'strategy': 'single_factor_coverage',
            'total_factors': coverage_info['total_factors'],
            'total_factor_values': total_values,
            'covered_factor_values': covered_count,
            'uncovered_factor_values': uncovered_count,
            'coverage_rate': f"{covered_count/total_values*100:.2f}%" if total_values > 0 else "N/A",
            'minimal_case_count': coverage_info['covered_count']
        },
        'details': {}
    }
    
    for factor_name, required_values in all_factor_values.items():
        covered = coverage_info['covered_values'].get(factor_name, set())
        uncovered = coverage_info['uncovered_values'].get(factor_name, set())
        
        report['details'][factor_name] = {
            'target_values': sorted([str(v) for v in required_values]),
            'covered_values': sorted([str(v) for v in covered]),
            'uncovered_values': sorted([str(v) for v in uncovered]),
            'coverage_rate': f"{len(covered)/len(required_values)*100:.2f}%" if required_values else "N/A"
        }
    
    return report


# ============ L1 相关函数 ============

def generate_pairwise_combinations(all_factor_values):
    """
    生成所有因子的两两组合（优化版：只考虑离散因子）
    
    Args:
        all_factor_values: Dict[str, List[Any]] - 因子名 -> 因子值列表
    
    Returns:
        Set[Tuple]: 所有两两组合集合
    
    优化说明：
        1. 只保留离散因子（dtype, dimensions, exist, format, 枚举值等）
        2. 过滤派生因子（.value, .shape）和连续值因子（.value_range）
        3. 过滤固定值因子（只有1个值的因子）
        
    性能提升：
        - 原始：192,810个组合（包含value_range）
        - 优化后：~500个组合（只有离散因子）
        - 性能提升：~400倍
    """
    # 定义离散因子模式
    discrete_patterns = [
        '.dtype',        # 数据类型（离散）
        '.dimensions',   # 维度数（离散）
        '.exist',        # 存在性（离散）
        '.format',       # 数据格式（离散）
        'cubeMathType.value',  # 枚举值（离散）
    ]
    
    def is_discrete_factor(factor_name):
        """判断是否为离散因子"""
        # 排除派生因子和连续值因子
        if factor_name.endswith('.value') and not factor_name == 'cubeMathType.value':
            return False
        if factor_name.endswith('.shape'):
            return False
        if factor_name.endswith('.value_range'):
            return False
        
        # 保留离散因子
        for pattern in discrete_patterns:
            if pattern in factor_name:
                return True
        
        return False
    
    # 过滤离散因子
    discrete_factors = {
        name: values 
        for name, values in all_factor_values.items()
        if is_discrete_factor(name)
    }
    
    # 过滤只有1个值的因子
    multi_value_factors = {
        name: values 
        for name, values in discrete_factors.items() 
        if len(values) > 1
    }
    
    fixed_value_factors = {
        name: values 
        for name, values in discrete_factors.items() 
        if len(values) == 1
    }
    
    # 统计过滤的因子
    filtered_continuous = {
        name: values 
        for name, values in all_factor_values.items()
        if name.endswith('.value_range') or (name.endswith('.value') and name != 'cubeMathType.value') or name.endswith('.shape')
    }
    
    if fixed_value_factors:
        print(f"[INFO] 过滤固定值因子: {len(fixed_value_factors)}个")
    if filtered_continuous:
        print(f"[INFO] 过滤连续值/派生因子: {len(filtered_continuous)}个")
    print(f"[INFO] 保留离散多值因子: {len(multi_value_factors)}个")
    
    factor_names = sorted(multi_value_factors.keys())
    pairwise_combinations = set()
    
    for i in range(len(factor_names)):
        for j in range(i + 1, len(factor_names)):
            factor1_name = factor_names[i]
            factor2_name = factor_names[j]
            
            values1 = multi_value_factors[factor1_name]
            values2 = multi_value_factors[factor2_name]
            
            for v1 in values1:
                for v2 in values2:
                    v1_key = make_hashable(v1)
                    v2_key = make_hashable(v2)
                    
                    pair = (
                        (factor1_name, v1_key),
                        (factor2_name, v2_key)
                    )
                    pairwise_combinations.add(pair)
    
    return pairwise_combinations


def select_cases_L1(df, pairwise_combinations, all_factor_values, verbose=False, sample_size=None):
    """
    L1: 使用贪心算法选择覆盖所有两两组合的用例集（优化版：只考虑离散因子）
    
    Args:
        df: 因子值 DataFrame
        pairwise_combinations: 所有两两组合
        all_factor_values: 所有因子及其值
        verbose: 是否输出详细信息
        sample_size: 每次迭代采样的候选用例数量（None表示全部考虑）
    
    Returns:
        Tuple[List[int], Dict]: (选中的用例索引列表, 覆盖度信息)
    
    优化说明：
        1. 只考虑离散因子（dtype, dimensions, exist, format, 枚举值等）
        2. 过滤连续值因子（value_range）和派生因子（value, shape）
    """
    # 定义离散因子模式
    discrete_patterns = [
        '.dtype',        # 数据类型（离散）
        '.dimensions',   # 维度数（离散）
        '.exist',        # 存在性（离散）
        '.format',       # 数据格式（离散）
        'cubeMathType.value',  # 枚举值（离散）
    ]
    
    def is_discrete_factor(factor_name):
        """判断是否为离散因子"""
        # 排除派生因子和连续值因子
        if factor_name.endswith('.value') and not factor_name == 'cubeMathType.value':
            return False
        if factor_name.endswith('.shape'):
            return False
        if factor_name.endswith('.value_range'):
            return False
        
        # 保留离散因子
        for pattern in discrete_patterns:
            if pattern in factor_name:
                return True
        
        return False
    
    # 过滤离散因子
    discrete_factors = {
        name: values 
        for name, values in all_factor_values.items()
        if is_discrete_factor(name)
    }
    
    # 过滤只有1个值的因子
    multi_value_factors = {
        name: values 
        for name, values in discrete_factors.items() 
        if len(values) > 1
    }
    
    if verbose:
        print(f"[INFO] 提取离散因子: {len(multi_value_factors)}个（已过滤连续值和派生因子）")
        print(f"[INFO] 因子值总数: {sum(len(v) for v in multi_value_factors.values())}")
    
    selected_indices = []
    uncovered = deepcopy(pairwise_combinations)
    covered = set()
    
    factor_names = [col for col in df.columns if col in multi_value_factors]
    all_indices = set(df.index.tolist())
    
    iteration = 0
    while uncovered:
        iteration += 1
        best_idx = None
        best_new_coverage = 0
        best_covered_pairs = set()
        
        # 采样候选用例以加速
        candidate_indices = list(all_indices - set(selected_indices))
        if sample_size and len(candidate_indices) > sample_size:
            candidate_indices = random.sample(candidate_indices, sample_size)
        
        for idx in candidate_indices:
            row = df.loc[idx]
            
            covered_pairs = set()
            
            for i in range(len(factor_names)):
                for j in range(i + 1, len(factor_names)):
                    factor1_name = factor_names[i]
                    factor2_name = factor_names[j]
                    
                    v1 = row[factor1_name]
                    v2 = row[factor2_name]
                    
                    v1_key = make_hashable(v1)
                    v2_key = make_hashable(v2)
                    
                    pair = (
                        (factor1_name, v1_key),
                        (factor2_name, v2_key)
                    )
                    
                    if pair in uncovered:
                        covered_pairs.add(pair)
            
            if len(covered_pairs) > best_new_coverage:
                best_new_coverage = len(covered_pairs)
                best_idx = idx
                best_covered_pairs = covered_pairs
        
        if best_idx is None or best_new_coverage == 0:
            break
        
        selected_indices.append(best_idx)
        covered.update(best_covered_pairs)
        uncovered -= best_covered_pairs
        
        if verbose and iteration % 10 == 0:
            print(f"[INFO] L1迭代{iteration}: 选择用例{best_idx}, 新覆盖{best_new_coverage}个两两组合, 剩余{len(uncovered)}")
    
    if verbose:
        print(f"[INFO] 完成用例选择: {len(selected_indices)}个用例")
        print(f"[INFO] 两两组合覆盖: {len(covered)}/{len(pairwise_combinations)} ({len(covered)/len(pairwise_combinations)*100:.2f}%)")
    
    coverage_info = {
        'total_pairwise': len(pairwise_combinations),
        'covered_pairwise': len(covered),
        'uncovered_pairwise': len(uncovered),
        'coverage_rate': len(covered) / len(pairwise_combinations) * 100 if pairwise_combinations else 0,
        'selected_count': len(selected_indices)
    }
    
    return selected_indices, coverage_info


def pad_cases(selected_indices, target_count, seed=None):
    """
    补齐用例到目标数量（L1专用）
    
    Args:
        selected_indices: 已选中的用例索引列表
        target_count: 目标用例数量
        seed: 随机数种子
    
    Returns:
        List[int]: 补齐后的用例索引列表
    """
    if len(selected_indices) >= target_count:
        return selected_indices[:target_count]
    
    if seed is not None:
        random.seed(seed)
    
    need_pad = target_count - len(selected_indices)
    padded_indices = selected_indices.copy()
    
    for _ in range(need_pad):
        random_idx = random.choice(selected_indices)
        padded_indices.append(random_idx)
    
    return padded_indices


def generate_L1_report(all_factor_values, pairwise_combinations, coverage_info, target_count):
    """生成L1覆盖度报告"""
    selected_count = coverage_info['selected_count']
    padded_count = max(0, target_count - selected_count)
    
    report = {
        'summary': {
            'level': 'L1',
            'strategy': 'pairwise_coverage',
            'total_factors': len(all_factor_values),
            'total_factor_values': sum(len(v) for v in all_factor_values.values()),
            'total_pairwise_combinations': coverage_info['total_pairwise'],
            'covered_pairwise_combinations': coverage_info['covered_pairwise'],
            'uncovered_pairwise_combinations': coverage_info['uncovered_pairwise'],
            'coverage_rate': f"{coverage_info['coverage_rate']:.2f}%",
            'selected_case_count': selected_count,
            'target_case_count': target_count,
            'padded_case_count': padded_count
        },
        'factor_statistics': {}
    }
    
    for factor_name, values in all_factor_values.items():
        report['factor_statistics'][factor_name] = {
            'value_count': len(values),
            'values': sorted([str(v) for v in values])
        }
    
    return report


# ============ 共享工具函数 ============

def load_yaml(file_path):
    """加载YAML文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_all_factor_values(factors_dict, df=None):
    """
    提取所有需要覆盖的因子值
    
    策略：
    1. 从 CSV DataFrame 中提取每个因子的实际值（如果提供了 df）
    2. 过滤 .shape 因子（由 dimensions 动态生成）
    3. 处理 value_range_{dtype} 模式：
       - YAML 中定义的是按 dtype 分组的可选值（如 value_range_float16）
       - CSV 中的列名是统一的 value_range
       - 应该从 CSV 中提取 value_range 列的实际值，而不是合并 YAML 中的所有可能值
    
    Args:
        factors_dict: YAML 中的因子定义
        df: CSV DataFrame（如果提供，从中提取实际值）
    
    Returns:
        Dict[str, Set]: 因子名 -> 值集合
    """
    all_factor_values = {}
    value_range_dtype_pattern = re.compile(r'^(.+)\.value_range_([a-z0-9]+)$')
    
    # 收集所有因子名（过滤 shape，处理 value_range_{dtype}）
    factor_names_from_yaml = set()
    for param_name, param_info in factors_dict.items():
        factors = param_info.get('factors', {})
        
        for factor_name in factors.keys():
            # 过滤 shape 因子
            if '.shape' in factor_name:
                continue
            
            # 检测 value_range_{dtype} 模式
            match = value_range_dtype_pattern.match(factor_name)
            if match:
                # 转换为统一的 value_range 名称
                base_factor_name = f"{match.group(1)}.value_range"
                factor_names_from_yaml.add(base_factor_name)
            else:
                factor_names_from_yaml.add(factor_name)
    
    # 如果提供了 DataFrame，从中提取实际值
    if df is not None:
        for factor_name in factor_names_from_yaml:
            if factor_name in df.columns:
                # 从 CSV 列中提取唯一值
                unique_values = df[factor_name].unique()
                all_factor_values[factor_name] = set(
                    make_hashable(v) for v in unique_values if not pd.isna(v)
                )
    else:
        # 从 YAML 定义中提取（兼容旧行为）
        for param_name, param_info in factors_dict.items():
            factors = param_info.get('factors', {})
            
            for factor_name, factor_values in factors.items():
                if '.shape' in factor_name:
                    continue
                
                match = value_range_dtype_pattern.match(factor_name)
                if match:
                    base_factor_name = f"{match.group(1)}.value_range"
                    if base_factor_name not in all_factor_values:
                        all_factor_values[base_factor_name] = set()
                    all_factor_values[base_factor_name].update(
                        make_hashable(v) for v in factor_values
                    )
                else:
                    all_factor_values[factor_name] = set(
                        make_hashable(v) for v in factor_values
                    )
    
    return all_factor_values


def make_hashable(value):
    """将值转换为可哈希的类型"""
    if isinstance(value, list):
        return tuple(value)
    elif isinstance(value, dict):
        return tuple(sorted(value.items()))
    else:
        return value


def extract_aclnn_name(args):
    """从参数定义中提取算子名称"""
    # 1. 优先使用命令行指定的名称
    if args.aclnn_name:
        return args.aclnn_name
    
    # 2. 从参数定义中获取
    param_def_path = Path(args.param_def)
    try:
        with open(param_def_path, 'r', encoding='utf-8') as f:
            param_def = yaml.safe_load(f)
        if param_def.get('aclnn_name'):
            return param_def.get('aclnn_name')
    except Exception:
        pass
    
    # 3. 从文件路径中提取（查找包含 aclnn 的目录名）
    for part in param_def_path.parts:
        if part.startswith('aclnn'):
            return part
    
    return 'UnknownOperator'


def get_output_filenames(report_output, case_output, level, is_batch, aclnn_name):
    """
    确定输出文件名
    
    Args:
        report_output: 用户指定的报告文件名（可能为None）
        case_output: 用户指定的用例文件名（可能为None）
        level: 当前级别 (L0/L1)
        is_batch: 是否批量生成
    
    Returns:
        Tuple[str, str]: (报告文件名, 用例文件名)
    """
    # 报告文件名
    if report_output:
        report_name = report_output
        if is_batch:
            report_name = f"{aclnn_name}_{level}_{report_name}"
    else:
        if is_batch:
            report_name = f"{aclnn_name}_{level}_coverage_report.yaml"
        else:
            report_name = f"{aclnn_name}_{level.lower()}_coverage_report.yaml"
    
    # 用例文件名
    if case_output:
        case_name = case_output
        if is_batch:
            case_name = f"{level}_{case_name}"
    else:
        if is_batch:
            case_name = f"{level}_test_cases.csv"
        else:
            case_name = f"{level.lower()}_test_cases.csv"
    
    return report_name, case_name


def _parse_tensor_length(row, param_name):
    length_raw = row.get(f"{param_name}.length", 1)
    try:
        return int(float(str(length_raw))) if pd.notna(length_raw) else 1
    except (ValueError, TypeError):
        return 1


def _resolve_format(fmt_raw):
    if isinstance(fmt_raw, list):
        return random.choice(fmt_raw) if fmt_raw else fmt_raw
    return fmt_raw


def _is_attr_param(io_type, param_type):
    if io_type not in ['input', 'output']:
        return True
    if io_type == 'input' and param_type not in ['aclTensor', 'aclTensorList']:
        return True
    return False


def _process_attributes(row, param_def):
    attrs = {}
    attr_idx = 0
    for param_name, param_info in param_def.items():
        param_type = param_info.get('type', '')
        io_type = param_info.get('io_type', '')
        if not _is_attr_param(io_type, param_type):
            continue
        exist_col = f"{param_name}.exist"
        if exist_col in row.index and row[exist_col] == False:
            continue
        attr_prefix = '' if attr_idx == 0 else f'.{attr_idx}'
        attrs[f'attr_name{attr_prefix}'] = param_name
        attrs[f'attr_type{attr_prefix}'] = get_attr_type(param_type)
        attrs[f'attr_dtype{attr_prefix}'] = get_attr_dtype(
            param_type, row.get(f"{param_name}.dtype", '')
        )
        attrs[f'attr_value{attr_prefix}'] = format_attr_value(
            row.get(f"{param_name}.value", '')
        )
        attr_idx += 1
    return attrs


def parse_list_value(value):
    """解析列表值"""
    if isinstance(value, str):
        try:
            return literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def parse_format_value(fmt_raw):
    if isinstance(fmt_raw, list):
        return fmt_raw
    if not isinstance(fmt_raw, str):
        return [str(fmt_raw)]
    try:
        parsed = literal_eval(fmt_raw)
        if isinstance(parsed, list):
            if parsed and all(isinstance(item, str) for item in parsed):
                return parsed
            if parsed and all(isinstance(item, list) for item in parsed):
                return parsed
            return [fmt_raw]
        return [fmt_raw]
    except (ValueError, SyntaxError):
        return [fmt_raw]


def expand_for_tensorlist(values, length):
    """将值列表扩展为指定长度（用于TensorList的format/range/dtype等字段）
    
    对于字符串元素（如format）：['ND'] -> ['ND', 'ND', ...]
    对于非字符串元素（如range/dtype）：[[min, max]] -> [[min, max], [min, max], ...]
    """
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], str):
        return values * length
    elif isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
        return [values[0] for _ in range(length)]
    elif isinstance(values, list):
        if len(values) >= length:
            return values[:length]
        else:
            return values * (length // len(values)) + values[:length % len(values)]
    else:
        return [values] * length


def expand_format_for_tensorlist(fmt, length):
    return expand_for_tensorlist(fmt, length)


def _format_range_value(v):
    import math
    if isinstance(v, float) and v == 0.0:
        if math.copysign(1, v) < 0:
            return '-0'
        return '+0'
    if isinstance(v, str) and v in ('+0', '-0'):
        return v
    return str(v)


def format_list(items):
    formatted = []
    for item in items:
        if isinstance(item, (list, tuple)):
            inner = ','.join(_format_range_value(v) for v in item)
            formatted.append(f'[{inner}]')
        else:
            formatted.append(str(item))
    return f'[{",".join(formatted)}]'


def format_quoted_list(items):
    if not items:
        return "[]"
    formatted = []
    for item in items:
        if isinstance(item, list):
            formatted.append(format_quoted_list(item))
        else:
            formatted.append(f"'{item}'")
    return f"[{','.join(formatted)}]"


def convert_dtype_format(dtype_str):
    """将数据类型从 float32/float16/bfloat16/float64 转换为 fp32/fp16/bf16/fp64"""
    dtype_mapping = {
        'float32': 'fp32',
        'float16': 'fp16',
        'bfloat16': 'bf16',
        'float64': 'fp64'
    }
    return dtype_mapping.get(dtype_str, dtype_str)


def get_attr_type(param_type):
    """获取属性类型"""
    type_mapping = {
        'aclScalar': 'scalar',
        'aclScalarList': 'scalar_list',
        'aclIntArray': 'int_array',
        'aclFloatArray': 'float_array',
        'aclBoolArray': 'bool_array',
        'int4_t': 'buildins',
        'int8_t': 'buildins',
        'int16_t': 'buildins',
        'int32_t': 'buildins',
        'int64_t': 'buildins',
        'uint1_t': 'buildins',
        'uint8_t': 'buildins',
        'uint16_t': 'buildins',
        'uint32_t': 'buildins',
        'uint64_t': 'buildins',
        'float': 'buildins',
        'double': 'buildins',
        'float16': 'buildins',
        'bfloat16': 'buildins',
        'float32': 'buildins',
        'bool': 'buildins',
        'char': 'buildins',
        'string': 'buildins',
    }
    return type_mapping.get(param_type, 'buildins')


def get_attr_dtype(param_type, dtype_value):
    """获取属性dtype"""
    type_dtype_map = {
        'int4_t': 'int4', 'int8_t': 'int8', 'int16_t': 'int16',
        'int32_t': 'int32', 'int64_t': 'int64', 'uint1_t': 'uint1',
        'uint8_t': 'uint8', 'uint16_t': 'uint16', 'uint32_t': 'uint32',
        'uint64_t': 'uint64', 'bool': 'bool', 'float': 'fp32',
        'float16': 'fp16', 'bfloat16': 'bf16', 'float32': 'fp32',
        'double': 'double', 'char': 'int8', 'string': 'string',
    }
    if param_type in ('aclIntArray', 'aclFloatArray', 'aclBoolArray', 'aclScalarList'):
        return 'list'
    if param_type == 'aclScalar':
        dtype = dtype_value if dtype_value else 'float32'
    elif param_type in type_dtype_map:
        dtype = type_dtype_map[param_type]
    else:
        dtype = param_type
    return convert_dtype_format(dtype)


def format_attr_value(value):
    """格式化属性值"""
    import math
    if pd.isna(value):
        return ''
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str) and value in ('+0', '-0'):
        return value
    if isinstance(value, (list, tuple)):
        return str(list(value))
    if isinstance(value, float):
        if value == 0.0:
            if math.copysign(1, value) < 0:
                return '-0'
            elif math.copysign(1, value) > 0:
                return '+0'
        elif value != value:
            return "'nan'"
        elif value == float('inf'):
            return "'inf'"
        elif value == float('-inf'):
            return "'-inf'"
    return str(value)


def save_results(params: SaveResultsParams):
    """保存结果文件"""
    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存覆盖度报告
    report_path = output_dir / params.report_file
    with open(report_path, 'w', encoding='utf-8') as f:
        yaml.dump(params.report, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # 保存ttk用例csv
    ttk_csv = params.aclnn_name + "_" + str(params.case_file).split(".")[0] + ".csv"
    case_path_ = output_dir / ttk_csv
    params.case_df_ttk.to_csv(case_path_, index=False)
    
    if params.verbose:
        logging.info(f"生成覆盖度报告: {report_path}")
        logging.info(f"生成用例: {case_path_} ({len(params.case_df_ttk)}条)")


def convert_to_ttk_format(df, param_def, aclnn_name, case_level):
    """
    转换为TTK用例CSV格式
    """
    cases = []

    # 统一参数格式
    if isinstance(param_def.get('parameters'), list):
        params_dict = {p['name']: p for p in param_def['parameters'] if p.get('name')}
        param_def = params_dict

    param_names = list(param_def.keys())

    # 逐行生成用例
    for idx, row in df.iterrows():
        ctx = TtkProcessContext(
            row=row,
            idx=idx,
            param_def=param_def,
            param_names=param_names,
            aclnn_name=aclnn_name,
            case_level=case_level
        )
        case = _process_single_ttk_case(ctx)
        cases.append(case)

    # 列定义
    columns = [
        'testcase_name', 'api_name', 'tensor_view_shapes', 'tensor_dtypes',
        'scalar_dtypes', 'attributes', 'output_tensor_indexes',
        'precision_tolerances', 'absolute_precision', 'input_data_ranges',
        'scalar_data_ranges', 'tensor_list_distribution'
    ]

    return pd.DataFrame(cases)[columns]


def _process_single_ttk_case(ctx: TtkProcessContext):
    """处理单行数据，生成单个 TTK 用例"""
    testcase_name = f"{ctx.aclnn_name}_{ctx.case_level}_{ctx.idx+1:03d}"
    api_name = ctx.aclnn_name
    case = _init_ttk_case_base(testcase_name, api_name)

    # 初始化容器
    input_shapes, input_dtypes, input_ranges = [], [], []
    output_shapes, output_dtypes, output_indexes = [], [], []
    scalar_dtypes_list, scalar_ranges_list, attrs_dict = [], [], {}
    tensor_list_lengths = []

    for param_name, param_info in ctx.param_def.items():
        io_type = param_info.get('io_type', 'input')
        param_type = param_info.get('type', '')
        exist_col = f"{param_name}.exist"

        if exist_col in ctx.row and not ctx.row[exist_col]:
            continue
        
        # 封装成对象
        data = TensorProcessData(
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            input_ranges=input_ranges,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            output_indexes=output_indexes
        )

        if param_type == 'aclTensor':
            _process_acl_tensor(ctx.row, param_name, io_type, data, ctx.param_names)
        elif param_type == 'aclTensorList':
            data = TensorProcessData(
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                input_ranges=input_ranges,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
                output_indexes=output_indexes,
                tensor_list_lengths=tensor_list_lengths  # 加上这行
            )
            _process_acl_tensor_list(row, param_name, io_type, data, param_names)
        else:
            _process_scalar_and_attrs(ctx.row, param_name, param_type, scalar_dtypes_list, attrs_dict)

    # 封装并构建用例
    data = TtkCaseData(
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        input_ranges=input_ranges,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        output_indexes=output_indexes,
        scalar_dtypes_list=scalar_dtypes_list,
        attrs_dict=attrs_dict,
        tensor_list_lengths=tensor_list_lengths
    )
    _build_ttk_case(case, data)
    return case


def _process_acl_tensor(row, param_name, io_type, data: TensorProcessData, param_names):
    """处理单个 aclTensor 类型参数"""
    shape = parse_list_value(row.get(f"{param_name}.shape", '[]'))
    dtype = row.get(f"{param_name}.dtype", 'float32')

    if io_type == 'input':
        value_range = parse_list_value(row.get(f"{param_name}.value_range", '[]'))
        data.input_shapes.append(shape)
        data.input_dtypes.append(dtype)
        data.input_ranges.append(value_range if value_range else [-2, 2])
    else:
        data.output_shapes.append(shape)
        data.output_dtypes.append(dtype)
        data.output_indexes.append(param_names.index(param_name))


def _process_acl_tensor_list(row, param_name, io_type, data: TensorProcessData, param_names):
    """处理 aclTensorList 类型参数"""
    shape_list = parse_list_value(row.get(f"{param_name}.shape_list", '[]'))
    dtype = row.get(f"{param_name}.dtype", 'float32')
    length = _parse_tensor_length(row, param_name)

    if io_type == 'input':
        value_range = parse_list_value(row.get(f"{param_name}.value_range", '[]'))
        data.input_shapes.append(shape_list)
        data.input_dtypes.extend([dtype] * length)
        data.input_ranges.extend([value_range if value_range else [-2, 2]] * length)
    else:
        data.output_shapes.append(shape_list)
        data.output_dtypes.extend([dtype] * length)
        data.output_indexes.append(param_names.index(param_name))

    data.tensor_list_lengths.append(length)


def _process_scalar_and_attrs(row, param_name, param_type, scalar_dtypes_list, attrs_dict):
    """处理标量、属性类型参数"""
    if param_type == 'aclScalar':
        dtype = row.get(f"{param_name}.dtype", 'float')
        scalar_dtypes_list.append(dtype)
        value = row.get(f"{param_name}.value", '')
        if pd.notna(value) and value != '':
            attrs_dict[param_name] = format_ttk_attr_value(value, param_type)
        return

    # 数组 / 列表类型
    if param_type in ['aclIntArray', 'aclFloatArray', 'aclBoolArray', 'aclScalarList']:
        value = row.get(f"{param_name}.value", '')
        if pd.notna(value) and value != '':
            attrs_dict[param_name] = format_ttk_attr_value(value, param_type)
        return

    # 基础类型
    basic_types = [
        'int4_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t',
        'uint1_t', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
        'float', 'double', 'float16', 'bfloat16', 'float32', 'bool', 'char', 'string'
    ]
    if param_type in basic_types:
        value = row.get(f"{param_name}.value", '')
        if pd.notna(value) and value != '':
            attrs_dict[param_name] = format_ttk_attr_value(value, param_type)


def _build_ttk_case(case, data: TtkCaseData):
    """构建最终的 TTK 用例字段"""
    all_tensor_shapes = data.input_shapes + data.output_shapes
    all_tensor_dtypes = data.input_dtypes + data.output_dtypes

    case['tensor_view_shapes'] = format_ttk_tuple(all_tensor_shapes)
    case['tensor_dtypes'] = format_ttk_tuple_str(all_tensor_dtypes)

    if data.scalar_dtypes_list:
        case['scalar_dtypes'] = format_ttk_tuple_str(data.scalar_dtypes_list)
    if data.attrs_dict:
        case['attributes'] = format_ttk_dict(data.attrs_dict)
    if data.output_indexes:
        num_input = len(data.input_dtypes)
        output_pos = [num_input + i for i in range(len(data.output_indexes))]
        case['output_tensor_indexes'] = format_ttk_tuple(output_pos)
    if data.input_ranges:
        case['input_data_ranges'] = format_ttk_tuple(data.input_ranges)
    if data.tensor_list_lengths:
        case['tensor_list_distribution'] = format_ttk_tuple(data.tensor_list_lengths)


def format_ttk_tuple(items):
    """格式化TTK元组格式：每个子项转为元组字面量，外层包装为嵌套元组"""
    if not items:
        return "()"
    formatted = []
    for item in items:
        if isinstance(item, (list, tuple)):
            if len(item) == 0:
                formatted.append("((),)")
            else:
                inner = ",".join(str(x) for x in item)
                formatted.append(f"({inner},)")
        else:
            formatted.append(str(item))
    result = ",".join(formatted)
    return f"({result},)"


def format_ttk_tuple_str(items):
    """格式化字符串元组（添加引号），单元素带尾逗号"""
    if not items:
        return ""
    formatted = [f"'{item}'" for item in items]
    return f"({','.join(formatted)},)"


def format_ttk_dict(d):
    """格式化字典为字符串"""
    if not d:
        return ""

    def custom_serializer(obj):
        if isinstance(obj, complex):
            return f"({obj.real}+{obj.imag}j)"
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json.dumps(d, ensure_ascii=False, default=custom_serializer)


def format_ttk_attr_value(value, param_type):
    """格式化TTK属性值"""
    if pd.isna(value):
        return ''
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        parsed = literal_eval(str(value))
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return parsed
    except (ValueError, SyntaxError):
        return str(value)


# ============ L2 异常用例生成函数 ============

def generate_error_cases(param_def: Dict, factors: Dict, constraints_def: Dict,
                          verbose: bool = False) -> List[Dict]:
    """
    生成异常场景测试用例（L2级别）

    入口文件：04_测试因子.yaml 和 05_约束定义.yaml
    每个异常场景只生成一条用例

    Args:
        param_def: 参数定义（03_参数定义.yaml）- 用于参数结构信息
        factors: 测试因子（04_测试因子.yaml）- 用于提取支持的因子值
        constraints_def: 约束定义（05_约束定义.yaml）- 用于提取约束信息
        verbose: 详细输出

    Returns:
        List[Dict]: 异常用例列表
    """
    if verbose:
        logging.info("[INFO] 开始生成异常用例")
        logging.info("[INFO] 入口文件: 04_测试因子.yaml, 05_约束定义.yaml")

    error_cases = []
    aclnn_name = param_def.get('aclnn_name', 'UnknownOperator')

    # 从04_测试因子.yaml提取支持的因子值（主要入口）
    supported_info = extract_supported_info_from_factors(factors)
    # 从05_约束定义.yaml提取约束信息（辅助入口）
    constraint_info = extract_constraint_info(constraints_def)

    # 计算不支持的dtype和format
    unsupported_dtypes = [d for d in ALL_DTYPES if d not in supported_info['dtypes']]
    unsupported_formats = [f for f in ALL_FORMATS if f not in supported_info['formats']]
    max_dimensions = constraint_info.get('max_dimensions', 8)

    # 打印不支持信息
    _print_unsupported_info(verbose, supported_info, unsupported_dtypes, unsupported_formats, constraint_info)

    # 生成各类异常用例
    dtype_error_cases = generate_dtype_error_cases(param_def, unsupported_dtypes, aclnn_name, verbose)
    dtype_combo_cases = generate_dtype_combo_error_cases(
        param_def, supported_info['dtypes'], constraint_info, aclnn_name, verbose
    )
    dim_error_cases = generate_dimension_error_cases(param_def, max_dimensions, aclnn_name, verbose)
    format_error_cases, config = generate_format_error_cases(param_def, unsupported_formats, aclnn_name, verbose)
    shape_error_cases = generate_shape_mismatch_cases(param_def, constraint_info, aclnn_name, verbose)
    empty_tensor_cases = generate_empty_tensor_cases(param_def, aclnn_name, verbose, config)
    attr_error_cases = generate_attr_error_cases(param_def, supported_info['dtypes'], aclnn_name, verbose)

    error_cases.extend(dtype_error_cases)
    error_cases.extend(dtype_combo_cases)
    error_cases.extend(dim_error_cases)
    error_cases.extend(format_error_cases)
    error_cases.extend(shape_error_cases)
    error_cases.extend(empty_tensor_cases)
    error_cases.extend(attr_error_cases)

    # 打印统计信息
    summary = ErrorCaseSummary(
        dtype_error_cases=dtype_error_cases,
        dtype_combo_cases=dtype_combo_cases,
        dim_error_cases=dim_error_cases,
        format_error_cases=format_error_cases,
        shape_error_cases=shape_error_cases,
        empty_tensor_cases=empty_tensor_cases,
        attr_error_cases=attr_error_cases,
    )
    _print_final_error_summary(verbose, summary)

    return error_cases


def _print_unsupported_info(verbose, supported_info, unsupported_dtypes, unsupported_formats, constraint_info):
    """打印不支持的类型与约束信息"""
    if not verbose:
        return
    logging.info("[INFO] 从04_测试因子.yaml提取:")
    logging.info(f"[INFO]   支持的dtype: {supported_info['dtypes']}")
    logging.info(f"[INFO]   支持的format: {supported_info['formats']}")
    logging.info(f"[INFO]   支持的dimensions: {supported_info['dimensions']}")
    logging.info(f"[INFO] 不支持的dtype: {unsupported_dtypes}")
    logging.info(f"[INFO] 不支持的format: {unsupported_formats}")
    logging.info("[INFO] 从05_约束定义.yaml提取:")
    logging.info(f"[INFO]   最大维度限制: {constraint_info.get('max_dimensions', 8)}")
    logging.info(f"[INFO]   约束数量: {constraint_info.get('constraint_count', 0)}")


def _print_final_error_summary(verbose: bool, summary: ErrorCaseSummary):
    """打印最终异常用例统计"""
    if not verbose:
        return

    total = (
        len(summary.dtype_error_cases)
        + len(summary.dtype_combo_cases)
        + len(summary.dim_error_cases)
        + len(summary.format_error_cases)
        + len(summary.shape_error_cases)
        + len(summary.empty_tensor_cases)
        + len(summary.attr_error_cases)
    )

    logging.info(f"[INFO] 异常用例生成完成，总数: {total}")
    logging.info(f"[INFO]   dtype异常: {len(summary.dtype_error_cases)}条")
    logging.info(f"[INFO]   dtype组合异常: {len(summary.dtype_combo_cases)}条")
    logging.info(f"[INFO]   维度异常: {len(summary.dim_error_cases)}条")
    logging.info(f"[INFO]   格式异常: {len(summary.format_error_cases)}条")
    logging.info(f"[INFO]   形状异常: {len(summary.shape_error_cases)}条")
    logging.info(f"[INFO]   空张量异常: {len(summary.empty_tensor_cases)}条")
    logging.info(f"[INFO]   attr属性异常: {len(summary.attr_error_cases)}条")


def extract_supported_info_from_factors(factors: Dict) -> Dict:
    """
    从04_测试因子.yaml提取支持的因子值（主要入口）
    
    Args:
        factors: 测试因子（04_测试因子.yaml）
    
    Returns:
        Dict: {'dtypes': [...], 'formats': [...], 'dimensions': [...], 'tensor_params': [...], 'scalar_params': [...]}
    """
    dtypes = set()
    formats = set()
    dimensions = set()
    tensor_params = []
    scalar_params = []
    
    # 遍历每个参数的因子
    for param_name, param_factors in factors.items():
        if not isinstance(param_factors, dict):
            continue
        
        param_type = param_factors.get('type', '')
        factor_dict = param_factors.get('factors', {})
        
        # 提取dtype因子值
        dtype_key = f"{param_name}.dtype"
        if dtype_key in factor_dict:
            dtype_values = factor_dict[dtype_key]
            if isinstance(dtype_values, list):
                dtypes.update(dtype_values)
        
        # 提取format因子值
        format_key = f"{param_name}.format"
        if format_key in factor_dict:
            format_values = factor_dict[format_key]
            if isinstance(format_values, list):
                formats.update(format_values)
        
        # 提取dimensions因子值
        dim_key = f"{param_name}.dimensions"
        if dim_key in factor_dict:
            dim_values = factor_dict[dim_key]
            if isinstance(dim_values, list):
                dimensions.update(dim_values)
        
        # 记录参数类型
        if param_type == 'aclTensor':
            tensor_params.append(param_name)
        elif param_type == 'aclScalar':
            scalar_params.append(param_name)
    
    return {
        'dtypes': sorted(list(dtypes)),
        'formats': sorted(list(formats)),
        'dimensions': sorted(list(dimensions)),
        'tensor_params': tensor_params,
        'scalar_params': scalar_params
    }


def extract_constraint_info(constraints_def: Dict) -> Dict:
    """
    从05_约束定义.yaml提取约束信息（辅助入口）
    
    Args:
        constraints_def: 约束定义（05_约束定义.yaml）
    
    Returns:
        Dict: {'max_dimensions': int, 'constraint_count': int, 'dtype_constraints': [...], 'shape_constraints': [...]}
    """
    max_dimensions = 8  # 默认最大维度
    dtype_constraints = []
    shape_constraints = []
    
    # 提取约束列表
    constraints = constraints_def.get('constraints', [])
    constraint_count = len(constraints)
    
    for constraint in constraints:
        constraint_type = constraint.get('type', '')
        target = constraint.get('target', '')
        
        # 从约束中提取维度限制
        if 'dim' in target.lower() or 'dimension' in target.lower():
            # 查找维度约束
            if constraint_type == 'calculate':
                expression = constraint.get('expression', '')
                # 解析维度限制
                
        # 收集dtype约束
        if 'dtype' in target.lower():
            dtype_constraints.append({
                'id': constraint.get('id', ''),
                'type': constraint_type,
                'target': target,
                'sources': constraint.get('sources', []),
            })
        
        # 收集shape约束
        if 'shape' in target.lower():
            shape_constraints.append({
                'id': constraint.get('id', ''),
                'type': constraint_type,
                'target': target,
                'sources': constraint.get('sources', []),
            })
    
    # 从metadata提取信息
    metadata = constraints_def.get('metadata', {})
    
    return {
        'max_dimensions': max_dimensions,
        'constraint_count': constraint_count,
        'dtype_constraints': dtype_constraints,
        'shape_constraints': shape_constraints,
        'metadata': metadata
    }


def _process_acl_tensor_v(param, dtypes, formats):
    """处理 aclTensor 类型参数，提取 dtype、format、维度信息"""
    # 处理 dtype
    dtype_ranges = param.get('dtype_with_ranges', [])
    for dr in dtype_ranges:
        dtype = dr.get('dtype', '')
        if dtype:
            dtypes.add(dtype)

    # 处理 format
    param_formats = param.get('format', [])
    if isinstance(param_formats, list):
        formats.update(param_formats)

    # 处理维度
    dimensions = param.get('dimensions', [])
    current_max = 8
    if dimensions:
        current_max = max(dimensions) if isinstance(dimensions, list) else dimensions
    return current_max


def _process_acl_scalar(param, dtypes):
    """处理 aclScalar 类型参数，提取 dtype"""
    dtype_values = param.get('dtype_with_values', [])
    for dv in dtype_values:
        dtype = dv.get('dtype', '')
        if dtype:
            dtypes.add(dtype)


def extract_supported_info(param_def: Dict) -> Dict:
    """
    从参数定义提取支持的dtype和format

    Args:
        param_def: 参数定义

    Returns:
        Dict: {'dtypes': [...], 'formats': [...], 'max_dimensions': int}
    """
    dtypes = set()
    formats = set()
    max_dimensions = 8

    params = param_def.get('parameters', [])
    if not isinstance(params, list):
        return {
            'dtypes': sorted(list(dtypes)),
            'formats': sorted(list(formats)),
            'max_dimensions': max_dimensions
        }

    for param in params:
        param_type = param.get('type', '')
        io_type = param.get('io_type', 'input')

        # 只处理输入参数
        if io_type != 'input':
            continue

        if param_type == 'aclTensor':
            dim = _process_acl_tensor_v(param, dtypes, formats)
            max_dimensions = max(max_dimensions, dim)

        elif param_type == 'aclScalar':
            _process_acl_scalar(param, dtypes)

    return {
        'dtypes': sorted(list(dtypes)),
        'formats': sorted(list(formats)),
        'max_dimensions': max_dimensions
    }


def _get_input_params(param_def):
    """提取输入的 Tensor 和 Scalar 参数（拆分深度）"""
    tensor_params = []
    scalar_params = []
    params = param_def.get('parameters', [])
    
    if not isinstance(params, list):
        return tensor_params, scalar_params
    
    for param in params:
        if param.get('io_type') != 'input':
            continue
        
        param_type = param.get('type')
        param_name = param.get('name', '')
        
        if param_type == 'aclTensor':
            tensor_params.append(param_name)
        elif param_type == 'aclScalar':
            scalar_params.append(param_name)
    
    return tensor_params, scalar_params


def generate_dtype_error_cases(param_def: Dict, unsupported_dtypes: List[str], 
                                aclnn_name: str, verbose: bool) -> List[Dict]:
    """
    生成dtype不支持异常用例（每种场景只生成一条）
    
    Args:
        param_def: 参数定义
        unsupported_dtypes: 不支持的dtype列表
        aclnn_name: 算子名称
        verbose: 详细输出
    
    Returns:
        List[Dict]: dtype异常用例列表
    """
    cases = []
    
    # 提取输入参数（拆分成函数，深度大幅下降）
    tensor_params, scalar_params = _get_input_params(param_def)
    
    # 1. Tensor参数dtype不支持
    if unsupported_dtypes and tensor_params:
        dtype = unsupported_dtypes[0]
        param_name = tensor_params[0]
        case = create_error_case_template(param_def, aclnn_name)
        case[f"{param_name}.dtype"] = dtype
        case[f"{param_name}.shape"] = generate_random_shape(3)
        case['error_type'] = 'dtype_not_supported'
        case['error_param'] = param_name
        case['case_name'] = f"{aclnn_name}_{dtype}_random_{param_name}_dtype_exception"
        cases.append(case)
    
    # 2. Scalar参数dtype不支持
    if unsupported_dtypes and scalar_params:
        dtype = unsupported_dtypes[0]
        param_name = scalar_params[0]
        case = create_error_case_template(param_def, aclnn_name)
        case[f"{param_name}.dtype"] = dtype
        case[f"{param_name}.value"] = generate_random_value(dtype)
        case['error_type'] = 'dtype_not_supported'
        case['error_param'] = param_name
        case['case_name'] = f"{aclnn_name}_{dtype}_random_{param_name}_dtype_exception"
        cases.append(case)
    
    if verbose and cases:
        logging.info(f"[INFO] 生成dtype不支持异常用例: {len(cases)}条")
    
    return cases


def generate_dtype_combo_error_cases(param_def: Dict, supported_dtypes: List[str],
                                      constraint_info: Dict, aclnn_name: str, verbose: bool) -> List[Dict]:
    """
    生成dtype组合不匹配异常用例（只生成一条）
    
    Args:
        param_def: 参数定义
        supported_dtypes: 支持的dtype列表（从04_测试因子.yaml提取）
        constraint_info: 约束信息（从05_约束定义.yaml提取）
        aclnn_name: 算子名称
        verbose: 详细输出
    
    Returns:
        List[Dict]: dtype组合异常用例列表
    """
    cases = []
    
    # 【替换重复代码】调用通用函数
    tensor_params = _get_input_tensor_params(param_def)
    
    if len(tensor_params) < 2:
        return cases
    
    # 只生成一条不兼容组合用例（选第一个不兼容组合）
    for dtype1, dtype2 in INCOMPATIBLE_DTYPE_COMBOS:
        if dtype1 in supported_dtypes or dtype2 in supported_dtypes:
            case = create_error_case_template(param_def, aclnn_name)
            case[f"{tensor_params[0]}.dtype"] = dtype1
            case[f"{tensor_params[0]}.shape"] = generate_random_shape(3)
            case[f"{tensor_params[1]}.dtype"] = dtype2
            case[f"{tensor_params[1]}.shape"] = generate_random_shape(3)
            case['error_type'] = 'dtype_combo_mismatch'
            case['error_param'] = f"{tensor_params[0]},{tensor_params[1]}"
            case['case_name'] = f"{aclnn_name}_{dtype1}_{dtype2}_random_dtype_combo_exception"
            cases.append(case)
            break  # 只生成一条
    
    if verbose and cases:
        print(f"[INFO] 生成dtype组合异常用例: {len(cases)}条")
    
    return cases


def generate_dimension_error_cases(param_def: Dict, max_dimensions: int, 
                                    aclnn_name: str, verbose: bool) -> List[Dict]:
    """
    生成维度超限异常用例（只生成一条）
    """
    cases = []
    overflow_dims = [d for d in OVERFLOW_DIMENSIONS if d > max_dimensions]
    
    if not overflow_dims:
        return cases
    
    if verbose:
        print(f"[INFO] 维度限制: <= {max_dimensions}, 超限维度: {overflow_dims}")
    
    # 公共方法
    tensor_params = _get_tensor_input_params(param_def)
    if not tensor_params:
        return cases
    
    supported_dtype = _get_supported_tensor_dtype(param_def)
    
    # 生成用例
    dim = overflow_dims[0]
    param_name = tensor_params[0]
    case = create_error_case_template(param_def, aclnn_name)
    case[f"{param_name}.dtype"] = supported_dtype
    case[f"{param_name}.shape"] = generate_overflow_shape(dim)
    case[f"{param_name}.dimensions"] = dim
    case['error_type'] = 'dimension_overflow'
    case['error_param'] = param_name
    case['case_name'] = f"{aclnn_name}_{supported_dtype}_random_{param_name}_dim_exception"
    cases.append(case)
    
    if verbose and cases:
        print(f"[INFO] 生成维度超限异常用例: {len(cases)}条")
    
    return cases


def generate_format_error_cases(param_def: Dict, unsupported_formats: List[str],
                                 aclnn_name: str, verbose: bool) -> List[Dict]:
    """
    生成格式不支持异常用例（只生成一条）
    """
    cases = []
    if not unsupported_formats:
        return cases

    tensor_params = _get_tensor_input_params(param_def)
    if not tensor_params:
        return cases

    supported_dtype = _get_supported_tensor_dtype(param_def)
    format_name = unsupported_formats[0]
    param_name = tensor_params[0]

    # 通用函数创建用例
    config = ErrorCaseConfig(
        param_def=param_def,
        aclnn_name=aclnn_name,
        param_name=param_name,
        supported_dtype=supported_dtype,
        error_type="format_not_supported",
        error_param=param_name,
        case_name=case_name_str,
        shape_values={param_name: generate_random_shape(3)}
    )
    case = _create_and_fill_error_case(config)

    # 单独设置 format
    case[f"{param_name}.format"] = format_name
    cases.append(case)

    if verbose and cases:
        print(f"[INFO] 生成格式不支持异常用例: {len(cases)}条")

    return cases, config


def _get_tensor_input_params(param_def: Dict) -> list:
    """获取所有输入 Tensor 参数名（抽取重复代码）"""
    tensor_params = []
    params = param_def.get('parameters', [])
    if isinstance(params, list):
        for param in params:
            if param.get('io_type') == 'input' and param.get('type') == 'aclTensor':
                tensor_params.append(param.get('name', ''))
    return tensor_params


def _get_supported_tensor_dtype(param_def: Dict) -> str:
    """获取 Tensor 支持的第一个 dtype（抽取重复代码）"""
    params = param_def.get('parameters', [])
    for param in params:
        if param.get('type') == 'aclTensor':
            dtype_ranges = param.get('dtype_with_ranges', [])
            if dtype_ranges:
                return dtype_ranges[0].get('dtype', 'float32')
    return 'float32'


def generate_shape_mismatch_cases(param_def: Dict, constraint_info: Dict,
                                   aclnn_name: str, verbose: bool) -> List[Dict]:
    """
    生成形状不匹配异常用例（只生成一条）
    """
    cases = []

    # 公共方法获取参数
    tensor_params = _get_tensor_input_params(param_def)

    if len(tensor_params) < 2:
        if verbose:
            print(f"[INFO] 单Tensor算子，不生成形状不匹配异常用例")
        return cases

    # 公共方法获取 dtype
    supported_dtype = _get_supported_tensor_dtype(param_def)

    # 生成用例
    case = create_error_case_template(param_def, aclnn_name)
    case[f"{tensor_params[0]}.dtype"] = supported_dtype
    case[f"{tensor_params[0]}.shape"] = [2, 3]
    case[f"{tensor_params[1]}.dtype"] = supported_dtype
    case[f"{tensor_params[1]}.shape"] = [2, 3, 4]
    case['error_type'] = 'shape_dimension_mismatch'
    case['error_param'] = f"{tensor_params[0]},{tensor_params[1]}"
    case['case_name'] = f"{aclnn_name}_{supported_dtype}_ND_ND_ND_exception_shape_dim_error"
    cases.append(case)

    if verbose and cases:
        print(f"[INFO] 生成形状不匹配异常用例: {len(cases)}条")

    return cases


def generate_empty_tensor_cases(param_def: Dict, aclnn_name: str, 
                                 verbose: bool, config: Dict) -> List[Dict]:
    """
    生成空张量异常用例（只生成一条）
    """
    cases = []
    # 【替换重复代码】复用通用函数，重复代码彻底消除
    tensor_params = _get_input_tensor_params(param_def)

    if len(tensor_params) < 2:
        return cases

    supported_dtype = _get_supported_tensor_dtype(param_def)
    p0, p1 = tensor_params[0], tensor_params[1]

    # 通用函数创建用例
    case = _create_and_fill_error_case(config)

    cases.append(case)

    if verbose and cases:
        print(f"[INFO] 生成空张量异常用例: {len(cases)}条")
    
    return cases


def _get_input_tensor_params(param_def: Dict) -> List[str]:
    """抽取重复代码：获取所有输入aclTensor参数名"""
    tensor_params = []
    params = param_def.get('parameters', [])
    if isinstance(params, list):
        for param in params:
            if param.get('io_type') == 'input' and param.get('type') == 'aclTensor':
                tensor_params.append(param.get('name', ''))
    return tensor_params


def _create_and_fill_error_case(config: ErrorCaseConfig):
    """通用：创建并填充异常用例（抽取所有重复代码）"""
    case = create_error_case_template(config.param_def, config.aclnn_name)
    case[f"{config.param_name}.dtype"] = config.supported_dtype

    # 如果传了 shape 字典，批量设置 shape
    if config.shape_values:
        for pname, shape_val in config.shape_values.items():
            case[f"{pname}.shape"] = shape_val

    case["error_type"] = config.error_type
    case["error_param"] = config.error_param
    case["case_name"] = config.case_name
    return case


def _get_supported_tensor_dtype(param_def: Dict) -> str:
    """获取第一个支持的 Tensor dtype（公共方法）"""
    params = param_def.get('parameters', [])
    for param in params:
        if param.get('type') == 'aclTensor':
            dtype_ranges = param.get('dtype_with_ranges', [])
            if dtype_ranges:
                return dtype_ranges[0].get('dtype', 'float32')
    return 'float32'


def _set_tensor_default(case, param_name, param):
    """设置 aclTensor 默认值"""
    dtype_ranges = param.get('dtype_with_ranges', [])
    if dtype_ranges:
        case[f"{param_name}.dtype"] = dtype_ranges[0].get('dtype', 'float32')
    case[f"{param_name}.format"] = param.get('format', ['ND'])[0]
    case[f"{param_name}.dimensions"] = 2
    case[f"{param_name}.shape"] = [2, 3]
    case[f"{param_name}.value_range"] = [-1, 1]


def _set_scalar_default(case, param_name, param):
    """设置 aclScalar 默认值"""
    dtype_values = param.get('dtype_with_values', [])
    if dtype_values:
        case[f"{param_name}.dtype"] = dtype_values[0].get('dtype', 'float32')
    case[f"{param_name}.value"] = 1.0


def _set_tensor_list_default(case, param_name, param):
    """设置 aclTensorList 默认值"""
    dtype_ranges = param.get('dtype_with_ranges', [])
    if dtype_ranges:
        case[f"{param_name}.dtype"] = dtype_ranges[0].get('dtype', 'float32')
    case[f"{param_name}.length"] = 2
    case[f"{param_name}.shape_list"] = [[2, 3], [2, 3]]


def _set_array_default(case, param_name, param_type):
    """设置数组/基础类型默认值"""
    if param_type in ['aclIntArray', 'aclFloatArray', 'aclBoolArray']:
        case[f"{param_name}.value"] = [1, 2]
    else:
        case[f"{param_name}.value"] = 1


def _set_output_tensor_default(case, param_name, param):
    """设置输出 Tensor 默认值"""
    dtype_ranges = param.get('dtype_with_ranges', [])
    if dtype_ranges:
        case[f"{param_name}.dtype"] = dtype_ranges[0].get('dtype', 'float32')
    case[f"{param_name}.format"] = param.get('format', ['ND'])[0]
    case[f"{param_name}.dimensions"] = 2
    case[f"{param_name}.shape"] = [2, 3]


def create_error_case_template(param_def: Dict, aclnn_name: str) -> Dict:
    """
    创建异常用例模板

    Args:
        param_def: 参数定义
        aclnn_name: 算子名称

    Returns:
        Dict: 用例模板
    """
    case = {
        'aclnn_name': aclnn_name,
        'case_name': '',
        'error_type': '',
        'error_param': '',
    }

    params = param_def.get('parameters', [])
    if not isinstance(params, list):
        return case

    for param in params:
        param_name = param.get('name', '')
        param_type = param.get('type', '')
        io_type = param.get('io_type', 'input')
        case[f"{param_name}.exist"] = True

        # 输入参数
        if io_type == 'input':
            if param_type == 'aclTensor':
                _set_tensor_default(case, param_name, param)
            elif param_type == 'aclScalar':
                _set_scalar_default(case, param_name, param)
            elif param_type == 'aclTensorList':
                _set_tensor_list_default(case, param_name, param)
            else:
                _set_array_default(case, param_name, param_type)

        # 输出参数
        elif io_type == 'output' and param_type == 'aclTensor':
            _set_output_tensor_default(case, param_name, param)

    return case


def generate_random_shape(dimensions: int) -> List[int]:
    """生成随机shape"""
    return [random.randint(1, 100) for _ in range(dimensions)]


def generate_overflow_shape(dimensions: int) -> List[int]:
    """生成超限shape（维度>8）"""
    return [random.randint(1, 10) for _ in range(dimensions)]


def generate_random_value(dtype: str) -> Any:
    """根据dtype生成随机值"""
    if dtype in ['float32', 'float', 'float16', 'bfloat16', 'float64', 'double']:
        return random.uniform(-10, 10)
    elif dtype in ['int32', 'int64', 'int16', 'int8']:
        return random.randint(-100, 100)
    elif dtype in ['uint8', 'uint16', 'uint32', 'uint64']:
        return random.randint(0, 100)
    elif dtype == 'bool':
        return random.choice([True, False])
    elif dtype in ['complex64', 'complex128']:
        return complex(random.uniform(-10, 10), random.uniform(-10, 10))
    else:
        return 1


def generate_attr_error_cases(param_def: Dict, supported_dtypes: List[str],
                               aclnn_name: str, verbose: bool) -> List[Dict]:
    """
    生成attr属性异常用例（Scalar参数value异常）
    
    Args:
        param_def: 参数定义
        supported_dtypes: 支持的dtype列表
        aclnn_name: 算子名称
        verbose: 详细输出
    
    Returns:
        List[Dict]: attr属性异常用例列表
    
    异常类型：
        1. 特殊值异常：inf, -inf, nan（浮点类型）
        2. 边界值异常：超出dtype范围的值
        3. 非法值异常：如非因子数的groups等
    """
    cases = []

    # 获取所有Scalar输入参数
    scalar_params = []
    params = param_def.get('parameters', [])
    if isinstance(params, list):
        for param in params:
            if param.get('io_type') == 'input' and param.get('type') == 'aclScalar':
                scalar_params.append(param.get('name', ''))
    
    if not scalar_params:
        return cases
    
    # 对所有Scalar参数生成attr异常（每个参数每个场景一条）
    for param_name in scalar_params:
        # 获取一个支持的浮点dtype（用于生成特殊值异常）
        float_dtype = next((d for d in supported_dtypes if d in ['float32', 'float16', 'float64', 'bfloat16']), None)
        
        # 1. 特殊值异常（inf/nan等）- 只对浮点类型生成一条
        if float_dtype:
            case = create_error_case_template(param_def, aclnn_name)
            case[f"{param_name}.dtype"] = float_dtype
            case[f"{param_name}.value"] = 'nan'
            case['error_type'] = 'attr_special_value'
            case['error_param'] = param_name
            case['case_name'] = f"{aclnn_name}_{float_dtype}_random_{param_name}_attr_nan_exception"
            cases.append(case)
        
        # 2. 边界值异常（超出范围）
        int_dtype = next((d for d in supported_dtypes if d in ['int32', 'int64', 'int16', 'int8']), None)
        if int_dtype and int_dtype in DTYPE_BOUNDARIES:
            boundaries = DTYPE_BOUNDARIES[int_dtype]
            case = create_error_case_template(param_def, aclnn_name)
            case[f"{param_name}.dtype"] = int_dtype
            overflow_value = boundaries['max'] + 1 if boundaries['max'] else 999999999
            case[f"{param_name}.value"] = overflow_value
            case['error_type'] = 'attr_boundary_overflow'
            case['error_param'] = param_name
            case['case_name'] = f"{aclnn_name}_{int_dtype}_random_{param_name}_attr_boundary_exception"
            cases.append(case)
    
    if verbose and cases:
        logging.info(f"[INFO] 生成attr属性异常用例: {len(cases)}条")
    
    return cases


def create_error_ttk_dataframe(error_cases: List[Dict], param_def: Dict) -> pd.DataFrame:
    """
    创建TTK格式的异常用例DataFrame

    Args:
        error_cases: 异常用例列表
        param_def: 参数定义

    Returns:
        pd.DataFrame: TTK格式用例DataFrame
    """
    cases = []

    # 构建参数字典
    params = param_def.get('parameters', [])
    if isinstance(params, list):
        param_def_dict = {p.get('name'): p for p in params if p.get('name')}
    else:
        param_def_dict = param_def

    for idx, error_case in enumerate(error_cases):
        testcase_name = error_case.get('case_name', f"error_case_{idx:03d}")
        api_name = error_case.get('aclnn_name', '')
        case = _init_ttk_case_base(testcase_name, api_name)

        # 初始化容器
        input_shapes, input_dtypes, input_ranges = [], [], []
        output_shapes, output_dtypes = [], []
        scalar_dtypes_list, attrs_dict = [], {}

        # 遍历参数
        for param_name, param_info in param_def_dict.items():
            io_type = param_info.get('io_type', 'input')
            param_type = param_info.get('type', '')

            if param_type == 'aclTensor' and io_type == 'input':
                data = InputTensorData(
                    input_shapes=input_shapes,
                    input_dtypes=input_dtypes,
                    input_ranges=input_ranges
                )
                _process_input_tensor(param_name, param_info, error_case, data)

            elif param_type == 'aclTensor' and io_type == 'output':
                _process_output_tensor(param_name, param_info, error_case,
                                      output_shapes, output_dtypes)

            else:
                _process_scalar_types(param_name, param_info, error_case,
                                      scalar_dtypes_list, attrs_dict)

        # 填充 TTK 格式
        data = TtkCaseData(
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            input_ranges=input_ranges,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            scalar_dtypes_list=scalar_dtypes_list,
            attrs_dict=attrs_dict
        )
        _fill_ttk_case_fields(case, data)
        
        cases.append(case)

    columns = [
        'testcase_name', 'api_name', 'tensor_view_shapes', 'tensor_dtypes',
        'scalar_dtypes', 'attributes', 'output_tensor_indexes',
        'precision_tolerances', 'absolute_precision', 'input_data_ranges',
        'scalar_data_ranges', 'tensor_list_distribution'
    ]

    return pd.DataFrame(cases)[columns] if cases else pd.DataFrame(columns=columns)


def _init_ttk_case_base(testcase_name, api_name):
    """初始化 TTK 用例通用空模板（消除重复代码）"""
    return {
        'testcase_name': testcase_name,
        'api_name': api_name,
        'tensor_view_shapes': '',
        'tensor_dtypes': '',
        'scalar_dtypes': '',
        'attributes': '',
        'output_tensor_indexes': '',
        'precision_tolerances': '',
        'absolute_precision': '',
        'input_data_ranges': '',
        'scalar_data_ranges': '',
        'tensor_list_distribution': ''
    }


def _process_input_tensor(param_name, param_info, error_case, data: InputTensorData):
    """处理输入 Tensor"""
    shape = error_case.get(f"{param_name}.shape", [])
    dtype = error_case.get(f"{param_name}.dtype", 'float32')
    value_range = error_case.get(f"{param_name}.value_range", [])
    
    if shape:
        data.input_shapes.append(list(shape) if isinstance(shape, (list, tuple)) else shape)
    
    data.input_dtypes.append(dtype)
    data.input_ranges.append(list(value_range) if value_range else [-2, 2])


def _process_output_tensor(param_name, param_info, error_case, output_shapes, output_dtypes):
    """处理输出 Tensor"""
    shape = error_case.get(f"{param_name}.shape", [])
    dtype = error_case.get(f"{param_name}.dtype", 'float32')
    if shape:
        output_shapes.append(list(shape) if isinstance(shape, (list, tuple)) else shape)
    output_dtypes.append(dtype)


def _process_scalar_types(param_name, param_info, error_case, scalar_dtypes_list, attrs_dict):
    """处理 Scalar / Array 类型参数"""
    param_type = param_info.get('type', '')

    if param_type == 'aclScalar':
        dtype = error_case.get(f"{param_name}.dtype", 'float32')
        scalar_dtypes_list.append(dtype)
        value = error_case.get(f"{param_name}.value", '')
        if pd.notna(value) and value != '':
            attrs_dict[param_name] = format_ttk_attr_value(value, param_type)

    elif param_type in ['aclIntArray', 'aclFloatArray', 'aclBoolArray']:
        value = error_case.get(f"{param_name}.value", '')
        if pd.notna(value) and value != '':
            attrs_dict[param_name] = format_ttk_attr_value(value, param_type)


def _fill_ttk_case_fields(case, data: TtkCaseData):
    """填充 TTK 格式最终字段"""
    all_tensor_shapes = data.input_shapes + data.output_shapes
    all_tensor_dtypes = data.input_dtypes + data.output_dtypes

    case['tensor_view_shapes'] = format_ttk_tuple(all_tensor_shapes)
    case['tensor_dtypes'] = format_ttk_tuple_str(all_tensor_dtypes)

    if data.scalar_dtypes_list:
        case['scalar_dtypes'] = format_ttk_tuple_str(data.scalar_dtypes_list)
    if data.attrs_dict:
        case['attributes'] = format_ttk_dict(data.attrs_dict)
    if data.input_ranges:
        case['input_data_ranges'] = format_ttk_tuple(data.input_ranges)


if __name__ == '__main__':
    main()
