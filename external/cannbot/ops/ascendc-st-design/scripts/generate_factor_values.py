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
因子取值生成器

功能：
1. 解析求解配置、约束定义、测试因子
2. 对锚点因子进行BC组合
3. 逐层推导其他因子（优先非shape类因子）
4. 返回取值范围而非随机值
5. 支持value_range_dtype扩展
6. 生成完整的因子取值表

使用方法：
    python generate_factor_values.py <求解配置.yaml> <约束定义.yaml> <测试因子.yaml> [输出.csv]
    
示例：
    python generate_factor_values.py 07_求解配置.yaml 05_约束定义.yaml 04_测试因子.yaml factor_values.csv
"""

import sys
import yaml
import csv
import random
import copy
from typing import Dict, List, Set, Any, Optional, Tuple, Union
from pathlib import Path
from itertools import product
from collections import defaultdict
import math

try:
    from utils import (
        infer_dtypes, 
        infer_two_dtypes, 
        normalize_dtype, 
        normalize_dtype_list,
        get_convertible_source_dtypes,
        generate_random_shape,
        generate_random_value_by_dtype,
        FLOAT_DTYPES,
        INTEGER_DTYPES,
        generate_broadcast_shapes,
        generate_unidirectional_broadcast_shapes,
        get_broadcast_result,
        MAX_SHAPE_PRODUCT
    )
except ImportError:
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import (
        infer_dtypes, 
        infer_two_dtypes, 
        normalize_dtype, 
        normalize_dtype_list,
        get_convertible_source_dtypes,
        generate_random_shape,
        generate_random_value_by_dtype,
        FLOAT_DTYPES,
        INTEGER_DTYPES,
        generate_broadcast_shapes,
        generate_unidirectional_broadcast_shapes,
        get_broadcast_result,
        MAX_SHAPE_PRODUCT
    )


class FactorValueGenerator:
    """因子取值生成器"""
    
    TENSOR_TYPES = {'aclTensor', 'aclTensorList'}
    ARRAY_TYPES = {'aclIntArray', 'aclFloatArray', 'aclBoolArray', 'aclScalarList'}
    SCALAR_TYPES = {
        'int4_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t',
        'uint1_t', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
        'bool', 'float', 'float16', 'bfloat16', 'float32', 'double',
        'char', 'string',
    }
    
    SHAPE_RELATED_ATTRS = {'shape'}
    
    def __init__(self):
        self.solver_config = {}
        self.constraints_data = {}
        self.test_factors = {}
        self.factors_def = {}
        self.constraints = []
        self.anchors = []
        self.derivation_order = {}
    
    @staticmethod
    def _make_hashable(value):
        """
        将值转换为可哈希类型
        
        Args:
            value: 任意值
            
        Returns:
            可哈希的值
        """
        if isinstance(value, list):
            return tuple(FactorValueGenerator._make_hashable(v) for v in value)
        elif isinstance(value, dict):
            return frozenset((k, FactorValueGenerator._make_hashable(v)) for k, v in value.items())
        else:
            return value
    
    def load_configs(self, solver_config_path: str, constraints_path: str, factors_path: str):
        """加载配置文件"""
        with open(solver_config_path, 'r', encoding='utf-8') as f:
            self.solver_config = yaml.safe_load(f)
        
        with open(constraints_path, 'r', encoding='utf-8') as f:
            self.constraints_data = yaml.safe_load(f)
            
        with open(factors_path, 'r', encoding='utf-8') as f:
            self.test_factors = yaml.safe_load(f)
        
        self.factors_def = self.constraints_data.get('factors', {})
        self.constraints = self.constraints_data.get('constraints', [])
        
        solver = self.solver_config.get('solver', {})
        self.anchors = solver.get('anchors', [])
        self.derivation_order = solver.get('derivation_order', {})
        
    def get_factor_domain(self, factor_name: str) -> List[Any]:
        """获取因子的取值范围"""
        parts = factor_name.split('.')
        if len(parts) != 2:
            return []
        
        param_name, attr = parts
        param_data = self.test_factors.get(param_name, {})
        factors = param_data.get('factors', {})
        
        if attr == 'exist':
            return factors.get(f'{param_name}.exist', [True])
        elif attr == 'format':
            format_list = factors.get(f'{param_name}.format', [])
            return format_list
        elif attr == 'dimensions':
            return factors.get(f'{param_name}.dimensions', [])
        elif attr == 'dtype':
            return factors.get(f'{param_name}.dtype', [])
        elif attr == 'enum_values':
            return factors.get(f'{param_name}.enum_values', [])
        # elif attr == 'value':
        #     return factors.get(f'{param_name}.enum_values', [])
        
        exact_key = f'{param_name}.{attr}'
        if exact_key in factors:
            return factors[exact_key]
        
        if attr == 'value':
            return []
        
        for key in factors:
            if key.startswith(f'{param_name}.{attr}_'):
                return factors[key]
        
        return []
    
    def get_value_range_for_dtype(self, param_name: str, dtype: str) -> List[Any]:
        """根据dtype获取value_range取值范围"""
        param_data = self.test_factors.get(param_name, {})
        factors = param_data.get('factors', {})
        
        key = f'{param_name}.value_range_{dtype}'
        if key in factors:
            return factors[key]
        
        key = f'{param_name}.value_range'
        if key in factors:
            return factors[key]
        
        return []
    
    def is_shape_factor(self, factor_name: str) -> bool:
        """判断是否为shape类因子"""
        parts = factor_name.split('.')
        if len(parts) != 2:
            return False
        return parts[1] in self.SHAPE_RELATED_ATTRS
    
    def factor_depends_on_shape(self, factor_name: str) -> bool:
        """
        检查因子的约束是否依赖shape类因子
        
        Args:
            factor_name: 因子名称
            
        Returns:
            True如果该因子依赖shape类因子，否则False
            
        注意：
            检查所有约束，只要任一约束依赖shape类因子就返回True
        """
        constraints = self.find_all_constraints_for_factor(factor_name)
        if not constraints:
            return False
        
        for constraint in constraints:
            sources = constraint.get('sources', [])
            for source in sources:
                if self.is_shape_factor(source):
                    return True
        
        return False
    
    def generate_bc_combinations(self, max_combinations: int = 100000000000) -> List[Dict[str, Any]]:
        """生成锚点因子的BC组合"""
        anchor_domains = {}
        
        for anchor in self.anchors:
            domain = self.get_factor_domain(anchor)
            if domain:
                anchor_domains[anchor] = domain
        
        if not anchor_domains:
            return []
        
        all_combinations = list(product(*[anchor_domains[a] for a in anchor_domains.keys()]))
        
        if len(all_combinations) > max_combinations:
            random.seed(42)
            all_combinations = random.sample(all_combinations, max_combinations)
        
        combinations = []
        anchor_names = list(anchor_domains.keys())
        
        for combo in all_combinations:
            case = {}
            for i, anchor in enumerate(anchor_names):
                case[anchor] = combo[i]
            combinations.append(case)
        
        return combinations
    
    def find_all_constraints_for_factor(self, factor_name: str) -> List[Dict]:
        """
        查找所有匹配的约束定义
        
        Args:
            factor_name: 因子名称
            
        Returns:
            所有target为该因子的约束列表
        """
        matched_constraints = []
        for constraint in self.constraints:
            target = constraint.get('target', '')
            if target == factor_name:
                matched_constraints.append(constraint)
        
        return matched_constraints
    
    def solve_constraint_range(self, constraint: Dict, context: Dict[str, Any]) -> Union[List[Any], Any]:
        """
        求解约束，返回取值范围而非单个值
        
        根据参数类型分发到对应的推导方法：
        - aclTensorList: _solve_tensor_list_derivation
        - aclIntArray/aclFloatArray/aclBoolArray/aclScalarList: _solve_array_derivation
        - aclScalar/18种标量类型: _solve_scalar_derivation
        - aclTensor: 原有逻辑
        """
        constraint_type = constraint.get('type', '')
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        
        source_values = [context.get(s) for s in sources]
        
        if any(v is None for v in source_values):
            return None
        
        if constraint_type == 'calculate':
            expression = constraint.get('expression', 'sources[0]')
            param_name = target.split('.')[0]
            param_type = self._get_param_type(param_name)
            target_attr = target.split('.')[-1] if '.' in target else ''
            
            try:
                result = self._dispatch_derivation(
                    expression, param_type, target_attr,
                    source_values, context, target
                )
                if result is not None:
                    return result
                return eval(expression, {
                    'sources': source_values,
                    'get_broadcast_result': lambda *shapes: get_broadcast_result(list(shapes)),
                })
            except:
                return None
        
        elif constraint_type == 'inferable_filter':
            return random.choice(self._solve_inferable_filter_range(constraint, context))
        
        elif constraint_type == 'convertible':
            return random.choice(self._solve_convertible_range(constraint, context))
        
        elif constraint_type == 'match':
            return random.choice(self._solve_match_range(constraint, context))
        
        elif constraint_type == 'broadcast_dim':
            return random.choice(self._solve_broadcast_dim_range(constraint, context))
        
        elif constraint_type == 'broadcast_shape':
            return self._solve_broadcast_shape_range(constraint, context)
        
        return None
    
    def _get_param_type(self, param_name: str) -> str:
        param_data = self.test_factors.get(param_name, {})
        return param_data.get('type', '')
    
    def _dispatch_derivation(self, expression, param_type, target_attr,
                             source_values, context, target):
        """
        按参数类型分发推导逻辑
        
        (1) aclTensorList:
            a. derive_length_from_length_ranges: 根据 length_ranges 随机生成 length
            b. derive_shape_list_from_length_and_dimensions: length 决定 shape_list 长度，
               根据 dimensions 生成每个 shape
            c. derive_value_range_from_dtype / derive_shape_from_dimensions: 同 aclTensor
        
        (2) aclIntArray/aclFloatArray/aclBoolArray/aclScalarList:
            a. derive_length_from_length_ranges: 根据 length_ranges 随机生成 length
            b. derive_array_value_from_length_and_value_range: 根据 value_range 随机生成
               length 个值，保存在 list 中
            c. derive_value_range_from_dtype: 同 aclTensor
        
        (3) 18种标量类型 + aclScalar:
            a. derive_value_from_range: 根据 value_range 随机生成 value
            b. derive_value_range_from_dtype: 同 aclTensor
        """
        hashable_context = frozenset((k, self._make_hashable(v)) for k, v in context.items())
        seed = hash(hashable_context) % (2**32)
        
        # === 通用推导：shape from dimensions ===
        if expression.startswith('derive_shape_from_dimensions'):
            dimensions = source_values[0]
            if isinstance(dimensions, int):
                return generate_random_shape(dimensions, seed=seed)
            return None
        
        # === 通用推导：value_range from dtype ===
        if expression.startswith('derive_value_range_from_dtype'):
            dtype = source_values[0]
            param_name = target.split('.')[0]
            value_ranges = self.get_value_range_for_dtype(param_name, dtype)
            if value_ranges:
                random.seed(seed)
                return random.choice(value_ranges)
            return None
        
        # === 通用推导：value from value_range ===
        if expression.startswith('derive_value_from_range'):
            value_range = source_values[0]
            param_name = target.split('.')[0]
            dtype_value = context.get(f'{param_name}.dtype', 'float32')
            return self._derive_scalar_value(value_range, dtype_value, seed)
        
        # === (1) aclTensorList 专属推导 ===
        if param_type == 'aclTensorList':
            if expression.startswith('derive_length_from_length_ranges'):
                return self._derive_length(source_values[0], seed)
            
            if expression.startswith('derive_shape_list_from_length_and_dimensions'):
                return self._derive_shape_list(source_values[0], source_values[1], seed)
        
        # === 通用推导：多shape广播结果计算 ===
        if expression.startswith('get_broadcast_result'):
            shapes = [v for v in source_values if isinstance(v, list)]
            if len(shapes) >= 2:
                result = get_broadcast_result(shapes[:2])
                if result is not None:
                    return result
            elif len(shapes) == 1:
                return shapes[0]
            return None

        # === (2) Array 类型专属推导 ===
        if param_type in self.ARRAY_TYPES:
            if expression.startswith('derive_length_from_length_ranges'):
                return self._derive_length(source_values[0], seed)
            
            if expression.startswith('derive_array_value_from_length_and_value_range'):
                return self._derive_array_value(
                    source_values[0], source_values[1],
                    context, target, seed
                )
        
        return None
    
    def _derive_scalar_value(self, value_range, dtype_value, seed):
        """根据 value_range 和 dtype 随机生成标量 value"""
        if not isinstance(value_range, list) or len(value_range) != 2:
            return None
        signed_zero = self._check_signed_zero_range(value_range)
        if signed_zero is not None:
            return signed_zero
        result = generate_random_value_by_dtype(dtype_value, value_range, seed=seed)
        if isinstance(result, list):
            if len(result) == 2 and all(isinstance(v, (int, float)) for v in result):
                mid = (result[0] + result[1]) / 2
                if dtype_value in INTEGER_DTYPES or dtype_value in ('bool',):
                    mid = int(mid)
                return mid
            return None
        return result
    
    def _derive_length(self, length_ranges, seed):
        """根据 length_ranges 随机生成 length"""
        if not isinstance(length_ranges, list) or not length_ranges:
            return None
        random.seed(seed)
        if isinstance(length_ranges[0], list):
            selected_range = random.choice(length_ranges)
            return random.randint(int(selected_range[0]), int(selected_range[1]))
        else:
            return random.randint(int(length_ranges[0]), int(length_ranges[1]))
    
    def _derive_shape_list(self, length_val, dimensions_val, seed):
        """
        根据 length 和 dimensions 生成 shape_list
        length 决定 shape_list 中包含几个 shape
        根据 dimensions 生成每个 shape（参考 aclTensor 的 shape 生成规则）
        """
        if not isinstance(length_val, int) or not isinstance(dimensions_val, int):
            return None
        return [generate_random_shape(dimensions_val, seed=seed + i) for i in range(length_val)]
    
    @staticmethod
    def _check_signed_zero_range(value_range):
        if isinstance(value_range, list) and len(value_range) == 2:
            if value_range[0] == '+0' and value_range[1] == '+0':
                return '+0'
            if value_range[0] == '-0' and value_range[1] == '-0':
                return '-0'
        return None

    def _derive_array_value(self, length_val, value_range_val, context, target, seed):
        """
        根据 value_range 随机生成 length 个数，保存在 list 中
        适用于 aclIntArray/aclFloatArray/aclBoolArray/aclScalarList
        """
        if not isinstance(length_val, int):
            return None
        if not isinstance(value_range_val, list) or len(value_range_val) != 2:
            return None
        signed_zero = self._check_signed_zero_range(value_range_val)
        if signed_zero is not None:
            return [signed_zero] * length_val
        param_name = target.split('.')[0]
        dtype_value = context.get(f'{param_name}.dtype', 'float32')
        random.seed(seed)
        return [generate_random_value_by_dtype(dtype_value, value_range_val) for _ in range(length_val)]
    
    def solve_all_constraints_for_factor(self, factor_name: str, context: Dict[str, Any]) -> Union[List[Any], Any, None]:
        """
        求解因子的所有约束，返回满足所有约束的值
        
        Args:
            factor_name: 因子名称
            context: 上下文（已知因子的值）
            
        Returns:
            - List: 返回取值范围列表
            - 单个值: 固定值
            - None: 无法求解
            
        处理策略：
            1. 获取所有约束
            2. 用第一个约束推导初始候选值
            3. 用后续约束验证和过滤候选值
            4. 返回满足所有约束的值
        """
        constraints = self.find_all_constraints_for_factor(factor_name)
        
        if not constraints:
            return None
        
        result = None
        
        for i, constraint in enumerate(constraints):
            new_result = self.solve_constraint_range(constraint, context)
            if new_result is not None and new_result != [] and new_result:
                result = new_result
            context[factor_name] = result if result is not None else new_result
        
        return result

    def _solve_match_range(self, constraint: Dict, context: Dict[str, Any]) -> List[int]:
        """求解match约束"""
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        source_index = constraint.get('source_index')
        target_index = constraint.get('target_index')
        
        source_shape = context.get(sources[0])
        target_shape = context.get(target)
        if source_shape is None or not isinstance(source_shape, list):
            return []
        
        if source_index is None or source_index >= len(source_shape):
            return []
        
        if target_shape is None or target_index >= len(target_shape):
            return []
        
        source_dim = source_shape[source_index]
        
        return [target_shape[:target_index] + [source_dim] + target_shape[target_index + 1:]]
 
    
    def _solve_inferable_filter_range(self, constraint: Dict, context: Dict[str, Any]) -> List[Any]:
        """
        求解inferable_filter约束，返回兼容类型列表
        
        返回所有与sources兼容的类型列表，而非随机选择一个
        """
        sources = constraint.get('sources', [])
        target_domain = constraint.get('target_domain', [])
        
        source_values = [context.get(s) for s in sources]
        
        if any(v is None for v in source_values):
            return []
        
        if len(source_values) == 1:
            inferred = source_values[0]
        elif len(source_values) >= 2:
            inferred = source_values[0]
            for i in range(1, len(source_values)):
                if inferred is None or source_values[i] is None:
                    return []
                inferred_str = str(inferred) if inferred else None
                source_val_str = str(source_values[i]) if source_values[i] else None
                if inferred_str is None or source_val_str is None:
                    return []
                inferred = infer_two_dtypes(inferred_str, source_val_str)
                if inferred is None:
                    return []
        else:
            return []
        
        if target_domain:
            compatible = []
            inferred_normalized = normalize_dtype(inferred)
            
            for candidate in target_domain:
                candidate_normalized = normalize_dtype(candidate)
                if inferred_normalized is None or candidate_normalized is None:
                    continue
                result = infer_two_dtypes(candidate_normalized, inferred_normalized)
                # if result == inferred_normalized:
                if result is not None:
                    compatible.append(candidate)
            
        #     return compatible if compatible else ([inferred] if inferred in target_domain else [])
        # else:
        #     return [inferred]
            return compatible if compatible else []
        else:
            return []
    
    def _solve_convertible_range(self, constraint: Dict, context: Dict[str, Any]) -> List[Any]:
        """
        求解convertible约束，返回可转换类型列表
        """
        sources = constraint.get('sources', [])
        target_domain = constraint.get('target_domain', [])
        
        target_dtype = context.get(sources[0])
        if target_dtype is None:
            return []
        
        if not target_domain:
            return []
        
        convertible_types = get_convertible_source_dtypes(target_dtype, target_domain)
        return convertible_types
    
    def _solve_broadcast_dim_range(self, constraint: Dict, context: Dict[str, Any]) -> List[int]:
        """求解broadcast_dim约束"""
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        source_index = constraint.get('source_index')
        target_index = constraint.get('target_index')
        
        source_shape = context.get(sources[0])
        target_shape = context.get(target)
        if source_shape is None or not isinstance(source_shape, list):
            return []
        
        if source_index is None or source_index >= len(source_shape):
            return []
        
        if target_shape is None or target_index >= len(target_shape):
            return []
        
        source_dim = source_shape[source_index]
        result = [target_shape[:target_index] + [1] + target_shape[target_index + 1:]]
        broadcast_dim_shape = target_shape[:target_index] + [source_dim] + target_shape[target_index + 1:]
        if math.prod(broadcast_dim_shape) <= MAX_SHAPE_PRODUCT:
            result.append(broadcast_dim_shape)
            result = [list(x) for x in dict.fromkeys(map(tuple, result))]
        return result
    
    def _solve_broadcast_shape_range(self, constraint: Dict, context: Dict[str, Any]) -> List[List[int]]:
        """求解broadcast_shape约束"""
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        mode = constraint.get('mode', 'unidirectional')
        
        source_shape = context.get(sources[0])
        if source_shape is None:
            return []
        
        if mode == 'unidirectional':
            return generate_unidirectional_broadcast_shapes(source_shape)
        else:
            param_name = target.split('.')[0]
            target_dims = context.get(f'{param_name}.dimensions')
            
            max_attempts = 100
            for _ in range(max_attempts):
                shape = generate_broadcast_shapes(source_shape)
                if target_dims is None or len(shape) == target_dims:
                    return shape
            return source_shape
    
    def expand_value_range_for_dtype(self, combinations: List[Dict[str, Any]], random_select: bool = False) -> List[Dict[str, Any]]:
        """
        根据dtype因子值扩展value_range
        
        Args:
            combinations: 组合列表
            random_select: 是否随机选择模式
                - False（默认）：扩展模式，每个value_range都生成独立组合
                - True：随机选择模式，从value_range列表中随机选择一个
        
        当dtype确定后，获取 参数名.value_range_dtype 的取值范围，并更新组合集
        如果该参数的value_range已存在，跳过处理
        """
        expanded_combos = []
        
        for combo in combinations:
            dtype_factors = [(k, v) for k, v in combo.items() if k.endswith('.dtype')]
            
            if not dtype_factors:
                expanded_combos.append(combo)
                continue
            
            current_combos = [combo]
            
            for dtype_factor, dtype_value in dtype_factors:
                param_name = dtype_factor.split('.')[0]
                value_range_key = f'{param_name}.value_range'
                
                value_ranges = self.get_value_range_for_dtype(param_name, dtype_value)
                
                if not value_ranges:
                    continue
                
                new_combos = []
                for current in current_combos:
                    if value_range_key in current:
                        new_combos.append(current)
                        continue
                    
                    if isinstance(value_ranges, list) and len(value_ranges) > 0:
                        if isinstance(value_ranges[0], list):
                            if random_select:
                                random.seed(hash(frozenset(current.items())) % (2**32))
                                selected_range = random.choice(value_ranges)
                                new_combo = dict(current)
                                new_combo[value_range_key] = selected_range
                                new_combos.append(new_combo)
                            else:
                                for vr in value_ranges:
                                    new_combo = dict(current)
                                    new_combo[value_range_key] = vr
                                    new_combos.append(new_combo)
                        else:
                            new_combo = dict(current)
                            new_combo[value_range_key] = value_ranges
                            new_combos.append(new_combo)
                    else:
                        new_combo = dict(current)
                        new_combo[value_range_key] = value_ranges
                        new_combos.append(new_combo)
                
                current_combos = new_combos if new_combos else current_combos
            
            expanded_combos.extend(current_combos)
        
        return expanded_combos
    
    def derive_non_shape_factors(self, combinations: List[Dict[str, Any]], skip_shape_dependent: bool = True) -> List[Dict[str, Any]]:
        """
        推导非shape类因子
        
        Args:
            combinations: 组合列表
            skip_shape_dependent: 是否跳过依赖shape类因子的因子
            
        Returns:
            更新后的组合列表
            
        注意：
            只跳过 *.shape 因子，*.dimensions 因子与其他非shape类因子一样正常处理
            skip_shape_dependent=True 时，额外跳过依赖shape类因子的因子
        """
        sorted_levels = sorted([k for k in self.derivation_order.keys() if k.startswith('level_')])
        
        for level_key in sorted_levels:
            level_factors = self.derivation_order[level_key]
            non_shape_factors = [f for f in level_factors if not self.is_shape_factor(f)]
            
            if skip_shape_dependent:
                non_shape_factors = [f for f in non_shape_factors if not self.factor_depends_on_shape(f)]
            
            if not non_shape_factors:
                continue

            combinations = self._derive_factors_for_level(combinations, non_shape_factors)
        
        return combinations
    
    def _derive_factors_for_level(self, combinations: List[Dict[str, Any]], factors: List[str]) -> List[Dict[str, Any]]:
        """
        为指定因子列表推导取值范围并扩展组合集
        
        更新：使用 solve_all_constraints_for_factor 处理多个约束
        """
        for factor in factors:
            new_combinations = []
            
            for combo in combinations:
                if factor in combo:
                    new_combinations.append(combo)
                    continue
                
                result = self.solve_all_constraints_for_factor(factor, combo)
                
                if result is not None:
                    new_combo = dict(combo)
                    new_combo[factor] = result
                    new_combinations.append(new_combo)
                    # if isinstance(result, list):
                    #     for value in result:
                    #         new_combo = dict(combo)
                    #         new_combo[factor] = value
                    #         new_combinations.append(new_combo)
                    # else:
                    #     new_combo = dict(combo)
                    #     new_combo[factor] = result
                    #     new_combinations.append(new_combo)
                else:
                    domain = self.get_factor_domain(factor)
                    if domain:
                        for value in domain:
                            new_combo = dict(combo)
                            new_combo[factor] = value
                            new_combinations.append(new_combo)
                    else:
                        new_combinations.append(combo)
            
            combinations = new_combinations
            
            if not combinations:
                break
        
        return combinations
    
    def derive_shape_factors(self, combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        推导shape类因子
        
        按照求解配置中的shape因子顺序依次处理：
        1. 优先根据 参数名.dimensions 生成 参数.shape 因子值
        2. 再根据 shape约束关系 更新shape值
        """
        sorted_levels = sorted([k for k in self.derivation_order.keys() if k.startswith('level_')])
        
        for level_key in sorted_levels:
            level_factors = self.derivation_order[level_key]
            
            shape_factors = [f for f in level_factors if self.is_shape_factor(f)]
            
            dimensions_factors = [f for f in shape_factors if f.endswith('.dimensions')]
            shape_only_factors = [f for f in shape_factors if f.endswith('.shape')]
            
            for dim_factor in dimensions_factors:
                combinations = self._derive_single_factor(combinations, dim_factor)
            
            for shape_factor in shape_only_factors:
                combinations = self._derive_shape_factor_with_dimensions(combinations, shape_factor)
        
        return combinations
    
    def derive_shape_dependent_factors(self, combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        推导依赖shape类因子的因子
        
        在shape类因子推导完成后，继续推导那些依赖shape类因子的因子
        """
        sorted_levels = sorted([k for k in self.derivation_order.keys() if k.startswith('level_')])
        
        for level_key in sorted_levels:
            level_factors = self.derivation_order[level_key]
            
            shape_dependent_factors = [
                f for f in level_factors 
                if not self.is_shape_factor(f) and self.factor_depends_on_shape(f)
            ]
            
            if not shape_dependent_factors:
                continue
            
            combinations = self._derive_factors_for_level(combinations, shape_dependent_factors)
        
        return combinations
    
    def _is_enum_param(self, param_name: str, sample_combo: Dict[str, Any]) -> bool:
        """
        判断参数是否为枚举类型
        
        Args:
            param_name: 参数名
            sample_combo: 采样组合（用于检查因子）
        
        Returns:
            True 如果是枚举参数，否则 False
        
        判断标准：
            - 定义了 param.enum_values 因子
            - 未定义 param.value 因子
        """
        enum_values_factor = f'{param_name}.enum_values'
        value_factor = f'{param_name}.value'
        
        # 判断条件：定义了 enum_values 且未定义 value
        has_enum_values = enum_values_factor in sample_combo
        has_value = value_factor in sample_combo
        
        return has_enum_values and not has_value
    
    def expand_value_factors(self, combinations: List[Dict[str, Any]], enum_only: bool = False) -> List[Dict[str, Any]]:
        """
        扩展非tensor类参数的value因子
        
        Args:
            combinations: 组合列表
            enum_only: 是否只处理枚举类型参数
                - False（默认）：处理所有非tensor类参数（枚举 + 非枚举）
                - True：只处理枚举类型参数
        
        对于非tensor类参数（没有shape因子的参数），根据不同场景生成value：
        场景1：存在参数名.enum_values - 扩展参数名.value因子，取值与enum_values相等
        场景2：存在参数名.value_range - 扩展参数名.value因子，调用generate_random_value_by_dtype生成随机值
        """
        non_tensor_params = self._find_non_tensor_params()
        
        if not non_tensor_params:
            return combinations
        
        result_combos = list(combinations)
        
        for param_name in non_tensor_params:
            value_factor = f'{param_name}.value'
            enum_values_factor = f'{param_name}.enum_values'
            value_range_factor = f'{param_name}.value_range'
            dtype_factor = f'{param_name}.dtype'
            
            # 检查是否为枚举类型参数
            is_enum_param = self._is_enum_param(param_name, result_combos[0] if result_combos else {})
            
            # 如果 enum_only=True 且不是枚举参数，跳过
            if enum_only and not is_enum_param:
                continue
            
            new_result = []
            
            for combo in result_combos:
                if value_factor in combo:
                    new_result.append(combo)
                    continue
                
                enum_values = combo.get(enum_values_factor)
                value_range = combo.get(value_range_factor)
                dtype_value = combo.get(dtype_factor)
                
                if enum_values is not None:
                    if isinstance(enum_values, list):
                        for enum_val in enum_values:
                            new_combo = dict(combo)
                            new_combo[value_factor] = enum_val
                            new_result.append(new_combo)
                    else:
                        new_combo = dict(combo)
                        new_combo[value_factor] = enum_values
                        new_result.append(new_combo)
                
                # 非枚举参数处理：从 value_range 生成 value
                # 只在 enum_only=False 时处理
                elif value_range is not None and not enum_only:
                    if dtype_value is None:
                        dtype_value = 'float32'
                    
                    if isinstance(value_range, list) and len(value_range) == 2:
                        try:
                            signed_zero = self._check_signed_zero_range(value_range)
                            if signed_zero is not None:
                                new_combo = dict(combo)
                                new_combo[value_factor] = signed_zero
                                new_result.append(new_combo)
                            else:
                                hashable_combo = frozenset((k, self._make_hashable(v)) for k, v in combo.items())
                                random.seed(hash(hashable_combo) % (2**32))
                                random_value = generate_random_value_by_dtype(dtype_value, value_range)
                                new_combo = dict(combo)
                                new_combo[value_factor] = random_value
                                new_result.append(new_combo)
                        except Exception:
                            new_result.append(combo)
                    else:
                        new_result.append(combo)
                else:
                    new_result.append(combo)
            
            result_combos = new_result
        
        return result_combos
    
    def _find_non_tensor_params(self) -> List[str]:
        """
        查找非tensor类参数（没有shape因子的参数）
        
        Returns:
            非tensor类参数名称列表
        """
        non_tensor_params = []
        
        for param_name, param_data in self.test_factors.items():
            factors = param_data.get('factors', {})
            param_type = param_data.get('type', '')

            if 'Tensor' not in param_type and 'tensor' not in param_type:
                non_tensor_params.append(param_name)
            
            # if param_type == 'scalar':
            #     non_tensor_params.append(param_name)
            #     continue
            
            # has_shape = any(
            #     k.endswith('.shape') or k.endswith('.dimensions') 
            #     for k in factors.keys()
            # )
            
            # if not has_shape:
            #     has_value_factor = any(
            #         k.endswith('.value') or k.endswith('.value_range')
            #         for k in factors.keys()
            #     )
            #     if has_value_factor:
            #         non_tensor_params.append(param_name)
        
        return non_tensor_params
    
    def _derive_single_factor(self, combinations: List[Dict[str, Any]], factor: str) -> List[Dict[str, Any]]:
        """
        推导单个因子
        
        更新：使用 solve_all_constraints_for_factor 处理多个约束
        """
        new_combinations = []
        
        for combo in combinations:
            if factor in combo:
                new_combinations.append(combo)
                continue
            
            result = self.solve_all_constraints_for_factor(factor, combo)
            
            if result is not None:
                new_combo = dict(combo)
                new_combo[factor] = result
                new_combinations.append(new_combo)
                # if isinstance(result, list):
                #     # for value in result:
                #     #     new_combo = dict(combo)
                #     #     new_combo[factor] = value
                #     #     new_combinations.append(new_combo)
                #     value = random.choice(result)
                #     new_combo = dict(combo)
                #     new_combo[factor] = value
                #     new_combinations.append(new_combo)
                # else:
                #     new_combo = dict(combo)
                #     new_combo[factor] = result
                #     new_combinations.append(new_combo)
            else:
                if factor.endswith('.value'):
                    param_name = factor.split('.')[0]
                    value_range = combo.get(f'{param_name}.value_range')
                    dtype_value = combo.get(f'{param_name}.dtype', 'float32')
                    if value_range is not None and isinstance(value_range, list) and len(value_range) == 2:
                        hashable_combo = frozenset((k, self._make_hashable(v)) for k, v in combo.items())
                        seed = hash(hashable_combo) % (2**32)
                        scalar_value = generate_random_value_by_dtype(dtype_value, value_range, seed=seed)
                        if not isinstance(scalar_value, list):
                            new_combo = dict(combo)
                            new_combo[factor] = scalar_value
                            new_combinations.append(new_combo)
                            continue
                    new_combinations.append(combo)
                else:
                    domain = self.get_factor_domain(factor)
                    if domain:
                        value = random.choice(domain)
                        new_combo = dict(combo)
                        new_combo[factor] = value
                        new_combinations.append(new_combo)
                    else:
                        new_combinations.append(combo)
        
        return new_combinations
    
    def _derive_shape_factor_with_dimensions(self, combinations: List[Dict[str, Any]], shape_factor: str) -> List[Dict[str, Any]]:
        """
        根据dimensions因子生成shape因子值
        
        调用utils.py中的generate_random_shape生成shape
        
        更新：使用 solve_all_constraints_for_factor 处理多个约束
        """
        param_name = shape_factor.split('.')[0]
        dim_factor = f'{param_name}.dimensions'
        
        new_combinations = []
        
        for idx, combo in enumerate(combinations):
            if shape_factor not in combo:
                dimensions_value = combo.get(dim_factor)
                if dimensions_value is not None and isinstance(dimensions_value, int):
                    hashable_combo = frozenset((k, self._make_hashable(v)) for k, v in combo.items())
                    random.seed(hash(hashable_combo) % (2**32))
                    shape_value = generate_random_shape(dimensions_value)
                    combo[shape_factor] = shape_value
            
            constraint_result = self.solve_all_constraints_for_factor(shape_factor, combo)
            
            if constraint_result is not None:
                if isinstance(constraint_result, list) and constraint_result:
                    if isinstance(constraint_result[0], list):
                        for shape_value in constraint_result:
                            new_combo = dict(combo)
                            new_combo[shape_factor] = shape_value
                            new_combinations.append(new_combo)
                    else:
                        new_combo = dict(combo)
                        new_combo[shape_factor] = constraint_result
                        new_combinations.append(new_combo)
                else:
                    new_combo = dict(combo)
                    new_combo[shape_factor] = constraint_result
                    new_combinations.append(new_combo)
                continue
            
            else:
                new_combinations.append(combo)
        
        return new_combinations
    
    def generate_complete_cases(self, anchor_combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        生成完整的测试用例
        
        处理流程（简化版）：
        1. 按 06_求解配置.yaml 的层级顺序求解所有因子
        2. 所有约束（包括隐式约束）统一处理
        """
        combinations = list(anchor_combinations)
        
        # 按 derivation_order 顺序求解所有因子
        sorted_levels = sorted([k for k in self.derivation_order.keys() if k.startswith('level_')])
        
        for level_key in sorted_levels:
            level_factors = self.derivation_order[level_key]
            
            # 调整 shape 和 dimensions 的顺序（dimensions 在前）
            level_factors = self.adjust_shape_dimensions_order(level_factors)
            
            for factor in level_factors:
                combinations = self._derive_single_factor(combinations, factor)
        
        return combinations

    def adjust_shape_dimensions_order(self, level_factors: list[str]) -> list[str]:
        """
        调整level_factors顺序：相同参数名的元素中，确保参数名.dimensions在参数名.shape前面
        核心：直接在原列表中交换shape和dimensions的位置，不影响其他元素顺序
        """
        # 复制原列表，避免修改输入
        result = level_factors.copy()
        
        # 遍历每个元素，找到需要调整的shape位置
        for i in range(len(result)):
            current = result[i]
            if not current.endswith('.shape'):
                continue
            
            # 拆分参数名和因子名
            param_name = current.split('.')[0]
            dimensions_item = f"{param_name}.dimensions"
            
            # 查找当前shape之后是否有对应的dimensions
            for j in range(i + 1, len(result)):
                if result[j] == dimensions_item:
                    # 交换位置：将dimensions移到shape前面
                    # 步骤1：删除j位置的dimensions
                    dim_item = result.pop(j)
                    # 步骤2：插入到shape（i位置）前面
                    result.insert(i, dim_item)
                    # 调整后跳出内层循环，避免重复处理
                    break
        
        return result
   
    def derive_shape_and_dependent_factors(self, combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        推导shape类因子和依赖shape类因子的因子
        
        按层级顺序处理：
        1. 先处理dimensions因子
        2. 再处理shape因子（可能依赖dimensions)z
        3. 最后处理依赖shape类因子的非shape因子
        """
        sorted_levels = sorted([k for k in self.derivation_order.keys() if k.startswith('level_')])
        
        for level_key in sorted_levels:
            level_factors = self.derivation_order[level_key]
            level_factors = self.adjust_shape_dimensions_order(level_factors)
            for factor in level_factors:
                if factor.endswith('.shape'):
                    combinations = self._derive_shape_factor_with_dimensions(combinations, factor)
                else:
                    combinations = self._derive_single_factor(combinations, factor)
        
        return combinations
    
    def generate_all_cases(self, max_cases: int = 10000) -> List[Dict[str, Any]]:
        anchor_combinations = self.generate_bc_combinations(max_cases)
        
        all_cases = self.generate_complete_cases(anchor_combinations)
        while len(all_cases) < max_cases:
            all_cases = all_cases + self.generate_complete_cases(anchor_combinations)
        return all_cases[:max_cases]
    
    def save_to_csv(self, cases: List[Dict[str, Any]], output_path: str):
        """保存到CSV文件"""
        if not cases:
            return
        
        all_factors = set()
        for case in cases:
            all_factors.update(case.keys())
        
        sorted_factors = sorted(all_factors)
        
        def format_value(v):
            if isinstance(v, list):
                return str(v)
            if isinstance(v, float):
                if v != v:
                    return "'nan'"
                elif v == float('inf'):
                    return "'inf'"
                elif v == float('-inf'):
                    return "'-inf'"
                elif v == 0.0:
                    if math.copysign(1, v) < 0:
                        return '-0'
                    else:
                        return '+0'
            return v
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted_factors)
            writer.writeheader()
            for case in cases:
                formatted_case = {k: format_value(v) for k, v in case.items()}
                writer.writerow(formatted_case)


def print_summary(cases: List[Dict[str, Any]]):
    """打印摘要"""
    print("\n" + "=" * 70)
    print("因子取值生成摘要")
    print("=" * 70)
    
    print(f"\n生成用例数: {len(cases)}")
    
    if cases:
        all_factors = set()
        for case in cases:
            all_factors.update(case.keys())
        
        print(f"因子总数: {len(all_factors)}")
        
        dtype_factors = [f for f in all_factors if '.dtype' in f]
        shape_factors = [f for f in all_factors if '.shape' in f]
        exist_factors = [f for f in all_factors if '.exist' in f]
        value_range_factors = [f for f in all_factors if '.value_range' in f]
        
        print(f"\n因子类型分布:")
        print(f"  - dtype 因子: {len(dtype_factors)}个")
        print(f"  - shape 因子: {len(shape_factors)}个")
        print(f"  - exist 因子: {len(exist_factors)}个")
        print(f"  - value_range 因子: {len(value_range_factors)}个")
        
        print(f"\n前3个用例预览:")
        for i, case in enumerate(cases[:3]):
            dtype_info = {k: v for k, v in case.items() if '.dtype' in k or '.value_range' in k}
            print(f"  用例{i+1}: {dtype_info}")
    
    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 4:
        print("使用方法: python generate_factor_values.py <求解配置.yaml> <约束定义.yaml> <测试因子.yaml> [输出.csv] [--max-cases N]")
        print("示例: python generate_factor_values.py 07_求解配置.yaml 05_约束定义.yaml 04_测试因子.yaml factor_values.csv --max-cases 100")
        sys.exit(1)
    
    solver_config_path = sys.argv[1]
    constraints_path = sys.argv[2]
    factors_path = sys.argv[3]
    output_path = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith('--') else 'factor_values.csv'
    
    max_cases = 10000
    for i, arg in enumerate(sys.argv):
        if arg == '--max-cases' and i + 1 < len(sys.argv):
            max_cases = int(sys.argv[i + 1])
    
    for path in [solver_config_path, constraints_path, factors_path]:
        if not Path(path).exists():
            print(f"错误: 文件不存在: {path}")
            sys.exit(1)
    
    print(f"正在读取配置文件...")
    print(f"  - 求解配置: {solver_config_path}")
    print(f"  - 约束定义: {constraints_path}")
    print(f"  - 测试因子: {factors_path}")
    
    generator = FactorValueGenerator()
    generator.load_configs(solver_config_path, constraints_path, factors_path)
    
    print(f"\n锚点因子: {len(generator.anchors)}个")
    print(f"推导层级: {len(generator.derivation_order)}层")
    
    print(f"\n正在生成测试用例 (最大 {max_cases} 个)...")
    cases = generator.generate_all_cases(max_cases)
    
    print(f"正在保存到: {output_path}")
    generator.save_to_csv(cases, output_path)
    
    print_summary(cases)
    
    print(f"\n✅ 因子取值已保存到: {output_path}")


if __name__ == "__main__":
    main()
