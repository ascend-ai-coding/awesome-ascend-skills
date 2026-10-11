# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
浮点计算类算子精度检查脚本 - 混合容差(Mixed Tolerance, atol/rtol)
对齐生态算子开源精度标准(experimental_standard.md)第 2 节。
"""
import numpy as np


_TOLERANCE_TABLE = {
    'float16': (2 ** (-9), 2 ** (-9), 0.99, 1e-1),
    'bfloat16': (2 ** (-6), 2 ** (-6), 0.99, 1e-0),
    'float32': (2 ** (-10), 2 ** (-16), 0.99, 1e-2),
    'hifloat32': (2 ** (-9), 2 ** (-10), 0.99, 1e-1),
    'float8e4m3': (2 ** (-2), 2 ** (-4), 0.99, 1e-0),
    'float8e4m3fn': (2 ** (-2), 2 ** (-4), 0.99, 1e-0),
    'float8e5m2': (2 ** (-1), 2 ** (-3), 0.99, 1e-1),
}

_ULP_FACTOR = 32

_ULP_TABLE = {
    'float16': 2 ** (-10),
    'bfloat16': 2 ** (-7),
    'float32': 2 ** (-23),
    'hifloat32': 2 ** (-24),
    'float8e4m3': 2 ** (-3),
    'float8e4m3fn': 2 ** (-3),
    'float8e5m2': 2 ** (-2),
}


def _normalize_dtype(dtype):
    return str(dtype).lower().replace(' ', '').replace('_', '')


def get_tolerance_by_dtype(dtype):
    """
    根据数据类型获取混合容差阈值

    Returns:
        tuple: (rtol, atol, required_matched_ratio, max_abs_error_limit_fixed)
    """
    dtype_str = _normalize_dtype(dtype)
    if dtype_str in _TOLERANCE_TABLE:
        return _TOLERANCE_TABLE[dtype_str]
    if 'float8e4m3' in dtype_str:
        return _TOLERANCE_TABLE['float8e4m3']
    if 'float8e5m2' in dtype_str:
        return _TOLERANCE_TABLE['float8e5m2']
    raise ValueError(f'Unsupported dtype: {dtype}')


def _ulp_at_one(dtype):
    dtype_str = _normalize_dtype(dtype)
    if dtype_str in _ULP_TABLE:
        return _ULP_TABLE[dtype_str]
    if 'float8e4m3' in dtype_str:
        return _ULP_TABLE['float8e4m3']
    if 'float8e5m2' in dtype_str:
        return _ULP_TABLE['float8e5m2']
    raise ValueError(f'Unsupported dtype: {dtype}')


def check_mixed_tolerance(npu_output, golden_output):
    """
    检查浮点算子精度(混合容差 atol/rtol 标准)

    通过标准(同时满足):
    1. matched_ratio >= required_matched_ratio
    2. max_abs_error <= max_abs_error_limit
       max_abs_error_limit = max(fixed_limit, 32 * ULP)

    Args:
        npu_output: NPU 算子输出(numpy array)
        golden_output: CPU 标杆输出(numpy array)

    Returns:
        dict: 包含 is_pass 和各误差指标的字典
    """
    rtol, atol, required_matched_ratio, fixed_limit = get_tolerance_by_dtype(npu_output.dtype)

    if npu_output.shape != golden_output.shape:
        raise ValueError(
            f"Shape mismatch: npu {npu_output.shape} vs golden {golden_output.shape}"
        )

    ulp_limit = _ULP_FACTOR * _ulp_at_one(npu_output.dtype)
    max_abs_error_limit = max(fixed_limit, ulp_limit)

    abs_error = np.abs(npu_output.astype(np.float64) - golden_output.astype(np.float64))
    element_threshold = atol + rtol * np.abs(golden_output.astype(np.float64))
    element_passed = abs_error <= element_threshold

    total = npu_output.size
    if total == 0:
        matched_ratio = 1.0
        max_abs_error = 0.0
    else:
        matched_ratio = float(np.sum(element_passed)) / total
        max_abs_error = float(np.max(abs_error))

    ratio_pass = matched_ratio >= required_matched_ratio
    max_err_pass = max_abs_error <= max_abs_error_limit
    is_pass = ratio_pass and max_err_pass

    result = {
        'is_pass': is_pass,
        'matched_ratio': matched_ratio,
        'required_matched_ratio': required_matched_ratio,
        'max_abs_error': max_abs_error,
        'max_abs_error_limit': max_abs_error_limit,
        'max_abs_error_limit_fixed': fixed_limit,
        'max_abs_error_limit_ulp': ulp_limit,
        'rtol': rtol,
        'atol': atol,
        'ratio_pass': ratio_pass,
        'max_err_pass': max_err_pass,
        'npu_dtype': str(npu_output.dtype),
        'golden_dtype': str(golden_output.dtype),
        'shape': npu_output.shape,
    }

    if not is_pass:
        result['failure_reasons'] = []
        if not ratio_pass:
            result['failure_reasons'].append(
                f'matched_ratio {matched_ratio:.6f} < required {required_matched_ratio}'
            )
        if not max_err_pass:
            result['failure_reasons'].append(
                f'max_abs_error {max_abs_error:.6g} > limit {max_abs_error_limit:.6g}'
            )

    return result


def check_mixed_tolerance_batch(outputs_list):
    """
    批量检查多个用例的浮点算子精度(混合容差标准)

    Args:
        outputs_list: [(npu_output, golden_output), ...] 列表

    Returns:
        dict: 包含汇总信息的字典
    """
    results = []
    pass_count = 0
    matched_ratios = []

    for npu_out, golden_out in outputs_list:
        result = check_mixed_tolerance(npu_out, golden_out)
        results.append(result)
        if result['is_pass']:
            pass_count += 1
        matched_ratios.append(result['matched_ratio'])

    total = len(outputs_list)
    summary = {
        'total_cases': total,
        'pass_count': pass_count,
        'fail_count': total - pass_count,
        'pass_rate': pass_count / total if total > 0 else 0,
        'matched_ratio_mean': float(np.mean(matched_ratios)) if matched_ratios else 0.0,
        'matched_ratio_min': float(np.min(matched_ratios)) if matched_ratios else 0.0,
        'detail_results': results,
    }

    return summary
