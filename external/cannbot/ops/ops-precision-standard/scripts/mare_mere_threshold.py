# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
浮点计算类算子精度检查脚本 - 生态标准(Threshold)
用于生态贡献算子的精度验证(单标杆比对)
"""
import numpy as np


def calculate_mare(actual, golden):
    """
    计算最大相对误差(MARE)
    MARE = max(|actual - golden| / (|golden| + 1e-7))
    """
    relative_errors = np.abs(actual - golden) / (np.abs(golden) + 1e-7)
    return np.max(relative_errors)


def calculate_mere(actual, golden):
    """
    计算平均相对误差(MERE)
    MERE = avg(|actual - golden| / (|golden| + 1e-7))
    """
    relative_errors = np.abs(actual - golden) / (np.abs(golden) + 1e-7)
    return np.mean(relative_errors)


def get_threshold_by_dtype(dtype):
    """
    根据数据类型获取通过阈值
    
    Returns:
        float: Threshold值
    """
    dtype_str = str(dtype).lower().replace(' ', '').replace('_', '')

    thresholds = {
        'float16': 2 ** (-10),
        'bfloat16': 2 ** (-7),
        'float32': 2 ** (-13),
        'float64': 2 ** (-13),
        'hifloat32': 2 ** (-11),
    }

    if 'float8e4m3' in dtype_str or 'float8e4m3fn' in dtype_str:
        return 2 ** (-3)
    elif 'float8e5m2' in dtype_str:
        return 2 ** (-2)

    return thresholds.get(dtype_str, 2 ** (-13))


def check_precision_threshold(npu_output, golden_output):
    """
    检查浮点算子精度(单标杆比对,Threshold标准)
    
    通过标准:
    - MERE < Threshold
    - MARE < 10 * Threshold
    
    Args:
        npu_output: NPU算子输出(numpy array)
        golden_output: CPU标杆输出(numpy array)
    
    Returns:
        dict: 包含is_pass和各种误差指标的字典
    """
    # 计算误差指标
    mare = calculate_mare(npu_output, golden_output)
    mere = calculate_mere(npu_output, golden_output)
    
    # 获取阈值
    threshold = get_threshold_by_dtype(npu_output.dtype)
    mare_threshold = 10 * threshold
    
    # 判定是否通过
    mere_pass = mere < threshold
    mare_pass = mare < mare_threshold
    
    is_pass = mere_pass and mare_pass
    
    result = {
        'is_pass': is_pass,
        'mare': mare,
        'mere': mere,
        'threshold': threshold,
        'mare_threshold': mare_threshold,
        'mere_pass': mere_pass,
        'mare_pass': mare_pass,
        'npu_dtype': str(npu_output.dtype),
        'golden_dtype': str(golden_output.dtype),
        'shape': npu_output.shape
    }
    
    if not is_pass:
        result['failure_reasons'] = []
        if not mere_pass:
            result['failure_reasons'].append(f'MERE {mere:.6f} >= threshold {threshold:.6f}')
        if not mare_pass:
            result['failure_reasons'].append(f'MARE {mare:.6f} >= mare_threshold {mare_threshold:.6f}')
    
    return result


def check_precision_threshold_batch(outputs_list):
    """
    批量检查多个用例的浮点算子精度(Threshold标准)
    
    Args:
        outputs_list: [(npu_output, golden_output), ...]列表
    
    Returns:
        dict: 包含汇总信息的字典
    """
    results = []
    pass_count = 0
    
    mare_values = []
    mere_values = []
    
    for npu_out, golden_out in outputs_list:
        result = check_precision_threshold(npu_out, golden_out)
        results.append(result)
        
        if result['is_pass']:
            pass_count += 1
        
        mare_values.append(result['mare'])
        mere_values.append(result['mere'])
    
    total = len(outputs_list)
    summary = {
        'total_cases': total,
        'pass_count': pass_count,
        'fail_count': total - pass_count,
        'pass_rate': pass_count / total if total > 0 else 0,
        'mare_mean': np.mean(mare_values),
        'mare_max': np.max(mare_values),
        'mere_mean': np.mean(mere_values),
        'mere_max': np.max(mere_values),
        'detail_results': results
    }
    
    return summary


if __name__ == '__main__':
    # 示例用法
    np.random.seed(42)
    
    # 测试1: FP16通过
    golden1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    npu1 = np.array([1.0001, 2.0001, 3.0001, 4.0001], dtype=np.float16)
    
    result1 = check_precision_threshold(npu1, golden1)
    print(f"测试1 - FP16通过: {result1['is_pass']}")
    print(f"  MERE: {result1['mere']:.6f}, threshold: {result1['threshold']:.6f}")
    print(f"  MARE: {result1['mare']:.6f}, mare_threshold: {result1['mare_threshold']:.6f}")
    
    # 测试2: FP32通过
    golden2 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    npu2 = np.array([10.00001, 20.00001, 30.00001], dtype=np.float32)
    
    result2 = check_precision_threshold(npu2, golden2)
    print(f"测试2 - FP32通过: {result2['is_pass']}")
    print(f"  MERE: {result2['mere']:.6f}, threshold: {result2['threshold']:.6f}")
    
    # 测试3: BF16不通过
    golden3 = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 模拟BF16精度
    npu3 = np.array([1.01, 2.01, 3.01], dtype=np.float32)
    
    # 手动设置BF16阈值测试
    threshold_bf16 = 2 ** (-7)
    mare3 = calculate_mare(npu3, golden3)
    mere3 = calculate_mere(npu3, golden3)
    
    print(f"测试3 - BF16阈值测试: mere={mere3:.6f}, threshold={threshold_bf16:.6f}")
    print(f"  是否通过: mere < threshold? {mere3 < threshold_bf16}")
    
    # 测试4: 批量检查
    outputs_list = [
        (np.array([1.0, 2.0], dtype=np.float16), np.array([1.0, 2.0], dtype=np.float16)),
        (np.array([3.0, 4.0], dtype=np.float16), np.array([3.0, 4.0], dtype=np.float16))
    ]
    
    summary = check_precision_threshold_batch(outputs_list)
    print(f"测试4 - 批量: 通过率 {summary['pass_rate']:.2%}")