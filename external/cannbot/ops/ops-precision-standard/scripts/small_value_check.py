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
小值域通过标准检查脚本
当输出结果接近0时,相对误差不稳定,使用小值域标准
"""
import numpy as np


def get_small_value_threshold(dtype):
    """
    获取小值域阈值
    
    注意: Golden文档(COMMERCIAL_OPS_PRECISION_DOCS.md:515-556行)表格列名为:
    - "小值域阈值(Small Value Threshold)": 用于判定是否是小值域元素(|golden| < threshold)
    - "小值域error指标": 用于判定误差是否过大(|actual - golden| > error)
    
    本函数返回的字典中:
    - 'threshold': 对应Golden文档的"小值域阈值",用于判定是否启用小值域标准
    - 'error': 对应Golden文档的"小值域error指标",用于计算ErrorCount
    
    Returns:
        dict: 包含threshold和error阈值的字典
    """
    dtype_str = str(dtype)
    
    # Golden文档命名惯例: 
    # threshold = "小值域阈值(Small Value Threshold)"，用于判定 |golden| < threshold
    # error = "小值域error指标"，用于判定 |actual - golden| > error
    thresholds = {
        'float16': {'threshold': 2 ** (-11), 'error': 2 ** (-16)},
        'bfloat16': {'threshold': 2 ** (-8), 'error': 2 ** (-16)},
        'float32': {'threshold': 2 ** (-14), 'error': 2 ** (-30)},
        'float64': {'threshold': 2 ** (-14), 'error': 2 ** (-30)},
        'hifloat32': {'threshold': 2 ** (-12), 'error': 2 ** (-28)},
    }
    
    # 处理float8的特殊情况
    if 'float8_e4m3' in dtype_str.lower():
        return {'threshold': 2 ** (-4), 'error': 2 ** (-6)}
    elif 'float8_e5m2' in dtype_str.lower():
        return {'threshold': 2 ** (-3), 'error': 2 ** (-5)}
    
    return thresholds.get(dtype_str, {'threshold': 2 ** (-14), 'error': 2 ** (-30)})


def calculate_error_count(actual, golden, threshold, error):
    """
    计算小值域数值错误数量(ErrorCount)
    
    Golden文档(COMMERCIAL_OPS_PRECISION_DOCS.md:558-565行)公式:
    ErrorCount = sum(|golden| < threshold and |actual - golden| > error)
    
    Args:
        actual: NPU输出
        golden: CPU标杆
        threshold: 小值域阈值(Small Value Threshold)，判定是否是小值域元素
        error: 小值域error指标，判定误差是否过大
    
    Returns:
        int: ErrorCount数量
    """
    # 判定是否是小值域元素: |golden| < threshold(小值域阈值)
    is_small_value = np.abs(golden) < threshold
    # 判定误差是否过大: |actual - golden| > error(小值域error指标)
    is_large_error = np.abs(actual - golden) > error
    
    error_count = np.sum(is_small_value & is_large_error)
    
    return error_count


def check_small_value_precision(npu_output, golden_output, third_party_output):
    """
    检查小值域精度
    
    通过标准:
    ErrorCount_npu / max(ErrorCount_third_party, 1) <= 2
    
    注意: 小值域标准要求双标杆比对,必须提供三方芯片输出。
    根据Golden标准(Section 4.5.3),小值域通过标准公式明确要求:
    ErrorCount_npu / max(ErrorCount_三方芯片, 1) <= 2
    
    Args:
        npu_output: NPU算子输出(numpy array)
        golden_output: CPU标杆输出(numpy array)
        third_party_output: 三方芯片输出(numpy array,必须提供)
    
    Returns:
        dict: 包含is_pass和ErrorCount信息的字典
    """
    if third_party_output is None:
        return {
            'is_pass': False,
            'error': '小值域标准要求双标杆比对,必须提供third_party_output',
            'npu_dtype': str(npu_output.dtype),
            'golden_dtype': str(golden_output.dtype)
        }
    
    # 获取阈值
    thresholds = get_small_value_threshold(npu_output.dtype)
    threshold = thresholds['threshold']
    error = thresholds['error']
    
    # 计算NPU的ErrorCount
    error_count_npu = calculate_error_count(npu_output, golden_output, threshold, error)
    
    # 计算三方芯片的ErrorCount
    error_count_third = calculate_error_count(third_party_output, golden_output, threshold, error)
    
    # 统计小值域内的总元素数
    small_value_count = np.sum(np.abs(golden_output) < threshold)
    
    # 计算ratio
    ratio = error_count_npu / max(error_count_third, 1)
    
    is_pass = ratio <= 2
    
    result = {
        'npu_dtype': str(npu_output.dtype),
        'golden_dtype': str(golden_output.dtype),
        'small_value_threshold': threshold,
        'small_value_error': error,
        'error_count_npu': error_count_npu,
        'error_count_third': error_count_third,
        'error_count_ratio': ratio,
        'small_value_count': small_value_count,
        'total_elements': npu_output.size,
        'comparison_method': 'dual_benchmark',
        'is_pass': is_pass,
        'threshold_ratio': 2.0
    }
    
    if not is_pass:
        result['failure_reason'] = f'ErrorCount ratio {ratio:.2f} > 2.0'
    
    return result


def should_use_small_value_standard(npu_output, golden_output, min_ratio=0.01):
    """
    判断是否应该启用小值域标准

    当golden值小于对应dtype的Small Value Threshold时，相对误差计算不稳定，
    需使用小值域标准。根据Golden标准(Section 4.5.3):
    "当真值小于Small Value Threshold时，采用小值域通过标准。"

    注意: 小值域标准只针对小值域子集计算ErrorCount,与正常值域的MARE/MERE评估
    并行使用,而非互斥替换。

    Args:
        npu_output: NPU输出
        golden_output: CPU标杆
        min_ratio: 小值域元素占比阈值(默认0.01即1%)。
                   当小值域元素占比超过此比例时才启用小值域标准,
                   避免极少数小值元素导致不必要的评估切换。
                   设为0时回退到"任意元素触发"的保守策略。

    Returns:
        bool: 是否应启用小值域标准
    """
    thresholds = get_small_value_threshold(npu_output.dtype)
    threshold = thresholds['threshold']

    total_elements = golden_output.size
    if total_elements == 0:
        return False

    small_value_count = np.sum(np.abs(golden_output) < threshold)

    if min_ratio <= 0:
        # 保守策略: 任意小值域元素即触发
        return small_value_count > 0

    # 比例策略: 小值域元素占比超过阈值才触发
    small_value_ratio = small_value_count / total_elements
    return small_value_ratio >= min_ratio


if __name__ == '__main__':
    # 示例用法
    np.random.seed(42)
    
    # 测试1: FP16小值域通过
    golden1 = np.array([2**(-12), 2**(-12), 2**(-12), 2**(-12)], dtype=np.float16)  # 小于阈值
    npu1 = np.array([2**(-12) + 2**(-17), 2**(-12), 2**(-12), 2**(-12)], dtype=np.float16)  # 一个误差较大
    third1 = np.array([2**(-12) + 2**(-17), 2**(-12), 2**(-12), 2**(-12)], dtype=np.float16)
    
    result1 = check_small_value_precision(npu1, golden1, third1)
    print(f"测试1 - FP16小值域: pass={result1['is_pass']}, "
          f"error_count_npu={result1['error_count_npu']}, "
          f"ratio={result1.get('error_count_ratio', 'N/A')}")
    
    # 测试2: FP32小值域
    golden2 = np.array([2**(-15), 2**(-15), 2**(-15)], dtype=np.float32)
    npu2 = np.array([2**(-15), 2**(-15), 2**(-15)], dtype=np.float32)
    third2 = np.array([2**(-15), 2**(-15), 2**(-15)], dtype=np.float32)
    
    result2 = check_small_value_precision(npu2, golden2, third2)
    print(f"测试2 - FP32小值域: pass={result2['is_pass']}, "
          f"small_value_count={result2['small_value_count']}")
    
    # 测试3: 判断是否应启用小值域标准(默认min_ratio=0.01)
    golden3 = np.random.randn(1000).astype(np.float16)
    golden3[:10] = 2**(-12)  # 设置10个小值元素,占比1%

    # 默认min_ratio=0.01(1%), 10/1000=1%刚好触发
    should_use_default = should_use_small_value_standard(np.zeros_like(golden3), golden3)
    print(f"测试3a - 默认min_ratio=0.01: {should_use_default}")
    
    # min_ratio=0(保守策略),任意小值域元素即触发
    should_use_conservative = should_use_small_value_standard(np.zeros_like(golden3), golden3, min_ratio=0)
    print(f"测试3b - 保守策略min_ratio=0: {should_use_conservative}")
    
    # min_ratio=0.05(5%), 1%占比不触发
    should_use_strict = should_use_small_value_standard(np.zeros_like(golden3), golden3, min_ratio=0.05)
    print(f"测试3c - 严格策略min_ratio=0.05: {should_use_strict}")
    
    # 测试4: 缺少三方芯片(应该报错)
    golden4 = np.array([2**(-12), 2**(-12)], dtype=np.float16)
    npu4 = np.array([2**(-12) + 2**(-17), 2**(-12)], dtype=np.float16)
    
    result4 = check_small_value_precision(npu4, golden4, None)
    print(f"测试4 - 缺少三方芯片: pass={result4['is_pass']}, error={result4.get('error', 'None')}")
