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
浮点计算类算子精度检查脚本 - 商用标准(Ratio)
用于matmul, conv, softmax等浮点算子的精度验证(双标杆比对)
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


def calculate_rmse(actual, golden):
    """
    计算均方根误差(RMSE)
    RMSE = sqrt(mean((actual - golden)^2))
    """
    squared_errors = (actual - golden) ** 2
    return np.sqrt(np.mean(squared_errors))


def check_precision_ratio(npu_output, golden_output, third_party_output, precision_level='L1'):
    """
    检查浮点算子精度(双标杆比对,Ratio标准)
    
    Args:
        npu_output: NPU算子输出(numpy array)
        golden_output: CPU标杆输出(numpy array)
        third_party_output: 三方芯片输出(numpy array)
        precision_level: 精度等级(L0/L1/L2)
    
    Returns:
        dict: 包含is_pass和各种误差指标的字典
    """
    # 计算NPU相对于Golden的误差
    mare_npu = calculate_mare(npu_output, golden_output)
    mere_npu = calculate_mere(npu_output, golden_output)
    rmse_npu = calculate_rmse(npu_output, golden_output)
    
    # 计算三方芯片相对于Golden的误差
    mare_third = calculate_mare(third_party_output, golden_output)
    mere_third = calculate_mere(third_party_output, golden_output)
    rmse_third = calculate_rmse(third_party_output, golden_output)
    
    # 计算Ratio
    # 当三方芯片误差为0时: 若NPU误差也为0则ratio=1(同等完美), 否则ratio=float('inf')(NPU劣于三方)
    if mare_third > 0:
        mare_ratio = mare_npu / mare_third
    elif mare_npu > 0:
        mare_ratio = float('inf')
    else:
        mare_ratio = 1.0

    if mere_third > 0:
        mere_ratio = mere_npu / mere_third
    elif mere_npu > 0:
        mere_ratio = float('inf')
    else:
        mere_ratio = 1.0

    if rmse_third > 0:
        rmse_ratio = rmse_npu / rmse_third
    elif rmse_npu > 0:
        rmse_ratio = float('inf')
    else:
        rmse_ratio = 1.0
    
    # 精度等级阈值
    thresholds = {
        'L0': {'mare_ratio': 10.0, 'mere_ratio': 2.0, 'rmse_ratio': 2.0},
        'L1': {'mare_ratio': 5.0, 'mere_ratio': 1.5, 'rmse_ratio': 1.5},
        'L2': {'mare_ratio': 2.0, 'mere_ratio': 1.2, 'rmse_ratio': 1.2}
    }
    
    threshold = thresholds.get(precision_level, thresholds['L0'])
    
    # 判定是否通过
    mare_pass = mare_ratio <= threshold['mare_ratio']
    mere_pass = mere_ratio <= threshold['mere_ratio']
    rmse_pass = rmse_ratio <= threshold['rmse_ratio']
    
    is_pass = mare_pass and mere_pass and rmse_pass
    
    result = {
        'is_pass': is_pass,
        'precision_level': precision_level,
        'mare_npu': mare_npu,
        'mere_npu': mere_npu,
        'rmse_npu': rmse_npu,
        'mare_third': mare_third,
        'mere_third': mere_third,
        'rmse_third': rmse_third,
        'mare_ratio': mare_ratio,
        'mere_ratio': mere_ratio,
        'rmse_ratio': rmse_ratio,
        'mare_pass': mare_pass,
        'mere_pass': mere_pass,
        'rmse_pass': rmse_pass,
        'thresholds_used': threshold,
        'npu_dtype': str(npu_output.dtype),
        'golden_dtype': str(golden_output.dtype),
        'third_dtype': str(third_party_output.dtype),
        'shape': npu_output.shape
    }
    
    if not is_pass:
        result['failure_reasons'] = []
        if not mare_pass:
            result['failure_reasons'].append(f'MARE ratio {mare_ratio:.3f} > {threshold["mare_ratio"]}')
        if not mere_pass:
            result['failure_reasons'].append(f'MERE ratio {mere_ratio:.3f} > {threshold["mere_ratio"]}')
        if not rmse_pass:
            result['failure_reasons'].append(f'RMSE ratio {rmse_ratio:.3f} > {threshold["rmse_ratio"]}')
    
    return result


def check_precision_ratio_batch(outputs_list, precision_level='L1'):
    """
    批量检查多个用例的浮点算子精度(Ratio标准)
    
    Args:
        outputs_list: [(npu_output, golden_output, third_party_output), ...]列表
        precision_level: 精度等级
    
    Returns:
        dict: 包含汇总信息的字典
    """
    results = []
    pass_count = 0
    
    mare_ratios = []
    mere_ratios = []
    rmse_ratios = []
    
    for npu_out, golden_out, third_out in outputs_list:
        result = check_precision_ratio(npu_out, golden_out, third_out, precision_level)
        results.append(result)
        
        if result['is_pass']:
            pass_count += 1
        
        mare_ratios.append(result['mare_ratio'])
        mere_ratios.append(result['mere_ratio'])
        rmse_ratios.append(result['rmse_ratio'])
    
    total = len(outputs_list)
    summary = {
        'total_cases': total,
        'pass_count': pass_count,
        'fail_count': total - pass_count,
        'pass_rate': pass_count / total if total > 0 else 0,
        'mare_ratio_mean': np.mean(mare_ratios),
        'mare_ratio_max': np.max(mare_ratios),
        'mere_ratio_mean': np.mean(mere_ratios),
        'mere_ratio_max': np.max(mere_ratios),
        'rmse_ratio_mean': np.mean(rmse_ratios),
        'rmse_ratio_max': np.max(rmse_ratios),
        'detail_results': results
    }
    
    return summary


if __name__ == '__main__':
    # 示例用法
    np.random.seed(42)
    
    # 测试1: L1标准,通过
    golden1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    npu1 = np.array([1.001, 2.001, 3.002, 4.001], dtype=np.float16)
    third1 = np.array([1.002, 2.002, 3.003, 4.002], dtype=np.float16)
    
    result1 = check_precision_ratio(npu1, golden1, third1, precision_level='L1')
    print(f"测试1 - L1通过: {result1['is_pass']}")
    print(f"  MARE ratio: {result1['mare_ratio']:.3f}, MERE ratio: {result1['mere_ratio']:.3f}, RMSE ratio: {result1['rmse_ratio']:.3f}")
    
    # 测试2: L2标准,不通过
    golden2 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    npu2 = np.array([10.05, 20.05, 30.06], dtype=np.float32)  # 误差较大
    third2 = np.array([10.02, 20.02, 30.03], dtype=np.float32)
    
    result2 = check_precision_ratio(npu2, golden2, third2, precision_level='L2')
    print(f"测试2 - L2不通过: {result2['is_pass']}")
    if not result2['is_pass']:
        print(f"  失败原因: {result2['failure_reasons']}")
    
    # 测试3: 批量检查
    outputs_list = [
        (np.array([1.0, 2.0], dtype=np.float16), 
         np.array([1.0, 2.0], dtype=np.float16),
         np.array([1.001, 2.001], dtype=np.float16)),
        (np.array([3.0, 4.0], dtype=np.float16),
         np.array([3.0, 4.0], dtype=np.float16),
         np.array([3.001, 4.001], dtype=np.float16))
    ]
    
    summary = check_precision_ratio_batch(outputs_list, precision_level='L0')
    print(f"测试3 - 批量(L0): 通过率 {summary['pass_rate']:.2%}")
    print(f"  平均MARE ratio: {summary['mare_ratio_mean']:.3f}")