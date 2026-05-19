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
Bootstrap中位数计算脚本
用于复检流程中计算误差比值中位数的置信区间
"""
import math
import numpy as np


def bootstrap_median(data, n_bootstrap=2000, confidence_level=0.95):
    """
    使用Bootstrap方法计算中位数的置信区间
    
    Args:
        data: 样本数据(numpy array)
        n_bootstrap: Bootstrap重采样次数(默认2000)
        confidence_level: 置信水平(默认0.95)
    
    Returns:
        dict: 包含中位数、置信区间等信息的字典
    """
    n = len(data)
    
    if n < 200:
        return {
            'error': '小样本熔断: N < 200,无法进行Bootstrap统计推断',
            'sample_size': n,
            'is_valid': False
        }
    
    # Bootstrap重采样
    boot_medians = []
    
    for _ in range(n_bootstrap):
        # 有放回地抽取n个样本
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        # 计算中位数
        median = np.median(bootstrap_sample)
        boot_medians.append(median)
    
    boot_medians = np.array(boot_medians)
    
    alpha = 1 - confidence_level
    # 根据golden标准5.3.4节: CI_Lower=第2.5%分位数, CI_Upper=第97.5%分位数
    # 对于N=2000: CI_Lower=第50小(索引49), CI_Upper=第1950小(索引1949)
    # 使用math.ceil确保与golden标准的"第N小"定义一致(1-indexed转0-indexed)
    lower_idx = max(0, math.ceil(alpha / 2 * n_bootstrap) - 1)
    upper_idx = min(n_bootstrap - 1, math.ceil((1 - alpha / 2) * n_bootstrap) - 1)
    
    boot_medians_sorted = np.sort(boot_medians)
    ci_lower = boot_medians_sorted[lower_idx]
    ci_upper = boot_medians_sorted[upper_idx]
    
    # 原始数据的中位数
    original_median = np.median(data)
    
    result = {
        'original_median': original_median,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap,
        'sample_size': n,
        'is_valid': True,
        'boot_medians_mean': np.mean(boot_medians),
        'boot_medians_std': np.std(boot_medians)
    }
    
    return result


def check_ratio_confidence_interval(ratio_data, threshold=1.0, n_bootstrap=2000):
    """
    检查误差比值是否满足置信区间判定标准
    
    通过标准: CI_Lower <= 1.0 (置信区间下限不大于1)
    
    Args:
        ratio_data: Ratio样本数据(numpy array)
        threshold: 判定阈值(默认1.0)
        n_bootstrap: Bootstrap次数
    
    Returns:
        dict: 包含is_pass和置信区间信息的字典
    """
    bootstrap_result = bootstrap_median(ratio_data, n_bootstrap)
    
    if not bootstrap_result['is_valid']:
        return {
            'is_pass': False,
            'error': bootstrap_result['error'],
            'sample_size': bootstrap_result['sample_size']
        }
    
    # 判定: CI_Lower <= threshold
    ci_lower = bootstrap_result['ci_lower']
    is_pass = ci_lower <= threshold
    
    result = {
        'is_pass': is_pass,
        'ci_lower': ci_lower,
        'ci_upper': bootstrap_result['ci_upper'],
        'median': bootstrap_result['original_median'],
        'threshold': threshold,
        'confidence_level': bootstrap_result['confidence_level'],
        'sample_size': bootstrap_result['sample_size']
    }
    
    if not is_pass:
        result['failure_reason'] = f'CI_lower {ci_lower:.3f} > threshold {threshold},表明NPU存在系统性精度恶化'
    
    return result


def bootstrap_median_batch(ratio_data_list, threshold=1.0):
    """
    批量计算多个Ratio数据集的置信区间
    
    Args:
        ratio_data_list: 多个Ratio样本数据的列表
        threshold: 判定阈值
    
    Returns:
        dict: 包含汇总信息的字典
    """
    results = []
    pass_count = 0
    invalid_count = 0
    
    medians = []
    ci_lowers = []
    ci_uppers = []
    
    for ratio_data in ratio_data_list:
        result = check_ratio_confidence_interval(ratio_data, threshold)
        results.append(result)
        
        is_valid = result.get('is_valid', True)
        if not is_valid:
            invalid_count += 1
        elif result.get('is_pass', False):
            pass_count += 1
        
        if is_valid:
            medians.append(result['median'])
            ci_lowers.append(result['ci_lower'])
            ci_uppers.append(result['ci_upper'])
    
    total = len(ratio_data_list)
    valid_count = total - invalid_count
    
    summary = {
        'total_datasets': total,
        'valid_datasets': valid_count,
        'invalid_datasets': invalid_count,
        'pass_count': pass_count,
        'fail_count': valid_count - pass_count,
        'pass_rate': pass_count / valid_count if valid_count > 0 else 0,
        'median_mean': np.mean(medians) if medians else None,
        'ci_lower_mean': np.mean(ci_lowers) if ci_lowers else None,
        'ci_upper_mean': np.mean(ci_uppers) if ci_uppers else None,
        'detail_results': results
    }
    
    return summary


if __name__ == '__main__':
    # 示例用法
    np.random.seed(42)
    
    # 测试1: 大样本,CI_Lower <= 1.0 (通过)
    # 生成Ratio数据,中位数约0.8
    ratio_data1 = np.random.normal(0.8, 0.1, 1000)
    result1 = check_ratio_confidence_interval(ratio_data1, threshold=1.0)
    print(f"测试1 - 通过: is_pass={result1['is_pass']}")
    print(f"  中位数: {result1['median']:.3f}")
    print(f"  置信区间: [{result1['ci_lower']:.3f}, {result1['ci_upper']:.3f}]")
    
    # 测试2: 大样本,CI_Lower > 1.0 (不通过)
    # 生成Ratio数据,中位数约1.3
    ratio_data2 = np.random.normal(1.3, 0.15, 1000)
    result2 = check_ratio_confidence_interval(ratio_data2, threshold=1.0)
    print(f"测试2 - 不通过: is_pass={result2['is_pass']}")
    if not result2['is_pass']:
        print(f"  失败原因: {result2['failure_reason']}")
    
    # 测试3: 小样本熔断(N < 200)
    ratio_data3 = np.random.normal(1.0, 0.1, 150)
    result3 = check_ratio_confidence_interval(ratio_data3)
    print(f"测试3 - 小样本熔断: is_pass={result3['is_pass']}, error={result3.get('error', 'None')}")
    
    # 测试4: 批量计算
    ratio_data_list = [
        np.random.normal(0.9, 0.1, 500),
        np.random.normal(1.1, 0.1, 500),
        np.random.normal(0.85, 0.1, 500)
    ]
    
    summary = bootstrap_median_batch(ratio_data_list)
    print(f"测试4 - 批量: 通过率 {summary['pass_rate']:.2%}")
    print(f"  平均中位数: {summary['median_mean']:.3f}")
    print(f"  平均CI_Lower: {summary['ci_lower_mean']:.3f}")