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
随机数生成算子分布检验脚本
用于rand, randn, uniform等随机数生成算子的精度验证
"""
import numpy as np
from scipy import stats


def ks_test_distribution(npu_output, expected_distribution='uniform', params=None):
    """
    使用KS检验(Kolmogorov-Smirnov test)检查随机数分布
    
    Args:
        npu_output: NPU生成的随机数(numpy array)
        expected_distribution: 期望分布类型('uniform'/'normal'/'exponential'等)
        params: 分布参数(dict)
            - uniform: {'low': 0, 'high': 1}
            - normal: {'mean': 0, 'std': 1}
    
    Returns:
        dict: 包含p值和检验结果的字典
    """
    if params is None:
        params = {}
    
    # 执行KS检验
    if expected_distribution == 'uniform':
        low = params.get('low', 0)
        high = params.get('high', 1)
        statistic, p_value = stats.kstest(npu_output, 'uniform', args=(low, high - low))
    
    elif expected_distribution == 'normal':
        mean = params.get('mean', 0)
        std = params.get('std', 1)
        statistic, p_value = stats.kstest(npu_output, 'norm', args=(mean, std))
    
    elif expected_distribution == 'exponential':
        scale = params.get('scale', 1)
        statistic, p_value = stats.kstest(npu_output, 'expon', args=(0, scale))
    
    else:
        return {
            'is_valid': False,
            'error': f'Unsupported distribution: {expected_distribution}'
        }
    
    # 判定标准
    alpha = 0.01  # 显著性水平
    is_pass = p_value > alpha
    
    result = {
        'is_valid': True,
        'ks_statistic': statistic,
        'p_value': p_value,
        'alpha': alpha,
        'is_pass': is_pass,
        'expected_distribution': expected_distribution,
        'params': params,
        'sample_size': len(npu_output),
        'interpretation': ''
    }
    
    if is_pass:
        result['interpretation'] = f'p_value {p_value:.4f} > alpha {alpha},分布与期望一致'
    else:
        result['interpretation'] = f'p_value {p_value:.4f} <= alpha {alpha},分布与期望不一致'
    
    return result


def check_random_distribution_batch(outputs_list, expected_distribution='uniform', 
                                     params=None, N=100, alpha=0.01):
    """
    批量检查随机数生成算子的分布
    
    通过标准:
    至少((1-α) + z×√(α(1-α)/N))×100%的测试用例满足p>α
    z为正态分布99.9%截尾点=-3.0902
    
    Args:
        outputs_list: 多个随机数输出样本的列表
        expected_distribution: 期望分布
        params: 分布参数
        N: 测试次数(默认100),与Golden标准中的N参数一致
        alpha: 显著性水平(默认0.01)
    
    Returns:
        dict: 包含汇总信息的字典
    """
    results = []
    pass_count = 0
    
    p_values = []
    
    for output in outputs_list:
        result = ks_test_distribution(output, expected_distribution, params)
        results.append(result)
        
        if result['is_valid'] and result['is_pass']:
            pass_count += 1
        
        if result['is_valid']:
            p_values.append(result['p_value'])
    
    total = len(outputs_list)
    
    # 计算理论通过率阈值
    z = -3.0902  # 正态分布99.9%截尾点
    theoretical_pass_rate = (1 - alpha) + z * np.sqrt(alpha * (1 - alpha) / N)
    theoretical_pass_rate_percent = theoretical_pass_rate * 100
    
    # 实际通过率
    actual_pass_rate = pass_count / total if total > 0 else 0
    actual_pass_rate_percent = actual_pass_rate * 100
    
    # 判定
    is_pass = actual_pass_rate_percent >= theoretical_pass_rate_percent
    
    summary = {
        'total_tests': total,
        'pass_count': pass_count,
        'fail_count': total - pass_count,
        'actual_pass_rate': actual_pass_rate,
        'actual_pass_rate_percent': actual_pass_rate_percent,
        'theoretical_pass_rate': theoretical_pass_rate,
        'theoretical_pass_rate_percent': theoretical_pass_rate_percent,
        'is_pass': is_pass,
        'expected_distribution': expected_distribution,
        'alpha': alpha,
        'N': N,
        'p_value_mean': np.mean(p_values) if p_values else None,
        'p_value_std': np.std(p_values) if p_values else None,
        'detail_results': results
    }
    
    if not is_pass:
        summary['failure_reason'] = (
            f'实际通过率 {actual_pass_rate_percent:.2f}% '
            f'< 理论阈值 {theoretical_pass_rate_percent:.2f}%'
        )
    
    return summary


def check_distribution_statistics(npu_output, expected_distribution='uniform', params=None):
    """
    检查随机数分布的统计特性(均值、方差等)
    
    Args:
        npu_output: NPU生成的随机数
        expected_distribution: 期望分布
        params: 分布参数
    
    Returns:
        dict: 包含统计特性信息的字典
    """
    if params is None:
        params = {}
    
    actual_mean = np.mean(npu_output)
    actual_std = np.std(npu_output)
    actual_min = np.min(npu_output)
    actual_max = np.max(npu_output)
    
    if expected_distribution == 'uniform':
        low = params.get('low', 0)
        high = params.get('high', 1)
        expected_mean = (low + high) / 2
        expected_std = (high - low) / np.sqrt(12)
        expected_min = low
        expected_max = high
    
    elif expected_distribution == 'normal':
        mean = params.get('mean', 0)
        std = params.get('std', 1)
        expected_mean = mean
        expected_std = std
        expected_min = None
        expected_max = None
    
    else:
        return {
            'is_valid': False,
            'error': f'Unsupported distribution: {expected_distribution}'
        }
    
    result = {
        'is_valid': True,
        'actual_mean': actual_mean,
        'actual_std': actual_std,
        'actual_min': actual_min,
        'actual_max': actual_max,
        'expected_mean': expected_mean,
        'expected_std': expected_std,
        'expected_min': expected_min,
        'expected_max': expected_max,
        'mean_error': abs(actual_mean - expected_mean),
        'std_error': abs(actual_std - expected_std),
        'sample_size': len(npu_output)
    }
    
    return result


if __name__ == '__main__':
    # 示例用法
    np.random.seed(42)
    
    # 测试1: 均匀分布KS检验(通过)
    uniform_data1 = np.random.uniform(0, 1, 1000)
    result1 = ks_test_distribution(uniform_data1, 'uniform', {'low': 0, 'high': 1})
    print(f"测试1 - 均匀分布KS检验: pass={result1['is_pass']}, p_value={result1['p_value']:.4f}")
    
    # 测试2: 正态分布KS检验(通过)
    normal_data2 = np.random.normal(0, 1, 1000)
    result2 = ks_test_distribution(normal_data2, 'normal', {'mean': 0, 'std': 1})
    print(f"测试2 - 正态分布KS检验: pass={result2['is_pass']}, p_value={result2['p_value']:.4f}")
    
    # 测试3: 分布不匹配(不通过)
    uniform_data3 = np.random.uniform(0.5, 1.5, 1000)  # 不是标准均匀分布
    result3 = ks_test_distribution(uniform_data3, 'uniform', {'low': 0, 'high': 1})
    print(f"测试3 - 分布不匹配: pass={result3['is_pass']}, p_value={result3['p_value']:.4f}")
    print(f"  解释: {result3['interpretation']}")
    
    # 测试4: 批量检查
    outputs_list = [np.random.uniform(0, 1, 100) for _ in range(100)]
    summary4 = check_random_distribution_batch(outputs_list, 'uniform', {'low': 0, 'high': 1}, N=100)
    print(f"测试4 - 批量检查: pass={summary4['is_pass']}")
    print(f"  实际通过率: {summary4['actual_pass_rate_percent']:.2f}%")
    print(f"  理论阈值: {summary4['theoretical_pass_rate_percent']:.2f}%")
    
    # 测试5: 统计特性检查
    uniform_data5 = np.random.uniform(0, 1, 10000)
    stats_result = check_distribution_statistics(uniform_data5, 'uniform', {'low': 0, 'high': 1})
    print(f"测试5 - 统计特性:")
    print(f"  实际均值: {stats_result['actual_mean']:.4f}, 期望均值: {stats_result['expected_mean']:.4f}")
    print(f"  实际标准差: {stats_result['actual_std']:.4f}, 期望标准差: {stats_result['expected_std']:.4f}")