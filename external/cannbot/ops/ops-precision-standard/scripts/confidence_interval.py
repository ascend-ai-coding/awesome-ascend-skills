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
置信区间计算脚本(复检工具)
用于精度复检流程中的统计分析
"""
import math
import numpy as np
from scipy import stats


def calculate_confidence_interval(data, confidence_level=0.95, method='bootstrap'):
    """
    计算置信区间
    
    Args:
        data: 样本数据(numpy array)
        confidence_level: 置信水平(默认0.95)
        method: 计算方法('bootstrap'或'parametric')
    
    Returns:
        dict: 包含置信区间信息的字典
    """
    n = len(data)
    
    if n < 200:
        return {
            'is_valid': False,
            'error': '小样本熔断: N < 200',
            'sample_size': n
        }
    
    if method == 'bootstrap':
        # Golden标准(5.3.4节791行)推荐1000次Bootstrap重采样
        # 498行示例使用N=1000: CI_Lower=第25小(索引24), CI_Upper=第976小(索引975)
        n_bootstrap = 1000
        boot_medians = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            boot_medians.append(np.median(sample))
        
        boot_medians = np.array(boot_medians)
        boot_medians_sorted = np.sort(boot_medians)
        
        alpha = 1 - confidence_level
        # 根据golden标准5.3.4节: CI_Lower=第2.5%分位数, CI_Upper=第97.5%分位数
        # 对于N=1000次Bootstrap: CI_Lower=第25小(索引24), CI_Upper=第976小(索引975)
        # 使用math.ceil确保与golden标准的"第N小"定义一致(1-indexed转0-indexed)
        lower_idx = max(0, math.ceil(alpha / 2 * n_bootstrap) - 1)
        upper_idx = min(n_bootstrap - 1, math.ceil((1 - alpha / 2) * n_bootstrap) - 1)
        
        ci_lower = boot_medians_sorted[lower_idx]
        ci_upper = boot_medians_sorted[upper_idx]
        
        return {
            'is_valid': True,
            'method': 'bootstrap',
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'median': np.median(data),
            'confidence_level': confidence_level,
            'n_bootstrap': n_bootstrap,
            'sample_size': n
        }
    
    elif method == 'parametric':
        # 参数方法(假设正态分布)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        # 计算t分布临界值
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, n - 1)
        
        margin = t_critical * std / np.sqrt(n)
        
        ci_lower = mean - margin
        ci_upper = mean + margin
        
        return {
            'is_valid': True,
            'method': 'parametric',
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean': mean,
            'std': std,
            'confidence_level': confidence_level,
            'sample_size': n
        }
    
    else:
        return {
            'is_valid': False,
            'error': f'Unknown method: {method}'
        }


def check_systematic_error(ratio_data, threshold=1.0, confidence_level=0.95):
    """
    检查是否存在系统性精度恶化
    
    通过标准: 置信区间下限 CI_Lower <= threshold
    
    Args:
        ratio_data: 误差比值样本(numpy array)
        threshold: 判定阈值(默认1.0)
        confidence_level: 置信水平
    
    Returns:
        dict: 包含is_pass和统计信息的字典
    """
    ci_result = calculate_confidence_interval(ratio_data, confidence_level, method='bootstrap')
    
    if not ci_result['is_valid']:
        return {
            'is_pass': False,
            'error': ci_result['error'],
            'sample_size': ci_result['sample_size']
        }
    
    ci_lower = ci_result['ci_lower']
    median = ci_result['median']
    
    # 判定规则
    is_pass = ci_lower <= threshold
    
    result = {
        'is_pass': is_pass,
        'ci_lower': ci_lower,
        'ci_upper': ci_result['ci_upper'],
        'median': median,
        'threshold': threshold,
        'confidence_level': confidence_level,
        'sample_size': ci_result['sample_size'],
        'interpretation': ''
    }
    
    if is_pass:
        result['interpretation'] = f'CI_Lower {ci_lower:.3f} <= {threshold},无统计学证据表明NPU存在系统性精度恶化'
    else:
        result['interpretation'] = f'CI_Lower {ci_lower:.3f} > {threshold},有统计学证据表明NPU存在系统性精度恶化'
    
    return result


def analyze_recheck_ratios(ratio_samples, confidence_level=0.95):
    """
    分析精度复检中预先计算好的误差比值样本,进行Bootstrap统计推断。

    注意: 本函数仅负责统计分析,不负责:
    - 更换随机种子重新生成输入
    - 执行算子获取输出
    - 计算误差和Ratio

    调用方需先完成上述步骤,将计算好的Ratio数组传入本函数。

    Args:
        ratio_samples: N次误差比值样本(np.array,长度N)
                      即每次更换随机种子后计算的 Ratio = npu_error / golden_error
        confidence_level: 置信水平

    Returns:
        dict: 包含完整复检结果的字典
    """
    ratios = np.asarray(ratio_samples, dtype=np.float64)

    if len(ratios) < 200:
        return {
            'recheck_pass': False,
            'error': f'小样本熔断: N={len(ratios)} < 200',
            'need_more_samples': True,
            'sample_size': len(ratios)
        }

    ci_result = calculate_confidence_interval(ratios, confidence_level)

    if not ci_result['is_valid']:
        return {
            'recheck_pass': False,
            'error': ci_result['error'],
            'need_more_samples': True,
            'sample_size': ci_result['sample_size']
        }

    check_result = check_systematic_error(ratios, threshold=1.0, confidence_level=confidence_level)

    result = {
        'recheck_pass': check_result['is_pass'],
        'ci_lower': check_result['ci_lower'],
        'ci_upper': check_result['ci_upper'],
        'median_ratio': check_result['median'],
        'mean_ratio': np.mean(ratios),
        'std_ratio': np.std(ratios),
        'interpretation': check_result['interpretation'],
        'sample_size': len(ratios),
        'confidence_level': confidence_level,
        'need_more_samples': False
    }

    return result


if __name__ == '__main__':
    # 示例用法
    np.random.seed(42)
    
    # 测试1: Bootstrap方法,通过
    data1 = np.random.normal(0.9, 0.15, 1000)
    ci1 = calculate_confidence_interval(data1, confidence_level=0.95, method='bootstrap')
    print(f"测试1 - Bootstrap: CI=[{ci1['ci_lower']:.3f}, {ci1['ci_upper']:.3f}], median={ci1['median']:.3f}")
    
    check1 = check_systematic_error(data1, threshold=1.0)
    print(f"  系统性误差检查: pass={check1['is_pass']}")
    print(f"  解释: {check1['interpretation']}")
    
    # 测试2: 参数方法
    data2 = np.random.normal(1.0, 0.2, 500)
    ci2 = calculate_confidence_interval(data2, method='parametric')
    print(f"测试2 - Parametric: CI=[{ci2['ci_lower']:.3f}, {ci2['ci_upper']:.3f}], mean={ci2['mean']:.3f}")
    
    # 测试3: 小样本熔断
    data3 = np.random.normal(1.0, 0.1, 150)
    ci3 = calculate_confidence_interval(data3)
    print(f"测试3 - 小样本: is_valid={ci3['is_valid']}, error={ci3.get('error', 'None')}")
    
    # 测试4: 精度复检流程（传入预先计算好的Ratio样本）
    npu_errors = np.random.normal(0.01, 0.002, 1000)  # NPU误差
    benchmark_errors = np.random.normal(0.008, 0.001, 1000)  # 标杆误差
    # 计算Ratio并处理除零情况
    safe_benchmark = np.where(benchmark_errors > 0, benchmark_errors, 1e-10)
    ratios = np.where(npu_errors > 0, npu_errors / safe_benchmark, 1.0)
    
    recheck_result = analyze_recheck_ratios(ratios)
    print(f"测试4 - 复检流程: pass={recheck_result['recheck_pass']}")
    print(f"  median_ratio={recheck_result['median_ratio']:.3f}")
    print(f"  CI=[{recheck_result['ci_lower']:.3f}, {recheck_result['ci_upper']:.3f}]")