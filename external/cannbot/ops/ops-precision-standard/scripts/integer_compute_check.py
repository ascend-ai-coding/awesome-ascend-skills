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
整数计算类算子精度检查脚本
用于INT8/INT16/INT32等整数算子的精度验证
"""
import numpy as np


def check_integer_compute(npu_output, golden_output):
    """
    检查整数计算类算子精度
    
    通过标准:
    1. 二进制一致: np.array_equal(npu_output, golden_output)
    2. 若不满足二进制一致但绝对误差为0也视为通过
    
    Args:
        npu_output: NPU算子输出(numpy array, int类型)
        golden_output: CPU标杆输出(numpy array, int类型)
    
    Returns:
        dict: 包含is_pass, bitwise_match, abs_error等信息的字典
    """
    # 检查数据类型
    if not (np.issubdtype(npu_output.dtype, np.integer) and 
            np.issubdtype(golden_output.dtype, np.integer)):
        return {
            'is_pass': False,
            'error': 'Output dtype must be integer type',
            'npu_dtype': str(npu_output.dtype),
            'golden_dtype': str(golden_output.dtype)
        }
    
    # 检查二进制一致
    is_bitwise_match = np.array_equal(npu_output, golden_output)
    
    # 检查绝对误差: 先统一转为int64避免不同位宽整数相减溢出
    npu_casted = npu_output.astype(np.int64)
    golden_casted = golden_output.astype(np.int64)
    abs_error = np.abs(npu_casted - golden_casted)
    max_abs_error = int(np.max(abs_error))
    mean_abs_error = float(np.mean(abs_error))
    
    # 通过条件: 二进制一致 OR 绝对误差为0
    is_abs_zero = (max_abs_error == 0)
    is_pass = is_bitwise_match or is_abs_zero
    
    result = {
        'is_pass': is_pass,
        'bitwise_match': is_bitwise_match,
        'abs_error_zero': is_abs_zero,
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'npu_dtype': str(npu_output.dtype),
        'golden_dtype': str(golden_output.dtype),
        'shape': npu_output.shape
    }
    
    if not is_pass:
        mismatch_count = np.sum(abs_error != 0)
        result['mismatch_count'] = mismatch_count
        result['total_elements'] = npu_output.size
        result['mismatch_ratio'] = mismatch_count / npu_output.size
        
        # 提供不匹配的样本位置和值
        mismatch_indices = np.where(abs_error != 0)
        if len(mismatch_indices[0]) > 0:
            sample_indices = mismatch_indices[0][:10]  # 只展示前10个
            result['sample_mismatch'] = {
                'indices': sample_indices,
                'npu_values': npu_output[sample_indices],
                'golden_values': golden_output[sample_indices],
                'errors': abs_error[sample_indices]
            }
    
    return result


def check_integer_compute_batch(outputs_list):
    """
    批量检查多个用例的整数计算精度
    
    Args:
        outputs_list: [(npu_output, golden_output), ...]列表
    
    Returns:
        dict: 包含汇总信息的字典
    """
    results = []
    pass_count = 0
    bitwise_match_count = 0
    abs_zero_count = 0
    
    for npu_out, golden_out in outputs_list:
        result = check_integer_compute(npu_out, golden_out)
        results.append(result)
        
        if result['is_pass']:
            pass_count += 1
            if result['bitwise_match']:
                bitwise_match_count += 1
            elif result['abs_error_zero']:
                abs_zero_count += 1
    
    total = len(outputs_list)
    summary = {
        'total_cases': total,
        'pass_count': pass_count,
        'fail_count': total - pass_count,
        'pass_rate': pass_count / total if total > 0 else 0,
        'bitwise_match_count': bitwise_match_count,
        'abs_zero_count': abs_zero_count,
        'detail_results': results
    }
    
    return summary


if __name__ == '__main__':
    # 示例用法
    np.random.seed(42)
    
    # 测试1: 完全二进制一致
    golden1 = np.array([10, 20, 30, 40], dtype=np.int32)
    npu1 = np.array([10, 20, 30, 40], dtype=np.int32)
    result1 = check_integer_compute(npu1, golden1)
    print(f"测试1 - 二进制一致: {result1['is_pass']}, bitwise_match={result1['bitwise_match']}")
    
    # 测试2: 绝对误差为0的情况(可能dtype不同但数值相同)
    golden2 = np.array([100, 200, 300], dtype=np.int64)
    npu2 = np.array([100, 200, 300], dtype=np.int64)
    result2 = check_integer_compute(npu2, golden2)
    print(f"测试2 - 绝对误差为0: {result2['is_pass']}, abs_error_zero={result2['abs_error_zero']}")
    
    # 测试3: 不通过的情况
    golden3 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    npu3 = np.array([1, 2, 4, 4, 5], dtype=np.int32)
    result3 = check_integer_compute(npu3, golden3)
    print(f"测试3 - 不通过: {result3['is_pass']}, max_abs_error={result3['max_abs_error']}")
    if 'sample_mismatch' in result3:
        print(f"  不匹配样本: indices={result3['sample_mismatch']['indices']}, "
              f"errors={result3['sample_mismatch']['errors']}")
    
    # 测试4: 非整数类型(应该报错)
    golden4 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    npu4 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result4 = check_integer_compute(npu4, golden4)
    print(f"测试4 - 非整数类型: {result4['is_pass']}, error={result4.get('error', 'None')}")