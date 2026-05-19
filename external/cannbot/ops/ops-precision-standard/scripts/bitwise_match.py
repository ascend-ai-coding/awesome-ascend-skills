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
二进制一致性比对脚本(Bitwise Match)
用于非计算类算子(搬移/Cast等)的精度验证
"""
import numpy as np


def check_bitwise_match(npu_output, golden_output):
    """
    检查NPU输出与Golden是否二进制一致
    
    Args:
        npu_output: NPU算子输出(numpy array)
        golden_output: CPU标杆输出(numpy array)
    
    Returns:
        dict: 包含is_pass, bitwise_match_result等信息的字典
    """
    is_bitwise_match = np.array_equal(npu_output, golden_output)
    
    result = {
        'is_pass': is_bitwise_match,
        'bitwise_match': is_bitwise_match,
        'npu_dtype': str(npu_output.dtype),
        'golden_dtype': str(golden_output.dtype),
        'npu_shape': npu_output.shape,
        'golden_shape': golden_output.shape
    }
    
    if not is_bitwise_match:
        mismatch_count = np.sum(npu_output != golden_output)
        result['mismatch_count'] = mismatch_count
        result['total_elements'] = npu_output.size
        result['mismatch_ratio'] = mismatch_count / npu_output.size
        
    return result


def check_bitwise_match_batch(outputs_list):
    """
    批量检查多个用例的二进制一致性
    
    Args:
        outputs_list: [(npu_output, golden_output), ...]列表
    
    Returns:
        dict: 包含汇总信息的字典
    """
    results = []
    pass_count = 0
    
    for npu_out, golden_out in outputs_list:
        result = check_bitwise_match(npu_out, golden_out)
        results.append(result)
        if result['is_pass']:
            pass_count += 1
    
    total = len(outputs_list)
    summary = {
        'total_cases': total,
        'pass_count': pass_count,
        'fail_count': total - pass_count,
        'pass_rate': pass_count / total if total > 0 else 0,
        'detail_results': results
    }
    
    return summary


if __name__ == '__main__':
    # 示例用法
    np.random.seed(42)
    
    # 测试1: 完全一致的情况
    golden1 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    npu1 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    result1 = check_bitwise_match(npu1, golden1)
    print(f"测试1 - 完全一致: {result1['is_pass']}")
    
    # 测试2: 不一致的情况
    golden2 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    npu2 = np.array([1.0, 2.0, 3.001], dtype=np.float16)
    result2 = check_bitwise_match(npu2, golden2)
    print(f"测试2 - 不一致: {result2['is_pass']}, 不匹配数量: {result2.get('mismatch_count', 0)}")
    
    # 测试3: 批量检查
    outputs_list = [
        (np.array([1, 2, 3], dtype=np.int32), np.array([1, 2, 3], dtype=np.int32)),
        (np.array([1, 2, 3], dtype=np.int32), np.array([1, 2, 4], dtype=np.int32)),
        (np.array([5, 6, 7], dtype=np.int32), np.array([5, 6, 7], dtype=np.int32))
    ]
    summary = check_bitwise_match_batch(outputs_list)
    print(f"批量测试: 通过率 {summary['pass_rate']:.2%}")