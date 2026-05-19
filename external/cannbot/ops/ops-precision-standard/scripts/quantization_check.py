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
量化计算类算子精度检查脚本
用于quantize/dequantize等量化算子的精度验证
"""
import numpy as np


def check_quantization(npu_output, golden_output, third_party_output=None,
                       input_dtype=None, output_dtype=None, precision_level='L0'):
    """
    检查量化计算类算子精度
    
    根据golden标准4.4节,量化算子通过标准由输入输出类型共同决定:
    +----------------+------------------+------------------+
    | 输入类型\\输出类型 | 整型输出         | 浮点输出         |
    +----------------+------------------+------------------+
    | 整型输入       | N/A(不常见场景)  | 参考浮点类标准   |
    | 浮点输入       | 绝对误差 ≤ 1     | 参考浮点类标准   |
    +----------------+------------------+------------------+
    
    Args:
        npu_output: NPU算子输出(numpy array)
        golden_output: CPU标杆输出(numpy array)
        third_party_output: 三方芯片输出(numpy array,可选,浮点输出时需要)
        input_dtype: 输入数据类型(str,可选). 未指定时自动推断
        output_dtype: 输出数据类型(str,可选). 未指定时自动推断
        precision_level: 精度等级(L0/L1/L2),默认L0,仅对浮点输出有效
    
    Returns:
        dict: 包含is_pass和各种误差指标的字典
    """
    # 推断输入输出类型
    is_int_input = input_dtype is not None and 'int' in str(input_dtype).lower()
    is_float_input = input_dtype is not None and ('float' in str(input_dtype).lower() or 'float' in str(input_dtype).lower())
    is_int_output = np.issubdtype(npu_output.dtype, np.integer)
    is_float_output = np.issubdtype(npu_output.dtype, np.floating)
    
    # 如果未指定input_dtype,根据常见量化算子场景推断:
    # 输出为整型时,默认输入为浮点(量化场景 FP→INT)
    # 输出为浮点时,默认输入为整型(反量化场景 INT→FP)
    if input_dtype is None:
        if is_int_output:
            is_float_input = True
            is_int_input = False
        elif is_float_output:
            is_int_input = True
            is_float_input = False
    
    result = {
        'npu_dtype': str(npu_output.dtype),
        'golden_dtype': str(golden_output.dtype),
        'input_dtype': input_dtype,
        'output_dtype': output_dtype,
        'is_int_input': is_int_input,
        'is_float_input': is_float_input,
        'is_int_output': is_int_output,
        'is_float_output': is_float_output,
        'shape': npu_output.shape
    }
    
    # 根据golden标准4.4节判定矩阵
    # 场景1: 整型输入 + 整型输出 → N/A(不常见场景)
    if is_int_input and is_int_output:
        result.update({
            'comparison_method': 'not_applicable',
            'is_pass': False,
            'error': '整型输入+整型输出为N/A场景,量化算子通常不涉及此组合',
            'scenario': 'int_input_int_output'
        })
        return result
    
    # 场景2: 整型输入 + 浮点输出 → 参考浮点类标准(双标杆比对)
    if is_int_input and is_float_output:
        if third_party_output is None:
            result.update({
                'comparison_method': 'dual_benchmark',
                'is_pass': False,
                'error': '整型输入+浮点输出(反量化场景)需要third_party_output进行双标杆比对',
                'scenario': 'int_input_float_output'
            })
            return result
        
        return _check_float_output(npu_output, golden_output, third_party_output,
                                    precision_level, result, 'int_input_float_output')
    
    # 场景3: 浮点输入 + 整型输出 → 绝对误差 ≤ 1(单标杆比对)
    if is_float_input and is_int_output:
        npu_casted = npu_output.astype(np.int64)
        golden_casted = golden_output.astype(np.int64)
        abs_error = np.abs(npu_casted - golden_casted)
        max_abs_error = int(np.max(abs_error))
        mean_abs_error = float(np.mean(abs_error))
        
        is_pass = (max_abs_error <= 1)
        
        result.update({
            'comparison_method': 'single_benchmark',
            'is_pass': is_pass,
            'max_abs_error': max_abs_error,
            'mean_abs_error': mean_abs_error,
            'threshold': 1,
            'scenario': 'float_input_int_output'
        })
        
        if not is_pass:
            mismatch_count = np.sum(abs_error > 1)
            result['mismatch_count'] = mismatch_count
            result['total_elements'] = npu_output.size
            result['mismatch_ratio'] = mismatch_count / npu_output.size
        
        return result
    
    # 场景4: 浮点输入 + 浮点输出 → 参考浮点类标准(双标杆比对)
    if is_float_input and is_float_output:
        if third_party_output is None:
            result.update({
                'comparison_method': 'dual_benchmark',
                'is_pass': False,
                'error': '浮点输入+浮点输出需要third_party_output进行双标杆比对',
                'scenario': 'float_input_float_output'
            })
            return result
        
        return _check_float_output(npu_output, golden_output, third_party_output,
                                    precision_level, result, 'float_input_float_output')
    
    # 兜底: 无法识别的类型组合
    result.update({
        'is_pass': False,
        'error': f'无法识别的类型组合: input={input_dtype}, output={npu_output.dtype}'
    })
    return result


def _check_float_output(npu_output, golden_output, third_party_output,
                         precision_level, result, scenario):
    """
    浮点输出场景的通用比对逻辑(双标杆比对,参考浮点类标准)
    适用于: 整型输入+浮点输出(反量化) 和 浮点输入+浮点输出(量化中间计算)
    """
    # 计算误差指标
    mare_npu = np.max(np.abs(npu_output - golden_output) / (np.abs(golden_output) + 1e-7))
    mere_npu = np.mean(np.abs(npu_output - golden_output) / (np.abs(golden_output) + 1e-7))
    rmse_npu = np.sqrt(np.mean((npu_output - golden_output) ** 2))
    
    mare_third = np.max(np.abs(third_party_output - golden_output) / (np.abs(golden_output) + 1e-7))
    mere_third = np.mean(np.abs(third_party_output - golden_output) / (np.abs(golden_output) + 1e-7))
    rmse_third = np.sqrt(np.mean((third_party_output - golden_output) ** 2))
    
    # 计算Ratio (当三方芯片误差为0时: 若NPU误差也为0则ratio=1, 否则ratio=float('inf'))
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
    
    result.update({
        'comparison_method': 'dual_benchmark',
        'mare_npu': mare_npu,
        'mere_npu': mere_npu,
        'rmse_npu': rmse_npu,
        'mare_third': mare_third,
        'mere_third': mere_third,
        'rmse_third': rmse_third,
        'mare_ratio': mare_ratio,
        'mere_ratio': mere_ratio,
        'rmse_ratio': rmse_ratio,
        'scenario': scenario
    })
    
    # 根据精度等级判断是否通过
    thresholds = {
        'L0': {'mare_ratio': 10, 'mere_ratio': 2, 'rmse_ratio': 2},
        'L1': {'mare_ratio': 5, 'mere_ratio': 1.5, 'rmse_ratio': 1.5},
        'L2': {'mare_ratio': 2, 'mere_ratio': 1.2, 'rmse_ratio': 1.2}
    }
    
    threshold = thresholds.get(precision_level, thresholds['L0'])
    
    mare_pass = mare_ratio <= threshold['mare_ratio']
    mere_pass = mere_ratio <= threshold['mere_ratio']
    rmse_pass = rmse_ratio <= threshold['rmse_ratio']
    
    is_pass = mare_pass and mere_pass and rmse_pass
    
    result.update({
        'precision_level': precision_level,
        'is_pass': is_pass,
        'mare_pass': mare_pass,
        'mere_pass': mere_pass,
        'rmse_pass': rmse_pass,
        'thresholds_used': threshold
    })
    
    return result


def check_quantization_with_level(npu_output, golden_output, third_party_output=None,
                                   precision_level='L0', input_dtype=None, output_dtype=None):
    """
    检查量化算子精度并根据精度等级判断是否通过
    
    Args:
        precision_level: 精度等级(L0/L1/L2),默认L0
    
    Returns:
        dict: 包含is_pass和完整判定信息的字典
    """
    result = check_quantization(npu_output, golden_output, third_party_output,
                                input_dtype, output_dtype, precision_level)
    
    return result


if __name__ == '__main__':
    # 示例用法
    np.random.seed(42)
    
    # 测试1: 浮点输入+整型输出(量化结果, 绝对误差≤1)
    golden_int = np.array([10, 20, 30, 40], dtype=np.int8)
    npu_int = np.array([10, 20, 31, 40], dtype=np.int8)  # 误差=1
    result1 = check_quantization(npu_int, golden_int, input_dtype='float16', output_dtype='int8')
    print(f"测试1 - 浮点输入+整型输出(量化): pass={result1['is_pass']}, max_abs_error={result1['max_abs_error']}, scenario={result1['scenario']}")
    
    # 测试2: 整型输入+浮点输出(反量化结果, 双标杆比对)
    golden_float = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float16)
    npu_float = np.array([1.501, 2.501, 3.502, 4.501], dtype=np.float16)
    third_float = np.array([1.502, 2.502, 3.503, 4.502], dtype=np.float16)
    
    result2 = check_quantization_with_level(
        npu_float, golden_float, third_float,
        precision_level='L1',
        input_dtype='int8',
        output_dtype='float16'
    )
    print(f"测试2 - 整型输入+浮点输出(反量化,L1): pass={result2['is_pass']}, "
          f"mare_ratio={result2['mare_ratio']:.3f}, mere_ratio={result2['mere_ratio']:.3f}, scenario={result2['scenario']}")
    
    # 测试3: 浮点输出缺少三方标杆
    result3 = check_quantization(npu_float, golden_float)
    print(f"测试3 - 缺少三方标杆: pass={result3['is_pass']}, error={result3.get('error', 'None')}")
    
    # 测试4: 整型输入+整型输出(N/A场景)
    golden_int4 = np.array([10, 20, 30], dtype=np.int8)
    npu_int4 = np.array([10, 20, 30], dtype=np.int8)
    result4 = check_quantization(npu_int4, golden_int4, input_dtype='int8', output_dtype='int8')
    print(f"测试4 - 整型输入+整型输出(N/A): pass={result4['is_pass']}, error={result4.get('error', 'None')}, scenario={result4.get('scenario', 'N/A')}")
    
    # 测试5: 浮点输入+浮点输出(双标杆比对)
    golden_float5 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    npu_float5 = np.array([1.001, 2.001, 3.001], dtype=np.float16)
    third_float5 = np.array([1.002, 2.002, 3.002], dtype=np.float16)
    result5 = check_quantization(npu_float5, golden_float5, third_float5,
                                  input_dtype='float16', output_dtype='float16', precision_level='L0')
    print(f"测试5 - 浮点输入+浮点输出(L0): pass={result5['is_pass']}, scenario={result5['scenario']}")