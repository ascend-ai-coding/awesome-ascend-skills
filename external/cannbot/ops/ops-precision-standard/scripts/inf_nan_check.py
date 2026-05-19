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
INF/NAN值通过标准检查脚本
处理inf/-inf/nan等特殊值的精度验证

根据Golden标准(5.3.3节):
- is_before_910a=True: inf不参与精度比较,仅检查nan一致性(对应Ascend910A及之前芯片,或910B+且INF_NAN_MODE_ENABLE=0)
- is_before_910a=False: strict模式,要求inf输出一致,NPU与Golden和标杆三者综合判定(对应Ascend910B+且INF_NAN_MODE_ENABLE=1)
"""
import numpy as np


def check_inf_nan_consistency(npu_output, golden_output, benchmark_output=None, is_before_910a=True):
    """
    检查INF/NAN值的通过标准

    根据Golden标准(5.3.3节)的判定规则:

    is_before_910a=True(Ascend910A及之前,或910B+且INF_NAN_MODE_ENABLE=0):
    - inf不参与精度比较,仅检查nan是否一致

    is_before_910a=False(Ascend910B+且INF_NAN_MODE_ENABLE=1):
    1. 正确条件(满足任一即可):
       - 不论Golden为何值: NPU值与标杆值一致(nan/inf/-inf) → 通过
       - 若Golden是inf/-inf/nan: NPU值与Golden完全一致 → 通过
    2. 计算错误条件:
       - 若Golden是inf/-inf/nan: NPU值与Golden不一致且标杆值与Golden一致 → 不通过（标杆正确但NPU错误）
    3. 需异常排查:
       - Golden、NPU结果、标杆值三者都不一致 → 需进一步验证

    Args:
        npu_output: NPU算子输出(numpy array)
        golden_output: CPU标杆输出(numpy array)
        benchmark_output: 三方芯片标杆输出(numpy array,可选)
        is_before_910a: True表示Ascend910A及之前芯片(或910B+且INF_NAN_MODE_ENABLE=0),
                       inf不参与比较仅检查nan; False表示Ascend910B+且INF_NAN_MODE_ENABLE=1,
                       启用strict模式要求inf输出一致

    Returns:
        dict: 包含is_pass和详细判定信息的字典
    """
    npu_inf = np.isinf(npu_output)
    npu_nan = np.isnan(npu_output)
    golden_inf = np.isinf(golden_output)
    golden_nan = np.isnan(golden_output)

    has_special_values = np.any(npu_inf | npu_nan | golden_inf | golden_nan)

    if not has_special_values:
        return {
            'has_special_values': False,
            'is_pass': None,
            'note': 'No inf/nan values, use normal precision standard'
        }

    result = {
        'has_special_values': True,
        'is_before_910a': is_before_910a,
        'npu_inf_count': int(np.sum(npu_inf)),
        'npu_nan_count': int(np.sum(npu_nan)),
        'golden_inf_count': int(np.sum(golden_inf)),
        'golden_nan_count': int(np.sum(golden_nan)),
        'total_elements': int(npu_output.size)
    }

    special_indices = np.where(npu_inf | npu_nan | golden_inf | golden_nan)

    if len(special_indices[0]) > 0:
        sample_indices = special_indices[0][:10]
        result['sample_special_values'] = {
            'indices': sample_indices.tolist(),
            'npu_values': npu_output[sample_indices].tolist(),
            'golden_values': golden_output[sample_indices].tolist()
        }

    pass_count = 0
    fail_count = 0
    need_check_count = 0
    skipped_count = 0

    for i in range(npu_output.size):
        golden_val = golden_output.flat[i]
        npu_val = npu_output.flat[i]

        is_golden_special = np.isinf(golden_val) or np.isnan(golden_val)
        is_npu_special = np.isinf(npu_val) or np.isnan(npu_val)

        if not is_golden_special and not is_npu_special:
            continue

        if is_before_910a:
            if np.isnan(golden_val) and np.isnan(npu_val):
                pass_count += 1
            elif np.isnan(golden_val) or np.isnan(npu_val):
                fail_count += 1
            else:
                skipped_count += 1
            continue

        benchmark_val = benchmark_output.flat[i] if benchmark_output is not None else None
        is_benchmark_special = (np.isinf(benchmark_val) or np.isnan(benchmark_val)) if benchmark_val is not None else False

        npu_matches_golden = (npu_val == golden_val) or (np.isnan(npu_val) and np.isnan(golden_val))
        npu_matches_benchmark = (benchmark_val is not None) and ((npu_val == benchmark_val) or (np.isnan(npu_val) and np.isnan(benchmark_val)))
        benchmark_matches_golden = (benchmark_val is not None) and ((benchmark_val == golden_val) or (np.isnan(benchmark_val) and np.isnan(golden_val)))

        if is_golden_special:
            if npu_matches_golden:
                pass_count += 1
                continue

            if benchmark_matches_golden:
                fail_count += 1
                continue

            if benchmark_val is not None:
                need_check_count += 1
            else:
                fail_count += 1
            continue

        if is_npu_special:
            if npu_matches_benchmark:
                pass_count += 1
            elif benchmark_val is not None and is_benchmark_special:
                need_check_count += 1
            else:
                fail_count += 1
        continue

    total_special = pass_count + fail_count + need_check_count + skipped_count

    result.update({
        'special_value_count': total_special,
        'pass_count': pass_count,
        'fail_count': fail_count,
        'need_check_count': need_check_count,
        'skipped_count': skipped_count,
        'has_benchmark': benchmark_output is not None
    })

    if total_special == 0:
        result['is_pass'] = True
    else:
        result['is_pass'] = (fail_count == 0 and need_check_count == 0)

        if not result['is_pass']:
            result['failure_reasons'] = []
            if fail_count > 0:
                result['failure_reasons'].append(f'{fail_count} special values mismatch with golden')
            if need_check_count > 0:
                result['failure_reasons'].append(f'{need_check_count} special values need abnormal check')

    return result


if __name__ == '__main__':
    np.random.seed(42)

    golden1 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    npu1 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    result1 = check_inf_nan_consistency(npu1, golden1)
    print(f"测试1 - 无特殊值: has_special={result1['has_special_values']}")

    golden2 = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    npu2 = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    result2 = check_inf_nan_consistency(npu2, golden2, is_before_910a=True)
    print(f"测试2 - 910A模式NPU与Golden一致(nan): pass={result2['is_pass']}, "
          f"pass_count={result2['pass_count']}, fail_count={result2['fail_count']}")

    golden3 = np.array([np.inf, 2.0], dtype=np.float32)
    npu3 = np.array([1.0, 2.0], dtype=np.float32)
    result3 = check_inf_nan_consistency(npu3, golden3, is_before_910a=True)
    print(f"测试3 - 910A模式inf不参与比较: pass={result3['is_pass']}, "
          f"skipped_count={result3['skipped_count']}")

    golden4 = np.array([np.inf, -np.inf, np.nan, 1.0, 2.0], dtype=np.float32)
    npu4 = np.array([np.inf, -np.inf, np.nan, 1.0, 2.0], dtype=np.float32)
    result4 = check_inf_nan_consistency(npu4, golden4, is_before_910a=False)
    print(f"测试4 - strict模式混合特殊值且一致: pass={result4['is_pass']}")

    golden5 = np.array([np.inf, 2.0], dtype=np.float32)
    npu5 = np.array([1.0, 2.0], dtype=np.float32)
    benchmark5 = np.array([np.inf, 2.0], dtype=np.float32)
    result5 = check_inf_nan_consistency(npu5, golden5, benchmark5, is_before_910a=False)
    print(f"测试5 - strict模式NPU不一致且标杆与Golden一致: pass={result5['is_pass']}, "
          f"fail_count={result5['fail_count']}")

    golden6 = np.array([np.inf, 2.0], dtype=np.float32)
    npu6 = np.array([np.inf, 2.0], dtype=np.float32)
    benchmark6 = np.array([1.0, 2.0], dtype=np.float32)
    result6 = check_inf_nan_consistency(npu6, golden6, benchmark6, is_before_910a=False)
    print(f"测试6 - strict模式NPU与Golden一致(标杆不一致): pass={result6['is_pass']}, "
          f"pass_count={result6['pass_count']}")

    golden7 = np.array([np.inf, 2.0], dtype=np.float32)
    npu7 = np.array([-np.inf, 2.0], dtype=np.float32)
    benchmark7 = np.array([1.0, 2.0], dtype=np.float32)
    result7 = check_inf_nan_consistency(npu7, golden7, benchmark7, is_before_910a=False)
    print(f"测试7 - strict模式三者都不一致: pass={result7['is_pass']}, "
          f"need_check_count={result7['need_check_count']}")

    golden8 = np.array([1.0, 2.0], dtype=np.float32)
    npu8 = np.array([np.inf, 2.0], dtype=np.float32)
    benchmark8 = np.array([1.0, 2.0], dtype=np.float32)
    result8 = check_inf_nan_consistency(npu8, golden8, benchmark8, is_before_910a=False)
    print(f"测试8 - strict模式NPU异常输出inf: pass={result8['is_pass']}, "
          f"fail_count={result8['fail_count']}")

    golden9 = np.array([1.0, 2.0], dtype=np.float32)
    npu9 = np.array([np.inf, 2.0], dtype=np.float32)
    benchmark9 = np.array([np.inf, 2.0], dtype=np.float32)
    result9 = check_inf_nan_consistency(npu9, golden9, benchmark9, is_before_910a=False)
    print(f"测试9 - strict模式Golden非特殊值但NPU与标杆一致(通过): pass={result9['is_pass']}, "
          f"pass_count={result9['pass_count']}")
