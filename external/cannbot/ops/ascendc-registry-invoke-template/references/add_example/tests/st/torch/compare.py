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
精度比对模块

对标 CANN 算子精度验收标准：
- 浮点计算类：MERE/MARE Threshold 方法
- 整数计算类：精确匹配

适配指南：
  根据算子类型选择验证方案，修改本模块：
  - 量化类算子 → 替换为量化精度比对逻辑
  - 随机数生成类算子 → 替换为分布统计检验逻辑
  如算子采用商用标准，请修改 FLOAT_THRESHOLDS 阈值。
"""

import torch


# ============================================================================
# 浮点精度阈值
#
# 通过标准：MERE < threshold 且 MARE < 10 * threshold
# 阈值跟随 CANN 社区标准，如标准更新或采用商用标准请同步修改。
# ============================================================================

FLOAT_THRESHOLDS = {
    torch.float16: 2 ** (-10),       # ~0.000977
    torch.bfloat16: 2 ** (-7),       # ~0.00781
    torch.float32: 2 ** (-13),       # ~0.000122
    torch.float64: 2 ** (-13),       # ~0.000122
}


def _get_threshold(dtype: torch.dtype) -> float:
    """根据 dtype 获取 MERE 阈值"""
    return FLOAT_THRESHOLDS.get(dtype, 2 ** (-13))


# ============================================================================
# 浮点比对：MERE/MARE Threshold
# ============================================================================

def _compare_float(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """浮点精度比对，按 MERE/MARE 阈值判定"""
    golden_f = golden.flatten().float()
    actual_f = actual.flatten().float()

    eps = torch.finfo(golden.dtype).tiny
    relative_errors = (golden_f - actual_f).abs() / (golden_f.abs() + eps)
    mere = relative_errors.mean().item()
    mare = relative_errors.max().item()

    threshold = _get_threshold(golden.dtype)
    mare_threshold = 10 * threshold

    mere_pass = mere < threshold
    mare_pass = mare < mare_threshold
    passed = mere_pass and mare_pass

    if not passed:
        print(f"    MERE: {mere:.6e} (阈值: {threshold:.6e}) {'PASS' if mere_pass else 'FAIL'}")
        print(f"    MARE: {mare:.6e} (阈值: {mare_threshold:.6e}) {'PASS' if mare_pass else 'FAIL'}")
        # 打印最大差异位置
        diff = (golden_f - actual_f).abs()
        max_idx = diff.argmax().item()
        print(f"    最大差异位置: [{max_idx}] Golden={golden_f[max_idx].item():.6f}, "
              f"Actual={actual_f[max_idx].item():.6f}")

    return passed


# ============================================================================
# 整型比对：精确匹配
# ============================================================================

def _compare_integer(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """整数精度比对（二进制一致或绝对误差为 0）"""
    passed = torch.equal(golden, actual)
    if not passed:
        mask = golden != actual
        mismatch = mask.sum().item()
        first_idx = mask.flatten().nonzero()[0].item()
        print(f"    不匹配元素数: {mismatch} (首个位置: {first_idx})")
        print(f"    Golden[{first_idx}] = {golden.flatten()[first_idx].item()}")
        print(f"    Actual[{first_idx}] = {actual.flatten()[first_idx].item()}")
    return passed


# ============================================================================
# 统一入口
# ============================================================================

def compare_results(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """比对 golden 与实际结果，失败时打印差异详情"""
    if golden.dtype.is_floating_point:
        return _compare_float(golden, actual)
    else:
        return _compare_integer(golden, actual)
