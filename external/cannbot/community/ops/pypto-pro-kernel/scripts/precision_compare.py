#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""在 CPU 上按实际输出 dtype 校验算子精度。

FP16、BF16、FP32、FP64 检查匹配率、最大绝对误差以及平均绝对和相对误差。
FP8 E4M3FN、E5M2 按 CANN opbase 混合容差标准检查匹配率和最大有限误差。
整数及布尔输出精确比较。FP4 契约要求编码一致时，传入实际与 golden 的有效字节。
"""

import math
from typing import List, Optional, Tuple, Union

import torch

_REQUIRED_MATCHED_RATIO = 0.99

# (相对容差，绝对容差，最大绝对误差上限的兜底值)
_THRESHOLDS = {
    torch.float16: (2.0**-9, 2.0**-9, 1e-1),
    torch.bfloat16: (2.0**-6, 2.0**-6, 1.0),
    torch.float32: (2.0**-10, 2.0**-16, 1e-2),
    torch.float64: (2.0**-10, 2.0**-16, 1e-2),
}

_DTYPE_NAME_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "half": torch.float16,
    "float": torch.float32,
    "double": torch.float64,
}

# PyTorch 版本可能未提供某一格式；其余 dtype 仍可独立使用。
_FP8_TOLERANCE = {
    getattr(torch, name): tolerance
    for name, tolerance in (("float8_e4m3fn", 2.0**-3), ("float8_e5m2", 2.0**-2))
    if hasattr(torch, name)
}
_DTYPE_NAME_MAP.update({str(dtype).removeprefix("torch."): dtype for dtype in _FP8_TOLERANCE})


def _get_threshold(dtype):
    """常规浮点类型的 ULP 取该 dtype 在 1.0 处的间距。"""
    if dtype not in _THRESHOLDS:
        return None
    rtol, atol, fixed_limit = _THRESHOLDS[dtype]
    return rtol, atol, _REQUIRED_MATCHED_RATIO, max(fixed_limit, 32 * torch.finfo(dtype).eps)


def _compare_integer(actual, golden, total):
    if torch.equal(actual, golden):
        return True, "exact match (integer)"
    mismatch = (actual != golden).sum().item()
    return False, f"integer mismatch: {mismatch}/{total} elements differ"


def _prepare_float_tensors(actual, golden, comparison_dtype):
    actual_float = actual.to(torch.float64)
    golden_float = golden.to(torch.float64)
    actual_nan = torch.isnan(actual_float)
    golden_nan = torch.isnan(golden_float)
    nan_mismatch = (actual_nan != golden_nan).sum().item()
    if nan_mismatch:
        return None, f"NaN position mismatch: {nan_mismatch} positions differ"

    actual_inf = torch.isinf(actual_float)
    golden_inf = torch.isinf(golden_float)
    both_inf = actual_inf & golden_inf
    inf_diff_sign = both_inf & (torch.sign(actual_float) != torch.sign(golden_float))
    if inf_diff_sign.any().item():
        return None, f"Inf sign mismatch: {inf_diff_sign.sum().item()} positions"

    actual_clean = actual_float.clone()
    golden_clean = golden_float.clone()
    finfo = torch.finfo(comparison_dtype)
    actual_only_inf = actual_inf & ~golden_inf
    golden_only_inf = golden_inf & ~actual_inf
    actual_clean[actual_only_inf & (actual_float > 0)] = finfo.max
    actual_clean[actual_only_inf & (actual_float < 0)] = finfo.min
    golden_clean[golden_only_inf & (golden_float > 0)] = finfo.max
    golden_clean[golden_only_inf & (golden_float < 0)] = finfo.min
    same_inf = both_inf & (torch.sign(actual_float) == torch.sign(golden_float))
    match_mask = same_inf | actual_nan | golden_nan
    return (actual_clean, golden_clean, match_mask), None


def _calculate_float_metrics(prepared, threshold, total):
    actual, golden, match_mask = prepared
    rtol, atol, required_ratio, max_abs_limit = threshold
    compare_mask = ~match_mask
    if not compare_mask.any().item():
        matched_ratio = 1.0
        max_abs_error = 0.0
        mean_abs_error = 0.0
        mean_rel_error = 0.0
    else:
        diff = (actual - golden).abs()
        element_pass = diff <= atol + rtol * golden.abs()
        pass_count = (element_pass & compare_mask).sum().item() + match_mask.sum().item()
        matched_ratio = pass_count / total
        max_abs_error = diff[compare_mask].max().item()
        mean_abs_error = diff[compare_mask].mean().item()
        golden_abs = golden.abs()[compare_mask]
        rel_valid = golden_abs > 0
        if rel_valid.any().item():
            mean_rel_error = (diff[compare_mask][rel_valid] / golden_abs[rel_valid]).mean().item()
        else:
            mean_rel_error = 0.0
    return {
        "matched_ratio": matched_ratio,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "mean_rel_error": mean_rel_error,
        "required_matched_ratio": required_ratio,
        "max_abs_error_limit": max_abs_limit,
        "rtol": rtol,
        "atol": atol,
    }


def _format_float_result(metrics):
    checks = [
        ("atol", metrics["mean_abs_error"], metrics["atol"], "GT"),
        ("rtol", metrics["mean_rel_error"], metrics["rtol"], "GT"),
        ("matched_ratio", metrics["matched_ratio"], metrics["required_matched_ratio"], "LT"),
        ("max_abs_error", metrics["max_abs_error"], metrics["max_abs_error_limit"], "GT"),
    ]
    failed = {
        name
        for name, value, thresh, op in checks
        if (value > thresh if op == "GT" else value < thresh)
    }
    passed = not failed
    lines = [f"precision {'PASS' if passed else 'FAIL'}"]
    for name, value, thresh, op in checks:
        if name == "matched_ratio":
            val_s, thr_s = f"{value:.4f}", f"{thresh:.4f}"
        else:
            val_s, thr_s = f"{value:.3e}", f"{thresh:.3e}"
        lines.append(
            f"    {name:<16}: {val_s:<10}thresh={thr_s:<10}check={op}"
            + ("  [CHECK_FAILED]" if name in failed else "")
        )
    lines.append("-" * 100)
    summary = "\n".join(lines)
    return passed, summary


def _fp8_ulp(magnitude, info):
    exponent = math.frexp(max(magnitude, info.tiny))[1] - 1
    return math.ldexp(info.eps, exponent)


def _fp8_error_limit(magnitude, info):
    """以二进制间距直接作 RNE，避免 FP64 经 FP32 中转产生双重舍入。"""
    step = _fp8_ulp(magnitude, info)
    if magnitude > info.max + _fp8_ulp(info.max, info) / 2:
        return None
    narrowed = round(magnitude / step) * step
    if narrowed > info.max:
        return None
    return max(1.0, 32 * _fp8_ulp(narrowed, info))


def _sum_tail(left, right, total):
    """恢复有限浮点和的舍入余量：精确值等于 total 加上余量。"""
    # FastTwoSum 从绝对值较大的加数减去已舍入的和。
    return torch.where(left.abs() >= right.abs(), (left - total) + right, (right - total) + left)


def _abs_error_tail(actual, golden, signed_diff):
    tail = _sum_tail(actual, -golden, signed_diff)
    return torch.where(signed_diff < 0, -tail, tail)


def _fp8_boundary_matches(actual, golden, tol):
    """只复核 FP64 下误差与容差舍入为同一值的点。"""
    # tol 是 2 的幂。比较 |a/tol - g/tol| 与 1 + |g|，避免极小的 g*tol 下溢。
    a, g = actual / tol, golden / tol
    signed_diff = a - g
    error = signed_diff.abs()
    magnitude = golden.abs()
    bound = 1 + magnitude
    error_tail = _abs_error_tail(a, g, signed_diff)
    bound_tail = _sum_tail(torch.ones_like(magnitude), magnitude, bound)
    return (error < bound) | ((error == bound) & (error_tail <= bound_tail))


def _compare_fp8(actual, golden):
    """FP8 数值输出与更高精度 golden 比较，特殊值计入匹配率。"""
    if golden.dtype not in _THRESHOLDS:
        return False, f"FP8 comparison requires a higher-precision floating golden, got {golden.dtype}"
    if actual.numel() == 0:
        return True, "empty tensor"

    a = actual.to(torch.float64)
    g = golden.to(torch.float64)
    finite = torch.isfinite(a) & torch.isfinite(g)
    info = torch.finfo(actual.dtype)
    # E5M2 的输出可为 Inf。有限 golden 在 RNE 溢出中点及以上收窄为同号 Inf。
    # E4M3FN 没有 Inf 编码，该项自然为假。
    overflow = info.max + _fp8_ulp(info.max, info) / 2
    same_inf = torch.isinf(a) & (torch.signbit(a) == torch.signbit(g)) & (g.abs() >= overflow)
    matched = ((torch.isnan(a) & torch.isnan(g)) | same_inf).sum().item()
    max_error, max_tail, limit = 0.0, 0.0, 1.0
    if finite.any():
        a_finite, g_finite = a[finite], g[finite]
        signed_diff = a_finite - g_finite
        diff = signed_diff.abs()
        tol = _FP8_TOLERANCE[actual.dtype]
        bound = tol + tol * g_finite.abs()
        matched += (diff < bound).sum().item()
        boundary = diff == bound
        if boundary.any():
            matched += _fp8_boundary_matches(a_finite[boundary], g_finite[boundary], tol).sum().item()
        max_error = diff.max().item()
        peaks = diff == max_error
        peak_golden = g_finite[peaks]
        if max_error:
            tails = _abs_error_tail(a_finite[peaks], peak_golden, signed_diff[peaks])
            max_tail = tails.max().item()
            peak_golden = peak_golden[tails == max_tail]
        # 真实误差仍并列时取最小 ULP，使判定与元素排列顺序无关。
        magnitude = peak_golden.abs().min().item()
        limit = _fp8_error_limit(magnitude, info)
        if limit is None:
            return False, "no finite FP8 ULP at the maximum-error golden value"
    ratio = matched / actual.numel()
    # 只有舍入后的误差恰好落在上限时，才由舍入余量决定是否越界。
    at_limit = math.isclose(max_error, limit, rel_tol=0.0, abs_tol=0.0)
    within_limit = max_error < limit or (at_limit and max_tail <= 0)
    summary = f"matched_ratio={ratio:.6g}, max_abs_error={max_error:.6g}, limit={limit:.6g}"
    if at_limit and max_tail:
        summary += f", max_error_roundoff={max_tail:+.6g}"
    return ratio >= _REQUIRED_MATCHED_RATIO and within_limit, summary


def _compare_exact_tensor(actual, golden, total):
    if actual.dtype != golden.dtype:
        return False, f"dtype mismatch for exact comparison: actual={actual.dtype}, golden={golden.dtype}"
    if actual.is_floating_point() or golden.is_floating_point():
        return False, f"unsupported floating dtype: {actual.dtype}"
    if actual.dtype != torch.bool:
        try:
            torch.iinfo(actual.dtype)
        except (TypeError, RuntimeError):
            return False, f"unsupported exact-comparison dtype: {actual.dtype}"
    if total == 0:
        return True, "empty tensor"
    return _compare_integer(actual, golden, total)


def _compare_single_tensor(actual: torch.Tensor, golden: torch.Tensor, threshold_dtype=None) -> Tuple[bool, str]:
    """按实际输出 dtype 返回单个张量的判定结果和摘要。"""
    for tensor in (actual, golden):
        if tensor.is_nested or tensor.layout != torch.strided:
            return False, "expected dense strided tensors"
        if tensor.device.type == "meta":
            return False, "meta tensors have no values to compare"
    if actual.shape != golden.shape:
        return False, f"shape mismatch: actual={tuple(actual.shape)} vs golden={tuple(golden.shape)}"
    actual, golden = actual.cpu(), golden.cpu()
    total = actual.numel()
    if actual.is_quantized or golden.is_quantized:
        return False, f"unsupported quantized dtype pair: actual={actual.dtype}, golden={golden.dtype}"
    if actual.is_complex() or golden.is_complex():
        return False, f"unsupported complex dtype pair: actual={actual.dtype}, golden={golden.dtype}"

    if threshold_dtype is not None and threshold_dtype != actual.dtype:
        return False, f"requested dtype {threshold_dtype} differs from actual {actual.dtype}"
    if actual.dtype in _FP8_TOLERANCE:
        return _compare_fp8(actual, golden)
    comparison_dtype = actual.dtype
    threshold = _get_threshold(comparison_dtype)
    if threshold is None:
        return _compare_exact_tensor(actual, golden, total)
    if golden.dtype not in _THRESHOLDS and golden.dtype not in _FP8_TOLERANCE:
        return False, f"unsupported dtype pair for floating comparison: actual={actual.dtype}, golden={golden.dtype}"
    if total == 0:
        return True, "empty tensor"
    prepared, error = _prepare_float_tensors(actual, golden, comparison_dtype)
    if error is not None:
        return False, error
    metrics = _calculate_float_metrics(prepared, threshold, total)
    return _format_float_result(metrics)


def _normalize_outputs(output) -> List[Optional[torch.Tensor]]:
    """统一单个或多个输出，None 可作为多输出的占位。"""
    if output is None:
        return [None]
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)):
        normalized = []
        for index, item in enumerate(output):
            if item is None or isinstance(item, torch.Tensor):
                normalized.append(item)
            else:
                raise TypeError(
                    f"unsupported output leaf at index {index}: {type(item).__name__}"
                )
        return normalized
    raise TypeError(f"unsupported output type: {type(output).__name__}")


def check_precision(
    actual: Union[torch.Tensor, tuple, list],
    golden: Union[torch.Tensor, tuple, list],
    dtype_str: Optional[str] = None,
) -> Tuple[bool, str]:
    """比较 actual 与独立 golden，返回 (是否全部通过, 逐输出摘要)。

    两者可为张量或张量序列。dtype_str 用于核对实际输出类型；
    混合类型的多输出应省略此参数，逐个输出按其 dtype 校验。
    """
    try:
        actuals = _normalize_outputs(actual)
        goldens = _normalize_outputs(golden)
    except TypeError as error:
        return False, str(error)

    if len(actuals) != len(goldens):
        return False, f"output count mismatch: actual={len(actuals)} vs golden={len(goldens)}"
    if dtype_str is not None and dtype_str not in _DTYPE_NAME_MAP:
        return False, f"unsupported dtype: {dtype_str}"
    threshold_dtype = _DTYPE_NAME_MAP.get(dtype_str)

    all_passed = True
    summaries = []
    compared_tensors = 0

    for i, (a, g) in enumerate(zip(actuals, goldens)):
        if a is None and g is None:
            continue
        if a is None or g is None:
            all_passed = False
            summaries.append(f"output[{i}]: one side is None")
            continue

        compared_tensors += 1
        passed, summary = _compare_single_tensor(a, g, threshold_dtype)
        if not passed:
            all_passed = False
        prefix = "" if len(actuals) == 1 else f"output[{i}]: "
        summaries.append(f"{prefix}{summary}")

    if compared_tensors == 0:
        all_passed = False
        summaries.append("no tensor output was compared")

    return all_passed, "; ".join(summaries)
