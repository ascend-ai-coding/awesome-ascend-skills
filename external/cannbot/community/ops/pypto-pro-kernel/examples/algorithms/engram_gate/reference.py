# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent CPU FP32 mathematics and fixed numerical comparison criteria.

Only the final result is storage-rounded. BF16 acceptance uses rtol=atol=2^-6,
matched_ratio>=0.99, and fixed maximum/mean absolute and mean relative error gates.
The dtype table below owns the executable thresholds and takes no test overrides.
"""

from typing import List, Optional, Tuple, Union

import torch


def make_inputs(t, hc, dim, dtype=torch.bfloat16, mask_kind="none", seed=20261012):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shapes = ((t, hc, dim), (t, hc, dim), (t, dim), (hc, dim))
    tensors = [
        torch.randn(shape, generator=generator)
        .clamp(-8, 8)
        .to(torch.float32 if index == 3 else dtype)
        for index, shape in enumerate(shapes)
    ]
    if mask_kind == "none":
        mask = None
    elif mask_kind == "partial":
        mask = torch.arange(t) % 2 == 1
    elif mask_kind == "all_false":
        mask = torch.zeros(t, dtype=torch.bool)
    elif mask_kind == "all_true":
        mask = torch.ones(t, dtype=torch.bool)
    else:
        raise ValueError(mask_kind)
    return (*tensors, mask)


def engram_gate_golden_cpu(
    x, key, value, weight, image_mask=None, eps=1e-6, clamp_value=1e-6
):
    inputs = (x, key, value, weight)
    if any(a.device.type != "cpu" for a in inputs):
        raise ValueError("reference inputs must be on CPU")
    output = torch.empty_like(x)
    w = weight.float()
    dim = x.shape[-1]
    for start in range(0, x.shape[0], 128):
        xx = x[slice(start, start + 128)].float()
        kk = key[slice(start, start + 128)].float()
        vv = value[slice(start, start + 128)].float()
        rstd = torch.rsqrt(xx.square().mean(-1) + eps) * torch.rsqrt(
            kk.square().mean(-1) + eps
        )
        dot = (xx * w * kk).sum(-1) * rstd * (dim**-0.5)
        signed = torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot)
        y = (xx + torch.sigmoid(signed).unsqueeze(-1) * vv.unsqueeze(-2)).to(x.dtype)
        if image_mask is not None:
            y = torch.where(
                image_mask[slice(start, start + 128), None, None],
                x[slice(start, start + 128)],
                y,
            )
        output[slice(start, start + 128)] = y
    return output


_REQUIRED_MATCHED_RATIO = 0.99

_THRESHOLDS = {
    torch.float16: (2.0**-9, 2.0**-9, 1e-1),  # 1.95e-3, 1.95e-3, 0.1
    torch.bfloat16: (2.0**-6, 2.0**-6, 1e-0),  # 1.56e-2, 1.56e-2, 1.0
    torch.float32: (2.0**-10, 2.0**-16, 1e-2),  # 9.77e-4, 1.53e-5, 0.01
    torch.float64: (2.0**-10, 2.0**-16, 1e-2),  # 同 float32
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


def _get_threshold(dtype):
    """查表返回 (rtol, atol, required_matched_ratio, max_abs_error_limit)。

    对已登记浮点类型返回阈值；其他 dtype 返回 None，由调用方再区分整数精确匹配
    与不支持类型，避免把未知浮点、复数或量化类型误当作整数。
    """
    if dtype in _THRESHOLDS:
        rtol, atol, fixed_limit = _THRESHOLDS[dtype]
        ulp = _compute_ulp(dtype)
        max_abs_limit = max(fixed_limit, 32.0 * ulp)
        return rtol, atol, _REQUIRED_MATCHED_RATIO, max_abs_limit
    # 整数类型 → 精确匹配
    return None


def _compute_ulp(dtype):
    """计算给定 dtype 的 ULP (Unit in the Last Place)。

    ULP = 2^(exponent - mantissa_bits)，取 1.0 附近的 ULP 值。
    """
    try:
        one = torch.tensor(1.0, dtype=dtype)
        nxt = torch.nextafter(one, torch.tensor(2.0, dtype=dtype))
        return (nxt - one).item()
    except (TypeError, RuntimeError):
        # 降级：用 finfo 推算
        fi = torch.finfo(dtype)
        return fi.eps


def _to_cpu(tensor):
    """转到 CPU，不改变 dtype。"""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu() if tensor.is_cuda or "npu" in str(tensor.device) else tensor
    return tensor


def _compare_integer(actual, golden, total):
    passed = torch.equal(actual, golden)
    if passed:
        return (
            True,
            "exact match (integer)",
            {"matched_ratio": 1.0, "max_abs_error": 0.0},
        )
    mismatch = (actual != golden).sum().item()
    metrics = {"matched_ratio": 1 - mismatch / total}
    return False, f"integer mismatch: {mismatch}/{total} elements differ", metrics


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
        pass_count = (
            element_pass & compare_mask
        ).sum().item() + match_mask.sum().item()
        matched_ratio = pass_count / total
        max_abs_error = diff[compare_mask].max().item()
        mean_abs_error = diff[compare_mask].mean().item()
        golden_abs = golden.abs()[compare_mask]
        rel_valid = golden_abs > 0
        if rel_valid.any().item():
            mean_rel_error = (
                (diff[compare_mask][rel_valid] / golden_abs[rel_valid]).mean().item()
            )
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
        (
            "matched_ratio",
            metrics["matched_ratio"],
            metrics["required_matched_ratio"],
            "LT",
        ),
        (
            "max_abs_error",
            metrics["max_abs_error"],
            metrics["max_abs_error_limit"],
            "GT",
        ),
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
    return passed, summary, metrics


def _compare_single_tensor(
    actual: torch.Tensor, golden: torch.Tensor, threshold_dtype=None
) -> Tuple[bool, str, dict]:
    """对比单个 tensor，返回 (passed, summary, metrics)。

    按本文件固定策略：
      - 浮点: 混合容差 + matched_ratio + max_abs_error_limit
      - 整数: torch.equal 精确匹配
      - NaN 位置必须一致
      - Inf: 同号视为匹配；异号 FAIL；一方 Inf 一方有限值→替换为 finfo.max 后正常判定
    """
    actual = _to_cpu(actual)
    golden = _to_cpu(golden)
    if actual.shape != golden.shape:
        return (
            False,
            f"shape mismatch: actual={tuple(actual.shape)} vs golden={tuple(golden.shape)}",
            {},
        )
    total = actual.numel()
    if actual.is_quantized or golden.is_quantized:
        return (
            False,
            f"unsupported quantized dtype pair: actual={actual.dtype}, golden={golden.dtype}",
            {},
        )
    if actual.is_complex() or golden.is_complex():
        return (
            False,
            f"unsupported complex dtype pair: actual={actual.dtype}, golden={golden.dtype}",
            {},
        )

    comparison_dtype = threshold_dtype or actual.dtype
    threshold = _get_threshold(comparison_dtype)
    if threshold is None:
        if actual.dtype != golden.dtype:
            return (
                False,
                f"dtype mismatch for exact comparison: actual={actual.dtype}, golden={golden.dtype}",
                {},
            )
        if actual.is_floating_point() or golden.is_floating_point():
            return False, f"unsupported floating dtype: {comparison_dtype}", {}
        if actual.dtype != torch.bool:
            try:
                torch.iinfo(actual.dtype)
            except (TypeError, RuntimeError):
                return False, f"unsupported exact-comparison dtype: {actual.dtype}", {}
        if total == 0:
            return True, "empty tensor", {}
        return _compare_integer(actual, golden, total)
    if not actual.is_floating_point() or not golden.is_floating_point():
        return (
            False,
            f"unsupported dtype pair for floating comparison: actual={actual.dtype}, golden={golden.dtype}",
            {},
        )
    if total == 0:
        return True, "empty tensor", {}
    prepared, error = _prepare_float_tensors(actual, golden, comparison_dtype)
    if error is not None:
        return False, error, {}
    metrics = _calculate_float_metrics(prepared, threshold, total)
    return _format_float_result(metrics)


def _normalize_outputs(output) -> List[Optional[torch.Tensor]]:
    """将输出标准化为 tensor 列表。

    支持: 单个 tensor / tuple / list / None 占位。None 只能作为多输出对齐占位；
    顶层 None 或全 None 会由 check_precision 判定为失败。
    """
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
    """精度对比主入口（方案A混合容差标准）。

    Args:
        actual: 算子输出（NPU tensor 或 CPU tensor，单输出或多输出）
        golden: golden 参考输出（CPU FP32 更高精度）
        dtype_str: 可选，指定 dtype 名称字符串（默认从 actual 推断）

    Returns:
        (passed, summary): passed=True/False, summary=人类可读的精度指标摘要

    本文件固定策略：
      - 逐元素: |actual - golden| <= atol + rtol * |golden|
      - 整体: matched_ratio >= 0.99 AND max_abs_error <= max_abs_error_limit
      - max_abs_error_limit = max(fixed_limit, 32 * ULP)
      - 整数: 精确匹配 (torch.equal)
      - NaN 位置必须一致
      - Inf: 同号匹配, 异号 FAIL, 一方Inf→替换为 finfo.max
    """
    try:
        actuals = _normalize_outputs(actual)
        goldens = _normalize_outputs(golden)
    except TypeError as error:
        return False, str(error)

    if len(actuals) != len(goldens):
        return (
            False,
            f"output count mismatch: actual={len(actuals)} vs golden={len(goldens)}",
        )
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
        passed, summary, _ = _compare_single_tensor(a, g, threshold_dtype)
        if not passed:
            all_passed = False
        prefix = "" if len(actuals) == 1 else f"output[{i}]: "
        summaries.append(f"{prefix}{summary}")

    if compared_tensors == 0:
        all_passed = False
        summaries.append("no tensor output was compared")

    return all_passed, "; ".join(summaries)
