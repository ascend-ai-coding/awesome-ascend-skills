#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import logging
import os
import sys
import traceback
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = SCRIPT_DIR.parent

# Import shared utility functions from ops-profiling.
# Canonical source: ops/ops-profiling/scripts/msprof_perf_summary.py
_PERF_SCRIPTS = (
    Path(__file__).resolve().parents[5] / "ops" / "ops-profiling" / "scripts"
)
if str(_PERF_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS))

from msprof_perf_summary import (
    _load_module,
    _find_cls as _find_model_class,
    _clone as _clone_value,
    _move as _move_to_device,
)


def _get_device():
    """Return default torch device (NPU)."""
    return torch.device("npu")


# ---------------------------------------------------------------------------
# 精度对比标准（参考 NPU Benchmark 精度对比方法，三项判定）
# ---------------------------------------------------------------------------

REQUIRED_MATCHED_RATIO = 0.9

# allclose 判定阈值 (atol, rtol)：|actual - golden| <= atol + rtol * |golden|
ALLCLOSE_TOLS_STR = {
    "float32": (1e-3, 2**(-13)),  # 2**(-13)=1.220703125e-4
    "float16": (9e-2, 2**(-10)),  # 2**(-10)=9.765625e-4
    "bfloat16": (1e-1, 2**(-7)),   # 2**(-7)=7.8125e-3
}
ALLCLOSE_DEFAULT_TOLS = ALLCLOSE_TOLS_STR["float32"]

# 量化整数输出的 LSB tolerance：dynamic quant / smooth quant 等算子的
# int8/int16 输出，在 NPU 实现侧通常经过 fp32→fp16→int 的中转 cast，
# 与 PyTorch CPU 全 fp32 .round() 比较时会出现 ±1 LSB 噪声。
INT_LSB_TOLERANCE = {
    torch.int8: 1,
    torch.int16: 1,
}


# ---------------------------------------------------------------------------
# dtype 精度优先级表（用于输入类型推断）
# ---------------------------------------------------------------------------

def _build_dtype_rank():
    """dtype 精度优先级表：值越大精度越高。"""
    rank = {
        torch.float64: 100,
        torch.float32: 90,
        torch.float16: 80,
        torch.bfloat16: 70,
        torch.int64: 50,
        torch.int32: 40,
        torch.int16: 30,
        torch.int8: 20,
        torch.uint8: 20,
        torch.bool: 10,
    }
    for name in ("float8_e4m3fn", "float8_e4m3", "float8_e5m2fn", "float8_e5m2"):
        dt = getattr(torch, name, None)
        if dt is not None:
            rank[dt] = 60
    return rank


_DTYPE_RANK = None


def _dtype_rank(dtype):
    global _DTYPE_RANK
    if _DTYPE_RANK is None:
        _DTYPE_RANK = _build_dtype_rank()
    return _DTYPE_RANK.get(dtype, 0)


def _is_int_like_dtype(dtype):
    """判断 dtype 属于"整型类"输入（含 bool；不含浮点/复数）。"""
    if dtype is None:
        return False
    if dtype == torch.bool:
        return True
    return (not dtype.is_floating_point) and (not dtype.is_complex)


def _is_integer_dtype(dtype):
    """判断 torch.dtype 是否为整数类型（不含 bool / 不含浮点 / 不含复数）。"""
    if dtype == torch.bool:
        return False
    return (not dtype.is_floating_point) and (not dtype.is_complex)


def _infer_input_type(inputs):
    """从 inputs 推断输入类型，返回 ("float" | "int" | "no_tensor", input_dtype | None)。

    判定优先级：
    1. 若存在 torch.Tensor 输入：取所有 tensor 中最高精度 dtype 作为输入类型
    2. 若不存在 tensor，但存在 list/tuple of Tensor：取第一个 tensor_list 的首元素 dtype
    3. 其他情况（全为标量 attr / 无输入）：返回 ("no_tensor", None)
    """
    tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
    candidate_dtypes = []
    if tensors:
        candidate_dtypes = [t.dtype for t in tensors]
        top_dtype = max(candidate_dtypes, key=_dtype_rank)
        input_type = "int" if _is_int_like_dtype(top_dtype) else "float"
        return input_type, top_dtype

    tensor_lists = []
    for x in inputs:
        if isinstance(x, (list, tuple)) and len(x) > 0 \
                and all(isinstance(e, torch.Tensor) for e in x):
            tensor_lists.append(x)
    if tensor_lists:
        top_dtype = tensor_lists[0][0].dtype
        input_type = "int" if _is_int_like_dtype(top_dtype) else "float"
        return input_type, top_dtype

    return "no_tensor", None


# ---------------------------------------------------------------------------
# NPU Benchmark 精度判定阈值
# ---------------------------------------------------------------------------

def _get_limits(data_type):
    """根据数据类型返回精度判定的三元组 (small_value_threshold, small_value_error, rel_threshold)。

    阈值表：
    | 数据类型      | small_value_threshold | small_value_error | rel_threshold |
    |--------------|-----------------------|-------------------|---------------|
    | FLOAT16      | 2^{-11}               | 2^{-16}           | 2^{-10}       |
    | BFLOAT16     | 2^{-8}                | 2^{-16}           | 2^{-7}        |
    | FLOAT32      | 2^{-14}               | 2^{-30}           | 2^{-13}       |
    | HiFloat32    | 2^{-12}               | 2^{-28}           | 2^{-11}       |
    | FLOAT8 E4M3  | 2^{-4}                | 2^{-6}            | 2^{-3}        |
    | FLOAT8 E5M2  | 2^{-3}                | 2^{-5}            | 2^{-2}        |
    """
    str_to_limits = {
        "float16": (2**(-11), 2**(-16), 2**(-10)),
        "bfloat16": (2**(-8), 2**(-16), 2**(-7)),
        "float32": (2**(-14), 2**(-30), 2**(-13)),
        "hifloat32": (2**(-12), 2**(-28), 2**(-11)),
        "float8_e4m3": (2**(-4), 2**(-6), 2**(-3)),
        "float8_e5m2": (2**(-3), 2**(-5), 2**(-2)),
        "fp8_e4m3": (2**(-4), 2**(-6), 2**(-3)),
        "fp8_e5m2": (2**(-3), 2**(-5), 2**(-2)),
    }
    if isinstance(data_type, str):
        return str_to_limits.get(data_type.lower(), (2**(-14), 2**(-30), 2**(-13)))

    dtype_limits_map = {
        torch.float16: (2**(-11), 2**(-16), 2**(-10)),
        torch.bfloat16: (2**(-8), 2**(-16), 2**(-7)),
        torch.float32: (2**(-14), 2**(-30), 2**(-13)),
    }

    float8_e4m3 = getattr(torch, 'float8_e4m3fn', None) or getattr(torch, 'float8_e4m3', None)
    if float8_e4m3 is not None:
        dtype_limits_map[float8_e4m3] = (2**(-4), 2**(-6), 2**(-3))

    float8_e5m2 = getattr(torch, 'float8_e5m2fn', None) or getattr(torch, 'float8_e5m2', None)
    if float8_e5m2 is not None:
        dtype_limits_map[float8_e5m2] = (2**(-3), 2**(-5), 2**(-2))

    return dtype_limits_map.get(data_type, (2**(-14), 2**(-30), 2**(-13)))


def _get_allclose_tols(data_type):
    """根据数据类型返回 allclose 判定的 (atol, rtol)。

    判定公式：|actual - golden| <= atol + rtol * |golden|

    阈值表：
    | 数据类型  | atol  | rtol            |
    |----------|-------|-----------------|
    | FLOAT32  | 2e-5  | 2**(-13)        |
    | FLOAT16  | 5e-3  | 2**(-10)        |
    | BFLOAT16 | 1e-2  | 2**(-7)         |
    """
    if isinstance(data_type, str):
        return ALLCLOSE_TOLS_STR.get(data_type.lower(), ALLCLOSE_DEFAULT_TOLS)

    dtype_map = {
        torch.float16: ALLCLOSE_TOLS_STR["float16"],
        torch.bfloat16: ALLCLOSE_TOLS_STR["bfloat16"],
        torch.float32: ALLCLOSE_TOLS_STR["float32"],
    }
    return dtype_map.get(data_type, ALLCLOSE_DEFAULT_TOLS)


# ---------------------------------------------------------------------------
# NPU Benchmark 三项精度判定
# ---------------------------------------------------------------------------

@dataclass
class _AccuracyResult:
    """封装 NPU Benchmark 三项精度判定的全部中间结果。"""
    is_pass: bool
    matched_ratio: float
    max_abs_diff: float
    mere: float | None
    rel_thr: float
    sv_thr: float
    sv_err: float
    atol: float
    rtol: float
    allclose_violation_count: int
    total_finite: int
    matched_count: int
    small_count: int
    normal_count: int
    allclose_ok: bool
    ratio_ok: bool
    mere_ok: bool


def _build_accuracy_metrics(acc: _AccuracyResult):
    """构建精度验证的 metrics 字典。"""
    return {
        "matched_ratio": acc.matched_ratio,
        "max_abs_diff": acc.max_abs_diff,
        "MERE": acc.mere,
        "rel_threshold": acc.rel_thr,
        "small_value_threshold": acc.sv_thr,
        "small_value_error": acc.sv_err,
        "atol": acc.atol,
        "rtol": acc.rtol,
        "max_error_cap_violation_count": acc.allclose_violation_count,
        "required_matched_ratio": REQUIRED_MATCHED_RATIO,
        "total_finite": acc.total_finite,
        "matched_count": acc.matched_count,
        "small_count": acc.small_count,
        "normal_count": acc.normal_count,
        "checks": {
            "max_error_cap": acc.allclose_ok,
            "required_matched_ratio": acc.ratio_ok,
            "MERE": acc.mere_ok,
        },
    }


def _format_accuracy_message(acc: _AccuracyResult):
    """构建精度验证的通过/失败消息。"""
    mere_str = f"{acc.mere:.6g}" if acc.mere is not None else "n/a"
    if acc.is_pass:
        return (
            f"allclose_ok, matched_ratio={acc.matched_ratio:.6f} "
            f"(req>={REQUIRED_MATCHED_RATIO}), MERE={mere_str} "
            f"(rel_thr={acc.rel_thr:.3e}), small={acc.small_count}, normal={acc.normal_count}"
        )
    return (
        f"max_error_cap_violations={acc.allclose_violation_count}/{acc.total_finite} "
        f"(atol={acc.atol:.6e}, rtol={acc.rtol:.6e}, max_abs_diff={acc.max_abs_diff:.6e}, "
        f"ok={acc.allclose_ok}), "
        f"matched_ratio={acc.matched_ratio:.6f} (req>={REQUIRED_MATCHED_RATIO}, ok={acc.ratio_ok}), "
        f"MERE={mere_str} (rel_thr={acc.rel_thr:.6e}, ok={acc.mere_ok}); "
        f"small_count={acc.small_count}, normal_count={acc.normal_count}"
    )


@dataclass
class _CheckParams:
    """封装 NPU Benchmark 精度检查的阈值参数。

    将 sv_thr/sv_err/rel_thr/atol/rtol 打包为具名形式，减少函数参数个数（G.FNM.03）。
    """
    sv_thr: float
    sv_err: float
    rel_thr: float
    atol: float
    rtol: float


def _compute_benchmark_checks(golden_f, actual_f, params):
    """计算 NPU Benchmark 三项精度检查，返回中间结果字典。

    三项判定：allclose 逐元素容差 / matched_ratio 分桶匹配率 / MERE 平均相对误差。
    """
    sv_thr = params.sv_thr
    sv_err = params.sv_err
    rel_thr = params.rel_thr
    atol = params.atol
    rtol = params.rtol

    abs_diff = (actual_f - golden_f).abs()
    abs_golden = golden_f.abs()

    # 分桶（小值域 vs 正常值域）
    small_mask = abs_golden < sv_thr
    normal_mask = ~small_mask

    # 元素级 matched（分桶判定）
    small_ok = abs_diff <= sv_err
    rel_err = abs_diff / (abs_golden + 1e-7)
    normal_ok = rel_err <= rel_thr
    matched_mask = torch.where(small_mask, small_ok, normal_ok)

    total_finite = matched_mask.numel()
    matched_count = int(matched_mask.sum().item())
    matched_ratio = matched_count / total_finite if total_finite > 0 else 1.0
    max_abs_diff = abs_diff.max().item() if total_finite > 0 else 0.0

    # allclose 逐元素判定
    allclose_bound = atol + rtol * abs_golden
    allclose_mask = abs_diff <= allclose_bound
    allclose_violations = int((~allclose_mask).sum().item()) if total_finite > 0 else 0
    allclose_ok = allclose_violations == 0

    # MERE
    normal_count = int(normal_mask.sum().item())
    small_count = int(small_mask.sum().item())
    if total_finite > 0:
        mere = rel_err.mean().item()
        mere_ok = mere < rel_thr
    else:
        mere = None
        mere_ok = True

    return dict(
        matched_ratio=matched_ratio, max_abs_diff=max_abs_diff,
        total_finite=total_finite, matched_count=matched_count,
        small_count=small_count, normal_count=normal_count,
        allclose_ok=allclose_ok, allclose_violation_count=allclose_violations,
        mere=mere, mere_ok=mere_ok,
    )


def _check_accuracy_npu_benchmark(golden, actual, data_type):
    """执行 NPU Benchmark 精度验证（三项判定：allclose, matched_ratio, MERE）。

    Returns:
        (passed, metrics_dict, message)
    """
    golden_f = golden.float()
    actual_f = actual.float()

    sv_thr, sv_err, rel_thr = _get_limits(data_type)
    atol, rtol = _get_allclose_tols(data_type)

    params = _CheckParams(sv_thr=sv_thr, sv_err=sv_err, rel_thr=rel_thr, atol=atol, rtol=rtol)
    d = _compute_benchmark_checks(golden_f, actual_f, params)

    ratio_ok = d["matched_ratio"] >= REQUIRED_MATCHED_RATIO
    is_pass = d["allclose_ok"] and ratio_ok and d["mere_ok"]

    acc = _AccuracyResult(
        is_pass=is_pass,
        matched_ratio=d["matched_ratio"],
        max_abs_diff=d["max_abs_diff"],
        mere=d["mere"],
        rel_thr=rel_thr,
        sv_thr=sv_thr,
        sv_err=sv_err,
        atol=atol,
        rtol=rtol,
        allclose_violation_count=d["allclose_violation_count"],
        total_finite=d["total_finite"],
        matched_count=d["matched_count"],
        small_count=d["small_count"],
        normal_count=d["normal_count"],
        allclose_ok=d["allclose_ok"],
        ratio_ok=ratio_ok,
        mere_ok=d["mere_ok"],
    )
    metrics = _build_accuracy_metrics(acc)
    msg = _format_accuracy_message(acc)
    return is_pass, metrics, msg


# ---------------------------------------------------------------------------
# 非计算类：二进制完全一致比对
# ---------------------------------------------------------------------------

def _view_int_dtype(dt):
    """将浮点/复数 dtype 映射为相同位宽的整型 dtype（用于 bit-exact 比较）。"""
    if dt in (torch.float64, torch.complex64):
        return torch.int64
    if dt in (torch.float32,):
        return torch.int32
    if dt in (torch.float16, torch.bfloat16):
        return torch.int16
    for name in ("float8_e4m3fn", "float8_e4m3", "float8_e5m2fn", "float8_e5m2"):
        fp8 = getattr(torch, name, None)
        if fp8 is not None and dt == fp8:
            return torch.int8
    return None


def _build_bit_mismatch_detail(fw, impl):
    """构建 bit-exact 不匹配时的详细错误信息。"""
    if fw.dtype.is_floating_point and not fw.dtype.is_complex:
        view_dt = _view_int_dtype(fw.dtype)
        fw_bits = fw.view(view_dt).flatten()
        impl_bits = impl.view(view_dt).flatten()
        diff_mask = fw_bits != impl_bits
        violation_count = int(diff_mask.sum().item())
        violation_idx = torch.where(diff_mask)[0]
        num_to_show = min(10, len(violation_idx))
        detail = f", first {num_to_show} bit mismatches:\n"
        fw_flat = fw.flatten()
        impl_flat = impl.flatten()
        for i in range(num_to_show):
            idx = violation_idx[i].item()
            detail += (
                f"    [{idx}]: ref={fw_flat[idx].item()} "
                f"(bits=0x{fw_bits[idx].item() & ((1 << view_dt.itemsize * 8) - 1):x}), "
                f"cand={impl_flat[idx].item()} "
                f"(bits=0x{impl_bits[idx].item() & ((1 << view_dt.itemsize * 8) - 1):x})\n"
            )
    else:
        fw_flat = fw.flatten()
        impl_flat = impl.flatten()
        diff_mask = fw_flat != impl_flat
        violation_count = int(diff_mask.sum().item())
        violation_idx = torch.where(diff_mask)[0]
        num_to_show = min(10, len(violation_idx))
        detail = f", first {num_to_show} mismatches:\n"
        for i in range(num_to_show):
            idx = violation_idx[i].item()
            detail += (
                f"    [{idx}]: ref={fw_flat[idx].item()}, cand={impl_flat[idx].item()}\n"
            )
    return violation_count, detail


def _compare_binary_exact(lhs, rhs, path):
    """非计算类：二进制完全一致比对。

    - 浮点 dtype：通过 view-as-int 比较底层 bit pattern，可识别 NaN payload 差异
    - 整型 / bool：直接 torch.equal
    - 复数：实部/虚部分别 view-as-int 比较

    Returns:
        (ok, message)
    """
    fw = lhs.contiguous().detach().cpu()
    impl = rhs.contiguous()
    if isinstance(impl, torch.Tensor):
        impl = impl.detach().cpu()
    else:
        return False, f"{path}: non-tensor impl output (type={type(impl).__name__})"

    if fw.shape != impl.shape:
        return False, f"{path}: shape mismatch: ref={tuple(fw.shape)}, cand={tuple(impl.shape)}"
    if fw.dtype != impl.dtype:
        return False, f"{path}: dtype mismatch: ref={fw.dtype}, cand={impl.dtype}"

    if fw.dtype.is_complex:
        fw_real_bits = torch.view_as_real(fw)
        impl_real_bits = torch.view_as_real(impl)
        view_dt = _view_int_dtype(torch.float32) if fw.dtype == torch.complex64 else torch.int64
        equal = torch.equal(fw_real_bits.view(view_dt), impl_real_bits.view(view_dt))
    elif fw.dtype.is_floating_point:
        view_dt = _view_int_dtype(fw.dtype)
        if view_dt is None:
            return False, f"{path}: unsupported float dtype for bit-exact: {fw.dtype}"
        equal = torch.equal(fw.view(view_dt), impl.view(view_dt))
    else:
        equal = torch.equal(fw, impl)

    if equal:
        return True, f"{path}: bit-exact matched"

    violation_count, detail = _build_bit_mismatch_detail(fw, impl)
    return False, (
        f"{path}: bit-exact mismatch, {violation_count}/{fw.numel()} elements differ "
        f"(dtype={fw.dtype}){detail}"
    )


# ---------------------------------------------------------------------------
# 结构化比较
# ---------------------------------------------------------------------------

def _normalize_output(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_normalize_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_output(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize_output(item) for key, item in value.items()}
    return value


def _contains_int8_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.dtype == torch.int8
    if isinstance(value, list):
        return any(_contains_int8_tensor(item) for item in value)
    if isinstance(value, tuple):
        return any(_contains_int8_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_int8_tensor(item) for item in value.values())
    return False


def _find_first_mismatch(lhs, rhs, mismatch_mask):
    """Return a string describing the first mismatched element, or empty if no mismatch."""
    if not mismatch_mask.numel():
        return ""
    mismatch_count = mismatch_mask.sum().item()
    if not mismatch_count:
        return ""
    first_linear_idx = int(torch.nonzero(mismatch_mask.reshape(-1), as_tuple=False)[0].item())
    if lhs.ndim == 0:
        first_index = ()
        lhs_val = lhs.item()
        rhs_val = rhs.item()
    else:
        rem = first_linear_idx
        first_index = [0] * lhs.ndim
        for d in range(lhs.ndim - 1, -1, -1):
            first_index[d] = rem % lhs.shape[d]
            rem //= lhs.shape[d]
        first_index = tuple(first_index)
        lhs_val = lhs[first_index].item()
        rhs_val = rhs[first_index].item()
    return f", first_mismatch(index={first_index}, ref={lhs_val}, cand={rhs_val})"


def _complex_diff_summary(lhs, rhs):
    """复数差异摘要（使用 NPU Benchmark 三项判定分别评估实部/虚部）。"""
    passed_r, metrics_r, _msg_r = _check_accuracy_npu_benchmark(rhs.real, lhs.real, rhs.real.dtype)
    passed_i, metrics_i, _msg_i = _check_accuracy_npu_benchmark(rhs.imag, lhs.imag, rhs.imag.dtype)
    lhs_fp = torch.view_as_real(lhs).to(torch.float32)
    rhs_fp = torch.view_as_real(rhs).to(torch.float32)
    diff = (lhs_fp - rhs_fp).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff.mean().item() if diff.numel() else 0.0
    mr_r = metrics_r["matched_ratio"]
    mr_i = metrics_i["matched_ratio"]
    mere_r = metrics_r["MERE"]
    mere_i = metrics_i["MERE"]
    thr_r = metrics_r["rel_threshold"]
    return (
        f"dtype(ref={lhs.dtype}, cand={rhs.dtype}), "
        f"max_abs_diff={max_abs:.6g}, mean_abs_diff={mean_abs:.6g}, "
        f"MR(real={mr_r:.6f}, imag={mr_i:.6f}), "
        f"MERE(real={mere_r:.6g}, imag={mere_i:.6g}), "
        f"rel_thr={thr_r:.3e}, "
        f"passed_real={passed_r}, passed_imag={passed_i}"
    )


def _float_diff_summary(lhs, rhs):
    """浮点差异摘要（使用 NPU Benchmark 三项判定）。"""
    lhs_fp = torch.nan_to_num(lhs.to(torch.float32))
    rhs_fp = torch.nan_to_num(rhs.to(torch.float32))
    diff = (lhs_fp - rhs_fp).abs()
    both_inf_mask = torch.isinf(lhs_fp) & torch.isinf(rhs_fp) & (torch.sign(lhs_fp) == torch.sign(rhs_fp))
    diff[both_inf_mask] = 0.0
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff.mean().item() if diff.numel() else 0.0
    passed, metrics, _msg = _check_accuracy_npu_benchmark(rhs, lhs, lhs.dtype)
    return (
        f"dtype(ref={lhs.dtype}, cand={rhs.dtype}), "
        f"max_abs_diff={max_abs:.6g}, mean_abs_diff={mean_abs:.6g}, "
        f"matched_ratio={metrics['matched_ratio']:.6f}, "
        f"MERE={metrics['MERE']:.6g}, "
        f"rel_thr={metrics['rel_threshold']:.3e}, "
        f"allclose_ok={metrics['checks']['max_error_cap']}, "
        f"passed={passed}"
    )


def _int_diff_summary(lhs, rhs, total):
    lhs_i32 = lhs.to(torch.int32)
    rhs_i32 = rhs.to(torch.int32)
    delta = rhs_i32 - lhs_i32
    abs_diff = delta.abs()
    max_abs = abs_diff.max().item() if abs_diff.numel() else 0
    mean_abs = abs_diff.float().mean().item() if abs_diff.numel() else 0.0
    mismatch_mask = delta != 0
    mismatch_count = mismatch_mask.sum().item() if delta.numel() else 0
    mismatch_ratio = (mismatch_count / total) if total else 0.0
    cand_gt_ref = ((delta > 0) & mismatch_mask).sum().item() if delta.numel() else 0
    cand_lt_ref = ((delta < 0) & mismatch_mask).sum().item() if delta.numel() else 0
    first_mismatch = _find_first_mismatch(lhs, rhs, mismatch_mask) if mismatch_count else ""
    return (
        f"dtype(ref={lhs.dtype}, cand={rhs.dtype}), "
        f"unequal_elements={mismatch_count}, mismatch_ratio={mismatch_ratio:.6%}, "
        f"max_abs_diff={max_abs}, mean_abs_diff={mean_abs:.6g}, "
        f"cand_gt_ref={cand_gt_ref}, cand_lt_ref={cand_lt_ref}"
        f"{first_mismatch}"
    )


def _tensor_diff_summary(lhs: torch.Tensor, rhs: torch.Tensor):
    if lhs.shape != rhs.shape:
        return f"shape mismatch: ref={tuple(lhs.shape)}, cand={tuple(rhs.shape)}"

    if lhs.is_complex() or rhs.is_complex():
        return _complex_diff_summary(lhs, rhs)

    if torch.is_floating_point(lhs) or torch.is_floating_point(rhs):
        return _float_diff_summary(lhs, rhs)

    return _int_diff_summary(lhs, rhs, lhs.numel())


def _check_nan_inf(lhs, rhs, path):
    """检查 NaN / Inf 一致性。返回 (ok, message)，ok=False 表示不匹配。"""
    actual_nan = torch.isnan(lhs)
    golden_nan = torch.isnan(rhs)
    if (actual_nan ^ golden_nan).any():
        fw_nan_count = golden_nan.sum().item()
        impl_nan_count = actual_nan.sum().item()
        return False, (
            f"{path}: NaN mask mismatch: ref={fw_nan_count}/{lhs.numel()}, "
            f"cand={impl_nan_count}/{rhs.numel()}"
        )

    actual_inf = torch.isinf(lhs)
    golden_inf = torch.isinf(rhs)
    if (actual_inf ^ golden_inf).any():
        fw_inf_count = golden_inf.sum().item()
        impl_inf_count = actual_inf.sum().item()
        return False, (
            f"{path}: Inf mask mismatch: ref={fw_inf_count}/{lhs.numel()}, "
            f"cand={impl_inf_count}/{rhs.numel()}"
        )
    if golden_inf.any():
        actual_sign = torch.sign(lhs[actual_inf])
        golden_sign = torch.sign(rhs[golden_inf])
        if not torch.equal(actual_sign, golden_sign):
            return False, f"{path}: Inf sign mismatch"

    return True, ""


# 张量比较公共上下文：统一封装 lhs_finite, rhs_finite, path, finite_count, total_numel
_TensorCmpCtx = namedtuple(
    "_TensorCmpCtx", ["lhs_finite", "rhs_finite", "path", "finite_count", "total_numel"])


def _compare_int_tensors(ctx: _TensorCmpCtx, input_type):
    """整型输出比较：按 input_type 分流为量化类 / 整数计算类。"""
    lhs_finite, rhs_finite = ctx.lhs_finite, ctx.rhs_finite
    path = ctx.path
    finite_count, total_numel = ctx.finite_count, ctx.total_numel
    # input_type == "float" → 量化类 (|diff|<=1)
    if input_type == "float":
        diff = (lhs_finite.to(torch.int64) - rhs_finite.to(torch.int64)).abs()
        violation_count = int((diff > 1).sum().item())
        if violation_count == 0:
            return True, (
                f"{path}: quant matched (fp->int, |diff|<=1, "
                f"{finite_count}/{total_numel} finite)"
            )
        max_diff = int(diff.max().item())
        violation_idx = torch.where(diff > 1)[0]
        num_to_show = min(10, len(violation_idx))
        detail = f", first {num_to_show} quant violations:\n"
        for i in range(num_to_show):
            idx = violation_idx[i].item()
            detail += (
                f"    [{idx}]: ref={lhs_finite[idx].item()}, "
                f"cand={rhs_finite[idx].item()}, |diff|={diff[idx].item()}\n"
            )
        return False, (
            f"{path}: quant mismatch (fp->int, |diff|<=1), "
            f"{violation_count}/{diff.numel()} violations, max_abs_diff={max_diff}"
            f"{detail}"
        )

    # 整数计算类：先用 LSB tolerance，否则严格相等
    tol = INT_LSB_TOLERANCE.get(lhs_finite.dtype) \
        if lhs_finite.dtype == rhs_finite.dtype else None
    if tol is not None:
        diff = (rhs_finite.to(torch.int32) - lhs_finite.to(torch.int32)).abs()
        max_abs = diff.max().item() if diff.numel() else 0
        if max_abs <= tol:
            return True, (
                f"{path}: int matched within ±{tol} LSB "
                f"(max_abs_diff={max_abs}, dtype={lhs_finite.dtype}, "
                f"{finite_count}/{total_numel} finite)"
            )
        return False, f"{path}: {_tensor_diff_summary(lhs_finite, rhs_finite)}"
    if torch.equal(lhs_finite, rhs_finite):
        return True, (
            f"{path}: int matched (strict, {finite_count}/{total_numel} finite)"
        )
    return False, f"{path}: {_tensor_diff_summary(lhs_finite, rhs_finite)}"


def _compare_bool_tensors(lhs_finite, rhs_finite, finite_count, total_numel, path):
    """Compare bool tensors with strict element-wise equality."""
    if torch.equal(lhs_finite, rhs_finite):
        return True, f"{path}: bool matched ({finite_count}/{total_numel} finite)"
    diff_idx = torch.where(lhs_finite != rhs_finite)[0]
    violation_count = int(diff_idx.numel())
    num_to_show = min(10, violation_count)
    detail = f", first {num_to_show} mismatches:\n"
    for i in range(num_to_show):
        idx = diff_idx[i].item()
        detail += f"    [{idx}]: ref={lhs_finite[idx].item()}, cand={rhs_finite[idx].item()}\n"
    return False, (
        f"{path}: bool mismatch, {violation_count}/{lhs_finite.numel()} elements differ"
        f"{detail}"
    )


def _compare_complex_tensors(lhs, rhs, path):
    """Compare complex tensors: run NPU Benchmark 3 checks on real/imag parts separately."""
    passed_r, metrics_r, _msg_r = _check_accuracy_npu_benchmark(rhs.real, lhs.real, rhs.real.dtype)
    passed_i, metrics_i, _msg_i = _check_accuracy_npu_benchmark(rhs.imag, lhs.imag, rhs.imag.dtype)
    passed = passed_r and passed_i
    if passed:
        return True, (
            f"{path}: matched, "
            f"MR(real={metrics_r['matched_ratio']:.6f}, imag={metrics_i['matched_ratio']:.6f}), "
            f"MERE(real={metrics_r['MERE']:.6g}, imag={metrics_i['MERE']:.6g}), "
            f"allclose_ok(real={metrics_r['checks']['max_error_cap']}, "
            f"imag={metrics_i['checks']['max_error_cap']})"
        )
    return False, f"{path}: {_tensor_diff_summary(lhs, rhs)}"


def _compare_float_tensors(ctx: _TensorCmpCtx, lhs):
    """Compare float tensors using NPU Benchmark 3 checks."""
    lhs_finite, rhs_finite = ctx.lhs_finite, ctx.rhs_finite
    path = ctx.path
    finite_count, total_numel = ctx.finite_count, ctx.total_numel
    if rhs_finite.dtype != lhs_finite.dtype:
        rhs_finite = rhs_finite.to(lhs_finite.dtype)
    passed, metrics, msg = _check_accuracy_npu_benchmark(rhs_finite, lhs_finite, lhs_finite.dtype)
    if passed:
        return True, f"{path}: matched, {msg} ({finite_count}/{total_numel} finite)"
    return False, f"{path}: {_tensor_diff_summary(lhs, rhs_finite)}"


def _compare_tensors(lhs, rhs, path, input_type=None, non_compute=False):
    """Compare two tensors using NPU Benchmark precision standards.

    决策矩阵（non_compute=False 时）：
        | 输出 dtype | 输入类型    | 类别        | 判定                    |
        |-----------|------------|------------|------------------------|
        | bool      | 任意        | bool 输出   | torch.equal            |
        | int       | int        | 整数计算类   | |                                 diff| == 0              |
        | int       | float      | 量化类      | |                                 diff| <= 1              |
        | int       | no_tensor  | 整数计算类   | |                                 diff| == 0              |
        | float     | 任意        | 浮点计算类   | NPU Benchmark 三项判定  |
    """
    if lhs.shape != rhs.shape:
        return False, f"{path}: shape mismatch: ref={tuple(lhs.shape)}, cand={tuple(rhs.shape)}"

    ok, msg = _check_nan_inf(lhs, rhs, path)
    if not ok:
        return False, msg

    # 非计算类：二进制完全一致
    if non_compute:
        return _compare_binary_exact(lhs, rhs, path)

    finite_mask = torch.isfinite(lhs) & torch.isfinite(rhs)
    finite_count = finite_mask.sum().item()

    # 所有元素均为双 NaN 或双 Inf
    if finite_count == 0:
        return True, f"{path}: matched (all non-finite)"

    lhs_finite = lhs[finite_mask]
    rhs_finite = rhs[finite_mask]

    # bool 输出：严格相等
    if lhs_finite.dtype == torch.bool:
        return _compare_bool_tensors(lhs_finite, rhs_finite, finite_count, lhs.numel(), path)

    # 输出整型：按 input_type 分流到量化类 / 整数计算类
    if _is_integer_dtype(lhs_finite.dtype):
        int_ctx = _TensorCmpCtx(lhs_finite, rhs_finite, path, finite_count, lhs.numel())
        return _compare_int_tensors(int_ctx, input_type)

    # 复数：实部/虚部分别做 NPU Benchmark 三项判定
    if lhs.is_complex() or rhs.is_complex():
        return _compare_complex_tensors(lhs, rhs, path)

    # 输出浮点（默认）：NPU Benchmark 三项判定
    float_ctx = _TensorCmpCtx(lhs_finite, rhs_finite, path, finite_count, lhs.numel())
    return _compare_float_tensors(float_ctx, lhs)


def _compare_structured(lhs, rhs, compare_leaf, path: str = "output"):
    """Generic structural comparison. Delegates leaf values to `compare_leaf`."""
    if type(lhs) is not type(rhs):
        return False, f"{path}: type mismatch: ref={type(lhs).__name__}, cand={type(rhs).__name__}"

    if isinstance(lhs, list):
        if len(lhs) != len(rhs):
            return False, f"{path}: list length mismatch: ref={len(lhs)}, cand={len(rhs)}"
        for index, (a, b) in enumerate(zip(lhs, rhs)):
            ok, message = _compare_structured(a, b, compare_leaf, f"{path}[{index}]")
            if not ok:
                return False, message
        return True, f"{path}: matched"
    if isinstance(lhs, tuple):
        if len(lhs) != len(rhs):
            return False, f"{path}: tuple length mismatch: ref={len(lhs)}, cand={len(rhs)}"
        for index, (a, b) in enumerate(zip(lhs, rhs)):
            ok, message = _compare_structured(a, b, compare_leaf, f"{path}[{index}]")
            if not ok:
                return False, message
        return True, f"{path}: matched"
    if isinstance(lhs, dict):
        if lhs.keys() != rhs.keys():
            return False, f"{path}: dict keys mismatch: ref={sorted(lhs.keys())}, cand={sorted(rhs.keys())}"
        for key in lhs:
            ok, message = _compare_structured(lhs[key], rhs[key], compare_leaf, f"{path}.{key}")
            if not ok:
                return False, message
        return True, f"{path}: matched"

    return compare_leaf(lhs, rhs, path)


def _compare_values(lhs, rhs, path: str = "output", input_type=None, non_compute=False):
    """Recursively compare values using NPU Benchmark precision for Tensors."""

    def _compare_leaf(a, b, p):
        if isinstance(a, torch.Tensor):
            return _compare_tensors(a, b, p, input_type=input_type, non_compute=non_compute)
        if a == b:
            return True, f"{p}: matched"
        return False, f"{p}: value mismatch: ref={a}, cand={b}"

    return _compare_structured(lhs, rhs, _compare_leaf, path)


# ---------------------------------------------------------------------------
# 验证主流程
# ---------------------------------------------------------------------------

def _resolve_task_dir(op: str, workdir: Path = WORKDIR) -> Path:
    op_path = Path(op)
    if op_path.is_dir():
        return op_path.resolve()

    direct = workdir / op
    if direct.is_dir():
        return direct

    raise FileNotFoundError(f"Cannot find task directory for op '{op}'")


def _format_tensor_summary(tensor: torch.Tensor) -> str:
    return f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device})"


def _summarize_value(value, name: str):
    if isinstance(value, torch.Tensor):
        return [f"{name}: {_format_tensor_summary(value)}"]
    if isinstance(value, list):
        lines = [f"{name}: list[{len(value)}]"]
        for index, item in enumerate(value):
            lines.extend(_summarize_value(item, f"{name}[{index}]"))
        return lines
    if isinstance(value, tuple):
        lines = [f"{name}: tuple[{len(value)}]"]
        for index, item in enumerate(value):
            lines.extend(_summarize_value(item, f"{name}[{index}]"))
        return lines
    if isinstance(value, dict):
        lines = [f"{name}: dict[{len(value)}]"]
        for key, item in value.items():
            lines.extend(_summarize_value(item, f"{name}.{key}"))
        return lines
    return [f"{name}: {type(value).__name__}({value})"]


def _get_input_groups(module):
    # Prefer get_input_groups(); fall back to get_inputs() wrapped in a list.
    if hasattr(module, "get_input_groups"):
        input_groups = module.get_input_groups()
        if not isinstance(input_groups, list) or not input_groups:
            raise ValueError("get_input_groups() must return a non-empty list")
        return input_groups

    if hasattr(module, "get_inputs"):
        inputs = module.get_inputs()
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("get_inputs() must return a non-empty list")
        return [inputs]

    raise AttributeError(f"Neither get_input_groups() nor get_inputs() found in {module.__file__}")


def _make_verification_report(op):
    return {
        "op": op,
        "ok": False,
        "device": str(_get_device()),
        "task_dir": "",
        "reference": "",
        "candidate": "",
        "kernel_build_dir": "",
        "non_compute": False,
        "inputs": [],
        "comparisons": [],
        "comparison": "",
        "error": "",
    }


def _setup_paths(kernel_build_dir):
    inserted_paths = []
    paths_to_add = [str(WORKDIR)]
    if kernel_build_dir.is_dir():
        paths_to_add.append(str(kernel_build_dir))
    else:
        import warnings
        warnings.warn(
            f"{kernel_build_dir} not found; assuming model_new_ascendc.py handles its own import path.",
            UserWarning, stacklevel=2,
        )
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)
            inserted_paths.append(p)
    return inserted_paths


def _execute_models(ref_model, cand_model, input_groups, device):
    """Run both models on all input groups, returning normalized outputs and summaries."""
    ref_outputs = []
    cand_outputs = []
    input_summaries = []
    for index, inputs in enumerate(input_groups):
        ref_inputs = _move_to_device(_clone_value(inputs), device)
        cand_inputs = _move_to_device(_clone_value(inputs), device)
        input_summaries.extend(_summarize_value(ref_inputs, f"inputs[{index}]"))

        with torch.no_grad():
            ref_out = ref_model(*ref_inputs)
            cand_out = cand_model(*cand_inputs)

        if hasattr(ref_model, "postprocess_output"):
            ref_out = ref_model.postprocess_output(ref_out, inputs)
            cand_out = ref_model.postprocess_output(cand_out, inputs)

        ref_outputs.append(_normalize_output(ref_out))
        cand_outputs.append(_normalize_output(cand_out))

    return ref_outputs, cand_outputs, input_summaries


def _run_comparisons(ref_model, cand_model, input_groups, device, non_compute=False):
    all_ok = True
    comparisons = []
    ref_outputs, cand_outputs, input_summaries = _execute_models(
        ref_model, cand_model, input_groups, device)
    for index, (inputs, ref_out, cand_out) in enumerate(zip(input_groups, ref_outputs, cand_outputs)):
        input_type, _input_dtype = _infer_input_type(inputs)
        ok, comparison = _compare_values(
            ref_out, cand_out, path=f"output[{index}]",
            input_type=input_type, non_compute=non_compute,
        )
        comparisons.append(f"case[{index}]: {comparison}")
        all_ok = all_ok and ok
    return all_ok, comparisons, input_summaries


def _run_verification(op: str, non_compute: bool = False):
    report = _make_verification_report(op)
    report["non_compute"] = non_compute

    task_dir = _resolve_task_dir(op)
    ref_path = task_dir / "model.py"
    cand_path = task_dir / "model_new_ascendc.py"
    kernel_build_dir = task_dir / "kernel" / "build"
    report["task_dir"] = str(task_dir)
    report["reference"] = str(ref_path)
    report["candidate"] = str(cand_path)
    report["kernel_build_dir"] = str(kernel_build_dir)

    if not ref_path.is_file():
        report["error"] = f"missing reference model: {ref_path}"
        return report
    if not cand_path.is_file():
        report["error"] = f"missing candidate model: {cand_path}"
        return report

    inserted_paths = _setup_paths(kernel_build_dir)
    try:
        ref_module = _load_module(ref_path, f"{op}_ref_model")
        cand_module = _load_module(cand_path, f"{op}_ascendc_model")

        ref_cls = _find_model_class(ref_module, "Model")
        cand_cls = _find_model_class(cand_module, "ModelNew")

        torch.manual_seed(0)
        if hasattr(cand_module, "get_init_inputs"):
            init_inputs = cand_module.get_init_inputs()
        else:
            init_inputs = getattr(ref_module, "get_init_inputs", lambda: [])()
        input_groups = _get_input_groups(ref_module)
        device = _get_device()

        ref_model = ref_cls(*_clone_value(init_inputs)).to(device).eval()
        cand_model = cand_cls(*_clone_value(init_inputs)).to(device).eval()

        all_ok, comparisons, input_summaries = _run_comparisons(
            ref_model, cand_model, input_groups, device, non_compute=non_compute)

        report["inputs"] = input_summaries
        report["comparisons"] = comparisons
        report["comparison"] = "\n".join(comparisons)
        report["ok"] = all_ok
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if os.environ.get("VERIFICATION_ASCENDC_DEBUG") == "1":
            raise
        report["traceback"] = traceback.format_exc()
        return report
    finally:
        for p in inserted_paths:
            if p in sys.path:
                sys.path.remove(p)


def verify(op: str, non_compute: bool = False) -> bool:
    return _run_verification(op, non_compute=non_compute)["ok"]


def _print_report(report, title="AscendC Verification Report",
                  extra_header_lines=None, debug_env_var="VERIFICATION_ASCENDC_DEBUG"):
    status = "PASS" if report["ok"] else "FAIL"
    lines = []
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)
    lines.append(f"Status       : {status}")
    lines.append(f"Operator     : {report['op']}")
    lines.append(f"Device       : {report['device']}")
    lines.append(f"Non-compute  : {report.get('non_compute', False)}")
    lines.append(f"Task Dir     : {report['task_dir']}")
    lines.append(f"Reference    : {report['reference']}")
    lines.append(f"Candidate    : {report['candidate']}")
    if extra_header_lines:
        lines.extend(extra_header_lines)

    if report["inputs"]:
        lines.append("-" * 72)
        lines.append("Inputs")
        lines.append("-" * 72)
        for line in report["inputs"]:
            lines.append(line)

    lines.append("-" * 72)
    lines.append("Comparison")
    lines.append("-" * 72)
    if report["comparison"]:
        lines.append(report["comparison"])
    elif report["error"]:
        lines.append(report["error"])
    else:
        lines.append("No comparison information available")

    if report["error"] and os.environ.get(debug_env_var) == "1":
        lines.append("-" * 72)
        lines.append("Traceback")
        lines.append("-" * 72)
        lines.append(report.get("traceback", ""))

    lines.append("-" * 72)
    lines.append(f"Result: {'pass' if report['ok'] else 'fail'}")

    logger.info("\n".join(lines))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AscendC 算子精度验证脚本")
    parser.add_argument("op", nargs="?", help="算子名称或任务目录路径")
    parser.add_argument(
        "--non-compute", action="store_true",
        help="非计算类算子（搬移 / Cast 等），走二进制完全一致判定",
    )
    args = parser.parse_args()

    if args.op is None:
        logger.error("Usage: python verification_ascendc.py <op> [--non-compute]")
        logger.error("Result: fail")
        raise SystemExit(1)

    report = _run_verification(args.op, non_compute=args.non_compute)
    _print_report(report,
                  extra_header_lines=[f"Kernel    : {report['kernel_build_dir']}"])
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
