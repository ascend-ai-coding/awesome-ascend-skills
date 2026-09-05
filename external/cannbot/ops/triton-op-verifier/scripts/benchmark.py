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

# 性能测试脚本 — 使用 torch_npu.profiler 测试生成算子的性能表现

import argparse
import gc
import importlib
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

# Baseline anchor gate — refuses to run if {work_dir}/{op_name}.py was tampered
# with after Phase 1 freeze. See _baseline_integrity.py for exit codes.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from _baseline_integrity import _check_baseline_integrity, BaselineGateError
except ImportError:
    _check_baseline_integrity = None  # graceful: gate disabled if module missing

    class BaselineGateError(Exception):  # fallback: never raised when gate disabled
        pass

# A5 NPU frequency locking helpers — optional, failures are warnings by default
try:
    from lock_npu_frequency import (
        detect_npu_devices,
        get_all_frequencies,
        lock_npu_frequency,
        FrequencyMonitor,
        format_frequency_report,
    )
except ImportError:
    detect_npu_devices = None
    get_all_frequencies = None
    lock_npu_frequency = None
    FrequencyMonitor = None
    format_frequency_report = None


# ============================================================================
# 日志配置
# ============================================================================

# 确保同目录下的 _log_utils 可被导入（脚本可能从其他工作目录调用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _log_utils import setup_logger as _setup_logger_shared  # noqa: E402	 
from _common_utils import describe_input as _describe_input_shared  # noqa: E402
from _common_utils import move_to_device  # noqa: E402
from npu_preflight import load_preflight_options, run_preflight, write_preflight_result  # noqa: E402

logger = logging.getLogger("triton_op_verifier.benchmark")


def _setup_logger() -> None:
    """配置 logger：复用 _log_utils.setup_logger。"""
    _setup_logger_shared(logger)


# ============================================================================
# 配置常量
# ============================================================================

WARMUP_DEFAULT = 5
REPEATS_DEFAULT = 5
TRITON_IMPL_NAME_DEFAULT = "triton_ascend_impl"
ERROR_MSG_LIMIT = 2000

# 坏行剔除参数（做法 B）：阈值 = 中位数 × BAD_ROW_K，配绝对下限防止中位数≈0 时阈值过小
BAD_ROW_K = 50
BAD_ROW_FLOOR_US = 10.0


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class BenchmarkConfig:
    """性能测试配置"""
    op_name: str
    verify_dir: str
    triton_impl_name: str = TRITON_IMPL_NAME_DEFAULT
    warmup: int = WARMUP_DEFAULT
    repeats: int = REPEATS_DEFAULT
    skip_framework: bool = False
    framework_latency_ms: float = 0.0
    clear_l2_cache: bool = False  # 是否清除 L2 cache
    keep_res: bool = False        # 是否保留 profiling 结果目录
    max_retries: int = 3          # 采集失败时的最大重试次数
    lock_frequency: bool = True
    lock_frequency_fail_action: str = "warn"  # "warn" | "error"
    freq_check_interval: float = 1.0



@dataclass
class CaseContext:
    """单个测试用例在整体序列中的定位（1-based）。"""
    case_idx: int
    total_cases: int


@dataclass
class MeasureContext:
    """单次 profiling 测量所需的全部参数。"""
    model: Any
    inputs: List[Any]
    warmup: int
    repeats: int
    profile_name: str
    device: Any


@dataclass
class Measurement:
    """单次 profiling 测量结果三元组：算子分项耗时 / 平均时延 / 峰值显存。"""
    operators: Dict[str, float]
    latency_ms: Optional[float]
    peak_memory: float


@dataclass
class ModelPair:
    """framework / impl 模型成对。"""
    framework: Any
    impl: Any


@dataclass
class BenchmarkModelSpec:
    """单 shape benchmark 所需的模型工厂三元组。"""
    framework_cls: Any
    impl_cls: Any
    get_init_inputs: Any


@dataclass
class SpeedupBuckets:
    """按 classify_speedup 把通过的 shape 分桶后的结果。"""
    valid_speedups: List[float] = field(default_factory=list)
    nan_indices: List[int] = field(default_factory=list)
    inf_indices: List[int] = field(default_factory=list)
    zero_indices: List[int] = field(default_factory=list)
    negative_indices: List[int] = field(default_factory=list)
    none_indices: List[int] = field(default_factory=list)


@dataclass
class PerfAggregate:
    """跨 shape 的 latency / memory / 算子分项耗时聚合结果。"""
    avg_fw: float
    avg_impl: float
    avg_fw_mem: float
    avg_impl_mem: float
    fw_ops: Dict[str, float]
    impl_ops: Dict[str, float]
    n: int


@dataclass
class PerformanceResult:
    """单次性能测试结果。

    operators: {Name: {"avg_us": 单次调用耗时(us)}}；per-shape 结果额外含 launch_count
    （一次 forward 内发射次数，为 per-shape 属性，不跨 shape 聚合）。
    """
    avg_latency_ms: float
    peak_memory_mb: float
    operators: Dict[str, Dict[str, Any]]


@dataclass
class SingleShapeResult:
    """单个 shape 的性能测试结果。

    失败用例时 framework / implementation / speedup_vs_torch 为 None，
    status="fail" 且附带 error_type / error_msg。
    通过用例但 speedup 异常（NaN/Inf/0/负数）时 status="pass"，
    speedup_vs_torch 落盘为 null，case_idx 收集到 BenchmarkResult 的对应分类列表。
    """
    case_idx: int
    input_desc: List[Dict[str, Any]]
    status: str = "pass"           # "pass" | "fail"
    framework: Optional[PerformanceResult] = None
    implementation: Optional[PerformanceResult] = None
    speedup_vs_torch: Optional[float] = None
    error_type: Optional[str] = None
    error_msg: Optional[str] = None


@dataclass
class BenchmarkResult:
    """完整性能测试结果。

    speedup_vs_torch: 各通过 shape 加速比的几何平均(不含异常 shape)。
    *_indices: 五类异常 shape 的 case_idx 列表(从 1 开始),与 passed_cases 不冲突,
               异常 shape 仍计入 passed_cases,只是 s_i 不进入几何平均。
    """
    op_name: str
    warmup: int
    repeats: int
    framework: Optional[PerformanceResult]
    implementation: Optional[PerformanceResult]
    speedup_vs_torch: Optional[float]
    total_cases: int = 1
    passed_cases: int = 0
    failed_cases: int = 0
    max_retries: int = 3
    nan_indices: List[int] = field(default_factory=list)
    inf_indices: List[int] = field(default_factory=list)
    zero_indices: List[int] = field(default_factory=list)
    negative_indices: List[int] = field(default_factory=list)
    none_indices: List[int] = field(default_factory=list)
    per_shape_results: List[SingleShapeResult] = field(default_factory=list)
    npu_preflight: Optional[Dict[str, Any]] = None
    failure_class: Optional[str] = None


@dataclass
class OverallAggregate:
    """compute_overall 的聚合结果：跨 shape 的整体均值与异常索引分类。"""
    framework: Optional[PerformanceResult]
    implementation: Optional[PerformanceResult]
    speedup_vs_torch: Optional[float]
    nan_indices: List[int] = field(default_factory=list)
    inf_indices: List[int] = field(default_factory=list)
    zero_indices: List[int] = field(default_factory=list)
    negative_indices: List[int] = field(default_factory=list)
    none_indices: List[int] = field(default_factory=list)


# ============================================================================
# 通用辅助函数
# ============================================================================

def truncate_error(msg: str, limit: int = ERROR_MSG_LIMIT) -> str:
    """截断过长错误信息：保留头尾各 limit/2 字符。"""
    if msg is None:
        return ""
    if len(msg) <= limit:
        return msg
    half = limit // 2
    return f"{msg[:half]}\n... [truncated {len(msg) - limit} chars] ...\n{msg[-half:]}"


def describe_input(inputs: List[Any]) -> List[Dict[str, Any]]:
    """将输入列表描述为结构化字段，便于写入 JSON。

    - torch.Tensor → {"type": "tensor", "shape": [...], "dtype": "..."}
    - 其他标量/对象 → {"type": "scalar", "value": repr(x)}
    """
    return _describe_input_shared(inputs)


def cleanup_npu_memory() -> None:
    """清理 NPU 显存，避免单个 shape 失败后连锁 OOM。"""
    try:
        import torch
        import torch_npu  # noqa: F401
        torch.npu.empty_cache()
    except Exception as e:
        # 例外：非 NPU 环境或 torch_npu 不可用时清理无意义，仅记录调试信息，不影响主流程
        logger.debug("跳过 NPU 显存清理（环境不支持 torch_npu）: %s: %s", type(e).__name__, e)
    gc.collect()


# ============================================================================
# 输入解析
# ============================================================================

def resolve_inputs(op_name: str, verify_dir: str):
    """解析任务文件的输入提供方式。

    支持两种格式：
        - get_inputs(): 旧格式，返回单组输入
        - get_input_groups(): 新格式，返回多组输入列表

    Returns:
        输入组列表 (List[List[Any]])
    """
    import torch  # noqa: F401
    sys.path.insert(0, verify_dir)
    torch_module = importlib.import_module(f"{op_name}_torch")

    if hasattr(torch_module, "get_input_groups"):
        return torch_module.get_input_groups()
    elif hasattr(torch_module, "get_inputs"):
        return [torch_module.get_inputs()]
    else:
        raise AttributeError(
            "模块必须提供 get_inputs() 或 get_input_groups() 方法"
        )


def prepare_model_fn(model: Any, inputs: List[Any], device: Any) -> callable:
    """准备模型用于性能测试，返回测试函数"""
    import torch
    import torch_npu  # noqa: F401

    with torch.no_grad():
        _ = model(*inputs)
    torch.npu.synchronize()

    def test_fn():
        with torch.no_grad():
            _ = model(*inputs)
        torch.npu.synchronize()

    return test_fn


def find_profile_file(profile_path: str, filename: str) -> Optional[str]:
    for root, _, files in os.walk(profile_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def cleanup_profile_path(profile_path: str) -> None:
    if os.path.exists(profile_path):
        shutil.rmtree(profile_path, ignore_errors=True)


# ============================================================================
# 性能分析逻辑
# ============================================================================

def _find_profile_file(profile_path: str, filename: str) -> Optional[str]:
    """在 profile_path 下递归查找精确文件名的路径。"""
    for root, _, files in os.walk(profile_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def _find_profile_file_prefix(profile_path: str, prefix: str, suffix: str) -> Optional[str]:
    """在 profile_path 下递归查找文件名以 prefix 开头、suffix 结尾的路径。"""
    for root, _, files in os.walk(profile_path):
        for f in files:
            if f.startswith(prefix) and f.endswith(suffix):
                return os.path.join(root, f)
    return None


def _aggregate_kernel_rows(name: str, durations, active_count: int) -> Tuple[float, int]:
    """做法 B：以中位数为基准剔除坏行，再按迭代段内求和、跨迭代取均值。

    kernel_details.csv 与 msprof 旧格式（op_summary / task_time）共用的聚合逻辑：
    - 识别坏行：阈值 = max(median × BAD_ROW_K, BAD_ROW_FLOOR_US)，超阈值行剔除，
      以中位数为基准天然规避 MAD≈0 的退化边界；
    - 聚合：剩余行按 active_count 迭代切段、段内直接求和（= 该迭代该 kernel 总耗时，
      与 framework 侧"实际总耗时=求和"口径一致），跨迭代取均值。
    返回 (单次调用平均耗时 avg_us, 发射次数 launch_count)。
    """
    if active_count <= 0 or durations is None or len(durations) == 0:
        return 0.0, 0
    launch_count_f = len(durations) / active_count
    launch_count = int(round(launch_count_f))
    if abs(launch_count_f - launch_count) > 0.01:
        logger.warning(
            "kernel %s 的发射次数 %.3f 非整数（CSV 混入了非 active 行或 forward 内条件分支）",
            name, launch_count_f,
        )
    if launch_count <= 0:
        return 0.0, 0

    durations = np.asarray(durations, dtype=float)
    median_m = float(np.median(durations))
    thr = max(median_m * BAD_ROW_K, BAD_ROW_FLOOR_US)
    clean = durations[durations <= thr]
    removed = len(durations) - len(clean)
    if removed > 0:
        logger.warning(
            "kernel %s 剔除 %d 个异常采样（阈值 %.2fus，中位数 %.2fus）",
            name, removed, thr, median_m,
        )
    n = len(clean)
    n -= n % active_count  # 只保留末尾可整除段，避免剔除坏行后 reshape 错位
    if n >= active_count:
        iter_sums = clean[-n:].reshape(active_count, -1).sum(axis=1)
        avg_us = float(iter_sums.mean())
    else:
        avg_us = float(clean.sum()) / active_count
    return avg_us, launch_count


def _parse_msprof_fallback(profile_path: str, active_count: int
                           ) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[float]]:
    """kernel_details.csv 缺失时回退解析 msprof 旧格式（mindstudio_profiler_output）。

    优先 op_summary_*.csv（Op Name / Task Duration(us)），其次 task_time_*.csv
    （kernel_name / task_time(us)）。窗口语义与 kernel_details 一致：只覆盖 active
    阶段，每 Name 行数 ≈ 发射次数 × active_count，可复用 _aggregate_kernel_rows。
    返回 (operators, total_ms)；格式缺失或不可解析时返回 (None, None)。
    """
    import pandas as pd

    op_summary_file = _find_profile_file_prefix(profile_path, "op_summary_", ".csv")
    task_time_file = _find_profile_file_prefix(profile_path, "task_time_", ".csv")

    operator_times = {}
    source = None
    try:
        if op_summary_file:
            source = os.path.basename(op_summary_file)
            df = pd.read_csv(op_summary_file)
            for name, group in df.groupby("Op Name"):
                avg_us, launch_count = _aggregate_kernel_rows(
                    name, group["Task Duration(us)"].to_numpy(), active_count)
                operator_times[name] = {"avg_us": avg_us, "launch_count": launch_count}
        elif task_time_file:
            source = os.path.basename(task_time_file)
            df = pd.read_csv(task_time_file)
            for name, group in df.groupby("kernel_name"):
                avg_us, launch_count = _aggregate_kernel_rows(
                    name, group["task_time(us)"].to_numpy(), active_count)
                operator_times[name] = {"avg_us": avg_us, "launch_count": launch_count}
    except Exception as e:
        logger.warning("msprof 旧格式回退解析失败 %s: %s: %s", profile_path, type(e).__name__, e)
        return None, None

    if not operator_times:
        logger.warning("msprof 旧格式回退解析为空（无有效 kernel 数据）: %s", profile_path)
        return None, None

    total_avg_ms = sum(v["avg_us"] for v in operator_times.values()) / 1e3
    logger.info(
        "kernel_details.csv 缺失，已回退 msprof 旧格式解析（%s，%d 个 kernel，共 %.4f ms）",
        source, len(operator_times), total_avg_ms,
    )
    return operator_times, round(total_avg_ms, 4)


def parse_operator_latency(
    profile_path: str, active_count: int, keep_res: bool = False,
) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[float]]:
    """从 kernel_details.csv 提取算子时延，计算每次 test_fn() 调用的平均总耗时（毫秒）。

    兼容一个 forward 内多次启动同一 kernel 的场景：
    - kernel_details.csv 只含 active 阶段的行（schedule 的 warmup 阶段不进入采集），
      每个 Name 行数 = 发射次数 L × active_count，据此反推 L，
      避免把多发射 kernel 的单次调用耗时低估为 1/L；
    - 每组先用做法 B 剔除异常采样（阈值 max(median×K, FLOOR)），再按迭代段内求和、
      跨迭代取均值，避免个别极端假采样污染聚合值；
    - kernel_details.csv 缺失或解析失败时回退 msprof 旧格式（op_summary / task_time）；
    - 所有失败路径的 cleanup 均受 keep_res 控制，保留失败证据。
    返回 operators 为 {Name: {"avg_us": 单次调用耗时(us), "launch_count": 发射次数 L}}。
    """
    import pandas as pd

    # 1) 主路径：kernel_details.csv
    kernel_details_file = _find_profile_file(profile_path, "kernel_details.csv")
    if kernel_details_file and os.path.exists(kernel_details_file):
        try:
            df = pd.read_csv(kernel_details_file)
            operator_times = {}
            for name, group in df.groupby("Name"):
                avg_us, launch_count = _aggregate_kernel_rows(
                    name, group["Duration(us)"].to_numpy(), active_count)
                operator_times[name] = {"avg_us": avg_us, "launch_count": launch_count}
            total_avg_ms = sum(v["avg_us"] for v in operator_times.values()) / 1e3
        except Exception as e:
            logger.warning("kernel_details.csv 解析失败 %s: %s: %s", kernel_details_file, type(e).__name__, e)
        else:
            if not keep_res:
                cleanup_profile_path(profile_path)
            return operator_times, round(total_avg_ms, 4)

    # 2) 回退：msprof 旧格式（模式 B 型采集失败时可恢复数据）
    operators, total_avg_ms = _parse_msprof_fallback(profile_path, active_count)
    if operators is not None:
        if not keep_res:
            cleanup_profile_path(profile_path)
        return operators, total_avg_ms

    # 3) 彻底失败：cleanup 受 keep_res 控制（修复无条件清理导致证据丢失的缺陷）
    if not keep_res:
        cleanup_profile_path(profile_path)
    return None, None


def run_profiler_with_config(test_fn: callable, warmup: int, repeats: int, profile_name: str,
                             clear_l2_cache: bool = False) -> str:
    """运行NPU profiler并返回生成的性能分析目录路径。

    schedule 的 warmup 阶段执行 warmup 次预热（不进入采集）；
    active 阶段只采集 repeats 次调用（每迭代 prof.step() 推进），
    因此 kernel_details.csv 只含 repeats 次采集行，不含 warmup 行。
    """
    import torch
    import torch_npu
    import triton.runtime as runtime

    # experimental_config 按 testing.py 方式构造
    # _ExperimentalConfig 为 torch_npu.profiler 暴露的实验配置入口，经 getattr 取用
    experimental_config_cls = getattr(torch_npu.profiler, "_ExperimentalConfig")
    experimental_config = experimental_config_cls(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )

    # profile_path 保持 benchmark.py 当前方式
    timestamp = int(time.time() * 1000)
    profile_path = os.path.join(os.getcwd(), f"{profile_name}_{timestamp}")

    # clear_l2_cache 支持
    if clear_l2_cache:
        buffer = runtime.driver.active.get_empty_cache_for_benchmark()
        buffer = buffer.float()
        buffer.sum()
        torch.npu.synchronize()

    # schedule：warmup 阶段不采集，active 阶段只采集 repeats 次调用；
    # 总 step 数 = warmup + repeats（wait=0、repeat=1 时的推荐公式 warmup + active）
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup, active=repeats, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(warmup + repeats):
            if clear_l2_cache:
                buffer.sum()
                torch.npu.synchronize()
            test_fn()
            torch.npu.synchronize()
            prof.step()

    if clear_l2_cache:
        del buffer

    return profile_path


def measure_single(
        ctx: MeasureContext,
        clear_l2_cache: bool = False,
        keep_res: bool = False,
) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[float], float]:
    """测量单次性能（warmup + profiling）"""
    import torch
    import torch_npu  # noqa: F401

    torch.npu.reset_peak_memory_stats()
    test_fn = prepare_model_fn(ctx.model, ctx.inputs, ctx.device)

    profile_path = run_profiler_with_config(
        test_fn, ctx.warmup, ctx.repeats, ctx.profile_name,
        clear_l2_cache=clear_l2_cache,
    )
    operators, latency_ms = parse_operator_latency(profile_path, ctx.repeats, keep_res=keep_res)

    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)
    return operators, latency_ms, round(peak_memory, 2)


# ============================================================================
# 主测试逻辑
# ============================================================================

def _move_inputs_to_device(inputs: List[Any], device: Any) -> List[Any]:
    """把张量类输入搬到目标 device（含嵌套 list/tuple），标量原样透传。

    复用 _common_utils.move_to_device 处理嵌套结构，从而与 verify._move_to_device
    保持一致：list/tuple 都会递归迁移并保留各自类型。顶层 inputs 始终为 list，
    因此返回值仍为 List[Any]。
    """
    return [move_to_device(x, device) for x in inputs]


def _measure_framework(
    framework_model: Any,
    inputs: List[Any],
    config: BenchmarkConfig,
    device: Any,
    case_idx: int,
) -> Tuple[Dict[str, Dict[str, Any]], Optional[float], float]:
    """测量 framework 端时延 / 显存 / 算子分项；skip_framework 时直接返回参考值。"""
    if config.skip_framework:
        logger.info("    跳过 Framework 测试，使用参考延迟: %.4f ms", config.framework_latency_ms)
        return {}, config.framework_latency_ms, 0.0

    inputs_framework = _move_inputs_to_device(inputs, device)
    logger.info("    测试 Framework (warmup=%d, active=%d)...", config.warmup, config.repeats)
    operators, latency_ms, peak_memory = measure_single(
        MeasureContext(
            model=framework_model,
            inputs=inputs_framework,
            warmup=config.warmup,
            repeats=config.repeats,
            profile_name=f"framework_profile_case{case_idx}",
            device=device,
        ),
        clear_l2_cache=config.clear_l2_cache,
        keep_res=config.keep_res,
    )
    return operators or {}, latency_ms, peak_memory


def _measure_impl(
    impl_model: Any,
    inputs_impl: List[Any],
    config: BenchmarkConfig,
    device: Any,
    case_idx: int,
) -> Tuple[Dict[str, Dict[str, Any]], Optional[float], float]:
    """测量 impl 端时延 / 显存 / 算子分项。"""
    logger.info("    测试 Implementation (warmup=%d, active=%d)...", config.warmup, config.repeats)
    operators, latency_ms, peak_memory = measure_single(
        MeasureContext(
            model=impl_model,
            inputs=inputs_impl,
            warmup=config.warmup,
            repeats=config.repeats,
            profile_name=f"impl_profile_case{case_idx}",
            device=device,
        ),
        clear_l2_cache=config.clear_l2_cache,
        keep_res=config.keep_res,
    )
    return operators or {}, latency_ms, peak_memory


def _compute_speedup(framework_latency_ms: float, impl_latency_ms: float) -> float:
    """framework / impl 都为正时按比值算 speedup，否则 0。"""
    if impl_latency_ms > 0 and framework_latency_ms > 0:
        return framework_latency_ms / impl_latency_ms
    return 0


def _build_perf_pair(
    framework: Measurement,
    impl: Measurement,
) -> Tuple[PerformanceResult, PerformanceResult]:
    """把 framework / impl 的三元测量结果打包成两份 PerformanceResult。"""
    return (
        PerformanceResult(
            avg_latency_ms=round(framework.latency_ms, 4),
            peak_memory_mb=round(framework.peak_memory, 2),
            operators=framework.operators,
        ),
        PerformanceResult(
            avg_latency_ms=round(impl.latency_ms, 4),
            peak_memory_mb=round(impl.peak_memory, 2),
            operators=impl.operators,
        ),
    )


class ProfilerCollectError(RuntimeError):
    """profiler 采集/导出失败：无法从 profile 输出提取有效时延数据。

    与真实性能失败区分：采集通道在高频会话下偶发失效（EventQueue 为空 /
    导出中断），不应与性能回归混为一谈，便于重测与统计。
    """


def run_single_benchmark(
    models: ModelPair,
    inputs: List[Any],
    config: BenchmarkConfig,
    device: Any,
    case_ctx: CaseContext,
) -> Tuple[PerformanceResult, PerformanceResult, float]:
    """对单组输入进行性能测试（支持跳过 framework 测试）。

    Returns:
        (framework_result, implementation_result, speedup)
    """
    case_idx = case_ctx.case_idx
    total_cases = case_ctx.total_cases
    logger.info("  测试第 %d/%d 组输入...", case_idx, total_cases)

    inputs_impl = _move_inputs_to_device(inputs, device)

    framework_operators, framework_latency_ms, framework_peak_memory = _measure_framework(
        models.framework, inputs, config, device, case_idx,
    )
    impl_operators, impl_latency_ms, impl_peak_memory = _measure_impl(
        models.impl, inputs_impl, config, device, case_idx,
    )

    if (not config.skip_framework and framework_latency_ms is None) or impl_latency_ms is None:
        raise ProfilerCollectError(
            f"[用例 {case_idx}/{total_cases}] PROFILER_COLLECT_FAIL: 无法从 profiler 提取有效时延数据"
        )

    speedup = _compute_speedup(framework_latency_ms, impl_latency_ms)
    fw_perf, impl_perf = _build_perf_pair(
        Measurement(framework_operators, framework_latency_ms, framework_peak_memory),
        Measurement(impl_operators, impl_latency_ms, impl_peak_memory),
    )
    return fw_perf, impl_perf, round(speedup, 4)


def classify_speedup(s: Any) -> str:
    """对单个 shape 的 speedup 值分类。

    判定优先级：none → nan → inf → negative → zero → valid

    Returns:
        "none" | "nan" | "inf" | "negative" | "zero" | "valid"
    """
    if s is None:
        return "none"
    if not isinstance(s, (int, float)):
        return "none"
    if math.isnan(s):
        return "nan"
    if math.isinf(s):
        return "inf"
    if s < 0:
        return "negative"
    if s == 0:
        return "zero"
    return "valid"


def _classify_passed_results(passed) -> SpeedupBuckets:
    """按 classify_speedup 把通过的 shape 分桶。"""
    buckets = SpeedupBuckets()
    for r in passed:
        category = classify_speedup(r.speedup_vs_torch)
        if category == "valid":
            buckets.valid_speedups.append(r.speedup_vs_torch)
        elif category == "nan":
            buckets.nan_indices.append(r.case_idx)
        elif category == "inf":
            buckets.inf_indices.append(r.case_idx)
        elif category == "negative":
            buckets.negative_indices.append(r.case_idx)
        elif category == "zero":
            buckets.zero_indices.append(r.case_idx)
        else:  # "none"
            buckets.none_indices.append(r.case_idx)
    return buckets


def _geomean(values: List[float]) -> float:
    """对数域几何平均（所有值必须为正）；空列表返回 0。"""
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def _aggregate_perf(passed) -> PerfAggregate:
    """聚合通过的 shape 的平均延时 / 显存 / 算子分项耗时。

    framework / impl 的总延时使用几何平均（对数域，仅计入正值）；
    显存与算子分项仍使用算术平均。launch_count 是 per-shape 属性，不跨 shape 聚合，
    由 per_shape_results 各自记录。
    """
    n = len(passed)
    avg_fw = _geomean([r.framework.avg_latency_ms for r in passed])
    avg_impl = _geomean([r.implementation.avg_latency_ms for r in passed])
    avg_fw_mem = sum(r.framework.peak_memory_mb for r in passed) / n
    avg_impl_mem = sum(r.implementation.peak_memory_mb for r in passed) / n

    fw_ops: Dict[str, float] = {}
    impl_ops: Dict[str, float] = {}
    for r in passed:
        for op, t in r.framework.operators.items():
            fw_ops[op] = fw_ops.get(op, 0) + t["avg_us"]
        for op, t in r.implementation.operators.items():
            impl_ops[op] = impl_ops.get(op, 0) + t["avg_us"]
    return PerfAggregate(avg_fw, avg_impl, avg_fw_mem, avg_impl_mem, fw_ops, impl_ops, n)


def _geomean_speedup(valid_speedups):
    """对数域几何平均，空列表返回 None。"""
    if not valid_speedups:
        return None
    return round(
        math.exp(sum(math.log(s) for s in valid_speedups) / len(valid_speedups)),
        4,
    )


def compute_overall(results: List[SingleShapeResult]) -> OverallAggregate:
    """基于通过的 shape 做几何平均聚合。

    异常 shape（speedup 为 None/NaN/Inf/负数/0）不进入几何平均，
    但其 case_idx 收集到对应类别列表中，供报告展示。
    异常 shape 仍计入 passed_cases（算子功能正常，只是测不准）。

    Returns:
        OverallAggregate；全部 shape 均失败时，framework/implementation/speedup 为 None，
        索引列表为空。
    """
    passed = [r for r in results if r.status == "pass" and r.framework and r.implementation]
    buckets = _classify_passed_results(passed)

    if not passed:
        return OverallAggregate(
            framework=None,
            implementation=None,
            speedup_vs_torch=None,
            nan_indices=buckets.nan_indices,
            inf_indices=buckets.inf_indices,
            zero_indices=buckets.zero_indices,
            negative_indices=buckets.negative_indices,
            none_indices=buckets.none_indices,
        )

    agg = _aggregate_perf(passed)
    overall_speedup = _geomean_speedup(buckets.valid_speedups)

    return OverallAggregate(
        framework=PerformanceResult(
            avg_latency_ms=round(agg.avg_fw, 4),
            peak_memory_mb=round(agg.avg_fw_mem, 2),
            operators={k: {"avg_us": round(v / agg.n, 4)} for k, v in agg.fw_ops.items()},
        ),
        implementation=PerformanceResult(
            avg_latency_ms=round(agg.avg_impl, 4),
            peak_memory_mb=round(agg.avg_impl_mem, 2),
            operators={k: {"avg_us": round(v / agg.n, 4)} for k, v in agg.impl_ops.items()},
        ),
        speedup_vs_torch=overall_speedup,
        nan_indices=buckets.nan_indices,
        inf_indices=buckets.inf_indices,
        zero_indices=buckets.zero_indices,
        negative_indices=buckets.negative_indices,
        none_indices=buckets.none_indices,
    )


def _load_benchmark_modules(config: BenchmarkConfig):
    """导入 framework / impl 模块；返回三元组 (FrameworkModel, ModelNew, get_init_inputs)。"""
    sys.path.insert(0, config.verify_dir)
    torch_module = importlib.import_module(f"{config.op_name}_torch")
    impl_module = importlib.import_module(f"{config.op_name}_{config.triton_impl_name}")
    return torch_module.Model, impl_module.ModelNew, torch_module.get_init_inputs


def _instantiate_bench_models(framework_cls, impl_cls, get_init_inputs, device):
    """同种子分别实例化 framework 与 impl 模型。"""
    import torch
    init_params = get_init_inputs()
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    framework_model = framework_cls(*init_params).to(device)
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    impl_model = impl_cls(*init_params).to(device)
    return framework_model, impl_model


def _safe_del_model(name, model_ref):
    """记录模型是否未创建；仅作为调试提示。"""
    if model_ref is None:
        # 例外：try 体内早期失败时变量未定义/为 None，删除无意义；仅记录调试信息
        logger.debug("%s 未创建，无需删除", name)


def _bench_once(config, model_spec: BenchmarkModelSpec,
                inputs, device, case_ctx: CaseContext) -> SingleShapeResult:
    """跑一次 benchmark（不含重试）；成功返回 pass 结果，失败向上抛异常。"""
    framework_model = None
    impl_model = None
    try:
        framework_model, impl_model = _instantiate_bench_models(
            model_spec.framework_cls, model_spec.impl_cls, model_spec.get_init_inputs, device,
        )
        fw_perf, impl_perf, speedup = run_single_benchmark(
            ModelPair(framework_model, impl_model), inputs, config, device, case_ctx,
        )
        return SingleShapeResult(
            case_idx=case_ctx.case_idx,
            input_desc=describe_input(inputs),
            status="pass",
            framework=fw_perf,
            implementation=impl_perf,
            speedup_vs_torch=speedup,
        )
    finally:
        _safe_del_model("framework_model", framework_model)
        _safe_del_model("impl_model", impl_model)
        framework_model = None
        impl_model = None
        cleanup_npu_memory()


def _log_bench_retry(config, case_ctx: CaseContext, attempt, error_type, exc):
    """非最后一次尝试时打印重试日志。"""
    if attempt >= config.max_retries - 1:
        return
    logger.warning(
        "  [用例 %d/%d] 第 %d/%d 次尝试失败: %s: %s，将重试...",
        case_ctx.case_idx, case_ctx.total_cases, attempt + 1,
        config.max_retries, error_type, exc,
    )


def _run_shape_case(config, model_spec: BenchmarkModelSpec,
                    inputs, device, case_ctx: CaseContext) -> SingleShapeResult:
    """执行单个 shape 的 benchmark；失败时重试，最终仍失败则返回 status=fail 的结果。"""
    last_error_type = None
    last_error_msg = None

    for attempt in range(config.max_retries):
        try:
            return _bench_once(config, model_spec, inputs, device, case_ctx)
        except ProfilerCollectError as e:
            last_error_type = "PROFILER_COLLECT_FAIL"
            last_error_msg = traceback.format_exc()
            _log_bench_retry(config, case_ctx, attempt, last_error_type, e)
        except Exception as e:
            last_error_type = type(e).__name__
            last_error_msg = traceback.format_exc()
            _log_bench_retry(config, case_ctx, attempt, last_error_type, e)

    logger.error(
        "  [用例 %d/%d] 失败（已重试 %d 次）: %s",
        case_ctx.case_idx, case_ctx.total_cases, config.max_retries, last_error_type,
    )
    return SingleShapeResult(
        case_idx=case_ctx.case_idx,
        input_desc=describe_input(inputs),
        status="fail",
        error_type=last_error_type,
        error_msg=truncate_error(last_error_msg),
    )


def _run_preflight_gate(config: BenchmarkConfig) -> Tuple[Dict[str, Any], Optional[BenchmarkResult]]:
    """执行 NPU preflight 并落盘 npu_preflight.json。

    ready 时返回 (preflight, None)；未 ready 时返回 (preflight, 阻塞态 B 类 BenchmarkResult)。
    """
    preflight = run_preflight(**load_preflight_options())
    write_preflight_result(preflight, os.path.join(config.verify_dir, "npu_preflight.json"))
    if preflight["status"] == "ready":
        return preflight, None
    return preflight, BenchmarkResult(
        op_name=config.op_name,
        warmup=config.warmup,
        repeats=config.repeats,
        max_retries=config.max_retries,
        framework=None,
        implementation=None,
        speedup_vs_torch=None,
        total_cases=0,
        failed_cases=0,
        failure_class="B",
        npu_preflight=preflight,
    )


def _benchmark_all_shapes(
    config: BenchmarkConfig,
    model_spec: BenchmarkModelSpec,
    input_groups: List[List[Any]],
    device: Any,
) -> List[SingleShapeResult]:
    """逐 shape 执行 benchmark，返回全量 per-shape 结果（含失败用例）。"""
    total_cases = len(input_groups)
    return [
        _run_shape_case(
            config, model_spec,
            inputs, device, CaseContext(case_idx=case_idx, total_cases=total_cases),
        )
        for case_idx, inputs in enumerate(input_groups, start=1)
    ]


def _assemble_result(
    config: BenchmarkConfig,
    preflight: Dict[str, Any],
    per_shape_results: List[SingleShapeResult],
) -> BenchmarkResult:
    """由 per-shape 结果聚合出完整 BenchmarkResult（几何平均 + 异常索引分类）。"""
    total_cases = len(per_shape_results)
    passed_cases = sum(1 for r in per_shape_results if r.status == "pass")
    overall = compute_overall(per_shape_results)

    return BenchmarkResult(
        op_name=config.op_name,
        warmup=config.warmup,
        repeats=config.repeats,
        max_retries=config.max_retries,
        framework=overall.framework,
        implementation=overall.implementation,
        speedup_vs_torch=overall.speedup_vs_torch,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=total_cases - passed_cases,
        nan_indices=overall.nan_indices,
        inf_indices=overall.inf_indices,
        zero_indices=overall.zero_indices,
        negative_indices=overall.negative_indices,
        none_indices=overall.none_indices,
        per_shape_results=per_shape_results,
        npu_preflight=preflight,
    )


def benchmark_implementations(config: BenchmarkConfig) -> BenchmarkResult:
    """执行完整的性能测试，支持多组输入。每个 shape 独立 try/except。"""
    preflight, blocked_result = _run_preflight_gate(config)
    if blocked_result is not None:
        return blocked_result

    import torch
    import torch_npu  # noqa: F401

    device = torch.device("npu")

    input_groups = resolve_inputs(config.op_name, config.verify_dir)
    model_spec = BenchmarkModelSpec(*_load_benchmark_modules(config))

    per_shape_results = _benchmark_all_shapes(config, model_spec, input_groups, device)
    return _assemble_result(config, preflight, per_shape_results)


def _perf_dict(p: PerformanceResult, with_launch: bool) -> Dict[str, Any]:
    """序列化单次性能结果；with_launch=True 时 operators 附带 launch_count（per-shape）。"""
    return {
        "avg_latency_ms": p.avg_latency_ms,
        "peak_memory_mb": p.peak_memory_mb,
        "operators": {
            name: (
                {"avg_us": round(item["avg_us"], 4), "launch_count": item["launch_count"]}
                if with_launch
                else {"avg_us": round(item["avg_us"], 4)}
            )
            for name, item in p.operators.items()
        },
    }


def _perf_to_dict(p: Optional[PerformanceResult]) -> Optional[Dict[str, Any]]:
    """顶层聚合结果序列化：operators 只含 avg_us（launch_count 不跨 shape 聚合）。"""
    if p is None:
        return None
    return _perf_dict(p, with_launch=False)


def _normalize_shape_speedup(s: Optional[float]) -> Optional[float]:
    """落盘前规范单 shape speedup：异常值统一写为 None（JSON 中即 null）。"""
    return s if classify_speedup(s) == "valid" else None


def result_to_dict(result: BenchmarkResult) -> Dict[str, Any]:
    """将 BenchmarkResult 转换为字典格式。"""
    base_dict: Dict[str, Any] = {
        "op_name": result.op_name,
        "warmup": result.warmup,
        "repeats": result.repeats,
        "max_retries": result.max_retries,
        "total_cases": result.total_cases,
        "passed_cases": result.passed_cases,
        "failed_cases": result.failed_cases,
        "nan_indices": result.nan_indices,
        "inf_indices": result.inf_indices,
        "zero_indices": result.zero_indices,
        "negative_indices": result.negative_indices,
        "none_indices": result.none_indices,
        "framework": _perf_to_dict(result.framework),
        "implementation": _perf_to_dict(result.implementation),
        "speedup_vs_torch": result.speedup_vs_torch,
        "npu_preflight": result.npu_preflight,
        "failure_class": result.failure_class,
    }

    # per_shape_results 保留全量（含失败用例），带 status 列；
    # 异常 speedup（NaN/Inf/0/负数/None）落盘为 null；
    # operators 带 launch_count（per-shape 属性，逐 shape 精确记录）
    base_dict["per_shape_results"] = [
        {
            "case_idx": r.case_idx,
            "input_desc": r.input_desc,
            "status": r.status,
            "framework": _perf_dict(r.framework, with_launch=True) if r.framework else None,
            "implementation": _perf_dict(r.implementation, with_launch=True) if r.implementation else None,
            "speedup_vs_torch": _normalize_shape_speedup(r.speedup_vs_torch),
            "error_type": r.error_type,
            "error_msg": r.error_msg,
        }
        for r in result.per_shape_results
    ]

    return base_dict


# ============================================================================
# 命令行入口
# ============================================================================

VERIFY_GATE_FAILURES_TO_PRINT = 5
VERIFY_GATE_EXIT_CODE = 2


class VerifyGateError(Exception):
    """L1 闸门未通过时抛出，由 main() 统一捕获并退出，避免在内部函数中调用 sys.exit。"""

    def __init__(self, message: str = "", exit_code: int = VERIFY_GATE_EXIT_CODE):
        super().__init__(message)
        self.exit_code = exit_code


def resolve_verify_json_name(triton_impl_name: str) -> str:
    """按 impl_name 推导 verify_result json 文件名。

    - triton_ascend_impl（默认）→ verify_result.json（Phase 3）
    - triton_baseline / triton_optimized → verify_result_{suffix}.json（Phase 4）
    - 其他自定义名 → verify_result_{name 去掉 triton_ 前缀}.json
    """
    if triton_impl_name == TRITON_IMPL_NAME_DEFAULT:
        return "verify_result.json"
    suffix = triton_impl_name
    if suffix.startswith("triton_"):
        suffix = suffix[len("triton_"):]
    return f"verify_result_{suffix}.json"


def _load_verify_json(verify_json_path, triton_impl_name):
    """读取 verify_result.json；缺失或读取失败抛出 VerifyGateError 由 main 处理。"""
    if not os.path.isfile(verify_json_path):
        logger.error(
            "[L1 闸门] 拒绝执行 benchmark：未找到 verify_result 文件\n"
            "  expected: %s\n"
            "  triton_impl_name: %s\n"
            "  请先运行 verify.py，或在确实不需要精度校验的场景下传 --verify_not_required",
            verify_json_path,
            triton_impl_name,
        )
        raise VerifyGateError("verify_result 文件不存在")

    try:
        with open(verify_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(
            "[L1 闸门] 拒绝执行 benchmark：verify_result 文件读取失败\n"
            "  path: %s\n"
            "  error: %s: %s",
            verify_json_path,
            type(e).__name__,
            e,
        )
        raise VerifyGateError("verify_result 文件读取失败") from e


def _emit_gate_failures(failures):
    """打印 verify failures 摘要。"""
    if not failures:
        return
    logger.error(
        "  前 %d 条 failures（共 %d 条）：",
        min(VERIFY_GATE_FAILURES_TO_PRINT, len(failures)),
        len(failures),
    )
    for f_item in failures[:VERIFY_GATE_FAILURES_TO_PRINT]:
        logger.error(
            "    - case_idx=%s error_type=%s input_desc=%s",
            f_item.get("case_idx"),
            f_item.get("error_type"),
            f_item.get("input_desc"),
        )


def check_verify_gate(verify_dir: str, triton_impl_name: str) -> None:
    """L1 闸门：benchmark 启动前必须确认对应 verify_result 全过。

    不通过时抛出 VerifyGateError（由 main 捕获并以非零码退出），
    stderr 打印路径 / 计数 / failures 摘要，便于上游 agent 把错误等价映射到
    verify 失败处理路径。
    """
    verify_json_name = resolve_verify_json_name(triton_impl_name)
    verify_json_path = os.path.join(verify_dir, verify_json_name)

    verify_data = _load_verify_json(verify_json_path, triton_impl_name)

    total = verify_data.get("total_cases", 0)
    passed = verify_data.get("passed_cases", 0)
    failures = verify_data.get("failures", []) or []

    if total == 0:
        logger.error(
            "[L1 闸门] 拒绝执行 benchmark：verify_result 中 total_cases=0\n"
            "  path: %s\n"
            "  说明 verify.py 未实际跑任何 shape，benchmark 无意义",
            verify_json_path,
        )
        raise VerifyGateError("verify_result total_cases=0")

    if passed != total:
        logger.error(
            "[L1 闸门] 拒绝执行 benchmark：精度校验未全通过\n"
            "  path: %s\n"
            "  passed_cases: %d/%d\n"
            "  triton_impl_name: %s",
            verify_json_path,
            passed,
            total,
            triton_impl_name,
        )
        _emit_gate_failures(failures)
        raise VerifyGateError("verify_result 未全部通过")


def _build_argparser():
    parser = argparse.ArgumentParser(description="性能测试脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument("--verify_dir", default=".", help="验证目录路径（默认当前目录）")
    parser.add_argument("--triton_impl_name", default=TRITON_IMPL_NAME_DEFAULT,
                       help="Triton 实现模块名")
    parser.add_argument("--warmup", type=int, default=WARMUP_DEFAULT, help="warmup 次数（默认 5）")
    parser.add_argument("--repeats", type=int, default=REPEATS_DEFAULT, help="正式测试次数（默认 5）")
    parser.add_argument("--output", help="输出文件路径（JSON 格式）")
    parser.add_argument("--skip_framework", action="store_true",
                       help="跳过 framework 性能测试（GPU Kernel 模式使用）")
    parser.add_argument("--framework_latency_ms", type=float, default=0.0,
                       help="预设的 framework 参考延迟（毫秒），用于计算 speedup")
    parser.add_argument("--verify_not_required", action="store_true",
                       help="跳过 L1 verify 闸门（默认强制要求 verify_result 全过）")
    parser.add_argument("--clear_l2_cache", action="store_true",
                       help="每次迭代前清除 L2 cache")
    parser.add_argument("--keep_res", action="store_true",
                       help="保留 profiling 结果目录（默认清理）")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="采集失败时的最大重试次数（默认 3）")
    parser.add_argument("--lock-frequency", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="性能测试前对 NPU 锁频并在测试期间监控频率（默认开启）")
    parser.add_argument("--lock-frequency-fail-action", default="warn",
                        choices=["warn", "error"],
                        help="锁频失败或频率漂移时的处理方式（默认 warn）")
    parser.add_argument("--freq-check-interval", type=float, default=1.0,
                        help="频率监控采样间隔，单位秒（默认 1.0）")
    return parser


def _emit_summary(result_dict):
    logger.info("\n性能测试结果:")
    logger.info("  通过率: %s/%s", result_dict["passed_cases"], result_dict["total_cases"])
    if result_dict["speedup_vs_torch"] is not None:
        logger.info("  框架实现 - 平均延迟: %.4f ms", result_dict["framework"]["avg_latency_ms"])
        logger.info("  生成实现 - 平均延迟: %.4f ms", result_dict["implementation"]["avg_latency_ms"])
        logger.info("  加速比 (几何平均): %.4fx", result_dict["speedup_vs_torch"])
    else:
        logger.info("  无可用加速比数据（全部 shape 失败或 speedup 异常）")

    excluded_total = (
        len(result_dict["nan_indices"]) + len(result_dict["inf_indices"])
        + len(result_dict["zero_indices"]) + len(result_dict["negative_indices"])
        + len(result_dict["none_indices"])
    )
    if excluded_total > 0:
        logger.info(
            "  异常 shape (不计入几何平均): "
            "nan=%s, inf=%s, zero=%s, neg=%s, none=%s",
            result_dict["nan_indices"],
            result_dict["inf_indices"],
            result_dict["zero_indices"],
            result_dict["negative_indices"],
            result_dict["none_indices"],
        )


def _save_or_print_result(result_dict, output_path):
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        logger.info("\n结果已保存到: %s", output_path)
    else:
        logger.info("\n结果:")
        logger.info("%s", json.dumps(result_dict, indent=2, ensure_ascii=False))


def _build_config(args, verify_dir):
    return BenchmarkConfig(
        op_name=args.op_name,
        verify_dir=verify_dir,
        triton_impl_name=args.triton_impl_name,
        warmup=args.warmup,
        repeats=args.repeats,
        skip_framework=args.skip_framework,
        framework_latency_ms=args.framework_latency_ms,
        clear_l2_cache=args.clear_l2_cache,
        keep_res=args.keep_res,
        max_retries=args.max_retries,
        lock_frequency=args.lock_frequency,
        lock_frequency_fail_action=args.lock_frequency_fail_action,
        freq_check_interval=args.freq_check_interval,
    )


def _check_gates(args, verify_dir):
    """L1 verify 闸门与基线完整性闸门。

    不通过时抛 VerifyGateError / BaselineGateError，由 main() 统一转成退出码，
    内部函数不直接调用 sys.exit。
    """
    if args.verify_not_required:
        logger.warning(
            "[L1 闸门] 已通过 --verify_not_required 跳过 verify 闸门检查 "
            "(triton_impl_name=%s)",
            args.triton_impl_name,
        )
    else:
        check_verify_gate(verify_dir, args.triton_impl_name)

    # Baseline gate: refuse to benchmark against a tampered baseline.
    # Exit 3 = anchor missing (Phase 1 freeze skipped); Exit 4 = baseline modified.
    if _check_baseline_integrity is not None:
        _check_baseline_integrity(verify_dir, args.op_name)


class FrequencyLockError(Exception):
    """锁频未达要求且策略为 error 时抛出，由 main() 统一捕获并退出。"""

    def __init__(self, message: str = "", exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


def _handle_lock_failure(config, lock_messages):
    """锁频未完全成功时按配置决定告警还是抛错终止。"""
    msg = (
        "NPU 锁频未完全成功；messages: "
        + "; ".join(f"dev{k}={v}" for k, v in lock_messages.items())
    )
    if config.lock_frequency_fail_action == "error":
        logger.error("[锁频] %s", msg)
        raise FrequencyLockError(msg)
    logger.warning("[锁频] %s", msg)


def _setup_frequency_monitor(config):
    """A5 性能测试前锁频，并返回频率监控上下文；未启用或无设备时返回 None。"""
    if not config.lock_frequency:
        return None
    if FrequencyMonitor is None:
        logger.warning("[锁频] 无法导入 lock_npu_frequency 模块，跳过锁频与监控")
        return None

    devices = detect_npu_devices()
    if not devices:
        logger.warning("[锁频] 未检测到 NPU 设备，跳过锁频与监控")
        return None

    logger.info("[锁频] 检测到 NPU 设备: %s", devices)
    lock_ok, _locked_devices, baseline_freqs, lock_messages = lock_npu_frequency(
        devices=devices, verify=True
    )
    if not lock_ok:
        _handle_lock_failure(config, lock_messages)
    return FrequencyMonitor(
        devices=devices,
        interval=config.freq_check_interval,
        baseline_freqs=baseline_freqs if baseline_freqs else None,
    )


def _run_and_emit(config, monitor_ctx, output):
    """执行 benchmark、落盘结果，返回进程退出码。"""
    if monitor_ctx is not None:
        with monitor_ctx:
            result = benchmark_implementations(config)
    else:
        result = benchmark_implementations(config)

    result_dict = result_to_dict(result)
    if result.failure_class == "B":
        result_dict["status"] = "blocked"
    _emit_summary(result_dict)
    _save_or_print_result(result_dict, output)

    # preflight 未 ready：B 类阻塞，直接以非零码返回，不再看频率漂移
    if result.failure_class == "B":
        return 1

    if monitor_ctx is not None and monitor_ctx.report.has_drift():
        logger.warning("%s", format_frequency_report(monitor_ctx.report))
        if config.lock_frequency_fail_action == "error":
            return 1
    return 0


def main():
    _setup_logger()
    args = _build_argparser().parse_args()

    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        logger.error("错误: 验证目录不存在: %s", verify_dir)
        sys.exit(1)

    try:
        _check_gates(args, verify_dir)
        config = _build_config(args, verify_dir)
        monitor_ctx = _setup_frequency_monitor(config)
    except (VerifyGateError, BaselineGateError, FrequencyLockError) as e:
        # 闸门/锁频未通过：以各自约定的退出码退出
        # （verify 闸门 2；基线 3=锚缺失、4=被篡改；锁频 1）
        sys.exit(e.exit_code)

    try:
        # 只要脚本正常跑完就 exit 0（由 Agent 读 JSON 判断）
        sys.exit(_run_and_emit(config, monitor_ctx, args.output))
    except Exception as e:
        if monitor_ctx is not None:
            logger.warning("%s", format_frequency_report(monitor_ctx.report))
        logger.error("性能测试失败: %s", e)
        logger.error("%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
