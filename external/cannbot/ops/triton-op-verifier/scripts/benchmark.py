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


# ============================================================================
# 日志配置
# ============================================================================

# 确保同目录下的 _log_utils 可被导入（脚本可能从其他工作目录调用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _log_utils import setup_logger as _setup_logger_shared  # noqa: E402	 
from _common_utils import describe_input as _describe_input_shared  # noqa: E402

logger = logging.getLogger("triton_op_verifier.benchmark")


def _setup_logger() -> None:
    """配置 logger：复用 _log_utils.setup_logger。"""
    _setup_logger_shared(logger)


# ============================================================================
# 配置常量
# ============================================================================

WARMUP_DEFAULT = 5
REPEATS_DEFAULT = 50
TRITON_IMPL_NAME_DEFAULT = "triton_ascend_impl"
ERROR_MSG_LIMIT = 2000


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
    """单次性能测试结果"""
    avg_latency_ms: float
    peak_memory_mb: float
    operators: Dict[str, float]


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
    nan_indices: List[int] = field(default_factory=list)
    inf_indices: List[int] = field(default_factory=list)
    zero_indices: List[int] = field(default_factory=list)
    negative_indices: List[int] = field(default_factory=list)
    none_indices: List[int] = field(default_factory=list)
    per_shape_results: List[SingleShapeResult] = field(default_factory=list)


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

def parse_operator_latency(profile_path: str, active_count: int) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """从 profiling 结果文件中提取算子时延数据，计算平均执行时间。"""
    import pandas as pd

    operator_details_file = find_profile_file(profile_path, "operator_details.csv")

    if not operator_details_file or not os.path.exists(operator_details_file):
        cleanup_profile_path(profile_path)
        return None, None

    try:
        df = pd.read_csv(operator_details_file)
    except Exception:
        cleanup_profile_path(profile_path)
        return None, None

    required_columns = ["Name", "Device Self Duration(us)"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        cleanup_profile_path(profile_path)
        return None, None

    if "Count" not in df.columns:
        return _parse_without_count(df, profile_path, active_count)

    return _parse_with_count(df, profile_path, active_count)


def _parse_without_count(
        df: Any, profile_path: str, active_count: int
) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    operator_avg_times = {}
    grouped = df.groupby("Name")["Device Self Duration(us)"].sum()
    for op_name_str, total_us in grouped.items():
        operator_avg_times[op_name_str] = total_us / active_count

    total_avg_us = sum(operator_avg_times.values())
    total_avg_ms = total_avg_us / 1000.0

    cleanup_profile_path(profile_path)
    return operator_avg_times, round(total_avg_ms, 4)


def _parse_with_count(
        df: Any, profile_path: str, active_count: int
) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    valid_ops = df[df["Count"] == active_count].copy()

    if valid_ops.empty:
        cleanup_profile_path(profile_path)
        return None, None

    operator_avg_times = {}
    grouped = valid_ops.groupby("Name")
    for op_name_str, group in grouped:
        total_us = group["Device Self Duration(us)"].sum()
        avg_us = total_us / active_count
        operator_avg_times[op_name_str] = avg_us

    total_avg_us = sum(operator_avg_times.values())
    total_avg_ms = total_avg_us / 1000.0

    cleanup_profile_path(profile_path)
    return operator_avg_times, round(total_avg_ms, 4)


def run_profiler_with_config(test_fn: callable, warmup: int, repeats: int, profile_name: str) -> str:
    """运行NPU profiler并返回生成的性能分析目录路径。"""
    import torch
    import torch_npu

    # 例外：torch_npu 未提供 _ExperimentalConfig 的公开等价物，
    # 上游官方示例同样以此方式配置 profiler，使用 getattr 间接访问以保持封装语义
    experimental_config_cls = getattr(torch_npu.profiler, "_ExperimentalConfig")
    experimental_config = experimental_config_cls(
        aic_metrics=None,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )

    test_fn()
    torch.npu.synchronize()

    skip_first = 1 + warmup
    total_steps = skip_first + repeats

    timestamp = int(time.time() * 1000)
    profile_path = os.path.join(os.getcwd(), f"{profile_name}_{timestamp}")

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=warmup, active=repeats, repeat=1, skip_first=skip_first
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total_steps):
            test_fn()
            prof.step()
            torch.npu.synchronize()

    return profile_path


def measure_single(
        ctx: MeasureContext,
) -> Tuple[Optional[Dict[str, float]], Optional[float], float]:
    """测量单次性能（warmup + profiling）"""
    import torch
    import torch_npu  # noqa: F401

    torch.npu.reset_peak_memory_stats()
    test_fn = prepare_model_fn(ctx.model, ctx.inputs, ctx.device)

    try:
        profile_path = run_profiler_with_config(test_fn, ctx.warmup, ctx.repeats, ctx.profile_name)
        operators, latency_ms = parse_operator_latency(profile_path, ctx.repeats)
    except Exception as e:
        logger.warning("torch_npu.profiler 获取数据失败: %s，使用兜底测试机制...", e)
        operators, latency_ms = None, None

    if operators is None or latency_ms is None or latency_ms <= 0.0001:
        logger.warning(
            "profiler 无法获取有效时延数据（当前:%s ms），将使用 time.perf_counter() 兜底...",
            latency_ms,
        )
        return measure_single_fallback(ctx)

    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)
    return operators, latency_ms, round(peak_memory, 2)


def measure_single_fallback(
        ctx: MeasureContext,
) -> Tuple[Optional[Dict[str, float]], Optional[float], float]:
    """使用time.perf_counter()的兜底测试机制"""
    import torch
    import torch_npu  # noqa: F401
    import statistics

    with torch.no_grad():
        for _ in range(ctx.warmup):
            _ = ctx.model(*ctx.inputs)
    torch.npu.synchronize()

    latencies = []
    for _ in range(ctx.repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = ctx.model(*ctx.inputs)
        torch.npu.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    avg_latency_ms = statistics.mean(latencies)
    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)
    return {}, round(avg_latency_ms, 4), round(peak_memory, 2)


# ============================================================================
# 主测试逻辑
# ============================================================================

def _move_inputs_to_device(inputs: List[Any], device: Any) -> List[Any]:
    """把张量类输入搬到目标 device，标量原样透传。"""
    import torch
    return [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]


def _measure_framework(
    framework_model: Any,
    inputs: List[Any],
    config: BenchmarkConfig,
    device: Any,
    case_idx: int,
) -> Tuple[Dict[str, float], Optional[float], float]:
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
        )
    )
    return operators or {}, latency_ms, peak_memory


def _measure_impl(
    impl_model: Any,
    inputs_impl: List[Any],
    config: BenchmarkConfig,
    device: Any,
    case_idx: int,
) -> Tuple[Dict[str, float], Optional[float], float]:
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
        )
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
        raise RuntimeError(
            f"[用例 {case_idx}/{total_cases}] 无法从 profiler 提取有效时延数据"
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


def _aggregate_perf(passed) -> PerfAggregate:
    """聚合通过的 shape 的平均延时 / 显存 / 算子分项耗时。"""
    n = len(passed)
    avg_fw = sum(r.framework.avg_latency_ms for r in passed) / n
    avg_impl = sum(r.implementation.avg_latency_ms for r in passed) / n
    avg_fw_mem = sum(r.framework.peak_memory_mb for r in passed) / n
    avg_impl_mem = sum(r.implementation.peak_memory_mb for r in passed) / n

    fw_ops: Dict[str, float] = {}
    impl_ops: Dict[str, float] = {}
    for r in passed:
        for op, t in r.framework.operators.items():
            fw_ops[op] = fw_ops.get(op, 0) + t
        for op, t in r.implementation.operators.items():
            impl_ops[op] = impl_ops.get(op, 0) + t
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
            operators={k: round(v / agg.n, 4) for k, v in agg.fw_ops.items()},
        ),
        implementation=PerformanceResult(
            avg_latency_ms=round(agg.avg_impl, 4),
            peak_memory_mb=round(agg.avg_impl_mem, 2),
            operators={k: round(v / agg.n, 4) for k, v in agg.impl_ops.items()},
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


def _run_shape_case(config, model_spec: BenchmarkModelSpec,
                    inputs, device, case_ctx: CaseContext) -> SingleShapeResult:
    """执行单个 shape 的 benchmark；失败时返回 status=fail 的结果。"""
    case_idx = case_ctx.case_idx
    total_cases = case_ctx.total_cases
    input_desc = describe_input(inputs)
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
            case_idx=case_idx,
            input_desc=input_desc,
            status="pass",
            framework=fw_perf,
            implementation=impl_perf,
            speedup_vs_torch=speedup,
        )
    except Exception as e:
        err_detail = traceback.format_exc()
        logger.error(
            "  [用例 %d/%d] 失败: %s: %s",
            case_idx, total_cases, type(e).__name__, e,
        )
        return SingleShapeResult(
            case_idx=case_idx,
            input_desc=input_desc,
            status="fail",
            error_type=type(e).__name__,
            error_msg=truncate_error(err_detail),
        )
    finally:
        _safe_del_model("framework_model", framework_model)
        _safe_del_model("impl_model", impl_model)
        framework_model = None
        impl_model = None
        cleanup_npu_memory()


def benchmark_implementations(config: BenchmarkConfig) -> BenchmarkResult:
    """执行完整的性能测试，支持多组输入。每个 shape 独立 try/except。"""
    import torch
    import torch_npu  # noqa: F401

    device = torch.device("npu")

    input_groups = resolve_inputs(config.op_name, config.verify_dir)
    total_cases = len(input_groups)

    framework_cls, impl_cls, get_init_inputs = _load_benchmark_modules(config)
    model_spec = BenchmarkModelSpec(framework_cls, impl_cls, get_init_inputs)

    per_shape_results: List[SingleShapeResult] = [
        _run_shape_case(
            config, model_spec,
            inputs, device, CaseContext(case_idx=case_idx, total_cases=total_cases),
        )
        for case_idx, inputs in enumerate(input_groups, start=1)
    ]

    passed_cases = sum(1 for r in per_shape_results if r.status == "pass")
    failed_cases = total_cases - passed_cases

    overall = compute_overall(per_shape_results)

    return BenchmarkResult(
        op_name=config.op_name,
        warmup=config.warmup,
        repeats=config.repeats,
        framework=overall.framework,
        implementation=overall.implementation,
        speedup_vs_torch=overall.speedup_vs_torch,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        nan_indices=overall.nan_indices,
        inf_indices=overall.inf_indices,
        zero_indices=overall.zero_indices,
        negative_indices=overall.negative_indices,
        none_indices=overall.none_indices,
        per_shape_results=per_shape_results,
    )


def _perf_to_dict(p: Optional[PerformanceResult]) -> Optional[Dict[str, Any]]:
    if p is None:
        return None
    return {
        "avg_latency_ms": p.avg_latency_ms,
        "peak_memory_mb": p.peak_memory_mb,
        "operators": {name: round(avg_us, 4) for name, avg_us in p.operators.items()},
    }


def _normalize_shape_speedup(s: Optional[float]) -> Optional[float]:
    """落盘前规范单 shape speedup：异常值统一写为 None（JSON 中即 null）。"""
    return s if classify_speedup(s) == "valid" else None


def result_to_dict(result: BenchmarkResult) -> Dict[str, Any]:
    """将 BenchmarkResult 转换为字典格式。"""
    base_dict: Dict[str, Any] = {
        "op_name": result.op_name,
        "warmup": result.warmup,
        "repeats": result.repeats,
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
    }

    # per_shape_results 保留全量（含失败用例），带 status 列；
    # 异常 speedup（NaN/Inf/0/负数/None）落盘为 null
    base_dict["per_shape_results"] = [
        {
            "case_idx": r.case_idx,
            "input_desc": r.input_desc,
            "status": r.status,
            "framework": (
                {
                    "avg_latency_ms": r.framework.avg_latency_ms,
                    "peak_memory_mb": r.framework.peak_memory_mb,
                } if r.framework else None
            ),
            "implementation": (
                {
                    "avg_latency_ms": r.implementation.avg_latency_ms,
                    "peak_memory_mb": r.implementation.peak_memory_mb,
                } if r.implementation else None
            ),
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
    parser.add_argument("--repeats", type=int, default=REPEATS_DEFAULT, help="正式测试次数（默认 50）")
    parser.add_argument("--output", help="输出文件路径（JSON 格式）")
    parser.add_argument("--skip_framework", action="store_true",
                       help="跳过 framework 性能测试（GPU Kernel 模式使用）")
    parser.add_argument("--framework_latency_ms", type=float, default=0.0,
                       help="预设的 framework 参考延迟（毫秒），用于计算 speedup")
    parser.add_argument("--verify_not_required", action="store_true",
                       help="跳过 L1 verify 闸门（默认强制要求 verify_result 全过）")
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
    )


def main():
    _setup_logger()
    args = _build_argparser().parse_args()

    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        logger.error("错误: 验证目录不存在: %s", verify_dir)
        sys.exit(1)

    if args.verify_not_required:
        logger.warning(
            "[L1 闸门] 已通过 --verify_not_required 跳过 verify 闸门检查 "
            "(triton_impl_name=%s)",
            args.triton_impl_name,
        )
    else:
        try:
            check_verify_gate(verify_dir, args.triton_impl_name)
        except VerifyGateError as e:
            sys.exit(e.exit_code)

    config = _build_config(args, verify_dir)

    try:
        result = benchmark_implementations(config)
        result_dict = result_to_dict(result)
        _emit_summary(result_dict)
        _save_or_print_result(result_dict, args.output)
        # 只要脚本正常跑完就 exit 0（由 Agent 读 JSON 判断）
        sys.exit(0)
    except Exception as e:
        logger.error("性能测试失败: %s", e)
        logger.error("%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
