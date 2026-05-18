#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layer Norm profiler：用例解析、op_statistic 汇总、torch_npu.profiler 单次采集。"""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch_npu

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# 与技能 ascendc-operator-performance-eval 一致：不可改为其它值
PROFILER_SCHEDULE_WARMUP = 5
PROFILER_SCHEDULE_ACTIVE = 5


def repo_root_from_here() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(16):
        if os.path.exists(os.path.join(here, "build.sh")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


def load_custom_library() -> str:
    try:
        import ascend_kernel  # noqa: F401

        return "import ascend_kernel"
    except Exception:
        root = repo_root_from_here()
        lib_pattern = os.path.join(
            root, "python", "ascend_kernel", "ascend_kernel", "lib", "*.so"
        )
        lib_files = glob.glob(lib_pattern)
        if not lib_files:
            cand = os.path.join(
                root,
                "python",
                "ascend_kernel",
                "ascend_kernel",
                "lib",
                "libascend_kernel.so",
            )
            if os.path.isfile(cand):
                torch.ops.load_library(cand)
                return cand
            raise FileNotFoundError(
                "ascend_kernel .so not found; build the project or set PYTHONPATH."
            )
        torch.ops.load_library(lib_files[0])
        return lib_files[0]


def load_cases(case_file: str) -> List[Dict[str, Any]]:
    if not str(case_file).endswith(".jsonl"):
        raise ValueError(f"性能用例仅支持 .jsonl：{case_file}")
    cases: List[Dict[str, Any]] = []
    with open(case_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_inputs(case: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    spec_map = {item["name"]: item for item in case["inputs"]}
    x_spec = spec_map["x"]
    norm_spec = spec_map["normalized_shape"]
    affine_spec = spec_map.get("use_affine", {"value": True})
    eps_spec = spec_map.get("eps", {"value": 1e-5})

    dt = DTYPE_MAP[x_spec["dtype"]]
    shape = tuple(x_spec["shape"])
    normalized_shape = tuple(norm_spec["value"])
    use_affine = bool(affine_spec.get("value", True))
    eps = float(eps_spec.get("value", 1e-5))

    x = torch.randn(shape, dtype=dt, device="cpu").to(device)
    if use_affine:
        weight = torch.randn(normalized_shape, dtype=dt, device="cpu").to(device)
        bias = torch.randn(normalized_shape, dtype=dt, device="cpu").to(device)
    else:
        weight = None
        bias = None

    return {
        "x": x,
        "normalized_shape": normalized_shape,
        "weight": weight,
        "bias": bias,
        "eps": eps,
    }


def _normalize_header(h: Optional[str]) -> str:
    if h is None:
        return ""
    return h.strip().lstrip("\ufeff")


def sum_total_time_us(csv_path: str) -> float:
    total = 0.0
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"empty csv header: {csv_path}")
        key = None
        for cand in reader.fieldnames:
            cn = _normalize_header(cand)
            if cn.replace(" ", "") == "TotalTime(us)" or cn == "Total Time(us)":
                key = cand
                break
        if key is None:
            for cand in reader.fieldnames:
                if "Total" in _normalize_header(cand) and "us" in _normalize_header(cand).lower():
                    key = cand
                    break
        if key is None:
            raise RuntimeError(
                f"no Total Time(us) column in {csv_path}, fields={reader.fieldnames}"
            )
        for row in reader:
            v = str(row.get(key, "")).strip()
            if v:
                total += float(v)
    return total


def newest_op_statistic_under(root: str) -> Tuple[str, str]:
    best_path: Optional[str] = None
    best_t = 0.0
    for dirpath, _, filenames in os.walk(root):
        if "op_statistic.csv" not in filenames:
            continue
        p = os.path.join(dirpath, "op_statistic.csv")
        try:
            t = os.path.getmtime(p)
        except OSError:
            continue
        if t > best_t:
            best_t = t
            best_path = p
    if not best_path:
        raise FileNotFoundError(
            f"op_statistic.csv not found under {root} (profiler 是否已写出？可稍等或检查 NPU 权限)"
        )
    rel = os.path.relpath(best_path, root)
    parts = rel.split(os.sep)
    trace_leaf = parts[0] if parts else root
    trace_root_dir = os.path.join(root, trace_leaf) if trace_leaf else root
    return best_path, trace_root_dir


def forward_layer_norm(
    mode: str,
    x: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    if mode == "custom":
        return torch.ops.npu.layer_norm(x, list(normalized_shape), weight, bias, eps)
    if mode == "native":
        return F.layer_norm(x, normalized_shape, weight, bias, eps)
    raise ValueError(f"unknown mode: {mode}")


def metric_divisor(active: int, repeat: int, divisor_mode: str) -> float:
    if divisor_mode == "active_only":
        return float(active) if active > 0 else 1.0
    return float(active * repeat) if (active * repeat) > 0 else 1.0


def run_one_profiler_block(
    forward_fn: Callable[[], torch.Tensor],
    handler_dir: str,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
    divisor_mode: str,
) -> Tuple[float, float, str, str]:
    steps = repeat * (wait + warmup + active)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    )
    if os.path.isdir(handler_dir):
        shutil.rmtree(handler_dir)
    os.makedirs(handler_dir, exist_ok=True)
    before = time.time()

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(handler_dir),
        experimental_config=experimental_config,
        schedule=torch_npu.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
            skip_first=0,
        ),
    ) as prof:
        for _ in range(steps):
            _ = forward_fn()
            prof.step()

    torch.npu.synchronize()
    deadline = time.time() + 120.0
    csv_path: Optional[str] = None
    trace_dir = ""
    while time.time() < deadline:
        try:
            csv_path, trace_dir = newest_op_statistic_under(handler_dir)
            if os.path.getmtime(csv_path) >= before - 1.0:
                break
        except FileNotFoundError:
            pass
        time.sleep(0.2)
    if not csv_path:
        csv_path, trace_dir = newest_op_statistic_under(handler_dir)

    total_us = sum_total_time_us(csv_path)
    denom = metric_divisor(active, repeat, divisor_mode)
    per_step_us = total_us / denom
    return total_us, per_step_us, csv_path, trace_dir


def shape_dtype(case: Dict[str, Any]) -> Tuple[str, str]:
    sh, dt = "?", "?"
    for item in case.get("inputs", []):
        if item.get("type") == "tensor" and item.get("name") == "x":
            sh = str(item.get("shape", "?"))
            dt = str(item.get("dtype", "?"))
            break
    return sh, dt


def profiler_test_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def default_case_file() -> str:
    return os.path.join(profiler_test_dir(), "layer_norm_perf_cases.jsonl")


def default_trace_root() -> str:
    return os.path.join(profiler_test_dir(), "profiler_trace")


def group_results_by_dtype(
    results: List[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        buckets[str(r.get("dtype", "?"))].append(r)
    preferred = ("float16", "bfloat16", "float32")
    out: List[Tuple[str, List[Dict[str, Any]]]] = []
    seen: set[str] = set()
    for dt in preferred:
        if dt in buckets:
            out.append((dt, buckets[dt]))
            seen.add(dt)
    for dt in sorted(k for k in buckets if k not in seen):
        out.append((dt, buckets[dt]))
    return out


def render_layer_norm_comparison_report_md(
    *,
    title: str,
    generated_at: str,
    case_file: str,
    trace_root: str,
    div: float,
    divisor_mode: str,
    results: List[Dict[str, Any]],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- 生成时间: {generated_at}",
        f"- 用例文件: `{case_file}`",
        f"- Trace: `{trace_root}`（`custom/case_XXX` 与 `native/case_XXX` 各对应一次 `with profile`；目录内为以 **`_ascend_pt` 为后缀** 的导出文件夹）",
        f"- 指标: `ASCEND_PROFILER_OUTPUT/op_statistic.csv` 各算子 **Total Time(us)** 求和 ÷ **{div}**（divisor_mode=`{divisor_mode}`）",
        f"- **固定 schedule**：warmup={PROFILER_SCHEDULE_WARMUP}、active={PROFILER_SCHEDULE_ACTIVE}（技能 ascendc-operator-performance-eval）",
        f"- 工程路径: `torch.ops.npu.layer_norm`（ascend_kernel）；原生路径: `torch.nn.functional.layer_norm`（同 NPU device）",
        "",
        "以下按 **输入张量 dtype** 分节列出对比表（节内不再重复 dtype 列）。",
    ]
    for dtype, rows in group_results_by_dtype(results):
        lines.extend(
            [
                "",
                f"## {dtype}",
                "",
                "| Case | Shape | 工程 per-step (us) | 原生 per-step (us) | native/custom |",
                "| ---- | ----- | ----------------- | ------------------- | ------------- |",
            ]
        )
        for r in rows:
            ratio = r.get("native_vs_custom_ratio")
            ratio_s = f"{ratio:.3f}" if ratio is not None else "n/a"
            lines.append(
                f"| {r['case_index']} | {r['shape']} | "
                f"{r['custom_per_active_step_us']:.3f} | {r['native_per_active_step_us']:.3f} | {ratio_s} |"
            )
        sub = [r["native_vs_custom_ratio"] for r in rows if r.get("native_vs_custom_ratio") is not None]
        avg = sum(sub) / len(sub) if sub else None
        cb = sum(1 for r in rows if (r.get("native_vs_custom_ratio") or 0) > 1.0)
        nb = sum(1 for r in rows if (r.get("native_vs_custom_ratio") or 0) < 1.0)
        avg_s = f"{avg:.3f}" if avg is not None else "n/a"
        lines.extend(
            [
                "",
                f"本节汇总（{dtype}）: 用例 {len(rows)}，平均 native/custom={avg_s}，工程更优 {cb}，原生更优 {nb}。",
            ]
        )

    ratios = [r["native_vs_custom_ratio"] for r in results if r.get("native_vs_custom_ratio") is not None]
    avg_ratio = sum(ratios) / len(ratios) if ratios else None
    custom_better = sum(1 for r in results if (r.get("native_vs_custom_ratio") or 0) > 1.0)
    native_better = sum(1 for r in results if (r.get("native_vs_custom_ratio") or 0) < 1.0)
    lines.extend(
        [
            "",
            "## 全量汇总",
            "",
            "| 指标 | 值 |",
            "| ---- | -- |",
            f"| 用例数 | {len(results)} |",
        ]
    )
    if avg_ratio is not None:
        lines.append(f"| 平均 native/custom（>1 表示原生更慢） | {avg_ratio:.3f} |")
    else:
        lines.append("| 平均 native/custom | n/a |")
    lines.extend(
        [
            f"| 工程更优（比值>1） | {custom_better} |",
            f"| 原生更优（比值<1） | {native_better} |",
            "",
            "### 按数据类型汇总",
            "",
            "| DType | 用例数 | 平均 native/custom | 工程更优 | 原生更优 |",
            "| ----- | ------ | ------------------- | -------- | -------- |",
        ]
    )
    for dtype, rows in group_results_by_dtype(results):
        sub = [r["native_vs_custom_ratio"] for r in rows if r.get("native_vs_custom_ratio") is not None]
        a = sum(sub) / len(sub) if sub else None
        a_s = f"{a:.3f}" if a is not None else "n/a"
        cb = sum(1 for r in rows if (r.get("native_vs_custom_ratio") or 0) > 1.0)
        nb = sum(1 for r in rows if (r.get("native_vs_custom_ratio") or 0) < 1.0)
        lines.append(f"| {dtype} | {len(rows)} | {a_s} | {cb} | {nb} |")
    lines.append("")
    return "\n".join(lines)


def run_layer_norm_profiler_cases(
    *,
    all_cases: List[Dict[str, Any]],
    case_indices: Optional[List[int]],
    device: torch.device,
    trace_root: str,
    trace_subdir: str,
    modes: List[str],
    load_ascend_kernel: bool,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
    divisor_mode: str,
) -> List[Dict[str, Any]]:
    if warmup != PROFILER_SCHEDULE_WARMUP or active != PROFILER_SCHEDULE_ACTIVE:
        raise ValueError(
            f"warmup/active 必须为 {PROFILER_SCHEDULE_WARMUP}/{PROFILER_SCHEDULE_ACTIVE}，收到 {warmup}/{active}"
        )
    if load_ascend_kernel:
        print(f"[INFO] custom lib: {load_custom_library()}")
    iter_cases: List[Tuple[int, Dict[str, Any]]]
    if case_indices is not None:
        iter_cases = [(i, all_cases[i]) for i in case_indices]
    else:
        iter_cases = list(enumerate(all_cases))

    base_out = os.path.join(trace_root, trace_subdir)
    results: List[Dict[str, Any]] = []

    for idx, case in iter_cases:
        shape_s, dtype_s = shape_dtype(case)
        inputs = build_inputs(case, device)
        x = inputs["x"]
        ns = inputs["normalized_shape"]
        w, b = inputs["weight"], inputs["bias"]
        eps = inputs["eps"]
        row: Dict[str, Any] = {"case_index": idx, "shape": shape_s, "dtype": dtype_s}

        for mode in modes:
            handler_dir = os.path.join(base_out, mode, f"case_{idx:03d}")

            def make_forward(m: str) -> Callable[[], torch.Tensor]:
                return lambda: forward_layer_norm(m, x, ns, w, b, eps)

            total_us, per_us, csv_p, trace_d = run_one_profiler_block(
                make_forward(mode),
                handler_dir,
                wait,
                warmup,
                active,
                repeat,
                divisor_mode,
            )
            prefix = "custom" if mode == "custom" else "native"
            row[f"{prefix}_sum_total_time_us"] = total_us
            row[f"{prefix}_per_active_step_us"] = per_us
            row[f"{prefix}_op_statistic_csv"] = csv_p
            row[f"{prefix}_trace_dir"] = trace_d
            print(
                f"[INFO] case {idx} {mode}: sum_total={total_us:.3f} us, "
                f"per_active_step={per_us:.3f} us -> {csv_p}"
            )

        if "custom" in modes and "native" in modes:
            c = row.get("custom_per_active_step_us")
            n = row.get("native_per_active_step_us")
            row["native_vs_custom_ratio"] = (n / c) if c and c > 0 else None

        results.append(row)

    return results
