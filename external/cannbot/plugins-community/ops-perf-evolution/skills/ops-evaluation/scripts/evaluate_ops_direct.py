#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
"""
ops仓算子 baseline vs evolved 直接对比评估工具（多 shape 版本）。

支持单 shape（旧 call_spec.json 顶层 inputs）和多 shape（target_shapes/
generalization_shapes）两种格式。旧格式在 normalize_call_spec 入口被
自动包装为单个 target shape，保持向后兼容。

每轮评估按 --shapes-mode 选择跑哪些 shape：
- target           : 只跑 target_shapes
- generalization   : 只跑 generalization_shapes
- all              : 跑 target_shapes + generalization_shapes

输出 evaluation_results.json 含：
- shape_results.{target,generalization} : per-shape baseline/evolved 时间和 speedup
- aggregate                              : target/generalization 聚合统计
- gating                                 : 5 enum {failed, target_regression,
                                           generalization_regression, partial_passed,
                                           fully_passed}
- baseline / evolved / comparison        : 旧字段（取 target[0] 数据）— 向后兼容

用法:
    python evaluate_ops_direct.py {op_name} \\
        --call-spec call_spec.json \\
        --baseline-path /path/to/baseline \\
        --evolved-path /path/to/evolved \\
        --device-id 0 --task-type vector \\
        --shapes-mode target \\
        --target-speedup 1.1 \\
        --output evaluation_results.json
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
PERF_SCRIPT = SCRIPT_DIR.parent.parent / "ascendc-evaluation" / "scripts" / "AscendPerformanceTest.py"

# Gating enum
GATING_FAILED = "failed"
GATING_TARGET_REGRESSION = "target_regression"
GATING_GENERALIZATION_REGRESSION = "generalization_regression"
GATING_PARTIAL_PASSED = "partial_passed"
GATING_FULLY_PASSED = "fully_passed"

SHAPES_MODE_TARGET = "target"
SHAPES_MODE_GENERALIZATION = "generalization"
SHAPES_MODE_ALL = "all"


sys.path.insert(0, str(SCRIPT_DIR))
from common.eval_utils import (  # noqa: E402
    acquire_eval_lock as _acquire_eval_lock,
    release_eval_lock as _release_eval_lock,
    detect_vendor_subdir,
)


# ============================================================
# Call spec 归一化与 shape 选取
# ============================================================

def normalize_call_spec(call_spec: dict) -> dict:
    """归一化 call_spec：旧单 shape 自动包装成 target_shapes=[{"name":"default",...}]。

    校验：
    - target_shapes / generalization_shapes 中 name 必须唯一（跨两组也唯一）
    - 至少存在一个 target shape
    - 每个 shape entry 含 inputs 字段
    """
    spec = dict(call_spec)  # 浅拷贝，不污染输入

    has_top_inputs = "inputs" in spec and isinstance(spec.get("inputs"), list)
    has_target_shapes = "target_shapes" in spec and isinstance(spec.get("target_shapes"), list)

    if has_top_inputs and not has_target_shapes:
        # 旧单 shape → 包装成单个 target shape
        default_entry = {
            "name": "default",
            "inputs": spec.get("inputs", []),
        }
        if "scalar_args" in spec:
            default_entry["scalar_args"] = spec["scalar_args"]
        if "tensor_kwargs" in spec:
            default_entry["tensor_kwargs"] = spec["tensor_kwargs"]
        spec["target_shapes"] = [default_entry]
        spec.setdefault("generalization_shapes", [])
        # 保留顶层 inputs/scalar_args/tensor_kwargs 不删除（向后兼容子进程对它们的引用）
    elif has_target_shapes:
        spec.setdefault("generalization_shapes", [])
    else:
        raise ValueError(
            "call_spec 必须含 target_shapes（多 shape 模式）或 inputs（向后兼容单 shape 模式）"
        )

    # 校验
    if not spec["target_shapes"]:
        raise ValueError("call_spec.target_shapes 不能为空")

    seen_names = set()
    for group, entries in (("target", spec["target_shapes"]),
                            ("generalization", spec["generalization_shapes"])):
        for i, entry in enumerate(entries):
            if "name" not in entry:
                raise ValueError(f"{group}_shapes[{i}] 缺少 name 字段")
            name = entry["name"]
            if name in seen_names:
                raise ValueError(f"shape name 重复: {name}（target_shapes 和 generalization_shapes 中不可重名）")
            seen_names.add(name)
            if "inputs" not in entry or not isinstance(entry["inputs"], list):
                raise ValueError(f"{group}_shapes[{i}] (name={name}) 缺少 inputs 列表")

    return spec


def select_shapes_to_run(call_spec: dict, shapes_mode: str) -> dict:
    """根据 shapes_mode 选出要跑的 shape 列表。

    Returns:
        {"target": [...], "generalization": [...]}, 每个列表内是 shape entry
    """
    if shapes_mode == SHAPES_MODE_TARGET:
        return {"target": call_spec["target_shapes"], "generalization": []}
    if shapes_mode == SHAPES_MODE_GENERALIZATION:
        return {"target": [], "generalization": call_spec.get("generalization_shapes", [])}
    if shapes_mode == SHAPES_MODE_ALL:
        return {
            "target": call_spec["target_shapes"],
            "generalization": call_spec.get("generalization_shapes", []),
        }
    raise ValueError(f"未知 shapes_mode: {shapes_mode}")


# ============================================================
# Aggregate / Gating 计算
# ============================================================

def _geomean(values):
    """几何平均（log 空间防溢出）；空列表返回 None。"""
    positives = [v for v in values if isinstance(v, (int, float)) and v > 0]
    if not positives:
        return None
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def compute_aggregate(target_results: list, generalization_results: list,
                       target_speedup: Optional[float]) -> dict:
    """聚合 per-shape speedup 计算 aggregate 字段。

    Args:
        target_results: list of {name, speedup, precision_passed, ...}（可能为空）
        generalization_results: 同上（可能为空）
        target_speedup: 用户设定的 target speedup 阈值（None 时所有 `all_target_meet_target` 字段为 None）

    Returns:
        aggregate dict（含 None 表示该项不适用）
    """
    agg = {
        "target_min_speedup": None,
        "target_geo_mean_speedup": None,
        "target_max_speedup": None,
        "all_target_meet_target": None,
        "all_target_above_baseline": None,
        "any_target_regression": None,
        "generalization_geo_mean_speedup": None,
        "any_generalization_regression": None,
    }

    if target_results:
        target_speedups = [r["speedup"] for r in target_results
                           if isinstance(r.get("speedup"), (int, float)) and r["speedup"] > 0]
        target_precision_ok = all(r.get("precision_passed", False) for r in target_results)
        if target_speedups and target_precision_ok:
            agg["target_min_speedup"] = round(min(target_speedups), 4)
            agg["target_max_speedup"] = round(max(target_speedups), 4)
            gm = _geomean(target_speedups)
            agg["target_geo_mean_speedup"] = round(gm, 4) if gm is not None else None
            agg["all_target_above_baseline"] = all(s >= 1.0 for s in target_speedups)
            agg["any_target_regression"] = any(s < 1.0 for s in target_speedups)
            if target_speedup is not None:
                agg["all_target_meet_target"] = all(s >= target_speedup for s in target_speedups)
        else:
            # target 跑了但失败（精度或时间无效）
            agg["all_target_above_baseline"] = False
            agg["any_target_regression"] = True
            agg["all_target_meet_target"] = False

    if generalization_results:
        gen_speedups = [r["speedup"] for r in generalization_results
                        if isinstance(r.get("speedup"), (int, float)) and r["speedup"] > 0]
        gen_precision_ok = all(r.get("precision_passed", False) for r in generalization_results)
        if gen_speedups and gen_precision_ok:
            gm = _geomean(gen_speedups)
            agg["generalization_geo_mean_speedup"] = round(gm, 4) if gm is not None else None
            agg["any_generalization_regression"] = (gm is not None and gm < 1.0)
        else:
            agg["any_generalization_regression"] = True

    return agg


def determine_gating(aggregate: dict, precision_ok: bool, compile_ok: bool) -> str:
    """根据 aggregate / 精度 / 编译状态判定 gating enum。

    优先级（先到先得）：
    1. 编译/精度/运行错 → failed
    2. any_target_regression == true → target_regression
    3. any_generalization_regression == true → generalization_regression
    4. all_target_meet_target == true → fully_passed
    5. 否则（所有 target ≥ 1.0x 但部分未达 target_speedup） → partial_passed
    """
    if not compile_ok or not precision_ok:
        return GATING_FAILED
    if aggregate.get("any_target_regression") is True:
        return GATING_TARGET_REGRESSION
    if aggregate.get("any_generalization_regression") is True:
        return GATING_GENERALIZATION_REGRESSION
    if aggregate.get("all_target_meet_target") is True:
        return GATING_FULLY_PASSED
    # all_target_above_baseline=True 但 all_target_meet_target!=True
    if aggregate.get("all_target_above_baseline") is True:
        return GATING_PARTIAL_PASSED
    return GATING_FAILED


# ============================================================
# 输入生成与子进程执行
# ============================================================

def _shape_input_dir(work_root: str, shape_name: str) -> str:
    """为某个 shape 生成的输入张量目录。"""
    return os.path.join(work_root, f"shape_{shape_name}")


def _make_input_tensor(inp: dict, dtype_map: dict):
    """按输入规格生成单个随机张量。"""
    import torch
    dtype = dtype_map.get(inp["dtype"], torch.float16)
    shape = inp["shape"]
    if dtype in (torch.int32, torch.int64):
        return torch.randint(0, 10, shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.uint8).to(torch.bool)
    return torch.randn(shape, dtype=dtype)


def _make_kwarg_tensor(kwarg_spec: dict, dtype_map: dict):
    """按 kwarg 规格生成张量：有 value 用之，否则按 shape 随机生成。"""
    import torch
    dtype = dtype_map.get(kwarg_spec["dtype"], torch.int64)
    value = kwarg_spec.get("value", None)
    if value is not None:
        return torch.tensor(value, dtype=dtype)
    shape = kwarg_spec["shape"]
    if dtype in (torch.int32, torch.int64):
        return torch.randint(0, 10, shape, dtype=dtype)
    return torch.randn(shape, dtype=dtype)


def _gen_tensor_kwargs(tensor_kwargs: dict, dtype_map: dict, out_dir: str):
    """为一个 shape 目录生成全部 tensor_kwargs 张量文件。"""
    import torch
    for kwarg_name, kwarg_spec in tensor_kwargs.items():
        t = _make_kwarg_tensor(kwarg_spec, dtype_map)
        torch.save(t, os.path.join(out_dir, f"kwarg_{kwarg_name}.pt"))


def generate_inputs(call_spec: dict, work_root: str,

                     shapes_to_run: dict) -> dict:
    """为所有要跑的 shape 生成输入张量，每个 shape 一个独立子目录。

    Returns:
        {shape_name: input_dir_path} 映射
    """
    import torch
    os.makedirs(work_root, exist_ok=True)

    torch.manual_seed(42)

    dtype_map = {
        "float16": torch.float16, "float32": torch.float32,
        "bfloat16": torch.bfloat16, "int32": torch.int32,
        "int64": torch.int64, "bool": torch.bool,
    }

    input_dirs = {}
    all_entries = shapes_to_run["target"] + shapes_to_run["generalization"]

    for entry in all_entries:
        name = entry["name"]
        out_dir = _shape_input_dir(work_root, name)
        os.makedirs(out_dir, exist_ok=True)

        for i, inp in enumerate(entry.get("inputs", [])):
            t = _make_input_tensor(inp, dtype_map)
            torch.save(t, os.path.join(out_dir, f"input_{i}.pt"))

        # tensor_kwargs（可能在 shape entry 或 call_spec 顶层）
        tensor_kwargs = entry.get("tensor_kwargs", call_spec.get("tensor_kwargs", {}))
        _gen_tensor_kwargs(tensor_kwargs, dtype_map, out_dir)

        input_dirs[name] = out_dir

    # 保存归一化后的 call_spec
    with open(os.path.join(work_root, "call_spec.json"), "w") as f:
        json.dump(call_spec, f, indent=2)

    return input_dirs


def _build_shape_plan(shapes_to_run: dict, input_dirs: dict, call_spec: dict) -> list:
    """构造子进程要 loop 的 shape 计划。

    每项含运行该 shape 所需的全部静态信息（input_dir / n_inputs / scalar_args /
    tensor_kwargs_keys / group）。
    """
    plan = []
    for group_name, entries in (("target", shapes_to_run["target"]),
                                 ("generalization", shapes_to_run["generalization"])):
        for entry in entries:
            name = entry["name"]
            scalar_args = entry.get("scalar_args", call_spec.get("scalar_args", {}))
            tensor_kwargs = entry.get("tensor_kwargs", call_spec.get("tensor_kwargs", {}))
            plan.append({
                "name": name,
                "group": group_name,
                "input_dir": input_dirs[name],
                "n_inputs": len(entry.get("inputs", [])),
                "scalar_args": scalar_args,
                "tensor_kwargs_keys": list(tensor_kwargs.keys()),
            })
    return plan


@dataclass
class DirectVersionConfig:
    """run_single_version 的参数封装。"""
    call_spec: dict
    install_path: str
    shape_plan: list
    device_id: int
    task_type: str
    profile_dir: str
    num_trials: int
    tag: str


def _dscript_env_setup(cfg: DirectVersionConfig, opp_path: str, lib_path: str,
                       shape_plan_json: str) -> str:
    """生成子进程脚本：环境设置 + 引擎初始化段。"""
    device_id = cfg.device_id
    profile_dir = cfg.profile_dir
    tag = cfg.tag
    op_namespace = cfg.call_spec["op_namespace"]
    op_func = cfg.call_spec["op_func"]
    is_omni_ops = cfg.call_spec.get("is_omni_ops", False)
    return f"""
        #!/usr/bin/env python3
        import os, sys, json, logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

        os.environ["ASCEND_CUSTOM_OPP_PATH"] = {repr(opp_path)}
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = {repr(lib_path)} + ":" + existing_ld
        os.environ["ASCEND_DEVICE_ID"] = str({device_id})

        perf_script_dir = {repr(str(PERF_SCRIPT.parent))}
        if perf_script_dir not in sys.path:
            sys.path.insert(0, perf_script_dir)

        from pathlib import Path
        import torch
        import torch_npu
        torch.npu.set_device({device_id})

        if {repr(is_omni_ops)}:
            try:
                import omni_custom_ops
                logging.info("omni_custom_ops imported successfully")
            except ImportError as e:
                logging.warning(f"Failed to import omni_custom_ops: {{e}}")
        try:
            import omni_custom_ops
        except ImportError:
            pass
        from AscendPerformanceTest import AdvancedPerformanceEngine

        op_fn = getattr(getattr(torch.ops, {repr(op_namespace)}), {repr(op_func)})

        shape_plan = json.loads({repr(shape_plan_json)})
        profile_root_base = Path({repr(profile_dir)})
        engine = AdvancedPerformanceEngine(logging.getLogger("DirectEval"))
"""


def _dscript_inputs_load() -> str:
    """生成子进程脚本：输入张量与 tensor_kwargs 加载段。"""
    return """
                inputs = []
                for i in range(n_inputs):
                    t = torch.load(os.path.join(input_dir, f"input_{i}.pt"), weights_only=True)
                    inputs.append(t.npu())

                tensor_kwargs = {}
                for kwarg_name in tensor_kwargs_keys:
                    kwarg_path = os.path.join(input_dir, f"kwarg_{kwarg_name}.pt")
                    if os.path.exists(kwarg_path):
                        kt = torch.load(kwarg_path, weights_only=True)
                        tensor_kwargs[kwarg_name] = kt.npu()

                def model(*args):
                    sa_copy = dict(scalar_args)
                    pos_args = []
                    for key in ["scale_value", "sparse_block_size"]:
                        if key in sa_copy:
                            pos_args.append(sa_copy.pop(key))
                    return op_fn(*args, *pos_args, **sa_copy, **tensor_kwargs)
"""


def _dscript_perf_measure(cfg: DirectVersionConfig) -> str:
    """生成子进程脚本：warmup_and_measure 性能测量 + pipeline 提取段。"""
    device_id = cfg.device_id
    task_type = cfg.task_type
    num_trials = cfg.num_trials
    tag = cfg.tag
    return f"""
                profile_root = profile_root_base / f"shape_{{shape_name}}"
                profile_root.mkdir(parents=True, exist_ok=True)
                median_time, perf_data, output_path, cv_pct = engine.warmup_and_measure(
                    model=model, inputs=inputs, device_id={device_id},
                    profile_root=profile_root, num_trials={num_trials},
                    task_type={repr(task_type)}, model_tag=f"{{shape_name}}_{repr(tag)[1:-1]}"
                )
                result["time_us"] = median_time if median_time is not None else -1
                result["cv_pct"] = cv_pct if cv_pct is not None else 0.0

                if perf_data and isinstance(perf_data, list):
                    pipeline = {{}}
                    for row in perf_data:
                        if isinstance(row, dict):
                            for key in ("aiv_mte2_ratio", "aiv_vec_ratio", "aiv_scalar_ratio",
                                        "aiv_mte3_ratio", "aiv_mte2_time(us)", "aiv_vec_time(us)",
                                        "aiv_scalar_time(us)", "aiv_mte3_time(us)",
                                        "aiv_icache_miss_rate", "cube_utilization(%)",
                                        "aic_mac_ratio", "aic_mte1_ratio", "aic_mte2_ratio"):
                                if key in row and row[key] is not None:
                                    pipeline[key] = row[key]
                    result["pipeline"] = pipeline
"""


def _dscript_output_save(cfg: DirectVersionConfig) -> str:
    """生成子进程脚本：日志 + 输出张量保存 + 异常处理段。"""
    profile_dir = cfg.profile_dir
    tag = cfg.tag
    return f"""
                tag_str = result["tag"]
                logging.info(f"[{{tag_str}}/{{shape_name}}] 性能: {{median_time:.2f}}us, cv={{cv_pct:.1f}}%")

                # 保存输出张量（per-shape，用于精度对比）
                with torch.no_grad():
                    output = model(*inputs)
                torch.save(output, os.path.join({repr(profile_dir)}, f"{tag}_output_{{shape_name}}.pt"))
                logging.info(f"[{{tag_str}}/{{shape_name}}] 输出已保存")

            except Exception as e:
                import traceback
                result["precision_passed"] = False
                result["correctness_message"] = f"评估异常: {{e}}"
                result["time_us"] = -1
                logging.error(f"[{{shape_name}}] 评估异常: {{e}}\n{{traceback.format_exc()}}")

            all_results.append(result)
"""


def _dscript_shape_loop(cfg: DirectVersionConfig) -> str:
    """生成子进程脚本：per-shape 评估循环段。"""
    tag = cfg.tag
    return (
        f"""
        all_results = []
        for shape_item in shape_plan:
            shape_name = shape_item["name"]
            group = shape_item["group"]
            input_dir = shape_item["input_dir"]
            n_inputs = shape_item["n_inputs"]
            scalar_args = shape_item["scalar_args"]
            tensor_kwargs_keys = shape_item["tensor_kwargs_keys"]

            result = {{
                "name": shape_name, "group": group, "tag": {repr(tag)},
                "precision_passed": True, "correctness_message": "",
                "time_us": -1, "pipeline": {{}}, "bottleneck": "unknown", "cv_pct": 0.0,
            }}
            try:
"""
        + _dscript_inputs_load()
        + _dscript_perf_measure(cfg)
        + _dscript_output_save(cfg)
    )


def _dscript_output() -> str:
    """生成子进程脚本：JSON 结果输出段。"""
    return """
        print("--- EVAL_RESULT_JSON ---")
        print(json.dumps(all_results, ensure_ascii=False, indent=2))
        print("--- END_EVAL_RESULT_JSON ---")
"""


def _build_direct_eval_script(cfg: DirectVersionConfig, opp_path: str,
                              lib_path: str, shape_plan_json: str) -> str:
    """生成子进程评估脚本（loop 所有 shape 跑性能+保存输出张量）。"""
    return textwrap.dedent(
        _dscript_env_setup(cfg, opp_path, lib_path, shape_plan_json)
        + _dscript_shape_loop(cfg)
        + _dscript_output()
    )


def _direct_error_results(shape_plan: list, tag: str, error: str) -> list:
    """为所有 shape 生成错误结果项。"""
    return [
        {"name": s["name"], "group": s["group"], "tag": tag,
         "error": error, "precision_passed": False, "time_us": -1}
        for s in shape_plan
    ]


def _run_direct_subprocess(cfg: DirectVersionConfig, script_path: str,
                           opp_path: str, lib_path: str) -> list:
    """运行评估子进程并解析 EVAL_RESULT_JSON 输出。"""
    try:
        logging.info("[%s] 启动评估子进程: %s（%d 个 shape）",
                     cfg.tag, script_path, len(cfg.shape_plan))
        env = os.environ.copy()
        env["ASCEND_CUSTOM_OPP_PATH"] = opp_path
        env["LD_LIBRARY_PATH"] = lib_path + ":" + env.get("LD_LIBRARY_PATH", "")
        env["ASCEND_DEVICE_ID"] = str(cfg.device_id)

        # 超时按 shape 数量线性扩展（每 shape 上限 600s）
        timeout = max(600, 600 * len(cfg.shape_plan))
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=timeout, env=env,
        )

        stdout = proc.stdout
        if "--- EVAL_RESULT_JSON ---" in stdout:
            json_start = stdout.index("--- EVAL_RESULT_JSON ---") + len("--- EVAL_RESULT_JSON ---")
            json_end = stdout.index("--- END_EVAL_RESULT_JSON ---")
            return json.loads(stdout[json_start:json_end].strip())
        logging.error(
            "[%s] 子进程未输出结果\nstdout (last 1000):\n%s\nstderr (last 1000):\n%s",
            cfg.tag, stdout[-1000:], proc.stderr[-1000:],
        )
        return _direct_error_results(
            cfg.shape_plan, cfg.tag, f"子进程未输出结果: rc={proc.returncode}")

    except subprocess.TimeoutExpired:
        return _direct_error_results(cfg.shape_plan, cfg.tag, "评估超时")
    except Exception as e:
        return _direct_error_results(cfg.shape_plan, cfg.tag, str(e))


def run_single_version(cfg: DirectVersionConfig) -> list:
    """在子进程中评估单个版本（baseline / evolved），loop 所有 shape。

    Returns:
        list of per-shape dict: {name, group, time_us, precision_passed,
        correctness_message, pipeline, bottleneck, cv_pct, error?}
    """
    vendor_subdir = detect_vendor_subdir(cfg.install_path)
    opp_path = os.path.join(cfg.install_path, "vendors", vendor_subdir)
    lib_path = os.path.join(opp_path, "op_api", "lib")

    if not os.path.isdir(opp_path):
        return _direct_error_results(
            cfg.shape_plan, cfg.tag, f"OPP directory not found: {opp_path}")

    os.makedirs(cfg.profile_dir, exist_ok=True)

    shape_plan_json = json.dumps(cfg.shape_plan)
    eval_script = _build_direct_eval_script(cfg, opp_path, lib_path, shape_plan_json)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"eval_direct_{cfg.tag}_",
        delete=False, dir=cfg.profile_dir,
    ) as f:
        f.write(eval_script)
        script_path = f.name

    return _run_direct_subprocess(cfg, script_path, opp_path, lib_path)


# ============================================================
# 精度对比与 speedup 合并
# ============================================================

def compare_outputs_per_shape(baseline_dir: str, evolved_dir: str,
                               shape_names: list,
                               rtol: float = 1e-3, atol: float = 1e-3) -> dict:
    """per-shape 精度对比。Returns {shape_name: (passed, message)}."""
    import torch
    out = {}
    for name in shape_names:
        baseline_path = os.path.join(baseline_dir, f"baseline_output_{name}.pt")
        evolved_path = os.path.join(evolved_dir, f"evolved_output_{name}.pt")
        if not os.path.exists(baseline_path):
            out[name] = (False, f"Baseline output not found: {baseline_path}")
            continue
        if not os.path.exists(evolved_path):
            out[name] = (False, f"Evolved output not found: {evolved_path}")
            continue

        baseline_out = torch.load(baseline_path, weights_only=False, map_location="cpu")
        evolved_out = torch.load(evolved_path, weights_only=False, map_location="cpu")

        if isinstance(baseline_out, torch.Tensor):
            baseline_out = [baseline_out]
        if isinstance(evolved_out, torch.Tensor):
            evolved_out = [evolved_out]
        if isinstance(baseline_out, tuple):
            baseline_out = list(baseline_out)
        if isinstance(evolved_out, tuple):
            evolved_out = list(evolved_out)

        if len(baseline_out) != len(evolved_out):
            out[name] = (False, f"Output count mismatch: baseline={len(baseline_out)}, evolved={len(evolved_out)}")
            continue

        max_diff = 0.0
        passed = True
        msg = ""
        for i, (b, e) in enumerate(zip(baseline_out, evolved_out)):
            if not isinstance(b, torch.Tensor) or not isinstance(e, torch.Tensor):
                continue
            bf, ef = b.float(), e.float()
            abs_diff = (bf - ef).abs()
            diff = abs_diff.max().item()
            max_diff = max(max_diff, diff)
            if not torch.allclose(bf, ef, rtol=rtol, atol=atol):
                atol_ok = (abs_diff <= atol).all().item()
                rtol_ok = (abs_diff <= rtol * ef.abs()).all().item()
                passed = False
                msg = (
                    f"Output[{i}] mismatch: max_abs_diff={diff:.6f} "
                    f"(atol_ok={'Y' if atol_ok else 'N'}, rtol_ok={'Y' if rtol_ok else 'N'}, "
                    f"rtol={rtol}, atol={atol})"
                )
                break
        if passed:
            msg = f"All outputs match (max_abs_diff={max_diff:.6f})"
        out[name] = (passed, msg)

    return out


def merge_per_shape_results(baseline_results: list, evolved_results: list,
                              precision_per_shape: dict) -> tuple:
    """按 shape name 对齐 baseline/evolved，计算 speedup 和精度。

    Returns:
        (shape_results, compile_ok_flag)
        shape_results = {"target": [...], "generalization": [...]}
        compile_ok_flag = True if 所有 shape 子进程都跑出了有效时间
    """
    baseline_map = {r["name"]: r for r in baseline_results if "name" in r}
    evolved_map = {r["name"]: r for r in evolved_results if "name" in r}

    shape_results = {"target": [], "generalization": []}
    all_compile_ok = True

    for name, br in baseline_map.items():
        er = evolved_map.get(name)
        if er is None:
            continue

        group = br.get("group", "target")
        bt = br.get("time_us", -1)
        et = er.get("time_us", -1)

        if bt > 0 and et > 0:
            speedup = round(bt / et, 4)
        else:
            speedup = 0.0
            all_compile_ok = False

        precision_passed, precision_msg = precision_per_shape.get(name, (False, "not_evaluated"))
        # 子进程内部已经判过精度（计算 vs CPU 不做，只 baseline vs evolved 对齐）
        # 若子进程报错 precision_passed=False，叠加 evolved.precision_passed
        ev_precision = er.get("precision_passed", False) and precision_passed
        bl_precision = br.get("precision_passed", False)

        entry = {
            "name": name,
            "baseline_time_us": bt,
            "evolved_time_us": et,
            "speedup": speedup,
            "precision_passed": ev_precision,
            "baseline_precision_passed": bl_precision,
            "correctness_message": precision_msg,
            "pipeline": er.get("pipeline", {}),
            "bottleneck": er.get("bottleneck", "unknown"),
            "cv_pct": er.get("cv_pct", 0.0),
            "compilation_success": (bt > 0 and et > 0),
        }
        if "error" in er:
            entry["error"] = er.get("error")
            all_compile_ok = False
        shape_results.setdefault(group, []).append(entry)

    return shape_results, all_compile_ok


# ============================================================
# 向后兼容字段合成
# ============================================================

def _synthesize_legacy_fields(shape_results: dict, aggregate: dict) -> dict:
    """从 shape_results 合成旧 baseline/evolved/comparison 字段（取 target[0]）。"""
    legacy = {"baseline": None, "evolved": None, "comparison": None}
    target_list = shape_results.get("target") or []
    if not target_list:
        # 退化使用 generalization[0]
        target_list = shape_results.get("generalization") or []
    if not target_list:
        return legacy

    head = target_list[0]
    legacy["baseline"] = {
        "tag": "baseline",
        "time_us": head.get("baseline_time_us", -1),
        "precision_passed": head.get("baseline_precision_passed", False),
        "pipeline": {},
        "bottleneck": "unknown",
        "cv_pct": 0.0,
    }
    legacy["evolved"] = {
        "tag": "evolved",
        "time_us": head.get("evolved_time_us", -1),
        "precision_passed": head.get("precision_passed", False),
        "pipeline": head.get("pipeline", {}),
        "bottleneck": head.get("bottleneck", "unknown"),
        "cv_pct": head.get("cv_pct", 0.0),
        "correctness_message": head.get("correctness_message", ""),
    }
    # comparison.speedup 用 aggregate.target_min_speedup（min-of-target 语义）
    target_min = aggregate.get("target_min_speedup")
    head_speedup = head.get("speedup", 0.0)
    speedup_for_legacy = target_min if target_min is not None else head_speedup
    bt = head.get("baseline_time_us", -1)
    et = head.get("evolved_time_us", -1)
    legacy["comparison"] = {
        "compilation_success": head.get("compilation_success", False),
        "speedup": speedup_for_legacy,
        "time_delta_us": (et - bt) if (bt > 0 and et > 0) else 0.0,
        "precision_passed": all(r.get("precision_passed", False) for r in target_list),
        "precision_message": head.get("correctness_message", ""),
        "cv_pct": head.get("cv_pct", 0.0),
        "measurement_quality": (
            "good" if head.get("cv_pct", 100) < 5.0
            else "acceptable" if head.get("cv_pct", 100) < 15.0
            else "noisy"
        ),
    }
    return legacy


# ============================================================
# 主流程
# ============================================================

@dataclass
class DirectEvalConfig:
    """evaluate_ops_direct 的参数封装。"""
    op_name: str
    call_spec_path: str
    baseline_path: str
    evolved_path: str
    device_id: int = 0
    task_type: str = "vector"
    output_path: Optional[str] = None
    num_trials: int = 50
    eval_lock: Optional[str] = None
    eval_lock_timeout: float = 300.0
    baseline_cache: Optional[str] = None
    shapes_mode: str = SHAPES_MODE_ALL
    target_speedup: Optional[float] = None
    eval_backend: str = "default"


def _load_baseline_shape_cache(baseline_cache: str, shape_plan: list):
    """加载 per-shape baseline 缓存；覆盖全部待跑 shape 时返回结果列表，否则 None。"""
    if not baseline_cache or not os.path.isfile(baseline_cache):
        return None
    try:
        with open(baseline_cache, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_shape_results = cached.get("shape_results", {})
        cached_baseline = []
        for grp in ("target", "generalization"):
            for r in cached_shape_results.get(grp, []):
                cached_baseline.append({
                    "name": r["name"], "group": grp, "tag": "baseline",
                    "time_us": r.get("baseline_time_us", -1),
                    "precision_passed": r.get("baseline_precision_passed", True),
                    "pipeline": {}, "bottleneck": "unknown", "cv_pct": 0.0,
                })
        # 仅当缓存覆盖了所有要跑的 shape 时使用
        cached_names = {r["name"] for r in cached_baseline if r.get("time_us", -1) > 0}
        need_names = {s["name"] for s in shape_plan}
        if cached_names >= need_names:
            baseline_results = [r for r in cached_baseline if r["name"] in need_names]
            logging.info("使用 baseline 缓存（覆盖 %d 个 shape）", len(baseline_results))
            return baseline_results
    except (json.JSONDecodeError, OSError) as e:
        logging.warning("Baseline 缓存读取失败: %s", e)
    return None


def _run_baseline_version(cfg: DirectEvalConfig, call_spec: dict, shape_plan: list,
                          baseline_profile_dir: str, cached: list) -> list:
    """评估 baseline 版本：无缓存全量跑；有缓存时少量 trial 生成输出张量并合回缓存时间。"""
    if cached is None:
        logging.info("评估 baseline: %s", cfg.baseline_path)
        return run_single_version(DirectVersionConfig(
            call_spec=call_spec, install_path=cfg.baseline_path,
            shape_plan=shape_plan, device_id=cfg.device_id,
            task_type=cfg.task_type, profile_dir=baseline_profile_dir,
            num_trials=cfg.num_trials, tag="baseline",
        ))
    # 缓存命中但仍需为精度对比生成 baseline 输出张量
    logging.info("Baseline 使用缓存性能数据；仅运行少量 trial 生成输出张量")
    baseline_run = run_single_version(DirectVersionConfig(
        call_spec=call_spec, install_path=cfg.baseline_path,
        shape_plan=shape_plan, device_id=cfg.device_id,
        task_type=cfg.task_type, profile_dir=baseline_profile_dir,
        num_trials=3, tag="baseline",
    ))
    # 把缓存的时间合回（按 name 对齐）
    cached_map = {r["name"]: r for r in cached}
    for fresh in baseline_run:
        cn = fresh["name"]
        if cn in cached_map:
            fresh["time_us"] = cached_map[cn]["time_us"]
    return baseline_run


def _run_direct_evals(cfg: DirectEvalConfig, call_spec: dict, shape_plan: list,
                      baseline_profile_dir: str, evolved_profile_dir: str) -> tuple:
    """在评估锁保护下跑 baseline / evolved 两个版本评估。"""
    cached = _load_baseline_shape_cache(cfg.baseline_cache, shape_plan)

    lock_fd = None
    if cfg.eval_lock:
        logging.info("等待评估锁: %s (超时 %ss)", cfg.eval_lock, cfg.eval_lock_timeout)
        lock_fd = _acquire_eval_lock(cfg.eval_lock, cfg.eval_lock_timeout)
        logging.info("评估锁已获取")

    try:
        baseline_results = _run_baseline_version(
            cfg, call_spec, shape_plan, baseline_profile_dir, cached)

        logging.info("评估 evolved: %s", cfg.evolved_path)
        evolved_results = run_single_version(DirectVersionConfig(
            call_spec=call_spec, install_path=cfg.evolved_path,
            shape_plan=shape_plan, device_id=cfg.device_id,
            task_type=cfg.task_type, profile_dir=evolved_profile_dir,
            num_trials=cfg.num_trials, tag="evolved",
        ))
    finally:
        if lock_fd is not None:
            _release_eval_lock(lock_fd)
            logging.info("评估锁已释放")

    return baseline_results, evolved_results


def _load_call_spec_and_plan(cfg: DirectEvalConfig, work_dir: str) -> tuple:
    """加载 call_spec、生成输入张量并构建 shape 计划。

    Returns (call_spec, shape_plan)。
    """
    with open(cfg.call_spec_path, "r", encoding="utf-8") as f:
        raw_call_spec = json.load(f)
    call_spec = normalize_call_spec(raw_call_spec)
    shapes_to_run = select_shapes_to_run(call_spec, cfg.shapes_mode)

    input_root = os.path.join(work_dir, "eval_inputs")
    logging.info("生成输入张量: %s（shapes_mode=%s, target=%d, generalization=%d）",
                 input_root, cfg.shapes_mode,
                 len(shapes_to_run['target']), len(shapes_to_run['generalization']))
    input_dirs = generate_inputs(call_spec, input_root, shapes_to_run)
    shape_plan = _build_shape_plan(shapes_to_run, input_dirs, call_spec)
    return call_spec, shape_plan


def _fix_one_shape_result(r: dict):
    """自评场景下修正单个 shape 结果为 speedup=1.0。"""
    if r.get("baseline_time_us", -1) > 0:
        r["speedup"] = 1.0
        r["precision_passed"] = True
        r["compilation_success"] = True


def _apply_self_baseline_fix(cfg: DirectEvalConfig, shape_results: dict):
    """baseline 自评特殊情况：baseline_path == evolved_path 时所有 speedup 应为 1.0。"""
    if os.path.realpath(cfg.baseline_path) == os.path.realpath(cfg.evolved_path):
        for grp_list in shape_results.values():
            for r in grp_list:
                _fix_one_shape_result(r)


def _build_final_result(cfg: DirectEvalConfig, shape_results: dict,
                        compile_ok: bool) -> tuple:
    """计算 aggregate/gating/legacy 并组装最终结果。

    Returns (final_result, aggregate, gating)。
    """
    aggregate = compute_aggregate(
        shape_results["target"], shape_results["generalization"], cfg.target_speedup
    )

    precision_ok = (
        all(r.get("precision_passed", False) for r in shape_results["target"])
        and all(r.get("precision_passed", False) for r in shape_results["generalization"])
    )
    gating = determine_gating(aggregate, precision_ok=precision_ok, compile_ok=compile_ok)
    legacy = _synthesize_legacy_fields(shape_results, aggregate)

    final_result = {
        "op_name": cfg.op_name,
        "call_spec": os.path.basename(cfg.call_spec_path),
        "eval_backend": cfg.eval_backend,
        "shapes_mode": cfg.shapes_mode,
        "shape_results": shape_results,
        "aggregate": aggregate,
        "gating": gating,
        # 向后兼容
        "baseline": legacy["baseline"],
        "evolved": legacy["evolved"],
        "comparison": legacy["comparison"],
    }
    return final_result, aggregate, gating


def _write_eval_result(cfg: DirectEvalConfig, work_dir: str, final_result: dict):
    """写出 evaluation_results.json。"""
    output_path = cfg.output_path
    if output_path is None:
        output_path = os.path.join(work_dir, "evaluation_results.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    logging.info("结果已保存: %s", output_path)


def evaluate_ops_direct(cfg: DirectEvalConfig) -> dict:
    """baseline vs evolved 多 shape 直接对比评估。"""
    work_dir = os.path.dirname(cfg.evolved_path) if cfg.evolved_path else "."
    baseline_profile_dir = os.path.join(work_dir, "baseline_profiling")
    evolved_profile_dir = os.path.join(work_dir, "evolved_profiling")

    call_spec, shape_plan = _load_call_spec_and_plan(cfg, work_dir)

    baseline_results, evolved_results = _run_direct_evals(
        cfg, call_spec, shape_plan, baseline_profile_dir, evolved_profile_dir)

    # per-shape 精度对比
    all_shape_names = [s["name"] for s in shape_plan]
    precision_per_shape = compare_outputs_per_shape(
        baseline_profile_dir, evolved_profile_dir, all_shape_names
    )

    shape_results, compile_ok = merge_per_shape_results(
        baseline_results, evolved_results, precision_per_shape
    )

    _apply_self_baseline_fix(cfg, shape_results)
    final_result, aggregate, gating = _build_final_result(cfg, shape_results, compile_ok)
    _write_eval_result(cfg, work_dir, final_result)

    _log_direct_summary(cfg, shape_results, aggregate, gating)

    return final_result


def _log_direct_summary(cfg: DirectEvalConfig, shape_results: dict,
                        aggregate: dict, gating: str):
    """打印评估结果摘要到日志。"""
    logging.info("")
    logging.info("=" * 60)
    logging.info("评估结果: %s  (shapes_mode=%s, gating=%s)",
                 cfg.op_name, cfg.shapes_mode, gating)
    logging.info("=" * 60)
    for grp in ("target", "generalization"):
        if not shape_results[grp]:
            continue
        logging.info("  [%s]", grp)
        for r in shape_results[grp]:
            tag = "PASS" if r.get("precision_passed") else "FAIL"
            logging.info("    %-12s baseline=%.2fus evolved=%.2fus speedup=%.3fx  [%s]",
                         r['name'], r.get('baseline_time_us', -1),
                         r.get('evolved_time_us', -1), r.get('speedup', 0), tag)
    if aggregate.get("target_min_speedup") is not None:
        logging.info("  target min/geo/max = %.3f / %.3f / %.3f",
                     aggregate['target_min_speedup'],
                     aggregate.get('target_geo_mean_speedup') or 0,
                     aggregate.get('target_max_speedup') or 0)
    if aggregate.get("generalization_geo_mean_speedup") is not None:
        logging.info("  generalization geomean = %.3f",
                     aggregate['generalization_geo_mean_speedup'])
    logging.info("=" * 60)
    logging.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Direct baseline vs evolved evaluation (multi-shape) for ops repository operators"
    )
    parser.add_argument("op_name", type=str, help="算子名")
    parser.add_argument("--call-spec", required=True, help="call_spec.json 路径")
    parser.add_argument("--baseline-path", required=True, help="baseline 安装路径")
    parser.add_argument("--evolved-path", required=True, help="evolved 安装路径")
    parser.add_argument("--device-id", type=int, default=0, help="NPU 设备 ID")
    parser.add_argument("--task-type", type=str, default="vector",
                        choices=["vector", "cube", "cv-mix", "unknown"])
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")
    parser.add_argument("--num-trials", type=int, default=50, help="profiling 试验次数")
    parser.add_argument("--eval-lock", type=str, default=None, help="评估排队锁文件路径")
    parser.add_argument("--eval-lock-timeout", type=float, default=300)
    parser.add_argument("--baseline-cache", type=str, default=None,
                        help="baseline 评估结果缓存文件路径")
    parser.add_argument("--shapes-mode", choices=[SHAPES_MODE_TARGET, SHAPES_MODE_GENERALIZATION, SHAPES_MODE_ALL],
                        default=SHAPES_MODE_ALL,
                        help="跑哪些 shape：target / generalization / all（默认 all）")
    parser.add_argument("--target-speedup", type=float, default=None,
                        help="目标加速比，用于判定 all_target_meet_target / gating")
    parser.add_argument("--eval-backend", default="default",
                        help="评估后端标识，写入 evaluation_results.eval_backend")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    evaluate_ops_direct(DirectEvalConfig(
        op_name=args.op_name,
        call_spec_path=args.call_spec,
        baseline_path=args.baseline_path,
        evolved_path=args.evolved_path,
        device_id=args.device_id,
        task_type=args.task_type,
        output_path=args.output,
        num_trials=args.num_trials,
        eval_lock=args.eval_lock,
        eval_lock_timeout=args.eval_lock_timeout,
        baseline_cache=args.baseline_cache,
        shapes_mode=args.shapes_mode,
        target_speedup=args.target_speedup,
        eval_backend=args.eval_backend,
    ))


if __name__ == "__main__":
    main()
