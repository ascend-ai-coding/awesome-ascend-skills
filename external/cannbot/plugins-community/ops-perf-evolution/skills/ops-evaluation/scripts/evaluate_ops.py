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
ops仓算子 baseline vs evolved 对比评估工具。

通过子进程隔离分别评估 baseline 和 evolved 两个版本的精度和性能，
然后合并结果生成对比报告。

使用子进程隔离的原因：CANN 运行时加载 OPP 库后无法在同一进程中切换到另一个版本。

用法:
    python evaluate_ops.py {op_name} \
        --baseline-path {baseline_install_path} \
        --evolved-path {evolved_install_path} \
        --reference-py {reference.py} \
        --custom-py {custom.py} \
        --device-id 0 \
        --task-type vector

输出:
    evaluation_results.json - baseline vs evolved 对比报告
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
# evaluate.py lives in the ascendc-evaluation scripts dir
EVALUATE_SCRIPT = SCRIPT_DIR.parent.parent / "ascendc-evaluation" / "scripts" / "evaluate.py"


sys.path.insert(0, str(SCRIPT_DIR))
from common.eval_utils import (  # noqa: E402
    acquire_eval_lock as _acquire_eval_lock,
    release_eval_lock as _release_eval_lock,
    detect_vendor_subdir,
)


@dataclass
class VersionEvalConfig:
    """evaluate_single_version 的参数封装。"""
    op_name: str
    install_path: str
    reference_py: str
    custom_py: str
    device_id: int
    task_type: str
    profile_dir: str
    num_trials: int
    tag: str


def _script_env_setup(cfg: VersionEvalConfig, opp_path: str, lib_path: str) -> str:
    """生成子进程脚本：环境设置 + AscendBackend 初始化段。"""
    return f"""
        #!/usr/bin/env python3
        import os
        import sys
        import json
        import logging

        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

        # 设置 OPP 环境变量
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = {repr(opp_path)}

        # 添加 op_api/lib 到 LD_LIBRARY_PATH
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = {repr(lib_path)} + ":" + existing_ld

        # 设置设备
        os.environ["ASCEND_DEVICE_ID"] = str({cfg.device_id})

        # 添加 evaluate.py 所在目录到 sys.path
        eval_script_dir = {repr(str(EVALUATE_SCRIPT.parent))}
        if eval_script_dir not in sys.path:
            sys.path.insert(0, eval_script_dir)

        from pathlib import Path
        import torch
        import torch_npu

        # 设置 NPU 设备
        torch.npu.set_device({cfg.device_id})

        # 导入 AscendBackend
        from evaluate import AscendBackend

        # 读取代码
        ref_code = Path({repr(cfg.reference_py)}).read_text()
        custom_code = Path({repr(cfg.custom_py)}).read_text()

        # 注入 pybind_lib（如果存在）
        pybind_lib = os.path.join({repr(cfg.install_path)}, "pybind_lib")
        if os.path.isdir(pybind_lib) and pybind_lib not in sys.path:
            sys.path.insert(0, pybind_lib)

        # 创建后端
        backend = AscendBackend(custom_code, ref_code)

        result = {{
            "tag": {repr(cfg.tag)},
            "install_path": {repr(cfg.install_path)},
            "precision_passed": False,
            "correctness_message": "",
            "time_us": -1,
            "pipeline": {{}},
            "bottleneck": "unknown",
            "profiling_dir": {repr(cfg.profile_dir)},
        }}
"""


def _script_precision_eval() -> str:
    """生成子进程脚本：精度评估段。"""
    return """
        # 1. 精度评估
        try:
            success, message = backend.evaluate_correctness()
            result["precision_passed"] = success
            result["correctness_message"] = message
            tag_str = result["tag"]
            logging.info(f"[{tag_str}] 精度: {'PASS' if success else 'FAIL'} - {message[:200]}")
        except Exception as e:
            result["correctness_message"] = f"精度评估异常: {e}"
            tag_str = result["tag"]
            logging.error(f"[{tag_str}] 精度评估异常: {e}")
"""


def _script_perf_eval(cfg: VersionEvalConfig) -> str:
    """生成子进程脚本：性能评估 + 瓶颈推断段。"""
    return f"""
        # 2. 性能评估（仅在精度通过时执行）
        if result["precision_passed"]:
            try:
                profile_root = Path({repr(cfg.profile_dir)})
                ref_time, ref_data, ref_dir, ref_cv, custom_time, custom_data, custom_dir, custom_cv = \
                    backend.compare_performance_advanced(
                        profile_root=profile_root,
                        num_trials={cfg.num_trials},
                        task_type={repr(cfg.task_type)},
                    )
                result["time_us"] = custom_time
                result["ref_time_us"] = ref_time
                result["cv_pct"] = custom_cv if custom_cv is not None else 0.0

                # 提取 pipeline 信息
                if custom_data:
                    pipeline = {{}}
                    for row in custom_data:
                        if isinstance(row, dict):
                            for key in ("mte2_ratio", "vec_ratio", "scalar_ratio", "mte3_ratio",
                                        "mte2_pct", "vec_pct", "scalar_pct", "mte3_pct"):
                                if key in row:
                                    pipeline[key] = row[key]
                    result["pipeline"] = pipeline

                # 推断瓶颈类型
                if pipeline:
                    mte2 = pipeline.get("mte2_pct", pipeline.get("mte2_ratio", 0))
                    vec = pipeline.get("vec_pct", pipeline.get("vec_ratio", 0))
                    scalar = pipeline.get("scalar_pct", pipeline.get("scalar_ratio", 0))
                    if mte2 > 50:
                        result["bottleneck"] = "memory_bound"
                    elif vec > 60:
                        result["bottleneck"] = "compute_bound"
                    elif scalar > 30:
                        result["bottleneck"] = "scalar_bound"
                    else:
                        result["bottleneck"] = "balanced"

                tag_str = result["tag"]
                logging.info(f"[{{tag_str}}] 性能: custom={{custom_time:.2f}}us, ref={{ref_time:.2f}}us")
            except Exception as e:
                result["correctness_message"] += f"; 性能评估异常: {{e}}"
                tag_str = result["tag"]
                logging.error(f"[{{tag_str}}] 性能评估异常: {{e}}")
"""


def _script_output() -> str:
    """生成子进程脚本：JSON 结果输出段。"""
    return """
        # 输出 JSON 结果
        print("--- EVAL_RESULT_JSON ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("--- END_EVAL_RESULT_JSON ---")
"""


def _build_eval_script(cfg: VersionEvalConfig, opp_path: str, lib_path: str) -> str:
    """生成子进程评估脚本（在隔离进程中跑精度+性能评估）。

    通过 AscendBackend 实现精度与性能评估，支持 OPP 库隔离。
    """
    return textwrap.dedent(
        _script_env_setup(cfg, opp_path, lib_path)
        + _script_precision_eval()
        + _script_perf_eval(cfg)
        + _script_output()
    )


def _run_eval_subprocess(cfg: VersionEvalConfig, script_path: str,
                         opp_path: str, lib_path: str) -> dict:
    """运行评估子进程并解析 EVAL_RESULT_JSON 输出。"""
    try:
        logging.info("[%s] 启动评估子进程: %s", cfg.tag, script_path)

        env = os.environ.copy()
        env["ASCEND_CUSTOM_OPP_PATH"] = opp_path
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib_path + ":" + existing_ld
        env["ASCEND_DEVICE_ID"] = str(cfg.device_id)

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes
            env=env,
        )

        # 解析输出
        stdout = result.stdout
        if "--- EVAL_RESULT_JSON ---" in stdout:
            json_start = stdout.index("--- EVAL_RESULT_JSON ---") + len("--- EVAL_RESULT_JSON ---")
            json_end = stdout.index("--- END_EVAL_RESULT_JSON ---")
            json_str = stdout[json_start:json_end].strip()
            return json.loads(json_str)
        logging.error(
            "[%s] 子进程未输出结果JSON\nstdout (last 2000):\n%s\nstderr (last 2000):\n%s",
            cfg.tag, stdout[-2000:], result.stderr[-2000:],
        )
        return {
            "tag": cfg.tag,
            "error": f"子进程未输出结果: returncode={result.returncode}",
            "precision_passed": False,
            "time_us": -1,
            "stderr_tail": result.stderr[-500:],
        }

    except subprocess.TimeoutExpired:
        logging.error("[%s] 评估子进程超时（600秒）", cfg.tag)
        return {
            "tag": cfg.tag,
            "error": "评估超时",
            "precision_passed": False,
            "time_us": -1,
        }
    except Exception as e:
        logging.error("[%s] 评估异常: %s", cfg.tag, e)
        return {
            "tag": cfg.tag,
            "error": str(e),
            "precision_passed": False,
            "time_us": -1,
        }


def evaluate_single_version(cfg: VersionEvalConfig) -> dict:
    """在子进程中评估单个版本（baseline 或 evolved）。

    通过 AscendBackend 实现精度与性能评估，支持 OPP 库隔离。
    """
    vendor_subdir = detect_vendor_subdir(cfg.install_path)
    opp_path = os.path.join(cfg.install_path, "vendors", vendor_subdir)
    lib_path = os.path.join(opp_path, "op_api", "lib")

    if not os.path.isdir(opp_path):
        return {
            "tag": cfg.tag,
            "error": f"OPP directory not found: {opp_path}",
            "precision_passed": False,
            "time_us": -1,
        }

    # 创建 profiling 目录
    os.makedirs(cfg.profile_dir, exist_ok=True)

    # 生成并写入评估子脚本
    eval_script = _build_eval_script(cfg, opp_path, lib_path)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"eval_{cfg.tag}_",
        delete=False, dir=cfg.profile_dir,
    ) as f:
        f.write(eval_script)
        script_path = f.name

    # 运行子进程（临时脚本保留用于调试）
    return _run_eval_subprocess(cfg, script_path, opp_path, lib_path)


def compare_versions(baseline_result: dict, evolved_result: dict) -> dict:
    """
    计算 baseline 和 evolved 的对比指标。

    Args:
        baseline_result: baseline 评估结果
        evolved_result: evolved 评估结果

    Returns:
        dict: 对比指标
    """
    baseline_time = baseline_result.get("time_us", -1)
    evolved_time = evolved_result.get("time_us", -1)

    comparison = {
        "compilation_success": True,  # 如果到达这里，说明编译成功
        "precision_passed": (
            baseline_result.get("precision_passed", False) and
            evolved_result.get("precision_passed", False)
        ),
    }

    if baseline_time > 0 and evolved_time > 0:
        comparison["speedup"] = baseline_time / evolved_time
        comparison["time_delta_us"] = evolved_time - baseline_time
    else:
        comparison["speedup"] = 0.0
        comparison["time_delta_us"] = 0.0

    # 瓶颈变化
    baseline_bn = baseline_result.get("bottleneck", "unknown")
    evolved_bn = evolved_result.get("bottleneck", "unknown")
    comparison["bottleneck_change"] = f"{baseline_bn} -> {evolved_bn}"

    # 评测质量
    evolved_cv = evolved_result.get("cv_pct", 0.0)
    comparison["cv_pct"] = evolved_cv
    if evolved_cv < 5.0:
        comparison["measurement_quality"] = "good"
    elif evolved_cv < 15.0:
        comparison["measurement_quality"] = "acceptable"
    else:
        comparison["measurement_quality"] = "noisy"

    return comparison


@dataclass
class OpsEvalConfig:
    """evaluate_ops 的参数封装。"""
    op_name: str
    baseline_path: str
    evolved_path: str
    reference_py: str
    custom_py: str
    device_id: int = 0
    task_type: str = "vector"
    output_path: str = None
    num_trials: int = 50
    soc: str = ""
    repo_type: str = ""
    eval_lock: str = None
    eval_lock_timeout: float = 300.0
    baseline_cache: str = None


def _load_baseline_cache(baseline_cache: str):
    """加载 baseline 评估结果缓存，无缓存或不可用返回 None。"""
    if not baseline_cache or not os.path.isfile(baseline_cache):
        return None
    try:
        with open(baseline_cache, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_baseline = cached.get("baseline")
        if cached_baseline and cached_baseline.get("time_us", -1) > 0:
            logging.info("使用 baseline 缓存: %s (time_us=%.2f)",
                         baseline_cache, cached_baseline['time_us'])
            return cached_baseline
    except (json.JSONDecodeError, OSError) as e:
        logging.warning("Baseline 缓存读取失败，将重新评估: %s", e)
    return None


def _run_version_evals(cfg: OpsEvalConfig) -> tuple:
    """在评估锁保护下跑 baseline / evolved 两个版本的评估。"""
    baseline_profile_dir = os.path.join(
        os.path.dirname(cfg.evolved_path), "baseline_profiling"
    )
    evolved_profile_dir = os.path.join(
        os.path.dirname(cfg.evolved_path), "evolved_profiling"
    )

    # 尝试加载 baseline 缓存（在获取锁之前，减少持锁时间）
    baseline_result = _load_baseline_cache(cfg.baseline_cache)

    # 评估排队锁：多个子 agent 共享同一张卡时串行排队
    lock_fd = None
    if cfg.eval_lock:
        logging.info("等待评估锁: %s (超时 %ss)", cfg.eval_lock, cfg.eval_lock_timeout)
        lock_fd = _acquire_eval_lock(cfg.eval_lock, cfg.eval_lock_timeout)
        logging.info("评估锁已获取，开始评估")

    try:
        # 仅在无缓存时评估 baseline
        if baseline_result is None:
            logging.info("评估 baseline: %s", cfg.baseline_path)
            baseline_result = evaluate_single_version(VersionEvalConfig(
                op_name=cfg.op_name,
                install_path=cfg.baseline_path,
                reference_py=cfg.reference_py,
                custom_py=cfg.custom_py,
                device_id=cfg.device_id,
                task_type=cfg.task_type,
                profile_dir=baseline_profile_dir,
                num_trials=cfg.num_trials,
                tag="baseline",
            ))

        logging.info("评估 evolved: %s", cfg.evolved_path)
        evolved_result = evaluate_single_version(VersionEvalConfig(
            op_name=cfg.op_name,
            install_path=cfg.evolved_path,
            reference_py=cfg.reference_py,
            custom_py=cfg.custom_py,
            device_id=cfg.device_id,
            task_type=cfg.task_type,
            profile_dir=evolved_profile_dir,
            num_trials=cfg.num_trials,
            tag="evolved",
        ))
    finally:
        if lock_fd is not None:
            _release_eval_lock(lock_fd)
            logging.info("评估锁已释放")

    return baseline_result, evolved_result


def evaluate_ops(cfg: OpsEvalConfig) -> dict:
    """对比评估 baseline 和 evolved 两个版本。

    Returns:
        dict: 完整对比结果
    """
    baseline_result, evolved_result = _run_version_evals(cfg)

    # 对比
    comparison = compare_versions(baseline_result, evolved_result)

    # 组装最终结果
    final_result = {
        "op_name": cfg.op_name,
        "repo_type": cfg.repo_type,
        "soc": cfg.soc,
        "baseline": baseline_result,
        "evolved": evolved_result,
        "comparison": comparison,
        "eval_backend": "python_npu_event",
    }

    # Baseline sanity check: warn if baseline time seems abnormally high
    # (likely measuring host-side e2e instead of kernel-only)
    baseline_time = baseline_result.get("time_us", -1)
    evolved_time = evolved_result.get("time_us", -1)
    if baseline_time > 0 and evolved_time > 0:
        ratio = baseline_time / evolved_time if evolved_time > 0 else 0
        if baseline_time > 2000 and ratio > 0.8 and ratio < 1.2:
            logging.warning(
                "Baseline time %.1fus is unusually high and close to evolved "
                "time %.1fus. This backend (python_npu_event) measures host-side "
                "end-to-end time including framework overhead. If comparing with "
                "forge (msprof kernel time), results are NOT directly comparable.",
                baseline_time, evolved_time,
            )

    # 输出
    output_path = cfg.output_path
    if output_path is None:
        output_path = os.path.join(cfg.evolved_path, "evaluation_results.json")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    logging.info("评估结果已保存到: %s", output_path)

    _log_eval_summary(cfg.op_name, baseline_result, evolved_result, comparison)

    return final_result


def _log_eval_summary(op_name: str, baseline_result: dict,
                      evolved_result: dict, comparison: dict):
    """打印评估结果摘要到日志。"""
    logging.info("")
    logging.info("=" * 60)
    logging.info("评估结果摘要: %s", op_name)
    logging.info("=" * 60)
    logging.info("  Baseline:")
    logging.info("    精度: %s", 'PASS' if baseline_result.get('precision_passed') else 'FAIL')
    logging.info("    耗时: %.2f us", baseline_result.get('time_us', -1))
    logging.info("    瓶颈: %s", baseline_result.get('bottleneck', 'unknown'))
    logging.info("  Evolved:")
    logging.info("    精度: %s", 'PASS' if evolved_result.get('precision_passed') else 'FAIL')
    logging.info("    耗时: %.2f us", evolved_result.get('time_us', -1))
    logging.info("    瓶颈: %s", evolved_result.get('bottleneck', 'unknown'))
    logging.info("  Comparison:")
    logging.info("    加速比: %.3fx", comparison.get('speedup', 0))
    logging.info("    耗时差异: %.2f us", comparison.get('time_delta_us', 0))
    logging.info("    瓶颈变化: %s", comparison.get('bottleneck_change', 'unknown'))
    logging.info("=" * 60)


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ops repository operator: baseline vs evolved"
    )
    parser.add_argument("op_name", type=str, help="算子名")
    parser.add_argument("--baseline-path", required=True, help="baseline 安装路径")
    parser.add_argument("--evolved-path", required=True, help="evolved 安装路径")
    parser.add_argument("--reference-py", required=True, help="参考实现 Python 文件路径")
    parser.add_argument("--custom-py", required=True, help="自定义算子 Python 文件路径")
    parser.add_argument("--device-id", type=int, default=0,
                        help="NPU 设备 ID (default: 0)")
    parser.add_argument("--task-type", type=str, default="vector",
                        choices=["vector", "cube", "cv-mix", "unknown"],
                        help="算子类型 (default: vector)")
    parser.add_argument("--output", type=str, default=None,
                        help="评估结果输出路径 (default: evolved_path/evaluation_results.json)")
    parser.add_argument("--num-trials", type=int, default=50,
                        help="profiling 试验次数 (default: 50)")
    parser.add_argument("--soc", type=str, default="",
                        help="目标芯片 (如 ascend910b)")
    parser.add_argument("--repo-type", type=str, default="",
                        help="仓类型 (nn/cv/math/transformer)")
    parser.add_argument("--eval-lock", type=str, default=None,
                        help="评估排队锁文件路径。多个子 agent 共享同一张卡时，通过此锁串行排队评估")
    parser.add_argument("--eval-lock-timeout", type=float, default=300,
                        help="评估锁等待超时秒数 (default: 300)")
    parser.add_argument("--baseline-cache", type=str, default=None,
                        help="baseline 评估结果缓存文件路径（如 baseline_evaluation.json）。"
                             "若指定且文件存在，跳过 baseline 评估直接复用，减少持锁时间")
    return parser


def _config_from_args(args) -> OpsEvalConfig:
    return OpsEvalConfig(
        op_name=args.op_name,
        baseline_path=args.baseline_path,
        evolved_path=args.evolved_path,
        reference_py=args.reference_py,
        custom_py=args.custom_py,
        device_id=args.device_id,
        task_type=args.task_type,
        output_path=args.output,
        num_trials=args.num_trials,
        soc=args.soc,
        repo_type=args.repo_type,
        eval_lock=args.eval_lock,
        eval_lock_timeout=args.eval_lock_timeout,
        baseline_cache=args.baseline_cache,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = _build_main_parser().parse_args()
    evaluate_ops(_config_from_args(args))


if __name__ == "__main__":
    main()
