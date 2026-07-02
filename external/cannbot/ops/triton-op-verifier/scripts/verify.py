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

"""算子验证脚本 — 对比框架实现 (Model) 与生成实现 (ModelNew) 的输出一致性。

多 shape 模式下：每个 shape 独立 try/except，全部跑完后落盘 verify_result.json。
策略 A：passed < total 即整体判失败（exit 1），同时失败清单记录在 JSON 的 `failures` 字段。

用法:
    python verify.py --op_name <算子名> [--verify_dir <验证目录>] [--timeout <超时秒数>]
"""
import argparse
import gc
import importlib
import json
import logging
import os
import sys
import subprocess
import traceback
from dataclasses import dataclass


ERROR_MSG_LIMIT = 2000


@dataclass
class CaseContext:
    """单个测试用例在整体序列中的定位（1-based）。"""
    case_idx: int
    total_cases: int


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

# 确保同目录下的 _log_utils 可被导入（脚本可能从其他工作目录调用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _log_utils import setup_logger as _setup_logger_shared  # noqa: E402	 
from _common_utils import describe_input as _describe_input_shared  # noqa: E402

logger = logging.getLogger("triton_op_verifier.verify")


def _setup_logger() -> None:
    """配置 logger：复用 _log_utils.setup_logger。"""
    _setup_logger_shared(logger)


def truncate_error(msg: str, limit: int = ERROR_MSG_LIMIT) -> str:
    if msg is None:
        return ""
    if len(msg) <= limit:
        return msg
    half = limit // 2
    return f"{msg[:half]}\n... [truncated {len(msg) - limit} chars] ...\n{msg[-half:]}"


def describe_input(inputs):
    """输入列表的结构化描述（用于 JSON）。"""
    return _describe_input_shared(inputs)


def cleanup_npu_memory():
    try:
        import torch
        import torch_npu  # noqa: F401
        torch.npu.empty_cache()
    except Exception as e:
        # 例外：非 NPU 环境或 torch_npu 不可用时清理无意义，仅记录调试信息，不影响主流程
        logger.debug("跳过 NPU 显存清理（环境不支持 torch_npu）: %s: %s", type(e).__name__, e)
    gc.collect()


def get_limit(data_type):
    """根据数据类型获取精度阈值 - 使用 2 的幂次方阈值（与 NPU Benchmark 标准一致）

    参考文档: 精度对比方法.md
    数据类型: FLOAT16, BFLOAT16, FLOAT32, HiFloat32, FLOAT8 E4M3, FLOAT8 E5M2
    判定标准: MERE < threshold 且 MARE < 10 * threshold

    阈值表:
    | 数据类型      | 阈值 (2^n)      | 十进制值       |
    |--------------|----------------|---------------|
    | FLOAT16      | 2^{-10}        | 0.0009765625  |
    | BFLOAT16     | 2^{-7}         | 0.0078125     |
    | FLOAT32      | 2^{-13}        | 0.0001220703  |
    | HiFloat32    | 2^{-11}        | 0.0004882812  |
    | FLOAT8 E4M3  | 2^{-3}         | 0.125         |
    | FLOAT8 E5M2  | 2^{-2}         | 0.25          |

    由于 torch.dtype 中没有直接定义 HiFloat32，可通过字符串传入 "hifloat32" 获取对应阈值。
    """  # noqa: E501
    import torch

    # 支持字符串类型（用于 HiFloat32 或其他自定义类型）
    if isinstance(data_type, str):
        str_to_threshold = {
            "float16": 2**(-10),
            "bfloat16": 2**(-7),
            "float32": 2**(-13),
            "hifloat32": 2**(-11),
            "float8_e4m3": 2**(-3),
            "float8_e5m2": 2**(-2),
            "fp8_e4m3": 2**(-3),
            "fp8_e5m2": 2**(-2),
        }
        return str_to_threshold.get(data_type.lower(), 2**(-13))

    # torch.dtype 类型映射
    dtype_threshold_map = {
        torch.float16: 2**(-10),    # FLOAT16
        torch.bfloat16: 2**(-7),    # BFLOAT16
        torch.float32: 2**(-13),    # FLOAT32
    }

    # 安全获取 FP8 类型（PyTorch 2.0+ 支持）
    # FLOAT8 E4M3: 2^{-3}
    float8_e4m3 = getattr(torch, 'float8_e4m3fn', None) or getattr(torch, 'float8_e4m3', None)
    if float8_e4m3 is not None:
        dtype_threshold_map[float8_e4m3] = 2**(-3)

    # FLOAT8 E5M2: 2^{-2}
    float8_e5m2 = getattr(torch, 'float8_e5m2fn', None) or getattr(torch, 'float8_e5m2', None)
    if float8_e5m2 is not None:
        dtype_threshold_map[float8_e5m2] = 2**(-2)

    return dtype_threshold_map.get(data_type, 2**(-13))


def resolve_input_provider(torch_module):
    """解析任务文件的输入提供方式。"""
    if hasattr(torch_module, "get_input_groups"):
        groups = torch_module.get_input_groups()
        return groups, len(groups)
    elif hasattr(torch_module, "get_inputs"):
        return [torch_module.get_inputs()], 1
    else:
        raise AttributeError(
            f"模块必须提供 get_inputs() 或 get_input_groups() 方法"
        )


def compare(fw_out, impl_out, data_type):
    """对比框架输出和实现输出"""
    import torch
    fw_flat = fw_out.flatten().detach().cpu()
    impl_flat = impl_out.flatten()
    if isinstance(impl_flat, torch.Tensor):
        impl_flat = impl_flat.detach().cpu()
    else:
        impl_flat = torch.tensor(impl_flat, dtype=fw_flat.dtype)

    size = fw_flat.numel()

    if fw_flat.shape != impl_flat.shape:
        raise AssertionError(
            f"验证失败，输出形状不一致: framework={fw_flat.shape}, impl={impl_flat.shape}"
        )

    fw_nan_mask = torch.isnan(fw_flat)
    impl_nan_mask = torch.isnan(impl_flat)
    if not torch.equal(fw_nan_mask, impl_nan_mask):
        fw_nan_count = fw_nan_mask.sum().item()
        impl_nan_count = impl_nan_mask.sum().item()
        raise AssertionError(
            f"验证失败，NaN 位置不匹配: Framework={fw_nan_count}/{size}, "
            f"Implementation={impl_nan_count}/{size}"
        )

    fw_inf_mask = torch.isinf(fw_flat)
    impl_inf_mask = torch.isinf(impl_flat)
    if not torch.equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = fw_inf_mask.sum().item()
        impl_inf_count = impl_inf_mask.sum().item()
        raise AssertionError(
            f"验证失败，Inf 位置不匹配: Framework={fw_inf_count}/{size}, "
            f"Implementation={impl_inf_count}/{size}"
        )
    if fw_inf_mask.any():
        if not torch.equal(
            torch.sign(fw_flat[fw_inf_mask]),
            torch.sign(impl_flat[impl_inf_mask]),
        ):
            raise AssertionError("验证失败，Inf 符号不匹配")

    finite_mask = torch.isfinite(fw_flat) & torch.isfinite(impl_flat)
    finite_count = finite_mask.sum().item()
    if finite_count == 0:
        logger.warning("警告: 所有值都是非有限值，跳过精度检查")
        return

    fw_finite = fw_flat[finite_mask]
    impl_finite = impl_flat[finite_mask]

    if fw_finite.dtype == torch.bool:
        if not torch.equal(fw_finite, impl_finite):
            raise AssertionError(f"验证失败，布尔值不匹配: dtype={data_type}")
        return

    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.to(fw_finite.dtype)

    # 执行 NPU Benchmark 精度验证
    _check_accuracy_npu_benchmark(fw_finite, impl_finite, data_type)


def _check_accuracy_npu_benchmark(golden, actual, data_type):
    """执行 NPU Benchmark 精度验证。

    根据精度对比方法文档，验证两个张量的数值一致性：
    - 计算 MERE（平均相对误差）和 MARE（最大相对误差）
    - 使用 2 的幂次方作为阈值
    - 判定标准：MERE < threshold 且 MARE < 10 * threshold

    Args:
        golden: 参考输出（金标准）
        actual: 被测实现输出
        data_type: 数据类型，用于获取对应的阈值

    Raises:
        AssertionError: 当精度验证未通过时
    """
    import torch

    # 统一转换为 float32 进行计算
    golden_f = golden.float()
    actual_f = actual.float()

    # 先取 dtype 阈值，用作分母下界 clamp。
    # 当 |y_ref| < threshold 时，按 |diff| / threshold 衡量，等价于
    # "参考值已小到 dtype 精度极限时，改用绝对误差归一化"，避免零值/极小值附近误报。
    threshold = get_limit(data_type)

    diff = (actual_f - golden_f).abs()
    denom = golden_f.abs().clamp(min=threshold)
    relative_error = diff / denom

    # 计算误差指标
    mere = relative_error.mean().item()  # 平均相对误差 (Mean Relative Error)
    mare = relative_error.max().item()   # 最大相对误差 (Max Relative Error)

    # 判定标准：MERE < t 且 MARE < 10t
    is_pass = (mere < threshold) and (mare < 10 * threshold)

    if not is_pass:
        # 收集错误信息
        mismatch_mask = relative_error > threshold
        mismatch_indices = torch.where(mismatch_mask)[0]
        num_to_show = min(10, len(mismatch_indices))

        error_msg = (
            f"验证失败，输出不一致: MERE={mere:.6e}, MARE={mare:.6e}, "
            f"dtype={data_type}, threshold={threshold}\n"
        )
        if len(mismatch_indices) > 0:
            error_msg += f"前 {num_to_show} 个超出阈值的值:\n"
            for i in range(num_to_show):
                idx = mismatch_indices[i].item()
                error_msg += (
                    f"  位置[{idx}]: framework={golden[idx]:.6e}, "
                    f"impl={actual[idx]:.6e}, "
                    f"相对误差={relative_error[idx]:.6e}\n"
                )
        raise AssertionError(error_msg)


def run_single_case(framework_model, impl_model, inputs, device, case_ctx: CaseContext):
    """验证单组输入。失败时抛出 AssertionError。"""
    import torch

    case_idx = case_ctx.case_idx
    total_cases = case_ctx.total_cases
    logger.info("  测试第 %d/%d 组输入...", case_idx, total_cases)

    inputs_for_impl = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]
    inputs_for_framework = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]

    with torch.no_grad():
        impl_output = impl_model(*inputs_for_impl)
        framework_output = framework_model(*inputs_for_framework)

    if not isinstance(framework_output, (list, tuple)):
        framework_output = [framework_output]
    if not isinstance(impl_output, (list, tuple)):
        impl_output = [impl_output]

    if len(framework_output) != len(impl_output):
        raise AssertionError(
            f"[用例 {case_idx}/{total_cases}] 输出数量不一致: "
            f"framework={len(framework_output)}, impl={len(impl_output)}"
        )

    for i, (fw_out, impl_out) in enumerate(zip(framework_output, impl_output)):
        if fw_out is None or impl_out is None:
            raise AssertionError(
                f"[用例 {case_idx}/{total_cases}] 输出 {i} 为 None: "
                f"framework={fw_out is None}, impl={impl_out is None}"
            )
        if isinstance(fw_out, torch.Tensor) and isinstance(impl_out, torch.Tensor):
            try:
                data_type = fw_out.dtype
                compare(fw_out, impl_out, data_type)
            except AssertionError as e:
                raise AssertionError(f"[用例 {case_idx}/{total_cases}] {str(e)}") from e


def _load_verify_modules(op_name, verify_dir, triton_impl_name):
    """导入 framework / impl 模块并解析关键符号。"""
    sys.path.insert(0, verify_dir)
    torch_module = importlib.import_module(f"{op_name}_torch")
    impl_module = importlib.import_module(f"{op_name}_{triton_impl_name}")
    return {
        "torch_module": torch_module,
        "framework_cls": torch_module.Model,
        "impl_cls": impl_module.ModelNew,
        "get_init_inputs": torch_module.get_init_inputs,
    }


def _instantiate_models(framework_cls, impl_cls, get_init_inputs, device):
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


def _try_run_case(modules, case_idx, inputs, total_cases, device):
    """执行单个 case：成功返回 (True, None)；失败返回 (False, failure_dict)。"""
    framework_model = None
    impl_model = None
    try:
        framework_model, impl_model = _instantiate_models(
            modules["framework_cls"],
            modules["impl_cls"],
            modules["get_init_inputs"],
            device,
        )
        run_single_case(
            framework_model, impl_model, inputs, device,
            CaseContext(case_idx=case_idx, total_cases=total_cases),
        )
        return True, None
    except Exception as e:
        err_detail = traceback.format_exc()
        logger.error(
            "  [用例 %d/%d] 失败: %s: %s",
            case_idx, total_cases, type(e).__name__, e,
        )
        failure = {
            "case_idx": case_idx,
            "input_desc": describe_input(inputs),
            "error_type": type(e).__name__,
            "error_msg": truncate_error(err_detail),
        }
        return False, failure
    finally:
        del framework_model
        del impl_model
        cleanup_npu_memory()


def _write_verify_result(output_path, result):
    """落盘 verify_result.json；失败仅警告。"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("验证结果已保存到: %s", output_path)
    except Exception as e:
        logger.warning("警告: 无法写入 verify_result.json: %s", e)


def verify_implementations(op_name, verify_dir, triton_impl_name="triton_ascend_impl", output_path=None):
    """验证框架实现和生成实现的结果一致性。

    每个 shape 独立 try/except，全部跑完后写 verify_result.json。

    Returns:
        (passed_cases, total_cases)
    """
    import torch
    import torch_npu  # noqa: F401

    modules = _load_verify_modules(op_name, verify_dir, triton_impl_name)

    # 在获取输入之前设置种子，确保随机生成的输入可复现
    torch.manual_seed(0)
    torch.npu.manual_seed(0)

    input_groups, total_cases = resolve_input_provider(modules["torch_module"])
    device = torch.device("npu")

    failures = []
    passed_cases = 0
    for case_idx, inputs in enumerate(input_groups, start=1):
        ok, failure = _try_run_case(modules, case_idx, inputs, total_cases, device)
        if ok:
            passed_cases += 1
        else:
            failures.append(failure)

    failed_cases = total_cases - passed_cases

    if output_path is None:
        output_path = os.path.join(verify_dir, "verify_result.json")
    result = {
        "op_name": op_name,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "failures": failures,
    }
    _write_verify_result(output_path, result)

    if failed_cases == 0:
        logger.info("验证成功：共 %d 组测试用例全部通过", total_cases)
    else:
        logger.error(
            "验证失败：%d/%d 组通过，%d 组失败（详见 %s）",
            passed_cases,
            total_cases,
            failed_cases,
            output_path,
        )

    return passed_cases, total_cases


if __name__ == "__main__":
    _setup_logger()
    parser = argparse.ArgumentParser(description="算子验证脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument(
        "--verify_dir", default=".",
        help="验证目录，包含 {op_name}_torch.py 和 {op_name}_triton_ascend_impl.py（默认当前目录）",
    )
    parser.add_argument("--timeout", type=int, default=900, help="超时秒数（默认 900）")
    parser.add_argument(
        "--triton_impl_name", default="triton_ascend_impl",
        help="Triton 实现模块名（不含 op_name 前缀，默认 triton_ascend_impl）",
    )
    parser.add_argument(
        "--output", default=None,
        help="验证结果 JSON 输出路径（默认 {verify_dir}/verify_result.json）",
    )
    parser.add_argument(
        "--subprocess", action="store_true",
        help=argparse.SUPPRESS,  # 内部参数：子进程模式，直接执行验证
    )
    args = parser.parse_args()

    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        logger.error("错误: 验证目录不存在: %s", verify_dir)
        sys.exit(1)

    if args.subprocess:
        # 子进程模式：直接执行验证逻辑
        try:
            passed, total = verify_implementations(
                args.op_name, verify_dir, args.triton_impl_name, args.output
            )
        except Exception as e:
            logger.error("%s", e)
            logger.error("%s", traceback.format_exc())
            sys.exit(1)
        # 策略 A：passed < total → exit 1
        sys.exit(0 if passed == total and total > 0 else 1)
    else:
        # 主进程模式：启动子进程执行验证，超时后 kill 整个进程树
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--op_name", args.op_name,
            "--verify_dir", verify_dir,
            "--triton_impl_name", args.triton_impl_name,
            "--subprocess",
        ]
        if args.output:
            cmd.extend(["--output", args.output])
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate(timeout=args.timeout)

            sys.stdout.buffer.write(stdout)
            sys.stdout.buffer.flush()
            sys.stderr.buffer.write(stderr)
            sys.stderr.buffer.flush()
            sys.exit(proc.returncode)

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            logger.error("验证超时（%d秒），已终止子进程", args.timeout)
            sys.exit(1)
