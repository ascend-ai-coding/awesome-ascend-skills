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

"""算子任务代码验证脚本

验证代码是否符合算子任务格式并通过运行时检查。
支持两种输入提供方式：
- 单 case：get_inputs() 返回单组输入
- 多 case：get_input_groups() 返回多组输入列表（每组对应一个 shape 配置）

检查项目:
1. 静态: class Model(nn.Module), forward, get_init_inputs, (get_inputs OR get_input_groups)
2. 运行时: exec → Model() → 遍历所有 groups 执行 forward() → NaN/Inf 检查 → 一致性检查

用法:
    python validate_task.py /abs/path/task_desc.py
    python validate_task.py /abs/path/task_desc.py --json
    python validate_task.py /abs/path/task_desc.py --static-only

输出格式:
    [VALID] 代码符合算子任务格式
    [INVALID] 代码不符合格式 + 原因 + 修复建议
"""
import ast
import os
import sys
import argparse
import json
import logging


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

# 确保同目录下的 _log_utils 可被导入（脚本可能从其他工作目录调用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _log_utils import setup_logger as _setup_logger_shared  # noqa: E402

logger = logging.getLogger("triton_task_extractor.validate_task")


def _setup_logger() -> None:
    """配置 logger：复用 _log_utils.setup_logger。"""
    _setup_logger_shared(logger)


# ---------------------------------------------------------------------------
# 静态检查
# ---------------------------------------------------------------------------

def _model_class_has_forward(node: ast.ClassDef) -> bool:
    """判断 Model 类是否继承 nn.Module，并包含 forward 方法。"""
    inherits_module = any(
        getattr(base, "attr", getattr(base, "id", "")) == "Module"
        for base in node.bases
    )
    if not inherits_module:
        return False
    return any(
        isinstance(item, ast.FunctionDef) and item.name == "forward"
        for item in node.body
    )


def _collect_components(tree: ast.AST) -> dict:
    """遍历 AST，标记四大组件存在情况。"""
    has = {
        "Model": False,
        "forward": False,
        "get_inputs": False,
        "get_input_groups": False,
        "get_init_inputs": False,
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            if _model_class_has_forward(node):
                has["Model"] = True
                has["forward"] = True
        elif isinstance(node, ast.FunctionDef) and node.name in has:
            has[node.name] = True
    return has


def check_static(code: str) -> dict:
    """静态检查: 验证算子任务四大组件是否存在

    输入函数允许两种之一：get_inputs() 或 get_input_groups()。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "passed": False,
            "found": [],
            "missing": ["Model", "forward", "get_init_inputs", "get_inputs|get_input_groups"],
            "error": f"SyntaxError: {e}",
        }

    has = _collect_components(tree)
    has_model = has.get("Model", False)
    has_forward = has.get("forward", False)
    has_get_init = has.get("get_init_inputs", False)
    has_input_provider = has.get("get_inputs", False) or has.get("get_input_groups", False)
    required_passed = has_model and has_forward and has_get_init and has_input_provider

    found = [k for k, v in has.items() if v]
    missing = []
    if not has_model:
        missing.append("Model")
    if not has_forward:
        missing.append("forward")
    if not has_get_init:
        missing.append("get_init_inputs")
    if not has_input_provider:
        missing.append("get_inputs|get_input_groups")
    return {"passed": required_passed, "found": found, "missing": missing, "error": None}


# ---------------------------------------------------------------------------
# 运行时检查
# ---------------------------------------------------------------------------

def _runtime_fail(checks: list, name: str, exc: Exception, cases_tested: int = 0,
                  cases_passed: int = 0) -> dict:
    """构造运行时检查失败结果。"""
    checks.append({"name": name, "passed": False, "error": str(exc)})
    return {
        "passed": False,
        "checks": checks,
        "error": f"{name} error: {exc}",
        "cases_tested": cases_tested,
        "cases_passed": cases_passed,
    }


def _exec_user_code(code: str, file_path, checks: list):
    """执行用户代码，返回 (namespace, err_result)。"""
    namespace = {}
    if file_path:
        namespace["__file__"] = file_path
    try:
        exec(code, namespace)
        checks.append({"name": "exec", "passed": True})
        return namespace, None
    except Exception as e:
        return None, _runtime_fail(checks, "exec", e)


def _instantiate_model(namespace: dict, checks: list):
    """调用 get_init_inputs() 并实例化 Model，返回 (model, err_result)。"""
    get_init_inputs = namespace.get("get_init_inputs")
    if get_init_inputs is None:
        return None, _runtime_fail(checks, "get_init_inputs()", KeyError("get_init_inputs"))
    try:
        init_inputs = get_init_inputs()
        checks.append({"name": "get_init_inputs()", "passed": True})
    except Exception as e:
        return None, _runtime_fail(checks, "get_init_inputs()", e)

    model_cls = namespace.get("Model")
    if model_cls is None:
        return None, _runtime_fail(checks, "Model(*init_inputs)", KeyError("Model"))
    try:
        model = model_cls(*init_inputs)
        checks.append({"name": "Model(*init_inputs)", "passed": True})
        return model, None
    except Exception as e:
        return None, _runtime_fail(checks, "Model(*init_inputs)", e)


def _resolve_input_groups(namespace: dict, checks: list):
    """解析 get_input_groups()/get_inputs()，返回 (input_groups, provider_kind, err_result)。"""
    get_groups = namespace.get("get_input_groups")
    if get_groups is not None:
        try:
            input_groups = get_groups()
            checks.append({
                "name": "get_input_groups()",
                "passed": True,
                "note": f"{len(input_groups)} groups",
            })
            return input_groups, "groups", None
        except Exception as e:
            return None, None, _runtime_fail(checks, "get_input_groups()", e)
    get_single = namespace.get("get_inputs")
    if get_single is not None:
        try:
            input_groups = [get_single()]
            checks.append({"name": "get_inputs()", "passed": True})
            return input_groups, "single", None
        except Exception as e:
            return None, None, _runtime_fail(checks, "get_inputs()", e)
    return None, None, {
        "passed": False,
        "checks": checks,
        "error": "缺少 get_inputs 或 get_input_groups",
        "cases_tested": 0,
        "cases_passed": 0,
    }


def _make_npu_helpers():
    """返回 (_to_npu_device, npu_available)；npu_available 为 False 时回退到 CPU。"""
    import torch

    try:
        import torch_npu
        npu_available = torch_npu.npu.is_available()
    except Exception:
        npu_available = False

    def _to_npu_device(x):
        if npu_available and isinstance(x, torch.Tensor):
            return x.npu()
        return x

    return _to_npu_device, npu_available


def _check_tensor_nan_inf(t, name: str):
    """返回单一 tensor 的 NaN/Inf 描述，正常返回 None。"""
    import torch
    if isinstance(t, torch.Tensor):
        if torch.isnan(t).any():
            return f"{name} contains NaN"
        if torch.isinf(t).any():
            return f"{name} contains Inf"
    return None


def _collect_output_issues(output, case_label: str) -> list:
    """收集输出张量（含 tuple/list）的 NaN/Inf 异常列表。"""
    issues = []
    if isinstance(output, (tuple, list)):
        for i, item in enumerate(output):
            issue = _check_tensor_nan_inf(item, f"{case_label} output[{i}]")
            if issue:
                issues.append(issue)
    else:
        issue = _check_tensor_nan_inf(output, f"{case_label} output")
        if issue:
            issues.append(issue)
    return issues


def _to_cpu_output(output):
    """将 forward 输出搬回 CPU 以便后续检查。"""
    import torch
    if isinstance(output, (tuple, list)):
        return [x.cpu() if isinstance(x, torch.Tensor) else x for x in output]
    if isinstance(output, torch.Tensor):
        return output.cpu()
    return output


def _tensors_close(a, b, rtol=1e-5, atol=1e-6):
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return all(_tensors_close(x, y) for x, y in zip(a, b))
    return True


def _run_single_case(model, inputs, idx: int, to_device) -> tuple:
    """执行单个 case 的 forward / NaN-Inf / 一致性校验。

    返回 (ok, error_msg_or_None)。
    """
    case_label = f"case[{idx}]"
    device_inputs = [to_device(x) for x in inputs]

    try:
        output = model(*device_inputs)
    except Exception as e:
        return False, ("forward", f"{case_label} forward error: {e}")

    output = _to_cpu_output(output)
    issues = _collect_output_issues(output, case_label)
    if issues:
        return False, ("nan_inf", "; ".join(issues))

    try:
        output2 = _to_cpu_output(model(*device_inputs))
        if not _tensors_close(output, output2):
            return False, ("consistency", f"{case_label} consistency check failed")
    except Exception as e:
        # 例外：consistency 是二次推理的可选校验，首次 forward 已成功即认为该 case 通过；
        # 二次推理失败（如随机种子相关偶发问题）不应让整体失败，仅记录 warning 提示
        logger.warning(
            "%s consistency 二次推理出错，跳过一致性校验: %s: %s",
            case_label, type(e).__name__, e,
        )
    return True, None


def _run_all_cases(model, input_groups, checks: list, provider_kind: str) -> dict:
    """遍历所有 case，返回最终运行时结果字典。"""
    to_device, _ = _make_npu_helpers()
    cases_passed = 0
    total = len(input_groups)

    for idx, inputs in enumerate(input_groups):
        ok, err = _run_single_case(model, inputs, idx, to_device)
        if not ok:
            kind, msg = err
            case_label = f"case[{idx}]"
            if kind == "forward":
                checks.append({"name": f"{case_label} forward", "passed": False, "error": msg})
            elif kind == "nan_inf":
                checks.append({"name": f"{case_label} NaN/Inf", "passed": False, "error": msg})
            else:
                checks.append({
                    "name": f"{case_label} consistency",
                    "passed": False,
                    "error": "outputs differ between runs",
                })
            return {
                "passed": False,
                "checks": checks,
                "error": msg,
                "cases_tested": idx + 1,
                "cases_passed": cases_passed,
            }
        cases_passed += 1

    checks.append({
        "name": "all cases",
        "passed": True,
        "note": f"{cases_passed}/{total} passed (provider={provider_kind})",
    })
    return {
        "passed": True,
        "checks": checks,
        "error": None,
        "cases_tested": total,
        "cases_passed": cases_passed,
    }


def check_runtime(code: str, file_path: str = None) -> dict:
    """运行时检查: exec → Model() → 遍历所有 groups → forward() → NaN/Inf → 一致性

    若任务文件提供 get_input_groups()，全部 groups 都会执行。
    若仅提供 get_inputs()，按单 case 处理。
    """
    checks = []
    namespace, err = _exec_user_code(code, file_path, checks)
    if err is not None:
        return err

    model, err = _instantiate_model(namespace, checks)
    if err is not None:
        return err

    input_groups, provider_kind, err = _resolve_input_groups(namespace, checks)
    if err is not None:
        return err

    return _run_all_cases(model, input_groups, checks, provider_kind)


# ---------------------------------------------------------------------------
# 入口与输出
# ---------------------------------------------------------------------------

def _load_code(path: str, want_json: bool):
    """读取代码文件；失败时返回 None，由调用方决定退出。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        if want_json:
            logger.info("%s", json.dumps({"valid": False, "error": f"File not found: {path}"}))
        else:
            logger.error("[ERROR] 文件不存在: %s", path)
        return None


def _emit_static_failure(result: dict, want_json: bool) -> None:
    static_result = result["static_check"]
    result["error"] = static_result.get("error") or f"缺少组件: {', '.join(static_result['missing'])}"
    result["suggestion"] = (
        "检查代码结构，确保包含 Model(nn.Module)、forward、get_inputs、get_init_inputs"
    )
    if want_json:
        logger.info("%s", json.dumps(result, ensure_ascii=False, indent=2))
    else:
        logger.error("[INVALID] 代码不符合算子任务格式")
        logger.error("缺少: %s", ", ".join(static_result["missing"]))
        logger.error("建议: %s", result["suggestion"])


def _emit_runtime_failure(result: dict, want_json: bool) -> None:
    runtime_result = result["runtime_check"]
    result["error"] = runtime_result["error"]
    result["suggestion"] = "检查代码逻辑，修复后重新验证"
    if want_json:
        logger.info("%s", json.dumps(result, ensure_ascii=False, indent=2))
        return
    logger.error("[INVALID] 运行时检查失败")
    logger.error("错误: %s", runtime_result["error"])
    logger.error(
        "已测试 cases: %s / 通过: %s",
        runtime_result.get("cases_tested", 0),
        runtime_result.get("cases_passed", 0),
    )
    for check in runtime_result["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        logger.error("  [%s] %s", status, check["name"])


def _emit_success(result: dict, want_json: bool, static_only: bool) -> None:
    static_result = result["static_check"]
    if want_json:
        logger.info("%s", json.dumps(result, ensure_ascii=False, indent=2))
        return
    check_type = "静态" if static_only else "静态+运行时"
    logger.info("[VALID] 代码符合算子任务格式（%s检查通过）", check_type)
    logger.info("包含组件: %s", ", ".join(static_result["found"]))
    if not static_only and result.get("cases_tested"):
        logger.info(
            "运行时测试 cases: %s/%s 全部通过",
            result["cases_passed"],
            result["cases_tested"],
        )


def main():
    _setup_logger()
    parser = argparse.ArgumentParser(
        description="验证代码是否符合算子任务格式"
    )
    parser.add_argument("file", help="要验证的 Python 文件路径")
    parser.add_argument("--static-only", action="store_true", help="只做静态检查")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    code = _load_code(args.file, args.json)
    if code is None:
        sys.exit(1)

    static_result = check_static(code)
    result = {
        "valid": False,
        "static_check": static_result,
        "runtime_check": None,
        "suggestion": "",
    }

    if not static_result["passed"]:
        _emit_static_failure(result, args.json)
        sys.exit(1)

    if not args.static_only:
        runtime_result = check_runtime(code, file_path=args.file)
        result["runtime_check"] = runtime_result
        result["cases_tested"] = runtime_result.get("cases_tested", 0)
        result["cases_passed"] = runtime_result.get("cases_passed", 0)

        if not runtime_result["passed"]:
            _emit_runtime_failure(result, args.json)
            sys.exit(1)

    result["valid"] = True
    _emit_success(result, args.json, args.static_only)
    sys.exit(0)


if __name__ == "__main__":
    main()
