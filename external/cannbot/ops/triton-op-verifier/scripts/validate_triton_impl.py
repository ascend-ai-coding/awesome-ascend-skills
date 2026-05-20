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

"""Triton 实现退化检测脚本 — 通过 AST 静态分析检查生成代码是否退化为 PyTorch 原生实现。

检测三种退化类型：
  Type 1: 无 @triton.jit kernel，全部使用 PyTorch
  Type 2: 有 @triton.jit kernel 定义但 forward() 未调用
  Type 3: forward() 调用了 kernel 但仍有部分计算使用 torch 接口

用法:
    python validate_triton_impl.py <file_path> [--json]

退出码: 0 = 通过, 1 = 检测到退化
"""
import ast
import argparse
import json
import logging
import os
import sys


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

# 确保同目录下的 _log_utils 可被导入（脚本可能从其他工作目录调用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _log_utils import setup_logger as _setup_logger_shared  # noqa: E402

logger = logging.getLogger("triton_op_verifier.validate_triton_impl")


def _setup_logger() -> None:
    """配置 logger：复用 _log_utils.setup_logger。"""
    _setup_logger_shared(logger)


# ---------------------------------------------------------------------------
# 白名单：forward() 中允许的 torch 调用和 tensor 方法
# ---------------------------------------------------------------------------

ALLOWED_TORCH_FUNCS = {
    # buffer 分配
    "empty", "empty_like", "empty_strided",
    "zeros", "zeros_like",
    "ones", "ones_like",
    "full", "full_like",
    # tensor 创建（有时需要用于标量常量 / 索引）
    "tensor", "arange", "linspace",
    # 类型 / 设备
    "as_tensor",
}

ALLOWED_TENSOR_METHODS = {
    # 形状 / 元信息
    "size", "shape", "stride", "numel", "dtype", "device", "dim",
    "is_contiguous", "data_ptr", "element_size", "storage_offset",
    # 布局操作（不执行计算）
    "contiguous", "to", "view", "view_as", "reshape",
    "permute", "transpose", "expand", "expand_as",
    "flatten", "unflatten", "unsqueeze", "squeeze",
    "narrow", "clone", "detach", "t",
    "type", "float", "half", "bfloat16", "int", "long", "bool", "double",
    "cpu", "npu", "cuda",
    "item", "tolist",
    # 原地标记
    "requires_grad_", "zero_",
    # 切片相关（一般通过 __getitem__ 而非方法，但以防万一）
    "index_select",
}

ALLOWED_TRITON_ATTRS = {
    "cdiv", "next_power_of_2",
}

FORBIDDEN_TENSOR_METHODS = {
    # 计算操作
    "sum", "mean", "max", "min", "softmax", "log_softmax",
    "matmul", "mm", "bmm", "addmm", "add", "sub", "mul", "div",
    "relu", "sigmoid", "tanh", "gelu", "silu", "elu", "leaky_relu",
    "exp", "log", "log2", "log10", "sqrt", "pow", "abs",
    "norm", "layer_norm", "batch_norm", "group_norm",
    "conv1d", "conv2d", "conv3d", "conv_transpose2d", "linear",
    "dropout", "softplus", "hardtanh", "hardswish",
}

FUNCTIONAL_QUALIFIERS = {
    "F", "functional", "torch.nn.functional", "nn.functional",
}

# forward() 中禁止的 Python 控制流和结构
FORBIDDEN_PYTHON_STMTS = {
    "for": "Python for 循环",
    "while": "Python while 循环",
}


# ---------------------------------------------------------------------------
# AST 辅助函数
# ---------------------------------------------------------------------------

def _decorator_is_triton_jit(decorator):
    """判断装饰器节点是否为 triton.jit 或 @jit（从 triton 导入）。"""
    # @triton.jit
    if isinstance(decorator, ast.Attribute):
        if (isinstance(decorator.value, ast.Name)
                and decorator.value.id == "triton"
                and decorator.attr == "jit"):
            return True
    # @jit（直接导入）
    if isinstance(decorator, ast.Name) and decorator.id == "jit":
        return True
    # @triton.jit 作为 Call（如 @triton.jit 带参数，虽然少见）
    if isinstance(decorator, ast.Call):
        return _decorator_is_triton_jit(decorator.func)
    return False


def _decorator_is_triton_autotune(decorator):
    """判断装饰器是否为 triton.autotune。"""
    if isinstance(decorator, ast.Attribute):
        if (isinstance(decorator.value, ast.Name)
                and decorator.value.id == "triton"
                and decorator.attr == "autotune"):
            return True
    if isinstance(decorator, ast.Call):
        return _decorator_is_triton_autotune(decorator.func)
    return False


def _has_triton_decorator(func_node):
    """检查函数是否有 @triton.jit（可能与 @triton.autotune 组合）。"""
    for dec in func_node.decorator_list:
        if _decorator_is_triton_jit(dec):
            return True
    return False


def _resolve_call_name(node):
    """尝试从 ast.Call 节点提取被调用函数的名称字符串。

    返回 (qualifier, attr) 或 (None, name) 或 None。
    例如：torch.empty -> ('torch', 'empty')
          my_func    -> (None, 'my_func')
          self.conv  -> ('self', 'conv')
          kernel[g]  -> 返回 None（kernel launch 通过 Subscript）
    """
    func = node.func if isinstance(node, ast.Call) else node
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return (func.value.id, func.attr)
        # 处理 torch.nn.functional.relu 形式
        if isinstance(func.value, ast.Attribute):
            inner = func.value
            if isinstance(inner.value, ast.Name):
                return (f"{inner.value.id}.{inner.attr}", func.attr)
    if isinstance(func, ast.Name):
        return (None, func.id)
    return None


def _get_subscript_value_name(node):
    """从 kernel[grid](...) 的 Subscript 节点提取 kernel 名称。"""
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name):
            return node.value.id
        if isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name):
                return f"{node.value.value.id}.{node.value.attr}"
    return None


# ---------------------------------------------------------------------------
# 核心检查
# ---------------------------------------------------------------------------

def _kernel_uses_tl_api(func_node) -> bool:
    """判断 kernel 函数体内是否出现 tl.* 属性访问。"""
    for child in ast.walk(func_node):
        if isinstance(child, ast.Attribute):
            if isinstance(child.value, ast.Name) and child.value.id == "tl":
                return True
    return False


def find_triton_kernels(tree):
    """查找所有 @triton.jit 装饰的函数名，及其是否使用了 tl.* API。"""
    kernels = {}  # name -> {"has_tl_usage": bool, "line": int}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _has_triton_decorator(node):
            kernels[node.name] = {
                "has_tl_usage": _kernel_uses_tl_api(node),
                "line": node.lineno,
            }
    return kernels


def _find_forward_in_class(class_node):
    """从 ModelNew 类节点中找到 forward 方法。"""
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
            return item
    return None


def find_model_new_forward(tree):
    """找到 ModelNew 类的 forward 方法节点。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            forward = _find_forward_in_class(node)
            if forward is not None:
                return forward
    return None


def _call_invokes_kernel(call_node, kernel_names) -> bool:
    """判断单个 Call 节点是否启动了 triton kernel（直接或通过 Subscript）。"""
    if isinstance(call_node.func, ast.Subscript):
        return _get_subscript_value_name(call_node.func) in kernel_names
    resolved = _resolve_call_name(call_node)
    return bool(resolved and resolved[0] is None and resolved[1] in kernel_names)


def _func_calls_kernel(func_node, kernel_names) -> bool:
    """判断函数体内是否存在 kernel 启动调用。"""
    for child in ast.walk(func_node):
        if isinstance(child, ast.Call) and _call_invokes_kernel(child, kernel_names):
            return True
    return False


def find_wrapper_functions(tree, kernel_names):
    """找到模块级别或类级别的辅助函数，这些函数内部调用了 triton kernel。

    返回函数名集合。
    """
    wrappers = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name not in kernel_names:
            if _func_calls_kernel(node, kernel_names):
                wrappers.add(node.name)
    return wrappers


def _called_from_call_node(call_node, kernel_names, wrapper_names):
    """从单个 Call 节点中提取被调用的 kernel/wrapper 名称，找不到返回 None。"""
    if isinstance(call_node.func, ast.Subscript):
        name = _get_subscript_value_name(call_node.func)
        if name in kernel_names:
            return name
    resolved = _resolve_call_name(call_node)
    if not resolved:
        return None
    qual, attr = resolved
    if qual is None and attr in kernel_names:
        return attr
    if qual is None and attr in wrapper_names:
        return attr
    if qual == "self" and attr in wrapper_names:
        return attr
    return None


def check_kernel_calls_in_forward(forward_node, kernel_names, wrapper_names):
    """检查 forward 中是否调用了 triton kernel（直接或通过 wrapper）。

    返回被调用的 kernel/wrapper 名称集合。
    """
    called = set()
    if forward_node is None:
        return called
    for node in ast.walk(forward_node):
        if isinstance(node, ast.Call):
            name = _called_from_call_node(node, kernel_names, wrapper_names)
            if name is not None:
                called.add(name)
    return called


def _count_kernel_launches_in_forward(forward_node):
    """统计 forward() 中 kernel 启动调用（kernel[grid](...)）的次数。"""
    count = 0
    if forward_node is None:
        return count
    for node in ast.walk(forward_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            count += 1
    return count


# --- check_forbidden_torch_ops 拆分出的辅助规则 ---

def _violation_for_loop(node):
    """循环禁用规则：返回违规字典或 None。"""
    if isinstance(node, ast.For):
        return {
            "line": node.lineno,
            "call": "for 循环",
            "reason": "forward() 中禁止 Python for 循环，核心计算必须在单个 Triton kernel 内完成",
        }
    if isinstance(node, ast.While):
        return {
            "line": node.lineno,
            "call": "while 循环",
            "reason": (
                "forward() 中禁止 Python while 循环，"
                "核心计算必须在单个 Triton kernel 内完成"
            ),
        }
    return None


def _violation_matmul_op(node):
    """检测 @ 运算符。"""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        return {
            "line": node.lineno,
            "call": "@",
            "reason": "矩阵乘法 @ 运算符必须在 Triton kernel 中实现",
        }
    return None


def _violation_list_append(node):
    """检测 list.append 形式调用。"""
    if not isinstance(node, ast.Call):
        return None
    resolved = _resolve_call_name(node)
    if not resolved:
        return None
    qual, attr = resolved
    if attr == "append" and qual is not None:
        return {
            "line": node.lineno,
            "call": f"{qual}.append(...)",
            "reason": (
                "forward() 中禁止 list.append，"
                "动态状态维护必须在 Triton kernel 内完成"
            ),
        }
    return None


def _violation_for_torch_qual(node, qual, attr):
    """处理 qual == 'torch' 的调用。"""
    if attr in ALLOWED_TORCH_FUNCS:
        return None
    return {
        "line": node.lineno,
        "call": f"torch.{attr}",
        "reason": f"torch.{attr} 是计算操作，必须在 Triton kernel 中实现",
    }


def _violation_for_functional_qual(node, qual, attr):
    """处理 F./functional. 形式调用。"""
    return {
        "line": node.lineno,
        "call": f"{qual}.{attr}",
        "reason": f"{qual}.{attr} 是 PyTorch 计算操作，必须在 Triton kernel 中实现",
    }


def _violation_for_tensor_method(node, qual, attr):
    """处理被禁止的 tensor 方法调用。"""
    if attr not in FORBIDDEN_TENSOR_METHODS:
        return None
    # 排除已知安全的 qual（torch/F/triton 已在上面处理）
    skip_quals = {"torch", "F", "triton"} | FUNCTIONAL_QUALIFIERS
    if qual in skip_quals:
        return None
    return {
        "line": node.lineno,
        "call": f"{qual}.{attr}()" if qual else f"{attr}()",
        "reason": f"{attr} 是计算操作，必须在 Triton kernel 中实现",
    }


def _violation_for_self_call(node, qual, attr):
    """处理 self.xxx(...) 形式调用。"""
    if qual != "self" or attr == "forward":
        return None
    return {
        "line": node.lineno,
        "call": f"self.{attr}(...)",
        "reason": (
            f"self.{attr}() 疑似 nn.Module 前向调用，"
            "核心计算必须在 Triton kernel 中实现"
        ),
    }


def _violation_for_call(node):
    """对 Call 节点应用所有调用相关规则，返回首个命中或 None。"""
    v = _violation_list_append(node)
    if v is not None:
        return v

    # --- kernel launch: kernel[grid](...) —— 允许 ---
    if isinstance(node.func, ast.Subscript):
        return None

    resolved = _resolve_call_name(node)
    if resolved is None:
        return None

    qual, attr = resolved
    if qual == "torch":
        return _violation_for_torch_qual(node, qual, attr)
    if qual in FUNCTIONAL_QUALIFIERS:
        return _violation_for_functional_qual(node, qual, attr)
    # --- triton.cdiv 等 —— 允许 ---
    if qual == "triton" and attr in ALLOWED_TRITON_ATTRS:
        return None
    v = _violation_for_tensor_method(node, qual, attr)
    if v is not None:
        return v
    return _violation_for_self_call(node, qual, attr)


def _violation_for_node(node):
    """对任意 AST 节点应用所有规则，返回首个命中或 None。"""
    v = _violation_for_loop(node)
    if v is not None:
        return v
    v = _violation_matmul_op(node)
    if v is not None:
        return v
    if isinstance(node, ast.Call):
        return _violation_for_call(node)
    return None


def check_forbidden_torch_ops(forward_node):
    """检查 forward 中是否使用了禁止的 torch 计算操作或 Python 控制流。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]
    """
    violations = []
    if forward_node is None:
        return violations

    for node in ast.walk(forward_node):
        v = _violation_for_node(node)
        if v is not None:
            violations.append(v)

    # --- 规则 B: 如果 forward() 中 kernel 启动次数 > 1，视为 Type3 退化 ---
    kernel_launch_count = _count_kernel_launches_in_forward(forward_node)
    if kernel_launch_count > 1:
        violations.append({
            "line": forward_node.lineno,
            "call": f"kernel 启动 {kernel_launch_count} 次",
            "reason": (
                "forward() 中只能启动一次 Triton kernel，"
                "多次启动表明核心计算在 host 端循环中完成"
            ),
        })

    return violations


# ---------------------------------------------------------------------------
# 主验证逻辑
# ---------------------------------------------------------------------------

def _empty_result(filepath):
    return {
        "valid": False,
        "filepath": filepath,
        "checks": {
            "triton_kernel_exists": {"passed": False, "kernels": [], "error": None},
            "kernel_called_from_forward": {"passed": False, "called": [], "error": None},
            "no_forbidden_torch_ops": {"passed": False, "violations": [], "error": None},
        },
        "regression_type": None,
        "suggestion": "",
    }


def _check_kernel_exists(result, tree):
    """填充 triton_kernel_exists 检查；返回 (passed, kernel_names)。"""
    kernels = find_triton_kernels(tree)
    kernel_names = set(kernels.keys())
    result["checks"]["triton_kernel_exists"]["kernels"] = [
        {"name": k, "line": v["line"], "has_tl_usage": v["has_tl_usage"]}
        for k, v in kernels.items()
    ]

    if not kernel_names:
        result["checks"]["triton_kernel_exists"]["error"] = "未找到任何 @triton.jit 装饰的 kernel 函数"
        result["regression_type"] = 1
        result["suggestion"] = (
            "代码中没有 Triton kernel。必须创建至少一个 @triton.jit 装饰的函数，"
            "在其中使用 tl.load/tl.store 实现核心计算逻辑。"
        )
        return False, kernel_names

    kernels_without_tl = [k for k, v in kernels.items() if not v["has_tl_usage"]]
    if len(kernels_without_tl) == len(kernels):
        result["checks"]["triton_kernel_exists"]["error"] = (
            f"kernel 函数 {kernels_without_tl} 未使用任何 tl.* API，"
            "可能是空壳 kernel"
        )
        result["regression_type"] = 1
        result["suggestion"] = (
            "虽然存在 @triton.jit 装饰的函数，但没有使用 triton.language (tl) API。"
            "kernel 必须使用 tl.load/tl.store 等进行显式内存操作和计算。"
        )
        return False, kernel_names

    result["checks"]["triton_kernel_exists"]["passed"] = True
    return True, kernel_names


def _check_forward_calls_kernel(result, tree, kernel_names):
    """填充 kernel_called_from_forward 检查；返回 (passed, forward_node)。"""
    forward_node = find_model_new_forward(tree)
    if forward_node is None:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            "未找到 ModelNew.forward() 方法"
        )
        result["regression_type"] = 2
        result["suggestion"] = "代码缺少 ModelNew 类或 forward 方法。"
        return False, None

    wrapper_names = find_wrapper_functions(tree, kernel_names)
    called = check_kernel_calls_in_forward(forward_node, kernel_names, wrapper_names)
    result["checks"]["kernel_called_from_forward"]["called"] = list(called)

    if not called:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            f"@triton.jit kernel {list(kernel_names)} 已定义但 forward() 未调用任何 kernel"
        )
        result["regression_type"] = 2
        wrapper_hint = (
            f"也存在 wrapper 函数 {list(wrapper_names)} 但 forward 也未调用它们。"
            if wrapper_names else ""
        )
        result["suggestion"] = (
            f"已定义 kernel {list(kernel_names)} 但 ModelNew.forward() 中未调用。"
            "forward() 必须通过 kernel_name[grid](...) 形式启动 kernel。"
            f"{wrapper_hint}"
        )
        return False, forward_node

    result["checks"]["kernel_called_from_forward"]["passed"] = True
    return True, forward_node


def _check_no_forbidden_ops(result, forward_node):
    """填充 no_forbidden_torch_ops 检查；返回 passed。"""
    violations = check_forbidden_torch_ops(forward_node)
    result["checks"]["no_forbidden_torch_ops"]["violations"] = violations

    if not violations:
        result["checks"]["no_forbidden_torch_ops"]["passed"] = True
        return True

    result["checks"]["no_forbidden_torch_ops"]["error"] = (
        f"forward() 中发现 {len(violations)} 处禁止的 PyTorch 计算操作"
    )
    violation_details = "; ".join(
        f"第{v['line']}行 {v['call']}" for v in violations[:5]
    )
    result["regression_type"] = 3
    result["suggestion"] = (
        f"forward() 调用了 Triton kernel 但仍使用 PyTorch 进行部分计算: "
        f"{violation_details}。"
        "所有核心计算必须在 @triton.jit kernel 中完成，"
        "forward() 中只允许 buffer 分配（torch.empty 等）和形状操作（.view/.reshape 等）。"
    )
    return False


def validate(code, filepath="<unknown>"):
    """对生成代码执行完整的退化检查。

    返回结构化结果 dict。
    """
    result = _empty_result(filepath)

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["checks"]["triton_kernel_exists"]["error"] = f"SyntaxError: {e}"
        result["regression_type"] = 1
        result["suggestion"] = "代码存在语法错误，无法解析。"
        return result

    ok, kernel_names = _check_kernel_exists(result, tree)
    if not ok:
        return result

    ok, forward_node = _check_forward_calls_kernel(result, tree, kernel_names)
    if not ok:
        return result

    if not _check_no_forbidden_ops(result, forward_node):
        return result

    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_code(path, want_json):
    """读取代码文件；失败时返回 None，由调用方决定退出。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        if want_json:
            logger.info("%s", json.dumps({"valid": False, "error": f"文件不存在: {path}"}))
        else:
            logger.error("[ERROR] 文件不存在: %s", path)
        return None


def _emit_pass(result):
    kernels = result["checks"]["triton_kernel_exists"]["kernels"]
    called = result["checks"]["kernel_called_from_forward"]["called"]
    logger.info("[PASS] Triton 实现验证通过")
    logger.info(
        "  - 发现 %d 个 @triton.jit kernel: %s",
        len(kernels),
        ", ".join(k["name"] for k in kernels),
    )
    logger.info("  - forward() 调用: %s", ", ".join(called))
    logger.info("  - forward() 中无禁止的 PyTorch 计算操作")


def _emit_fail(result):
    rtype = result["regression_type"]
    type_desc = {
        1: "完全无 Triton kernel（纯 PyTorch）",
        2: "有 Triton kernel 但 forward() 未调用",
        3: "部分计算使用 PyTorch（需全部移入 Triton kernel）",
    }
    logger.info("[FAIL] 检测到 PyTorch 退化 — Type %s: %s", rtype, type_desc.get(rtype, "未知"))

    for check_name, check_result in result["checks"].items():
        status = "PASS" if check_result["passed"] else "FAIL"
        logger.info("  [%s] %s", status, check_name)
        if check_result["error"]:
            logger.info("         %s", check_result["error"])

    if result["checks"]["no_forbidden_torch_ops"]["violations"]:
        logger.info("  违规详情:")
        for v in result["checks"]["no_forbidden_torch_ops"]["violations"]:
            logger.info("    第 %s 行: %s — %s", v["line"], v["call"], v["reason"])

    logger.info("\n  修复建议: %s", result["suggestion"])


def main():
    _setup_logger()
    parser = argparse.ArgumentParser(
        description="检查生成代码是否退化为 PyTorch 原生实现（AST 静态分析）"
    )
    parser.add_argument("file", help="要检查的 Python 文件路径")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    code = _load_code(args.file, args.json)
    if code is None:
        sys.exit(1)
    result = validate(code, filepath=args.file)

    if args.json:
        logger.info("%s", json.dumps(result, ensure_ascii=False, indent=2))
    elif result["valid"]:
        _emit_pass(result)
    else:
        _emit_fail(result)

    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
