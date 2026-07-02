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
"""TileLang 实现退化检测脚本 — 通过 AST 静态分析检查生成代码是否退化为 PyTorch 原生实现。

检测四种退化类型：
  Type 1: 无 TileLang kernel 导入（纯 PyTorch）
  Type 2: 有 kernel 导入但 forward() 未调用
  Type 3: forward() 调用了 kernel 但仍有部分计算使用 torch 接口
  Type 4: forward() 中存在逐元素 Python for 循环（标量写法退化）

TileLang 正确模式：
  1. 从 design.tile_level.xxx 导入 kernel builder 函数
  2. forward() 中调用 builder 获取 kernel 对象: kernel = builder(M, N, ...)
  3. 调用 kernel 对象执行计算: result = kernel(x, y)

用法:
    python validate_tilelang_impl.py <file_path> [--json]

退出码: 0 = 通过, 1 = 检测到退化
"""
import ast
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 从 validate_ascendc_impl 导入共享常量、AST 工具和检查函数
_ASCENDC_SCRIPTS = (
    Path(__file__).resolve().parents[1] / "tilelang2ascend-translator" / "scripts"
)
if str(_ASCENDC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_ASCENDC_SCRIPTS))
from validate_ascendc_impl import (
    ALLOWED_TORCH_FUNCS_TYPE,
    ALLOWED_TENSOR_METHODS_TYPE,
    ALLOWED_BUILTIN_FUNCS,
    FORBIDDEN_TENSOR_METHODS,
    _resolve_call_name,
    find_model_forward,
    check_for_loops_over_tensors,
    _format_violation_details,
    _format_loop_details,
    _run_cli,
    _make_result,
    _print_validation_result,
    PrintConfig,
)

KERNEL_TYPE = "TileLang kernel"


# ---------------------------------------------------------------------------
# TileLang 专用: kernel 导入和调用检测
# ---------------------------------------------------------------------------

def _is_tilelang_design_module(module_path):
    """检查模块路径是否匹配 TileLang 设计模块的模式。

    匹配模式：
    - design.tile_level.xxx
    - design.tile_level.xxx.yyy
    """
    if not module_path:
        return False
    parts = module_path.split(".")
    if len(parts) >= 3 and parts[0] == "design" and parts[1] == "tile_level":
        return True
    return False


def find_tilelang_kernel_imports(tree):
    """查找所有从 design.tile_level.* 导入的 TileLang kernel builder 函数。

    检测模式：
    1. from design.tile_level.xxx import yyy [as zzz]
    2. from design.tile_level.xxx import (a, b, c)

    返回 dict: {used_name: {"actual_name": str, "module": str,
                              "alias": str|None, "line": int}}
    """
    kernels = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not _is_tilelang_design_module(node.module):
            continue
        for alias in node.names:
            used_name = alias.asname if alias.asname else alias.name
            kernels[used_name] = {
                "actual_name": alias.name,
                "module": node.module,
                "alias": alias.asname,
                "line": node.lineno,
            }

    return kernels


def _method_calls_builder(method_node, kernel_builder_names):
    """Return True if the method body contains a call to any kernel builder."""
    for child in ast.walk(method_node):
        if isinstance(child, ast.Call):
            resolved = _resolve_call_name(child)
            if resolved and resolved[0] is None and resolved[1] in kernel_builder_names:
                return True
    return False


def find_build_kernel_methods(class_node, kernel_builder_names):
    """查找类中通过 kernel builder 构建 kernel 的辅助方法。

    典型模式：
      def _build_kernel(self, x):
          return tl_matmul_leakyrelu(m, n, k)

    返回方法名集合（这些方法返回可调用的 kernel 对象）。
    """
    builder_methods = set()
    if class_node is None:
        return builder_methods

    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name != "forward":
            if _method_calls_builder(item, kernel_builder_names):
                builder_methods.add(item.name)
    return builder_methods


def _function_calls_builder(func_node, kernel_builder_names):
    """Return True if the function contains a call to any kernel builder."""
    for child in ast.walk(func_node):
        if isinstance(child, ast.Call):
            resolved = _resolve_call_name(child)
            if resolved and resolved[0] is None and resolved[1] in kernel_builder_names:
                return True
    return False


def find_module_wrapper_functions(tree, kernel_builder_names):
    """查找模块级别的辅助函数，它们调用 kernel builder 或 kernel 对象。

    返回函数名集合。
    """
    wrappers = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and _function_calls_builder(node, kernel_builder_names):
            wrappers.add(node.name)
    return wrappers


def _scan_kernel_builder_assigns(tree, kernel_builder_names, builder_method_names):
    """Scan for kernel builder assignment patterns and return (called_list, var_names)."""
    called = []
    var_names = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and isinstance(node.value, ast.Call)):
            continue
        resolved = _resolve_call_name(node.value)
        if not resolved:
            continue
        qual, attr = resolved
        if qual is None and attr in kernel_builder_names:
            var_names.add(target.id)
            called.append({"call": attr, "line": node.lineno, "pattern": "builder_assign"})
        elif qual == "self" and attr in builder_method_names:
            var_names.add(target.id)
            called.append({"call": f"self.{attr}", "line": node.lineno, "pattern": "builder_method"})
    return called, var_names


def _scan_kernel_invocations(tree, kernel_var_names, wrapper_func_names, kernel_builder_names):
    """Scan for kernel invocation patterns and return called_list."""
    called = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call_name(node)
        if resolved:
            qual, attr = resolved
            if qual is None and attr in kernel_var_names:
                called.append({"call": f"{attr}(...)", "line": node.lineno, "pattern": "kernel_invoke"})
            elif qual is None and attr in wrapper_func_names:
                called.append({"call": attr, "line": node.lineno, "pattern": "wrapper_call"})
            elif qual == "self" and attr in wrapper_func_names:
                called.append({"call": f"self.{attr}", "line": node.lineno, "pattern": "wrapper_call"})
        if isinstance(node.func, ast.Call):
            inner = _resolve_call_name(node.func)
            if inner and inner[0] is None and inner[1] in kernel_builder_names:
                called.append({
                    "call": f"{inner[1]}(...)(...)",
                    "line": node.lineno,
                    "pattern": "inline_build_invoke",
                })
    return called


def check_kernel_calls_in_forward(forward_node, kernel_builder_names,
                                  builder_method_names, wrapper_func_names):
    """检查 forward 中是否调用了 TileLang kernel。"""
    called = []
    if forward_node is None:
        return called

    assigns, kernel_var_names = _scan_kernel_builder_assigns(
        forward_node, kernel_builder_names, builder_method_names)
    called.extend(assigns)

    called.extend(_scan_kernel_invocations(
        forward_node, kernel_var_names, wrapper_func_names, kernel_builder_names))

    return called


def _has_kernel_invocation(called_list):
    """检查 called 列表中是否有实际的 kernel 调用（不仅仅是 builder 赋值）。"""
    invoke_patterns = {"kernel_invoke", "wrapper_call", "inline_build_invoke"}
    return any(c["pattern"] in invoke_patterns for c in called_list)


# ---------------------------------------------------------------------------
# TileLang 专用: check_forbidden_torch_ops（内置 kernel builder 白名单）
# ---------------------------------------------------------------------------

def _check_call_for_violation(qual, attr, node, kernel_builder_names, builder_method_names):
    """Check a single resolved call and return a violation dict or None."""
    # --- TileLang kernel builder 调用 —— 允许 ---
    if qual is None and attr in kernel_builder_names:
        return None
    if qual == "self" and attr in builder_method_names:
        return None

    # --- torch.xxx(...) ---
    if qual == "torch":
        if attr not in ALLOWED_TORCH_FUNCS_TYPE:
            return {"line": node.lineno, "call": f"torch.{attr}",
                    "reason": f"torch.{attr} 是计算操作，必须在 {KERNEL_TYPE} 中实现"}
        return None

    # --- F.xxx(...) / functional.xxx(...) ---
    if qual in ("F", "functional", "torch.nn.functional", "nn.functional"):
        return {"line": node.lineno, "call": f"{qual}.{attr}",
                "reason": f"{qual}.{attr} 是 PyTorch 计算操作，必须在 {KERNEL_TYPE} 中实现"}

    # --- Python 内建函数 —— 允许 ---
    if qual is None and attr in ALLOWED_BUILTIN_FUNCS:
        return None

    # --- tensor 方法计算操作 ---
    if attr in FORBIDDEN_TENSOR_METHODS:
        if qual not in ("torch", "F", "functional", "torch.nn.functional", "nn.functional"):
            return {"line": node.lineno, "call": f"{qual}.{attr}()" if qual else f"{attr}()",
                    "reason": f"{attr} 是计算操作，必须在 {KERNEL_TYPE} 中实现"}
        return None

    # --- self.layer_name(x) —— 禁止 nn.Module 调用（但允许 builder 方法）---
    if qual == "self" and attr not in ("forward",) and attr not in builder_method_names:
        return {"line": node.lineno, "call": f"self.{attr}(...)",
                "reason": f"self.{attr}() 疑似 nn.Module 前向调用，核心计算必须在 {KERNEL_TYPE} 中实现"}

    return None


def check_forbidden_torch_ops(forward_node, kernel_builder_names,
                              builder_method_names):
    """检查 forward 中是否使用了禁止的 torch 计算操作。"""
    violations = []
    if forward_node is None:
        return violations

    for node in ast.walk(forward_node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            violations.append({
                "line": node.lineno, "call": "@",
                "reason": f"矩阵乘法 @ 运算符必须在 {KERNEL_TYPE} 中实现",
            })
            continue
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call_name(node)
        if resolved is None:
            continue
        violation = _check_call_for_violation(
            resolved[0], resolved[1], node,
            kernel_builder_names, builder_method_names)
        if violation:
            violations.append(violation)

    return violations


# ---------------------------------------------------------------------------
# 主验证逻辑
# ---------------------------------------------------------------------------

def _check_kernel_imported(result, tree):
    """Check 1: TileLang kernel imports exist. Returns kernel_builder_names or None."""
    kernels = find_tilelang_kernel_imports(tree)
    kernel_builder_names = set(kernels.keys())

    result["checks"]["tilelang_kernel_imported"]["kernels"] = [
        {"used_name": k, "actual_name": v["actual_name"],
         "module": v["module"], "line": v["line"]}
        for k, v in kernels.items()
    ]

    if not kernel_builder_names:
        result["checks"]["tilelang_kernel_imported"]["error"] = (
            "未找到任何从 design.tile_level.* 导入的 TileLang kernel builder"
        )
        result["regression_type"] = 1
        result["suggestion"] = (
            "代码中没有从 design.tile_level.* 导入 TileLang kernel builder。"
            "model_new_tilelang.py 必须从 design/tile_level/ 导入 kernel builder 函数"
            "（如 from design.tile_level.xxx import xxx），"
            "并在 forward() 中调用 builder 构建 kernel 再执行计算。"
        )
        return None

    result["checks"]["tilelang_kernel_imported"]["passed"] = True
    return kernel_builder_names


def _check_kernel_called(result, tree, kernel_builder_names):
    """Check 2: forward() calls kernel. Returns (forward_node, builder_method_names) or (None, None)."""
    forward_node, class_name, class_node = find_model_forward(tree)
    if forward_node is None:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            "未找到 ModelNew.forward() 或 Model.forward() 方法"
        )
        result["regression_type"] = 2
        result["suggestion"] = "代码缺少 ModelNew（或 Model）类或 forward 方法。"
        return None, None

    builder_method_names = find_build_kernel_methods(class_node, kernel_builder_names)
    wrapper_func_names = find_module_wrapper_functions(tree, kernel_builder_names)

    called = check_kernel_calls_in_forward(
        forward_node, kernel_builder_names,
        builder_method_names, wrapper_func_names,
    )
    result["checks"]["kernel_called_from_forward"]["called"] = [
        {"call": c["call"], "line": c["line"], "pattern": c["pattern"]}
        for c in called
    ]

    has_invocation = _has_kernel_invocation(called)

    if not called:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            f"已导入 kernel builder {list(kernel_builder_names)} 但 "
            f"{class_name}.forward() 未调用任何 kernel"
        )
        result["regression_type"] = 2
        result["suggestion"] = (
            f"已导入 TileLang kernel builder {list(kernel_builder_names)} 但 "
            f"{class_name}.forward() 中未使用。"
            "forward() 必须通过 kernel = builder(M, N, ...); kernel(x, y) 模式调用。"
        )
        return None, None

    if not has_invocation:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            "forward() 中调用了 kernel builder 但未执行返回的 kernel 对象"
        )
        result["regression_type"] = 2
        result["suggestion"] = (
            "forward() 构建了 kernel 对象但没有执行它。"
            "请确保在 kernel = builder(M, N, ...) 之后调用 kernel(x, y, ...)。"
        )
        return None, None

    result["checks"]["kernel_called_from_forward"]["passed"] = True
    return forward_node, builder_method_names


def _check_forbidden_ops(result, forward_node, kernel_builder_names, builder_method_names):
    """Check 3: no forbidden torch ops. Returns True to continue."""
    violations = check_forbidden_torch_ops(
        forward_node, kernel_builder_names, builder_method_names,
    )
    result["checks"]["no_forbidden_torch_ops"]["violations"] = violations

    if violations:
        violation_details = _format_violation_details(violations)
        result["checks"]["no_forbidden_torch_ops"]["error"] = (
            f"forward() 中发现 {len(violations)} 处禁止的 PyTorch 计算操作"
        )
        result["regression_type"] = 3
        result["suggestion"] = (
            f"forward() 调用了 TileLang kernel 但仍使用 PyTorch 进行部分计算: "
            f"{violation_details}。"
            "所有核心计算必须在 TileLang kernel 中完成，"
            "forward() 中只允许 buffer 分配（torch.empty 等）和形状操作（.view/.reshape 等）。"
        )
        return False

    result["checks"]["no_forbidden_torch_ops"]["passed"] = True
    return True


def _check_scalar_loops(result, forward_node):
    """Check 4: no scalar for loops. Returns True to continue."""
    loop_violations = check_for_loops_over_tensors(forward_node, KERNEL_TYPE)
    result["checks"]["no_scalar_for_loops"]["violations"] = loop_violations

    if loop_violations:
        loop_details = _format_loop_details(loop_violations)
        result["checks"]["no_scalar_for_loops"]["error"] = (
            f"forward() 中发现 {len(loop_violations)} 处逐元素 Python for 循环"
        )
        result["regression_type"] = 4
        result["suggestion"] = (
            f"forward() 中存在逐元素 Python for 循环: {loop_details}。"
            "不能用标量逐元素写法，必须使用 TileLang kernel 的向量化 / 块级操作。"
        )
        return False

    result["checks"]["no_scalar_for_loops"]["passed"] = True
    return True


def validate(code, filepath="<unknown>"):
    """对生成代码执行完整的退化检查。

    返回结构化结果 dict。
    """
    result = _make_result(filepath, first_check_name="tilelang_kernel_imported",
                          first_check_field="kernels")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["checks"]["tilelang_kernel_imported"]["error"] = f"SyntaxError: {e}"
        result["regression_type"] = 1
        result["suggestion"] = "代码存在语法错误，无法解析。"
        return result

    kernel_builder_names = _check_kernel_imported(result, tree)
    if kernel_builder_names is None:
        return result

    forward_node, builder_method_names = _check_kernel_called(result, tree, kernel_builder_names)
    if forward_node is None:
        return result

    if not _check_forbidden_ops(result, forward_node, kernel_builder_names, builder_method_names):
        return result

    if not _check_scalar_loops(result, forward_node):
        return result

    result["valid"] = True
    return result


def _print_result(result):
    cfg = PrintConfig(
        first_check_name="tilelang_kernel_imported",
        first_check_field="kernels",
        kernel_label="kernel builder",
        pass_header="[PASS] TileLang 实现验证通过",
        type_descs={
            1: "无 TileLang kernel 导入（纯 PyTorch）",
            2: "有 kernel 导入但 forward() 未调用",
            3: "部分计算仍使用 PyTorch（需全部移入 TileLang kernel）",
            4: "存在逐元素 Python for 循环（需使用向量化操作）",
        },
    )
    _print_validation_result(result, cfg)


def main():
    sys.exit(_run_cli(
        description="检查 TileLang 生成代码是否退化为 PyTorch 原生实现（AST 静态分析）",
        validate_fn=validate,
        print_result_fn=_print_result,
    ))


if __name__ == "__main__":
    main()
