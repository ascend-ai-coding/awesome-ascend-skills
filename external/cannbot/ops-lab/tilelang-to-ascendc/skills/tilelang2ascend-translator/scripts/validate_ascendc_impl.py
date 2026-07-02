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
"""AscendC 实现退化检测脚本 — 通过 AST 静态分析检查生成代码是否退化为 PyTorch 原生实现。

检测四种退化类型：
  Type 1: 无 AscendC kernel 扩展导入（纯 PyTorch）
  Type 2: 有扩展导入但 forward() 未调用 kernel 函数
  Type 3: forward() 调用了 kernel 但仍有部分计算使用 torch 接口
  Type 4: forward() 中存在逐元素 Python for 循环（标量写法退化）

用法:
    python validate_ascendc_impl.py <file_path> [--json]

退出码: 0 = 通过, 1 = 检测到退化
"""
import ast
import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# 从 validate_triton_impl 导入共享的 tensor 方法白名单（复用，消除重复代码）
_TRITON_SCRIPTS = (
    Path(__file__).resolve().parents[5] / "ops" / "triton-op-verifier" / "scripts"
)
if str(_TRITON_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_TRITON_SCRIPTS))
from validate_triton_impl import ALLOWED_TENSOR_METHODS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_stdout_logger = logging.getLogger(__name__ + "._stdout")
_stdout_logger.propagate = False
_stdout_logger.setLevel(logging.INFO)
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(logging.Formatter("%(message)s"))
_stdout_logger.addHandler(_stdout_handler)


# ---------------------------------------------------------------------------
# 白名单：forward() 中允许的 torch 调用和 tensor 方法
# (同时供 validate_tilelang_impl.py 导入复用)
# ---------------------------------------------------------------------------

ALLOWED_TORCH_FUNCS_TYPE = {
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

ALLOWED_TENSOR_METHODS_TYPE = ALLOWED_TENSOR_METHODS | {"is_npu", "is_cuda"}

ALLOWED_BUILTIN_FUNCS = {
    # Python 内建函数（非 tensor 方法）
    "min", "max", "abs", "len", "range", "int", "float", "bool",
    "list", "tuple", "str", "type", "isinstance", "print",
    "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "hasattr", "getattr", "setattr",
}

FORBIDDEN_TENSOR_METHODS = {
    # 归约操作
    "sum", "mean", "max", "min", "prod", "cumsum", "cumprod",
    "argmax", "argmin", "var", "std",
    # 矩阵 / 线性代数
    "matmul", "mm", "bmm", "addmm",
    # 逐元素算术
    "add", "sub", "mul", "div", "fmod", "remainder",
    "add_", "sub_", "mul_", "div_",
    # 激活函数
    "relu", "sigmoid", "tanh", "gelu", "silu", "elu", "leaky_relu",
    "relu_", "sigmoid_", "tanh_",
    # 数学函数
    "exp", "log", "log2", "log10", "sqrt", "pow", "abs",
    "sin", "cos", "clamp", "clamp_", "ceil", "floor", "round",
    "reciprocal", "neg", "sign",
    # softmax
    "softmax", "log_softmax",
    # 范数 / 归一化
    "norm", "layer_norm", "batch_norm", "group_norm",
    # 卷积 / 线性
    "conv1d", "conv2d", "conv3d", "conv_transpose2d", "linear",
    # 其他
    "dropout", "softplus", "hardtanh", "hardswish",
    # 比较（用于计算，非条件判断时）
    "eq", "ne", "lt", "gt", "le", "ge", "where",
}

# 已知的占位符导入名称（表示扩展模块未正确配置）
PLACEHOLDER_IMPORT_NAMES = {
    "TORCH_EXTENSION_NAME",
}

# AscendC 扩展模块的命名模式
ASCENDC_EXT_PATTERNS = [
    re.compile(r"_\w+_ext$"),          # _xxx_ext
    re.compile(r"\w+_ext$"),            # xxx_ext
    re.compile(r"\w+_ascendc\w*$"),     # xxx_ascendc, xxx_ascendc_ext
    re.compile(r"_ext$"),               # _ext
]

# torch.ops.load_library 调用模式（新的 whl/torch.ops 注册路径）
TORCH_OPS_LOAD_LIBRARY_PATTERNS = [
    re.compile(r"torch\.ops\.load_library"),
    re.compile(r"torch\.ops\.npu\."),
]


# ---------------------------------------------------------------------------
# AST 辅助函数（同时供 validate_tilelang_impl.py 导入复用）
# ---------------------------------------------------------------------------

def _resolve_call_name(node):
    """尝试从 ast.Call 节点提取被调用函数的名称字符串。

    返回 (qualifier, attr) 或 (None, name) 或 None。
    例如：torch.empty -> ('torch', 'empty')
          torch.ops.npu.rms_norm -> ('torch.ops.npu', 'rms_norm')
          _ext.run_kernel -> ('_ext', 'run_kernel')
          my_func -> (None, 'my_func')
    """
    func = node.func if isinstance(node, ast.Call) else node
    if isinstance(func, ast.Attribute):
        parts = []
        current = func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            parts.reverse()
            qual = ".".join(parts[:-1])
            attr = parts[-1]
            return (qual, attr) if qual else (None, attr)
    if isinstance(func, ast.Name):
        return (None, func.id)
    return None


def _find_forward_in_class(class_node):
    """Return the forward method node from a class node, or None."""
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
            return item
    return None


def find_model_forward(tree):
    """找到 ModelNew 或 Model 类的 forward 方法节点及其所属类。

    优先查找 ModelNew，若不存在则查找 Model。
    返回 (forward_node, class_name, class_node)。
    """
    model_new_class = None
    model_class = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name == "ModelNew":
                model_new_class = node
            elif node.name == "Model":
                model_class = node

    if model_new_class:
        forward = _find_forward_in_class(model_new_class)
        return forward, "ModelNew", model_new_class
    if model_class:
        forward = _find_forward_in_class(model_class)
        return forward, "Model", model_class
    return None, None, None


# ---------------------------------------------------------------------------
# AscendC 专用: _is_ext_module_name
# ---------------------------------------------------------------------------

def _is_ext_module_name(name):
    """检查名称是否匹配 AscendC 扩展模块的命名模式。"""
    if name in PLACEHOLDER_IMPORT_NAMES:
        return False
    for pattern in ASCENDC_EXT_PATTERNS:
        if pattern.match(name):
            return True
    return False


# ---------------------------------------------------------------------------
# AscendC 专用: 扩展导入检测
# ---------------------------------------------------------------------------

def _detect_importlib_extensions(tree):
    """检测 importlib 动态加载的扩展模块。"""
    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Call):
            continue
        resolved = _resolve_call_name(node.value)
        if not resolved:
            continue
        qual, attr = resolved
        if attr == "module_from_spec":
            result[target.id] = {
                "name": target.id, "alias": None,
                "line": node.lineno, "is_placeholder": False,
                "import_style": "importlib",
            }
    return result


def _check_torch_ops_npu_call(call_node, result):
    """Check if an ast.Call node is a standalone torch.ops.npu.<op>(...) invocation."""
    resolved = _resolve_call_name(call_node)
    if resolved is None:
        return
    if resolved[0] is None:
        return
    if not resolved[0].startswith("torch.ops.npu."):
        return
    result["torch.ops.npu"] = {
        "name": "torch.ops.npu", "alias": None,
        "line": call_node.lineno, "is_placeholder": False,
        "import_style": "torch_ops_npu_call",
    }


def _check_load_library_call(expr_value, result):
    """Check if an expression value node is a torch.ops.load_library(...) call."""
    resolved = _resolve_call_name(expr_value)
    if resolved is None:
        return
    qual, attr = resolved
    if attr != "load_library":
        return
    if qual != "torch.ops" and qual is not None:
        return
    result["torch.ops.npu"] = {
        "name": "torch.ops.npu", "alias": None,
        "line": expr_value.lineno, "is_placeholder": False,
        "import_style": "torch_ops_load_library",
    }


def _detect_torch_ops_extensions(tree):
    """检测 torch.ops.load_library 动态注册和 torch.ops.npu.* 直接调用的扩展。"""
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            _check_load_library_call(node.value, result)
        elif isinstance(node, ast.Call):
            _check_torch_ops_npu_call(node, result)
    return result


def _process_import_alias(alias, node, extensions, import_style):
    """Process a single import alias node and update extensions if it matches an AscendC pattern."""
    actual_name = alias.name
    used_name = alias.asname if alias.asname else alias.name
    is_placeholder = actual_name in PLACEHOLDER_IMPORT_NAMES
    if is_placeholder or _is_ext_module_name(actual_name):
        extensions[used_name] = {
            "name": actual_name,
            "alias": alias.asname,
            "line": node.lineno,
            "is_placeholder": is_placeholder,
            "import_style": import_style,
        }


def find_ascendc_extension_imports(tree):
    """查找所有 AscendC 扩展模块的导入信息。

    检测模式：
    1. import _xxx_ext [as alias]
    2. import xxx_ext [as alias]
    3. from path import _xxx_ext [as alias]
    4. importlib 动态加载

    返回 dict: {alias_or_name: {"name": str, "alias": str|None,
                                  "line": int, "is_placeholder": bool,
                                  "import_style": str}}
    """
    extensions = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        import_style = "import" if isinstance(node, ast.Import) else "from_import"
        for alias in node.names:
            _process_import_alias(alias, node, extensions, import_style)

    # --- importlib 动态加载检测 ---
    extensions.update(_detect_importlib_extensions(tree))

    # --- torch.ops.load_library 动态加载检测 (新 whl 注册路径) ---
    extensions.update(_detect_torch_ops_extensions(tree))

    return extensions


def _node_calls_ext(node, ext_names):
    """Return True if the AST node (a function) contains a call to any ext_name."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            resolved = _resolve_call_name(child)
            if resolved and resolved[0] in ext_names:
                return True
    return False


def find_wrapper_functions(tree, ext_names):
    """找到模块级别的辅助函数，这些函数内部调用了扩展模块。

    返回函数名集合。
    """
    wrappers = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and _node_calls_ext(node, ext_names):
            wrappers.add(node.name)
    return wrappers


def check_kernel_calls_in_forward(forward_node, ext_names, wrapper_names):
    """检查 forward 中是否调用了 AscendC 扩展模块的函数。

    检测模式：
    1. ext_module.function_name(...)  — 直接调用扩展模块方法
    2. wrapper_func(...)              — 通过 wrapper 函数调用
    3. self.wrapper_name(...)         — 通过类方法调用

    返回被调用信息列表 [{"call": str, "line": int}, ...]
    """
    called = []
    if forward_node is None:
        return called
    for node in ast.walk(forward_node):
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call_name(node)
        if resolved is None:
            continue
        qual, attr = resolved

        if qual in ext_names:
            called.append({"call": f"{qual}.{attr}", "line": node.lineno})

        if qual is None and attr in wrapper_names:
            called.append({"call": attr, "line": node.lineno})

        if qual == "self" and attr in wrapper_names:
            called.append({"call": f"self.{attr}", "line": node.lineno})
    return called


# ---------------------------------------------------------------------------
# 共享检查函数
# ---------------------------------------------------------------------------

def _is_allowed_npu_qual(qual):
    """Return True if qual belongs to torch.ops.npu.* or torch.ops namespace."""
    return qual and (qual == "torch.ops.npu" or qual.startswith("torch.ops.npu.") or qual == "torch.ops")


def _check_call_violation(node, qual, attr):
    """Check a single resolved call for forbidden ops. Returns a violation dict or None."""
    if _is_allowed_npu_qual(qual):
        return None

    if qual == "torch":
        if attr not in ALLOWED_TORCH_FUNCS_TYPE:
            return {"line": node.lineno, "call": f"torch.{attr}",
                    "reason": f"torch.{attr} 是计算操作，必须在 AscendC kernel 中实现"}
        return None

    if qual in ("F", "functional", "torch.nn.functional", "nn.functional"):
        return {"line": node.lineno, "call": f"{qual}.{attr}",
                "reason": f"{qual}.{attr} 是 PyTorch 计算操作，必须在 AscendC kernel 中实现"}

    if qual is None and attr in ALLOWED_BUILTIN_FUNCS:
        return None

    if attr in FORBIDDEN_TENSOR_METHODS:
        if qual not in ("torch", "F", "functional", "torch.nn.functional", "nn.functional"):
            return {"line": node.lineno,
                    "call": f"{qual}.{attr}()" if qual else f"{attr}()",
                    "reason": f"{attr} 是计算操作，必须在 AscendC kernel 中实现"}
        return None

    if qual == "self" and attr not in ("forward",):
        return {"line": node.lineno, "call": f"self.{attr}(...)",
                "reason": f"self.{attr}() 疑似 nn.Module 前向调用，核心计算必须在 AscendC kernel 中实现"}

    return None


def check_forbidden_torch_ops(forward_node):
    """检查 forward 中是否使用了禁止的 torch 计算操作。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]
    """
    violations = []
    if forward_node is None:
        return violations

    for node in ast.walk(forward_node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            violations.append({
                "line": node.lineno, "call": "@",
                "reason": "矩阵乘法 @ 运算符必须在 AscendC kernel 中实现",
            })
            continue
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call_name(node)
        if resolved is None:
            continue
        violation = _check_call_violation(node, resolved[0], resolved[1])
        if violation:
            violations.append(violation)

    return violations


def check_for_loops_over_tensors(forward_node, kernel_type="AscendC kernel"):
    """检查 forward 中是否存在用于计算的逐元素 Python for 循环（标量写法退化信号）。

    典型退化模式：
      for n in range(N):
          for c in range(C):
              x_nc = tensor[n, c]
              result = x_nc * weight + bias  # 逐元素计算
              output[n, c] = result.sum()    # 计算归约

    以下不视为退化：
      - 数据准备循环（仅做简单赋值 / 索引映射，无计算操作）

    返回违规列表 [{"line": N, "loop_var": str, "reason": str}, ...]
    """
    violations = []
    if forward_node is None:
        return violations

    for node in ast.walk(forward_node):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.iter, ast.Call):
            continue
        resolved = _resolve_call_name(node.iter)
        if not resolved or resolved != (None, "range"):
            continue
        loop_var = node.target.id if isinstance(node.target, ast.Name) else ""

        # 循环体必须同时满足两个条件才判定为退化：
        # 1. 存在 tensor 索引操作
        # 2. 循环体内含计算操作（禁止的 tensor 方法、torch 计算、
        #    BinOp 算术或 @ 矩阵乘法）
        has_tensor_indexing = _loop_has_tensor_indexing(node, loop_var)
        has_computation = _loop_has_computation(node)

        if has_tensor_indexing and has_computation:
            violations.append({
                "line": node.lineno,
                "loop_var": loop_var,
                "reason": (
                    f"for {loop_var} in range(...) 循环中存在 tensor 索引 + 计算操作，"
                    f"这是逐元素标量写法，必须使用 {kernel_type} 的向量化操作替代"
                ),
            })

    return violations


def _loop_has_tensor_indexing(for_node, loop_var):
    """检查 for 循环体中是否存在使用循环变量的 tensor 索引。"""
    if not loop_var:
        return False
    for child in ast.walk(for_node):
        if not isinstance(child, ast.Subscript):
            continue
        for sub_node in ast.walk(child.slice):
            if isinstance(sub_node, ast.Name) and sub_node.id == loop_var:
                return True
    return False


def _call_is_computation(qual, attr):
    """Return True if a resolved (qual, attr) call is a forbidden computation."""
    if qual is None and attr in ALLOWED_BUILTIN_FUNCS:
        return False
    if attr in FORBIDDEN_TENSOR_METHODS and qual not in ("torch", "F", "functional",
                                                         "torch.nn.functional", "nn.functional"):
        return True
    if qual == "torch" and attr not in ALLOWED_TORCH_FUNCS_TYPE:
        return True
    if qual in ("F", "functional", "torch.nn.functional", "nn.functional"):
        return True
    return False


def _loop_has_computation(for_node):
    """检查 for 循环体中是否包含实际的计算操作。

    计算操作包括：
    - 禁止的 tensor 方法（.sum(), .mul(), ...）
    - torch.xxx 计算调用
    - F.xxx 计算调用
    - BinOp 算术运算符（+, -, *, /, ** 等，作用于 tensor 时）
    - @ 矩阵乘法运算符
    """
    arithmetic_ops = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Pow, ast.Mod, ast.MatMult,
    )
    binop_count = 0

    for child in ast.walk(for_node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.MatMult):
            return True
        if isinstance(child, ast.BinOp) and isinstance(child.op, arithmetic_ops):
            binop_count += 1
            continue
        if isinstance(child, ast.Call):
            resolved = _resolve_call_name(child)
            if resolved and _call_is_computation(resolved[0], resolved[1]):
                return True

    # 阈值：5 个以上算术操作视为计算密集型循环
    if binop_count >= 5:
        return True
    return False


# ---------------------------------------------------------------------------
# 共享格式化工具函数（供 validate_tilelang_impl.py 导入复用）
# ---------------------------------------------------------------------------

def _format_violation_details(violations):
    """Format violation details for error messages (shared with TileLang validator)."""
    return "; ".join(
        f"第{v['line']}行 {v['call']}" for v in violations[:5]
    )


def _format_loop_details(loop_violations):
    """Format loop violation details for error messages (shared with TileLang validator)."""
    return "; ".join(
        f"第{v['line']}行 for {v['loop_var']} in range(...)"
        for v in loop_violations[:5]
    )


# ---------------------------------------------------------------------------
# 主验证逻辑
# NOTE: 此文件与 validate_tilelang_impl.py 共享相同的验证模式结构：
#   _make_result -> _check_ext_imported -> _check_kernel_called ->
#   _check_forbidden_ops -> _check_scalar_loops -> validate
#   _print_result -> main
# validate_tilelang_impl.py 导入了此文件的共享函数和常量。
# 新增的 _format_violation_details / _format_loop_details 也供其复用。
# ---------------------------------------------------------------------------

def _make_result(filepath, first_check_name="ascendc_ext_imported",
                 first_check_field="extensions"):
    return {
        "valid": False,
        "filepath": filepath,
        "checks": {
            first_check_name: {
                "passed": False, first_check_field: [], "error": None,
            },
            "kernel_called_from_forward": {
                "passed": False, "called": [], "error": None,
            },
            "no_forbidden_torch_ops": {
                "passed": False, "violations": [], "error": None,
            },
            "no_scalar_for_loops": {
                "passed": False, "violations": [], "error": None,
            },
        },
        "regression_type": None,
        "suggestion": "",
    }


def _check_ext_imported(result, tree):
    """Check 1: AscendC extension imports exist. Returns valid_ext_names or None (early return)."""
    extensions = find_ascendc_extension_imports(tree)
    ext_names = set(extensions.keys())

    result["checks"]["ascendc_ext_imported"]["extensions"] = [
        {"used_name": k, "actual_name": v["name"], "line": v["line"],
         "is_placeholder": v["is_placeholder"], "import_style": v["import_style"]}
        for k, v in extensions.items()
    ]

    if not ext_names:
        result["checks"]["ascendc_ext_imported"]["error"] = (
            "未找到任何 AscendC 扩展模块导入（如 import <op_name> 或 torch.ops.load_library）"
        )
        result["regression_type"] = 1
        result["suggestion"] = (
            "代码中没有加载 AscendC kernel 扩展。model_new_ascendc.py 必须通过 "
            "import <op_name> 导入 whl 包（__init__.py 内部调用 torch.ops.load_library()），"
            "并在 forward() 中通过 torch.ops.npu.<op_name>(...) 调用。"
        )
        return None

    placeholder_exts = [k for k, v in extensions.items() if v["is_placeholder"]]
    if len(placeholder_exts) == len(extensions):
        result["checks"]["ascendc_ext_imported"]["error"] = (
            f"扩展导入使用了占位符名称 {placeholder_exts}（如 TORCH_EXTENSION_NAME），"
            "扩展模块未正确配置"
        )
        result["regression_type"] = 1
        result["suggestion"] = (
            "扩展模块导入使用了占位符名称（如 import TORCH_EXTENSION_NAME），"
            "这表示 AscendC kernel 未正确编译或配置。"
            "请确保使用 NpuExtension 编译 kernel 并使用正确的模块名导入。"
        )
        return None

    result["checks"]["ascendc_ext_imported"]["passed"] = True
    return {k for k, v in extensions.items() if not v["is_placeholder"]}


def _check_kernel_called(result, tree, valid_ext_names):
    """Check 2: forward() calls kernel. Returns (forward_node, class_name) or (None, None)."""
    forward_node, class_name, _class_node = find_model_forward(tree)
    if forward_node is None:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            "未找到 ModelNew.forward() 或 Model.forward() 方法"
        )
        result["regression_type"] = 2
        result["suggestion"] = "代码缺少 ModelNew（或 Model）类或 forward 方法。"
        return None, None

    wrapper_names = find_wrapper_functions(tree, valid_ext_names)
    called = check_kernel_calls_in_forward(forward_node, valid_ext_names, wrapper_names)
    result["checks"]["kernel_called_from_forward"]["called"] = called

    if not called:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            f"已加载扩展 {list(valid_ext_names)} 但 {class_name}.forward() "
            f"未调用任何 kernel 函数"
        )
        result["regression_type"] = 2
        result["suggestion"] = (
            f"已加载 AscendC kernel 扩展 {list(valid_ext_names)} 但 "
            f"{class_name}.forward() 中未调用。"
            "forward() 必须通过 torch.ops.npu.<op_name>(...) 形式调用 kernel，"
            "或通过 ext_module.function_name(...) 调用。"
            f"{'也存在 wrapper 函数 ' + str(list(wrapper_names)) + ' 但 forward 也未调用它们。' if wrapper_names else ''}"
        )
        return None, None

    result["checks"]["kernel_called_from_forward"]["passed"] = True
    return forward_node, class_name


def _check_forbidden_ops(result, forward_node):
    """Check 3: no forbidden torch ops. Returns True to continue."""
    violations = check_forbidden_torch_ops(forward_node)
    result["checks"]["no_forbidden_torch_ops"]["violations"] = violations

    if violations:
        violation_details = _format_violation_details(violations)
        result["checks"]["no_forbidden_torch_ops"]["error"] = (
            f"forward() 中发现 {len(violations)} 处禁止的 PyTorch 计算操作"
        )
        result["regression_type"] = 3
        result["suggestion"] = (
            f"forward() 调用了 AscendC kernel 但仍使用 PyTorch 进行部分计算: "
            f"{violation_details}。"
            "所有核心计算必须在 AscendC kernel 中完成，"
            "forward() 中只允许 buffer 分配（torch.empty 等）和形状操作（.view/.reshape 等）。"
        )
        return False

    result["checks"]["no_forbidden_torch_ops"]["passed"] = True
    return True


def _check_scalar_loops(result, forward_node):
    """Check 4: no scalar for loops. Returns True to continue."""
    loop_violations = check_for_loops_over_tensors(forward_node, "AscendC kernel")
    result["checks"]["no_scalar_for_loops"]["violations"] = loop_violations

    if loop_violations:
        loop_details = _format_loop_details(loop_violations)
        result["checks"]["no_scalar_for_loops"]["error"] = (
            f"forward() 中发现 {len(loop_violations)} 处逐元素 Python for 循环"
        )
        result["regression_type"] = 4
        result["suggestion"] = (
            f"forward() 中存在逐元素 Python for 循环: {loop_details}。"
            "不能用标量逐元素写法，必须使用 AscendC kernel 的向量化 / 块级操作。"
        )
        return False

    result["checks"]["no_scalar_for_loops"]["passed"] = True
    return True


def validate(code, filepath="<unknown>"):
    """对生成代码执行完整的退化检查。

    返回结构化结果 dict。
    """
    result = _make_result(filepath)

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["checks"]["ascendc_ext_imported"]["error"] = f"SyntaxError: {e}"
        result["regression_type"] = 1
        result["suggestion"] = "代码存在语法错误，无法解析。"
        return result

    valid_ext_names = _check_ext_imported(result, tree)
    if valid_ext_names is None:
        return result

    forward_node, _class_name = _check_kernel_called(result, tree, valid_ext_names)
    if forward_node is None:
        return result

    if not _check_forbidden_ops(result, forward_node):
        return result

    if not _check_scalar_loops(result, forward_node):
        return result

    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class PrintConfig:
    first_check_name: str
    first_check_field: str
    kernel_label: str
    pass_header: str
    type_descs: dict


def _print_validation_result(result, cfg):
    """Shared logging for validation results (also used by validate_tilelang_impl.py)."""
    if result["valid"]:
        items = result["checks"][cfg.first_check_name][cfg.first_check_field]
        called = result["checks"]["kernel_called_from_forward"]["called"]
        logger.info(cfg.pass_header)
        logger.info("  - 导入 %d 个%s: %s",
                    len(items), cfg.kernel_label,
                    ', '.join(i['used_name'] for i in items))
        logger.info("  - forward() 调用: %s",
                    ', '.join(c['call'] for c in called))
        logger.info("  - forward() 中无禁止的 PyTorch 计算操作")
        logger.info("  - forward() 中无逐元素 Python for 循环")
        return

    rtype = result["regression_type"]
    logger.error("[FAIL] 检测到 PyTorch 退化 — Type %d: %s",
                 rtype, cfg.type_descs.get(rtype, '未知'))

    for check_name, check_result in result["checks"].items():
        status = "PASS" if check_result["passed"] else "FAIL"
        logger.info("  [%s] %s", status, check_name)
        if check_result["error"]:
            logger.info("         %s", check_result["error"])

    torch_violations = result["checks"]["no_forbidden_torch_ops"]["violations"]
    if torch_violations:
        logger.info("  torch 操作违规详情:")
        for v in torch_violations:
            logger.info("    第 %d 行: %s — %s", v['line'], v['call'], v['reason'])

    loop_violations = result["checks"]["no_scalar_for_loops"]["violations"]
    if loop_violations:
        logger.info("  for 循环违规详情:")
        for v in loop_violations:
            logger.info("    第 %d 行: for %s in range(...) — %s",
                        v['line'], v['loop_var'], v['reason'])

    logger.info("  修复建议: %s", result['suggestion'])


def _print_result(result):
    cfg = PrintConfig(
        first_check_name="ascendc_ext_imported",
        first_check_field="extensions",
        kernel_label="扩展模块",
        pass_header="[PASS] AscendC 实现验证通过",
        type_descs={
            1: "无 AscendC 扩展导入（纯 PyTorch / 占位符导入）",
            2: "有扩展导入但 forward() 未调用 kernel",
            3: "部分计算仍使用 PyTorch（需全部移入 AscendC kernel）",
            4: "存在逐元素 Python for 循环（需使用向量化操作）",
        },
    )
    _print_validation_result(result, cfg)


def _run_cli(description, validate_fn, print_result_fn):
    """Shared CLI entry point (also used by validate_tilelang_impl.py)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("file", help="要检查的 Python 文件路径")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        logger.error("文件不存在: %s", args.file)
        if args.json:
            _stdout_logger.info(json.dumps({"valid": False, "error": f"文件不存在: {args.file}"}))
        return 1

    result = validate_fn(code, filepath=args.file)

    if args.json:
        _stdout_logger.info(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result_fn(result)

    return 0 if result["valid"] else 1


def main():
    sys.exit(_run_cli(
        description="检查 AscendC 生成代码是否退化为 PyTorch 原生实现（AST 静态分析）",
        validate_fn=validate,
        print_result_fn=_print_result,
    ))


if __name__ == "__main__":
    main()
