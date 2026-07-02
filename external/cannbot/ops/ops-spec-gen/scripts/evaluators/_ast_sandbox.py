# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""AST sandbox shared by shape_eval / dtype_eval / formula_eval.

三个 evaluator 都是「在受限 numpy 子集 AST 上 exec 表达式」的形式，沙箱策略一致：
  * AST 节点白名单 — 拒绝 import / def / class / for / while / try / lambda
  * 标识符黑名单 — 拒绝 __import__ / exec / getattr 等绕沙箱通路
  * 全量 dunder 名字拒绝（Name + Attribute 两侧对称） — 阻断
    `__builtins__["__import__"]("os")`、`().__class__.__bases__[0].__subclasses__()`
    这类逃逸路径；spec.yaml 不需要 dunder 标识符
  * 受限 builtins — 只暴露 abs / min / max / range / len / int / float / bool / True / False / None
    （`__import__` 必须留，numpy 内部 submodule 解析依赖；安全性靠 dunder Name 黑名单
    保证用户拿不到 `__builtins__` 字典）
  * 可选 SIGALRM 超时 — POSIX 主线程下硬中断；其他平台软退（formula 都是小 shape 向量化操作）

维护要点：
  * 修改 _ALLOWED_BUILTINS 时必须同步更新 _BANNED_NAMES
  * 三个 evaluator 业务语义不同，但都先经过 _validate_ast 同一闸口
"""

from __future__ import annotations

import ast
import signal
import threading
from contextlib import contextmanager


_ALLOWED_AST_NODES = frozenset([
    "Module", "Expr",
    "Assign", "AugAssign",
    "BinOp", "UnaryOp", "BoolOp", "Compare",
    "Call", "Attribute", "Subscript",
    "Name", "Constant", "Load", "Store",
    "Tuple", "List", "Slice",
    "IfExp",
    "keyword",   # func(name=val)
    "Starred",   # *args 解包
    # operators
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow",
    "USub", "UAdd",
    "And", "Or", "Not",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot",
    "Index",  # py<3.9 兼容
])

_BANNED_NAMES = frozenset([
    "__import__", "exec", "eval", "compile", "open", "input",
    "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "hasattr", "breakpoint", "exit", "quit",
])

# 受限 builtins。`__import__` 必须留：numpy 内部解析 submodule 时通过
# `__builtins__["__import__"]` 走，删掉会触发 `KeyError: '__import__'`。
# 安全性靠 validate_ast 里禁止 `__builtins__` / `__import__` 等 dunder Name 来保证——
# 用户无法在 formula 里写出 `__builtins__["__import__"](...)`，也无法直接调 `__import__`。
_ALLOWED_BUILTINS = {
    "abs": abs, "min": min, "max": max, "range": range,
    "len": len, "int": int, "float": float, "bool": bool,
    "True": True, "False": False, "None": None,
    "__import__": __import__,
}


class SandboxError(Exception):
    """AST 阶段拒绝；evaluator 各自包装为自己的领域错误（DslError / FormulaError）。"""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def validate_ast(tree: ast.AST) -> None:
    """走白名单 + 黑名单 + dunder 检查；不通过抛 SandboxError。"""
    for node in ast.walk(tree):
        cls = type(node).__name__
        if cls not in _ALLOWED_AST_NODES:
            raise SandboxError(
                "ast_disallowed",
                f"AST 节点 {cls} 不在白名单（表达式必须是纯赋值 / 调用，"
                f"禁止 import / def / class / for / while / try / lambda）",
            )
        if isinstance(node, ast.Name):
            if node.id in _BANNED_NAMES:
                raise SandboxError(
                    "banned_name",
                    f"标识符 {node.id!r} 不允许（绕沙箱）",
                )
            # 全量 dunder 标识符封禁：__builtins__ / __class__ / __globals__ / __dict__ ...
            # 这些是 `__builtins__["__import__"]` 类逃逸链的入口。
            if node.id.startswith("__") and node.id.endswith("__"):
                raise SandboxError(
                    "banned_name",
                    f"dunder 标识符 {node.id!r} 不允许（沙箱逃逸路径）",
                )
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxError(
                "banned_name",
                f"属性名 {node.attr!r} 不允许（dunder 通向 object 子类树等逃逸路径）",
            )


def _signal_alarm_supported() -> bool:
    """SIGALRM 仅 POSIX 主线程可用；其他平台走软超时退化。"""
    return (hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread())


@contextmanager
def timeout(seconds: int, on_timeout_code: str = "timeout"):
    """Wallclock 超时上下文。POSIX 主线程下用 SIGALRM 硬中断。

    on_timeout_code 让调用方决定超时事件的错误码（如 formula_timeout / shape_eval_timeout）。
    """
    if _signal_alarm_supported():
        def _handler(signum, frame):
            raise SandboxError(on_timeout_code, f"执行超时（>{seconds}s）")
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        # 软超时：调用方自行兜底；小 shape 向量化都 < 100ms
        yield


def make_globals(extra: dict) -> dict:
    """构造受限 globals。extra 是 evaluator 自己暴露的命名空间（如 np / inputs / attrs）。"""
    g = {"__builtins__": _ALLOWED_BUILTINS}
    g.update(extra)
    return g
