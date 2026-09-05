# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""沙箱逃逸回归测试。

历史漏洞：`__builtins__["__import__"]("os").system(...)` 在 exec 模式下能绕过
Name 黑名单（`__builtins__` 当时只查 _BANNED_NAMES 列表，未拦 dunder Name）。
修复方式：validate_ast 增加全量 dunder Name 拒绝。本测试锁死这条契约。
"""
from __future__ import annotations

import ast

import pytest

from evaluators._ast_sandbox import SandboxError, validate_ast


class TestDunderNameBlocked:
    """所有 __X__ 形态的 Name 必须被沙箱拒绝。"""

    @pytest.mark.parametrize("src", [
        # 主漏洞向量：通过 __builtins__ 字典拿到 __import__
        'y = __builtins__["__import__"]("os").system("echo pwned")',
        # 等价变体
        '__builtins__["eval"]("1+1")',
        # 直接拿 __builtins__ 列出键
        'k = __builtins__',
        # __class__ / __globals__ / __dict__ 等其他常见逃逸入口
        'c = __class__',
        'g = __globals__',
        'd = __dict__',
    ])
    def test_dunder_name_rejected(self, src: str):
        tree = ast.parse(src, mode="exec")
        with pytest.raises(SandboxError, match="dunder|banned_name"):
            validate_ast(tree)


class TestBannedNamesStillBlocked:
    """显式黑名单标识符仍被拦下（防止重构时误删）。"""

    @pytest.mark.parametrize("src", [
        '__import__("os")',
        'eval("1+1")',
        'exec("print(1)")',
        'getattr(np, "abs")',
        'open("/etc/passwd")',
    ])
    def test_banned_name_rejected(self, src: str):
        tree = ast.parse(src, mode="exec")
        with pytest.raises(SandboxError, match="banned_name"):
            validate_ast(tree)


class TestDunderAttributeBlocked:
    """().__class__.__bases__ 风格的逃逸链仍被拦下。"""

    @pytest.mark.parametrize("src", [
        'x = ().__class__',
        'x = ().__class__.__bases__',
        'x = ().__class__.__bases__[0].__subclasses__()',
        'x = f.__globals__',
    ])
    def test_dunder_attribute_rejected(self, src: str):
        tree = ast.parse(src, mode="exec")
        with pytest.raises(SandboxError, match="dunder|banned_name"):
            validate_ast(tree)


class TestLegitimateFormulasStillPass:
    """确认沙箱加固没误伤合法 formula 写法。"""

    @pytest.mark.parametrize("src", [
        'y = np.exp(x)',
        'm = x.max(axis=dim, keepdims=True)',
        'e = np.exp(x - m)',
        'y = e / e.sum(axis=dim, keepdims=True)',
        'c = np.matmul(a, b)',
        'out = np.broadcast_shapes(a.shape, b.shape)',
        'y = a if cond else b',
    ])
    def test_normal_formula_passes(self, src: str):
        tree = ast.parse(src, mode="exec")
        validate_ast(tree)  # must not raise
