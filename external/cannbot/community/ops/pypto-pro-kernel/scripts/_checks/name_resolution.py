# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Optional lexical-name checking without importing candidate or framework code.

PyPTO tiling keys inject names outside Python's lexical bindings. Files using
that mechanism, star imports or dynamic namespaces are left unverified.
"""

import ast
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .source import Finding, read

RULE = "undefined-name"


def backend():
    constraints = Path(__file__).resolve().parent.parent / "requirements-checks.txt"
    expected = next(
        (
            line.split("==", 1)[1]
            for line in constraints.read_text().splitlines()
            if line.startswith("pyflakes==")
        ),
        None,
    )
    try:
        from pyflakes import api

        installed = version("pyflakes")
    except (ImportError, PackageNotFoundError):
        return None, "unavailable"
    if installed != expected or not callable(getattr(api, "check", None)):
        return None, f"unsupported {installed}"
    return api, installed


def inspect(path: Path) -> tuple[list[Finding], dict]:
    source = read(path)
    api, installed = backend()
    scope = {"checked": 0, "unverified": 1, "backend": installed, "reason": ""}
    for node in ast.walk(source.tree):
        if isinstance(node, ast.keyword):
            if node.arg == "tiling_key":
                scope["reason"] = "framework-injected tiling-key names"
                return [], scope
            if node.arg is None:
                scope["reason"] = (
                    "keyword expansion may supply injected-name configuration"
                )
                return [], scope
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            scope["reason"] = "star-import names"
            return [], scope
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"exec", "eval", "globals", "locals"}
        ):
            scope["reason"] = "dynamic namespace"
            return [], scope
    if api is None:
        scope["reason"] = (
            "install scripts/requirements-checks.txt for lexical-name checks"
        )
        return [], scope
    findings = []
    errors = []

    class Reporter:
        def flake(self, message):
            if type(message).__name__ == "UndefinedName":
                findings.append(
                    Finding(
                        source.path,
                        message.lineno,
                        RULE,
                        message.message % message.message_args,
                    )
                )

        def syntaxError(self, filename, message, lineno, offset, text):
            errors.append(message)

        def unexpectedError(self, filename, message):
            errors.append(message)

    try:
        api.check(path.read_text(encoding="utf-8"), str(path), reporter=Reporter())
    except (AttributeError, TypeError) as exc:
        errors.append(f"unsupported checker protocol: {exc}")
    if errors:
        scope["reason"] = "; ".join(errors)
        return [], scope
    scope.update(checked=1, unverified=0)
    return findings, scope


def check(path: Path) -> list[Finding]:
    return inspect(path)[0]
