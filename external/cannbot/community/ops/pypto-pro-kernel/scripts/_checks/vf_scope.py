# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check VF scope, procedure bodies and result-producing instruction nesting."""

import ast
import hashlib
from functools import lru_cache
from pathlib import Path

from .paths import PYPTO_ROOT
from .source import (
    Finding,
    apply_claim,
    attribute_path,
    counterexample_claims,
    has_decorator,
    read,
)

RULE = "vf-operation-needs-vector-function"
RETURN_RULE = "simd-vf-cannot-return"
DIVISION_RULE = "vf-scalar-division"
NESTED_RULE = "vf-result-nested-call"


@lru_cache(maxsize=8)
def _result_operations(pypto_root):
    """Read the selected parser's assignment table, guarded by its rejection form."""
    path = Path(pypto_root) / "python/pypto_pro/language/parser/_call_parser.py"
    identity = str(path)
    try:
        payload = path.read_bytes()
        identity += " (SHA256=" + hashlib.sha256(payload).hexdigest()[:12] + ")"
        tree = ast.parse(payload)
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CallParserMixin"
        )
        table = next(
            node.value
            for node in cls.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_VF_OP_DST_COUNT"
        )
        counts = ast.literal_eval(table)
        parser = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "_parse_vf_op"
        )
        guard = ast.parse(
            "dst_count is not None and dst_count > 0 and op_name != 'bit_cast'",
            mode="eval",
        ).body
        recognized = any(
            isinstance(node, ast.If)
            and ast.dump(node.test) == ast.dump(guard)
            and any(
                isinstance(part, ast.Constant)
                and isinstance(part.value, str)
                and "cannot be nested" in part.value
                for part in ast.walk(node)
            )
            for node in ast.walk(parser)
        )
        if not recognized or not isinstance(counts, dict) or not counts:
            return None, identity
        if not all(
            isinstance(name, str) and type(count) is int
            for name, count in counts.items()
        ):
            return None, identity
        return {name for name, count in counts.items() if count > 0} - {
            "bit_cast"
        }, identity
    except (OSError, SyntaxError, ValueError, StopIteration):
        return None, identity


def _simd_vf(function, alias):
    if not has_decorator(function, alias, "vector_function"):
        return False
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if attribute_path(decorator.func) != f"{alias}.vector_function":
            continue
        for keyword in decorator.keywords:
            if keyword.arg is None:
                return False
            if keyword.arg == "mode":
                return isinstance(
                    keyword.value, ast.Constant
                ) and keyword.value.value in (None, "simd")
    return True


def _number(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _number(node.operand)
        if value is not None:
            return -value if isinstance(node.op, ast.USub) else value
    return None


def _procedure_findings(source, function, pypto_root):
    found = []
    result_ops, identity = _result_operations(pypto_root)
    arguments = (
        function.args.posonlyargs + function.args.args + function.args.kwonlyargs
    )
    bound = {arg.arg for arg in arguments}
    bound.update(
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    prefixes = {f"{source.pl}.Vf"} if source.pl not in bound else set()
    if source.vf and source.vf not in bound:
        prefixes.add(source.vf)

    class Visitor(ast.NodeVisitor):
        call_depth = 0

        def visit_Call(self, node):
            name = attribute_path(node.func)
            if self.call_depth and name and name.rpartition(".")[0] in prefixes:
                operation = name.rpartition(".")[2]
                if result_ops is None:
                    if not any(item.rule == NESTED_RULE for item in found):
                        found.append(
                            Finding(
                                source.path,
                                node.lineno,
                                NESTED_RULE,
                                f"无法核对嵌套 VF 调用：{identity} 缺失或解析规则未识别；"
                                "用 --pypto-root 指定当前安装对应的源码，再核对 assignment-form 契约。",
                                advisory=True,
                            )
                        )
                elif operation in result_ops:
                    found.append(
                        Finding(
                            source.path,
                            node.lineno,
                            NESTED_RULE,
                            f"{name} 产生寄存器结果，不能嵌套在调用参数内；先用独立赋值绑定结果，"
                            f"再把变量传给外层调用。依据：{identity}。",
                        )
                    )
            self.call_depth += 1
            self.generic_visit(node)
            self.call_depth -= 1

        def visit_Return(self, node):
            found.append(
                Finding(
                    source.path,
                    node.lineno,
                    RETURN_RULE,
                    "SIMD @pl.vector_function 不能包含 return；结果通过 tile 参数写出。"
                    "寄存器计算留在 VF 内，跨 VF 调用的状态使用显式 UB tile。"
                    '普通 inline helper 和 mode="simt" 的返回规则不同。',
                )
            )
            self.generic_visit(node)

        def visit_BinOp(self, node):
            left, right = _number(node.left), _number(node.right)
            literal = left is not None and right is not None
            forbidden = isinstance(node.op, ast.Div) or (
                isinstance(node.op, (ast.FloorDiv, ast.Mod))
                and (isinstance(left, float) or isinstance(right, float))
            )
            if forbidden:
                found.append(
                    Finding(
                        source.path,
                        node.lineno,
                        DIVISION_RULE,
                        "VF 不支持标量 / 或浮点 //、%；静态倒数/系数放到 Python 工厂，"
                        "运行期标量除法放到外层 JIT。寄存器除法使用 vf.div。"
                        + ("" if literal else "操作数类型未静态确定，请核对实际类型。"),
                        advisory=not literal,
                    )
                )
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            pass

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

    visitor = Visitor()
    for statement in function.body:
        visitor.visit(statement)
    result = []
    for rule in (RETURN_RULE, DIVISION_RULE, NESTED_RULE):
        result += apply_claim(
            source,
            function,
            rule,
            [item for item in found if item.rule == rule],
            counterexample_claims(function),
        )
    return result


def check(path: Path, *, pypto_root: Path = PYPTO_ROOT):
    source = read(path)
    if not source.pl:
        return []
    findings = []
    for function in ast.walk(source.tree):
        if _simd_vf(function, source.pl):
            findings += _procedure_findings(source, function, pypto_root)
        if not has_decorator(function, source.pl, "jit"):
            continue
        arguments = (
            function.args.posonlyargs + function.args.args + function.args.kwonlyargs
        )
        bound = {arg.arg for arg in arguments}
        # Ambiguous namespace bindings stay unknown. This is not an interprocedural resolver.
        bound.update(
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
        if source.pl in bound:
            continue
        prefixes = {f"{source.pl}.Vf"}
        if source.vf and source.vf not in bound:
            prefixes.add(source.vf)
        found = []

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node):
                name = attribute_path(node.func)
                if name and name.rpartition(".")[0] in prefixes:
                    found.append(
                        Finding(
                            source.path,
                            node.lineno,
                            RULE,
                            f"{name} 必须在 @pl.vector_function 中执行；section_vector 不是 VF 作用域。"
                            "包括 mask/寄存器初始化和 VF load/store；将这些操作放入 VF helper，"
                            "从 JIT 调用并传入 tile/标量。",
                        )
                    )
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                pass

            visit_AsyncFunctionDef = visit_FunctionDef
            visit_ClassDef = visit_FunctionDef
            visit_Lambda = visit_FunctionDef

        visitor = Visitor()
        for statement in function.body:
            visitor.visit(statement)
        findings += apply_claim(
            source, function, RULE, found, counterexample_claims(function)
        )
    return findings
