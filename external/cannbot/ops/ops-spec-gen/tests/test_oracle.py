# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stage 9 oracle_reachable 单测——单 callable 模式 + composition DAG 模式。

单 api 模式覆盖：reachable / api typo / framework mismatch / placeholder / absent 签字 /
                严格占位符 regex；
composition 模式覆盖：DAG 通过 / output 不存在 / id 重复 / id 撞 input 名 / 前向引用 /
                     args 引用未知 input / args/kwargs 占位符不存在 / output 占位符禁用 /
                     节点 api typo / 节点 framework mismatch / kwargs 字面量直传。
"""
from __future__ import annotations

import pytest

from evaluators.oracle_check import stage_9, _check_placeholders  # noqa: E402


def _oracle_spec(framework="numpy", api="numpy.add", absent=False, kwargs=None) -> dict:
    return {
        "math_semantics": {
            "reference_oracle": {
                "framework": framework,
                "api": api,
                "absent": absent,
                "kwargs": kwargs or {},
                "available_for_dtype": ["float32"],
            },
        },
        "attributes": [{"name": "dim"}],
        "inputs": [{"name": "x"}],
        "outputs": [{"name": "y"}],
        "dtype_policy": {
            "supported_combinations": [
                {"inputs": {"x": "float32"}, "outputs": {"y": "float32"}},
            ],
        },
    }


class TestStage9:
    def test_numpy_add_reachable(self):
        # numpy is always installed in our test env
        s = _oracle_spec(framework="numpy", api="numpy.add")
        status, findings = stage_9(s)
        assert status == "PASS", findings

    def test_api_typo(self):
        s = _oracle_spec(framework="numpy", api="numpy.adds")  # typo
        status, findings = stage_9(s)
        assert status == "FAIL"
        assert any("api_not_found" in f["rule_id"] for f in findings)

    def test_framework_mismatch(self):
        s = _oracle_spec(framework="torch", api="numpy.add")  # head mismatch
        status, findings = stage_9(s)
        assert status == "FAIL"
        assert any("api_framework_mismatch" in f["rule_id"] for f in findings)

    def test_placeholder_unresolved(self):
        s = _oracle_spec(
            framework="numpy", api="numpy.add",
            kwargs={"axis": "${attr.nonexistent}"},
        )
        status, findings = stage_9(s)
        assert status == "FAIL"
        assert any("placeholder_unresolved" in f["rule_id"] for f in findings)

    def test_placeholder_valid(self):
        s = _oracle_spec(
            framework="numpy", api="numpy.add",
            kwargs={"axis": "${attr.dim}"},  # 'dim' exists in attributes
        )
        status, findings = stage_9(s)
        assert status == "PASS", findings

    def test_absent_acknowledged_skip(self):
        s = _oracle_spec(absent=True)
        status, findings = stage_9(s)
        assert status == "SKIP"
        assert any("oracle_reachable.absent" in f["rule_id"] for f in findings)

    def test_absent_not_required_to_ack(self):
        # absent=true 自身即视为 spec 作者显式声明，无需额外签字。
        s = _oracle_spec(absent=True)
        status, _ = stage_9(s)
        assert status == "SKIP"

    def test_placeholder_subscript_rejected(self):
        # Strict regex (#26): ${attr.foo[0]} should not match → not validated → no false-positive
        s = _oracle_spec(
            framework="numpy", api="numpy.add",
            kwargs={"axis": "${attr.foo[0]}"},   # malformed placeholder syntax
        )
        status, _ = stage_9(s)
        # Strict regex → placeholder regex doesn't match → no placeholder check → PASS
        assert status == "PASS"


def _composition_spec(composition: list[dict], output: str = "out") -> dict:
    """构造一个最小化、能跑 stage 9 composition 路径的 spec。"""
    return {
        "math_semantics": {
            "reference_oracle": {
                "framework": "numpy",
                "composition": composition,
                "output": output,
                "absent": False,
                "available_for_dtype": ["float32"],
            },
        },
        "attributes": [{"name": "scale"}, {"name": "zero_point"}],
        "inputs": [{"name": "x"}, {"name": "w"}],
        "outputs": [{"name": "y"}],
        "dtype_policy": {
            "supported_combinations": [
                {"inputs": {"x": "float32", "w": "float32"}, "outputs": {"y": "float32"}},
            ],
        },
    }


class TestStage9Composition:
    def test_simple_dag_passes(self):
        # numpy.subtract → numpy.multiply → numpy.matmul（裸 input + 前序节点 id 引用）
        spec = _composition_spec([
            {"id": "w_dq", "api": "numpy.subtract", "args": ["w", "${attr.zero_point}"]},
            {"id": "w_scaled", "api": "numpy.multiply", "args": ["w_dq", "${attr.scale}"]},
            {"id": "out", "api": "numpy.matmul", "args": ["x", "w_scaled"]},
        ])
        status, findings = stage_9(spec)
        assert status == "PASS", findings

    def test_output_unresolved(self):
        spec = _composition_spec([
            {"id": "w_dq", "api": "numpy.subtract", "args": ["w", 0]},
        ], output="nope")
        status, findings = stage_9(spec)
        assert status == "FAIL"
        assert any("composition_output_unresolved" in f["rule_id"] for f in findings)

    def test_id_collision(self):
        spec = _composition_spec([
            {"id": "n1", "api": "numpy.subtract", "args": ["w", 0]},
            {"id": "n1", "api": "numpy.multiply", "args": ["w", 1]},
            {"id": "out", "api": "numpy.matmul", "args": ["x", "n1"]},
        ])
        status, findings = stage_9(spec)
        assert status == "FAIL"
        assert any("composition_id_collision" in f["rule_id"] for f in findings)

    def test_id_shadows_input(self):
        spec = _composition_spec([
            {"id": "x", "api": "numpy.subtract", "args": ["w", 0]},  # 撞 input 名
            {"id": "out", "api": "numpy.matmul", "args": ["x", "w"]},
        ])
        status, findings = stage_9(spec)
        assert status == "FAIL"
        assert any("composition_id_shadows_input" in f["rule_id"] for f in findings)

    def test_forward_reference_rejected(self):
        # 第一节点引用尚未声明的 future 节点 id（即潜在循环）
        spec = _composition_spec([
            {"id": "out", "api": "numpy.matmul", "args": ["x", "w_dq"]},
            {"id": "w_dq", "api": "numpy.subtract", "args": ["w", 0]},
        ])
        status, findings = stage_9(spec)
        assert status == "FAIL"
        assert any("composition_arg_unresolved" in f["rule_id"] for f in findings)

    def test_arg_unknown_input(self):
        spec = _composition_spec([
            {"id": "out", "api": "numpy.matmul", "args": ["x", "nonexistent_input"]},
        ])
        status, findings = stage_9(spec)
        assert status == "FAIL"
        assert any("composition_arg_unresolved" in f["rule_id"] for f in findings)

    def test_arg_attr_placeholder_unresolved(self):
        spec = _composition_spec([
            {"id": "out", "api": "numpy.matmul", "args": ["x", "${attr.nonexistent}"]},
        ])
        status, findings = stage_9(spec)
        assert status == "FAIL"
        assert any("placeholder_unresolved" in f["rule_id"] for f in findings)

    def test_output_placeholder_in_args_rejected(self):
        # ${output.X} 在 args 里非法（output 是另一个名字空间）
        spec = _composition_spec([
            {"id": "out", "api": "numpy.matmul", "args": ["x", "${output.y}"]},
        ])
        status, findings = stage_9(spec)
        assert status == "FAIL"
        assert any("不允许" in (f.get("message") or "") for f in findings)

    def test_node_api_typo(self):
        spec = _composition_spec([
            {"id": "out", "api": "numpy.matmuls", "args": ["x", "w"]},  # 拼错
        ])
        status, findings = stage_9(spec)
        assert status == "FAIL"
        assert any("api_not_found" in f["rule_id"] for f in findings)

    def test_node_framework_mismatch(self):
        # 节点 api 头段必须等于 framework
        spec = _composition_spec([
            {"id": "out", "api": "torch.matmul", "args": ["x", "w"]},  # torch ≠ numpy
        ])
        status, findings = stage_9(spec)
        assert status == "FAIL"
        assert any("api_framework_mismatch" in f["rule_id"] for f in findings)

    def test_kwargs_attr_placeholder_unresolved(self):
        # P0 #1：composition 节点的 kwargs 中 ${attr.X} 占位符必须校验存在
        spec = _composition_spec([
            {"id": "out", "api": "numpy.matmul", "args": ["x", "w"],
             "kwargs": {"axis": "${attr.nonexistent}"}},
        ])
        status, findings = stage_9(spec)
        assert status == "FAIL"
        assert any("placeholder_unresolved" in f["rule_id"] for f in findings)

    def test_kwargs_attr_placeholder_valid(self):
        # 合法占位符应通过（kwargs 引用真实 attr）
        spec = _composition_spec([
            {"id": "out", "api": "numpy.matmul", "args": ["x", "w"],
             "kwargs": {"alpha": "${attr.scale}"}},   # scale 是 _composition_spec 中已声明的 attr
        ])
        status, findings = stage_9(spec)
        assert status == "PASS", findings

    def test_kwargs_output_placeholder_rejected(self):
        # ${output.X} 在 kwargs 中也禁止（与 args 同）
        spec = _composition_spec([
            {"id": "out", "api": "numpy.matmul", "args": ["x", "w"],
             "kwargs": {"alpha": "${output.y}"}},
        ])
        status, findings = stage_9(spec)
        assert status == "FAIL"
        assert any("不允许" in (f.get("message") or "") for f in findings)

    def test_kwargs_literals_pass(self):
        # 字面量 kwargs（如 approximate='tanh'）不走占位符校验，直接通过
        spec = _composition_spec([
            {"id": "out", "api": "numpy.matmul", "args": ["x", "w"],
             "kwargs": {"approximate": "tanh", "out": None}},
        ])
        status, findings = stage_9(spec)
        assert status == "PASS", findings

