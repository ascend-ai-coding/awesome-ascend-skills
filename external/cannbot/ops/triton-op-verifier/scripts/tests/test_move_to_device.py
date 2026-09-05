# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""_common_utils.move_to_device 及其在 verify.py / benchmark.py 中复用的单元测试。

测试目标：
1. move_to_device 对 tensor / list / tuple / 标量 / 嵌套结构的迁移与类型保留。
2. verify_move_to_device 直接复用共享实现（抽取重复代码后不再单独维护）。
3. benchmark_move_inputs_to_device 在复用共享实现后具备 tuple 迁移能力（与 verify 对齐）。
4. verify 与 benchmark 对同一嵌套输入产生结构一致的结果。

运行环境仅需 CPU torch（无需 NPU）：通过 monkeypatch torch.Tensor.to 记录 device 实参，
验证“每个 tensor 都被 .to(device) 调用且 device 实参正确”。
"""
import torch
import pytest

import _common_utils as cu
import verify
import benchmark


CPU = torch.device("cpu")

# 白盒测试需断言 verify/benchmark 复用了共享实现。这些是模块级实现细节（下划线命名），
# 用 getattr 取出后在用例中引用，避免以 . 直接访问下划线成员。
verify_move_to_device = getattr(verify, "_move_to_device")
benchmark_move_inputs_to_device = getattr(benchmark, "_move_inputs_to_device")


def _flatten_tensors(x):
    """收集嵌套 list/tuple 中的所有 tensor，用于计数与取值校验。"""
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        out = []
        for e in x:
            out.extend(_flatten_tensors(e))
        return out
    return []


@pytest.fixture
def spy_to(monkeypatch):
    """包装 torch.Tensor.to，记录每次调用的 device 实参，随后委托原始实现。"""
    calls = []
    original = torch.Tensor.to

    def wrapper(self, *args, **kwargs):
        dev = args[0] if args else kwargs.get("device", None)
        calls.append(dev)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", wrapper)
    return calls


# ---------------------------------------------------------------------------
# 共享实现 move_to_device
# ---------------------------------------------------------------------------

def test_move_tensor_calls_to_with_device(spy_to):
    t = torch.tensor([1.0, 2.0])
    out = cu.move_to_device(t, CPU)

    assert isinstance(out, torch.Tensor)
    assert torch.equal(out.cpu(), t)
    assert spy_to == [CPU], "tensor 应恰好触发一次 .to(device) 且 device 实参正确"


def test_move_list_preserves_type_and_moves_tensors(spy_to):
    a, b = torch.tensor([1.0]), torch.tensor([2.0])
    out = cu.move_to_device([a, b], CPU)

    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(e, torch.Tensor) for e in out)
    assert len(spy_to) == 2


def test_move_tuple_preserves_type_and_moves_tensors(spy_to):
    a, b = torch.tensor([1.0]), torch.tensor([2.0])
    out = cu.move_to_device((a, b), CPU)

    assert isinstance(out, tuple), "tuple 输入必须保留 tuple 类型"
    assert len(out) == 2
    assert all(isinstance(e, torch.Tensor) for e in out)
    assert len(spy_to) == 2


def test_move_nested_mixed_preserves_structure():
    """嵌套 list/tuple + 标量 + None：结构/类型/标量值均保留，tensor 被迁移。"""
    t1, t2, t3 = torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])
    scalar_obj = 3.14  # 用同一对象校验“原样返回”（身份保持）
    none_obj = None
    data = [t1, (t2, "tag", scalar_obj), [t3, none_obj], 7]

    out = cu.move_to_device(data, CPU)

    assert isinstance(out, list)
    assert isinstance(out[1], tuple), "内层 tuple 必须保留 tuple 类型"
    assert isinstance(out[2], list), "内层 list 必须保留 list 类型"
    # tensor 取值保持
    assert torch.equal(out[0].cpu(), t1)
    assert torch.equal(out[1][0].cpu(), t2)
    assert torch.equal(out[2][0].cpu(), t3)
    # 标量 / 字符串 / None 原样透传（身份保持）
    assert out[1][1] == "tag"
    assert out[1][2] is scalar_obj
    assert out[2][1] is none_obj
    assert out[3] == 7


def test_move_scalars_returned_as_is():
    """标量（int/float/str/bool/None）应原样返回（同一对象）。"""
    for value in (1, 2.5, "hello", True, None):
        out = cu.move_to_device(value, CPU)
        assert out is value, f"标量 {value!r} 应原样返回"


def test_move_empty_containers_preserve_type():
    assert cu.move_to_device([], CPU) == []
    assert isinstance(cu.move_to_device([], CPU), list)
    assert cu.move_to_device((), CPU) == ()
    assert isinstance(cu.move_to_device((), CPU), tuple)


def test_move_deep_nesting_moves_all_tensors(spy_to):
    data = [[torch.tensor([1.0]), (torch.tensor([2.0]),)], torch.tensor([3.0])]
    out = cu.move_to_device(data, CPU)

    assert len(spy_to) == 3, "深层嵌套中的每个 tensor 都应触发 .to(device)"
    assert _flatten_tensors(out) and all(
        isinstance(e, torch.Tensor) for e in _flatten_tensors(out)
    )


# ---------------------------------------------------------------------------
# verify.py 复用共享实现
# ---------------------------------------------------------------------------

def test_verify_move_to_device_is_shared():
    """抽取重复代码后，verify_move_to_device 应直接指向共享实现。"""
    assert verify_move_to_device is cu.move_to_device


def test_verify_move_handles_tuple():
    a = torch.tensor([1.0])
    out = verify_move_to_device((a, 1), CPU)
    assert isinstance(out, tuple)
    assert torch.equal(out[0].cpu(), a)
    assert out[1] == 1


# ---------------------------------------------------------------------------
# benchmark.py 复用共享实现（补充 tuple 能力）
# ---------------------------------------------------------------------------

def test_benchmark_top_level_returns_list():
    out = benchmark_move_inputs_to_device([torch.tensor([1.0]), 2], CPU)
    assert isinstance(out, list), "顶层 inputs 为 list，返回仍应为 list"


def test_benchmark_preserves_nested_tuple():
    """新增能力：benchmark 现在与 verify 一致，嵌套 tuple 保留类型并迁移 tensor。"""
    a, b = torch.tensor([1.0]), torch.tensor([2.0])
    out = benchmark_move_inputs_to_device([(a, b), [a, "x"], 5], CPU)

    assert isinstance(out, list)
    assert isinstance(out[0], tuple), "嵌套 tuple 必须保留 tuple 类型（新增能力）"
    assert isinstance(out[1], list), "嵌套 list 必须保留 list 类型"
    assert torch.equal(out[0][0].cpu(), a)
    assert torch.equal(out[0][1].cpu(), b)
    assert out[1][1] == "x"
    assert out[2] == 5


def test_benchmark_scalars_passthrough():
    value = 42
    out = benchmark_move_inputs_to_device([value, None, "s"], CPU)
    assert out[0] is value
    assert out[1] is None
    assert out[2] == "s"


# ---------------------------------------------------------------------------
# verify 与 benchmark 一致性
# ---------------------------------------------------------------------------

def test_verify_and_benchmark_consistent_on_nested_input():
    """对同一顶层 list 输入，两者产出结构/取值一致。"""
    data = [torch.tensor([1.0]), (torch.tensor([2.0]), "k"), [torch.tensor([3.0]), None], 9]

    out_v = verify_move_to_device(data, CPU)
    out_b = benchmark_move_inputs_to_device(data, CPU)

    assert type(out_v) is type(out_b) is list
    assert isinstance(out_v[1], tuple) and isinstance(out_b[1], tuple)
    assert isinstance(out_v[2], list) and isinstance(out_b[2], list)
    for tv, tb in zip(_flatten_tensors(out_v), _flatten_tensors(out_b)):
        assert torch.equal(tv.cpu(), tb.cpu())
    assert out_v[1][1] == out_b[1][1] == "k"
    assert out_v[3] == out_b[3] == 9
