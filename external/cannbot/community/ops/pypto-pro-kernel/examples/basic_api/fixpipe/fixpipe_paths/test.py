# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fixpipe 的 Acc→GM、Acc→Vec→GM 与 Acc→Mat→GEMM 验证。

FP16 小整数输入使独立 NumPy 数学参考保持 rtol=atol=0。ND/NZ 输出同时检查数据和物理排布，
链式 GEMM 使用显式 FP16 中间舍入。GM Tensor 注解仅接受 ND/NZ，DN 反例只检查解析期拒绝，
不编译或执行不受支持的落盘路径。
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import pypto_pro as pp
from pypto_pro._errors import InvalidArgument

from . import reference as ref
from .kernel import (
    K,
    M,
    N,
    acc_to_gm,
    acc_to_mat_chain,
    acc_to_vec_dn,
    acc_to_vec_nd,
    acc_to_vec_nz,
)

_LAYOUTS = [("ND", acc_to_vec_nd), ("NZ", acc_to_vec_nz)]


def _dev(array, device):
    return torch.from_numpy(array).to(device)


def _out(device):
    return torch.zeros((M, N), dtype=torch.float32, device=device)


def _run(kernel, device):
    a, b = ref.make_operands(M, K, N)
    out = _out(device)
    kernel(_dev(a, device), _dev(b, device), out)
    torch.npu.synchronize()
    return a, b, out.cpu().numpy()


# ---------------------------------------------------------------------------
# 三个目的地
# ---------------------------------------------------------------------------


def test_acc_to_gm_is_one_fixpipe(device):
    """`pl.store(out, acc, [0, 0])`——一行，一次 fixpipe，没有中间 Tile。"""
    a, b, got = _run(acc_to_gm, device)
    np.testing.assert_allclose(
        got, ref.matmul(a, b), rtol=0, atol=0, err_msg="Acc→GM 的结果不对"
    )


def test_acc_to_vec_then_store(device):
    """`pl.move(ub, acc)` + `pl.store(out, ub, ...)`——两步，只有第一步是 fixpipe。"""
    a, b, got = _run(acc_to_vec_nd, device)
    np.testing.assert_allclose(
        got, ref.matmul(a, b), rtol=0, atol=0, err_msg="Acc→Vec→GM 的结果不对"
    )


def test_acc_to_mat_feeds_the_next_gemm(device):
    """`pl.move(l1, acc)`：结果留在 L1 当下一次乘法的操作数，算 `fp16(a@b) @ c`。

    **这是 Acc→Mat 唯一的出路。** `Mat→Vec` 不在 `move` 的合法路径里，
    `Mat→GM` 不在 store 的组合表里——见 kernel.py 里那张表。
    `move` 的 docstring 却把 `Mat → Vec` 列成走 v 流水，与实现不符。

    那一步顺带做 `FP32 → FP16` 降位：**降位在 fixpipe 里免费**，
    不需要单独一趟向量转换。
    """
    a, b = ref.make_operands(M, K, N)
    c = ref.make_third(N)
    out = _out(device)
    acc_to_mat_chain(_dev(a, device), _dev(b, device), _dev(c, device), out)
    torch.npu.synchronize()
    np.testing.assert_allclose(
        out.cpu().numpy(),
        ref.matmul_chain(a, b, c),
        rtol=0,
        atol=0,
        err_msg="Acc→Mat 再进第二次 gemm 的结果不对",
    )


def test_acc_to_gm_and_acc_to_vec_agree(device):
    """Acc→GM 和 Acc→Vec→GM 的数值必须一致——否则其中一条的布局转换错了。

    这条不依赖参考实现：它是两个输出之间的关系。
    Acc→Mat 那条进不了这个对照（它的输出是另一个式子），单独由上一条测。
    """
    _, _, from_gm = _run(acc_to_gm, device)
    _, _, from_vec = _run(acc_to_vec_nd, device)

    np.testing.assert_allclose(
        from_vec, from_gm, rtol=0, atol=0, err_msg="Acc→Vec 与 Acc→GM 结果不一致"
    )


# ---------------------------------------------------------------------------
# 三种布局
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "kernel"), _LAYOUTS, ids=[n for n, _ in _LAYOUTS])
def test_each_layout_moves_the_same_values(name, kernel, device):
    """三种 `LayoutMode_t` 各自跑通，且搬出去的**值的集合**和参考一样。

    不逐位置比，因为 DN / NZ 的排布本来就和 ND 不同——比的是「没丢没多没改值」。
    逐位置的差异由下一条测。
    """
    a, b, got = _run(kernel, device)
    want = ref.matmul(a, b)
    np.testing.assert_allclose(
        np.sort(got.reshape(-1)),
        np.sort(want.reshape(-1)),
        rtol=0,
        atol=0,
        err_msg=f"{name} 布局搬出来的值集合和参考不一样",
    )


def test_nd_and_nz_differ(device):
    """防空测：`layout=` 必须真的改变排布。

    如果两种布局的输出逐位相同，说明 `layout=` 没被传下去，两个变体退化成同一个测试。
    """
    _, _, from_nd = _run(acc_to_vec_nd, device)
    _, _, from_nz = _run(acc_to_vec_nz, device)

    assert not np.array_equal(from_nd, from_nz), (
        "ND 与 NZ 的输出逐位相同——layout= 没起作用（或这批输入恰好对称）"
    )


def test_a_two_dimensional_dn_tile_has_no_store_path():
    """DN Tensor 注解在解析期拒绝，并指出支持的 ND/NZ 布局。"""
    with pytest.raises(InvalidArgument, match="Unknown layout: DN") as raised:
        acc_to_vec_dn.to_kernel_def().parse_target_program(pp.ir.SectionKind.Cube)
    assert "Tensor only supports pl.ND and pl.NZ" in str(raised.value)


# ---------------------------------------------------------------------------
# 结构检查
# ---------------------------------------------------------------------------


def test_the_input_has_both_signs():
    """不需要卡。`fixpipe_riders` 要用同一个构造函数验随路 relu，
    全正的输入下 relu 是恒等映射。这里先把前提钉住。
    """
    a, _ = ref.make_operands(M, K, N)
    assert ref.has_both_signs(a), "输入没有负数——随路 relu 那一例会验空"


def test_the_gemm_producer_is_shared():
    """不需要卡。本例三条输出路径共用一个 `_cube_groups()` 生产者。"""
    source = (Path(__file__).parent / "kernel.py").read_text()
    assert source.count("def _cube_groups(") == 1, "生产者应当只声明一次"
    calls = [
        line
        for line in source.splitlines()
        if "_cube_groups()" in line and not line.lstrip().startswith("def ")
    ]
    assert len(calls) == 3, (
        f"三个 kernel（acc_to_gm / 工厂 / acc_to_mat_chain）各调一次，实际 {len(calls)} 处"
    )
