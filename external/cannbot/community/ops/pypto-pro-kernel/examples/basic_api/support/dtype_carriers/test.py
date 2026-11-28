# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""22 个导出 dtype 的位宽、host 载体与逐字节搬运验证。

GM→Vec→GM 的正向路径按原始字节比较，固定 FP4 形状按两个逻辑元素一个字节校验。
FP8 参数拒绝错误位宽的 host 载体；动态 FP4 的具体算法宽度仍由调用方约束。
不受支持的数据通道和 reinterpret 路径保留负例，torch_npu 的创建限制单独验证。
"""

import numpy as np
import pytest
import torch

import pypto_pro.language as pl
from pypto_pro._errors import InvalidType

from . import reference as ref
from .kernel import (
    N,
    bool_via_reinterpret,
    gm_int8_to_tile_hf8,
    int4_via_reinterpret,
    make_dynamic_copy,
    make_static_copy,
)


def _filled_cpu(torch_dtype, last):
    """在 CPU 上造一个 (1, last) 的张量，按字节填非零花样。

    在 CPU 上造是必须的：`uint32` / `uint64` / `float8_e8m0fnu` 在卡上建不出来（§六）。
    按**字节**填而不是按值填：有些 dtype（fp8 / fp4）没有能直接赋值的 Python 标量路径，
    而这一例只关心字节搬对没搬对。
    """
    t = torch.zeros((1, last), dtype=torch_dtype)
    if torch_dtype == torch.bool:
        t.copy_((torch.arange(last, dtype=torch.int64) % 2).reshape(1, last).bool())
        return t
    raw = t.view(torch.uint8).reshape(-1)
    raw.copy_((torch.arange(raw.numel(), dtype=torch.int64) % 251 + 1).to(torch.uint8))
    return t


def _same_bytes(a, b):
    return bool((a.view(torch.uint8) == b.view(torch.uint8)).all().item())


# ---------------------------------------------------------------------------
# 一、22 个 dtype 的位宽
# ---------------------------------------------------------------------------


def test_the_22_dtypes_group_by_bit_width():
    """不需要卡。位宽从 `dtype.bits()` 读，分组与 reference 的表逐项对。"""
    names = sorted(n for n in dir(pl) if n.startswith("DT_"))
    assert len(names) == 22, (
        f"安装版本有 {len(names)} 个 DT_*，不是 22 个——reference 的表要重核"
    )

    measured = {}
    for name in names:
        measured.setdefault(getattr(pl, name).bits(), []).append(name)
    assert {k: sorted(v) for k, v in measured.items()} == {
        k: sorted(v) for k, v in ref.BITS.items()
    }

    # 69 = 2² + 6² + 4² + 3² + 2²：搬运表的组合数正好是「按位宽分组后组内任意配对」。
    # 每组要先剔掉不参与搬运的成员（4 位剔 4 个剩 2 个，8 位剔 BOOL 剩 6 个）。
    per_width = {}
    for width, members in ref.BITS.items():
        usable = [m for m in members if m not in ref.NOT_IN_DATA_PATH]
        per_width[width] = len(usable)
    assert sum(n * n for n in per_width.values()) == ref.ND_COMBO_COUNT, (
        f"按位宽两两配对得到 {sum(n * n for n in per_width.values())} 条，"
        f"搬运表是 {ref.ND_COMBO_COUNT} 条——「同位宽任意配对」这个规律不成立了"
    )
    assert sum(per_width.values()) == ref.IN_DATA_PATH_COUNT


# ---------------------------------------------------------------------------
# 二、5 个进不了搬运通道的
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dt_name,carrier,last",
    [
        ("DT_BOOL", torch.bool, N),  # 有映射，必须传 torch.bool
        ("DT_INT4", torch.uint8, N // 2),
        ("DT_UINT4", torch.uint8, N // 2),
        ("DT_FP4", torch.uint8, N // 2),
        ("DT_HF4", torch.uint8, N // 2),
    ],
)
def test_the_five_dtypes_outside_the_data_path_are_refused(
    device, dt_name, carrier, last
):
    """`load` 的组合表里没有它们。报错在 IR 层，指得很准。"""
    assert dt_name in ref.NOT_IN_DATA_PATH
    dtype = getattr(pl, dt_name)
    a = _filled_cpu(carrier, last).to(device)
    out = _filled_cpu(carrier, last).to(device)
    with pytest.raises(Exception, match=r"unsupported layout/dtype combination"):
        make_static_copy(dtype)(a, out)


# ---------------------------------------------------------------------------
# 三、能走通的载体
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dt_name,torch_name,ratio,needs_dynamic,note",
    ref.WORKING_CARRIERS,
    ids=[c[0] for c in ref.WORKING_CARRIERS],
)
def test_the_working_carriers_round_trip_byte_exact(
    device, dt_name, torch_name, ratio, needs_dynamic, note
):
    """每个载体走一趟 GM→Vec→GM，比字节。`note` 就是它为什么是这个载体。"""
    dtype = getattr(pl, dt_name)
    carrier = getattr(torch, torch_name)
    last = int(N * ratio)

    kernel = make_dynamic_copy(dtype) if needs_dynamic else make_static_copy(dtype)
    a = _filled_cpu(carrier, last).to(device)
    out = torch.zeros((1, last), dtype=carrier).to(device)  # CPU 建好再拷，见 §六
    kernel(a, out)
    torch.npu.synchronize()
    assert _same_bytes(a, out), f"{dt_name}（{note}）搬完字节不一致"


# ---------------------------------------------------------------------------
# 四、4 bit 静态宽度与缓存一致性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [pl.DT_FP4E2M1, pl.DT_FP4E1M2])
@pytest.mark.parametrize("wrong_first", [False, True])
def test_static_fp4_width_is_independent_of_call_order(device, dtype, wrong_first):
    """Logical width 64 accepts 32 storage bytes on both cold and cached launches."""
    kernel = make_static_copy(dtype)

    def call(host_last):
        a = _filled_cpu(torch.uint8, host_last).to(device)
        out = torch.zeros((1, host_last), dtype=torch.uint8, device=device)
        kernel(a, out)
        torch.npu.synchronize()
        return a, out

    if wrong_first:
        with pytest.raises(TypeError, match="expected 64, got 128"):
            call(N)
    for _ in range(2):
        a, out = call(N // 2)
        assert _same_bytes(a, out)
    with pytest.raises(TypeError, match="expected 64, got 128"):
        call(N)
    a, out = call(N // 2)
    assert _same_bytes(a, out)


def test_a_4_bit_tensor_with_the_wrong_host_width_goes_silently_wrong(device):
    """末轴是 `pl.DYNAMIC` 时形状不再被拦，给错宽度就**静默只搬一半**。

    这是本例第二个静默坑。正确值是 `N // 2`（字节数），这里故意给 `N`。
    """
    kernel = make_dynamic_copy(pl.DT_FP4E2M1)
    a = _filled_cpu(torch.uint8, N).to(device)  # ← 应该是 N // 2
    out = _filled_cpu(torch.uint8, N).to(device)
    out.zero_()
    kernel(a, out)  # 不报错
    torch.npu.synchronize()

    assert not _same_bytes(a, out), (
        "host 末轴给成逻辑元素数居然也搬对了——那这个坑不存在了，本例这条结论要重测"
    )
    a_cpu, out_cpu = a.cpu().numpy().reshape(-1), out.cpu().numpy().reshape(-1)
    np.testing.assert_array_equal(
        out_cpu[: N // 2], a_cpu[: N // 2], "前一半应该是搬对的"
    )
    assert (out_cpu[slice(N // 2, None)] == 0).all(), "后一半应该一个字节都没动"


# ---------------------------------------------------------------------------
# 五、host dtype 检查：有映射的严格，没映射的完全不查
# ---------------------------------------------------------------------------


def test_a_mapped_dtype_refuses_the_wrong_host_dtype(device):
    """`DT_UINT64` 在映射表里 → 只收 `torch.uint64`，`int64` 立刻报错。"""
    kernel = make_static_copy(pl.DT_UINT64)
    a = _filled_cpu(torch.int64, N).to(device)
    out = _filled_cpu(torch.int64, N).to(device)
    with pytest.raises(Exception, match=r"dtype mismatch: expected torch\.uint64"):
        kernel(a, out)


def test_fp8_refuses_a_host_carrier_with_the_wrong_width(device):
    """FP8 参数接受对应 torch FP8 或 uint8 载体，拒绝四字节的 float32 载体。"""
    kernel = make_static_copy(pl.DT_FP8E4M3FN)
    a = torch.arange(1, N + 1, dtype=torch.float32).reshape(1, N).to(device)
    out = torch.zeros((1, N), dtype=torch.float32).to(device)
    with pytest.raises(InvalidType) as rejected:
        kernel(a, out)
    message = str(rejected.value)
    assert "dtype mismatch" in message, message
    assert "torch.uint8" in message and "torch.float8_e4m3fn" in message, message
    assert "got torch.float32" in message, message


# ---------------------------------------------------------------------------
# 六、reinterpret：8 位有退路，4 位没有
# ---------------------------------------------------------------------------


def test_bool_travels_as_uint8_and_is_reinterpreted_on_chip(device):
    """`DT_BOOL` 搬不了，但声明成 `DT_UINT8` 搬进来、片上换视图可以。"""
    a = _filled_cpu(torch.uint8, N).to(device)
    out = torch.zeros((1, N), dtype=torch.uint8, device=device)
    bool_via_reinterpret(a, out)
    torch.npu.synchronize()
    assert _same_bytes(a, out)


def test_the_same_reinterpret_route_does_not_compile_for_int4(device):
    """4 位的 `DT_INT4` 走不通——编译就失败。钉住「没有可用路径」这个事实。"""
    a = _filled_cpu(torch.uint8, N // 2).to(device)
    out = _filled_cpu(torch.uint8, N // 2).to(device)
    with pytest.raises(Exception, match=r"Failed to compile kernel"):
        int4_via_reinterpret(a, out)


# ---------------------------------------------------------------------------
# 七、同位宽跨 dtype 搬运
# ---------------------------------------------------------------------------


def test_same_width_cross_dtype_movement_is_allowed(device):
    """GM 声明 `DT_INT8`、Tile 声明 `DT_HF8`——搬运表允许，不需要 `reinterpret`。

    这是 69 条组合那个规律的直接后果，也是「host 用 int8 装 hif8 字节」的正路。
    """
    a = _filled_cpu(torch.int8, N).to(device)
    out = torch.zeros((1, N), dtype=torch.int8, device=device)
    gm_int8_to_tile_hf8(a, out)
    torch.npu.synchronize()
    assert _same_bytes(a, out)


# ---------------------------------------------------------------------------
# 八、torch_npu 的限制
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("torch_name", ref.NO_DEVICE_SIDE_CREATION)
def test_three_torch_dtypes_cannot_be_created_on_the_device(device, torch_name):
    """`uint32` / `uint64` / `float8_e8m0fnu` 在卡上建不出张量，CPU 上建好再拷就行。

    这是 torch_npu 的限制，不是 pypto 的。钉住它，因为照常写 `torch.zeros(..., device=)`
    会在这三个 dtype 上莫名失败，而报错指向 `ZerosLikeKernelNpuOpApi.cpp`，看不出是 dtype 问题。
    """
    carrier = getattr(torch, torch_name)
    with pytest.raises(RuntimeError):
        torch.zeros((1, N), dtype=carrier, device=device)
    # 绕法：CPU 上建好再拷
    on_device = torch.zeros((1, N), dtype=carrier).to(device)
    assert on_device.dtype == carrier
