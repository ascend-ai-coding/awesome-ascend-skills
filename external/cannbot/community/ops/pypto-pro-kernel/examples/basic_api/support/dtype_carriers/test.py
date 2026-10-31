# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""22 个 `DT_*`：位宽、host 载体、进不了搬运通道的那 5 个，以及两个静默坑。

**写法：原始。** 这一例是 `vf.astype` 的前置——转换的两端各用什么载体，先在这里定清楚。

## 一、22 个 dtype 的形状

| 位宽 | 个数 | 成员 |
| --- | --- | --- |
| 4 | 6 | `FP4` `FP4E1M2` `FP4E2M1` `HF4` `INT4` `UINT4` |
| 8 | 7 | `BOOL` `FP8E4M3FN` `FP8E5M2` `FP8E8M0` `HF8` `INT8` `UINT8` |
| 16 | 4 | `BF16` `FP16` `INT16` `UINT16` |
| 32 | 3 | `FP32` `INT32` `UINT32` |
| 64 | 2 | `INT64` `UINT64` |

**其中 5 个根本进不了搬运通道**：`BOOL` `FP4` `HF4` `INT4` `UINT4`。板上报
`load: unsupported layout/dtype combination src=GM(ND,DT_X) -> dst=Vec(ND,DT_X)`。
剩下 17 个参与搬运，GM(ND)→Vec(ND) 共 **69** 条组合 = 2² + 6² + 4² + 3² + 2²
——**同位宽任意配对，不跨位宽**。所以 GM 声明 `INT8`、Tile 声明 `HF8` 是合法的，
不需要 `reinterpret`（`test_same_width_cross_dtype_movement_is_allowed`）。

## 二、载体：只有 11 个 dtype 的 host 张量 dtype 被检查

`runtime/jit.py` 的 `_PL_DTYPE_TO_TORCH` 只有 11 条，而检查是
`if expected_dtype is not None and arg.dtype != expected_dtype`。所以：

- **有映射的 11 个**（`FP16` `BF16` `FP32` `INT8` `INT16` `INT32` `INT64` `UINT8` `UINT64`
  `BOOL` `HF8`）：host dtype 严格对齐，错了立刻报 `dtype mismatch`。
  注意 `HF8` 映射到 **`uint8`**——torch 没有 hif8，传的是裸字节。
- **没映射的 11 个**：**完全不检查**。传 `torch.float32` 给一个 8 位参数会跑通、不报错、
  数据错（`test_an_unmapped_dtype_takes_any_host_dtype_and_goes_silently_wrong`）。

## 三、4 bit 张量形状按逻辑元素计数

固定声明 `[1, 64]` 对应 host `(1, 32)` 字节载体；冷调用与缓存命中使用同一口径。
`test_static_fp4_width_is_independent_of_call_order` 覆盖两种 FP4 和错误调用前后的状态。
动态末轴仍可使用，但调用方须自行确认 carrier 宽度足够且符合算法约定。

## 四、两个静默坑

| 坑 | 症状 |
| --- | --- |
| 没映射的 dtype 传错位宽的 host 张量 | 跑通，不报错，**只搬了声明位宽那么多字节**，尾部没动 |
| 动态末轴的 4 bit host 按逻辑元素数给（而不是字节数） | 跑通，不报错，**只搬了前一半** |

两个都钉了测试。共同点是**第二道检查被跳过时，第一道和第三道都能被形状「蒙对」**。

## 五、`pl.reinterpret`：8 位的退路，4 位没有

`DT_BOOL` 进不了搬运通道，但它是 8 位：声明 `DT_UINT8` 搬进来，片上
`pl.reinterpret(t, dtype=pl.DT_BOOL, shape=[1, N])` 换视图。**换 dtype 时必须同时给 `shape=`**
（`block_ops.py:3835`），即使同位宽、形状不变。

同样的路子对 `DT_INT4` **编译失败**。所以那 5 个里只有 `DT_BOOL` 有退路。

**`INT4` / `UINT4` / `FP4` / `HF4` 在 A5 上没有路径，因为硬件没有这些 4 位元素类型。**
三处相互独立地印证：搬运表里 4 位只有 `FP4E2M1` / `FP4E1M2`；`DT_INT4` 在整个
`pypto_pro.language` 面（`_api.py` / `_vf_api.py` / `_simt_api.py`）一次都没出现；
ascriptor（同一块硬件的上一代编译器）的 a5 dtype 表也只有 `fp4_e2m1` / `fp4_e1m2`，它的
`int4` 只作为塞在 int32 里的位负载存在，配套的是 `pack_int4_to_int32` / `unpack_int32_to_int4`，
不是可搬运的元素类型。

所以这 4 个 dtype 在枚举里是**占位**，不是能用的类型。要在 A5 上用 4 位数据，用
`FP4E2M1` / `FP4E1M2`。

## 六、torch_npu 的一个限制（不是 pypto 的）

`uint32` / `uint64` / `float8_e8m0fnu` 三种 dtype **在卡上建不出张量**：
`zeros_like` 和 `zeros(device=)` 都报 `ZerosLikeKernelNpuOpApi.cpp:26 ... error code is 161002`。
办法是在 CPU 上建好再 `.to(device)`。`test_three_torch_dtypes_cannot_be_created_on_the_device`
钉住这条，因为它会让「照常写 `torch.zeros_like(a)`」在这三个 dtype 上莫名失败。

判据：这一例不比数值，比**字节**——`a.view(torch.uint8) == out.view(torch.uint8)` 全等，
`rtol/atol` 无意义。搬运是逐字节的，任何一位不同都是错。
"""

import numpy as np
import pytest
import torch

import pypto_pro.language as pl

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


def test_an_unmapped_dtype_takes_any_host_dtype_and_goes_silently_wrong(device):
    """`DT_FP8E4M3FN` 不在映射表里 → host dtype 不检查。

    传 `torch.float32`（4 字节/元素）给一个 8 位参数：形状 64 == 64 蒙对了第一道，
    第二道跳过，第三道 pack 因子是 1 也蒙对——于是跑通，**只搬了前 64 字节**。
    这是本例第一个静默坑，也是这一整例存在的理由。
    """
    kernel = make_static_copy(pl.DT_FP8E4M3FN)
    a = torch.arange(1, N + 1, dtype=torch.float32).reshape(1, N).to(device)
    out = torch.zeros((1, N), dtype=torch.float32).to(device)
    kernel(a, out)  # 不报错
    torch.npu.synchronize()

    assert not _same_bytes(a, out), (
        "位宽差 4 倍居然搬对了——那这个坑不存在了，本例这条结论要重测"
    )
    moved = out.cpu().view(torch.uint8).reshape(-1).numpy()
    assert (moved[:N] != 0).any(), "前 64 字节应该搬过去了"
    assert (moved[N:] == 0).all(), (
        f"64 字节之后应该没动，实际有 {(moved[N:] != 0).sum()} 个非零"
    )


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
