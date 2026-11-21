# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""``dist=`` 里**位宽绑定**的那一半——展开、块广播、打包（Tile↔寄存器层）。

**写法：原始。** ``vf.*`` 没有语法糖。前置样例是 [`vf_align`](../vf_align/test.py)
（同一 ``dist=`` 参数在单一位宽 FP32 下的分布模式）。**同一批词、另一个层**：
``pack`` / ``interleave`` 作为**寄存器到寄存器算子**（``vf.pack`` / ``vf.astype``
本体等十个）在 [`vf_lane_layout`](../vf_lane_layout/test.py)；本例是它们在
``load_align`` / ``store_align`` 的 ``dist=`` 参数上的**搬运端口版本**。

## 一、为什么这些模式值得单独一例

``vf_align`` 里的分布（NORM/BRC/DINTLV/INTLV/…）在任何位宽下语义一致，位宽后缀只是
粒度；这里的四个机制**语义本身绑定「源 dtype × dist 粒度」的组合**，用对了是零成本
的跨位宽重排，用错了是三种不同形态的静默垃圾（子采样 / 块广播当扩展 / 位截断）。
机制的完整证据链在 references/api/vf.md 二点十三 / 二点十四；跨 dtype 的实战落地是
examples/algorithms/rmsnorm_gemm（FP16 输入的全 FP32 RMSNorm）。

## 二、实测语义（2026-09-29，A5，bit-exact）

| 机制 | 写法 | 结果 |
| --- | --- | --- |
| **保序宽化** | `load_align(src16, off, dist=UNPK)` + `astype(FP32, layout=ZERO)` | 64FP16→64FP32，**保源序**，等价「FP32源一次load」 |
| 块广播 | `load_align(src16, 0, dist=E2B)` | 每个源元素复制填满一个 32B block（FP16 下 16 lane）——**不是**文档说的「16→32 扩展」 |
| **窄化直出** | `astype(FP16, layout=ZERO)` + `store_align(dist=PACK_B32)` | 每 b32 容器低 16 bit 抽出连续写 → 稠密 FP16 |
| 位截断（陷阱） | `store_align(dist=PACK)`（粗名，FP16 寄存器） | 自动选 PK_B16 = 取每 b16 元素**低 8 bit** 打包——垃圾，不报错 |

## 三、三个坑，三种静默形态

### 3.1 宽化 ``astype`` 不带 UNPK 是源子采样

FP16 tile 上普通 ``load_align`` 一次取 **128** 个原生元素（寄存器宽度 = 256B ÷
dtype 字节数，`VL=64` 是 FP32 专属数字）；直接 ``astype(dtype=FP32)`` 拿到的是
**偶数下标**的 64 个（``layout=ONE`` 给奇数下标）——``ZERO/ONE`` 的本义是「b32
容器的低/高 16 bit 位置」，紧凑寄存器上恰好表现为偶/奇子采样。症状：输出 ≈ 源的
偶数列，不报错。保序入口只有 UNPK（本例验证 1）。

### 3.2 E2B 的官方描述是错的

``LoadDist.md`` 写「16位宽类型->32位宽类型扩展」；实测是 element-to-block 广播，
且与 ``vf_align`` 第六节 FP32 下的既有观察（每元素重复 8 次 = 32B/4B；当时被解读
为「不是它的语义」）互证——**块广播就是它的语义**，任何位宽下一致（本例 FP16 下
重复 16 次 = 32B/2B）。应以目标版本的官方文档和实测结果核对该行为。

### 3.3 PACK 粗名自动粒度会选错语义

粗名按**源寄存器 dtype** 自动选粒度：FP16 寄存器上 ``PACK`` → PK_B16 =「取每个
b16 元素的低 8 bit」——这是给整数打包用的**位截断**，对 FP16 数据输出垃圾
（1.0=0x3C00 的低字节是 0）。「窄化直出」要的是「取每 b32 容器的低 16 bit」=
**显式 ``PACK_B32``**。官方功能说明「低半部分 bit」按粒度字面成立，坑在粗名的
自动粒度落点。验证 4 把 PK_B16 的位截断行为钉成测试。

## 四、什么时候用哪个

- FP16 源要做 FP32 计算（归约、精度敏感的逐元素链）→ **UNPK + astype ZERO**
  （rmsnorm_gemm 的通路）；顺序无关的纯归约也可以用「偶/奇子采样各算再合并」省
  一次展开（vf.md 二点十三）。
- FP32 结果转 FP16 写 **ND** 目标 → **astype ZERO + PACK_B32**（单 chunk 一条
  store）；喂 vsstb 游标散写（要稠密 128 宽寄存器作源）→ 仍走两 chunk
  ``de_interleave`` 拼接（vf.md 二点十，mm_softmax_mm/rmsnorm_gemm 的写出段）。
- E2B 的正当用途是「每 block 一个常数」的图样构造，不是类型转换。

## 五、如何独立验证

1. UNPK 宽化：128 个已知 FP16 → FP32 输出逐位等于源值、顺序不变。
2. E2B：输出前 128 元素 = 前 8 个源元素各铺满 16 lane。
3. PACK_B32：128 个 FP16 可精确表示的 FP32 → FP16 输出逐位等于 ``fp16(源)``。
4. PACK 粗名：uint16 视角，前 64 槽 = 128 个源元素的低字节两两打包（小端），
   后 64 槽保留哨兵（不被写）——钉住位截断行为，行为一变（改报错/改语义）
   测试立刻失败。

## 六、这一例不讲什么

- 单一位宽下的分布模式（NORM/BRC/BLK/DINTLV/INTLV/DataBlock/位宽缺失陷阱）在
  ``vf_align``。
- **寄存器到寄存器**的排布/位宽/转换算子（``vf.interleave`` / ``vf.pack`` /
  ``vf.astype`` 本体、取整模式、``bit_cast`` 等十个）在 ``vf_lane_layout``——
  那一例的 ``pack`` 是「两个窄元素挤进一个宽槽位」的寄存器算子，与本例
  ``StoreDist.PACK``（store 端口打包）是同名的两层机制。
- `PACK4_B32` 的直接 INT32→byte 写出由本文件的部分 mask、跨寄存器与越界值测试覆盖；mask_reg 的 NORM/PACK 存取仍不在本例中。
"""

import numpy as np
import pytest
import torch

from . import reference as ref
from .kernel import (
    N,
    alignw_load_e2b,
    alignw_narrow_pack_b32,
    alignw_pack_coarse_trap,
    alignw_widen_unpk,
)

SENTINEL_U16 = 0x7FFF  # FP16 NaN 域的一个固定位型，当哨兵一眼可辨


@pytest.mark.parametrize("active", [1, 3, 31, 32, 33, 63, 64, 65, 127, 128])
@pytest.mark.parametrize("byte_base", [0, 128])
def test_u8_float_chain_preserves_order_and_partial_masks(device, active, byte_base):
    from .kernel import make_u8_float_chain

    values = (np.arange(N, dtype=np.uint16) + byte_base).astype(np.uint8).reshape(1, N)
    x = torch.from_numpy(values).to(device)
    buffers = [
        torch.full((N + 128,), -8192, dtype=dtype, device=device)
        for dtype in (torch.float32, torch.float16)
    ]
    outputs = [backing[64:-64].view(1, N) for backing in buffers]
    kernel = make_u8_float_chain(active)
    for _ in range(2):
        for out in outputs:
            out.fill_(-8192)
        kernel[None, 1](x, *outputs)
        torch.npu.synchronize()
        for out, backing in zip(outputs, buffers):
            actual = out.cpu().numpy()
            expected = np.full((1, N), -8192, dtype=actual.dtype)
            expected[:, :active] = values[:, :active]
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(backing[:64].cpu().numpy(), -8192)
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), -8192)
    np.testing.assert_array_equal(x.cpu().numpy(), values)


@pytest.mark.parametrize(
    "destination_offset,active",
    [(0, n) for n in [1, 3, 31, 63, 64, 65, 127, 128]]
    + [(32, n) for n in [1, 31, 63, 64, 65, 96]],
)
@pytest.mark.parametrize("signed", [True, False])
def test_pack4_b32_extracts_low_bytes_and_preserves_unwritten_tail(
    device, active, signed, destination_offset
):
    from .kernel import make_pack4_b32

    values = np.resize(
        np.array([-129, -128, -1, 0, 1, 127, 128, 255, 256, 511], dtype=np.int32),
        (1, N),
    )
    x = torch.from_numpy(values).to(device)
    dtype = torch.int8 if signed else torch.uint8
    sentinel = -73 if signed else 173
    backing = torch.full((N + 128,), sentinel, dtype=dtype, device=device)
    out = backing[64:-64].view(1, N)
    expected = np.full((1, N), sentinel, dtype=np.int8 if signed else np.uint8)
    expected[:, slice(destination_offset, destination_offset + active)] = ref.low_bytes(
        values, signed
    )[:, :active]
    kernel = make_pack4_b32(active, signed, destination_offset)
    for _ in range(2):
        out.fill_(sentinel)
        kernel[None, 1](x, out)
        torch.npu.synchronize()
        np.testing.assert_array_equal(out.cpu().numpy(), expected)
        np.testing.assert_array_equal(backing[:64].cpu().numpy(), sentinel)
        np.testing.assert_array_equal(backing[-64:].cpu().numpy(), sentinel)
    np.testing.assert_array_equal(x.cpu().numpy(), values)


@pytest.mark.parametrize("logical_offset", [1, 7, 31, 47])
@pytest.mark.parametrize("active", [1, 63])
@pytest.mark.parametrize("signed", [True, False])
def test_aligned_pack4_fragment_reaches_unaligned_logical_gm_offsets(
    device, logical_offset, active, signed
):
    from .kernel import make_pack4_fragment

    values = np.resize(
        np.array([-129, -128, -1, 0, 1, 127, 128, 255, 256, 511], dtype=np.int32),
        (1, 64),
    )
    x = torch.from_numpy(values).to(device)
    dtype = torch.int8 if signed else torch.uint8
    sentinel = -73 if signed else 173
    backing = torch.full((N + 128,), sentinel, dtype=dtype, device=device)
    out = backing[64:-64].view(1, N)
    expected = np.full((1, N), sentinel, dtype=np.int8 if signed else np.uint8)
    expected[:, slice(logical_offset, logical_offset + active)] = ref.low_bytes(
        values, signed
    )[:, :active]
    kernel = make_pack4_fragment(active, signed, logical_offset)
    for _ in range(2):
        out.fill_(sentinel)
        kernel[None, 1](x, out)
        torch.npu.synchronize()
        np.testing.assert_array_equal(out.cpu().numpy(), expected)
        np.testing.assert_array_equal(backing[:64].cpu().numpy(), sentinel)
        np.testing.assert_array_equal(backing[-64:].cpu().numpy(), sentinel)
    np.testing.assert_array_equal(x.cpu().numpy(), values)


def test_unpk_widening_preserves_source_order(device):
    """验证 1。UNPK + astype(FP32, ZERO) 逐位保序——FP16 源做 FP32 计算的入口。"""
    x = ref.make_fp16_input()
    out = torch.zeros([1, N], device=device, dtype=torch.float32)
    alignw_widen_unpk(torch.from_numpy(x).to(device), out)
    torch.npu.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy(), ref.widen_unpk(x))


def test_e2b_is_element_to_block_broadcast(device):
    """验证 2。E2B = 每个源元素铺满一个 32B block，不是「16→32 扩展」。"""
    x = ref.make_fp16_input()
    out = torch.zeros([1, N], device=device, dtype=torch.float16)
    alignw_load_e2b(torch.from_numpy(x).to(device), out)
    torch.npu.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy(), ref.load_e2b(x))


def test_pack_b32_extracts_dense_fp16_after_narrowing_astype(device):
    """验证 3。astype(FP16, ZERO) + PACK_B32 store = 窄化直出稠密。"""
    x = ref.make_fp32_input()
    out = torch.zeros([1, N], device=device, dtype=torch.float16)
    alignw_narrow_pack_b32(torch.from_numpy(x).to(device), out)
    torch.npu.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy(), ref.narrow_pack_b32(x))


def test_pack_coarse_name_on_fp16_is_bit_truncation(device):
    """验证 4。钉住陷阱：粗名 PACK on FP16 寄存器 = PK_B16 = 低 8 bit 位截断。

    输出按 uint16 位型比（值域是位模式不是数值）：前 64 槽 = 源低字节两两打包，
    后 64 槽保留哨兵（store 只写 128 字节）。这条测试不是说「位截断是对的用法」，
    是说「现在是位截断」——行为一旦变化（报错或改语义），它会立刻失败。
    """
    x = ref.make_fp16_input()
    out = torch.full([1, N], 0, device=device, dtype=torch.int16)
    out.fill_(SENTINEL_U16)
    out16 = out.view(torch.float16)
    alignw_pack_coarse_trap(torch.from_numpy(x).to(device), out16)
    torch.npu.synchronize()

    got = out16.cpu().numpy().view(np.uint16)
    want = ref.pack_coarse_trap_u16(x)
    np.testing.assert_array_equal(got[0, :64], want[0, :64], err_msg="低字节打包")
    np.testing.assert_array_equal(
        got[0, 64:],
        np.full(64, SENTINEL_U16, dtype=np.uint16),
        err_msg="后 64 槽应保留哨兵（store 只写 128 字节）",
    )
