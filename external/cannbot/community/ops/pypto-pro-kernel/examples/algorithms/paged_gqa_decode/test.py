# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Paged GQA 的独立页寻址、数值范围、运行期元数据与物理边界验收。

契约：FP16 Q[B,Hq,D]、K[P,S,Hkv,D]、V[P,S,Hkv,Dv]有限且在[-2,2]，INT32页表[B,L]/长度[B]；连续FP32 O[B,Hq,Dv]/LSE[B,Hq]。B=1..16，Hkv=1/2/4，
Hq/Hkv=1/2/4/8，S=16/32/64，D/Dv=64/128，L=1..256，P=1..4096，长度0..S*L。
独立性：reference.py按原始页表展开FP64数学attention，不采用候选的分片、公共参考、累加顺序或中间舍入；O/LSE固定atol=rtol=2e-3，空序列精确0/-inf。
边界：每项原始布局以非零存储偏移67个元素的连续view传入；检查前后守卫、输入字节不变、两种输出毒值、workers=1/4、页乱序/重复/共享/高页号、尾页和最大范围。
数值与状态：±2端点、恒定最高/最低分数、首末页峰值及抵消；同一设备buffer切换满/空/混合长度和页表，防止陈旧状态及固化运行期内容。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


CASES = [
    (1, 1, 1, 16, 64, 64, 1, 1, "zero"),
    (1, 1, 8, 16, 128, 64, 1, 1, "full"),
    (7, 2, 2, 16, 64, 128, 3, 7, "mixed"),
    (7, 4, 4, 32, 128, 64, 3, 1, "shared"),
    (7, 1, 8, 64, 64, 128, 3, 7, "mixed"),
    (1, 1, 1, 16, 64, 64, 2, 4096, "full"),
    (8, 2, 4, 32, 64, 64, 256, 31, "full"),
    (16, 4, 8, 64, 128, 128, 256, 31, "shared"),
]


def metadata(batch, pages, page_size, physical_pages, mode, rng):
    if mode == "zero":
        lengths = np.zeros(batch, np.int32)
    elif mode == "full":
        lengths = np.full(batch, pages * page_size, np.int32)
    else:
        choices = [
            0,
            1,
            page_size - 1,
            page_size,
            page_size + 1,
            pages * page_size - 1,
            pages * page_size,
        ]
        lengths = np.array(
            [min(pages * page_size, choices[b % 7]) for b in range(batch)], np.int32
        )
    table = np.full((batch, pages), -1, np.int32)
    for row, length in enumerate(lengths):
        count = (int(length) + page_size - 1) // page_size
        table[row, :count] = rng.integers(0, physical_pages, count, dtype=np.int32)
        if mode == "shared":
            table[row, :count] = physical_pages - 1
    if physical_pages == 4096:
        table[:, 0] = 4095
    return table, lengths


def make_case(spec):
    (
        batch,
        kv_heads,
        ratio,
        page_size,
        depth,
        value_depth,
        pages,
        physical_pages,
        mode,
    ) = spec
    rng = np.random.default_rng(batch * 71 + depth + value_depth + page_size)
    return [
        rng.uniform(-2, 2, (batch, kv_heads * ratio, depth)).astype(np.float16),
        rng.uniform(-2, 2, (physical_pages, page_size, kv_heads, depth)).astype(
            np.float16
        ),
        rng.uniform(-2, 2, (physical_pages, page_size, kv_heads, value_depth)).astype(
            np.float16
        ),
        *metadata(batch, pages, page_size, physical_pages, mode, rng),
    ]


def guarded(array, device):
    value = torch.from_numpy(array)
    backing = torch.full((value.numel() + 138,), -99, dtype=value.dtype, device=device)
    view = backing[slice(67, 67 + value.numel())].view(value.shape)
    view.copy_(value)
    return view, backing


def verify(originals, inputs, outputs, workers):
    expected = reference(*originals)
    run(
        *(value for value, _ in inputs),
        *(value for value, _ in outputs),
        workers=workers,
    )
    torch.npu.synchronize()
    actual = [value.cpu().numpy() for value, _ in outputs]
    for value, wanted in zip(actual, expected):
        np.testing.assert_allclose(value, wanted, atol=2e-3, rtol=2e-3)
    empty = originals[-1] == 0
    assert np.all(actual[0][empty] == 0)
    assert np.all(np.isneginf(actual[1][empty]))
    for value, backing in inputs + outputs:
        host = backing.cpu().numpy()
        assert np.all(host[:67] == -99) and np.all(
            host[slice(67 + value.numel(), None)] == -99
        )
    for (value, _), original in zip(inputs, originals):
        np.testing.assert_array_equal(
            value.cpu().numpy().view(np.uint8), original.view(np.uint8)
        )


def buffers(originals, device):
    batch, heads = originals[0].shape[:2]
    shapes = [(batch, heads, originals[2].shape[-1]), (batch, heads)]
    return (
        [guarded(value, device) for value in originals],
        [guarded(np.zeros(shape, np.float32), device) for shape in shapes],
    )


def check_case(originals, device, workers):
    inputs, outputs = buffers(originals, device)
    for poison in (123, -77):
        for value, _ in outputs:
            value.fill_(poison)
        verify(originals, inputs, outputs, workers)


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("workers", [1, 4])
def test_paged_gqa(device, spec, workers):
    check_case(make_case(spec), device, workers)


@pytest.mark.parametrize("depth", [64, 128])
@pytest.mark.parametrize("kind", ["high", "low", "first", "last", "cancel"])
@pytest.mark.parametrize("workers", [1, 4])
def test_score_range_and_cancellation(device, depth, kind, workers):
    pages = 256 if kind in ("high", "low", "cancel") else 2
    originals = make_case((2, 2, 2, 64, depth, 128, pages, 2, "full"))
    q, k, v, table, lengths = originals
    q.fill(2)
    k.fill(-2 if kind in ("low", "first", "last") else 2)
    v[:, :, 0].fill(0.5)
    v[:, :, 1].fill(-1.5)
    table[:] = np.arange(pages) % 2
    if kind in ("first", "last"):
        peak = 0 if kind == "first" else 1
        k[peak].fill(2)
        v[peak] *= -1
        q[:, 1::2] *= -1
    if kind == "cancel":
        v[:, 0::2].fill(2)
        v[:, 1::2].fill(-2)
        v[1, -1, :, 0] = 2
    check_case(originals, device, workers)


@pytest.mark.parametrize("value_depth", [64, 128])
@pytest.mark.parametrize("workers", [1, 4])
def test_runtime_metadata_changes_with_fixed_storage(device, value_depth, workers):
    originals = make_case((7, 2, 2, 32, 128, value_depth, 3, 7, "full"))
    inputs, outputs = buffers(originals, device)
    rng = np.random.default_rng(31239)
    for step, mode in enumerate(("full", "zero", "mixed", "shared", "full")):
        originals[-2:] = metadata(7, 3, 32, 7, mode, rng)
        for (value, _), original in zip(inputs[-2:], originals[-2:]):
            value.copy_(torch.from_numpy(original))
        for value, _ in outputs:
            value.fill_(123 if step % 2 else -77)
        verify(originals, inputs, outputs, workers)
