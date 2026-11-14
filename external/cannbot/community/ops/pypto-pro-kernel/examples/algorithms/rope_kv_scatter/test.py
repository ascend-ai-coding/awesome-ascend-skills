# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""RoPE 与唯一槽位 KV 写入的独立数值、逐位传输和边界验收。

契约：连续FP16 Q[B,T,Hq,D]、K/V[B,T,Hkv,D]，值有限且在[-2,2]；FP32 cos/sin[P,R/2]，INT32 positions/slots[B,T]；输出Qout与
cacheK/cacheV[C,Hkv,D]。B=1..4，T=1..513，Hkv=1/2/4，Hq/Hkv=1/2/4，D=1..129，R为0..D内偶数。非负slot唯一且<C；slot=-1只跳过cache写入，Qout仍完整定
义。
独立性：reference.py用实际FP16输入与FP32表做FP64相邻对旋转，再舍入FP16，按数学槽位赋值；旋转结果固定atol=rtol=2e-3，未旋转部分、V和未写cache必须逐位一致。
物理边界：连续view带67元素偏移和前后守卫；覆盖R=0/部分/全部、奇数D、首末position/slot、P=C=1、全跳过和混合跳过、最大T/D。Q/K/V包括±2、正负零与最小FP16次正规数，检查全部输入字节不变；初始cache
另含NaN payload和正负无穷，验证未写区域逐位保留。
状态与同步：workers=1/4、两种cache初值/输出毒值；同一设备buffer改变positions和slots，检查已有cache内容逐位保留。合法非负slots唯一，不用冲突写入测试替代契约。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


CASES = [
    (1, 1, 1, 1, 1, 0, "skip"),
    (1, 1, 4, 4, 2, 2, "all"),
    (2, 3, 2, 2, 63, 62, "mixed"),
    (1, 65, 1, 4, 64, 0, "all"),
    (3, 17, 4, 4, 65, 32, "mixed"),
    (1, 127, 2, 1, 127, 126, "all"),
    (4, 128, 1, 4, 128, 128, "skip"),
    (4, 513, 4, 4, 129, 128, "mixed"),
    (4, 513, 4, 4, 129, 0, "all"),
]


def make_inputs(spec):
    batch, tokens, kv_heads, ratio, depth, rotary, mode = spec
    rng = np.random.default_rng(batch * 71 + tokens + depth + rotary)
    minimal = batch * tokens == 1 and mode == "all"
    position_count = 1 if minimal else 19
    cache_rows = 1 if minimal else batch * tokens + 7
    q = rng.uniform(-2, 2, (batch, tokens, kv_heads * ratio, depth)).astype(np.float16)
    k = rng.uniform(-2, 2, (batch, tokens, kv_heads, depth)).astype(np.float16)
    v = rng.uniform(-2, 2, k.shape).astype(np.float16)
    for value in (q, k, v):
        value.flat[::37] = 2
        value.flat[::41] = -2
        value.flat[::29] = np.float16(-0.0)
        value.flat[::31] = np.nextafter(np.float16(0), np.float16(1))
    angles = rng.uniform(-np.pi, np.pi, (position_count, rotary // 2))
    angles.flat[::17] = 0
    angles.flat[::19] = np.pi / 2
    angles.flat[::23] = np.pi
    cos, sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
    positions = rng.integers(0, position_count, (batch, tokens), dtype=np.int32)
    positions.flat[0] = 0
    if positions.size > 1:
        positions.flat[-1] = position_count - 1
    slots = (
        rng.permutation(cache_rows)[: batch * tokens]
        .astype(np.int32)
        .reshape(batch, tokens)
    )
    if mode == "skip":
        slots.fill(-1)
    elif mode == "mixed":
        slots.flat[::3] = -1
    else:
        flat = slots.ravel()
        flat[0] = 0
        if flat.size > 1:
            flat[-1] = cache_rows - 1
            flat[1:-1] = rng.permutation(np.arange(1, cache_rows - 1))[: flat.size - 2]
    caches = [
        rng.uniform(-2, 2, (cache_rows, kv_heads, depth)).astype(np.float16)
        for _ in range(2)
    ]
    for initial in caches:
        bits = initial.view(np.uint16)
        bits.flat[::13] = 0x8000
        bits.flat[::17] = 0x0001
        bits.flat[::23] = 0x7E35
        bits.flat[::29] = 0x7C00
        bits.flat[::31] = 0xFC00
    return [q, k, v, cos, sin, positions, slots], caches


def guarded(array, device):
    host = torch.from_numpy(array)
    backing = torch.full((host.numel() + 138,), -99, dtype=host.dtype, device=device)
    value = backing[slice(67, 67 + host.numel())].view(host.shape)
    value.copy_(host)
    return value, backing


def verify(originals, inputs, outputs, cache_bases, workers):
    expected = reference(*originals, *cache_bases)
    run(
        *(value for value, _ in inputs),
        *(value for value, _ in outputs),
        workers=workers,
    )
    torch.npu.synchronize()
    actual = [value.cpu().numpy() for value, _ in outputs]
    for got, wanted in zip(actual[:2], expected[:2]):
        np.testing.assert_allclose(got, wanted, atol=2e-3, rtol=2e-3)
    rotary = originals[3].shape[1] * 2
    for got, wanted in zip(actual[:2], expected[:2]):
        np.testing.assert_array_equal(
            got[..., rotary:].view(np.uint16), wanted[..., rotary:].view(np.uint16)
        )
    np.testing.assert_array_equal(
        actual[2].view(np.uint16), expected[2].view(np.uint16)
    )
    written = np.zeros(cache_bases[0].shape[0], bool)
    written[originals[-1][originals[-1] >= 0]] = True
    np.testing.assert_array_equal(
        actual[1][~written].view(np.uint16), cache_bases[0][~written].view(np.uint16)
    )
    for (value, _), original in zip(inputs, originals):
        np.testing.assert_array_equal(
            value.cpu().numpy().view(np.uint8), original.view(np.uint8)
        )
    for value, backing in inputs + outputs:
        host = backing.cpu().numpy()
        assert np.all(host[:67] == -99) and np.all(
            host[slice(67 + value.numel(), None)] == -99
        )
    return expected[1:]


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("workers", [1, 4])
def test_rope_kv_scatter(device, spec, workers):
    originals, caches = make_inputs(spec)
    inputs = [guarded(value, device) for value in originals]
    outputs = [
        guarded(np.zeros_like(originals[0]), device),
        *[guarded(value, device) for value in caches],
    ]
    for repeat in range(2):
        bases = [value if repeat == 0 else -value for value in caches]
        outputs[0][0].fill_(123 if repeat == 0 else -77)
        for (value, _), base in zip(outputs[1:], bases):
            value.copy_(torch.from_numpy(base))
        verify(originals, inputs, outputs, bases, workers)


@pytest.mark.parametrize("workers", [1, 4])
def test_runtime_positions_and_slots_with_fixed_storage(device, workers):
    originals, bases = make_inputs((1, 7, 2, 4, 65, 64, "all"))
    inputs = [guarded(value, device) for value in originals]
    outputs = [
        guarded(np.zeros_like(originals[0]), device),
        *[guarded(value, device) for value in bases],
    ]
    first_slots = originals[-1].copy()
    for step in range(4):
        originals[-2] = (originals[-2] + 1) % originals[3].shape[0]
        originals[-1] = (
            first_slots.copy() if step % 2 else np.full_like(first_slots, -1)
        )
        if step == 3:
            originals[-1] = np.flip(first_slots, axis=1).copy()
            originals[-1][0, ::2] = -1
        for (value, _), original in zip(inputs[-2:], originals[-2:]):
            value.copy_(torch.from_numpy(original))
        outputs[0][0].fill_(123 if step % 2 else -77)
        bases = verify(originals, inputs, outputs, bases, workers)
