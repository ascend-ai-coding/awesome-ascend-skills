# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立 FP64 参考验证因果边界、归一化梯度、流式尾部与输出所有权。

恒定 payload/key 检查非零内部项的梯度抵消；集中权重及中间位置的上游梯度检查有效域。
所有输入与输出使用连续偏移视图，重复毒值初始化并逐位检查输入和前后守卫。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


def guarded(value, device):
    dtype = torch.float16 if value.dtype == np.float16 else torch.float32
    backing = torch.full((value.size + 81,), 128, dtype=dtype, device=device)
    tensor = backing[slice(17, 17 + value.size)].reshape(value.shape)
    tensor.copy_(torch.from_numpy(value))
    return tensor, backing


CASES = [
    (b, h, s, d, kind, w)
    for b, h, s, d in ((1, 1, 1, 32), (1, 3, 7, 64), (2, 1, 33, 128), (1, 2, 65, 32))
    for kind, w in (("random", 1), ("uniform", 4), ("zero_grad", 16))
]
CASES += [
    (b, h, s, d, kind, w)
    for b, h, s, d in ((1, 1, 257, 64), (2, 4, 513, 32), (1, 2, 1024, 128))
    for kind, w in (("random", 16), ("structured", 4))
]
CASES += [
    (1, 1, 257, 128, "constant_payload", 16),
    (1, 1, 1024, 128, "constant_payload", 4),
    (1, 2, 129, 64, "constant_keys", 16),
    (2, 4, 1024, 128, "concentrated", 16),
    (1, 2, 255, 32, "impulse", 1),
    (1, 1, 1023, 128, "random", 4),
]
CASES += [(2, 4, 1024, 128, "random", w) for w in (1, 4, 16)]


@pytest.mark.parametrize("b,h,s,d,kind,workers", CASES)
def test_attention_backward(device, b, h, s, d, kind, workers):
    rng = np.random.default_rng(7513 + b * 11 + h * 19 + s * 31 + d)
    host = [rng.uniform(-1, 1, (b, h, s, d)).astype(np.float16) for _ in range(4)]
    if kind == "uniform":
        host[0].fill(0)
    elif kind == "zero_grad":
        host[3].fill(0)
    elif kind == "structured":
        host[0][:] = np.resize(np.array([-1, 1, 0, 0], np.float16), host[0].shape)
        host[1][:] = np.resize(np.array([1, -1, -1, 1, 0], np.float16), host[1].shape)
    elif kind == "constant_payload":
        host[1].fill(1)
        host[2].fill(1)
        host[3].fill(1)
    elif kind == "constant_keys":
        host[1][:] = host[1][..., :1, :]
    elif kind == "concentrated":
        host[0].fill(1)
        host[1].fill(-1)
        host[1][..., ::7, :] = 1
    elif kind == "impulse":
        host[3].fill(0)
        host[3][..., s // 2, :] = 1
    expected = reference(*host)
    inputs = [guarded(a, device) for a in host]
    outputs = [guarded(np.empty((b, h, s, d), np.float32), device) for _ in range(3)]
    for poison in (float("nan"), 13):
        for tensor, _ in outputs:
            tensor.fill_(poison)
        run(*(a for a, _ in inputs + outputs), workers=workers)
        torch.npu.synchronize()
        for (tensor, _), wanted in zip(outputs, expected):
            np.testing.assert_allclose(
                tensor.cpu().numpy(), wanted, rtol=3e-3, atol=3e-3
            )
        for (tensor, _), original in zip(inputs, host):
            np.testing.assert_array_equal(
                tensor.cpu().numpy().view(np.uint16), original.view(np.uint16)
            )
        for tensor, backing in inputs + outputs:
            assert bool(torch.all(backing[:17].cpu() == 128))
            assert bool(
                torch.all(backing[slice(17 + tensor.numel(), None)].cpu() == 128)
            )
