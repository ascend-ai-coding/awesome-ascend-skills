# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证稳定分组、稀疏/空路由、逆映射和尾部初始化。

连续 GM payload 由唯一 token owner 写入；两个原生任务的同 Stream 顺序保护计数依赖。
独立 CPU 稳定排序给出位置，位比较覆盖特殊 FP32 payload；偏移视图、毒化和守卫检查物理访问。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import stable_dispatch_reference


def data(rows, cols, experts, kind):
    rng = np.random.default_rng(rows * cols + experts * 97)
    x = rng.uniform(-100, 100, (rows, cols)).astype(np.float32)
    special = np.array(
        [0x80000000, 0, 1, 0x7F800000, 0xFF800000, 0x7FC12345], dtype=np.uint32
    )
    payload = x.view(np.uint32).reshape(-1)[::31]
    payload[:] = np.resize(special, payload.shape)
    ids = rng.integers(-1, experts, rows, dtype=np.int32)
    if kind == "invalid":
        ids[:] = -1
    elif kind == "single":
        ids[:] = experts - 1
    elif kind == "cycle":
        ids = (np.arange(rows) % experts).astype(np.int32)
    elif kind == "reverse":
        ids = (experts - 1 - np.arange(rows) % experts).astype(np.int32)
        ids[::7] = -1
    elif kind == "sparse":
        ids[:] = -1
        for i, position in enumerate(
            [0, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024, rows - 1]
        ):
            if position < rows:
                ids[position] = i % experts
    return x, ids


def guarded(value, device, head=64):
    dtype = torch.float32 if value.dtype == np.float32 else torch.int32
    backing = torch.full((value.size + head + 64,), 8192, dtype=dtype, device=device)
    tensor = backing[slice(head, head + value.size)].view(*value.shape)
    tensor.copy_(torch.from_numpy(value).to(device))
    return tensor, backing


@pytest.mark.parametrize(
    "rows,cols,experts",
    [
        (65, 33, 5),
        (257, 65, 17),
        (4097, 129, 8),
        (1, 1, 1),
        (17, 7, 32),
        (129, 257, 3),
        (8192, 17, 32),
    ],
)
@pytest.mark.parametrize(
    "kind", ["random", "invalid", "single", "cycle", "reverse", "sparse"]
)
@pytest.mark.parametrize("workers", [1, 3])
def test_stable_dispatch(device, rows, cols, experts, kind, workers):
    x, ids = data(rows, cols, experts, kind)
    expected = stable_dispatch_reference(x, ids, experts)
    inputs = [guarded(x, device), guarded(ids, device)]
    outputs = [
        guarded(np.zeros_like(value), device, head=8 if (workers + i) % 2 else 64)
        for i, value in enumerate(expected)
    ]
    for poison in [float("nan"), 12345.0]:
        for i, (tensor, _) in enumerate(outputs):
            tensor.fill_(poison if i == 0 else -777)
        run(
            *(tensor for tensor, _ in inputs),
            *(tensor for tensor, _ in outputs),
            experts,
            workers=workers,
        )
        torch.npu.synchronize()
        valid_count = int(expected[-1][-1])
        for i, ((tensor, _), value) in enumerate(zip(outputs, expected)):
            actual = tensor.cpu().numpy()
            if i == 0:
                np.testing.assert_array_equal(
                    actual[:valid_count].view(np.uint32),
                    value[:valid_count].view(np.uint32),
                )
                np.testing.assert_array_equal(actual[valid_count:], 0)
            else:
                np.testing.assert_array_equal(actual, value)
        for tensor, backing in inputs + outputs:
            np.testing.assert_array_equal(
                backing[: tensor.storage_offset()].cpu().numpy(), 8192
            )
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 8192)
        np.testing.assert_array_equal(
            inputs[0][0].cpu().numpy().view(np.uint32), x.view(np.uint32)
        )
        np.testing.assert_array_equal(inputs[1][0].cpu().numpy(), ids)
