# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证完整直方图：两阶段初始化顺序、局部屏障、尾部和空输入。

独立 bincount 判据为逐元素精确相等；每次更换输入并毒化旧输出，检查输入和两侧守卫。
直接 GM、线程内合并与 UB 聚合都必须通过。性能使用共享计时器另测，不作为默认 pytest 的阈值。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


@pytest.mark.parametrize(
    "n,kind",
    [
        (0, "random"),
        (1, "hot"),
        (257, "random"),
        (4099, "edge"),
        (65537, "hot"),
        (131101, "skewed"),
        (8197, "runs"),
    ],
)
@pytest.mark.parametrize(
    "local,combine_runs,workers,threads",
    [(False, False, 1, 128), (False, True, 3, 128), (True, False, 3, 256)],
)
def test_histogram(device, n, kind, local, combine_runs, workers, threads):
    rng = np.random.default_rng(74081 + n)
    raw = torch.full((384,), -123456789, dtype=torch.int32, device=device)
    out = raw[64:-64]
    for repeat in range(2):
        values = rng.integers(0, 256, n, dtype=np.uint8)
        if kind == "hot":
            values[:] = 0 if repeat == 0 else 255
        elif kind == "edge":
            values[:] = np.where(np.arange(n) % 2, 0, 255)
        elif kind == "skewed":
            values[np.arange(n) % 8 != 0] = 127 + repeat
        elif kind == "runs":
            values[:] = (np.arange(n) // (3 * workers * threads) + repeat) % 256
        x = torch.from_numpy(values).to(device)
        out.fill_(10000 + repeat)
        run(
            x,
            out,
            workers=workers,
            threads=threads,
            local=local,
            combine_runs=combine_runs,
        )
        torch.npu.synchronize()
        np.testing.assert_array_equal(out.cpu().numpy(), reference(values))
        np.testing.assert_array_equal(x.cpu().numpy(), values)
        assert torch.all(raw[:64].cpu() == -123456789)
        assert torch.all(raw[-64:].cpu() == -123456789)
