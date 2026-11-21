# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证连续 GM 与共享 UB 转置：双屏障、tile 尾部、多 worker 和不同 UB 行跨度。

独立参考逐位比较，包含正负零、subnormal；重复调用更换输入并毒化输出。
输入和输出 storage 的前后守卫均检查，padding 的性能差异用共享计时器另测。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


@pytest.mark.parametrize(
    "rows,cols", [(1, 1), (17, 33), (64, 96), (65, 97), (129, 257)]
)
@pytest.mark.parametrize("padding,workers,threads", [(0, 1, 128), (2, 3, 256)])
def test_transpose(device, rows, cols, padding, workers, threads):
    rng = np.random.default_rng(78101 + rows * cols)
    for repeat in range(2):
        bits = rng.integers(0, 0x7F000000, rows * cols, dtype=np.int32)
        bits[::3] |= np.int32(-2147483648)
        special = np.array([0, -2147483648, 1, -2147483647, 0x00800000], dtype=np.int32)
        bits[: min(bits.size, special.size)] = special[: min(bits.size, special.size)]
        values = bits.view(np.float32).reshape(rows, cols)
        source_raw = torch.full(
            (rows * cols + 128,), -777, dtype=torch.float32, device=device
        )
        x = source_raw[64:-64].reshape(rows, cols)
        x.copy_(torch.from_numpy(values).to(device))
        out_raw = torch.full_like(source_raw, -777)
        out = out_raw[64:-64].reshape(cols, rows)
        out.fill_(123 + repeat)
        run(x, out, workers=workers, threads=threads, padding=padding)
        torch.npu.synchronize()
        np.testing.assert_array_equal(
            out.cpu().numpy().view(np.int32), reference(values).view(np.int32)
        )
        np.testing.assert_array_equal(
            x.cpu().numpy().view(np.int32), bits.reshape(rows, cols)
        )
        for raw in (source_raw, out_raw):
            assert torch.all(raw[:64].cpu() == -777)
            assert torch.all(raw[-64:].cpu() == -777)
