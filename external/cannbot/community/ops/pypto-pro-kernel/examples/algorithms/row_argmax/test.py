# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""四问：逐行极值/首索引；二维 UB 与一维紧密 GM；单槽 auto_mutex 加 DMA；数学参考、毒化输出和守卫。"""

from functools import lru_cache

import numpy as np
import pytest
import torch

from .kernel import make_row_argmax
from .reference import row_argmax_reference

_make = lru_cache(maxsize=None)(make_row_argmax)


@pytest.mark.parametrize("workers", [1, 4, 8])
@pytest.mark.parametrize(
    "rows,cols", [(m, n) for m in (1, 17, 65) for n in (63, 130, 257)]
)
def test_exact_values_indices_and_output_guards(device, rows, cols, workers):
    kernel = _make(cols, workers)
    for kind in range(4):
        rng = np.random.default_rng(8521 + rows * cols + kind)
        x = rng.integers(-100, 101, (rows, cols)).astype(np.float32)
        if kind == 1:
            x = -rng.uniform(1, 100, (rows, cols)).astype(np.float32)
        elif kind == 2:
            x.fill(-7.25)
        elif kind == 3:
            x.fill(-99)
            for row in range(rows):
                first = (row * 37 + cols // 2) % cols
                second = (first + 65) % cols
                x[row, [first, second]] = 98
        expected_values, expected_indices = row_argmax_reference(x)
        tx = torch.from_numpy(x).to(device)
        values_backing = torch.full(
            (rows + 2,), 777.0, device=device, dtype=torch.float32
        )
        indices_backing = torch.full(
            (rows + 2,), 12345, device=device, dtype=torch.int32
        )
        values = values_backing[1:-1]
        indices = indices_backing[1:-1]
        values.fill_(float("nan"))
        indices.fill_(-1)
        kernel[None, workers](tx, values, indices)
        torch.npu.synchronize()
        np.testing.assert_array_equal(values.cpu().numpy(), expected_values)
        np.testing.assert_array_equal(indices.cpu().numpy(), expected_indices)
        np.testing.assert_array_equal(
            values_backing[[0, rows + 1]].cpu().numpy(), 777.0
        )
        np.testing.assert_array_equal(
            indices_backing[[0, rows + 1]].cpu().numpy(), 12345
        )
        np.testing.assert_array_equal(tx.cpu().numpy(), x)
