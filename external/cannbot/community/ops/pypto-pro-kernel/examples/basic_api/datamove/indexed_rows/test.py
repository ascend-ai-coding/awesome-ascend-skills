# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bit-preserving MTE rows: special values, tails, guards and same-pointer updates."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import reference


@pytest.mark.parametrize(
    "count,rows,columns", [(3, 7, 1), (19, 13, 17), (67, 23, 4096)]
)
@pytest.mark.parametrize("workers", [1, 8])
def test_indexed_rows(device, count, rows, columns, workers):
    rng = np.random.default_rng(40817 + columns)
    source_raw = torch.full(
        (rows * columns + 128,), -777, dtype=torch.float32, device=device
    )
    index_raw = torch.full((count + 128,), -777, dtype=torch.int32, device=device)
    output_raw = torch.full(
        (count * columns + 128,), -777, dtype=torch.float32, device=device
    )
    table = source_raw[64:-64].reshape(rows, columns)
    indices = index_raw[64:-64]
    out = output_raw[64:-64].reshape(count, columns)
    for repeat in range(2):
        bits = rng.integers(0, 0x7F000000, (rows, columns), dtype=np.int32)
        bits.flat[::3] |= np.int32(-2147483648)
        special = np.array(
            [
                0x00000000,
                0x80000000,
                0x00000001,
                0x80000001,
                0x7F800000,
                0xFF800000,
                0x7FC12345,
                0xFFC54321,
                0x7FA12345,
                0x7F7FFFFF,
                0xFF7FFFFF,
            ],
            dtype=np.uint32,
        ).view(np.int32)
        special[6] ^= np.int32(repeat)
        bits.flat[: min(bits.size, special.size)] = special[
            : min(bits.size, special.size)
        ]
        selected = rng.integers(0, rows, count, dtype=np.int32)
        selected[:3] = [0, rows - 1, 0]
        values = bits.view(np.float32)
        table.copy_(torch.from_numpy(values).to(device))
        indices.copy_(torch.from_numpy(selected).to(device))
        out.fill_(123 + repeat)
        run(table, indices, out, workers=workers)
        torch.npu.synchronize()
        np.testing.assert_array_equal(
            out.cpu().numpy().view(np.int32), reference(values, selected).view(np.int32)
        )
        np.testing.assert_array_equal(table.cpu().numpy().view(np.int32), bits)
        np.testing.assert_array_equal(indices.cpu().numpy(), selected)
        for raw in (source_raw, index_raw, output_raw):
            assert bool((raw[:64].cpu() == -777).all()) and bool(
                (raw[-64:].cpu() == -777).all()
            )
