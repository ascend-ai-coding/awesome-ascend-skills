# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Consume both UB packing paths through Cube with exact quarter-valued inputs.

Changed inputs, non-square geometry, both AIVs and output guards distinguish
layout mistakes and stale handovers. These tests establish no timing result.
"""

import os

import pytest
import torch
import torch_npu  # noqa: F401

from .kernel import COLS, OUT_COLS, handover
from .reference import reference


@pytest.mark.parametrize("rows", [8, 16, 32])
@pytest.mark.parametrize("direct", [0, 1])
def test_handover(rows, direct):
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    for seed in (0, 3):
        x_cpu = (
            (torch.arange(2 * rows * COLS).reshape(2 * rows, COLS) + seed) % 31 - 15
        ).half() / 4
        rhs_cpu = (
            (torch.arange(COLS * OUT_COLS).reshape(COLS, OUT_COLS) * 3 + seed) % 7 - 3
        ).half() / 4
        x, rhs = x_cpu.to(device), rhs_cpu.to(device)
        backing = torch.full(
            (2 * rows + 2, OUT_COLS), -12345.0, dtype=torch.float32, device=device
        )
        out = backing[1:-1]
        handover[torch.npu.current_stream(), 1, {"ROWS": rows, "DIRECT": direct}](
            x, rhs, out
        )
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), reference(x_cpu, rhs_cpu), rtol=0, atol=0)
        assert torch.all(backing[0] == -12345.0).item()
        assert torch.all(backing[-1] == -12345.0).item()
        assert torch.equal(x.cpu(), x_cpu)
        assert torch.equal(rhs.cpu(), rhs_cpu)
