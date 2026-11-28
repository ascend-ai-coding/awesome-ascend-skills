# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证独立频数、逐 key 唯一 ticket、输出覆盖、输入不变和两侧守卫。"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import validate


@pytest.mark.parametrize(
    "n,bins,kind",
    [
        (0, 7, "random"),
        (1, 1, "hot"),
        (259, 17, "random"),
        (8197, 31, "runs"),
        (65539, 17, "hot"),
        (131101, 63, "no_runs"),
    ],
)
@pytest.mark.parametrize(
    "combine_runs,workers,threads", [(False, 1, 128), (True, 3, 256)]
)
def test_ticket_reservation(device, n, bins, kind, combine_runs, workers, threads):
    rng = np.random.default_rng(90419 + n)
    raw_counts = torch.full((bins + 128,), -9876543, dtype=torch.int32, device=device)
    raw_tickets = torch.full((n + 128,), -9876543, dtype=torch.int32, device=device)
    counts, tickets = raw_counts[64:-64], raw_tickets[64:-64]
    for repeat in range(2):
        values = rng.integers(0, bins, n, dtype=np.int32)
        if kind == "hot":
            values[:] = bins - 1 if repeat == 0 else 0
        elif kind == "runs":
            values[:] = (np.arange(n) // (3 * workers * threads) + repeat) % bins
        elif kind == "no_runs":
            values[:] = (np.arange(n) // (workers * threads) + repeat) % bins
        keys = torch.from_numpy(values).to(device)
        counts.fill_(313 + repeat)
        tickets.fill_(313 + repeat)
        run(
            keys,
            counts,
            tickets,
            workers=workers,
            threads=threads,
            combine_runs=combine_runs,
        )
        torch.npu.synchronize()
        validate(values, counts.cpu().numpy(), tickets.cpu().numpy())
        np.testing.assert_array_equal(keys.cpu().numpy(), values)
        for raw in (raw_counts, raw_tickets):
            assert (raw[:64].cpu() == -9876543).all() and (
                raw[-64:].cpu() == -9876543
            ).all()
