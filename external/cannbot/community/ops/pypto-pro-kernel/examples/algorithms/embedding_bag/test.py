# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Runtime segments, indirect reads, direct/split paths, repeat calls and guards."""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import embedding_bag_reference


def guarded(value, device):
    dtype = torch.float32 if value.dtype == np.float32 else torch.int32
    backing = torch.full((value.size + 128,), 1234, dtype=dtype, device=device)
    tensor = backing[64:-64].view(*value.shape)
    tensor.copy_(torch.from_numpy(value).to(device))
    return tensor, backing


@pytest.mark.parametrize(
    "vocab,dims,bags,tokens",
    [
        (17, 33, 5, 19),
        (257, 65, 33, 257),
        (513, 128, 7, 4097),
        (23, 129, 4, 73),
        (29, 67, 5, 2051),
    ],
)
@pytest.mark.parametrize("kind", ["random", "empty", "duplicate", "padding", "long"])
@pytest.mark.parametrize("workers", [1, 3])
def test_embedding_bag(device, vocab, dims, bags, tokens, kind, workers):
    rng = np.random.default_rng(vocab * dims + tokens)
    table = rng.uniform(-1, 1, (vocab, dims)).astype(np.float32)
    indices = rng.integers(-1, vocab, tokens, dtype=np.int32)
    lengths = rng.multinomial(tokens, np.full(bags, 1 / bags))
    if kind == "empty":
        lengths[:] = 0
        lengths[-1] = tokens
    elif kind == "duplicate":
        indices.fill(vocab - 1)
    elif kind == "padding":
        indices.fill(-1)
    elif kind == "long":
        lengths[:] = 0
        lengths[0], lengths[-1] = tokens - 1, 1
        indices[::2] = -1
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    originals = [table, indices, offsets]
    expected = embedding_bag_reference(*originals)
    inputs = [guarded(value, device) for value in originals]
    output = guarded(np.zeros((bags, dims), dtype=np.float32), device)
    for _ in range(2):
        output[0].fill_(float("nan"))
        run(*(pair[0] for pair in inputs), output[0], workers=workers)
        torch.npu.synchronize()
        np.testing.assert_allclose(
            output[0].cpu().numpy(), expected, atol=2e-4, rtol=2e-4
        )
        for _, backing in inputs + [output]:
            np.testing.assert_array_equal(backing[:64].cpu().numpy(), 1234)
            np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 1234)
    for (tensor, _), original in zip(inputs, originals):
        np.testing.assert_array_equal(
            tensor.cpu().numpy().view(np.uint32), original.view(np.uint32)
        )
