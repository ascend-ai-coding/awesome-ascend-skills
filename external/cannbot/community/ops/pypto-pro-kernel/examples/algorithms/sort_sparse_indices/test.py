# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Index sort: exact key inversion, worker reuse and guarded writes.

The native sorter uses disjoint 20 KiB UB and TileGroup dependency tracking.
The reference orders INT32 values directly. Both the numeric and ordered-bit
keys must preserve endpoints and duplicates. No floating tolerance is used.
"""

import pytest
import torch

from .reference import reference


def make_inputs():
    generator = torch.Generator(device="cpu").manual_seed(47)
    rows = torch.randint(
        0, 1024, (9, 512), generator=generator, dtype=torch.int32, device="cpu"
    )
    rows[0] = torch.arange(511, -1, -1, dtype=torch.int32, device="cpu")
    rows[1] = torch.tensor([0, 1, 65535, 65536, 2**24 - 1, 2**24, 7, 7]).repeat(64)
    rows[2].fill_(37)
    return rows


def test_reference_domain():
    original = make_inputs()
    result = reference(original)
    assert torch.equal(result[0], torch.arange(512, dtype=torch.int32, device="cpu"))
    assert result[1, 0] == 0 and result[1, -1] == 2**24
    assert torch.equal(result[2], original[2])
    for outside in (-1, 2**24 + 1):
        invalid = original.clone()
        invalid[0, 0] = outside
        with pytest.raises(ValueError):
            reference(invalid)


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("key_encoding", ["ordered_bits", "numeric"])
def test_sort_and_reuse(workers, key_encoding, device):
    from .kernel import make_kernel, sort_sparse_indices

    kernel = make_kernel(key_encoding)
    original = make_inputs()
    source = original.to(device)
    backing = torch.full(
        (original.numel() + 128,), -123, dtype=torch.int32, device=device
    )
    output = backing[64:-64].view_as(source)
    for shift in (0, 13):
        current = (original.roll(shift, dims=-1) + shift) % (2**24 + 1)
        source.copy_(current)
        output.fill_(-123)
        kernel[None, workers](source, output)
        torch.npu.synchronize()
        assert torch.equal(output.cpu(), reference(current))
        assert torch.equal(source.cpu(), current)
        assert bool((backing[:64] == -123).all())
        assert bool((backing[-64:] == -123).all())
    actual = sort_sparse_indices(source, key_encoding=key_encoding)
    torch.npu.synchronize()
    assert torch.equal(actual.cpu(), reference(current))
    assert torch.equal(source.cpu(), current)
