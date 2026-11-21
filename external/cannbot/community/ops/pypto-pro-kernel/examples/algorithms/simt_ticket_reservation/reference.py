# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent per-key counts and unordered ticket-set contract."""

import numpy as np


def reference(keys, bins):
    keys = np.asarray(keys)
    if (
        keys.dtype != np.int32
        or keys.ndim != 1
        or not 1 <= bins <= 256
        or np.any(keys < 0)
        or np.any(keys >= bins)
    ):
        raise ValueError("require valid INT32 keys and 1<=B<=256")
    return np.bincount(keys, minlength=bins).astype(np.int32)


def validate(keys, counts, tickets):
    expected = reference(keys, len(counts))
    assert counts.dtype == tickets.dtype == np.int32
    np.testing.assert_array_equal(counts, expected)
    assert tickets.shape == keys.shape
    for key, count in enumerate(expected):
        np.testing.assert_array_equal(
            np.sort(tickets[keys == key]), np.arange(count, dtype=np.int32)
        )
