# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Find neighboring reduction lengths for a constant-input variance test."""
import numpy as np


def find_variance_pair(lengths, constants, *, input_dtype, accumulator_dtype,
                       output_dtype, offset, matches):
    """Return (probe_length, control_length, stored_constant), or None.

    lengths and constants are bounded candidates from the caller's legal domain.
    matches(actual, expected) applies the operator's output comparison to scalars.
    The calculation models rounded sums followed by reciprocal multiplication.
    """
    store = np.dtype(input_dtype).type
    acc = np.dtype(accumulator_dtype).type
    out = np.dtype(output_dtype).type
    lengths = sorted(set(lengths))
    best_pair = None
    largest_error = 0.0
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        expected = out(acc(offset))
        if not np.isfinite(expected):
            return None
        for value in constants:
            c = acc(store(value))
            square = acc(c * c)
            if not np.isfinite(square):
                continue
            controls = set()
            errors = {}
            for n in lengths:
                if n <= 0 or int(n) != n or not np.isfinite(acc(n)) or int(acc(n)) != n:
                    continue
                count = acc(n)
                reciprocal = acc(acc(1) / count)
                mean = acc(acc(count * c) * reciprocal)
                moment = acc(acc(count * square) * reciprocal)
                variance = out(acc(acc(moment - acc(mean * mean)) + acc(offset)))
                if not np.isfinite(variance):
                    continue
                if matches(variance, expected):
                    controls.add(n)
                else:
                    errors[n] = abs(float(variance) - float(expected))
            for n, error in errors.items():
                if error > largest_error:
                    for neighbor in (n - 1, n + 1):
                        if neighbor in controls:
                            best_pair = n, neighbor, float(c)
                            largest_error = error
                            break
    return best_pair
