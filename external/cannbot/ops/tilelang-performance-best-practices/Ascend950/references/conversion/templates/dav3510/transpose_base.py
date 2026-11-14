# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Adapt an explicitly supplied, validated transpose factory from the current repository.

The factory receives full shape_x, shape_y and dtype and returns the callable
kernel. Device validation and the factory's input/output contract remain the
caller's responsibility; this adapter does not implement a transpose kernel.
"""

__all__ = ["TransposeTiling", "build"]

from ..transpose_tiling_data import TransposeTiling, select_tiling

STATUS = "PARTIAL"


def build(shape_x: int, shape_y: int, dtype, *, kernel_factory=None):
    cfg = select_tiling(shape_x, shape_y)
    cfg.validate(shape_x, shape_y)
    if not callable(kernel_factory):
        raise ValueError(
            "Pass a validated kernel_factory(shape_x, shape_y, dtype) from the current repository"
        )
    return kernel_factory(shape_x, shape_y, dtype)
