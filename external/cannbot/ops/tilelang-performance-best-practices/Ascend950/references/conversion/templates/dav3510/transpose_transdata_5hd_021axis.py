# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass(frozen=True)
class Transpose021Plan:
    batches: int
    height: int
    width: int
    split_axis: str


def select_plan(
    batches: int, height: int, width: int, num_cores: int = 64
) -> Transpose021Plan:
    split_axis = "H" if batches < num_cores and height >= num_cores else "N"
    return Transpose021Plan(batches, height, width, split_axis)


def reference(x):
    """Exact [N,H,W] -> [N,W,H] semantics."""
    return x.transpose(1, 2).contiguous()
