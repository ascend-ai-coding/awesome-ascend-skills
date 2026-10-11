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
class SoftmaxTiling:
    rows: int
    cols: int
    tile_cols: int
    num_cores: int
    strategy: str

    @property
    def active_cores(self) -> int:
        return min(self.rows, self.num_cores)


@dataclass(frozen=True)
class SoftmaxStrategy:
    name: str
    layout: str
    input_passes: int
    fp32_state: tuple[str, ...]
    selection: str
