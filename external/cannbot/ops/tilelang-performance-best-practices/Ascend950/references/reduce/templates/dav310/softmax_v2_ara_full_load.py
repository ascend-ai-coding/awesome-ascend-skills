# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from ..softmax_v2_tiling_data import SoftmaxStrategy

STATUS = "DESIGN_ONLY"
STRATEGY = SoftmaxStrategy(
    "ara_full_load",
    "[A1, R, A0]",
    1,
    ("max[A0]", "sum[A0]", "exp tile"),
    "R x tile_A0 and fp32 workspace fit UB",
)
