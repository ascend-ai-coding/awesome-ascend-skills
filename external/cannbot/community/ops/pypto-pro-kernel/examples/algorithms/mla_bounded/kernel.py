# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合。Native MLA paths for materialized, fused and streaming decode calls.

All paths consume independent V and split score features. The materialized path
supports FP16/BF16 and both layouts; fused and single-query decode paths use FP16.
Physical tile placement and synchronization remain explicit in _kernels/.
"""

from ._kernels.materialized import make_kernels, mla
from ._kernels.fused import make_fused, mla_fused
from ._kernels.decode import make_decode_kernel, mla_decode

__all__ = [
    "make_kernels",
    "mla",
    "make_fused",
    "mla_fused",
    "make_decode_kernel",
    "mla_decode",
]
