# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。Native variants with explicit addresses and automatic mutex IDs."""

__all__ = ["K", "M", "N", "matmul_nn", "matmul_nt", "matmul_tn", "matmul_tt"]

from ._kernels.nn import K, M, N, matmul_nn
from ._kernels.nt import matmul_nt
from ._kernels.tn import matmul_tn
from ._kernels.tt import matmul_tt
