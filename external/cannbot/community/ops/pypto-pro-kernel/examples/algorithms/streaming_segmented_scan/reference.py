# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""由数学段起点与全局前缀和计算参考，不复刻 kernel 的分块和摘要合并。"""

import numpy as np


def reference(x, heads):
    prefix = np.concatenate(([0], np.cumsum(x, dtype=np.int64)))
    starts = np.maximum.accumulate(np.where(heads != 0, np.arange(x.size), 0))
    return (prefix[1:] - prefix[starts]).astype(np.int32)
