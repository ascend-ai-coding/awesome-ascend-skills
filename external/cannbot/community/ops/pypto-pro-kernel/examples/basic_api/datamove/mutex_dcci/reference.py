# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""五个 kernel 都算 `x * 2`，参考就是它。

**为什么可以 `rtol=0, atol=0` 比。** `x * 2` 在 FP32 上精确（指数加一、尾数不变）。这一例改的
是同步与 cache 管理，不是算术——**判据要能把"同步／cache 错了"和"算错了"分开**，
所以中间那一步取成一个不引入误差的运算。

输入不含 0：这样「读到了没初始化的 UB」不会碰巧等于正确答案。`dma_pad` 那次就是残留恰好是 0,
让一个错的结果看着像对的。
"""

import numpy as np


def make_input(rows, cols):
    return (np.arange(rows * cols, dtype=np.float32).reshape(rows, cols) + 1.0) * 10.0


def doubled(x):
    return x * 2.0
