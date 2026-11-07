# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""十四个算子的 CPU 参考，以及每个算子的容差。

**容差是量出来的，不是猜的。** `TOLERANCE` 里每一项都附了实测的最大相对误差；
放宽任何一项都要先有实测依据（`references/verification/evidence.md` 第 5 条：放宽阈值要有文档依据，
而且这一类错起来最隐蔽）。

**为什么不能一律 `rtol=0`。** 超越函数在硬件上是近似实现，和 numpy 的双精度实现不会逐位相同。
但实测分成了清楚的两组，所以**分开定容差，不要一个数糊到底**：

| 组 | 算子 | 实测最大相对误差 | 容差 |
| --- | --- | ---: | --- |
| 逐位相等 | `fma` `abs` `min` `max` | **0** | `rtol=0, atol=0` |
| 约 1 ulp | `sqrt` `rsqrt` `exp` `exp2` `log` `log2` `log1p` `sin` `cos` | ≤ 1.18e-07 | `rtol=2e-07` |
| 约 3 ulp | `tanh` | 3.48e-07 | `rtol=5e-07` |

FP32 的 eps 是 1.19e-07，所以第二组就是"最后一位"。

**`sqrt` 不在逐位组里**——我最初以为它会在（IEEE 754 要求 `sqrt` 精确舍入），实测 1.12e-07,
也就是硬件上它和别的超越函数一样是近似实现。**这句话是被实测改过的，不是一开始就写对的。**

**`fma` 逐位相等这件事本身是个结论**：参考用 float64 算 `a*b+a` 再转 float32，和硬件的融合结果
完全一致，说明它真的只舍入一次（分开算两次舍入的话这一组过不了 `rtol=0`）。

`atol` 一律取 0，这是有实测依据的：`b` 里含 0，`sin(0)` 和 `tanh(0)` 硬件返回的是**精确的 0**
（否则上面那张表里它们的相对误差会是天文数字，因为分母是 0）。
"""

import numpy as np

THREADS = 32

OPS = (
    "sqrt",
    "rsqrt",
    "exp",
    "exp2",
    "log",
    "log2",
    "log1p",
    "sin",
    "cos",
    "tanh",
    "fma",
    "abs",
    "min",
    "max",
)
"""行顺序。**`kernel.py` 的 ROW_* 常量和这张表必须一致**——这里是唯一的顺序来源。"""


def make_inputs(threads=THREADS):
    """`a` 严格正、`b` 含 0 和负数。

    `a` 取 `0.25, 0.5, ... 8.0` 的等比+线性混合，覆盖小于 1 和大于 1 两侧（`log` 变号的地方）。
    `b` 取 `-4 .. 4`，**含 0**（`sin(0)`、`abs(0)`、`tanh(0)` 的边界）。
    """
    a = np.linspace(0.25, 8.0, threads, dtype=np.float32)
    b = np.linspace(-4.0, 4.0, threads, dtype=np.float32)
    b[threads // 2] = 0.0
    # **显式塞一个 0 进去**：`linspace(-4, 4, 32)` 点数是偶数，正好跨过 0 而不落在 0 上。
    # 不含 0 的话 `sin(0)` / `tanh(0)` / `abs(0)` 这几个边界一个都验不到。
    return a.reshape(1, threads), b.reshape(1, threads)


def expected(a, b):
    """逐行的 CPU 参考。用 float64 算再转回 float32：**参考要比被测的更准**，
    否则两边都有误差时分不清是谁的。
    """
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    rows = {
        "sqrt": np.sqrt(a64),
        "rsqrt": 1.0 / np.sqrt(a64),
        "exp": np.exp(b64),
        "exp2": np.exp2(b64),
        "log": np.log(a64),
        "log2": np.log2(a64),
        "log1p": np.log1p(a64),
        "sin": np.sin(b64),
        "cos": np.cos(b64),
        "tanh": np.tanh(b64),
        "fma": a64 * b64 + a64,
        "abs": np.abs(b64),
        "min": np.minimum(a64, b64),
        "max": np.maximum(a64, b64),
    }
    return np.stack([rows[name].astype(np.float32) for name in OPS])


EXACT = ("fma", "abs", "min", "max")
"""实测逐位相等的四个。**不给它们任何容差**——松一点就再也发现不了退化。"""

RTOL = {name: 2e-07 for name in OPS}
"""每个算子的相对容差。**数字来自实测**（见模块 docstring 的表），不是估的。"""
RTOL.update({name: 0.0 for name in EXACT})
RTOL["tanh"] = 5e-07
