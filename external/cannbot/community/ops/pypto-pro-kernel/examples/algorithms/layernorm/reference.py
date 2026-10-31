# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""数学定义参考：逐行 LayerNorm（无仿射），有偏（总体）方差。

out[i, :] = (x[i, :] - mean(x[i, :])) / sqrt(var(x[i, :]) + eps)

用 float64 中间精度算 mean / var / normalize，只在最后转回 float32——让参考自身的
舍入误差远小于被测 FP32 kernel 的舍入误差，比较时看到的基本是 kernel 那一侧的舍入。
不复现 kernel 的 64-lane 分块或 tail 掩码路径：那是「照抄实现」，不是「从数学定义写起」。
"""

import numpy as np

EPS = 1e-5  # 必须与 kernel.py 的编译期常量一致


def layernorm_reference(x: np.ndarray) -> np.ndarray:
    """x: [rows, cols]，任意可转 float 的数组。返回 float32 [rows, cols]。"""
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=1, keepdims=True)
    var = ((x64 - mean) ** 2).mean(axis=1, keepdims=True)  # 有偏/总体方差，ddof=0
    out = (x64 - mean) / np.sqrt(var + EPS)
    return out.astype(np.float32)
