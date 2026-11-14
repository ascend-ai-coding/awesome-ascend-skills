# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证长序列分段前缀、动态元数据、尾部物理写入与三阶段依赖。

输入/输出使用有非零存储偏移的连续视图；完整检查前后守卫与输入不变。
第二次调用在相同地址更新输入和段首，避免把缓存旧摘要误判为正确。
参考从INT64全局前缀与数学段起点求差，不采用设备端块摘要算法。
同Stream调用完成跨阶段依赖；第二阶段的稠密标量输出只有一个worker写。
性能需按完整三次任务计量；单worker的额外遍历开销独立于多worker收益。
"""

import numpy as np
import pytest
import torch

from .kernel import CHUNK, run
from .reference import reference


def data(size, kind, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(-4, 5, size, dtype=np.int32)
    heads = np.zeros(size, dtype=np.int32)
    if kind == "none":
        x.fill(4)
    elif kind == "all":
        heads.fill(1)
    elif kind == "sparse":
        heads[:] = rng.random(size) < 0.002
    elif kind == "dense":
        heads[:] = rng.random(size) < 0.31
    elif kind == "boundaries":
        for step in (64, CHUNK):
            for delta in (-1, 0, 1):
                heads[np.arange(step + delta, size, step)] = 1
        heads[-1] = 1
    return x, heads


CASES = [
    (size, "boundaries", 4)
    for size in (
        1,
        7,
        8,
        9,
        63,
        64,
        65,
        CHUNK - 1,
        CHUNK,
        CHUNK + 1,
        2 * CHUNK + 1,
        65537,
        131073,
    )
]
CASES += [
    (1048576, kind, workers)
    for kind in ("none", "all", "sparse", "boundaries")
    for workers in (1, 4, 16)
]
CASES += [(10007, "dense", workers) for workers in (1, 16)]


@pytest.mark.parametrize("size,kind,workers", CASES)
def test_scan(device, size, kind, workers):
    backing = [
        torch.full((size + 96,), 937123, dtype=torch.int32, device=device)
        for _ in range(3)
    ]
    x, heads, out = [buffer[slice(17, 17 + size)] for buffer in backing]
    for iteration in range(2):
        values, flags = data(
            size, kind if iteration == 0 else "sparse", 31973 + size + iteration
        )
        x.copy_(torch.from_numpy(values).to(device))
        heads.copy_(torch.from_numpy(flags).to(device))
        out.fill_(-123456789)
        run(x, heads, out, workers=workers)
        torch.npu.synchronize()
        np.testing.assert_array_equal(out.cpu().numpy(), reference(values, flags))
        np.testing.assert_array_equal(x.cpu().numpy(), values)
        np.testing.assert_array_equal(heads.cpu().numpy(), flags)
        for buffer in backing:
            np.testing.assert_array_equal(buffer[:17].cpu().numpy(), 937123)
            np.testing.assert_array_equal(
                buffer[slice(17 + size, None)].cpu().numpy(), 937123
            )
