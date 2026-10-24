# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""单核L0B复用的独立精度和重复调用对照；不声称性能收益。

输入FP16在[-1,1]，两个M块内容不同；每元素误差<=1e-3+3e-4*绝对项和。输入和输出带非零偏移及守卫；相同物理输入地址更新B，验证常驻仅限单次调用。每次覆盖全部C，手写链与语法糖结果逐位一致。
"""

import numpy as np
import torch

from .kernel import make_kernel
from .reference import reference


def test_retained_operand_is_reloaded_on_each_call(device):
    rng = np.random.default_rng(912)
    a = rng.uniform(-1, 1, (128, 128)).astype(np.float16)
    b = rng.uniform(-1, 1, (128, 128)).astype(np.float16)
    backings = [
        torch.full((128 * 128 + 128,), -99, dtype=dtype, device=device)
        for dtype in (torch.float16, torch.float16, torch.float32)
    ]
    ad, bd, cd = [v[64:-64].view(128, 128) for v in backings]
    ad.copy_(torch.from_numpy(a))
    for scale in (1.0, -0.5, 0.25):
        current_b = (b * scale).astype(np.float16)
        bd.copy_(torch.from_numpy(current_b))
        expected, term_scale = reference(a, current_b)
        before = None
        for reuse in (False, True):
            cd.fill_(123 if reuse else -77)
            make_kernel(reuse)[None, 1](ad, bd, cd)
            torch.npu.synchronize()
            actual = cd.cpu().numpy()
            assert np.all(np.abs(actual - expected) <= 1e-3 + 3e-4 * term_scale)
            if before is not None:
                np.testing.assert_array_equal(
                    before.view(np.uint8), actual.view(np.uint8)
                )
            before = actual.copy()
            for backing in backings:
                assert bool(torch.all(backing[:64] == -99))
                assert bool(torch.all(backing[-64:] == -99))
        np.testing.assert_array_equal(ad.cpu().numpy(), a)
        np.testing.assert_array_equal(bd.cpu().numpy(), current_b)
