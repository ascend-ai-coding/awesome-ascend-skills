#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
TileLang-Ascend simple test script
Used to verify if the environment is correctly configured
"""

import logging

import torch
import tilelang
import tilelang.language as T

logging.basicConfig(level=logging.INFO)


@tilelang.jit(out_idx=[-1])
def simple_add(m=16, n=16, dtype="float"):
    @T.prim_func
    def kernel(
        a: T.Tensor((m, n), dtype),
        b: T.Tensor((m, n), dtype),
        c: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((m, n), dtype)
            b_ub = T.alloc_ub((m, n), dtype)
            c_ub = T.alloc_ub((m, n), dtype)

            with T.Scope("V"):
                T.copy(a[0, 0], a_ub)
                T.copy(b[0, 0], b_ub)
                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()
                T.copy(c_ub, c[0, 0])

    return kernel


def main():
    tilelang.cache.clear_cache()

    m, n = 16, 16
    func = simple_add(m, n)

    a = torch.randn(m, n).npu()
    b = torch.randn(m, n).npu()
    c = func(a, b)

    torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)
    logging.info("✓ TileLang 环境验证通过!")


if __name__ == "__main__":
    main()
