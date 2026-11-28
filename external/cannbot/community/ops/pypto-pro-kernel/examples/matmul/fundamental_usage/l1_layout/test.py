# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""L1 的四种 layout 组合，以及那条走不通的路。

**解决什么.** 操作数在 GM 里可能是转置存放的。把它喂给 cube 单元时，L1 Tile 的
layout 和 ``pl.load`` 的 ``order`` 必须配套——配不对的组合**不是慢，是编译不过或
算错**。这里把四种组合摆齐：

| case | A 在 GM | ``a_l1`` layout | A 的 load | B 在 GM | ``b_l1`` layout | B 的 load |
|---|---|---|---|---|---|---|
| NN | ``[M, K]`` | NZ | 默认 | ``[K, N]`` | NZ | 默认 |
| NT | ``[M, K]`` | NZ | 默认 | ``[N, K]`` | **ZN** | ``order=[1, 0]`` |
| TN | ``[K, M]`` | **ZN** | ``order=[1, 0]`` | ``[K, N]`` | NZ | 默认 |
| TT | ``[K, M]`` | **ZN** | ``order=[1, 0]`` | ``[N, K]`` | **ZN** | ``order=[1, 0]`` |

四个 case 都算同一个 ``C[M, N] = A[M, K] @ B[K, N]``，区别只在操作数怎么存。

**走不通的那条路.** ``pl.load`` 只有两条通路：

```
GM 的 ND  -> L1 的 NZ      （默认，不转置）
GM 的 DN  -> L1 的 ZN      （order=[1, 0]，转置）
```

**没有 ND -> ZN。** 所以「L1 用 ZN」和「从转置过的 GM 读」是同一件事的两面，不能
只做一半。想让一块常规存放的 ``[M, K]`` 落成 ZN 的 L1 Tile——没有这条路径。

**L0 的 layout 一个都不变.** 四个 kernel 里 ``l0a`` 永远是 NZ、``l0b`` 永远是 ZN、
``acc`` 永远是 NZ。这三块由 cube 单元决定，和操作数在 GM 里怎么放没有关系。
**L1 的 layout 跟着搬运走，L0 的 layout 跟着硬件走**——这两件事经常被混成一件，
而混了之后看到的现象是「改了 L1 就好了」或者「改了 L0 就好了」，两种都会误导。

Tile 的 **shape 也不变**：``a_l1`` 在四个 case 里都是 ``[M, K]``，``b_l1`` 都是
``[K, N]``。Tile 描述的是逻辑操作数，转置发生在搬运里。

**为何这样同步.** 四个 kernel 的五个显式地址组均使用 ``mutex_ids="auto", depth=1``，由 ``auto_mutex`` 插入同步。
转置不改变同步：``order`` 只影响 ``pl.load`` 怎么读 GM，读进来之后的依赖关系
（load 写 L1、move 读 L1）与 NN 完全一样。单缓冲，没有重叠。

**如何独立验证.** 形状取 ``M=64, K=128, N=32`` 的非方阵——方阵会让 ``[M, K]`` 和
``[K, M]`` 长得一样，维度搞反也测不出来。每个 case 的 reference 先把 GM 里的存放
方向还原成逻辑操作数，再按定义算矩阵乘；转置写在 reference 里而不是喂给 kernel
之前，这样 kernel 拿到的就是真正转置存放的张量。

源于 ``pypto/python/tests/st/pypto_pro/frontend/matmul/test_matmul_api_basic.py``
的 t01–t24 系列（``t14_sq_nd_nz_single`` 是 NN，``t11_nsq_dn_zn_tt_single`` 是 TT）。
"""

import pytest
import torch

from .kernel import K, M, N, matmul_nn
from .kernel import matmul_nt
from .kernel import matmul_tn
from .kernel import matmul_tt
from .reference import reference_nn
from .reference import reference_nt
from .reference import reference_tn
from .reference import reference_tt

# case 名 -> (kernel, reference, A 在 GM 的形状, B 在 GM 的形状)
CASES = {
    "nn": (matmul_nn, reference_nn, [M, K], [K, N]),
    "nt": (matmul_nt, reference_nt, [M, K], [N, K]),
    "tn": (matmul_tn, reference_tn, [K, M], [K, N]),
    "tt": (matmul_tt, reference_tt, [K, M], [N, K]),
}


@pytest.mark.parametrize("case", sorted(CASES), ids=sorted(CASES))
def test_l1_layout(case, device):
    kernel, reference, a_shape, b_shape = CASES[case]
    torch.manual_seed(0)
    a = torch.randn(a_shape, device=device, dtype=torch.float16)
    b = torch.randn(b_shape, device=device, dtype=torch.float16)
    out = torch.zeros([M, N], device=device, dtype=torch.float32)

    kernel(a, b, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, reference(a, b), rtol=2e-2, atol=2e-2)


def test_every_case_computes_the_same_product(device):
    """四个 case 算的是同一个乘积，只是操作数存放方向不同。

    单看四条 parametrize 会漏掉一种错法：某个 case 的 kernel 和它的 reference
    一起把维度搞反了，两边对得上，但算的已经不是原来那个矩阵乘。这里用同一对
    逻辑矩阵喂四个 case，要求输出彼此一致。
    """
    torch.manual_seed(1)
    a_logical = torch.randn([M, K], device=device, dtype=torch.float16)
    b_logical = torch.randn([K, N], device=device, dtype=torch.float16)

    outputs = {}
    for case, (kernel, _, a_shape, _b_shape) in CASES.items():
        a = a_logical if a_shape == [M, K] else a_logical.t().contiguous()
        b = b_logical if case in ("nn", "tn") else b_logical.t().contiguous()
        out = torch.zeros([M, N], device=device, dtype=torch.float32)
        kernel(a, b, out)
        torch.npu.synchronize()
        outputs[case] = out

    for case, out in outputs.items():
        torch.testing.assert_close(
            out,
            outputs["nn"],
            rtol=2e-2,
            atol=2e-2,
            msg=lambda m, c=case: f"{c} 与 nn 不一致\n{m}",
        )
