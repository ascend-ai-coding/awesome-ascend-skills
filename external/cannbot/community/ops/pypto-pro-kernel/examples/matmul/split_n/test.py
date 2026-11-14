# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""N 维切条带：条带是累加器内部的视图，和它共享地址与 mutex ID。

**解决什么.** 右操作数太宽、L0B 一次装不下时，把 N 切成若干条带，每条带算一遍。
与 ``split_k`` 的区别是**切的轴不同带来的后果完全不同**：

| | 切 K | 切 N |
|---|---|---|
| 每步算的是 | 同一片输出的部分和 | 输出的**不同列** |
| 累加器 | 一块，逐步累加 | 一块，逐步**写不同的列条带** |
| 每步之间 | 有依赖（累加） | 无依赖（各写各的列） |

所以 ``splitn`` 不需要 ``matmul_acc``：每条带都是 ``init=True`` 写入自己那几列。

**物理访问.** 五个资源组使用显式地址和 ``mutex_ids="auto"``；条带**不出现在声明里**。
``gemm`` 在解析期从累加器内部切出若干视图，
每个视图是 ``[M, SN]``，地址是 ``acc_base + j * stride``，并且**携带累加器自己的
mutex ID**。

这是 mutex ID 语义的一个直接例子：ID 不是唯一性标签，它表示「这几个句柄指向同一
片字节」。条带和累加器确实是同一片字节，自动规划让条带继承父缓冲的同步所有权，
不会将它们当作独立分配。

要声明的只有 ``l0b``，宽度是**一条带**的宽度而不是 N。

**尾条带与 compact.** ``splitn`` 不要求整除。``n=104, splitn=48`` 切成
``48 + 48 + 8``，最后那 8 列会**向上取整到 16**——NZ 的累加器 Tile 以 16 列为
分形粒度，取到 16 就停，不会取到 32（那会跨进下一条带的列）。

``n`` 收窄 ``l0b`` **和累加器**，所以**两块都要 ``compact=1``**。漏掉任何一块都
会直接报错，不会静默算错：

```text
pl.gemm() needs 'l0b' declared with compact=1 to take an effective extent
pl.gemm() needs 'ac'  declared with compact=1 to take an effective extent
```

收窄关系记在 gemm 的 API 页 (`pypto/docs/zh/api/pro_api/SIMD-API/cube_computation/gemm.md`)：
``m`` 收窄 ``l0a`` 和 ``acc``，``n`` 收窄 ``l0b`` 和 ``acc``，``k`` 收窄 ``l0a``
和 ``l0b``。只考虑被切维度对应的 L0 会漏掉累加器。

整除的那个 case 没有任何一步被收窄，所以两块都不要 compact，也不该加。

**两个不能做的事.**

- ``n`` 在 ``splitn`` 之下是循环上界，必须是编译期常量。
- ``m`` 与 ``splitn`` **互斥**。在 M 上被收窄的累加器在 M 方向也是 compact 的，
  那会缩短每个列分形，从而改变列条带的寻址步长——条带就落错位置了。gemm 直接
  拒绝这个组合（``cannot combine 'm' with 'splitn'``），不让它静默算错。

**为何这样同步.** ``auto_mutex``。条带共用累加器的 ID，所以各步的 matmul 在那把
锁上串行——这比必要的严格（它们写的是不相交的列），但安全，而且这正是「同一片
字节共用一把锁」这条不变量的直接后果。``l0b`` 是双槽的，两步的搬运可以重叠。

**如何独立验证.** 参考实现整个 N 一次算完，不切条带。两个 case：

1. 整除：``splitn=64`` 切 128 列，全列与参考比。
2. 尾条带：``n=104, splitn=48``，比 ``[:, :104]``。条带落错位置是这个机制最典型的
   错法，而且是**静默**的——地址错了结果照样是一堆合理的数，只是放错了列。所以
   这里比的是整片区域而不是抽样。

源于 ``pypto/python/tests/st/pypto_pro/frontend/matmul/test_gemm_st.py`` 的
``gemm_splitn``，尾条带的取整规则来自 ``test_gemm.py`` 的
``test_gemm_cuts_a_narrower_last_strip``。
"""

import torch

from .kernel import K, M, N, N_EXTENT, split_n_even, split_n_tail
from .reference import matmul_reference


def _run(kernel, a, b):
    out = torch.zeros([M, N], device=a.device, dtype=torch.float32)
    kernel(a, b, out)
    torch.npu.synchronize()
    return out


def test_strips_land_on_their_own_columns(device):
    torch.manual_seed(0)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)

    out = _run(split_n_even, a, b)

    # 整片比较：条带落错列是静默的，抽样看不出来。
    torch.testing.assert_close(out, matmul_reference(a, b), rtol=2e-2, atol=2e-2)


def test_a_tail_strip_is_rounded_up_not_widened(device):
    """``n=104, splitn=48`` -> ``48 + 48 + 8``，末条带取整到 16 列。"""
    torch.manual_seed(1)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)

    out = _run(split_n_tail, a, b)

    reference = matmul_reference(a, b)
    torch.testing.assert_close(
        out[:, :N_EXTENT], reference[:, :N_EXTENT], rtol=2e-2, atol=2e-2
    )
