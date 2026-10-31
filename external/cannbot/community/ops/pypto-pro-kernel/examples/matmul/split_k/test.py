# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""K 维分块归约：首块写入，其余累加。

**解决什么.** 归约维 K 大到一次装不进 L0 时，把它切成若干步，每步算一个部分积，
累加在同一块 L0C 上。这里 ``K=256`` 切成两步 ``SK=128``。

关键是**只有第一步写入累加器，其余都累加**。写错方向的后果是静默的：全都用
``matmul`` 会让后一步覆盖前一步，结果是最后一块部分积；全都用 ``matmul_acc`` 会
把 L0C 上的残留值一起加进去，跑第二次才看得出不对。

**物理访问.** A 和 B 各一次整体搬进 L1（``[M, 256]`` 和 ``[256, N]``），循环里从
L1 上按 K 的偏移取切片搬进 L0：

```python
pl.move(al, cur_a, offset=[0, ki])    # L1 的 [M, 256] 里取 [M, 128]
pl.move(br, cur_b, offset=[ki, 0])    # L1 的 [256, N] 里取 [128, N]
```

``l0a`` / ``l0b`` 是**双槽**的：K 的两步交替使用，前一步还在算的时候后一步就能
搬进来。累加器是**单槽**的——每一步都往同一块 L0C 上累加，这正是分块归约的定义，
给它两个槽等于把部分和拆开放，那就不是累加了。

**为何这样同步.** ``make_tile_group`` 的显式地址将 L0A / L0B 各分成两个独立槽，
``mutex_ids="auto"`` 规划 ID，``auto_mutex`` 据此插同步。所以第 2 步的 ``move`` 不必等第
1 步的 ``matmul`` 读完——它写的是另一个槽、另一把锁。累加器只分到一个 ID，
两步的 matmul 因此串行，这是对的：它们在累加同一片字节。

``AccPhase`` 不在这个样例里。K 分块配合硬件 unitFlag 的用法（Partial / Final）
属于另一个机制，留给 ``acc_phase`` 样例。**不配 phase 也是正确的**，只是少了那一
层流水优化。

**如何独立验证.** 两层：

1. 手写循环与 ``pl.gemm(splitk=SK)`` 逐值比较（``rtol=0``）。gemm 展开出来的就是
   上面那个循环，不该有任何差异。
2. 与 ``reference.py`` 比。参考实现**一次算完整个 K，不分块**——照着分一遍只能
   验出分块被抄对了，验不出分块归约等于不分块归约。

第 2 条还顺带覆盖了「全用 matmul」这种错法：那会得到最后一块部分积，与完整归约
差得很远。

源于 ``pypto/python/tests/st/pypto_pro/frontend/matmul/test_matmul_api_basic.py``
的 K-loop 系列，以及 ``test_gemm_st.py`` 的 ``gemm_splitk``。
"""

import torch

from .kernel import K, M, N, split_k_by_gemm, split_k_by_hand
from .reference import matmul_reference


def _run(kernel, a, b):
    out = torch.zeros([M, N], device=a.device, dtype=torch.float32)
    kernel(a, b, out)
    torch.npu.synchronize()
    return out


def test_split_k_accumulates_over_the_whole_reduction(device):
    torch.manual_seed(0)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)

    by_hand = _run(split_k_by_hand, a, b)
    by_gemm = _run(split_k_by_gemm, a, b)

    # gemm 展开出来的就是手写的那个循环。
    torch.testing.assert_close(by_gemm, by_hand, rtol=0, atol=0)

    # 与一次算完的参考比：只算了最后一块部分积的话，这里差得很远。
    torch.testing.assert_close(by_hand, matmul_reference(a, b), rtol=2e-2, atol=2e-2)


def test_running_twice_gives_the_same_answer(device):
    """累加器每次都被首块写入，不会把上一次 launch 的残留值加进来。

    全用 ``matmul_acc`` 而漏掉首块写入时，第一次跑可能恰好对（L0C 初值为零），
    第二次就会翻倍。
    """
    torch.manual_seed(1)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)

    first = _run(split_k_by_hand, a, b)
    second = _run(split_k_by_hand, a, b)

    torch.testing.assert_close(second, first, rtol=0, atol=0)
