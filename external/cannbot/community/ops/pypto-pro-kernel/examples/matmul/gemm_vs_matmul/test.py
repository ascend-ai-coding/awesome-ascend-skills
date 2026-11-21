# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""``pl.gemm`` 与手写 move+move+matmul：省掉了什么，没省掉什么。

**解决什么.** 同一个 ``C = A @ B`` 写两遍，逐值对照。读完知道 ``pl.gemm`` 能替换
哪三行、以及**它不替换什么**——这是后面 ``split_k`` / ``split_n`` / ``tail_extent``
三个样例的共同前提，那三个都是 gemm 参数的展开。

**物理访问.** 两个 kernel 共用 ``_groups()`` 的五条显式地址声明。
数据通路也相同：GM → L1 → L0A/L0B → L0C → GM。差别只在 section 内部：

```python
pl.move(l0a.current(), cur_a)                   # 手写：三行
pl.move(l0b.current(), cur_b)
pl.matmul(ac, l0a.current(), l0b.current())

pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)     # gemm：一行
```

``pl.gemm`` 接收的是 **L1 上的两个操作数**，自己把它们搬进调用方给的 L0 暂存
Tile 再发矩阵乘；``pl.matmul`` 接收的是**已经在 L0 上**的操作数。这是两者唯一
的定位差别。

**没省掉的部分才是要点**：L0A / L0B 的形状和缓冲深度仍由调用方声明；
地址由调用方按 memory space 分别规划，``mutex_ids="auto"`` 交给框架分配核内同步 ID。双缓冲仍是调用方的选择——``l0a`` / ``l0b``
以 TileGroup 的形式传进去，gemm 每个 split 步骤取一次 ``next()``。传单块 Tile
也可以。

gemm 在**解析期**展开成上面那三行，不引入新的 IR 形态。删掉语法糖，剩下的仍是
一个普通 kernel。

**为何这样同步.** 两边都只靠 ``auto_mutex``。因为 gemm 展开出来的就是普通的
``move`` / ``matmul``，它们照常按 Tile 的 mutex ID 拿到同步，不需要也没有第二套
机制。这也是这个样例用**逐值相等**而不是"都对"来比较的原因：如果 gemm 的展开
改变了同步或地址，两边的数值会分开。

**如何独立验证.** 两层：

1. 两个 kernel 用同一批输入，``rtol=0, atol=0`` 逐值比较。同样的 op、同样的顺序、
   同样的字节，不该有任何差异。
2. 再与 ``reference.py`` 从定义算出的结果比（fp16 输入 / fp32 累加，按 2e-2）。
   只比第 1 条会漏掉"两边错得一样"。

源于 ``pypto/python/tests/st/pypto_pro/frontend/matmul/test_gemm_st.py`` 的
``gemm_plain``，以及同目录 ``test_gemm_matrix_st.py`` 的手写版与 GEMM 版对照。
"""

import torch

from .kernel import K, M, N, matmul_by_gemm, matmul_by_hand
from .reference import matmul_reference


def _run(kernel, a, b):
    """跑一次，返回输出。"""
    out = torch.zeros([M, N], device=a.device, dtype=torch.float32)
    kernel(a, b, out)
    torch.npu.synchronize()
    return out


def test_gemm_matches_the_hand_written_form(device):
    torch.manual_seed(0)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)

    by_hand = _run(matmul_by_hand, a, b)
    by_gemm = _run(matmul_by_gemm, a, b)

    # 同样的 op、同样的顺序、同样的字节：这里出现差异只可能是降级挪动了什么，
    # 不可能是舍入不同。
    torch.testing.assert_close(by_gemm, by_hand, rtol=0, atol=0)

    # 再确认两边算的确实是矩阵乘——否则「完全一致」什么都证明不了。
    reference = matmul_reference(a, b)
    torch.testing.assert_close(by_hand, reference, rtol=2e-2, atol=2e-2)
