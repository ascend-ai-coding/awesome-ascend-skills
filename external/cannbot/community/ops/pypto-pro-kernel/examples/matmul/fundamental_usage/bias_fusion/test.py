# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""融合 bias：它走的不是 a / b 那条通路。

**解决什么.** ``C = A @ B + bias`` 不需要在 vector 侧补一次加法——cube 单元可以在
矩阵乘的同时把 ``[1, N]`` 的 bias 沿 M 铺进累加器。省掉的是一次 L0C→UB→L0C 的往返
和一次跨核交接。

代价是 bias 有**自己的一套规则**，和 a / b 那条通路处处不同。

**物理访问：三处不同.**

```
a / b :  GM --load--> L1 (Mat)  --move--> L0A (Left) / L0B (Right)   dtype 不变
bias  :  GM --load--> L1 (Mat)  --move--> L0  (Bias)                 FP16 -> FP32
```

| | a / b | bias |
|---|---|---|
| L0 的 space | ``Left`` / ``Right`` | **``Bias``** |
| L0 的 dtype | 与输入相同（FP16） | **必须 FP32** |
| shape | ``[M, K]`` / ``[K, N]`` | **``[1, N]``**，沿 M 广播 |

**三条会被拒绝的写法**（都是编译期失败，不是静默算错）：

1. **直接 load 进 Bias 空间**——``pl.load`` 只支持 ``Vec`` / ``Mat``。必须先落
   ``Mat``，再 ``move`` 过去。所以 bias 的 L1 中转**不是浪费，是唯一通路**。
2. **L0 那块不是 FP32**（写成 FP16 / BF16），或者不在 ``Bias`` 空间（写成
   ``Right`` / ``Vec`` / ``Mat``）。
3. **shape 写成 ``[M, N]``**。bias 是一行，广播是硬件做的。

搬进 ``Bias`` 的合法来源只有 ``Mat``。``Left`` → ``Bias``、``Acc`` → ``Bias``、
``Vec`` → ``Bias`` 都会被拒。

那个 ``Mat``(FP16) → ``Bias``(FP32) 的 ``move`` **顺带做类型转换**——这是 ``move``
少见的一种用法，不是写错了。

**K 分段时 bias 放哪一段.** 只能放**第一段**，理由在 op 的语义里：

```python
pl.matmul(ac, al, br, bl)        # 覆盖写累加器（init），顺带铺 bias
pl.matmul_acc(ac, ac, al, br)    # 累加
```

``pl.matmul`` 带不带 bias 都是**覆盖写**。挂到中间段或最后一段，会把前面几段的部分
和冲掉。框架直接拒绝这个组合，不让它静默算错。bias 在第一段铺进去之后就留在累加器
里，后面几段累加上去——所以后面几段**不用也不能再提它**。

**两种传参都行.** ``pl.matmul(ac, al, br, bl)`` 位置传，或
``pl.matmul(ac, al, br, bias_tile=bl)`` 关键字传，两种在 ST 里都有。

**为何这样同步.** 显式地址组使用 ``mutex_ids="auto"``，由 ``auto_mutex`` 插入同步。bias 的两块 Tile 和
别的 Tile 一样各拿自己的 ID，没有特殊之处——``Bias`` 是一个普通的 memory space，
只是地址检查对它**没有强制对齐**（``DDR`` / ``Scaling`` / ``Bias`` 三个空间在对齐表
里没有条目）。

K 分段版本里 bias 只在第一次迭代搬，所以它是单槽；``a_l1`` / ``l0a`` 每段都换，是
双槽。

**这个样例为什么不放负例.** 上面那几条被拒绝的写法都是编译期失败，本来可以演示，
但按 [tooling/examples.md](../../../../references/tooling/examples.md) 的准入规则负例不放这里。规则记在上面，
可执行的反例进 ``tests/negative/``。

**如何独立验证.** 三条：

1. 单次 matmul + bias 与参考比。参考用 torch 的广播，不手写——手写会把 kernel 的
   广播方式抄一遍。
2. K 分三段 + bias 与**同一个**参考比。bias 只加一次这件事，只有在分段版本里才
   可能出错（加三次、或者被第二段冲掉），而两种错法**数值差异都很大**，不会被
   2e-2 的容差盖过去。
3. bias 全零时，带 bias 的 kernel 必须与 ``acc_phase`` 样例里不带 bias 的同形 kernel
   算出一样的数。这条把「bias 通路本身没有副作用」单独钉住：如果那条 ``Mat`` →
   ``Bias`` 的转换搬运顺带改了别的东西，前两条仍可能通过。

源于 ``pypto/python/tests/st/pypto_pro/frontend/matmul/test_matmul_bias.py``
（25 个 test）与 ``test_matmul_bias_constraints.py``（约束与报错，13 个 test）。
"""

import torch

from .kernel import K, M, N, TILE, matmul_bias, matmul_bias_k_split
from .reference import matmul_bias_reference


def _run(kernel, a, b, bias):
    out = torch.zeros([M, N], device=a.device, dtype=torch.float32)
    kernel(a, b, bias, out)
    torch.npu.synchronize()
    return out


def test_bias_is_broadcast_over_every_row(device):
    """单次 matmul + bias。``[1, N]`` 要铺到全部 M 行。"""
    torch.manual_seed(0)
    a = torch.randn([M, TILE], device=device, dtype=torch.float16)
    b = torch.randn([TILE, N], device=device, dtype=torch.float16)
    bias = torch.randn([1, N], device=device, dtype=torch.float16)

    out = _run(matmul_bias, a, b, bias)

    torch.testing.assert_close(
        out, matmul_bias_reference(a, b, bias), rtol=2e-2, atol=2e-2
    )


def test_bias_is_added_once_across_a_k_split(device):
    """K 分三段，bias 只在第一段。加三次或被冲掉都会在这里露出来。"""
    torch.manual_seed(1)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)
    bias = torch.randn([1, N], device=device, dtype=torch.float16)

    out = _run(matmul_bias_k_split, a, b, bias)

    torch.testing.assert_close(
        out, matmul_bias_reference(a, b, bias), rtol=2e-2, atol=2e-2
    )


def test_a_zero_bias_changes_nothing(device):
    """bias 通路本身没有副作用：bias 全零时结果等于不加 bias 的纯矩阵乘。

    前两条都通过、而这条失败的情况是存在的——那条 ``Mat`` -> ``Bias`` 的转换搬运
    如果顺带改了别的东西，带 bias 的两条仍然自洽。
    """
    torch.manual_seed(2)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)
    zero = torch.zeros([1, N], device=device, dtype=torch.float16)

    with_zero_bias = _run(matmul_bias_k_split, a, b, zero)

    # 同一个 K 分段结构，不带 bias：acc_phase 样例里的那个。
    from ..acc_phase.kernel import k_split_no_phase

    plain = torch.zeros([M, N], device=device, dtype=torch.float32)
    k_split_no_phase(a, b, plain)
    torch.npu.synchronize()

    torch.testing.assert_close(with_zero_bias, plain, rtol=2e-2, atol=2e-2)
