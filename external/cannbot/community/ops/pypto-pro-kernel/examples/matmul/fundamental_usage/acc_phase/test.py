# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""``AccPhase`` / ``STPhase``：unitFlag，不是累加控制。

**先纠一个名字.** ``AccPhase`` 里的 Acc 指的是 **L0C 累加器这块存储**，不是「累加
这个动作」。它和 ``pl.matmul_acc`` 是**两件无关的事**，凑在同一个 K 分段循环里出现
才容易被看成一件：

| | 管什么 | 不配会怎样 |
|---|---|---|
| ``pl.matmul`` vs ``pl.matmul_acc`` | **算什么**：覆盖写 vs 累加 | 结果错（少加了前几段） |
| ``phase=AccPhase.*`` | **怎么并行**：硬件 unitFlag | 结果不变，慢一点；**配错会卡死** |

把 ``kernel.py`` 里两个 kernel 的 ``matmul`` / ``matmul_acc`` 摘出来对比，会看到
这一维两边**完全一样**——不同的只有 phase。

**unitFlag 是什么.** 硬件给 L0C Buffer 里**每一个 512B 内存块**一个标志位，表示这块
能不能被读写，从而在硬件层面做 L0C 的读写同步。它让 Mmad（矩阵计算）和 Fixpipe
（搬出）能细粒度并行：不必等整块算完才开始搬。

两个取值的差别就一句话：

| | 检查标志位并据此控制写 | 执行后 |
|---|---|---|
| ``Partial`` | 是 | **不改**标志位 |
| ``Final`` | 是 | 把标志位**置 1** |
| ``Unspecified`` | 否（不启用） | — |

所以 ``Final`` 是「我写完了，这块可以搬了」的那一次。K 分三段就是
``Partial, Partial, Final``。

**物理访问：配对规则.** 这条是硬性的，写在
AccPhase 的 API 页 (`pypto/docs/zh/api/pro_api/SIMD-API/basic_data_structures/AccPhase.md`)
和 STPhase 的 (`pypto/docs/zh/api/pro_api/SIMD-API/basic_data_structures/STPhase.md`)：

- ``matmul`` / ``matmul_acc`` / ``matmul_mx`` / ``matmul_mx_acc`` 配了 ``AccPhase``，
  对应的 ``store`` / ``store_tile`` / ``move`` / ``insert`` **必须**配 ``STPhase``。
- 矩阵计算的**最后一次写**用 ``AccPhase.Final``；结果的**最后一次搬出**用
  ``STPhase.Final``。同一份结果搬多次时，最后一次之前用 ``STPhase.Partial``。
- 配不上**可能卡死**（官方原话「否则可能出现卡死现象」），不是算错。

**两条额外约束.**

- ``store`` / ``store_tile`` 的 ``STPhase`` **不能**与 Tile 类型的 scale 同时使用。
- A5 独有：Atlas A2 / A3 系列不支持。

**这个样例为什么不放负例.** 配错的失败形态是**死锁**，不是错数。按
[tooling/examples.md](../../../../references/tooling/examples.md) 的准入规则，会死锁的负例不放这里，要先在模型
里跑。所以上面那两条只是记下来，没有可执行的反例。

**为何这样同步.** 两个 kernel 均显式声明地址与深度，使用 ``mutex_ids="auto"``，
由 ``auto_mutex`` 插入核内同步。资源声明见 ``gemm_vs_matmul`` 样例。unitFlag 是**硬件层**的读写同步，
和 mutex 那一层不冲突也不替代：mutex 管的是 Tile 槽位之间的依赖，unitFlag 管的是
同一块 L0C 内部 512B 粒度的 Mmad/Fixpipe 重叠。

**如何独立验证.** 两条：

1. 开 phase 与不开 phase 的两个 kernel ``rtol=0, atol=0`` 逐位比较。这条把「phase
   是性能机制」钉成事实——一旦出现数值差异，要么是 unitFlag 影响了结果（那就和文档
   矛盾），要么是我把两个 kernel 写得不等价了。
2. 两版都与参考实现比。参考整个 K 一次归约，不分段——只比第 1 条会漏掉「两边都少
   加了一段」这种错法，而那恰好是 ``matmul`` / ``matmul_acc`` 用反时的表现。

源于 ``pypto/python/tests/st/pypto_pro/frontend/matmul/test_matmul_bias.py`` 的
``kernel_k_split_bias``（这里去掉了 bias，那是 ``bias_fusion`` 样例的事），
以及 ``test_matmul_hf8_fp32.py`` 里 ``make_matmul_no_phase`` 与带 phase 版本的对照
手法。
"""

import torch

from .kernel import K, M, N, k_split_no_phase, k_split_with_phase
from .reference import matmul_reference


def _run(kernel, a, b):
    out = torch.zeros([M, N], device=a.device, dtype=torch.float32)
    kernel(a, b, out)
    torch.npu.synchronize()
    return out


def _inputs(device, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn([M, K], device=device, dtype=torch.float16),
        torch.randn([K, N], device=device, dtype=torch.float16),
    )


def test_phase_does_not_change_the_numbers(device):
    """unitFlag 是性能机制：开与不开必须逐位相同。"""
    a, b = _inputs(device)

    with_phase = _run(k_split_with_phase, a, b)
    no_phase = _run(k_split_no_phase, a, b)

    torch.testing.assert_close(with_phase, no_phase, rtol=0, atol=0)


def test_both_accumulate_all_three_k_blocks(device):
    """确认两边都把三段都加上了——只比上一条会漏掉「两边都少加一段」。"""
    a, b = _inputs(device, seed=1)

    reference = matmul_reference(a, b)
    torch.testing.assert_close(
        _run(k_split_with_phase, a, b), reference, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        _run(k_split_no_phase, a, b), reference, rtol=2e-2, atol=2e-2
    )


def test_the_last_write_is_the_only_final(device):
    """把配对规则钉在源码上：Final 恰好各一个，Partial 是其余的。

    这条不上板，读的是源码。规则写在注释里会随注释腐烂，写成断言不会。
    """
    import inspect

    from . import kernel as kernel_mod

    # 读模块源码，不是 kernel 对象——@pl.jit 之后那个名字绑的是 _TileJitKernel，
    # inspect.getsource 不认。
    src = inspect.getsource(kernel_mod)
    src = src[
        slice(src.index("def k_split_with_phase"), src.index("def k_split_no_phase"))
    ]
    assert src.count("AccPhase.Final") == 1, "最后一次写 L0C 才用 Final"
    assert src.count("AccPhase.Partial") == 2, "K 分三段，前两段是 Partial"
    assert src.count("STPhase.Final") == 1, "结果只搬出一次，那一次是 Final"
    # 配了 AccPhase 就必须配 STPhase。
    assert "AccPhase" in src and "STPhase" in src
