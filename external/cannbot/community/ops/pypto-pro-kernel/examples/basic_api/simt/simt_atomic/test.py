# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""九个原子族：七个顺序无关，两个不是。

**写法：原始。**

## 一、原子操作唯一值得验的形状是"所有线程打同一个地址"

各打各的地址的话普通写也"对"，原子性一点都验不到。所以这一例让 32 个线程全打同一个槽。

## 二、本例 INT32 的七个顺序无关运算与两个例外

| 算子 | 最终值顺序无关吗 | 判据 |
| --- | --- | --- |
| ``atomic_add`` / ``atomic_sub`` | **是** | 等于所有贡献之和 / 负和 |
| ``atomic_max`` / ``atomic_min`` | **是** | 等于候选集的极值 |
| ``atomic_and`` / ``atomic_or`` / ``atomic_xor`` | **是** | 等于位运算的归约 |
| ``atomic_exch`` | **否** | 只断言最终值**落在候选集里** |
| ``atomic_cas`` | **否** | 只断言**恰好一个赢家** |

后两个不写"具体是谁"——那是时序。本章 ``datamove/strided_view`` §六 的教训：
**把时序相关的东西写进判据，会得到一条偶发失败的测试，而那比没有测试更糟。**

## 三、`atomic_add` 的返回值证明原子性（这一例最有力的判据）

`atomic_add` 返回**加之前的旧值**。让 32 个线程各加 1，收回来的 32 个旧值必须**恰好是
`0..31` 的一个排列**：

- 少一个数 → 有更新丢了（两个线程读到同一个旧值，一次加法被覆盖）；
- 多一个重复 → 同一个旧值被两个线程看到，同上；
- 排列成立 → 32 次加法每一次都看到了前一次的结果。

**这比看最终值有力**：最终值等于 32 也可能是别的原因（比如硬件把 32 次加法合并成一次加 32）。
排列判据把"每一次都真的发生了"钉住。

## 四、输入怎么挑

- `max` / `min` 的候选值用 `(tid * 7) % 32 + 1`——**32 的一个置换**（7 与 32 互质），
  覆盖 `1..32` 但**不单调**。单调的话"取最大"和"取最后一个"分不开。
- 位掩码是 `(1 << (tid % 5)) | 0x100`。那个**公共位 `0x100` 不可省**：没有它 `and` 的结果是 0,
  而 0 可能来自任何地方（初值没设对、算子没执行、掩码全是 0）。有了它期望值是 `0x100`。
- `and` 的初值必须是全 1（INT32 的 `-1`），`min` 的初值必须比所有候选都大——
  **初值挑错判据就失去意义**。

## 五、如何独立验证

1. 七个顺序无关的算子各自等于自己的归约值。
2. `atomic_exch` 的最终值 ∈ `{1..32}`，**不是 0**（初值）。
3. `atomic_cas` 的最终值 ∈ `{1..32}`，**不是 0**：恰好一个线程看到了 `compare=0`。
4. `atomic_add` 的 32 个返回值排序后等于 `0..31`。
5. 操作数不 cast 到目标 dtype 时被解析期拒（**不上板**）。

## 六、启动方式是这一例语义的一部分

**必须 `[None, 1]`。** 不带启动括号的话 56 个 block 各跑一遍，每个累加槽被打 56 倍。

本仓栽过一次（`references/repeated-mistakes.md` #9）：`fixpipe_riders` 的 atomic 测试直接
`kernel(...)` 调用，拿到期望值的 28 倍，而第一反应是去查 atomic 的语义。
**累加类的东西，先回答"几个 block 在跑"。**

## 七、操作数的 dtype 必须和目标完全一致

`tid + 1` 和 Python 字面量在 SIMT 里都是 **INT64**。目标是 INT32 计数器时解析期直接拒：

```
simt.atomic_add operand 0 dtype must match target dtype int32, but got int64
```

`simt_ops.py:578`，消息带源码 span 和两边的 dtype 名字——**很准，撞上不用猜**。
所以这一例里每个标量操作数都过一次 `pl.simt.cast`；从 INT32 张量读出来的值不用再 cast。

**写这一例时 10 条判据一起红**，全是这一条。值得记的是症状：**报错指向第一个 atomic 调用**,
而真正的原因是"那个 `+ 1`"。同一条规则在 ``simt_ring_fence`` 里以 `limit` 的形式又出现一次。
两处都记进了 [`api/constraints.md`](../../../../references/api/constraints.md)。

## 八、这一例不讲什么

- **`atomic_inc` / `atomic_dec`**——它们带 `limit` 参数、语义是**环形**增减，和这九个不同,
  在 ``simt_ring_fence``。
- **`threadfence` 系**——同上。
- **非 INT32 的原子**——ST 里 `atomic_add` 覆盖了 INT32/UINT32/FP16/BF16/FP32 五种，
  这一例只测 INT32。**明说跳过**：dtype 矩阵是 ST 的职责，这一例要讲的是顺序无关性。
- **跨 block 的原子**——只有一个 block 在跑（见 §六）。跨 block 的争用是另一个题目。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 三节。
"""

import numpy as np
import pypto_pro as pp
import pytest
import torch
from pypto_pro._errors import CommonExternal

from . import reference as ref

from .kernel import (
    SLOT_ADD,
    SLOT_ADD_RET,
    SLOT_AND,
    SLOT_CAS,
    SLOT_EXCH,
    SLOT_MAX,
    SLOT_MIN,
    SLOT_OR,
    SLOT_SUB,
    SLOT_XOR,
    SLOTS,
    THREADS,
    operand_dtype_mismatch,
    simt_nine_atomic_families,
)

_SLOT_OF = {
    "add": SLOT_ADD,
    "sub": SLOT_SUB,
    "max": SLOT_MAX,
    "min": SLOT_MIN,
    "and": SLOT_AND,
    "or": SLOT_OR,
    "xor": SLOT_XOR,
}


def _run(device):
    state = torch.from_numpy(ref.initial_state(SLOTS)).to(device)
    vals = torch.from_numpy(ref.make_values()).to(device)
    returns = torch.full((1, THREADS), -1, dtype=torch.int32, device=device)
    simt_nine_atomic_families[None, 1](state, vals, returns)
    torch.npu.synchronize()
    return state.cpu().numpy().ravel(), returns.cpu().numpy().ravel()


@pytest.mark.parametrize("op", ["add", "sub", "max", "min", "and", "or", "xor"])
def test_the_order_independent_families_reach_the_reduction(device, op):
    """验证 1。七个算子各自等于自己的归约值——**和线程完成的顺序无关**。"""
    state, _ = _run(device)
    want = ref.expected_order_independent()[op]
    assert int(state[_SLOT_OF[op]]) == want, (
        f"`pl.simt.atomic_{op}` 最终值 {int(state[_SLOT_OF[op]])}，期望 {want}"
    )


def test_exchange_leaves_one_of_the_thread_values(device):
    """验证 2。`atomic_exch` 的最终值 ∈ `{1..32}`，**不是初值 0**。

    **不断言具体是哪个**：那取决于哪个线程最后换进去，是时序。能断言的是"每个线程都换过一次",
    其后果就是最终值一定不是初值。
    """
    state, _ = _run(device)
    final = int(state[SLOT_EXCH])
    assert final != 0, "最终值还是初值 —— 一次交换都没发生？"
    assert final in ref.exch_candidates(), f"最终值 {final} 不是任何线程的值"


def test_compare_and_swap_has_exactly_one_winner(device):
    """验证 3。初值等于 `compare`，所以**恰好一个线程**能赢下这次交换。

    赢家是谁取决于时序，但"有且只有一个赢家"不是——判据落在这一点上：最终值是某个线程的值
    （说明有人赢了），而且它不是初值（说明这次交换真的发生了）。
    """
    state, _ = _run(device)
    final = int(state[SLOT_CAS])
    assert final != 0, "没有线程赢下 cas —— compare 值和初值不匹配？"
    assert final in ref.cas_candidates(), f"最终值 {final} 不是任何线程的值"


def test_the_returned_old_values_form_a_permutation(device):
    """验证 4，**这一例最有力的判据**：32 个返回的旧值排序后正好是 `0..31`。

    少一个数说明有更新丢了，多一个重复说明两个线程读到了同一个旧值。
    **只看最终值证不到这一步**——最终值等于 32 也可能是硬件把 32 次加法合并成了一次。
    """
    state, returns = _run(device)
    np.testing.assert_array_equal(np.sort(returns), ref.add_return_permutation(THREADS))
    assert int(state[SLOT_ADD_RET]) == THREADS, (
        f"最终值 {int(state[SLOT_ADD_RET])}，期望 {THREADS}：返回值是排列但最终值不对？"
    )


def test_an_int64_operand_is_refused():
    """验证 5。`tid + 1` 是 INT64，目标是 INT32。**不上板**——解析期就拒了。

    判据匹配两边的 dtype 名字，不匹配错误码。
    """
    with pytest.raises(CommonExternal) as raised:
        operand_dtype_mismatch.to_kernel_def().parse_target_program(
            pp.ir.SectionKind.Vector
        )
    message = str(raised.value)
    assert "must match target dtype int32" in message, message
    assert "int64" in message, message
