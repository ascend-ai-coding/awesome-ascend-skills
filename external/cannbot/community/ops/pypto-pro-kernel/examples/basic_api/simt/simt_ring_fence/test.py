# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""环形计数（`atomic_inc` / `atomic_dec`）与两个内存栅栏。

**写法：原始。**

## 一、环形计数的周期是 `limit + 1`，不是 `limit`（实测）

计数器在 `0 .. limit` 之间转，**含两端**：

| | 规则 | 第 k 次（k 从 0 起）拿到的旧值 |
| --- | --- | --- |
| ``atomic_inc(t, limit)`` | `old >= limit ? 0 : old + 1` | `k mod (limit + 1)` |
| ``atomic_dec(t, limit)`` | `old == 0 ? limit : old - 1` | `(-k) mod (limit + 1)` |

`limit=6`、32 个线程（周期 7）实测：`inc` 最终值 **4** = `32 mod 7`，`dec` 最终值 **3**。
**这一组参数能把两种候选规则分开**：周期 7 给 4，周期 6 给 2。`limit` 取成和线程数互素
是刻意的——整除的话最终值回到初值，"转了几圈"就验不出来。

## 二、`limit` 的 dtype 必须和目标一致

目标是 UINT32 计数器时，`limit` 写成 Python 字面量会被拒（字面量是 INT64）：

```
simt.atomic_inc operand 0 dtype must match target dtype uint32, but got int64
```

所以这一例的 `limit` 从一个 UINT32 张量里读。**这条消息很准**，撞上不用猜。

## 三、判据落在返回值的**多重集**上

线程完成的顺序不确定，所以"第几个线程拿到几"不能写进判据。但**"哪些旧值各出现几次"是确定的**：

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `inc` 的旧值出现次数 | 5 | 5 | 5 | 5 | 4 | 4 | 4 |
| `dec` 的旧值出现次数 | 5 | 4 | 4 | 4 | 5 | 5 | 5 |

两行都和实测逐个吻合。**这比只看最终值有力**：最终值对了也可能是别的路径
（比如 32 次操作被合并），多重集要求每一次都真的发生过，而且每次都看到了上一次的结果。

## 四、如何独立验证

1. `atomic_inc` 的最终值 = `32 mod 7 = 4`。
2. `atomic_dec` 的最终值 = `(-32) mod 7 = 3`。
3. `inc` 的 32 个返回值排序后等于 `[k mod 7 for k in 0..31]` 排序。
4. `dec` 的 32 个返回值排序后等于 `[(-k) mod 7 for k in 0..31]` 排序。
5. 两个栅栏加进一条正确的链里，结果不变。
6. `limit` 写成字面量被解析期拒（dtype 不匹配）。
7. `syncthreads` 放在运行期 `if` 里**不被拒**（见 §六）。
8. `threadfence_block` 放在运行期 `if` 里**允许**（docstring 说的）。

## 五、栅栏本身没有验到，这是能力边界

两个栅栏「只排序、不等待」（docstring：「does not wait for other threads」）。它们排序的是
**这个线程自己的内存操作对别人的可见顺序**——要验得构造生产者／消费者：一个线程写数据再写标志,
另一个线程自旋等标志再读数据，没有栅栏时后者可能先看到标志。

**没做**，理由是自旋等待在 SIMT 里有挂死风险（线程分叉 + 没有前进保证），在共享机上不值当。
验证 5 能证明的只有「**加了栅栏不会把对的弄坏**」——这一点明说，不假装它验到了可见性顺序。

## 六、`syncthreads` 的位置约束：docstring 写了，解析期不拦

docstring 原话：「The barrier must be reached uniformly by every thread in the block.
It **cannot** be placed inside runtime ``if``, ``for``, or ``while`` control flow.
This no-return operation must be used as a standalone statement.」

实测三条**全都不拦**：

| 写法 | docstring | 解析期 |
| --- | --- | --- |
| `syncthreads()` 在顶层 | 允许 | 通过 |
| `syncthreads()` 在运行期 `if` 里 | **不允许** | **通过** |
| `x = pl.simt.syncthreads()` 当表达式 | **不允许** | **通过** |
| `threadfence_block()` 在运行期 `if` 里 | 允许 | 通过 |

**第二行是危险的那一条**：一部分线程到了屏障、另一部分跳过，剩下的会等一个永远不来的信号
——症状是挂死，而且解析、编译都不吭声。

**这个 kernel 只解析，绝不上板。** 在共享机上试一个会挂死的东西不值当，而且挂死之后可能要重置卡。
验证 7 断言的是「**没有检查**」——框架哪天加上了，那条会红。

落点按 [skill constraints](../../../../SKILL.md#constraints-and-workspace-policy)：**文档纠正不了、症状是挂死、静态可判 → 进
skill 的 `scripts/`**。已经是 `python scripts/check.py rules` 的第六条规则
`syncthreads-must-be-uniform`；缺口本身记在
[`api/constraints.md`](../../../../references/api/constraints.md) 与
[`api/simt.md`](../../../../references/api/simt.md)。

## 七、这一例不讲什么

- **其余九个原子族**——在 ``simt_atomic``，那一例讲的是顺序无关性。
- **`syncthreads` 的正常用法**——在 ``simt_launch``（读邻居的槽）。
- **跨 block 的可见性**——`threadfence` 是 device-wide 的，但这一例只有一个 block。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 三节。
"""

import numpy as np
import pytest
import torch

import pypto_pro as pp
from pypto_pro._errors import CommonExternal

from . import reference as ref
from .kernel import (
    COUNTER_DEC,
    COUNTER_INC,
    COUNTERS,
    LIMIT,
    RET_DEC,
    RET_INC,
    RET_ROWS,
    THREADS,
    fence_inside_a_branch,
    limit_dtype_mismatch,
    simt_fences,
    simt_ring_counters,
    syncthreads_inside_a_branch,
)


def _run_ring(device):
    counter = torch.zeros((COUNTERS, 1), dtype=torch.int32, device=device).to(
        torch.uint32
    )
    limits = torch.full((1, 1), LIMIT, dtype=torch.int32, device=device).to(
        torch.uint32
    )
    ret = torch.full((RET_ROWS, THREADS), -1, dtype=torch.int32, device=device)
    simt_ring_counters[None, 1](counter, limits, ret)
    torch.npu.synchronize()
    return counter.cpu().to(torch.int32).numpy().ravel(), ret.cpu().numpy()


def _parse(kernel):
    kernel.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)


def test_the_increment_wraps_at_limit_plus_one(device):
    """验证 1。`32 mod 7 = 4`。

    **这条参数选得能把两种候选规则分开**：周期 `limit+1=7` 给 4，周期 `limit=6` 给 2。
    """
    counter, _ = _run_ring(device)
    assert int(counter[COUNTER_INC]) == ref.inc_final(), (
        f"inc 最终值 {int(counter[COUNTER_INC])}，期望 {ref.inc_final()}"
        f"（周期 {ref.period()} = limit + 1）"
    )


def test_the_decrement_wraps_the_other_way(device):
    """验证 2。从 0 往下减，下一个是 `limit`，所以 32 次之后是 `(-32) mod 7 = 3`。"""
    counter, _ = _run_ring(device)
    assert int(counter[COUNTER_DEC]) == ref.dec_final(), (
        f"dec 最终值 {int(counter[COUNTER_DEC])}，期望 {ref.dec_final()}"
    )


def test_the_returned_old_values_form_the_expected_multiset(device):
    """验证 3 / 4，**这一例最有力的判据**。

    线程完成的顺序不确定，所以比的是**多重集**而不是逐位。它要求 32 次操作每一次都真的发生过、
    而且每次都看到了上一次的结果——只看最终值证不到这一步。
    """
    _, ret = _run_ring(device)
    np.testing.assert_array_equal(np.sort(ret[RET_INC]), ref.inc_returns())
    np.testing.assert_array_equal(np.sort(ret[RET_DEC]), ref.dec_returns())


def test_adding_fences_does_not_break_a_correct_chain(device):
    """验证 5。两个栅栏加进一条靠 `syncthreads` 定序的链里，结果不变。

    **这条不验证栅栏做了什么**（§五）：它排序的是可见性，要验得构造生产者／消费者 + 自旋等待,
    而那在 SIMT 里有挂死风险。能证明的只有"加了不会把对的弄坏"。
    """
    out = torch.full((1, THREADS), -1, dtype=torch.int32, device=device)
    simt_fences[None, 1](out)
    torch.npu.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy()[0], ref.rotated(THREADS))


def test_a_literal_limit_is_refused():
    """验证 6。字面量是 INT64，和 UINT32 的目标不匹配。

    判据匹配规则文本里的 dtype 名字，**不匹配错误码**。
    """
    with pytest.raises(CommonExternal) as raised:
        _parse(limit_dtype_mismatch)
    message = str(raised.value)
    assert "must match target dtype uint32" in message, message
    assert "int64" in message, message


def test_syncthreads_inside_a_branch_is_not_refused():
    """验证 7。docstring 说不能放在运行期控制流里，**解析期不拦**。

    **断言的是"没有检查"**——框架哪天加上了，这条会红，那正是我们想收到的通知。
    这个 kernel **绝不上板**：非一致的屏障会挂死（§六）。

    确定性检查已经落在 `python scripts/check.py rules` 的 `syncthreads-must-be-uniform`。
    """
    _parse(syncthreads_inside_a_branch)  # 不抛就是这条测试的全部内容


def test_a_fence_inside_a_branch_is_allowed():
    """验证 8。栅栏在运行期 `if` 里是**允许的**（docstring 明确说了）。

    **和验证 7 放在一起才有意义**：它说明 docstring 区分的是"屏障"和"栅栏"，
    不是笼统地禁止控制流。少了这一条，`syncthreads` 那条看起来像"控制流里什么都不能放"。
    """
    _parse(fence_inside_a_branch)
