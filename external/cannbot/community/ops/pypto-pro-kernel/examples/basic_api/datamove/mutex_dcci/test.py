# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""手工上锁（`mutex_lock` / `mutex_unlock`）与数据 cache 清理（`dcci`）。

**写法：原始。**

## 一、手工 mutex 就是 `auto_mutex` 帮你插的那两条 op

`@pl.jit(auto_mutex=True)` + `pl.make_tile_group(mutex_ids=...)` 在解析后生成的正是
`system.mutex_lock_dyn` / `system.mutex_unlock_dyn`。这一例把同样的 op 自己写出来，
验证 3 用 IR 比对钉住"是同一种 op"。

**方向不写在参数里。** `mutex_lock(pipe=P, mutex_id=n)` 让流水 `P` 停下来，直到上一个持有 `n`
的人 `mutex_unlock`；「谁等谁」全靠 `mutex_id` 的先后持有表达。flag 那一套把方向写在
`set_pipe` / `wait_pipe` 里，**读起来更难写错**——这是两套机制的实际差别，不只是风格。

`pipe` 是**要上锁的那条流水，不是要等的那条**。

## 二、三个 id 池子互不相干

| | 池子 | 定义处 |
| --- | --- | --- |
| flag 的 `event_id` | `[0, 7]` | `MAX_FLAG_EVENT_ID` |
| 跨核的 `event_id` | `[0, 15]` | `MAX_EVENT_ID` |
| mutex 的 `mutex_id` | `[0, 31]` | `MAX_MUTEX_ID` |

`system_ops.py:38-41` 的注释写明这三行是「the single definition of each limit」，扫流水的 pass
和 `make_tile_group` 的 parser 都 import 它们而不是各自重写一遍。**所以这三个数字有单一来源，
不要在别处硬编码。**

## 三、如何独立验证

1. ``manual_mutex_chain``：`out == x * 2`，三段共用一个 `mutex_id` 严格串行。
2. ``move_sync.chain_auto_mutex``：与验证 1 **逐位相同**，复用自动同步对照。
3. 两者的 IR 里都有 `mutex_lock_dyn` / `mutex_unlock_dyn`，**手工那条一条 flag op 都没有**
   （不上板）。这条钉住"同一种 op"，也顺带钉住手工写法真的落到了 mutex 路径上。
4. ``dcci_on_tile`` / ``dcci_on_tensor`` / ``dcci_single_cache_line`` /
   ``dcci_explicit_dst``：四种参数形态，结果都仍是 `x * 2`。
5. `mutex_id=32` 被解析期拒，范围消息是 `[0, 31]`。
6. `mutex_lock` 不配 `mutex_unlock` **解析期通过**（不上板，见 §六）。

**第 4 条是这一例最弱的判据，必须说清它证明了什么**，见 §五。

## 四、这一例不讲什么

- **`auto_mutex` 的适用边界**——它对裸 `pl.make_tile` 一条同步都不插，那件事在
  [`diagnostics/validation-traps.md`](../../../../references/diagnostics/validation-traps.md) #7 与 ``dma_pad`` §七。
- **flag 同步**——`sync_src` / `sync_dst` 的完整表面（合法路径表、AIV/AIC 差别）在 ``move_sync``。
- **`bar_*`**——流水内屏障，本章刻意不收，理由见 ``move_sync`` §六。

## 五、`dcci` 没有数值判据，这是能力边界不是偷懒

`dcci` 是 Data Cache Clean and Invalid。**它的效果在单核、单进程、同一份数据的往返里不可观测**：

- 清 UB tile 的 cache（`DcciDst.CACHELINE_UB`）之后再读，读到的仍是同一份数据——正确的程序
  在清与不清两种情况下结果相同，这正是 cache 该有的性质。
- 清 GM 的 cache（`CACHELINE_OUT`）要在**另一个观察者**（另一个核、host 侧的直读）看得见差别
  的场景里才有意义，那需要跨核或 host/device 一致性的构造，不是这一章的题目。

所以验证 4 断言的是「**加了这一条不会把对的弄坏**」，以及四种参数形态都被接受。
**不假装它验到了 cache 行为。** 硬凑一个数值判据只会得到一条以错误的理由通过的测试
（`references/verification/evidence.md` #14 是同一类）。

四种形态都过了解析与上板：`dcci(tile)`、`dcci(tensor, [0, 0])`、
`dcci(tile, cache_line=SINGLE_CACHE_LINE)`、`dcci(tile, dst=CACHELINE_UB)`。张量目标的 `offset`
可以是逐维列表或一个标量元素偏移，**tile 目标只接受标量**。

## 六、`mutex_lock` 不配 `mutex_unlock`：解析期不拦（明说跳过上板）

`mutex_lock` 的 docstring 写着「Must be paired with `mutex_unlock` on the same `pipe` and
`mutex_id`」，但**没有任何一层检查它**——验证 6 钉住这件事。

**没有上板测这个 kernel，理由是共享机。** 一个没释放的 buffer-id token 可能把那条流水的指令
队列留在阻塞态；在别人也在用的机器上试它，代价和收益不成比例。所以这一条只声明**解析期**的
事实，不声明运行期会发生什么——**这是跳过，不是"验过没问题"**。

要补这一格需要一台独占的机器，加上一个能在挂死后恢复的流程。记在这里而不是删掉，
因为「没覆盖」和「不需要覆盖」不是一回事。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 一节。
"""

import numpy as np
import pytest
import torch

import pypto_pro as pp
from pypto_pro._errors import OutOfRange

from ..move_sync.kernel import chain_auto_mutex
from .kernel import (
    COLS,
    ROWS,
    dcci_explicit_dst,
    dcci_on_tensor,
    dcci_on_tile,
    dcci_single_cache_line,
    lock_without_unlock,
    manual_mutex_chain,
    mutex_id_out_of_range,
)
from .reference import doubled, make_input


def _run(kernel, device):
    x = make_input(ROWS, COLS)
    out = torch.zeros((ROWS, COLS), device=device, dtype=torch.float32)
    kernel(torch.from_numpy(np.ascontiguousarray(x)).to(device), out)
    torch.npu.synchronize()
    return out.cpu().numpy(), x


def _parse(kernel):
    """解析并返回 IR 文本。`parse_target_program` 返回二元组，第二项这里用不到。"""
    program, _ = kernel.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)
    return pp.ir.python_print(program)


def test_the_manual_mutex_chain_is_correct(device):
    """验证 1。三段共用一个 `mutex_id`，严格串行。"""
    got, x = _run(manual_mutex_chain, device)
    np.testing.assert_array_equal(got, doubled(x))


def test_manual_and_automatic_agree(device):
    """验证 2。手工与 `auto_mutex` 逐位相同。

    比两个 kernel 的输出而不是各自比 golden：golden 相同只说明两者都算对了。
    """
    manual, _ = _run(manual_mutex_chain, device)
    automatic, _ = _run(chain_auto_mutex, device)
    np.testing.assert_array_equal(manual, automatic)


def test_both_forms_lower_to_the_mutex_ops():
    """验证 3，**不上板**：手工写法落在的确是 mutex 路径，不是 flag 路径。

    数值判据证不了这一点——两条链都算对了，但"算对了"和"走了哪条同步路径"是两件事。
    最后那条 `not in` 断言才是关键：少了它，这条测试在手工版被悄悄改成 flag 之后仍然会绿。
    """
    manual = _parse(manual_mutex_chain)
    automatic = _parse(chain_auto_mutex)

    for text in (manual, automatic):
        assert "system.mutex_lock_dyn" in text
        assert "system.mutex_unlock_dyn" in text

    assert "system.sync_src_dyn" not in manual, "手工版应该走 mutex，不是 flag"


@pytest.mark.parametrize(
    "kernel", [dcci_on_tile, dcci_on_tensor, dcci_single_cache_line, dcci_explicit_dst]
)
def test_dcci_does_not_disturb_a_correct_chain(device, kernel):
    """验证 4。三种 `dcci` 形态都被接受，且结果仍然正确。

    **这条不验证 cache 行为**，见 §五：`dcci` 的效果在单核单进程里不可观测。它验的是
    「加了这一条不会把对的弄坏」，以及四种参数形态都能过解析和上板。
    """
    got, x = _run(kernel, device)
    np.testing.assert_array_equal(got, doubled(x))


def test_a_mutex_id_above_thirty_one_is_refused():
    """验证 5。mutex 的池子是 `[0, 31]`；flag 的 `[0, 7]` 是另一个池子。"""
    with pytest.raises(OutOfRange) as raised:
        _parse(mutex_id_out_of_range)
    assert "mutex_id must be in [0, 31]" in str(raised.value), str(raised.value)


def test_an_unpaired_lock_is_not_refused_at_parse_time():
    """验证 6。docstring 说「Must be paired」，没有任何一层检查它。

    **这条断言的是"没有检查"**，所以它会在框架哪天加上检查时变红——那正是我们想收到的通知。
    没有上板测它，理由见 §六（共享机 + 可能把流水队列留在阻塞态）。
    """
    _parse(lock_without_unlock)  # 不抛就是这条测试的全部内容
