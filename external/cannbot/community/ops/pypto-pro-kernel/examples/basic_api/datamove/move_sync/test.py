# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""跨流水定序：`sync_src` / `sync_dst` 成对的 flag，以及核间的 `sync_all`。

**写法：原始。**

## 一、跨流水定序只有 flag 这一条路

`pl.load` 在 MTE2、`pl.add` 在 V、`pl.store` 在 MTE3。**默认这三段是并行的**，要定序就得说。

| 想让谁等谁 | 写法 |
| --- | --- |
| V 等 MTE2 | ``sync_src/sync_dst(set_pipe=MTE2, wait_pipe=V, event_id=n)`` |
| MTE3 等 V | ``sync_src/sync_dst(set_pipe=V, wait_pipe=MTE3, event_id=n)`` |

**两条必须成对，参数完全相同。** set 那一半在 `sync_src`，wait 那一半在 `sync_dst`；只写一条
不报错，也拦不住任何东西。

`event_id` 的池子是 `[0, 7]`（`system_ops.py` 的 `MAX_FLAG_EVENT_ID`）。跨核事件的 `[0, 15]`
（`MAX_EVENT_ID`）和 tile mutex 的 `[0, 31]`（`MAX_MUTEX_ID`）**是三个不同的池子**，别混。

## 二、两种同步手段，等价但插的 op 不同

| 写法 | 解析后 | 谁负责 |
| --- | --- | --- |
| `@pl.jit()` + 手工 flag | `sync_src_dyn` / `sync_dst_dyn` | 作者 |
| `@pl.jit(auto_mutex=True)` + `make_tile_group(mutex_ids=...)` | `mutex_lock_dyn` / `mutex_unlock_dyn` | 框架 |

验证 2 把两者的输出逐位比。**`auto_mutex` 只对组里的 tile 生效**：换成裸 `pl.make_tile`
它一条同步都不插也不报错，见 [`diagnostics/validation-traps.md`](../../../../references/diagnostics/validation-traps.md) #7。

## 三、`sync_all` 是核间的，不是流水间的

`pl.system.sync_all(core_type=...)` 管「所有核都到这儿了」。把它当流水定序用不报错也不生效
——这是两个不同层级的问题，容易混因为名字里都有 sync。

## 四、如何独立验证

1. ``chain_flag_sync``：`out == x * 2`，手工 flag。
2. ``chain_auto_mutex``：与验证 1 **逐位相同**，两种同步手段等价。
3. ``chain_with_sync_all``：链尾多一条 `sync_all(AIV_ONLY)`，结果不变。
4. `set_pipe == wait_pipe` 被解析期拒。
5. AIV 侧不存在的路径（`V → MTE1`）被拒，且消息**列出合法的 wait_pipe**。
6. 同一条 `MTE2 → V` 在 `section_cube()` 里被拒（说 `on AIC`），在 `section_vector()` 里合法。
7. `event_id=8` 被拒，范围消息是 `[0, 7]`。

**第 1 条单独几乎什么都不证明**，这一点必须说清：`pl.add` 之前不放任何同步，在冷进程里
**也会给出正确答案**（实测，见 §五）。所以这一例的判据重心在 4–7 那四条解析期规则上，
它们是确定的；第 1–3 条只保证「正确的写法不会反而出错」。

## 五、"测试绿了"不等于"同步对了"（实测）

同一条 load → add → store，只改同步手段，固定前置状态（先跑一个把两块 UB 都写成已知非零数据的
kernel），每种跑三次：

| 中间的同步 | 紧跟脏化 kernel 之后 | 冷进程单跑 |
| --- | --- | --- |
| `sync_src` / `sync_dst` 两对 | **3/3 正确** | 正确 |
| 完全不同步 | **3/3 错**（拿到上一个 kernel 留在 dst 里的数据） | **正确** |

**"完全不同步"在冷进程里是正确的。** 如果只在冷进程里跑一次，缺同步这件事根本不会暴露。
这就是为什么本章的判据不依赖数值：数值只在前置状态被控制住时才有话语权。

## 六、为什么这一例不收 `bar_*`（六个屏障）

**`bar_*` 是流水内的屏障，不是跨流水屏障**（用户 2026-09-28 指出）。它排空**一条**流水，
用在自动同步分析看不见依赖、需要强制流水内串行的少数场合。

用它做跨流水定序会**静默算错**。同一张表里：

| 中间的同步 | 紧跟脏化 kernel 之后 |
| --- | --- |
| `bar_mte2()` 后接 `bar_mte3()` | **3/3 错** |

两个真实用法（`easyasc`，同一批硬件的上一代）：

- `kernels/a2/vec_only/mxfp8e5m2.py:185`——`bar_mte3()` 在循环体**尾部**，把 MTE3 排到自己后面，
  下一轮复用同一块 UB scratch 之前先让这一轮的搬出落地。
- `kernels/a2/vec_only/dflash_anchor_mask/*.py:180`——注释写得最直白：自动同步按 DBuff 槽位
  建依赖，跨两个循环时看不见，于是 `bar_all()` 强制排空。「**The barrier drains it.**」

**不收录的理由**：一个"用 `bar_*` 同步"的样例会教错人，而正确的用法（流水内串行）需要一个
自动同步分析看不见依赖的场景才讲得清，那不是这一章的题目。事实记在
[`diagnostics/validation-traps.md`](../../../../references/diagnostics/validation-traps.md) #8。

## 七、这一例还不讲什么

- **`sync_all(MIX)` / `sync_all(AIC_ONLY)` 不上板。** 两者解析都通过，但本例是纯 vector
  kernel，`MIX` 按文档要连 AIC 一起同步——**这是明说跳过，不是"验过没问题"**：在共享机上
  试一个可能挂死的组合不值当。要覆盖它得有真正的 cube + vector 双侧 kernel。
- **跨核交接**——`set_cross_core` / `wait_cross_core` 在 `fixpipe/` 几例里已覆盖，
  设计在 `references/api/crosscore-contract.md`。
- **手工 mutex**——`mutex_lock` / `mutex_unlock` 在 ``mutex_dcci``。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 一节。
"""

import numpy as np
import pytest
import torch

import pypto_pro as pp
from pypto_pro._errors import InvalidOperation, NotSupported, OutOfRange

from .kernel import (
    COLS,
    ROWS,
    chain_auto_mutex,
    chain_flag_sync,
    chain_with_sync_all,
    event_id_out_of_range,
    path_only_legal_on_aiv,
    same_pipe_both_sides,
    unsupported_aiv_path,
)
from .reference import doubled, make_input


def _run(kernel, device):
    x = make_input(ROWS, COLS)
    out = torch.zeros((ROWS, COLS), device=device, dtype=torch.float32)
    kernel(torch.from_numpy(np.ascontiguousarray(x)).to(device), out)
    torch.npu.synchronize()
    return out.cpu().numpy(), x


def _parse(kernel, kind=pp.ir.SectionKind.Vector):
    """解析并返回 IR 文本。`parse_target_program` 返回的是二元组，第二项这里用不到。"""
    program, _ = kernel.to_kernel_def().parse_target_program(kind)
    return pp.ir.python_print(program)


def test_the_flag_paired_chain_is_correct(device):
    """验证 1。两对 flag 把 MTE2 → V → MTE3 串起来。

    **这条通过不代表同步是必需的**：不放同步的同一条链在冷进程里也对（§五）。它保证的只是
    「正确的写法不会反而出错」。这一例真正确定的判据是下面四条解析期规则。
    """
    got, x = _run(chain_flag_sync, device)
    np.testing.assert_array_equal(got, doubled(x))


def test_auto_mutex_reaches_the_same_result(device):
    """验证 2。两种同步手段等价，输出逐位相同。

    比两个 kernel 的输出而不是各自比 golden：golden 相同只说明两者都算对了，
    **逐位相同才是"等价"**。
    """
    manual, _ = _run(chain_flag_sync, device)
    automatic, _ = _run(chain_auto_mutex, device)
    np.testing.assert_array_equal(manual, automatic)


def test_the_two_forms_lower_to_different_ops():
    """验证 2 的另一半，**不上板**：等价不等于一样。

    手工那条插 `sync_src_dyn` / `sync_dst_dyn`，`auto_mutex` 那条插
    `mutex_lock_dyn` / `mutex_unlock_dyn`。这条钉住「哪种写法生成哪种 op」，也顺带钉住
    `auto_mutex` 在组上**确实**生效了——数值判据证不了这一点（缺同步也可能碰巧对）。
    """
    manual = _parse(chain_flag_sync)
    automatic = _parse(chain_auto_mutex)

    assert "system.sync_src_dyn" in manual and "system.sync_dst_dyn" in manual
    assert "system.mutex_lock_dyn" not in manual

    assert (
        "system.mutex_lock_dyn" in automatic and "system.mutex_unlock_dyn" in automatic
    )
    assert "system.sync_src_dyn" not in automatic


def test_sync_all_does_not_disturb_the_chain(device):
    """验证 3。核间同步加在链尾，结果不变。

    它验的是「多这一条不会把对的弄坏」。`sync_all` 真正的作用（所有 AIV 核对齐）在单 block
    的样例里观察不到——**这一点明说，不假装这条测试验到了它**。
    """
    got, x = _run(chain_with_sync_all, device)
    np.testing.assert_array_equal(got, doubled(x))


def test_the_same_pipe_on_both_sides_is_refused():
    """验证 4。自己等自己，`system_ops.py:130` 单独一条。"""
    with pytest.raises(InvalidOperation) as raised:
        _parse(same_pipe_both_sides)
    assert "set_pipe and wait_pipe must differ" in str(raised.value), str(raised.value)


def test_a_path_missing_from_the_aiv_table_is_refused():
    """验证 5。`V → MTE1` 在 AIV 侧不存在，而且消息**把合法值列全**。

    第二条断言比第一条值钱：它要求这个诊断**可操作**。只断言"被拒了"的话，消息退化成
    "unsupported" 也一样绿。
    """
    with pytest.raises(NotSupported) as raised:
        _parse(unsupported_aiv_path)
    message = str(raised.value)
    assert "unsupported A5 synchronization path on AIV" in message, message
    for legal in ("PipeType.MTE2", "PipeType.MTE3", "PipeType.S"):
        assert legal in message, message


def test_the_same_path_is_legal_on_aiv_and_refused_on_aic():
    """验证 6，这一例最值得记的一条：**合法性取决于这段代码落在哪一侧**。

    `MTE2 → V` 在 vector section 里是 `chain_flag_sync` 的第一对 flag，完全合法；同一行放进
    `section_cube()` 就被拒，因为 cube 侧没有 V 流水。消息会说 `on AIC` 并列出 AIC 的表。
    """
    _parse(chain_flag_sync)  # 同一条路径在 AIV 侧：通过

    with pytest.raises(NotSupported) as raised:
        _parse(path_only_legal_on_aiv, pp.ir.SectionKind.Cube)
    message = str(raised.value)
    assert "on AIC" in message, message
    assert "PipeType.MTE2 -> PipeType.V" in message, message


def test_an_event_id_above_seven_is_refused():
    """验证 7。flag 的池子是 `[0, 7]`；跨核事件的 `[0, 15]` 是另一个池子。"""
    with pytest.raises(OutOfRange) as raised:
        _parse(event_id_out_of_range)
    assert "event_id must be in [0, 7]" in str(raised.value), str(raised.value)
