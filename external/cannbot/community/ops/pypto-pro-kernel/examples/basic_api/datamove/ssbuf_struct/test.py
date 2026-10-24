# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""SSBUF：把结构化元数据从 vector 侧交给 cube 侧。

**写法：原始。**

## 一、SSBUF 传的是元数据，不是数据

数据走 GM↔UB↔L1；SSBUF 传形状、偏移、块号这类**小而有结构**的标量。一次交接固定四步：

```python
with pl.section_vector():
    msg.batch = 8                                  # 1. vector 侧填字段
    if pl.get_subblock_idx() == 0:                 # 2. 只让一个 subblock 写
        pl.ssbuf_store(msg, 0)
        pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=15)   # 3. 发跨核事件
with pl.section_cube():
    pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=15,      # 4. cube 侧等
                              sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    pl.ssbuf_load(msg, 0)
```

第 2 步不是可选的：一个 vector block 有多个 subblock，都去写同一个偏移就是多个写者抢同一块
地址。第 3、4 步也不是可选的：**没有跨核事件，cube 侧读到的是什么完全看时序。**

跨核事件的 `event_id` 池子是 `[0, 15]`（`MAX_EVENT_ID`），和 flag 的 `[0, 7]` 是**两个池子**。

## 二、`pl.struct` 与 `pl.make_tuple` 的分界

| | 生成 C++ struct | 字段可写 | 能进 SSBUF |
| --- | --- | --- | --- |
| ``pl.struct("Name", f=0, ...)`` | 是 | 是 | **是** |
| ``pl.make_tuple(f=expr, ...)`` | 否 | **否** | 否 |

`make_tuple` 只是给几个已有的 IR 值起名字（docstring：「without forcing a C++ struct to be
generated」），给它的字段赋值会被拒（验证 6）。**所以 SSBUF 这条路只能用 `pl.struct`。**

## 三、如何独立验证

1. ``handoff_one_struct``：三个字段原样到达 cube 侧 → `[8, 1, 32768]`。
2. ``two_structs_spaced_correctly``：偏移 `0` 和 `24`，六个字段全对。
3. ``two_structs_spaced_too_tightly``：偏移 `0` 和 `16`，**第一个 struct 的 `z` 读出第二个的
   `x`** → `[8, 1, 11, 11, 12, 13]`。这条把"每字段 8 字节"钉住。
4. ``struct_array_two_elements``：`pl.struct_array` 的两个元素，偏移规则一样。
5. `pl.struct` 不带名字被解析期拒。
6. 给 `pl.make_tuple` 的字段赋值被拒。
7. `msg.a, msg.b = 1, 2` 这种元组解包被拒。

**验证 3 是这一例的核心**，而不是验证 1。验证 1 只说明"通路是通的"；**只有验证 3 能说明 SSBUF
里的布局是什么样**，而布局是唯一会让人写错的东西（偏移随手写个 16 就静默串字段）。

## 四、`offset` 的单位与槽宽：模型，以及用预测验证它

官方 docstring 只写「offset: SSBUF offset」，**没有单位**；ST 里也只用过 `0`。所以这两个数字
是量出来的：

> **`offset` 以字节计，而每个字段在 SSBUF 里占 8 字节**，不是它的 dtype 宽度。

三字段 INT32 struct（12 字节数据）实测，第二个 struct 放在不同偏移：

| 第二个的 `offset` | 第一个读回来 | 模型预测 |
| ---: | --- | --- |
| 4 | `[8, 0, 0]` | 错半个槽 → 读到高 4 字节的 0 ✓ |
| 8 | `[8, 11, 12]` | 压在第 **2** 个槽 ✓ |
| 12 | `[8, 1, 0]` | 错半个槽 ✓ |
| 16 | `[8, 1, 11]` | 压在第 **3** 个槽 ✓ |
| **24** | `[8, 1, 32768]` | 刚好接在后面 ✓ |
| 32 / 64 / 128 | 正确 | ✓ |

**五个点能拟合出很多模型，所以又用另一个字段数做了预测**（这一步才是证据）：四字段 struct
按模型占 32 字节，于是预测 `off=16` 压第 3、4 个槽、`off=24` 只压第 4 个：

| 四字段的 `offset` | 预测 | 实测 |
| ---: | --- | --- |
| 16 | `[1, 2, 11, 12]` | `[1, 2, 11, 12]` ✓ |
| 24 | `[1, 2, 3, 11]` | `[1, 2, 3, 11]` ✓ |
| 32 | 正确 | 正确 ✓ |

**七个配置逐位吻合，其中四个是先写下预测再测的。** 这就是这一例敢把"损坏的样子"写进判据的理由
——它不是偶发现象（同一进程连跑三轮一致），是一个确定的布局规则。

`reference.py` 的 `clobbered_at()` 是这条规则的形式化，它和三组实测逐位相等。

## 五、这一例不讲什么

- **跨核同步本身**——`set_cross_core` / `wait_cross_core` 的完整语义、`CrossCoreSyncMode` 的
  三个值，在 `references/api/crosscore-contract.md` 与 `fixpipe/` 几例。这里只用它当交接的信号。
- **`pl.get_subblock_idx()`**——subblock / block 的划分在 support 一章的 ``block_grid``。
- **SSBUF 的总容量**——**没测**。要测得一直往上加偏移直到出错，而越界写 SSBUF 的后果未知,
  在共享机上不值当。**这是明说跳过。**
- **非 INT32 字段的槽宽**——只量了 INT32。8 字节的槽装得下 INT64，但**没验**，
  也没验混合宽度的 struct。模型只在 INT32 上成立。

## 六、`pl.struct` 的 "Legacy 匿名形式" 实际不能用

`pl.struct` 的 docstring 列了两种形式：

> - Legacy: ``pl.struct(field=val, ...)`` — anonymous struct, fields-only.
> - With name: ``pl.struct("StructName", field=val, ...)``

**第一种解析期就被拒**：

```
pl.struct("Name", ...) requires exactly one string struct name as first argument
```

`_struct_parser.py:172`。验证 5 钉住这一点——**文档和实现对不上时，以实测为准并说清楚以哪份为准**
（[skill constraints](../../../../SKILL.md#constraints-and-workspace-policy)）。这条落到 `pypto/` 该改 docstring，记在
[`diagnostics/regressions.md`](../../../../references/diagnostics/regressions.md) 的框架侧待办里。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 一节。
"""

import pytest
import torch

import pypto_pro as pp
from pypto_pro._errors import InvalidOperation, InvalidType, InvalidVal

from .kernel import (
    BATCH,
    BLOCK,
    OFFSET,
    SECOND,
    anonymous_struct,
    assign_a_make_tuple_field,
    handoff_one_struct,
    struct_array_two_elements,
    two_structs_spaced_correctly,
    two_structs_spaced_too_tightly,
    unpack_two_fields_at_once,
)
from .reference import clobbered_at, spacing

FIRST = [BATCH, BLOCK, OFFSET]


def _run(kernel, device, size):
    out = torch.full((size,), -1, dtype=torch.int32, device=device)
    kernel(out)
    torch.npu.synchronize()
    return out.cpu().numpy().tolist()


def _parse(kernel):
    """SSBUF 的 kernel 有 vector 和 cube 两段，这里只解析 vector 段——三条被拒的写法都在那一段。"""
    kernel.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)


def test_one_struct_crosses_from_vector_to_cube(device):
    """验证 1。三个字段原样到达。

    **这条只说明通路是通的**，说明不了 SSBUF 里的布局——布局要靠验证 3。
    """
    assert _run(handoff_one_struct, device, 3) == FIRST


def test_two_structs_spaced_by_the_slot_width_do_not_collide(device):
    """验证 2。偏移 `0` 和 `24 = 3 × 8`，六个字段全对。"""
    assert _run(two_structs_spaced_correctly, device, 6) == FIRST + list(SECOND)
    assert spacing(len(FIRST)) == 24


def test_a_tight_offset_silently_clobbers_the_third_field(device):
    """验证 3，**这一例的核心**。偏移 16 压在第一个 struct 的第 3 个槽上。

    判据是一个**确定的损坏形状**：`first.z` 读出 `second.x`。敢这么断言是因为模型在 7 个配置上
    逐位吻合，其中 4 个是先写下预测再测的（§四），而且同一进程连跑三轮一致。

    **它会在两种情况下变红**，两种都是我们想知道的：槽宽变了，或者框架开始拦这种重叠。
    """
    got = _run(two_structs_spaced_too_tightly, device, 6)
    expected_first = clobbered_at(FIRST, list(SECOND), slot_index=2)
    assert got == expected_first + list(SECOND), (
        f"SSBUF 的槽宽或重叠行为变了：期望 {expected_first + list(SECOND)}，实际 {got}"
    )
    assert got[2] == SECOND[0], (
        "第一个 struct 的 z 没有读出第二个的 x —— 槽宽不是 8 字节了？"
    )


def test_a_struct_array_follows_the_same_offset_rule(device):
    """验证 4。`pl.struct_array` 的两个元素，偏移还是要自己算。

    **"用数组"不等于"不用管布局"**：`ssbuf_store(m[1], 24)` 里那个 24 是手写的。
    """
    assert _run(struct_array_two_elements, device, 6) == FIRST + list(SECOND)


def test_an_anonymous_struct_is_refused():
    """验证 5。docstring 把匿名形式列为 "Legacy"，实际解析期就拒。

    判据匹配规则文本。这条**同时是一份文档缺陷的记录**：以实测为准，`pypto/` 侧该改 docstring。
    """
    with pytest.raises(InvalidVal) as raised:
        _parse(anonymous_struct)
    assert "requires exactly one string struct name" in str(raised.value), str(
        raised.value
    )


def test_assigning_a_make_tuple_field_is_refused():
    """验证 6。`make_tuple` 聚合已有值，不是可写的 struct——这是它和 `pl.struct` 的实际分界。"""
    with pytest.raises(InvalidOperation) as raised:
        _parse(assign_a_make_tuple_field)
    assert "the target is not a struct" in str(raised.value), str(raised.value)


def test_unpacking_into_two_fields_is_refused():
    """验证 7。解包的目标必须是变量名，不能是字段。

    写这一例时我自己撞了这条，两个探针 kernel 一起挂，而消息指向"解包"——**很容易误读成
    struct 的问题**，进而去怀疑一个没问题的接口。留着它就是为了下次一眼认出来。
    """
    with pytest.raises(InvalidType) as raised:
        _parse(unpack_two_fields_at_once)
    assert "Tuple unpacking target must be a variable name" in str(raised.value), str(
        raised.value
    )
