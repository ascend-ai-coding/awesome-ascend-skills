# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""CTRL 位域的读改复位，与 `pl.get_spr()`（AR）。

**写法：原始。** 四个 API **全无 ST**，每个数字都是 2026-09-28 在 Ascend950PR 上量的。

## 一、四个 API 的分工

| API | 干什么 | 受可写位限制 |
| --- | --- | --- |
| ``pl.get_ctrl_spr(start, end)`` | 读位区间 | **不受**，任意区间可读 |
| ``pl.set_ctrl_spr(start, end, value)`` | 写位区间 | 受：6-10、45、48、50、53、59、60 |
| ``pl.reset_ctrl_spr(start, end)`` | 恢复**硬件默认值** | 同上 |
| ``pl.get_spr()`` | 读 AR（地址寄存器） | — |

三个 CTRL 接口都只能在 `@pl.jit` 体里调，`start` / `end` 必须是编译期常量。

## 二、如何独立验证

1. ``read_entry_state``：整寄存器读数 == `0x1001000000000008`，且逐位读数与它一致。
2. ``set_then_reset``：59 位 `0 → set(1) → 1 → reset → 0`，邻位 60 始终为 1。
3. ``reset_bit48_changes_it``：**什么都没写**，只 reset，bit 48 从 `1` 变 `0`。
4. ``clear_bit48_then_read`` 连启动三次：每次读到的入口值都是 `1`。
5. ``ar_around_squeeze`` 与 ``ar_without_any_vf``：AR 的所有读数**彼此相等**——
   四个有效计数 × section 前后，加上完全不带 vf 的控制项。**不断言具体值。**
6. 四条被拒的写法：不可写位、两个未定义原子编码、`get_spr` 进 vf 体、`squeeze` 不接返回值。

**判据 1 的写法值得说**：整寄存器和单个位**都是实测量**，判据比的是这两个实测量彼此一致，
而不是拿它们去对一个我猜的默认值。猜错默认值不会让这条红——那件事由判据 1 的**第二半**
（和文档默认值的差集）负责，而那一半的期望值是「差集恰好是 {48}」，一个量出来的事实。

## 三、入口值与文档默认值不同（实测）

| | 值 | bit 48 |
| --- | --- | --- |
| 官方 `reset_ctrl_spr.md` 写的默认值 | `0x1000000000000008` | 0（饱和） |
| **实测 kernel 入口值** | `0x1001000000000008` | **1（非饱和）** |

差集恰好 `{48}`。其余六个可写位（6-10、45、50、53、59）与 60 都和文档一致。

**这件事的后果比看上去大**（判据 3）：`reset_ctrl_spr` 写的是 C++ 里那个写死的
`defaultCtrl = 0x1000000000000008`，不是读回来的入口值。所以

```python
pl.set_ctrl_spr(48, 48, 1)      # 临时改
...                              # 干活
pl.reset_ctrl_spr(48, 48)       # 「恢复原状」—— 并没有
```

这段代码把 bit 48 从入口的 1 变成了 0，**静默把浮点转换从非饱和切成饱和**。
要真正恢复原状得先 `get_ctrl_spr(48, 48)` 存下来，再 `set_ctrl_spr` 写回去。

三份证据的对齐（[skill constraints](../../../../SKILL.md#constraints-and-workspace-policy)）：官方文档和 C++ 常量一致（都是
`0x1000000000000008`）,
**实测与它们不符**。以实测为准，因为要判的是「这段代码在这台机器上会发生什么」。
文档描述的是复位目标值，那句话没错——错的是把它读成「入口值」。

## 四、CTRL 不跨 launch 残留（判据 4）

连启动三次同一个「读 bit 48 → 清零」的 kernel，三次读到的都是 `1`。

**这条是好消息，但必须量过才能依赖**：如果 CTRL 跨 launch 残留，那么任何改过 CTRL 的 kernel
都会污染后面所有 kernel，而症状是「换个测试顺序数值就变」——最难查的那一类。
判据 4 用**重复启动**而不是一次启动里读两遍，因为要验的正是 launch 边界。

## 五、`pl.get_spr()` 读不到 squeeze 的字节数

官方 `get_spr.md`：

> AddrReg 寄存器……`vf.squeeze` 会将有效元素的总字节数存储到 AddrReg 寄存器中，
> 随后可通过 `pl.get_spr` 读取该值。

`_api.py` 的 docstring 说同一件事。**实测 `get_spr()` 的读数与有效元素个数无关**：

| 变量 | 取值 | AR |
| --- | --- | --- |
| 有效元素个数 | 0 / 1 / 7 / 8 / 16 / 30 / 31 / 32 / 63 / 64 | 全部**同一个值** |
| dtype | FP32（64 lane）、FP16（128 lane） | 同一个值 |
| 读的位置 | squeeze 所在 section 之前 / 之后 | 同一个值 |
| **有没有 vf 指令** | **一个都没有**（纯搬运 kernel） | 同一个值 |

最后一行是控制项，也是定案的那一条：**不带 squeeze 也读到同一个值,
所以那个值不是 squeeze 写的。**

十个计数全同值的时候我没有直接下结论——`references/verification/evidence.md` 的防御条目说
「整列同值先怀疑仪器」，所以加了这个控制项。**加对了**：它把「squeeze 写了个巧合的值」
和「squeeze 根本没参与」分开了。

### 那个值不是常量（差一点写错的地方）

**它不是 128，也不是 0，它取决于之前跑了什么**：

| 进程 | AR |
| --- | ---: |
| 只跑这一例的 kernel（`pytest .../spr_ctrl`） | **128** |
| 跑完整个 `support` 一章再跑到这里 | **0** |

第一版判据写的是 `== 128`。单独跑 15 passed，**整章跑 5 条红**——
`references/verification/board-runs.md` §五点八 那个形状（「单独过、一起错」）,
而这次的原因不是同步，是**判据比被测量严**（§五点十）。

现在判据断言的是「所有读数彼此相等」，那是两个进程里都成立的不变量。
`AR_SEEN` 记着见过的两个值，只做参考，红起来会提示「加进去即可」。

为什么这条路没有出口：

| | |
| --- | --- |
| `pl.get_spr()` 在 `@pl.vector_function` 里 | **解析期拒**（判据 6，`_call_parser.py:1159`） |
| `vf.squeeze` 只能在 `@pl.vector_function` 里 | 那是 vf 指令 |

**能写 squeeze 的地方读不了 AR，能读 AR 的地方看不到 squeeze 写的值。** 按 [skill
constraints](../../../../SKILL.md#constraints-and-workspace-policy)
这是框架侧问题（文档与实测不符），记在
[`diagnostics/messages.md`](../../../../references/diagnostics/messages.md#parse-error-localization-and-regressions)。

**这一例只钉住「读到的是 128」这个事实，不声称「AR 一定没被 squeeze 写」**——
AR 也可能被写了、但在 section 边界上被恢复。两者从这一层分不开，需要看生成的 CCE。
分不开的地方说分不开。

## 六、`get_spr` 的「无 ST」核过两遍

`st/` 里有个 `kernel_83_get_spr_movemask`，名字里带 `get_spr`——**它不调 `pl.get_spr()`**,
调的是 `vf.get_mask_spr`。这是按名字搜的**假阳性**，和 `pl.pipeline` 那条假阴性正好相反
（见 `references/tooling/example-coverage.md` §五的更正）。所以「无 ST」这个结论成立。

而且那个 ST 只断言 `f0.dtype == torch.float32`——**是个 smoke test**，就算它真调了
`pl.get_spr()` 也给不出 AR 的值。

## 七、这一例不讲什么

- **饱和标志本身**——`set_saturation_flag` / `get_saturation_flag` 的极性反转与 CTRL[60] 的
  全局／单指令模式在 ``saturation``。那一例用的是**同一个 CTRL 寄存器的 48 / 50 / 53 / 59 位**,
  两例合起来才是完整的一张图。
- **原子操作配置（6-10 位）的实际效果**——只验了两个未定义编码被拒。**合法编码没上板试**：
  它改的是数据搬出 GM 时的原子行为，而 fixpipe 一章已有 atomic 样例走的是接口参数那条路。
  在共享机上改全局原子配置再跑别的东西风险不值当。**这是明说跳过。**
- **`vf.create_addr_reg`**——AR 的另一个写者，在 ``vf/vf_gather_scatter``。写这一例时顺手撞到
  它的一条硬约束：不在循环层里会编译失败（`parameters of vag is not bound into the correct
  loop layer`），那一例已记。
- **45 位（matmul 寄存器格式）**——读了默认值，没验效果。要构造 FP8E4M3FN 的复数 matmul,
  属于 cube 面。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 四节。
"""

import numpy as np
import pytest
import torch

import pypto_pro as pp
from pypto_pro._errors import InvalidVal, NotSupported

from .kernel import (
    DOCUMENTED_DEFAULT,
    N,
    NEIGHBOUR_BIT,
    PROBE_BIT,
    RANGES,
    ar_around_squeeze,
    ar_without_any_vf,
    clear_bit48_then_read,
    get_spr_inside_a_vector_function,
    read_entry_state,
    reset_bit48_changes_it,
    set_then_reset,
    squeeze_without_assignment,
    write_an_unwritable_bit,
    write_undefined_atomic_dtype,
    write_undefined_atomic_op,
)
from .reference import AR_SEEN, MEASURED_ENTRY, bits, differing_bits

LAUNCHES = 3
"""判据 4 连启动几次。3 次而不是 2 次：两次一致可能是巧合，三次一致才像规律——
`references/verification/board-runs.md` §五点八 那句「连跑三遍」的同一个道理。"""


def _run_scalars(kernel, device, size):
    """跑一个只输出 INT64 标量的 kernel。`[None, 1]` 是必须的:
    不带启动括号会解成全量核数（56），56 个 block 往同一片 GM 写同一串标量
    ——值虽然一样，但读回来是谁写赢的，见 ``simt/simt_launch`` §四。
    """
    out = torch.full((size,), -1, dtype=torch.int64, device=device)
    kernel[None, 1](out)
    torch.npu.synchronize()
    return [int(v) for v in out.cpu().numpy()]


def _run_with_input(kernel, device, count, ar_size=2):
    """输入里前 `count` 个元素 > 0，其余 < 0 —— 有效元素个数由输入决定。"""
    a = np.full(N, -1.0, dtype=np.float32)
    a[:count] = np.arange(1, count + 1, dtype=np.float32)
    out = torch.zeros((1, N), dtype=torch.float32, device=device)
    ar = torch.full((ar_size,), -1, dtype=torch.int64, device=device)
    kernel[None, 1](torch.from_numpy(a).reshape(1, N).to(device), out, ar)
    torch.npu.synchronize()
    return [int(v) for v in ar.cpu().numpy()], out.cpu().numpy()[0]


def _parse(kernel):
    kernel.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)


def _compile(kernel, device, size=1):
    """编译期的拒绝要真的走一遍编译。解析期不拦这三条（判据 6 的前三条）。"""
    out = torch.full((size,), -1, dtype=torch.int64, device=device)
    kernel[None, 1](out)
    torch.npu.synchronize()


# ---------------------------------------------------------------------------
# 一、入口状态
# ---------------------------------------------------------------------------


def test_the_whole_register_and_the_single_bits_agree(device):
    """验证 1 前半。整寄存器与逐位读数一致。

    **两边都是实测量**，所以这条验的是框架内部一致（`get_ctrl_spr` 的移位掩码算对了），
    不是「我猜的默认值对不对」。
    """
    got = _run_scalars(read_entry_state, device, len(RANGES))
    whole = got[0]
    for index, (lo, hi) in enumerate(RANGES[1:], start=1):
        assert got[index] == bits(whole, lo, hi), (
            f"get_ctrl_spr({lo}, {hi}) = {got[index]}，但整寄存器 {whole:#x} 的该区间是 "
            f"{bits(whole, lo, hi)}"
        )


def test_the_entry_value_differs_from_the_documented_default_in_bit_48(device):
    """验证 1 后半，**这一例的第一个核心结论**。

    期望值不是猜的：`0x1001000000000008` 是量出来的，判据钉住「和文档默认值的差集恰好是
    `{48}`」。文档改了、硬件改了、或者框架加了 prologue，这条都会红——三种都想知道。
    """
    got = _run_scalars(read_entry_state, device, len(RANGES))
    whole = got[0]
    assert whole == MEASURED_ENTRY, (
        f"入口 CTRL = {whole:#x}，2026-09-28 实测是 {MEASURED_ENTRY:#x}"
    )
    assert differing_bits(whole, DOCUMENTED_DEFAULT) == [48], (
        f"入口值与文档默认值 {DOCUMENTED_DEFAULT:#x} 的差集是 "
        f"{differing_bits(whole, DOCUMENTED_DEFAULT)}，实测只差 bit 48"
    )


# ---------------------------------------------------------------------------
# 二、写与复位
# ---------------------------------------------------------------------------


def test_set_and_reset_round_trip_on_one_bit(device):
    """验证 2。59 位 `0 → 1 → 0`，邻位 60 不受影响。

    选 59 而不是 48：59 的入口值和文档默认值一致，所以这条判据不会被「入口值异常」那件事
    带着一起红。两个问题分开验，红起来才指得准。
    """
    before, after_set, after_reset, neighbour = _run_scalars(set_then_reset, device, 4)
    assert (before, after_set, after_reset) == (0, 1, 0), (
        f"bit {PROBE_BIT} 的 set/reset 往返读到 {(before, after_set, after_reset)}，期望 (0, 1, 0)"
    )
    assert neighbour == 1, f"邻位 {NEIGHBOUR_BIT} 被动了：读到 {neighbour}，期望 1"


def test_resetting_bit_48_changes_it(device):
    """验证 3。**只 reset，什么都没写，bit 48 就从 1 变成 0。**

    这条是「用完恢复原状」那个直觉的反例。`reset_ctrl_spr` 写的是硬件默认值,
    而入口值不等于硬件默认值——对 bit 48 差的就是这一位。
    """
    before, after = _run_scalars(reset_bit48_changes_it, device, 2)
    assert before == 1, f"入口 bit 48 = {before}，实测是 1（非饱和）"
    assert after == 0, f"reset 之后 bit 48 = {after}，期望 0（硬件默认值里这一位是 0）"
    assert before != after, "reset 没有改变 bit 48 —— 入口值和文档默认值不再有差异了？"


def test_ctrl_does_not_survive_across_launches(device):
    """验证 4。连启动三次「读 bit 48 → 清零」，三次都读到 1。

    **这条靠重复启动成立**，不靠 kernel 内部：要验的正是 launch 边界。
    读到 0 会是坏消息——那意味着任何改过 CTRL 的 kernel 都会污染后面的。
    """
    seen = [_run_scalars(clear_bit48_then_read, device, 1)[0] for _ in range(LAUNCHES)]
    assert seen == [1] * LAUNCHES, (
        f"{LAUNCHES} 次启动读到 {seen}，期望全 1：CTRL 应该每次 launch 重置。"
        f"出现 0 说明上一次的清零残留了下来"
    )


# ---------------------------------------------------------------------------
# 三、AR
# ---------------------------------------------------------------------------


AR_COUNTS = (0, 1, 32, 64)
"""验证 5 扫的有效元素个数。**0 和 64 是两端**：一个有效元素都没有、和全都有效,
如果 AR 真的跟着计数走，这两端必然不同。"""


def test_ar_never_varies_with_the_squeeze(device):
    """验证 5，**这一例的第二个核心结论**。

    一次收集所有读数：四个有效计数 × (section 前, section 后)，加上**完全不带 vf 指令**的
    那个控制项。判据是**它们彼此全部相等**。

    **刻意不断言具体值**：实测它在只跑这一例的进程里是 128、跑完整章之后是 0
    （`reference.py` 的 `AR_SEEN`）。差一点就把 `== 128` 写进判据了——
    单独跑 15 passed，整章跑 5 条红，正是 [数值与竞争](../../../../references/verification/board-runs.md#numerics-and-races) 那个形状。

    **不带 vf 的那个控制项是定案的那一条**：没有它，「所有读数相等」只说明「相等」,
    说明不了「与 squeeze 无关」。
    """
    readings = []
    for count in AR_COUNTS:
        (before, after), packed = _run_with_input(ar_around_squeeze, device, count)
        readings.append((f"squeeze(有效={count}) 之前", before))
        readings.append((f"squeeze(有效={count}) 之后", after))
        if count:
            assert packed[0] == 1.0, (
                f"有效 {count} 个时 squeeze 没在工作：压紧结果首元素是 {packed[0]}"
            )

    (before, after), copied = _run_with_input(ar_without_any_vf, device, 32)
    readings.append(("不带任何 vf 指令 之前", before))
    readings.append(("不带任何 vf 指令 之后", after))
    assert copied[0] == 1.0, "直通搬运本身坏了，这条判据说明不了 AR 的事"

    values = {value for _, value in readings}
    assert len(values) == 1, (
        "AR 随 squeeze 变了——那 `get_spr()` 也许真能读到有效字节数：\n  "
        + "\n  ".join(f"{label}: {value}" for label, value in readings)
    )
    seen = values.pop()
    assert seen in AR_SEEN, (
        f"AR 读到 {seen}，之前只见过 {AR_SEEN}。**这不一定是坏事**："
        f"它证实了这个值取决于之前跑了什么，把新值加进 AR_SEEN 即可"
    )


# ---------------------------------------------------------------------------
# 四、被拒的写法
# ---------------------------------------------------------------------------


def test_an_unwritable_bit_is_refused_at_compile_time(device):
    """验证 6 之一。bit 20 不可写。**解析期不拦，编译期拦**，消息带可写位清单。"""
    _parse(write_an_unwritable_bit)  # 解析这一层过得去
    with pytest.raises(RuntimeError) as raised:
        _compile(write_an_unwritable_bit, device)
    assert "not writable on current device" in str(raised.value), str(raised.value)


def test_the_undefined_atomic_dtype_encoding_is_refused(device):
    """验证 6 之二。`CTRL[8:6] = 3'b111`。

    **这条被拒是好事**：C++ 注释说这个编码会让设备 fault，守卫把它挡在编译期,
    而且消息把合法取值列了出来（0-6 = none/float/half/int16/int32/int8/bfloat16）。
    """
    with pytest.raises(RuntimeError) as raised:
        _compile(write_undefined_atomic_dtype, device)
    message = str(raised.value)
    assert "undefined atomic operand dtype" in message, message
    assert "supported: 0-6" in message, (
        "消息里没列合法取值，那就只是「不行」而不是「该写什么」"
    )


def test_the_undefined_atomic_op_encoding_is_refused(device):
    """验证 6 之三。`CTRL[10:9] = 2'b11`（合法 0-2 = ADD/MAX/MIN）。

    **这条守卫只对字面量生效**：C++ 侧先 `ir::As<ir::ConstInt>` 取常量，取不到就跳过检查。
    写成运行期表达式能绕过去，而绕过去的后果按注释是设备 fault。
    **绕过的那一版刻意不写**——在共享机上试一个已知会 fault 的写法不值当。
    这是明说跳过，不是「验过没问题」。
    """
    with pytest.raises(RuntimeError) as raised:
        _compile(write_undefined_atomic_op, device)
    assert "undefined atomic op type" in str(raised.value), str(raised.value)


def test_get_spr_inside_a_vector_function_is_refused():
    """验证 6 之四。`pl.get_spr()` 进不了 vf 体。

    **这条和 §五 是同一件事的两半**：squeeze 只能写在 vf 体里，而 AR 只能在 vf 体外读。
    """
    with pytest.raises(NotSupported) as raised:
        _parse(get_spr_inside_a_vector_function)
    assert "'pl.get_spr' is not supported inside @pl.vector_function" in str(
        raised.value
    ), str(raised.value)


def test_a_squeeze_whose_result_is_discarded_is_refused():
    """验证 6 之五。`vf.squeeze(a, cond)` 当语句写会被拒，要求赋值形式。

    收这条是因为它正是「只为副作用调 squeeze」的第一反应——那条路连语法都不成立。
    """
    with pytest.raises(InvalidVal) as raised:
        _parse(squeeze_without_assignment)
    assert "must use the assignment form" in str(raised.value), str(raised.value)
