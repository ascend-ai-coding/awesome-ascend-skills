# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""八个非对齐搬运接口：`load_unalign_init` `load_unalign_pre` `load_unalign`
`store_unalign` `store_unalign_post` `squeeze_store_unalign` `squeeze_store_unalign_post`
`unalign_reg_for_store`。

**写法：原始。** 前置样例是 [`vf_align`](../vf_align/test.py)（对齐搬运）与
[`vf_lane_layout`](../vf_lane_layout/test.py)（`squeeze`）。

机制的完整解释在设计文档 §九——它解决什么问题、两条 store 链的区别、真实场景、四条约束。
**这里不重复讲，只演示和验证。**

## 一、三条链，八个接口

| 链 | 接口 | 谁决定本次推进多少字节 |
| --- | --- | --- |
| load | `load_unalign_init` → `load_unalign_pre` → `load_unalign(stride)` | 调用方传的 `stride` |
| store，调用方驱动 | `unalign_reg_for_store` → `store_unalign(stride)` → `store_unalign_post(stride)` | 调用方传的 `stride` |
| store，数据驱动 | unalign_reg_for_store→squeeze_store_unalign→squeeze_store_unalign_post | AR 特殊寄存器，由squeeze(STORE_REG)写入 |

`unalign_reg_for_store()` 两条 store 链共用，所以 8 = 3 + 3 + 2。

**`squeeze_store_unalign` 没有 `stride` 参数不是简化**——推进量不由调用方决定。这是分界线：
调用方驱动用在「每次写多长编译期或循环变量能算出来」，数据驱动用在「只有运行时才知道」
（谓词的 popcount）。

## 二、`*_post` 不是可选的

前面几次写出去的是「凑满整块的部分」，不满一块的尾巴留在跟踪寄存器里。`*_post` 把它冲出去。
**少了它，输出尾部会缺数据，而且不报错。**

## 三、同一条存储链的状态跨循环延续

`test_the_two_filter_passes_append_without_a_gap` 验的就是这件事：两轮筛选共用一个 `areg`、
中间不收尾，第二轮从第一轮停下的**字节位置**接着写，输出里没有空洞。这是 Top-K 索引收集
（`lightning_indexer` ST）的骨架。UnalignReg 保存尾字节，squeeze 链的累计位移还依赖共享 AR；独立输出必须各自初始化状态，不能只换跟踪寄存器。

## 三点五、两个静默陷阱（都是实测踩出来的）

| 陷阱 | 症状 | 规则 |
| --- | --- | --- |
| 整型 stride 须配 post_update=True | bisheng 报 `... 'POST_UPDATE'` | 标量 stride 必须 post_update=True（默认 False 不行） |
| `squeeze(STORE_REG)` 前漏 `vf.clear_spr()` | **不报错不挂**，输出是 Tile 残留内容，像「搬运没生效」 | AR 特殊寄存器，进 @pl.vector_function 时不保证为空 |
| 变长写只覆盖 Tile 的一部分，没写到的位置当成 0 | 尾部出现上一个 kernel 的残留值 | **UB 不是零初始化的**；变长输出要先 `store_align` 一遍 0 |

后两条的共同点：错的表现是「结果不对」，但排查方向是**初始化**，不是「搬运语义理解错了」。
第三条只在变长写里出现——定长写覆盖整块，看不见它。

## 四、一条不能上板的负面用例

连着两次 `squeeze(gather_mode=STORE_REG)` 而中间不夹 `squeeze_store_unalign`：第二次会覆盖
AR 里还没被消费的字节数，官方文档说这种用法会**挂死硬件**。

`test_two_squeezes_in_a_row_are_not_rejected_at_parse_time` **只解析，不执行**——它验的是
「工具链不拦」这个事实，不是「它能跑」。这条钉住的正是
[skill constraints](../../../../SKILL.md#constraints-and-workspace-policy) 里「同一类错误反复出现 → skill 的 `scripts/` 加确定性检查」
该拦的第一条规则：在一个 `@pl.vector_function` 体内，`squeeze(STORE_REG)` 与
`squeeze_store_unalign` 必须严格交替。

**这一例是整个 vf 系里唯一有「不能执行的代码」的样例。** 它待在 `kernel.py` 里是安全的
——解析在首次**调用**时发生，模块导入不触发。

## 五、判据

`rtol=0, atol=0`。搬运是逐字节的，操作数是 1..64 的整数。

**load / store 两条链的确切推进语义在写这一版时还没实测**（`stride` 是本次长度还是下次起点，
文档没写清），所以那两条先断言**关系**：哪些位置必须被搬到、两次写必须不同。
squeeze 那条链的语义是确定的，直接比值。
"""

import numpy as np
import pytest
import torch

import pypto_pro as pp

from . import reference as ref
from .kernel import (
    N,
    load_chain,
    make_independent_squeeze_streams,
    squeeze_chain,
    store_chain,
    two_squeezes_in_a_row,
)

_ALL_APIS = [
    "load_unalign_init",
    "load_unalign_pre",
    "load_unalign",
    "store_unalign",
    "store_unalign_post",
    "squeeze_store_unalign",
    "squeeze_store_unalign_post",
    "unalign_reg_for_store",
]


def _dev(array, device):
    return torch.from_numpy(array).to(device)


def _out(device):
    return torch.zeros((1, N), dtype=torch.float32, device=device)


@pytest.mark.parametrize("cols", [65, 130])
@pytest.mark.parametrize("kind", ["mixed", "positive", "zero"])
def test_independent_squeeze_outputs_reset_ar(device, cols, kind):
    rng = np.random.default_rng(cols)
    x = rng.uniform(-3, 3, (1, cols)).astype(np.float32)
    if kind == "positive":
        x[:] = np.abs(x) + np.float32(0.01)
    elif kind == "zero":
        x.fill(0)
    inputs = _dev(x, device)
    backing = [
        torch.full((cols + 128,), 1234, device=device, dtype=torch.float32)
        for _ in range(2)
    ]
    outputs = [buffer[64:-64].view(1, cols) for buffer in backing]
    expected = [np.zeros_like(x), np.zeros_like(x)]
    for target, keep in zip(expected, [x > 0, x < 0]):
        selected = x[keep]
        target[0, : selected.size] = selected
    kernel = make_independent_squeeze_streams(cols)
    for _ in range(2):
        for output in outputs:
            output.fill_(float("nan"))
        kernel[None, 1](inputs, *outputs)
        torch.npu.synchronize()
        for output, want, buffer in zip(outputs, expected, backing):
            np.testing.assert_array_equal(
                output.cpu().numpy().view(np.uint32), want.view(np.uint32)
            )
            np.testing.assert_array_equal(buffer[:64].cpu().numpy(), 1234)
            np.testing.assert_array_equal(buffer[-64:].cpu().numpy(), 1234)
    np.testing.assert_array_equal(
        inputs.cpu().numpy().view(np.uint32), x.view(np.uint32)
    )


def test_the_load_chain_reads_from_the_tile(device):
    """`init` → `pre` → `load` 三步走通，并且读到的是 Tile 里的真实数据。

    **确切的推进语义这一版没断言**（见模块 docstring §五）。断言的是：
    搬出来的东西必须来自输入，而不是零或垃圾。
    """
    a = ref.make_operands(N)
    out = _out(device)
    load_chain(_dev(a, device), out)
    torch.npu.synchronize()
    got = out.cpu().numpy().reshape(-1)
    src = a.reshape(-1)

    assert (got != 0).any(), "输出全是 0——load 链没搬到任何东西"
    # 搬出来的每个非零值都必须是输入里存在的值
    unexpected = [v for v in np.unique(got[got != 0]) if v not in set(src.tolist())]
    assert not unexpected, f"输出里出现了输入里没有的值 {unexpected[:6]}——读到了垃圾"


def test_the_store_chain_writes_two_chunks(device):
    """两次 `store_unalign` + 一次 `store_unalign_post`。

    断言：输出被写到了，且写进去的值来自输入。确切布局见 §五。
    """
    a = ref.make_operands(N)
    out = _out(device)
    store_chain(_dev(a, device), out)
    torch.npu.synchronize()
    got = out.cpu().numpy().reshape(-1)
    src = set(a.reshape(-1).tolist())

    assert (got != 0).any(), "输出全是 0——store 链什么都没写"
    written = got[got != 0]
    unexpected = [v for v in np.unique(written) if v not in src]
    assert not unexpected, f"写进去的值不在输入里 {unexpected[:6]}"
    assert len(written) >= ref.CHUNK, (
        f"至少该写 {ref.CHUNK} 个元素（一个 chunk），实际非零 {len(written)} 个"
    )


def test_the_two_filter_passes_append_without_a_gap(device):
    """**这一例的重点**：两轮变长筛选的结果紧密追加，中间没有空洞。

    输入 1..64：第一轮取 >48（49..64），第二轮取 <=16（1..16）。
    输出前 32 个应该正好是 `49..64, 1..16`——**顺序和位置都对**才说明游标跨轮存活了。
    """
    a = ref.make_operands(N)
    out = _out(device)
    squeeze_chain(_dev(a, device), out)
    torch.npu.synchronize()
    got = out.cpu().numpy()

    want, n_first, n_second = ref.squeeze_chain(a)
    assert n_first > 0 and n_second > 0, "两轮里有一轮没命中，这条验不到追加"

    np.testing.assert_allclose(
        got, want, rtol=0, atol=0, err_msg="两轮筛选的结果没有紧密追加"
    )

    flat = got.reshape(-1)
    assert flat[0] == 49.0, f"第一轮的第一个应该是 49，实际 {flat[0]}"
    assert flat[n_first] == 1.0, (
        f"第二轮应该从下标 {n_first} 开始（紧接第一轮），实际那里是 {flat[n_first]}"
    )
    assert (flat[slice(n_first + n_second, None)] == 0).all(), (
        "两轮之后的位置应该没被碰过"
    )


def test_the_squeeze_chain_is_not_just_two_aligned_stores(device):
    """防空测：如果游标没跨轮存活，第二轮会从 0 覆盖第一轮。

    那样输出前 16 个就是 `1..16` 而不是 `49..64`。
    """
    a = ref.make_operands(N)
    out = _out(device)
    squeeze_chain(_dev(a, device), out)
    torch.npu.synchronize()
    flat = out.cpu().numpy().reshape(-1)

    assert flat[0] != 1.0, "输出从 1 开始——第二轮覆盖了第一轮，游标没有跨轮存活"


def test_two_squeezes_in_a_row_are_not_rejected_at_parse_time():
    """**只解析，不执行。** 连着两次 `squeeze(STORE_REG)` 不夹 store——工具链不拦。

    这条钉住的是「解析期不拦」这个事实。它**不能上板**：第二次 squeeze 覆盖 AR 里还没被消费的
    字节数，官方文档说会挂死硬件。

    如果哪天解析器开始拦它，这条会失败——那是好事，改成 `pytest.raises` 并把
    skill 的 `scripts/` 里对应的检查撤掉。
    """
    program, _ = two_squeezes_in_a_row.to_kernel_def().parse_target_program(
        pp.ir.SectionKind.Vector
    )
    text = pp.ir.python_print(program)

    assert text.count("@vf.squeeze(") == 2, (
        f"IR 里应该有两个 vf.squeeze，实际 {text.count('@vf.squeeze(')} 个"
    )
    # 两次 squeeze 之间没有 squeeze_store_unalign —— 解析器照样放行
    first = text.index("@vf.squeeze(")
    second = text.index("@vf.squeeze(", first + 1)
    between = text[first:second]
    assert "squeeze_store_unalign" not in between, (
        "两次 squeeze 之间出现了 store，那这条负面用例的形状就不对了"
    )


@pytest.mark.parametrize("name", _ALL_APIS)
def test_every_api_in_this_example_is_actually_called(name):
    """不需要卡。防漏：表里列的 8 个接口每个都得在 kernel 源码里出现。"""
    from pathlib import Path

    source = (Path(__file__).parent / "kernel.py").read_text()
    assert f"vf.{name}(" in source, f"kernel.py 里没有调用 vf.{name}()"
