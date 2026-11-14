# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""``vf.*`` 的入口：为什么必须有 `@pl.vector_function`，以及掩码从哪来。

**这是 vf 系的第一个样例。** 库里在它之前的 21 个样例全在 cube 路径上，`vf.*` 一条都没覆盖过。
运算故意选最无聊的 `x + 1.0`——这样看得见的只有机制。

**写法：原始。** `vf.*` 没有语法糖，不存在「糖 vs 原始」这个选择。

## 一、`@pl.vector_function` 是唯一入口

`pl.section_vector()` **不是**入口。直接在它里面写 `vf.*` 会被解析期拒绝：

```
Operation 'vf.adds' can only be used inside @pl.vector_function
```

（`pypto_pro/language/parser/_call_parser.py:1144`）。section 是放 vf 体**调用**的地方，
vf 算子本身只能待在被装饰的函数里。验证 4 钉住这一点——它是这个装饰器存在的理由。

这条不是文档里读到的，是 2026-09-26 实测出来的：第一版 probe 把 `vf.*` 直接写进
`pl.section_vector()`，被响亮拒绝。

## 二、VL：为什么循环步长是 64

一个 FP32 寄存器装 **VL = 256 字节 = 64 个元素**。所以 `pl.range(0, N, 64)`，`N = 128` 是两个
整寄存器——取 128 而不是 64，是为了让「第二次迭代」真的存在，否则循环和直线代码没区别。

换 dtype 就换元素数：FP16 是 128 个，INT8 是 256 个。步长跟着 dtype 走，不是常数 64。

## 三、两种同步形式，同一个结果

vf 是 V 流水上的事，而数据要 MTE2 搬进来、MTE3 搬出去。三段之间必须缝上：

| kernel | 怎么缝 |
| --- | --- |
| ``vf_auto`` | ``@pl.jit(auto_mutex=True)``，框架按 `mutex_ids` 自己插 |
| ``vf_manual_sync`` | 每处交接一对 ``pl.system.sync_src`` / ``sync_dst``，``event_id`` 自己分配 |

验证 2 断言两者**逐位相同**：同步形式不改变算什么。两种形式在框架 ST 里都有（`test_vf_scope.py`
用 auto，`test_vf_basic_ops.py` 用手工，150 处），所以都要会读。

**但它们不是同一件事的两种拼写，是两套机制**（验证 5 在 IR 上钉住，2026-09-26 实测）：

| | 同步 op | 用的资源 |
| --- | --- | --- |
| ``vf_auto`` | 8 条 ``system.mutex_lock_dyn`` / ``mutex_unlock_dyn`` | buffer mutex（按 `mutex_ids` 上锁） |
| ``vf_manual_sync`` | 4 条 ``system.sync_src_dyn`` / ``sync_dst_dyn`` | pipe event（按 `event_id` 配对） |

而两版的 **vf 体 op 序列完全一致（5 条）**——所以「同步不改变算什么」这句话在 IR 上也成立，
不只是数值上成立。

**手工那条的代价**：`event_id` 是自己分配的资源，撞号不报错。`auto_mutex` 把这件事拿走了，
代价是多了 4 条 mutex op。这和 `pl.mem_space()` 拿走地址与 mutex id 是同一类交换。

## 四、两种掩码来源

| 来源 | 什么时候用 |
| --- | --- |
| ``vf.create_mask(pattern=pl.MaskPattern.ALL)`` | 长度是编译期已知的整寄存器 |
| ``vf.update_mask(active)`` | 长度运行期才知道，`active` 每次迭代现算 |

尾块的算式是 ``active = pl.max(0, pl.min(count - offset, VL))``。两层夹紧各有理由：
`min` 收最后一次迭代，`max` 处理 `count` 小于 `offset` 时整次迭代都空（`count=0` 会走到）。

**掩码之外的位置不被写。** 验证 3 用哨兵值把这件事变成可观测的：`out` 先填 `-10.0`，kernel 只写
前 `count` 个，剩下的必须还是 `-10.0`。如果掩码没生效，那些位置会变成 `x + 1`——两者差得很远，
不会被容差藏住。

## 四点五、`load`/`store` 与 `load_align`/`store_align` 是两套接口

不是一套的简写。差别在**有没有偏移和模式面**：

| | 定位 | 模式面 |
| --- | --- | --- |
| ``vf.load(tile[, stride])`` / ``vf.store(tile, src[, count])`` | tile 基址，一个寄存器 | 只有 post-update |
| ``vf.load_align(tile, offset)`` / ``vf.store_align(tile, src, preg, offset)`` | 基址+偏移 | 双搬入、DataBlock、`dist=` |

验证 6 让两套接口做同一个拷贝并逐位比，把「两套而不是两种写法」钉成可检验的。

**``vf.store`` 的 ``count`` 可以不传**——不传就是一个整寄存器（2026-09-27 板上实测）。

写这一例时三份证据不一致：官方 `store.md` 的原型和参数说明都把它写成必填，Python 签名
``store(tile, src, count)`` 也没有默认值；但那一页的**调用示例里就没传**，框架 docstring 写着
「optional, defaults to 256/elem_bytes」。实测站在调用示例这边，**错的两份已经修了**（`pypto`
的 `f8f172bdc`：签名改成 ``count=None``，官方文档的原型和参数行跟上）。

## 四点六、`move` 与 `full` 的 MergeMode 默认值相反，而另一种都不支持

| | 默认 | 未选中的 lane | 另一种模式 |
| --- | --- | --- | --- |
| ``vf.move`` | **MERGING** | 保留目的寄存器**原值** | ``ZEROING`` 不支持 |
| ``vf.full`` | **ZEROING** | **置 0** | ``MERGING`` 不支持 |

两个「不支持」都不在解析期拦，**在 codegen 拦**，消息清楚且带源码行（验证 9）：

```
vf.move only supports MERGING mode on current device   # backend_cce_vf_ops.cpp:6614
vf.full only supports ZEROING mode on current device   # backend_cce_vf_ops.cpp:162
```

**`move` 那一行的「原值」是谁，取决于 LHS 是不是新名字。** `_assignment_parser.py` 的 Step 2
只在**第一次定义**时声明寄存器，「For existing variables, reuse them directly -- do NOT create a
new register, as that would lose the accumulated value in self-referential ops」：

| LHS | 目的寄存器 | 未选中的 lane | 有判据吗 |
| --- | --- | --- | --- |
| 已赋过值的名字 | **复用同一个** | 保留它上一次的内容 | **有**，验证 10 |
| 新名字 | 新分配 | 原值未定义 | 没有 |

实测（2026-09-27，`keep` 取 1001..1064、`src` 取 1..64、半掩码）：

```
复用已有名字   [1..32, 1033..1064]     ← 后半正是 keep，这就是 MERGING 的用途
新名字         [1..32,   33..64]       ← 后半是**源寄存器自己的值**
```

**新名字那一支比给出垃圾更危险**：输出看起来就是一次完整拷贝，一条「move 拷贝了整块」的断言会以
错误的理由通过（同 [verification/evidence.md](../../../../references/verification/evidence.md)
#7 / #14）。所以 `_simple_pair` 里那个新名字的 `vf.move` 掩码给满，而**新名字配偏掩码没有测试**
——未定义行为没有判据。有判据的是复用那一支，`_move_merging` 演的就是它。

## 五、如何独立验证

1. ``vf_auto`` 与参考 ``rtol=0, atol=0``。可以逐位比的理由见
   [reference.py](reference.py)：输入是 `arange`，加 1 之后仍然精确可表示。
2. ``vf_manual_sync`` 与 ``vf_auto`` 逐位相同——两种同步形式算出同一个东西。
3. ``vf_tail`` 在 `count = 0 / 1 / 17 / 64 / 79 / 128` 六个值上，前 `count` 个加一、其余保持哨兵。
   `0` 和 `128` 是两端，`64` 正好一个整寄存器，`17` / `79` 是两种尾块（第一次和第二次迭代里的）。
4. 把 ``vf.adds`` 直接写进 ``pl.section_vector()``，解析期必须拒绝，且消息里点名
   ``@pl.vector_function``。
5. 两版的 IR：vf 体 op 序列相同，同步 op 不同类。这条防的是验证 2 变成空测——如果两版编译出
   一样的东西，「两种同步形式」就是句空话。

6. ``vf_simple_pair`` 与 ``vf_aligned_pair`` 逐位相同，且都等于输入——两套搬运接口做同一件事。
   顺带证明 ``vf.store`` 不传 `count` 能用（不能用的话解析就过不去）。
7. ``vf_broadcast`` 第 0 行：前 32 个 lane 是 `7.0`，后 32 个是 `0.0`。这条验的是 `vf.full` 的
   `ZEROING` 默认——未选中的 lane 被置 0，不是保留原值。
8. ``vf_broadcast`` 第 1 / 2 行：`LOWEST` 给元素 0、`HIGHEST` 给元素 63。输入取 `arange(1..64)`，
   两端的值不同（1.0 与 64.0），所以两个 `pos` 不会互相冒充。
9. ``move(ZEROING)`` 与 ``full(MERGING)`` 必须在 codegen 被拒，消息里点名**这一层自己的**规则
   （"only supports ... mode on current device"），**不匹配错误码**——匹配错误码会让判据变成
   同义反复（[证据判断](../../../../references/verification/evidence.md) #14）。
10. ``vf_move_merging``：前 32 个来自 `x`、后 32 个保留 `keep`。两个输入的值域完全不重叠
    （1..64 与 1001..1064），所以「保留了谁」不会含糊。这条验的是复用寄存器下的 `MERGING`。

第 1 条单独不够：它只证明整块对。第 3 条才检验掩码，第 2 条才检验同步，第 4 条才检验入口约束，
第 5 条才保证第 2 条不是空的，第 6 条才让「两套接口」不是句空话。

## 六、这一例不讲什么

- **具体算什么**——`adds` 只是载体，算术面在 ``vf_arith``。
- **对齐搬运的模式面**——``load_align`` / ``store_align`` 这里只用了最简单的连续单搬入。它们还有
  双搬入（de-interleave）、DataBlock 非连续、以及 `dist=` 的 broadcast / 上下采样，在 ``vf_align``。
- **非对齐搬运**——`load_align` 要求偏移对齐。变长紧密输出那套在 ``vf_unalign``。
- **``vf.load`` 的 post-update 模式**——传 `stride` 会在搬完后推进源地址。这里不演：连续多段搬运
  是 ``vf_align`` / ``vf_unalign`` 的主题，而且**探它要先把每一段都填上已知数据**。
  2026-09-27 第一次探时没填，读到上一次运行留在 UB 里的东西，同一个 case 两次跑给出不同的数——
  差点把「残留」当成缺陷（`references/verification/board-runs.md`）。
- **``load.md`` 说的调用次数上限**——那一页的约束说明写「接口内部定义了一个
  ``vf.unalign_reg_for_store``，该寄存器数量上限为 4」。**实测一个 vf 体里 5 次 ``vf.load``
  不被拦**（普通模式和 post-update 模式都试了，2026-09-27 板上）。所以那条约束的触发条件
  还不知道——**这是「没复现」，不是「文档写错了」**，记在设计文档 §七点五。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §八 与
[api/vf.md](../../../../references/api/vf.md) §一。
"""

import re

import numpy as np
import pytest
import torch

import pypto_pro as pp
import pypto_pro.language as pl
from pypto_pro._errors import InvalidOperation
from pypto_pro.language import Vf as vf  # noqa: N813

from .kernel import (
    ACTIVE_LANES,
    BROADCAST_SCALAR,
    N,
    VL_FP32,
    vf_aligned_pair,
    vf_auto,
    vf_broadcast,
    vf_manual_sync,
    vf_move_merging,
    vf_simple_pair,
    vf_tail,
)
from .reference import (
    SENTINEL,
    add_one,
    add_one_prefix,
    broadcast_rows,
    make_input,
    move_merging,
    one_register,
)

FP32 = pl.DT_FP32


def _dev(array, device):
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def _run_whole(kernel, device):
    x = make_input(N)
    out = torch.zeros([1, N], device=device, dtype=torch.float32)
    kernel(_dev(x, device), out)
    torch.npu.synchronize()
    return out.cpu().numpy(), x


def test_the_whole_register_form_matches_the_reference(device):
    """验证 1。整寄存器，两次迭代，逐位相等。"""
    got, x = _run_whole(vf_auto, device)
    np.testing.assert_array_equal(got, add_one(x))


def test_both_sync_forms_give_the_same_numbers(device):
    """验证 2。auto_mutex 与手工 sync_src/sync_dst 逐位相同。

    两个 kernel 的 vf 体是**同一个** `_add_one_whole`，差别只在同步怎么来的。所以出现数值
    差异只可能是同步错了，不可能是运算不同。
    """
    auto, _ = _run_whole(vf_auto, device)
    manual, _ = _run_whole(vf_manual_sync, device)
    np.testing.assert_array_equal(auto, manual)


@pytest.mark.parametrize("count", [0, 1, 17, VL_FP32, 79, N])
def test_the_runtime_mask_leaves_the_tail_untouched(device, count):
    """验证 3。前 count 个加一，其余保持哨兵值。

    哨兵是这条测试的关键：不写哨兵就只能证明前 count 个对，证明不了后面**没被碰**。
    """
    x = make_input(N)
    out = torch.full((1, N), SENTINEL, device=device, dtype=torch.float32)
    vf_tail(_dev(x, device), out, count)
    torch.npu.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy(), add_one_prefix(x, count))


def test_a_vf_op_outside_vector_function_is_refused():
    """验证 4。`pl.section_vector()` 不是入口——这是那个装饰器存在的理由。

    不需要卡：解析就会拒绝。
    """

    @pl.jit(name="vf_outside", auto_mutex=False)
    def bad(x: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
        """反例：vf-operation-needs-vector-function —— 仅解析，断言框架拒绝 VF 作用域之外的操作。"""
        tile_type = pl.TileType(
            shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        src = pl.make_tile(tile_type, addr=0)
        dst = pl.make_tile(tile_type, addr=N * 4)
        with pl.section_vector():
            pl.load(src, x, [0, 0])
            # 没有 @pl.vector_function 包着
            preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
            reg = vf.load_align(src, 0)
            reg = vf.adds(reg, pl.const(1.0, FP32), preg)
            vf.store_align(dst, reg, preg, 0)
            pl.store(out, dst, [0, 0])

    with pytest.raises(InvalidOperation) as raised:
        bad.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)

    message = raised.value.message
    assert "vector_function" in message, message


def _vector_ir(kernel):
    program, _ = kernel.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)
    return pp.ir.python_print(program)


def _op_names(text):
    return re.findall(r"@([\w.]+)\(", text)


def test_the_two_sync_forms_compile_to_different_mechanisms():
    """验证 5。防验证 2 变成空测：两版必须真的不同，而 vf 体必须真的相同。

    不需要卡。auto_mutex 走 buffer mutex（按 ``mutex_ids`` 上锁），手工走 pipe event（按
    ``event_id`` 配对）——不是同一件事的两种写法。
    """
    auto_ops = _op_names(_vector_ir(vf_auto))
    manual_ops = _op_names(_vector_ir(vf_manual_sync))

    # vf 体是同一个 Python 函数，op 序列必须逐条相同
    auto_vf = [op for op in auto_ops if op.startswith("vf.")]
    manual_vf = [op for op in manual_ops if op.startswith("vf.")]
    assert auto_vf == manual_vf, (auto_vf, manual_vf)
    assert auto_vf, "vf 体一条 op 都没有，说明没解析到 vf"

    # 同步机制必须不同类
    auto_sync = [op for op in auto_ops if "mutex" in op]
    manual_sync = [op for op in manual_ops if "sync_" in op]
    assert auto_sync, f"auto_mutex 版没有 mutex op: {auto_ops}"
    assert manual_sync, f"手工版没有 sync op: {manual_ops}"
    assert not [op for op in auto_ops if "sync_" in op], "auto 版不该有 pipe event"
    assert not [op for op in manual_ops if "mutex" in op], "手工版不该有 buffer mutex"


# ---------------------------------------------------------------------------
# 两套搬运接口，与 `vf.full` 的两种模式
# ---------------------------------------------------------------------------


def _run_one_register(kernel, device):
    """一个寄存器进出。`out` 先填哨兵——不填就分不清「拷贝对了」和「kernel 什么都没写」。"""
    x = make_input(VL_FP32)
    out = torch.full((1, VL_FP32), SENTINEL, device=device, dtype=torch.float32)
    kernel(_dev(x, device), out)
    torch.npu.synchronize()
    return out.cpu().numpy(), x


def test_the_simple_pair_copies_one_register(device):
    """验证 6a。`vf.load` → `vf.move`（满掩码）→ `vf.store`，逐位等于输入。

    顺带证明 **`vf.store` 不传 `count` 能用**：不能用的话解析就过不去，跑不到这里。
    """
    got, x = _run_one_register(vf_simple_pair, device)
    np.testing.assert_array_equal(got, one_register(x, VL_FP32))


def test_the_simple_and_aligned_pairs_agree(device):
    """验证 6b。两套搬运接口做同一个拷贝，逐位相同。

    这条让「两套接口而不是两种写法」成为可检验的：如果 `vf.load`/`vf.store` 其实是
    `load_align`/`store_align` 的别名，这条照样过——它证明的不是实现不同，而是**语义等价**。
    差异会出现在哪里，见模块 docstring §四点五那张表。
    """
    simple, _ = _run_one_register(vf_simple_pair, device)
    aligned, _ = _run_one_register(vf_aligned_pair, device)
    np.testing.assert_array_equal(simple, aligned)


def test_the_broadcast_rows_match_the_reference(device):
    """验证 7 + 8 一起跑，三行一次比完。

    - 第 0 行钉 `vf.full` 的 `ZEROING` 默认：前 32 个是 `7.0`，后 32 个是 `0.0`。
    - 第 1 / 2 行钉 `pos`：`LOWEST` 给元素 0（`1.0`），`HIGHEST` 给元素 63（`64.0`）。
      两端的值不同，所以两个 `pos` 不会互相冒充。
    """
    x = make_input(VL_FP32)
    out = torch.full((3, VL_FP32), SENTINEL, device=device, dtype=torch.float32)
    vf_broadcast(_dev(x, device), out)
    torch.npu.synchronize()
    expected = broadcast_rows(x, BROADCAST_SCALAR, ACTIVE_LANES, VL_FP32)
    np.testing.assert_array_equal(out.cpu().numpy(), expected)


@pytest.mark.parametrize(
    "mode_name, rule",
    [
        ("move_zeroing", "vf.move only supports MERGING mode"),
        ("full_merging", "vf.full only supports ZEROING mode"),
    ],
)
def test_the_unsupported_merge_modes_are_refused_at_codegen(device, mode_name, rule):
    """验证 9。两个「当前不支持」的 MergeMode **解析期通过、codegen 拒**。

    判据匹配**这一层自己的**规则文本（`backend_cce_vf_ops.cpp` 出的那句），不匹配错误码——
    匹配错误码会让判据变成同义反复（[证据判断](../../../../references/verification/evidence.md) #14）。

    这两个形状不上板：codegen 就失败了，设备没被碰过，所以不会污染同进程里后面的用例。
    """

    @pl.vector_function
    def _body(src, dst):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        if mode_name == "move_zeroing":
            reg = vf.load_align(src, 0)
            result = vf.move(reg, preg, mode=pl.MergeMode.ZEROING)
        else:
            result = vf.full(
                BROADCAST_SCALAR, preg, dtype=FP32, mode=pl.MergeMode.MERGING
            )
        vf.store_align(dst, result, preg, 0)

    @pl.jit(auto_mutex=True, name=f"vf_{mode_name}")
    def kernel(x: pl.Tensor[[1, VL_FP32], FP32], out: pl.Tensor[[1, VL_FP32], FP32]):
        tile_type = pl.TileType(
            shape=[1, VL_FP32], dtype=FP32, target_memory=pl.MemorySpace.Vec
        )
        src_group = pl.make_tile_group(type=tile_type, addrs=[0x400], mutex_ids=[0])
        dst_group = pl.make_tile_group(type=tile_type, addrs=[0x500], mutex_ids=[1])
        with pl.section_vector():
            src = src_group.next()
            dst = dst_group.next()
            pl.load(src, x, [0, 0])
            _body(src, dst)
            pl.store(out, dst, [0, 0])

    x = make_input(VL_FP32)
    out = torch.zeros((1, VL_FP32), device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError) as raised:
        kernel(_dev(x, device), out)
    assert rule in str(raised.value), str(raised.value)


def test_move_onto_an_existing_register_preserves_the_lanes_outside_the_mask(device):
    """验证 10。`MERGING` 在**复用寄存器**时的语义，这是 `vf.move` 偏掩码唯一有判据的形状。

    两个输入的值域完全不重叠（`1..64` 与 `1001..1064`），所以「后半保留了谁」不会含糊——
    如果目的寄存器其实是新分配的，后半会是 `33..64`（源寄存器自己的值），判据立刻红。
    用的就是探针那两组数，所以 docstring 里引的实测结果和这条测试是同一件事。
    """
    x = make_input(VL_FP32) + 1.0
    keep = make_input(VL_FP32) + 1001.0
    out = torch.full((1, VL_FP32), SENTINEL, device=device, dtype=torch.float32)
    vf_move_merging(_dev(x, device), _dev(keep, device), out)
    torch.npu.synchronize()
    np.testing.assert_array_equal(
        out.cpu().numpy(), move_merging(x, keep, ACTIVE_LANES, VL_FP32)
    )
