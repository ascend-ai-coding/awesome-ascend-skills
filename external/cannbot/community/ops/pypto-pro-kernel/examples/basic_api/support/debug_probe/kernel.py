# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。Native printf, dump_data, pto_assert and trap probes.

Vector functions expose registers by storing to a tile before a host-visible
dump outside VF. test.py isolates log capture and context-failing probes in
child processes and verifies the loaded framework identity.
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
I32 = pl.DT_INT32

VL_FP32 = 64  # 一个 FP32 寄存器 = 256 字节 = 64 个元素
N = 64

_A = 0
_B = 0x400


def _tile_type():
    return pl.TileType(shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、printf：编译期格式串 + 运行期标量
# ---------------------------------------------------------------------------


@pl.jit()
def probe_printf(out: pl.Tensor[[1], I32], v_int: pl.DT_INT32, v_flt: pl.DT_FP32):
    """`%d` 取有符号整数，`%f` 只支持 FP32。格式串必须是编译期常量。

    `out[0] = 1` 是为了让 host 侧能断言 kernel 真的跑到了这里——只看打印内容不够，
    打印可能来自上一次运行的缓冲。
    """
    with pl.section_vector():
        pl.printf("PROBE_PRINTF int=%d flt=%f\n", v_int, v_flt)
        out[0] = 1


# ---------------------------------------------------------------------------
# 二、机器形状：打印是「每个核各打一份」，先拿到核数
# ---------------------------------------------------------------------------


@pl.jit()
def probe_block_shape(out: pl.Tensor[[2], I32]):
    """把核数写进张量，不靠数打印行数反推。

    `get_block_num()` 是**实际启动**的 worker 数，可能小于 host 申请的
    `kernel[stream, block_dim]`；所以它才是用来算「打印会出现几份」的那个数。
    """
    with pl.section_vector():
        out[0] = pl.get_block_num()
        out[1] = pl.get_subblock_num()


# ---------------------------------------------------------------------------
# 三、守卫：同一次运行里，一个 dump 点打 1 份，另一个打 block_num 份
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def probe_dump_guard(a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
    """两个 dump 点放在同一个 kernel 里，才能排除「是不是这次只启动了一个核」。

    `if pl.get_block_idx() == 0:` 是 kernel 体里的运行期分支——kernel 体是被解析的，
    这个 `if` 会编成设备上的条件跳转，不是 Python 期的分支。
    """
    g = pl.make_tile_group(type=_tile_type(), addrs=_A, mutex_ids=[0])
    with pl.section_vector():
        t = g.current()
        pl.load(t, a, [0, 0])
        if pl.get_block_idx() == 0:
            pl.dump_data(t, offsets=[0, 0], shapes=[1, 4], flag="GUARDED")
        pl.dump_data(t, offsets=[0, 0], shapes=[1, 4], flag="UNGUARDED")
        pl.store(out, t, [0, 0])


# ---------------------------------------------------------------------------
# 四、看 vf 寄存器的唯一一条路
# ---------------------------------------------------------------------------


@pl.vector_function
def _scale_by_100(src, dst):
    """vf 体内**不能**放 `pl.printf` / `pl.dump_data`（见 test.py 的两条负面用例）。

    想看 `r` 里是什么，就把它 `store_align` 出来，在外面 dump 那个 Tile。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    r = vf.load_align(src, 0)
    r = vf.muls(r, pl.const(100.0, FP32), preg)
    vf.store_align(dst, r, preg, 0)


@pl.jit(auto_mutex=True)
def probe_register_via_tile(a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
    """`dump_data` 在 vf 体外、`store_align` 之后——这时 Tile 里就是寄存器的内容。

    `flag=` 给 dump 点起名。多个 dump 点时没有 flag 就分不清哪份是哪个点的，
    尤其是每个点还要乘以核数份。
    """
    g = pl.make_tile_group(type=_tile_type(), addrs=_A, mutex_ids=[0])
    h = pl.make_tile_group(type=_tile_type(), addrs=_B, mutex_ids=[1])
    with pl.section_vector():
        t, out_tile = g.current(), h.current()
        pl.load(t, a, [0, 0])
        _scale_by_100(t, out_tile)
        if pl.get_block_idx() == 0:
            pl.dump_data(out_tile, offsets=[0, 0], shapes=[1, 4], flag="REGISTER")
        pl.store(out, out_tile, [0, 0])


# ---------------------------------------------------------------------------
# 五、标量写 GM 不能用来探每个核的状态
# ---------------------------------------------------------------------------


@pl.jit()
def probe_scalar_gm_writes(out: pl.Tensor[[128], I32]):
    """看起来最自然的「每个核把自己的 id 写进 out」——**大部分写会丢**。

    实测四次分别只有 7 / 6 / 6 / 5 个槽位落地（共 56 个核），每次丢的还不一样。
    同一批测试里 `probe_dump_guard` 的 UNGUARDED 打了 56 份，证明核确实都跑了，
    所以丢的是写，不是核。机制没查（大概是标量写 GM 的回写时机），但结论够用：
    **探每个核的状态用 `dump_data` / `printf`，不要手写标量落盘。**
    """
    with pl.section_vector():
        out[pl.get_block_idx()] = pl.get_block_idx() + 1000


# ---------------------------------------------------------------------------
# 六、pto_assert：条件不成立就中止
# ---------------------------------------------------------------------------
#
# **接口名是 `pl.pto_assert`，不是 `assert_` 也不是 `assert`**（`assert` 是 Python 关键字，
# 不能当函数名）。按 `assert` 去 grep API 会找不到它——ST 里有 22 次调用，目录是
# `pypto/python/tests/st/pypto_pro/frontend/assert/`。


@pl.jit()
def probe_assert_passing(
    out: pl.Tensor[[1], I32], flag: pl.DT_BOOL, offset: pl.DT_INT32
):
    """三种写法：裸条件、条件 + 消息、条件 + 格式化消息。条件成立时什么都不发生。

    `condition` 必须是 BOOL 标量；`format_str` 是编译期常量，和 `printf` 一样。
    """
    with pl.section_vector():
        pl.pto_assert(flag)
        pl.pto_assert(flag, "flag is false")
        pl.pto_assert(offset != 2, "offset=%d", offset)
        out[0] = 1


@pl.jit()
def _assert_failing_kernel(out: pl.Tensor[[1], I32], offset: pl.DT_INT32):
    """**实测：断言失败不中止。** `out[0]` 最后是 9，不是 7；host 侧收不到任何异常。

    设备日志打的是**条件的源码文本**：`Assertion failed: offset == 12345`，随后是格式化消息。
    每个核各打一份。

    `_api.py` 的 docstring 说 *"abort if condition is false"*，那句是错的；官方文档
    `pto_assert.md:20` 和框架自己的 ST 都写明只打日志、不中止。要真的停用 `pl.trap()`。
    """
    with pl.section_vector():
        out[0] = 7
        pl.pto_assert(offset == 12345, "PROBE_ASSERT_MSG offset=%d", offset)
        out[0] = 9  # ← 照样执行


# ---------------------------------------------------------------------------
# 七、trap：无条件中止
# ---------------------------------------------------------------------------


@pl.jit()
def probe_trap_unreached(out: pl.Tensor[[1], I32]):
    """`trap` 在永不成立的分支里——编进去了但不执行，kernel 正常结束。

    这一条能放进测试套；下面那条真触发的不能，见 `run_trap_in_this_process`。
    """
    with pl.section_vector():
        out[0] = 7
        if pl.get_block_idx() == 9999:
            pl.trap()


@pl.jit()
def _trap_kernel(out: pl.Tensor[[1], I32]):
    with pl.section_vector():
        out[0] = 7
        pl.trap()
        out[0] = 9  # trap 之后的代码不会执行


# ---------------------------------------------------------------------------
# 八、子进程入口：断言设备打印内容的唯一可靠途径
