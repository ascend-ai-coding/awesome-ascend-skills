# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""五种取整、三个数值谓词，以及 `cast` / `bitcast` / `popcount` / `mul_hi` / `fmod`。

**写法：原始。**

## 一、最小的 SIMT kernel：GM 张量直接进去

SIMT 函数可以直接收 GM 张量——不需要 `make_tile` 再搬。所以这一例的 kernel 体只有一条启动，
**没有 tile、没有同步**：

```python
@pl.jit()
def kernel(src, out):
    with pl.section_vector():
        body[THREADS](src, out)
```

前一例 ``simt_launch`` 用了 tile（因为要让线程之间交换数据），这一例不需要。**两种都对**,
挑哪种看线程之间要不要共享 UB。

## 二、`rint` 与 `round` 在 `.5` 上取法相反（实测）

| | `.5` 怎么办 | `-2.5` | `-0.5` | `0.5` | `2.5` |
| --- | --- | ---: | ---: | ---: | ---: |
| ``pl.simt.rint`` | **就近取偶** | `-2` | `0` | `0` | `2` |
| ``pl.simt.round`` | **远离零** | `-3` | `-1` | `1` | `3` |

实测它们在 32 个输入里**恰好在 4 个位置不同**，全是 `.5`（`±1.5` 不在里面：`2` 是偶数,
两种取法给出同一个答案）。验证 3 直接断言"这 4 个位置不同"——**比各自对上参考更有力**,
因为它把两个算子的**区别**本身变成了判据。

## 三、`cast(FP32 → INT32)` 是**截断**，不是就近（实测）

`pl.simt.cast` 的 `mode` 缺省是 `RoundMode.CAST_NONE`，实测结果和 `trunc` 逐位相同、
和 `rint` 不同：`1.7 → 1`、`-1.7 → -1`。

**这是个容易错的地方**：一个叫 `cast` 的接口配一个叫 `CAST_NONE` 的缺省值，读不出"向零截断"。
判据同时断言"等于 trunc"和"不等于 rint"，两条一起才说明它是截断而不是碰巧。

`cast` 与 `bitcast` 的区别和 GM 侧 `pl.make_tensor(dtype=)`、numpy 的 `astype` / `view` 是同一个：

| | `1.5` 变成 |
| --- | --- |
| ``pl.simt.cast(v, I32)`` | `1` |
| ``pl.simt.bitcast(v, I32)`` | `1069547520` |

## 四、如何独立验证

1. `floor` / `ceil` / `trunc` / `fmod` 逐位等于 numpy 的同名函数。
2. `rint` 等于就近取偶，`round` 等于远离零（各自的参考）。
3. **两者恰好在 4 个 `.5` 位置不同**，其余 28 个位置相同。
4. `isnan` / `isinf` / `isfinite` 三个谓词逐位等于 numpy（输入里有 `nan` / `±inf`）。
5. `cast` 在有限值上等于 `trunc`，且**不等于** `rint`。
6. `bitcast` 等于 numpy 的 `view(np.int32)`。
7. `popcount` 等于 `bin(v).count("1")`（输入含 0 位、单高位、全 1）。
8. `mul_hi` 等于 `(int64(v) * int64(v)) >> 32`（输入含负数和会溢出的值）。

输入前 16 个是**设计过的边界**（`±0.5 / ±1.5 / ±2.5 / ±1.7 / ±0.0 / nan / ±inf`），后 16 个是
普通值，确认特殊处理没把普通情况带坏。`mul_hi` 的输入含 `65536`（平方正好溢出一位）和负数
——**不溢出的话"高半"永远是 0，什么都验不到**。

## 五、`np.array_equal` 遇到 NaN 一律返回 False

写这一例时第一轮所有浮点判据都红，而逐值打印出来前 13 个全对。原因不在硬件：
**输入里有 `nan`，而 `np.array_equal(a, b)` 在任一侧有 NaN 时返回 `False`**（NaN != NaN）。

要比就得 `np.array_equal(a, b, equal_nan=True)`，或者用
`np.testing.assert_array_equal`（它自己把 NaN 当相等）。

**这条值得写下来**：它会让一批正确的判据一起变红，而第一反应是去查被测的东西。
本仓的判据设计原则里因此多一条：**输入含 NaN 时，比较函数要显式选定 NaN 的语义**。

## 六、这一例不讲什么

- **超越函数与精度**——`sqrt` / `exp` / `sin` 这些在 ``simt_math``，那一例标定了容差。
- **`cast` 的其他 `mode`**——`CAST_ODD`（只支持 FP32→FP16）、饱和模式（docstring 说"尚未暴露"）
  **没测**。这一例只用缺省的 `CAST_NONE`。**这是明说跳过。**
- **`cast(nan)` / `cast(inf)` 到整数**——结果没有定义，判据用 `finite_mask` 把那三列排除了。
  **没有声称它们是什么。**
- **UINT64 / INT64 的 `popcount` / `mul_hi`**——只测了 32 位那一半。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 三节。
"""

import numpy as np
import torch

from . import reference as ref
from .kernel import (
    ROW_BITCAST,
    ROW_CAST,
    ROW_CEIL,
    ROW_FLOOR,
    ROW_FMOD,
    ROW_ISFINITE,
    ROW_ISINF,
    ROW_ISNAN,
    ROW_MUL_HI,
    ROW_POPCOUNT,
    ROW_RINT,
    ROW_ROUND,
    ROW_TRUNC,
    ROWS_B,
    ROWS_F,
    ROWS_I,
    THREADS,
    simt_round_and_predicates,
)


def _run(device):
    f, u, i = ref.make_inputs()
    out_f = torch.zeros((ROWS_F, THREADS), dtype=torch.float32, device=device)
    out_b = torch.zeros((ROWS_B, THREADS), dtype=torch.bool, device=device)
    out_i = torch.zeros((ROWS_I, THREADS), dtype=torch.int32, device=device)
    simt_round_and_predicates[None, 1](
        torch.from_numpy(f).to(device),
        torch.from_numpy(u.astype(np.int32)).to(device).to(torch.uint32),
        torch.from_numpy(i).to(device),
        out_f,
        out_b,
        out_i,
    )
    torch.npu.synchronize()
    return (
        out_f.cpu().numpy(),
        out_b.cpu().numpy(),
        out_i.cpu().numpy().astype(np.int32),
        f,
        u,
        i,
    )


def test_floor_ceil_trunc_and_fmod_match_numpy(device):
    """验证 1。四个都是精确定义的运算，`rtol=0`。

    用 `assert_array_equal` 而不是 `np.array_equal`：**前者把 NaN 当相等**（§五），
    而输入里有 `nan`。
    """
    got_f, _, _, f, _, _ = _run(device)
    flat = f.astype(np.float64).ravel()
    np.testing.assert_array_equal(got_f[ROW_FLOOR], np.floor(flat).astype(np.float32))
    np.testing.assert_array_equal(got_f[ROW_CEIL], np.ceil(flat).astype(np.float32))
    np.testing.assert_array_equal(got_f[ROW_TRUNC], np.trunc(flat).astype(np.float32))
    np.testing.assert_array_equal(got_f[ROW_FMOD], ref.fmod_trunc(f))


def test_rint_rounds_half_to_even_and_round_rounds_away_from_zero(device):
    """验证 2。两个算子各自对上自己的参考。"""
    got_f, _, _, f, _, _ = _run(device)
    np.testing.assert_array_equal(got_f[ROW_RINT], ref.rint_half_to_even(f))
    np.testing.assert_array_equal(got_f[ROW_ROUND], ref.round_half_away(f))


def test_the_two_rounding_modes_differ_exactly_on_the_half_cases(device):
    """验证 3，这一例最有力的一条：**把两个算子的区别本身变成判据**。

    验证 2 只说明各自对上了参考——参考写错了它也可能通过。这一条要求两者**恰好在 `.5` 的
    位置不同**：那些位置的输入必须满足 `frac(|x|) == 0.5` 且取整到奇数（`±1.5` 不在里面,
    因为 `2` 是偶数，两种取法给出同一答案）。
    """
    got_f, _, _, f, _, _ = _run(device)
    flat = f.ravel()
    # **必须先排除非有限值**：`nan != nan`，所以 `nan` 那一列永远落在 "differ" 里而它跟取整无关。
    # 第一版忘了这一步，判据以 "两者在 nan 上不同，但它不是 .5 的情形" 失败——
    # 一条对的规则被一个错的掩码打红了。
    finite = np.isfinite(flat)
    differ = (got_f[ROW_RINT] != got_f[ROW_ROUND]) & finite

    assert differ.any(), "rint 和 round 在所有输入上都相同 —— 输入里没有 .5 的情形？"
    for value in flat[differ]:
        assert abs(value) % 1.0 == 0.5, f"两者在 {value} 上不同，但它不是 .5 的情形"
    # 不同的那些位置，rint 给偶数、round 的绝对值更大
    assert (got_f[ROW_RINT][differ] % 2 == 0).all(), "rint 在 .5 上没有取偶"
    assert (np.abs(got_f[ROW_ROUND][differ]) > np.abs(got_f[ROW_RINT][differ])).all(), (
        "round 在 .5 上没有远离零"
    )


def test_the_three_predicates_match_numpy(device):
    """验证 4。输入里有 `nan` / `±inf`，三个谓词各自对上。

    三个一起验才有意义：只验 `isnan` 的话，一个"永远返回 False"的实现在没有 NaN 的输入上
    也能通过，而 `isfinite` 会立刻抓住它。
    """
    _, got_b, _, f, _, _ = _run(device)
    flat = f.ravel()
    np.testing.assert_array_equal(got_b[ROW_ISNAN], np.isnan(flat))
    np.testing.assert_array_equal(got_b[ROW_ISINF], np.isinf(flat))
    np.testing.assert_array_equal(got_b[ROW_ISFINITE], np.isfinite(flat))


def test_cast_truncates_toward_zero(device):
    """验证 5。`cast` 等于 `trunc`，**且不等于** `rint`。

    两条一起才说明它是截断：只断言"等于 trunc"的话，在一组恰好没有小数的输入上
    "就近取整"也会通过。`nan` / `inf` 三列排除掉——`cast` 到整数在那里没有定义，
    **不声称它们是什么**。
    """
    _, _, got_i, f, _, _ = _run(device)
    finite = ref.finite_mask(f)
    flat = f.ravel()[finite]

    np.testing.assert_array_equal(
        got_i[ROW_CAST][finite], np.trunc(flat).astype(np.int32)
    )
    assert not np.array_equal(
        got_i[ROW_CAST][finite], np.rint(flat).astype(np.int32)
    ), "cast 的结果和就近取整一样 —— 输入里没有能区分两者的小数？"


def test_bitcast_reinterprets_the_bits(device):
    """验证 6。`bitcast` 等于 numpy 的 `view`，**不是** `astype`。"""
    _, _, got_i, f, _, _ = _run(device)
    np.testing.assert_array_equal(got_i[ROW_BITCAST], ref.bitcast_to_i32(f))


def test_popcount_counts_set_bits(device):
    """验证 7。输入含 0 位、单独的高位、全 1（`0xFFFFFFFF` → 32）。"""
    _, _, got_i, _, u, _ = _run(device)
    np.testing.assert_array_equal(got_i[ROW_POPCOUNT], ref.popcount(u))
    assert got_i[ROW_POPCOUNT].max() == 32, (
        "没有一个输入的位数是 32 —— 全 1 那一列丢了？"
    )


def test_mul_hi_returns_the_upper_half(device):
    """验证 8。输入含 `65536`（平方正好溢出一位）和负数。

    **不溢出的话"高半"永远是 0**，那样这条判据在一个恒返回 0 的实现上也会通过。
    """
    _, _, got_i, _, _, i = _run(device)
    np.testing.assert_array_equal(got_i[ROW_MUL_HI], ref.mul_hi_i32(i))
    assert (got_i[ROW_MUL_HI] != 0).any(), "所有高半都是 0 —— 输入里没有会溢出的值？"
