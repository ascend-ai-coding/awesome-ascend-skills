# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""饱和标志：读回、极性，以及全局模式与单指令模式谁说了算。

**写法：原始。** 两个 API **全无 ST**，每个数字都是 2026-09-28 在 Ascend950PR 上量的。

## 一、两个接口，四个模式

```python
pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=True)   # 两个参数都是 keyword-only
flag = pl.get_saturation_flag(mode=pl.SaturationFlagMode.CAST)         # -> bool
```

| 模式 | CTRL 位 | 管什么 |
| --- | ---: | --- |
| `FLOAT` | 48 | 浮点计算与浮点精度转换 |
| `FLOAT8` | 50 | float8 的 NaN 处理 |
| `INT` | 53 | 整数指令 |
| `CAST` | 59 | 浮点转整数 / 整数转整数 |

两个都只能在 `@pl.jit` 体里调，进不了 `@pl.vector_function`（验证 5）。

## 二、如何独立验证

1. 四个模式各自的 `set → get` 往返：`enable=True` 后 flag 为真，`False` 后为假。
2. **极性**：对 FLOAT / FLOAT8 / CAST，`enable=True` 让 CTRL 位变成 **0**；对 INT 变成 1。
3. 入口默认：FLOAT **关**，FLOAT8 / CAST 开，INT 关。
4. **六种配置的真值表**（§四）：CTRL[60] 决定听 `set_saturation_flag` 还是听 `astype(saturate=)`。
5. 两条被拒的写法：标志设在 vf 体里、两个参数不用 keyword。

验证 2 的判据把 C++ 的 `IsInvertedPolarity` 和实测的位值对起来。**写在 `kernel.py` 的
`MODES` 表里那一列 `inverted` 是抄来的，判据是量来的**——抄错了会红。

## 三、极性对三个模式是反的

`backend_cce_ops.cpp` 的 `IsInvertedPolarity` 原话：

> FLOAT/FLOAT8/CAST: polarity inverted (bit=0 means saturation ON, bit=1 means OFF)
> INT: polarity normal (bit=1 means saturation ON, bit=0 means OFF)

实测（`enable=` → CTRL 位）：

| 模式 | 位 | `enable=True` | `enable=False` | 入口时 flag |
| --- | ---: | ---: | ---: | --- |
| FLOAT | 48 | **0** | 1 | **关** |
| FLOAT8 | 50 | **0** | 1 | 开 |
| INT | 53 | 1 | 0 | 关 |
| CAST | 59 | **0** | 1 | 开 |

**混用两个接口时这一点会骗人**：`set_saturation_flag(mode=CAST, enable=True)` 之后
`get_ctrl_spr(59, 59)` 读到 0，看着像没生效。`get_saturation_flag` 自己是对称的
（读的时候也反回来），单用它不会错。

「FLOAT 入口时是关的」这一条不是独立事实：它就是 ``spr_ctrl`` §三那件事的另一个说法——
bit 48 入口为 1，而 FLOAT 极性反转，所以 flag 读作关。**两例说的是同一个位。**

`_api.py` 的 docstring 说 `enable` 是「`True` to enable saturation (clamp...), `False` to
disable (truncate)」——**这句话在语义层是对的**，验证 1 和 §四 都印证了。它没说的是
CTRL 位不等于 flag 值，而那正是会让人误判的地方。三份证据这里是一致的
（官方位表、C++ 实现、实测），只是**官方位表描述的是位，docstring 描述的是语义**,
读的时候得知道各自在说哪一层。

## 四、CTRL[60]：全局模式还是单指令模式（实测六行）

FP32 → INT16，输入 `[50000, -50000, 100000, -100000, 100, -100]`：

| 饱和 | `[32767, -32768, 32767, -32768, 100, -100]` |
| 截断 | `[-15536, 15536, -31072, 31072, 100, -100]` |

| CTRL[60] | `set_saturation_flag(CAST)` | `astype(saturate=)` | 实测 | 判别 |
| ---: | --- | --- | --- | --- |
| 1 | `enable=True` | `OFF` | 饱和 | |
| 1 | `enable=False` | `OFF` | 截断 | |
| 1 | `enable=True` | `ON` | 饱和 | |
| 1 | `enable=False` | **`ON`** | **截断** | **✓ `saturate` 被忽略** |
| 0 | `enable=False` | **`ON`** | **饱和** | **✓ `saturate` 说了算** |
| 0 | **`enable=True`** | `OFF` | **截断** | **✓ 标志被忽略** |

**后三行才是证据**，前三行两种模型都能解释。第 4 行和第 5 行是同一组参数
（`enable=False` + `saturate=ON`）只改 CTRL[60]，结果相反——这一对最干净。

而 **CTRL[60] 的入口值是 1**（见 ``spr_ctrl``），所以：

> 默认情况下 `vf.astype(saturate=...)` 不起作用，饱和由 `set_saturation_flag` 决定。

**这是这一例最该记住的一句。** 写 kernel 时给 `astype` 传了 `saturate=ON` 却没生效,
第一反应会是「参数没传对」，而实际要改的是 CTRL[60]。

范围内的两个值（100、-100）在六种配置下都相同——**这是控制项**：如果它们也变了,
说明变的不是「越界怎么办」，那整张表的解释就得推翻。

## 五、这一例不讲什么

- **CTRL 寄存器本身**——位布局、可写位清单、`reset` 恢复的是哪个值，在 ``spr_ctrl``。
  这一例只用 `get_ctrl_spr` 当**观测手段**，用 `set_ctrl_spr(60, 60, ...)` 当自变量。
- **`vf.astype` 的其余参数面**——`layout` / `round_mode` / `mode` 在 ``vf/vf_lane_layout``。
  这里固定 `layout=CastLayout.ZERO`、`round_mode` 用缺省，因为探针值都精确可表示。
- **FLOAT / FLOAT8 / INT 三个模式的数值效果**——**只验了标志的读回与极性，没验效果**。
  效果要各自构造越界输入：FLOAT 要 inf/NaN 的低精度转换，FLOAT8 要 FP8 的 NaN,
  INT 要整数指令溢出。三条都能做，但 `CAST` 这一条已经把「标志 → 数值」的机制钉住了,
  另外三条是同一机制的不同位。**这是明说跳过，不是「验过一样」。**
- **饱和对性能的影响**——没测。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 四节。
"""

import numpy as np
import pytest
import torch

import pypto_pro as pp
import pypto_pro.language as pl
from pypto_pro._errors import InvalidArgument, NotSupported

from .kernel import (
    CONFIGS,
    MODES,
    N,
    make_narrowing_kernel,
    make_polarity_probe,
    set_flag_inside_a_vector_function,
    set_flag_without_keywords,
)
from .reference import IN_RANGE_FROM, PROBE, modes_differ_on, saturated, truncated

_MODE_IDS = [row[0] for row in MODES]


def _run_probe(mode, bit, device):
    out = torch.full((6,), -1, dtype=torch.int64, device=device)
    make_polarity_probe(mode, bit)[None, 1](out)
    torch.npu.synchronize()
    return [int(v) for v in out.cpu().numpy()]


def _run_narrowing(kernel, device):
    a = np.zeros(N, dtype=np.float32)
    a[: len(PROBE)] = PROBE
    out = torch.zeros((1, N), dtype=torch.float32, device=device)
    kernel[None, 1](torch.from_numpy(a).reshape(1, N).to(device), out)
    torch.npu.synchronize()
    return out.cpu().numpy()[0][: len(PROBE)]


def _parse(kernel):
    kernel.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# 一、读回与极性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "mode", "bit", "inverted", "entry_saturating"), MODES, ids=_MODE_IDS
)
def test_the_flag_round_trips(device, name, mode, bit, inverted, entry_saturating):
    """验证 1。`enable=True` 后读到真，`False` 后读到假。四个模式各一次。

    **这条不碰 CTRL 位**，只用 `get_saturation_flag`。单用它是对称的,
    所以这一条和极性那件事无关——它验的是两个接口自己成对。
    """
    _, _, on_flag, _, off_flag, _ = _run_probe(mode, bit, device)
    assert on_flag == 1, f"{name}: enable=True 之后 get_saturation_flag 读到 {on_flag}"
    assert off_flag == 0, (
        f"{name}: enable=False 之后 get_saturation_flag 读到 {off_flag}"
    )


@pytest.mark.parametrize(
    ("name", "mode", "bit", "inverted", "entry_saturating"), MODES, ids=_MODE_IDS
)
def test_the_ctrl_bit_polarity_matches_the_implementation(
    device, name, mode, bit, inverted, entry_saturating
):
    """验证 2。FLOAT / FLOAT8 / CAST 的位是反的，INT 是正的。

    `kernel.py` 的 `MODES` 里那一列 `inverted` 抄自 C++ 的 `IsInvertedPolarity`,
    这条判据拿实测的位值去对它——**抄错了会红**，所以它验的是「抄对了」而不是「猜对了」。
    """
    _, _, _, on_bit, _, off_bit = _run_probe(mode, bit, device)
    expected_on, expected_off = (0, 1) if inverted else (1, 0)
    assert (on_bit, off_bit) == (expected_on, expected_off), (
        f"{name}(CTRL[{bit}]): enable=True/False 读到位值 {(on_bit, off_bit)}，"
        f"按 {'反' if inverted else '正'}极性应为 {(expected_on, expected_off)}"
    )


@pytest.mark.parametrize(
    ("name", "mode", "bit", "inverted", "entry_saturating"), MODES, ids=_MODE_IDS
)
def test_the_entry_default_of_each_mode(
    device, name, mode, bit, inverted, entry_saturating
):
    """验证 3。入口默认：FLOAT 关，FLOAT8 / CAST 开，INT 关。

    期望值从 `MODES` 表最后一列取，那一列是实测的。**FLOAT 和 INT 那两个「关」来路不同**：
    FLOAT 是「位为 1 + 极性反转」，INT 是「位为 0 + 极性正常」。判据的后半段拿位值
    按极性推一遍，和 flag 对上——两个接口自己得自洽。

    **这一列和 `inverted` 一开始被我并成了一列**，这条判据当场红了。留着这句话:
    「入口是否饱和」和「极性是否反转」是两件事，合并它们会得出 INT 极性反转的错结论。
    """
    default_flag, default_bit, *_ = _run_probe(mode, bit, device)
    assert bool(default_flag) is entry_saturating, (
        f"{name}: 入口 flag 读到 {default_flag}，实测是 {int(entry_saturating)}"
    )
    derived = (default_bit == 0) if inverted else (default_bit == 1)
    assert derived is bool(default_flag), (
        f"{name}: CTRL[{bit}]={default_bit} 按极性推出 {derived}，"
        f"但 get_saturation_flag 读到 {bool(default_flag)} —— 两个接口自己对不上"
    )


# ---------------------------------------------------------------------------
# 二、标志改不改数值结果
# ---------------------------------------------------------------------------


def test_the_probe_input_can_tell_the_two_modes_apart():
    """**先证明仪器能分辨**，再读数（`references/verification/board-runs.md` §七）。

    这条不上板：它只检查参考里那两个函数在探针输入上确实不同，而且**恰好在越界的那四个位置
    不同**。如果探针值选得不好（全在范围内），下面四条判据会全绿而什么都没验到。
    """
    np.testing.assert_array_equal(modes_differ_on(), np.arange(IN_RANGE_FROM))
    np.testing.assert_array_equal(
        saturated()[IN_RANGE_FROM:], truncated()[IN_RANGE_FROM:]
    )


@pytest.mark.parametrize(
    ("global_bit", "enable", "saturate", "decisive"),
    CONFIGS,
    ids=[
        f"ctrl60={g}-enable={e}-saturate={str(s).split('.')[-1]}"
        for g, e, s, _ in CONFIGS
    ],
)
def test_who_decides_saturation(device, global_bit, enable, saturate, decisive):
    """验证 4。六种配置，每一种落在饱和或截断上。

    期望值由官方那句约束推出来：**`CTRL[60]=1` 听 `set_saturation_flag`,
    `CTRL[60]=0` 听 `astype(saturate=)`。** 六行全部吻合。

    后三行标了 `decisive`——它们是把两种模型分开的那几行。前三行两种解释都成立,
    单独看它们证不到任何东西，留着是为了让表完整（也为了万一哪天它们红了）。
    """
    # 官方那句约束直接写成代码：全局模式看标志，单指令模式看参数。
    expect_saturating = (
        bool(enable) if global_bit == 1 else saturate is pl.SaturateMode.ON
    )
    expected = saturated() if expect_saturating else truncated()
    got = _run_narrowing(
        make_narrowing_kernel(
            global_bit=global_bit, cast_enable=enable, saturate=saturate
        ),
        device,
    )
    np.testing.assert_array_equal(
        got,
        expected,
        err_msg=(
            f"CTRL60={global_bit} enable={enable} saturate={saturate}："
            f"期望{'饱和' if expect_saturating else '截断'}，实际 {got.tolist()}"
        ),
    )


def test_the_in_range_values_are_untouched_by_the_mode(device):
    """验证 4 的控制项。范围内的两个值在六种配置下都一样。

    **没有这条，整张表可以被别的解释掉**：比如「某个配置下 astype 整体坏了」。
    范围内的值不变，说明变的确实只是「越界怎么办」。
    """
    seen = []
    for global_bit, enable, saturate, _ in CONFIGS:
        got = _run_narrowing(
            make_narrowing_kernel(
                global_bit=global_bit, cast_enable=enable, saturate=saturate
            ),
            device,
        )
        seen.append(got[IN_RANGE_FROM:].tolist())
    assert all(row == seen[0] for row in seen), f"范围内的值随配置变了：{seen}"
    np.testing.assert_array_equal(
        np.array(seen[0], dtype=np.float32), PROBE[IN_RANGE_FROM:]
    )


# ---------------------------------------------------------------------------
# 三、被拒的写法
# ---------------------------------------------------------------------------


def test_setting_the_flag_inside_a_vector_function_is_refused():
    """验证 5 之一。标志设在 vf 体里被拒。

    这一步很自然会想错：饱和影响的是 vf 指令，**所以会想在 vf 里设**。
    但它是标量流水上的全局状态，设它的地方在 kernel 体。
    """
    with pytest.raises(NotSupported) as raised:
        _parse(set_flag_inside_a_vector_function)
    assert "is not supported inside @pl.vector_function" in str(raised.value), str(
        raised.value
    )


def test_positional_arguments_are_refused():
    """Keyword-only errors identify the public parameters and their call site."""
    with pytest.raises(InvalidArgument) as raised:
        _parse(set_flag_without_keywords)
    assert "keyword-only" in raised.value.message
    assert "mode" in raised.value.message and "enable" in raised.value.message
    assert raised.value.span is not None
