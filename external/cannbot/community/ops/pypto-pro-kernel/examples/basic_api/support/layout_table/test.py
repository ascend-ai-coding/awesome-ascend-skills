# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""六个布局在各存储空间的合法组合，以及 `set_mm_layout_transform`。

**写法：原始。**

## 一、布局的合法性有两层检查

布局的合法性有**三层**检查，在三个地方，按这个顺序拦：

| 层 | 在哪 | 问的是 | 撞上它的例子 |
| --- | --- | --- | --- |
| 1. 算子操作数 | `block_ops.py:1388` / `:559` | `load` 的 dst、`store` 的 src 在哪个空间 | `load` 直进 L0A |
| 2. tile 声明 | `block_ops.py:1995` / `:3829` | 这个空间能不能声明成这个布局、这个形状 | `Mat(NN)` + FP16 |
| 3. 数据通路 | `_A5_LOAD_COMBOS` 等三张表 | (空间, 布局, dtype) 组合能不能搬 | `GM(ND) → Mat(ZN)` |

**写这一例时是一层一层撞出来的**，顺序恰好是倒着的：先按压掉 dtype 的通路表选布局,
撞第 3 层；改对之后撞第 2 层（`Mat(NN)` 根本声明不出来）；最后两条反例撞第 1 层
（而我按第 3 层的消息写了判据）。三层都要讲，而且要讲清顺序——
只知道一层的人会在另一层的报错前面找不着方向。

**前两层的消息质量明显好**：第 1 层直说「dst 该在 Vec 或 Mat」,
第 2 层带许可集和缺省值；第 3 层只说 `unsupported layout/dtype combination`,
要自己去源码里查表。所以这一例的三条判据分别断言三种消息——
**看到哪种消息决定了下一步该查什么**。

## 二、第一层：tile 声明（A5，FP16）

| 空间 | 缺省 | 可声明 |
| --- | --- | --- |
| `Mat`（L1） | `NZ` | `NZ` `ZN` `ND` `DN` |
| `Left`（L0A） | `NZ` | `NZ` |
| `Right`（L0B） | `ZN` | `ZN` |
| `Acc`（L0C） | `NZ` | `NZ` |
| `Bias` | `ND` | `ND` |
| `Scaling` | `ND` | `ND` |
| `ScaleLeft` | `ZZ` | `ZZ` |
| `ScaleRight` | `NN` | `NN` |

**`Bias` 还有一条形状约束**：必须恰好一行（`block_ops.py:3829`，硬件 `Rows==1`）。
它的探针用 `[1, K]`，别的空间用 `[16, 64]`。

**`Mat` 的 `ZZ` / `NN` 只在 `dtype=FP8E8M0` 时可声明**（MX 缩放视图，`fractal` 被强制成 32）。
所以「L1 接受全部六个布局」要加 dtype 条件才成立——验证 2 把两个方向都测了。

## 三、如何独立验证

1. 每个空间：缺省布局能声明、许可集里的能声明、不在许可集里的被拒。
2. `Mat(ZZ)` / `Mat(NN)`：`FP16` **被拒**，`FP8E8M0` **通过**。
3. 两种拒绝消息分开：`Left(ZZ)` 说 "do not support ZZ layout on 'a5'",
   `Left(ND)` 说 "require layout in {NZ}"。
4. `SPACE_LAYOUTS` / `OP_SPACE_PAIRS` / `FP16_LAYOUT_PAIRS` 与三张组合表逐项相等。
5. ``make_gm_to_l1(NZ)`` 跑得通；`ZN` **被拒**（第二层）。
6. ``make_l1_to_l0``：`L1(NZ) → L0A(NZ)` 与 `L1(NZ) → L0B(ZN)` 都跑得通。
7. ``make_with_layout_transform(True/False)``：两种都跑得通，结果相同。
8. 三条被拒的写法，**各撞一层**：`load` 直进 L0A（第 1 层）、L1 直接 `store`（第 1 层）、
   `enabled` 不用 keyword（参数解析）。
9. 布局开关放进 vector 段**不被拒**——没有任何一层拦它。

**判据 1 到 4 不上板**：布局规则是**静态**的，拿板上跑一遍只能证明「这几条能跑」,
证不到「只有这几条能跑」。判据 5 的「被拒」那一半才是价值所在。

## 四、压掉 dtype 维只给**必要**条件（这一例栽的第一跤）

`SPACE_LAYOUTS` 是把 dtype 压掉的，它回答「一个空间**可能**持有哪些布局」,
**不回答「从 X 布局搬到 Y 布局行不行」**。

| | 压掉 dtype 看到的 `GM → Mat` | **FP16 实际** |
| --- | --- | --- |
| | `DN→NN` `DN→NZ` `DN→ZN` `DN→ZZ` `ND→ND` `ND→NN` `ND→NZ` `ND→ZZ` `NZ→NZ` | `DN→NZ` `DN→ZN` `ND→NZ` `NZ→NZ` |

`ND→ND`、`ND→NN`、`ND→ZZ` **全来自 8 bit / 4 bit 的行**。所以 FP16 从一个 `ND` 的 GM
张量出发**只能落到 `Mat(NZ)`**。拿压过的表当充分条件用，三条判据当场红了：

```
load: unsupported layout/dtype combination src=GM(ND,DT_FP16) -> dst=Mat(ZN,DT_FP16)
```

两张表都留着，各答一问。**把它们合成一张会再犯一次同样的错。**

## 五、第一层比第二层先拦（这一例栽的第二跤）

改成「按 FP16 的通路表选布局」之后，`Mat(NN)` 那条又红了，但报错换了个地方：

```
Mat tiles require layout in {DN, ND, NZ, ZN}, got NN. Default for 'a5' is NZ.
```

`block_ops.py:1995`——**这是 tile 声明那一层**，跟搬运没关系。通路表里确实有
`Mat(NN) → ScaleRight(NN)` 这一行，但那一行的源 tile 要 `dtype=FP8E8M0` 才声明得出来。

教训：**「表里有这一行」不等于「这个 tile 声明得出来」。** 两层检查的规则不同,
读了一层的源码就去写 kernel，会在另一层的报错前面找不着方向。

## 六、两种拒绝消息（判据 3）

| 情形 | 消息 | 规则 |
| --- | --- | --- |
| `Left`/`Right`/`Acc` 的 `ZZ`/`NN` | `tiles do not support ZZ layout on 'a5'; ...` | `_REJECTED_LAYOUTS_ON_A5` 那条单独的 |
| 其余不在许可集里的 | `tiles require layout in {NZ}, got ND.` | 许可集那条 |

为什么分两条：`Left` 的许可集算出来是 `{ZZ, NZ}`（**A3 的缺省是 `ZZ`**，A5 是 `NZ`,
算法把两代的缺省都放进候选），所以 `ZZ` 过得了许可集那一关,
得靠前面那条 A5 专用规则拦。**这是两代硬件共用一份代码留下的痕迹**,
而它的可观测后果就是消息不一样。

判据 3 分别断言两段文本。合成一条「反正会拒」会丢掉这个区别,
而下一个人看到 "do not support ... on 'a5'" 时最想知道的正是「那 A3 上行吗」。

## 七、`set_mm_layout_transform` 只验到「被接受」

```python
pl.system.set_mm_layout_transform(enabled=True)     # fixpipe 按 N 方向排出 L0C
pl.system.set_mm_layout_transform(enabled=False)    # 恢复 M 方向
```

开了之后 cube 和 fixpipe 在同一个 L0C 槽里沿**正交的轴**访问，于是没有 RAW 冲突,
单缓冲 L0C 也能做 K 累加（`system_ops.py:520` 的 docstring）。

**这一例给不出它的数值判据**，和 ``datamove/mutex_dcci`` 的 `dcci` 同一类：
要看见差别得有一个 K 累加的 matmul，并且 L0C 单缓冲——那是 `examples/matmul/` 的题目。
验证 7 断言「开着和关着结果相同」，也就是**它不会把对的弄坏**。
**不假装验到了排出方向。**

ST 里的用法都是**开了再关**，这一例照做。**不关会不会影响后面的 kernel 没量过**——
``support/spr_ctrl`` §四 量到 CTRL 寄存器不跨 launch 残留，但这个开关是不是同一个机制,
没有独立证据。**这是明说跳过。**

**放进 vector 段没有任何一层拦**（判据 9）。写这一例时我把它当反例，
判据红在「DID NOT RAISE」——收着它是为了钉住这里没有检查。

## 八、两处消息里的小错

| 出处 | 消息 | 问题 |
| --- | --- | --- |
| `block_ops.py:3829` | `Use shape=[1, N] for the Bias (L0B) tile.` | **`Bias` 不是 L0B**，L0B 是 `Right` |
| `ir.pyi` 的 `TensorLayout` | 只列了 `ND` / `DN` / `NZ` | stub 不全，`pl` 导出六个 |

两条都不影响行为，但都会把读的人带偏：前者让人以为 `Bias` 和 `Right` 是同一块,
后者让人以为只有三个布局。按 [skill constraints](../../../../SKILL.md#constraints-and-workspace-policy) 都落在 `pypto/`。

## 九、`TensorLayout` 的 stub 不全

`pypto/python/pypto/pypto_impl/ir.pyi` 的 `class TensorLayout` **只列了 `ND` / `DN` / `NZ`**,
而 `pl` 导出六个（`language/__init__.py:476-481`）。按 stub 写代码的人会以为只有三个。
判据 4 顺手钉住六个都在。

## 十、这一例不讲什么

- **布局到底怎么排字节**——`NZ` 的分形块大小、`ZN` 的转置关系这些。这一例只管
  「哪个空间接受哪个布局」，不管布局自己的内存排布。
- **dtype 维的全貌**——三张表的每一行还带源／目的 dtype，`load` 那张 477 行里绝大部分
  差别在 dtype。dtype 面在 ``support/dtype_carriers``。这一例只取 FP16 一个切片,
  **外加 `Mat(ZZ/NN)` 需要 FP8E8M0 这一条**。
- **`quant` 那张额外的 store 表**——`_A5_STORE_QUANT_COMBOS` 没进这一例，量化的 store
  在 fixpipe 一章。
- **A3 上的规则**——只验 A5。`Left` 的 A3 缺省是 `ZZ` 这件事从源码读到,
  **没有 A3 机器可测**。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §二 四节。
"""

from collections import defaultdict

import numpy as np
import pytest
import torch

import pypto_pro as pp
import pypto_pro.language as pl
from pypto_pro._errors import InvalidOperation, InvalidType, InvalidVal
from pypto_pro.ir.op import block_ops

from .kernel import (
    FP8E8M0,
    K,
    M,
    layout_transform_in_a_vector_section,
    layout_transform_without_the_keyword,
    load_straight_into_l0a,
    make_declaration_probe,
    make_default_layout_probe,
    make_gm_to_l1,
    make_l1_to_l0,
    make_with_layout_transform,
    store_from_l1_to_gm,
)
from .reference import (
    DECLARABLE,
    DEFAULT_LAYOUT,
    FP16_LAYOUT_PAIRS,
    LAYOUTS,
    MAT_TO_L0_TARGET,
    MX_SCALE_ONLY,
    OP_SPACE_PAIRS,
    SPACE_LAYOUTS,
    TABLE_ROWS,
    declarable_layouts,
    refusal_kind,
    spaces_with_layout,
)

ROW_WIDTH = 6
"""组合表每行的元组长度：(源空间, 目的空间, 源布局, 目的布局, 源 dtype, 目的 dtype)。"""

ROW_TOLERANCE = 0.2
"""行数的容差。判据只断言量级相符——差一两行是正常演进，不该让判据红。"""

DECLARING_SPACES = sorted(DECLARABLE)

SPACE_ROWS = {"Bias": 1}
"""按空间给 tile 行数。**`Bias` 必须恰好一行**（硬件 `Rows==1`），其余用缺省的 16。"""


def _rows(space_name):
    return SPACE_ROWS.get(space_name, M)


def _tables():
    """三张表，一起取。**这是能力探针**：取不到就说明内部接口变了。"""
    return {
        "load": block_ops._A5_LOAD_COMBOS,
        "store": block_ops._A5_STORE_COMBOS,
        "move": block_ops._A5_MOVE_COMBOS,
    }


def _layout_name(value):
    """把表里的布局值变成名字。通配项（`_ANY_LAYOUT`，一个裸 `object()`）返回 None。"""
    text = str(value)
    if "TensorLayout" not in text:
        return None
    name = text.split(".")[-1].split(":")[0].strip("<>")
    return name if name in LAYOUTS else None


def _project_space_layouts():
    found = defaultdict(set)
    for table in _tables().values():
        for row in table:
            for space, layout in (
                (row[0], _layout_name(row[2])),
                (row[1], _layout_name(row[3])),
            ):
                found[space]
                if layout is not None:
                    found[space].add(layout)
    return dict(found)


def _project_space_pairs():
    return {
        name: {(row[0], row[1]) for row in table} for name, table in _tables().items()
    }


def _project_fp16_pairs():
    """三张表里 **FP16 → FP16** 的行。混宽度的行（比如 `fp16 → int16`）另算。"""
    found = defaultdict(set)
    for op, table in _tables().items():
        for row in table:
            if str(row[4]) != "fp16" or str(row[5]) != "fp16":
                continue
            src, dst = _layout_name(row[2]), _layout_name(row[3])
            if src is not None and dst is not None:
                found[(op, row[0], row[1])].add((src, dst))
    return dict(found)


def _parse(kernel, kind=pp.ir.SectionKind.Cube):
    kernel.to_kernel_def().parse_target_program(kind)


def _run(kernel, device):
    rng = np.random.default_rng(13)
    a = rng.standard_normal((M, K)).astype(np.float16)
    out = torch.zeros((M, K), dtype=torch.float16, device=device)
    kernel[None, 1](torch.from_numpy(a).to(device), out)
    torch.npu.synchronize()
    return a, out.cpu().numpy()


# ---------------------------------------------------------------------------
# 一、第一层：tile 声明
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("space_name", DECLARING_SPACES)
def test_the_default_layout_of_each_space(space_name):
    """验证 1 前段。**不给 `layout=`** 时框架填的缺省值，和显式给缺省值等价。

    这一条把 `DEFAULT_LAYOUT` 那张手写表钉住：缺省值变了，显式版和隐式版就不再都通过。
    """
    space = getattr(pl.MemorySpace, space_name)
    rows = _rows(space_name)
    _parse(make_default_layout_probe(space, rows=rows), pp.ir.SectionKind.Vector)
    explicit = getattr(pl, DEFAULT_LAYOUT[space_name])
    _parse(make_declaration_probe(space, explicit, rows=rows), pp.ir.SectionKind.Vector)


@pytest.mark.parametrize("space_name", DECLARING_SPACES)
def test_each_space_declares_exactly_the_layouts_we_wrote_down(space_name):
    """验证 1 后段。许可集里的六个布局逐个试：该通的通、该拒的拒。

    **六个都试**而不是只试许可集里的：只试合法的那些，多出来的许可（框架放宽了）
    就看不出来。
    """
    space = getattr(pl.MemorySpace, space_name)
    for layout_name in LAYOUTS:
        probe = make_declaration_probe(
            space, getattr(pl, layout_name), rows=_rows(space_name)
        )
        expected = refusal_kind(space_name, layout_name)
        if expected == "allowed":
            _parse(probe, pp.ir.SectionKind.Vector)
            continue
        with pytest.raises(InvalidVal) as raised:
            _parse(probe, pp.ir.SectionKind.Vector)
        assert space_name in str(raised.value), str(raised.value)


def test_mat_takes_zz_and_nn_only_with_the_mx_scale_dtype():
    """验证 2，**这一例的第二个核心结论**。`Mat(ZZ)` / `Mat(NN)`：FP16 拒，FP8E8M0 通。

    **两个方向都测**：只测 FP8E8M0 通过的话，「只在这个 dtype 下」这半句就没验到。
    """
    for layout_name in sorted(MX_SCALE_ONLY):
        layout = getattr(pl, layout_name)
        with pytest.raises(InvalidVal) as raised:
            _parse(
                make_declaration_probe(pl.MemorySpace.Mat, layout),
                pp.ir.SectionKind.Vector,
            )
        assert "require layout in" in str(raised.value), str(raised.value)

        _parse(
            make_declaration_probe(pl.MemorySpace.Mat, layout, dtype=FP8E8M0),
            pp.ir.SectionKind.Vector,
        )

    assert declarable_layouts("Mat", dtype_is_mx_scale=True) == set(LAYOUTS), (
        "加上 MX 的两个之后 Mat 应该覆盖全部六个布局"
    )
    assert declarable_layouts("Mat") == DECLARABLE["Mat"]


def test_the_two_refusal_messages_are_different():
    """验证 3。`Left(ZZ)` 和 `Left(ND)` 撞的是两条不同的规则，消息不同。

    分两条断言是因为「A5 不支持」和「这个空间不接受」是两件事：
    看到前者最想知道的是「那 A3 上行吗」（源码里 A3 的 `Left` 缺省确实是 `ZZ`）。
    """
    with pytest.raises(InvalidVal) as rejected:
        _parse(
            make_declaration_probe(pl.MemorySpace.Left, pl.ZZ), pp.ir.SectionKind.Vector
        )
    assert "do not support ZZ layout on 'a5'" in str(rejected.value), str(
        rejected.value
    )
    assert refusal_kind("Left", "ZZ") == "rejected-on-a5"

    with pytest.raises(InvalidVal) as not_allowed:
        _parse(
            make_declaration_probe(pl.MemorySpace.Left, pl.ND), pp.ir.SectionKind.Vector
        )
    assert "require layout in" in str(not_allowed.value), str(not_allowed.value)
    assert refusal_kind("Left", "ND") == "not-allowed"


# ---------------------------------------------------------------------------
# 二、第二层：和三张组合表对账
# ---------------------------------------------------------------------------


def test_the_combo_tables_are_still_there_and_still_six_tuples():
    """验证 4 的第一步。**表没了或格式变了要红在这儿**，不是红在一堆看不懂的组合上。

    行数只断言量级（±20%）：差一两行是正常演进。精确行数记在 `TABLE_ROWS` 供对比。
    """
    for name, table in _tables().items():
        assert table, f"{name} 的组合表是空的"
        assert all(len(row) == ROW_WIDTH for row in table), (
            f"{name} 的行不再是 {ROW_WIDTH} 元组"
        )
        expected = TABLE_ROWS[name]
        assert abs(len(table) - expected) <= expected * ROW_TOLERANCE, (
            f"{name} 表 {len(table)} 行，2026-09-28 是 {expected} 行 —— 量级变了，"
            f"这一例记的布局规则要重新投影一遍"
        )


def test_each_space_appears_with_exactly_the_layouts_we_wrote_down():
    """验证 4 之二。压掉 dtype 的投影与手写的 `SPACE_LAYOUTS` 逐项相等。

    `SPACE_LAYOUTS` 是手写的，不是从表算出来的——**算出来再和表比是恒真的**。
    """
    projected = _project_space_layouts()
    assert set(projected) == set(SPACE_LAYOUTS), (
        f"空间集合变了：表里 {sorted(projected)}，手写的 {sorted(SPACE_LAYOUTS)}"
    )
    for space in sorted(SPACE_LAYOUTS):
        assert projected[space] == SPACE_LAYOUTS[space], (
            f"{space} 出现的布局变了：表里 {sorted(projected[space])}，"
            f"手写的 {sorted(SPACE_LAYOUTS[space])}"
        )


def test_each_op_covers_exactly_the_space_pairs_we_wrote_down():
    """验证 4 之三。`load` 只有 `GM → Mat` / `GM → Vec`，`store` 只有 `Vec → GM` / `Acc → GM`。"""
    projected = _project_space_pairs()
    for name in sorted(OP_SPACE_PAIRS):
        assert projected[name] == OP_SPACE_PAIRS[name], (
            f"{name} 的空间对变了：表里 {sorted(projected[name])}，"
            f"手写的 {sorted(OP_SPACE_PAIRS[name])}"
        )


def test_the_fp16_layout_pairs_are_exactly_what_we_wrote_down():
    """验证 4 之四，**§四 那一跤的判据**。FP16 → FP16 的布局对逐项相等。

    和上面那条的区别就是「压不压 dtype」：那条答「可能持有什么」，这条答「能不能这么搬」。
    **两条都要有**，合成一条会丢掉这个区别。
    """
    projected = _project_fp16_pairs()
    for key in sorted(FP16_LAYOUT_PAIRS):
        assert projected.get(key) == FP16_LAYOUT_PAIRS[key], (
            f"{key} 的 FP16 布局对变了：表里 {sorted(projected.get(key, set()))}，"
            f"手写的 {sorted(FP16_LAYOUT_PAIRS[key])}"
        )


def test_l1_is_the_only_space_that_ever_shows_all_six_layouts():
    """验证 4 之五。`Left` 收 `NZ`、`Right` 收 `ZN`，两个反着。

    **`Mat` 那个「六个全有」来自表里含 FP8E8M0 的 MX 行**——判据 2 已经把那个条件钉住了,
    这里只确认组合表这一侧的投影和它一致。
    """
    assert SPACE_LAYOUTS["Mat"] == set(LAYOUTS)
    assert SPACE_LAYOUTS["Left"] == {MAT_TO_L0_TARGET["Left"]} == {"NZ"}
    assert SPACE_LAYOUTS["Right"] == {MAT_TO_L0_TARGET["Right"]} == {"ZN"}
    assert spaces_with_layout("ZZ") == {"Mat", "ScaleLeft"}
    assert spaces_with_layout("NN") == {"Mat", "ScaleRight"}


def test_all_six_layouts_are_exported():
    """§八 的判据。`pl` 导出六个布局，而 `.pyi` stub 只列了三个。"""
    for name in LAYOUTS:
        assert hasattr(pl, name), f"pl.{name} 不在了"
    assert len({getattr(pl, name) for name in LAYOUTS}) == len(LAYOUTS), (
        "六个布局有重复值"
    )


# ---------------------------------------------------------------------------
# 三、上板
# ---------------------------------------------------------------------------


def test_from_an_nd_gm_tensor_fp16_can_only_land_in_l1_nz(device):
    """验证 5，**§四 的结论上板**。`GM(ND) → Mat(NZ)` 通，`Mat(ZN)` 被拒。

    `NZ` 跑通只说明「这一条能跑」；**`ZN` 被拒才说明「只有这一条能跑」**。
    `ZN` 在第一层是可声明的（`Mat` 的许可集里有），所以它撞的是**第二层**——
    判据断言消息里的 "unsupported layout/dtype combination" 就是为了钉住这一点。
    """
    a, got = _run(make_gm_to_l1(pl.NZ), device)
    np.testing.assert_array_equal(got, a)

    assert "ZN" in DECLARABLE["Mat"], "ZN 不再可声明的话这条判据验的就不是第二层了"
    with pytest.raises(InvalidType) as raised:
        _parse(make_gm_to_l1(pl.ZN))
    message = str(raised.value)
    assert "unsupported layout/dtype combination" in message, message
    assert "Mat(ZN" in message, message


@pytest.mark.parametrize(
    ("space_name", "layout_name"), [("Left", "NZ"), ("Right", "ZN")]
)
def test_l1_nz_reaches_both_l0_operands(device, space_name, layout_name):
    """验证 6。同一个 `L1(NZ)` 源，`move` 到 L0A 要 `NZ`、到 L0B 要 `ZN`。

    **源布局相同**，所以差别只能来自目的空间——这是「两个 L0 反着」的判据。
    到 L0B 那一条顺带是一次真的布局转换（`NZ → ZN`），到 L0A 那一条不是。
    """
    kernel = make_l1_to_l0(
        getattr(pl.MemorySpace, space_name), getattr(pl, layout_name)
    )
    a, got = _run(kernel, device)
    np.testing.assert_array_equal(got, a)


@pytest.mark.parametrize("enabled", [True, False])
def test_the_layout_transform_switch_does_not_change_a_plain_move(device, enabled):
    """验证 7。布局开关开着和关着，一趟普通搬运的结果相同。

    **这条不验排出方向**，验的是「加了这一条不会把对的弄坏」——和
    ``datamove/mutex_dcci`` 验证 4 同一个形状。见 §七。
    """
    a, got = _run(make_with_layout_transform(enabled), device)
    np.testing.assert_array_equal(got, a)


def test_the_layout_switch_in_a_vector_section_is_not_refused(device):
    """验证 9。布局开关放进 vector 段**不被拒**，解析和上板都通。

    写这一例时我把它当反例，判据红在「DID NOT RAISE」——**收着它是为了钉住这里没有检查**。
    """
    _parse(layout_transform_in_a_vector_section, pp.ir.SectionKind.Vector)
    a, got = _run(layout_transform_in_a_vector_section, device)
    np.testing.assert_array_equal(got, a)


# ---------------------------------------------------------------------------
# 四、被拒的写法
# ---------------------------------------------------------------------------


def test_loading_straight_into_l0a_is_refused():
    """验证 8 之一，**撞第 1 层**。`load` 的 dst 只能在 Vec 或 Mat。

    **这是最容易想当然的一步**：L0A 也是个 buffer，为什么不能直接搬进去。
    答案是片外到 L0 一定是两跳。

    我原本按第 3 层的消息写了判据（`unsupported layout/dtype combination`），红了——
    实际拦它的是算子操作数检查，**而那条消息好得多**：它直说 dst 该在哪。
    """
    with pytest.raises(InvalidOperation) as raised:
        _parse(load_straight_into_l0a)
    message = str(raised.value)
    assert "dst tile must be in Vec (UB) or Mat (L1) memory" in message, message
    assert "got Left" in message, message


def test_storing_from_l1_to_gm_is_refused():
    """验证 8 之二，**也撞第 1 层**。`store` 的 src 只能在 Vec 或 Acc，L1 出不去。

    和上一条成对：`load` 管 dst、`store` 管 src，两个检查在 `block_ops.py` 的
    两个不同位置（`:1388` 和 `:559`），消息形状一样。
    """
    with pytest.raises(InvalidOperation) as raised:
        _parse(store_from_l1_to_gm)
    message = str(raised.value)
    assert "src tile must be in Vec (UB) or Acc (L0C) memory" in message, message
    assert "got Mat" in message, message


def test_a_positional_enabled_is_refused():
    """验证 8 之三。`enabled` 是 keyword-only，而 `set_mm_layout_transform(True)` 读起来很顺。"""
    with pytest.raises(Exception) as raised:
        _parse(layout_transform_without_the_keyword)
    assert "enabled" in str(raised.value), str(raised.value)
