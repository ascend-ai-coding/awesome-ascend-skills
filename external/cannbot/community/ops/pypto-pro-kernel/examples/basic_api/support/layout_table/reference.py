# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""布局规则的**独立写下来的**期望。不 import 编译器。

这一份不算数值，它把两层检查各自的规则手写一遍。判据拿它和框架自己的表对,
**两边对不上就红**。

为什么值得这么做：那些表是非公开接口（下划线开头），随时可能改。
手写一份期望之后，表改了这里会红，而红的信息是「布局规则变了」——
比「某个 kernel 突然编不过」指得准得多。

**两份期望都是 2026-09-28 从源码和实测投影出来的**，不是从文档抄的。
`pypto/docs/` 里没有这两张表；布局的说明按算子分散写在各接口页。
"""

from __future__ import annotations

LAYOUTS = ("ND", "DN", "NZ", "ZN", "NN", "ZZ")
"""`pl` 导出的六个布局。`pypto/python/pypto/pypto_impl/ir.pyi` 的 `TensorLayout` stub
**只列了 ND / DN / NZ 三个**（2026-09-28 核）——stub 不全，以 `pl.__all__` 为准。
"""

# ---------------------------------------------------------------------------
# 第一层：tile 声明
# ---------------------------------------------------------------------------

DEFAULT_LAYOUT = {
    "Mat": "NZ",
    "Left": "NZ",
    "Right": "ZN",
    "Acc": "NZ",
    "Bias": "ND",
    "Scaling": "ND",
    "ScaleLeft": "ZZ",
    "ScaleRight": "NN",
}
"""A5 上每个空间的缺省布局（`_DEFAULT_LAYOUTS_A5`）。`Vec` / `DDR` 没有缺省。

**`Left` 的 A5 缺省是 `NZ`，A3 是 `ZZ`**——这是两代硬件的差别,
而 `allowed_layouts` 的算法把两代的缺省都放进候选集，再靠另一条规则把 A5 不支持的拒掉。
"""

DECLARABLE = {
    "Mat": {"NZ", "ZN", "ND", "DN"},
    "Left": {"NZ"},
    "Right": {"ZN"},
    "Acc": {"NZ"},
    "Bias": {"ND"},
    "Scaling": {"ND"},
    "ScaleLeft": {"ZZ"},
    "ScaleRight": {"NN"},
}
"""A5 上 **FP16** tile 可声明的布局。

`Mat` 的 `ZZ` / `NN` **不在这里**：它们只在 `dtype=FP8E8M0` 时允许（见 `MX_SCALE_ONLY`）。
所以「L1 接受全部六个布局」这句话要加 dtype 条件才成立。
"""

MX_SCALE_ONLY = {"ZZ", "NN"}
"""`Mat` 上这两个布局只在 `dtype=FP8E8M0` 时可声明（MX 缩放视图），
而且 `fractal` 会被强制成 32。
"""

REJECTED_ON_A5 = {"Left", "Right", "Acc"}
"""这三个空间上 `ZZ` / `NN` 走**另一条**规则被拒，报错文本也不同
（`tiles do not support ... layout on 'a5'` 而不是 `tiles require layout in {...}`）。
判据 3 把两种消息分开断言。
"""

# ---------------------------------------------------------------------------
# 第二层：数据通路
# ---------------------------------------------------------------------------

SPACE_LAYOUTS = {
    "GM": {"ND", "DN", "NZ"},
    "Vec": {"ND", "DN", "NZ"},
    "Mat": {"ND", "DN", "NZ", "ZN", "NN", "ZZ"},
    "Left": {"NZ"},
    "Right": {"ZN"},
    "Acc": {"NZ"},
    "Bias": {"ND"},
    "ScaleLeft": {"ZZ"},
    "ScaleRight": {"NN"},
    "Scaling": set(),
}
"""三张组合表里每个空间出现过的布局（通配项不算），**dtype 维已压掉**。

这张表回答「一个空间**可能**持有哪些布局」，**不回答「能不能这么搬」**——见
`FP16_LAYOUT_PAIRS`。`Mat` 这里是六个全都出现，因为表里含 FP8E8M0 的 MX 行。
`Scaling` 只以通配出现，空集的意思是「表里没给出具体布局」，不是「不支持」。
"""

OP_SPACE_PAIRS = {
    "load": {("GM", "Mat"), ("GM", "Vec")},
    "store": {("Vec", "GM"), ("Acc", "GM")},
    "move": {
        ("Acc", "Mat"),
        ("Acc", "Vec"),
        ("Mat", "Bias"),
        ("Mat", "Left"),
        ("Mat", "Right"),
        ("Mat", "ScaleLeft"),
        ("Mat", "ScaleRight"),
        ("Mat", "Scaling"),
        ("Vec", "Mat"),
        ("Vec", "Vec"),
    },
}
"""三个搬运接口各自覆盖的空间对。

**`load` 到不了 L0A/L0B/L0C，`store` 出不了 L1。** 片外到 L0 一定是两跳:
先 `load` 进 L1 或 UB，再 `move`。
"""

FP16_LAYOUT_PAIRS = {
    ("load", "GM", "Mat"): {("DN", "NZ"), ("DN", "ZN"), ("ND", "NZ"), ("NZ", "NZ")},
    ("load", "GM", "Vec"): {("DN", "DN"), ("ND", "ND"), ("NZ", "NZ")},
    ("store", "Vec", "GM"): {("DN", "DN"), ("ND", "ND"), ("NZ", "NZ")},
    ("move", "Mat", "Left"): {("ND", "NZ"), ("NZ", "NZ"), ("ZN", "NZ"), ("ZZ", "NZ")},
    ("move", "Mat", "Right"): {("ND", "ZN"), ("NZ", "ZN"), ("ZN", "ZN"), ("ZZ", "ZN")},
    ("move", "Mat", "ScaleLeft"): {("ZZ", "ZZ")},
    ("move", "Mat", "ScaleRight"): {("NN", "NN")},
}
"""**FP16 → FP16** 的布局对，按 (算子, 源空间, 目的空间) 分组。

对比最能说明问题的一格：压过 dtype 的表里 `GM → Mat` 有 `ND→ND`、`ND→NN`、`ND→ZZ`,
**FP16 一个都没有**——那几条全来自 8 bit / 4 bit 的行。FP16 从 `ND` 的 GM 张量出发
只能落到 `Mat(NZ)`。

只列了含 `Mat` / `GM` 的几组：`Vec → Mat` 和 `Vec → Vec` 的目的是通配,
`Mat → Bias` / `Mat → Scaling` 的 FP16 行不存在（那两个走 INT32/FP32）。
"""

TABLE_ROWS = {"load": 477, "store": 219, "move": 210}
"""2026-09-28 实测的行数。**这是能力探针**（[skill constraints](../../../../SKILL.md#constraints-and-workspace-policy)）：
判据先确认表还在、行数量级还对，再用它。差一两行不该红——所以判据只断言「非空且量级相符」,
具体数字记在这儿供对比。
"""

MAT_TO_L0_TARGET = {"Left": "NZ", "Right": "ZN"}
"""`Mat → Left` / `Mat → Right` 的目的布局：源可以是好几种，目的**各只有一个**，而且反着。"""


def spaces_with_layout(layout: str) -> set[str]:
    """哪些空间的组合表里出现过某个布局。判据用它反查。"""
    return {space for space, layouts in SPACE_LAYOUTS.items() if layout in layouts}


def declarable_layouts(space: str, dtype_is_mx_scale: bool = False) -> set[str]:
    """某个空间在给定 dtype 下可声明的布局。

    `dtype_is_mx_scale` 指 `FP8E8M0`——**只有它能让 `Mat` 接受 `ZZ` / `NN`**。
    """
    allowed = set(DECLARABLE[space])
    if space == "Mat" and dtype_is_mx_scale:
        allowed |= MX_SCALE_ONLY
    return allowed


def refusal_kind(space: str, layout: str) -> str:
    """这个 (空间, 布局) 会撞上哪一条规则。判据 3 用它选断言哪段消息。

    - `"rejected-on-a5"`：`Left` / `Right` / `Acc` 上的 `ZZ` / `NN`，走单独那条规则;
    - `"not-allowed"`：其余不在许可集里的;
    - `"allowed"`：合法。
    """
    if space in REJECTED_ON_A5 and layout in MX_SCALE_ONLY:
        return "rejected-on-a5"
    if layout in DECLARABLE.get(space, set()):
        return "allowed"
    return "not-allowed"
