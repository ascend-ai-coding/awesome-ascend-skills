# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。六个布局在各存储空间的合法组合，以及 matmul 的布局开关。

**布局的合法性有两层检查，在两个地方**，写这一例时是一层一层撞出来的：

| 层 | 在哪 | 问的是 | 消息质量 |
| --- | --- | --- | --- |
| 1. 算子操作数 | `block_ops.py:1388` / `:559` | `load` 的 dst、`store` 的 src **在哪个空间** | 好，直说该在哪 |
| 2. tile 声明 | `block_ops.py:1995` / `:3829` | 这个空间**能不能声明**成这个布局、这个形状 | 好，带许可集和缺省值 |
| 3. 数据通路 | `_A5_LOAD_COMBOS` 等三张表 | 这个 (空间, 布局, dtype) 组合**能不能搬** | 只说「不支持」，要自己去查表 |

**按这个顺序拦**：`load` 进 L0A 撞第 1 层，`Mat(NN)` + FP16 撞第 2 层,
`GM(ND) → Mat(ZN)` 撞第 3 层。三层的规则不同，读了一层的源码就去写 kernel,
会在另一层的报错前面找不着方向。

## 第一层：tile 声明（A5）

| 空间 | 缺省布局 | 可声明的布局 |
| --- | --- | --- |
| `Mat`（L1） | `NZ` | `NZ` `ZN` `ND` `DN`，**外加 `ZZ` `NN` 但只在 `dtype=FP8E8M0`** |
| `Left`（L0A） | `NZ` | 只有 `NZ` |
| `Right`（L0B） | `ZN` | 只有 `ZN` |
| `Acc`（L0C） | `NZ` | 只有 `NZ` |
| `Bias` | `ND` | 只有 `ND` |
| `Scaling` | `ND` | 只有 `ND` |
| `ScaleLeft` | `ZZ` | 只有 `ZZ` |
| `ScaleRight` | `NN` | 只有 `NN` |
| `Vec`（UB）/ `DDR`（GM） | 无 | 这一层不管 |

**`Bias` 还有一条形状约束**：必须恰好一行（`block_ops.py:3829`，硬件 `Rows==1`），
所以它的探针用 `[1, K]`。顺带一提，那条消息把 `Bias` 写成「Bias (L0B)」,
而 L0B 是 `Right`——**消息里的空间名标错了**，记在 `test.py` §八。

**`Mat` 的 `ZZ` / `NN` 是 MX 缩放视图**：`dtype=FP8E8M0` 时才允许，而且会把 `fractal`
强制成 32（`_MX_SCALE_FRACTAL`）。所以「L1 接受全部六个布局」这句话要加条件才成立。

`Left` 那一格值得单说：`allowed_layouts` 算出来是 `{ZZ, NZ}`（A3 的缺省是 `ZZ`），
但 A5 上 `ZZ` / `NN` 对 `Left` / `Right` / `Acc` 被单独一条规则拒掉,
**报错消息也不一样**（见 `test.py` 验证 3）。

## 第二层：数据通路，依赖 dtype

FP16 → FP16 的 `GM → Mat` **只有四条**：`DN→NZ`、`DN→ZN`、`ND→NZ`、`NZ→NZ`。
把 dtype 维压掉之后会多看到 `ND→ND`、`ND→NN`、`ND→ZZ`——**那些全来自 8 bit / 4 bit 的行**。

所以从一个 `ND` 的 FP16 GM 张量出发，L1 只能落在 `NZ`。

三个搬运接口的方向也是分开的：

| 接口 | 空间对 |
| --- | --- |
| `load` / `load_tile` | 只有 `GM → Mat`、`GM → Vec` |
| `store` / `store_tile` | 只有 `Vec → GM`、`Acc → GM` |
| `move` | 片上十个空间对 |

**`load` 到不了 L0A/L0B/L0C，`store` 出不了 L1。** 片外到 L0 一定是两跳。

第一手来源：`ir/op/block_ops.py` 的 `_DEFAULT_LAYOUTS_A5` / `_REJECTED_LAYOUTS_ON_A5` /
`_MX_SCALE_FRACTAL` 与三张组合表；`system_ops.py:520` 的 `set_mm_layout_transform`;
`datacopy/test_move_phase.py`、`system/test_cube_vector_sync.py`（布局开关的 ST 用法）。
"""

import pypto_pro.language as pl

FP16 = pl.DT_FP16
FP8E8M0 = pl.DT_FP8E8M0

M, K = 16, 64
_GM_UB = 0x0000
_L1 = 0x0000
_L0 = 0x0000

MX_SCALE_FRACTAL = 32
"""`Mat` 的 `ZZ` / `NN` 视图被强制成这个 fractal（`_MX_SCALE_FRACTAL`）。"""


# ---------------------------------------------------------------------------
# 一、第一层：tile 能不能声明成这个布局
# ---------------------------------------------------------------------------


def make_declaration_probe(space, layout, dtype=FP16, rows=M):
    """只声明一个 tile，什么都不搬。**声明那一层的检查在声明处就发生**，所以这就够了。

    工厂让 (空间, 布局, dtype, 行数) 四个变量走同一份代码——手写几十个 kernel 的话,
    「被拒的原因是哪一个变量」就说不清了。`rows` 是给 `Bias` 的（它要恰好一行）。
    """

    @pl.jit(auto_mutex=False)  # Parse-only layout probe; never launched.
    def probe(out: pl.Tensor[[M, K], FP16]):
        pl.make_tile(
            pl.TileType(
                shape=[rows, K], dtype=dtype, target_memory=space, layout=layout
            ),
            addr=_L1,
        )
        ub = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Vec),
            addr=_GM_UB,
        )
        with pl.section_vector():
            pl.store(out, ub, [0, 0])

    return probe


def make_default_layout_probe(space, dtype=FP16, rows=M):
    """**不给 `layout=`**，让框架填缺省值。对照上面那个显式版本。"""

    @pl.jit(auto_mutex=False)  # Parse-only layout probe; never launched.
    def probe(out: pl.Tensor[[M, K], FP16]):
        pl.make_tile(
            pl.TileType(shape=[rows, K], dtype=dtype, target_memory=space), addr=_L1
        )
        ub = pl.make_tile(
            pl.TileType(shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Vec),
            addr=_GM_UB,
        )
        with pl.section_vector():
            pl.store(out, ub, [0, 0])

    return probe


# ---------------------------------------------------------------------------
# 二、第二层：数据通路
# ---------------------------------------------------------------------------


def make_gm_to_l1(layout):
    """`GM(ND) → L1(layout)`。**FP16 下只有 `layout=NZ` 合法**。

    工厂保留是为了让合法的那一个和被拒的那个走同一份代码:
    只有布局一个变量，所以「被拒的原因是布局」这句话才站得住。
    """

    @pl.jit(auto_mutex=True)
    def kernel(a: pl.Tensor[[M, K], FP16], out: pl.Tensor[[M, K], FP16]):
        l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K],
                dtype=FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=layout,
            ),
            addrs=_L1,
            mutex_ids=[0],
        )
        ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=_GM_UB,
            mutex_ids=[1],
        )
        with pl.section_cube():
            pl.load(l1.current(), a, [0, 0])
        with pl.section_vector():
            t = ub.current()
            pl.load(t, a, [0, 0])
            pl.store(out, t, [0, 0])

    return kernel


def make_l1_to_l0(space, layout):
    """`L1(NZ) → L0A(NZ)` 或 `L1(NZ) → L0B(ZN)`。**同一个源，两个目的两个布局。**

    这一对是「L0A 收 `NZ`、L0B 收 `ZN`，两个反着」的判据：源布局相同,
    所以差别只能来自目的空间。到 L0B 那一条顺带是一次真的布局转换（`NZ → ZN`）,
    到 L0A 那一条不是。
    """

    @pl.jit(auto_mutex=True)
    def kernel(a: pl.Tensor[[M, K], FP16], out: pl.Tensor[[M, K], FP16]):
        l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
            ),
            addrs=_L1,
            mutex_ids=[0],
        )
        l0 = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=FP16, target_memory=space, layout=layout
            ),
            addrs=_L0,
            mutex_ids=[1],
        )
        ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=_GM_UB,
            mutex_ids=[2],
        )
        with pl.section_cube():
            pl.load(l1.current(), a, [0, 0])
            pl.move(l0.current(), l1.current())
        with pl.section_vector():
            t = ub.current()
            pl.load(t, a, [0, 0])
            pl.store(out, t, [0, 0])

    return kernel


@pl.jit(auto_mutex=True)
def load_straight_into_l0a(a: pl.Tensor[[M, K], FP16], out: pl.Tensor[[M, K], FP16]):
    """反例：`load` 直接进 L0A。表里 `load` 只有 `GM → Mat` 和 `GM → Vec`。

    **这是最容易想当然的一步**：L0A 也是个 buffer，为什么不能直接搬进去。
    """
    l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ
        ),
        addrs=_L0,
        mutex_ids=[0],
    )
    ub = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Vec),
        addrs=_GM_UB,
        mutex_ids=[1],
    )
    with pl.section_cube():
        pl.load(l0a.current(), a, [0, 0])
    with pl.section_vector():
        t = ub.current()
        pl.load(t, a, [0, 0])
        pl.store(out, t, [0, 0])


@pl.jit(auto_mutex=True)
def store_from_l1_to_gm(a: pl.Tensor[[M, K], FP16], out: pl.Tensor[[M, K], FP16]):
    """反例：从 L1 直接 `store` 回 GM。表里 `store` 只有 `Vec → GM` 和 `Acc → GM`。"""
    l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
        ),
        addrs=_L1,
        mutex_ids=[0],
    )
    with pl.section_cube():
        pl.load(l1.current(), a, [0, 0])
        pl.store(out, l1.current(), [0, 0])


# ---------------------------------------------------------------------------
# 三、set_mm_layout_transform
# ---------------------------------------------------------------------------


def make_with_layout_transform(enabled):
    """把布局开关套在一趟 `GM → L1 → L0A` 外面。

    ST 里的用法都是**开了再关**（`test_move_phase.py`、`test_cube_vector_sync.py`），
    这里照做：不关的话它对后面的 kernel 有没有影响没量过。
    """

    @pl.jit(auto_mutex=True)
    def kernel(a: pl.Tensor[[M, K], FP16], out: pl.Tensor[[M, K], FP16]):
        l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
            ),
            addrs=_L1,
            mutex_ids=[0],
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K],
                dtype=FP16,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
            ),
            addrs=_L0,
            mutex_ids=[1],
        )
        ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=_GM_UB,
            mutex_ids=[2],
        )
        with pl.section_cube():
            pl.system.set_mm_layout_transform(enabled=enabled)
            pl.load(l1.current(), a, [0, 0])
            pl.move(l0a.current(), l1.current())
            pl.system.set_mm_layout_transform(enabled=False)
        with pl.section_vector():
            t = ub.current()
            pl.load(t, a, [0, 0])
            pl.store(out, t, [0, 0])

    return kernel


@pl.jit(auto_mutex=True)
def layout_transform_in_a_vector_section(
    a: pl.Tensor[[M, K], FP16], out: pl.Tensor[[M, K], FP16]
):
    """**不是反例**：布局开关放进 vector 段，解析和上板都通得过（实测）。

    它管的是 fixpipe 排出 L0C 的方向，放 vector 段没有意义，但**没有任何一层拦它**。
    写这一例时我把它当反例写，判据红在「DID NOT RAISE」——收着它就是为了钉住这个空缺。
    """
    ub = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Vec),
        addrs=_GM_UB,
        mutex_ids=[0],
    )
    with pl.section_vector():
        pl.system.set_mm_layout_transform(enabled=True)
        t = ub.current()
        pl.load(t, a, [0, 0])
        pl.store(out, t, [0, 0])


@pl.jit(auto_mutex=True)
def layout_transform_without_the_keyword(
    a: pl.Tensor[[M, K], FP16], out: pl.Tensor[[M, K], FP16]
):
    """反例：`enabled` 是 keyword-only（`system_ops.py:520` 的签名里有 `*`）。"""
    ub = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Vec),
        addrs=_GM_UB,
        mutex_ids=[0],
    )
    with pl.section_cube():
        pl.system.set_mm_layout_transform(True)
    with pl.section_vector():
        t = ub.current()
        pl.load(t, a, [0, 0])
        pl.store(out, t, [0, 0])
