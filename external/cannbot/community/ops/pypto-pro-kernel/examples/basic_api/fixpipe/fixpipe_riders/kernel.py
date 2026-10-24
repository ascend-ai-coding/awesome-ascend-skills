# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。fixpipe 的四个 rider：`relu_pre_mode` `atomic` `phase` `order`。

全部走 **Acc → GM**（`pl.store`）这一条——它不需要跨核交接
（见 [`fixpipe_paths`](../fixpipe_paths/kernel.py) 的对照），所以 kernel 里只剩 rider 本身。

## 四个 rider 各自是什么

| rider | 取值 | 做什么 | 判据怎么写 |
| --- | --- | --- | --- |
| `relu_pre_mode` | `NormalRelu`（ISA 侧只有它和 `NoRelu`） | 量化**之前**做 ReLU | 和不带 relu 的对照组比；输入必须有负数 |
| `atomic` | `AtomicNone`（默认）/ `AtomicAdd` | 写 GM 时原子累加而非覆盖 | **同一个 kernel 跑两次**，结果翻倍 |
| `phase` | `Unspecified` / `Partial` / `Final` | fixpipe 的 unit_flag 同步 | 与生产者的 `AccPhase` 配对；数值不变 |
| `order` | **升序**的两根目的轴，如 `[0, 2]` | 2 维 Tile 落到高维 Tensor 的哪两根轴 | 数据落在指定那一页，别的页仍是 0 |

**`ReluPreMode` 只有一个成员这件事不是缺口。** ISA 侧的 `pto::ReluPreMode` 也只有
`NoRelu` / `NormalRelu` 两个值，Python 侧「1 个成员 + `None` 表示不开」是完整的。
值得写下来，免得每次看到「枚举只有 1 项」都要重新怀疑一遍。
（ISA 另有一个独立的 `ClipReluMode_t`（钳位 relu），**Python 侧没有暴露**——
那是真缺口，见设计文档 §十四点二。）

## `phase` 要和生产者配对

`phase` 不是 fixpipe 单方面的开关：它和矩阵乘那边的 `AccPhase` 成对。
形状抄自 `pypto/python/tests/st/pypto_pro/frontend/datacopy/test_scale_phase_atomic_order.py`：

```python
pl.matmul(acc, l0a, l0b, phase=pl.AccPhase.Partial)
pl.matmul_acc(acc, acc, l0a, l0b, phase=pl.AccPhase.Final)
pl.store(out, acc, [0, 0], scale=s, phase=pl.STPhase.Partial)
pl.store(out, acc, [0, 0], scale=s, phase=pl.STPhase.Final)
```

这一段**不能用 `pl.gemm`**——`phase` 是 `pl.matmul` / `pl.matmul_acc` 的参数，
要手工把 L1 搬进 L0A/L0B。这是 phase 那一条比其他三条长的原因。

## 目的类型取 FP16，不取 FP32

`FP32 → FP32` 带 scale 时 **scale 被静默丢弃**（见
[defect](../../../../references/diagnostics/defects/fixpipe-fp32-to-fp32-drops-scale.md)），
用它做 rider 的载体会把 rider 的效果和那个缺陷混在一起。
`FP32 → FP16` 带 scale 是逐位精确的，所以判据能写 `atol=0`。

## `atomic` 必须指定 block_dim

`kernel(...)` 直接调用时 block_dim 走 `_AUTO_BLOCK_DIM = 0`，由 native 侧解成
**当前 stream 的全量核数**（`jit.py:1883`）。这几个 kernel 都不按
`get_block_idx()` 分工，于是每个 block 算同一份、写同一片 GM：

- 覆盖模式下每个 block 写的字节完全相同，**看不出来跑了几个 block**；
- `AtomicAdd` 下它们互相累加，输出变成 `block_dim × scale × acc`。

所以 atomic 那几条一律用 `kernel[None, block_dim](...)` 显式指定
（官方 ST `test_scale_phase_atomic_order.py::test_atomic_add` 也是 `[None, 1]`）。
**这不是 atomic 的缺陷，是「默认 block_dim 不是 1」被 atomic 暴露了出来。**
"""

import pypto_pro.language as pl

FP16 = pl.DT_FP16
FP32 = pl.DT_FP32

M = N = K = 64

ND = pl.TensorLayout.ND

#: 量化比例。2 的幂，浮点乘法精确。
SCALE = 0.5


def _cube_groups():
    """L1 两个操作数 + L0A/L0B + Acc。和 `fixpipe_paths` 的同名函数一致。"""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[M, N], dtype=FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    return a_l1, b_l1, l0a, l0b, acc


# ---------------------------------------------------------------------------
# 一、relu_pre_mode：量化之前做 ReLU
# ---------------------------------------------------------------------------


def make_store_with_relu(*, relu):
    """`pl.store(out, acc, [0, 0], scale=, relu_pre_mode=)`。

    `relu=False` 是对照组：**同一条链不带 relu**。「relu 生效了吗」只能靠两份输出
    对比来判——输入全正时 relu 是恒等映射，单看一份分不出来。
    所以 `reference.py` 的输入**必须有负数**，并且有一条测试专门钉这个前提。

    `relu_pre_mode` 名字里的 `pre` 指的是**在量化之前**：先 ReLU 再乘 scale 再转类型。
    对 `scale > 0` 来说两个顺序结果一样（都是钳掉负数），所以本例验不出先后。
    要验先后得用负的 scale——**那不是这一例的题目**，而且负 scale 的合法性没查。
    """

    @pl.jit(auto_mutex=True)
    def kernel(
        a: pl.Tensor[[M, K], FP16],
        b: pl.Tensor[[K, N], FP16],
        out: pl.Tensor[[M, N], FP16],
    ):
        a_l1, b_l1, l0a, l0b, acc = _cube_groups()
        with pl.section_cube():
            cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
            pl.load(cur_a, a, [0, 0])
            pl.load(cur_b, b, [0, 0])
            pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
            if relu:
                pl.store(
                    out,
                    ac,
                    [0, 0],
                    scale=SCALE,
                    relu_pre_mode=pl.ReluPreMode.NormalRelu,
                )
            else:
                pl.store(out, ac, [0, 0], scale=SCALE)

    return kernel


store_with_relu = make_store_with_relu(relu=True)
store_without_relu = make_store_with_relu(relu=False)


# ---------------------------------------------------------------------------
# 二、atomic：写 GM 时累加而不是覆盖
# ---------------------------------------------------------------------------


def make_store_atomic(*, atomic):
    """`atomic=pl.AtomicType.AtomicAdd`——把结果**加**到 GM 上，不覆盖。

    验法是**同一个 kernel 跑两次**：覆盖模式下两次结果一样，累加模式下第二次翻倍。
    这比「和参考比一次」强——它验的是「累加」这个语义本身。

    调用方要负责把输出**清零**（`AtomicAdd` 加在原有内容上）。测试里每次
    从零张量开始，跑两次拿 2×。

    **必须 `kernel[None, 1](...)` 启动**，见模块 docstring「`atomic` 必须指定
    block_dim」：默认 block_dim 是全量核数，每个 block 都累加一份，
    一次启动就拿到 `block_dim ×`。这一点是被这一例的失败测出来的，
    不是先知道的。
    """
    mode = pl.AtomicType.AtomicAdd if atomic else pl.AtomicType.AtomicNone

    @pl.jit(auto_mutex=True)
    def kernel(
        a: pl.Tensor[[M, K], FP16],
        b: pl.Tensor[[K, N], FP16],
        out: pl.Tensor[[M, N], FP16],
    ):
        a_l1, b_l1, l0a, l0b, acc = _cube_groups()
        with pl.section_cube():
            cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
            pl.load(cur_a, a, [0, 0])
            pl.load(cur_b, b, [0, 0])
            pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
            pl.store(out, ac, [0, 0], scale=SCALE, atomic=mode)

    return kernel


store_atomic_add = make_store_atomic(atomic=True)
store_overwrite = make_store_atomic(atomic=False)


# ---------------------------------------------------------------------------
# 三、phase：和生产者的 AccPhase 配对
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def store_phase_pair(
    a: pl.Tensor[[M, K], FP16], b: pl.Tensor[[K, N], FP16], out: pl.Tensor[[M, N], FP16]
):
    """`AccPhase.Partial/Final` 配 `STPhase.Partial/Final`。

    **`phase` 是同步优化，不改数值。** 它让 fixpipe 用 unit_flag 而不是显式同步
    等矩阵乘的结果，省一次同步开销。所以这一例的判据是「结果和不带 phase 的版本一致」,
    不是「结果变了」。

    这一段必须手工搬 L0A/L0B——`phase` 是 `pl.matmul` / `pl.matmul_acc` 的参数,
    `pl.gemm` 把那两步包掉了就传不进去。

    累加两次，所以 `acc = 2 * (a @ b)`；两次 store 都写同一片 GM，
    第二次覆盖第一次（`atomic` 是默认的 `AtomicNone`），**最终值只由第二次决定**。
    """
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        la, lb = l0a.current(), l0b.current()
        pl.move(la, cur_a)
        pl.move(lb, cur_b)
        pl.matmul(ac, la, lb, phase=pl.AccPhase.Partial)
        pl.matmul_acc(ac, ac, la, lb, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], scale=SCALE, phase=pl.STPhase.Partial)
        pl.store(out, ac, [0, 0], scale=SCALE, phase=pl.STPhase.Final)


@pl.jit(auto_mutex=True)
def store_no_phase(
    a: pl.Tensor[[M, K], FP16], b: pl.Tensor[[K, N], FP16], out: pl.Tensor[[M, N], FP16]
):
    """上一个的对照组：同样累加两次，但**不带任何 phase**。

    两份输出必须相同——`phase` 是同步优化，不是数值语义。
    """
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        la, lb = l0a.current(), l0b.current()
        pl.move(la, cur_a)
        pl.move(lb, cur_b)
        pl.matmul(ac, la, lb)
        pl.matmul_acc(ac, ac, la, lb)
        pl.store(out, ac, [0, 0], scale=SCALE)


# ---------------------------------------------------------------------------
# 四、order：2 维 Tile 落到高维 Tensor 的哪两根轴
# ---------------------------------------------------------------------------

#: 目的 Tensor 是 `[M, PAGES, N]`——中间多一根「页」轴，`order` 要把它跳过去。
PAGES = 2

#: 写第 1 页；第 0 页留空当对照，用来证明偏移真的落在指定页上。
PAGE = 1


@pl.jit(auto_mutex=True)
def store_order_page(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    out: pl.Tensor[[M, PAGES, N], FP16],
):
    """`order=[0, 2]`：2 维 Tile 的维 0 落到轴 0，维 1 落到轴 2，**跳过中间那根轴**。

    **`order` 不是转置。** 它必须升序（`block_ops.py:347`，降序报
    `order must be ascending`），所以拿它做转置这条路是关不通的——见
    `test_a_descending_order_is_rejected`。它回答的是另一个问题：
    **一个 2 维 Tile 要落到一个高维 Tensor 的哪两根轴上。**

    | | 谁做 | 改什么 |
    | --- | --- | --- |
    | `layout=`（目的 Tile/Tensor） | fixpipe | NZ 分形 → ND/DN/NZ 的**排布** |
    | `order=`（store 的参数） | 地址计算 | 2 维 Tile 映到哪两根 Tensor 轴 |

    不传 `order` 时默认是**最后两根轴** `list(range(ndim - 2, ndim))`
    （`block_ops.py:1223`）——本例的目的 Tensor 最后两根是 `[PAGES, N]`，
    Tile 的 64 行放不进 `PAGES=2`，所以这里**必须**显式传 `order`。
    「不传就不对」正是这一条验得出东西的原因。

    偏移仍然按 Tensor 的秩给满（三个），非 Tile 轴那一位是**下标**：
    `[0, PAGE, 0]` 选第 `PAGE` 页。
    """
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.store(out, ac, [0, PAGE, 0], scale=SCALE, order=[0, 2])


@pl.jit(auto_mutex=True)
def store_order_descending(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    out: pl.Tensor[[M, PAGES, N], FP16],
):
    """降序 `order=[2, 0]`——**解析期就被拒**，不上卡。

    这个 kernel 的存在就是为了钉住那条拒绝。它和上面那个只差 `order` 的顺序。
    """
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.store(out, ac, [0, PAGE, 0], scale=SCALE, order=[2, 0])


@pl.jit(auto_mutex=True)
def store_order_nz_out(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    out: pl.Tensor[[M, PAGES, N], FP16, pl.TensorLayout.NZ],
):
    """同一个升序 `order=[0, 2]`，但目的 Tensor 是 **NZ**——也被拒。

    `_validate_nz_transfer_axes`（`block_ops.py:142`）：NZ 搬运只允许映到
    **最后两根轴**。所以 `order` 的自由度只对 ND 目的成立,
    NZ 目的下它退化成「只能是默认值」。

    两条拒绝是不同的规则：上一个 kernel 拒的是**降序**，这个拒的是**NZ + 非末两轴**。
    """
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.store(out, ac, [0, PAGE, 0], scale=SCALE, order=[0, 2])
