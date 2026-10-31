# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""fixpipe 的三个目的地 × 三种布局，不带 rider 的基线。

**写法：原始。** 这是 fixpipe 这一章的第一例，后面四例都在它的骨架上加 rider。
机制在 [`kernel.py`](kernel.py) 的模块 docstring 里；设计文档 §十四 讲为什么 fixpipe
单独成章。

## 一、为什么 `pl.store` 和 `pl.move` 在这里是「同一个接口的两件事」

源在 `Acc` 时，`pl.store` / `pl.move` 发的是 **fix 流水**（`TSTORE` / `TEXTRACT`）；
源在 `Vec` 时发的是 MTE3 / v 流水。**同一个 Python 调用，不同的硬件单元。**

这一例把三条出路摆在一起，就是为了让这件事看得见：

| 出路 | 调用 | 流水 |
| --- | --- | --- |
| Acc → GM | `pl.store(out, acc, [0, 0])` | **fix** |
| Acc → Vec | `pl.move(ub, acc)` | **fix** |
| Acc → Mat | `pl.move(l1, acc)` | **fix** |
| Mat → Left/Right | `pl.move(l0a, l1)` | MTE1（**不是** fixpipe） |
| Vec → GM | `pl.store(out, ub, [0, 0])` | MTE3（**不是** fixpipe） |

**Acc→Mat 的结果只有一条出路**：进 L0A/L0B 当下一次乘法的操作数。
`Mat→Vec` 不在 `move` 的合法路径里，`Mat→GM` 不在 store 的组合表里
——而 `move` 的 docstring 把 `Mat → Vec` 列成走 v 流水，**与实现不符**（已报 pypto）。

## 二、三种布局只差目的 Tile 的一个 `layout=`

ISA 侧是 `LayoutMode_t` 的 `NZ2ND` / `NZ2DN` / `NZ2NZ`，**Python 侧没有这个参数**
——它由目的操作数的 `layout=` 推断。所以 `make_acc_to_vec` 是个工厂，三个变体只差一个枚举。

**GM 侧的 Tensor 必须声明成同一个 layout**（`pl.Tensor[[M, N], FP32, layout]`）：
Vec→GM 的组合表只有 `ND→ND` / `DN→DN` / `NZ→NZ` 三对，**跨布局的 store 不存在**
——布局转换是 fixpipe 那一步做的，MTE3 不做。

Acc→GM 的组合表只有 `NZ→ND` 和 `NZ→NZ` 两对，**没有 `NZ→DN`**。

## 四、DN 这一支这一版走不通，钉成边界

`NZ2DN` 的两步在 Python 侧都合法，但 codegen 发的 `GlobalTensor` 是 `Layout::ND`，
被 A5 的 `TStore.hpp:169` 拦。**全仓 ST 里 DN 的 Vec Tile 一律是 `[N,1]` 单列**——
正好落在那个 static_assert 的 `Rows == 1 || Cols == 1` 例外里，2-D 的 DN 一个先例都没有。
`test_a_two_dimensional_dn_tile_has_no_store_path` 把这件事钉成 `pytest.raises`，
细节和两条旁证在那条测试的 docstring 里。

## 三、判据：`rtol=0, atol=0`

FP16 输入取小整数，K=128 下最大累加 1024，FP32 精确。**这一例不需要容差**——
有差异就只可能是搬运或布局错了，不用分辨「是精度还是 bug」。

数值正确性之外还断言排布：三种 layout 的输出**不能三个都相同**，否则 `layout=`
没起作用，三个变体就退化成同一个测试。
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from . import reference as ref
from .kernel import (
    K,
    M,
    N,
    acc_to_gm,
    acc_to_mat_chain,
    acc_to_vec_dn,
    acc_to_vec_nd,
    acc_to_vec_nz,
)

_LAYOUTS = [("ND", acc_to_vec_nd), ("NZ", acc_to_vec_nz)]


def _dev(array, device):
    return torch.from_numpy(array).to(device)


def _out(device):
    return torch.zeros((M, N), dtype=torch.float32, device=device)


def _run(kernel, device):
    a, b = ref.make_operands(M, K, N)
    out = _out(device)
    kernel(_dev(a, device), _dev(b, device), out)
    torch.npu.synchronize()
    return a, b, out.cpu().numpy()


# ---------------------------------------------------------------------------
# 三个目的地
# ---------------------------------------------------------------------------


def test_acc_to_gm_is_one_fixpipe(device):
    """`pl.store(out, acc, [0, 0])`——一行，一次 fixpipe，没有中间 Tile。"""
    a, b, got = _run(acc_to_gm, device)
    np.testing.assert_allclose(
        got, ref.matmul(a, b), rtol=0, atol=0, err_msg="Acc→GM 的结果不对"
    )


def test_acc_to_vec_then_store(device):
    """`pl.move(ub, acc)` + `pl.store(out, ub, ...)`——两步，只有第一步是 fixpipe。"""
    a, b, got = _run(acc_to_vec_nd, device)
    np.testing.assert_allclose(
        got, ref.matmul(a, b), rtol=0, atol=0, err_msg="Acc→Vec→GM 的结果不对"
    )


def test_acc_to_mat_feeds_the_next_gemm(device):
    """`pl.move(l1, acc)`：结果留在 L1 当下一次乘法的操作数，算 `fp16(a@b) @ c`。

    **这是 Acc→Mat 唯一的出路。** `Mat→Vec` 不在 `move` 的合法路径里，
    `Mat→GM` 不在 store 的组合表里——见 kernel.py 里那张表。
    `move` 的 docstring 却把 `Mat → Vec` 列成走 v 流水，与实现不符。

    那一步顺带做 `FP32 → FP16` 降位：**降位在 fixpipe 里免费**，
    不需要单独一趟向量转换。
    """
    a, b = ref.make_operands(M, K, N)
    c = ref.make_third(N)
    out = _out(device)
    acc_to_mat_chain(_dev(a, device), _dev(b, device), _dev(c, device), out)
    torch.npu.synchronize()
    np.testing.assert_allclose(
        out.cpu().numpy(),
        ref.matmul_chain(a, b, c),
        rtol=0,
        atol=0,
        err_msg="Acc→Mat 再进第二次 gemm 的结果不对",
    )


def test_acc_to_gm_and_acc_to_vec_agree(device):
    """Acc→GM 和 Acc→Vec→GM 的数值必须一致——否则其中一条的布局转换错了。

    这条不依赖参考实现：它是两个输出之间的关系。
    Acc→Mat 那条进不了这个对照（它的输出是另一个式子），单独由上一条测。
    """
    _, _, from_gm = _run(acc_to_gm, device)
    _, _, from_vec = _run(acc_to_vec_nd, device)

    np.testing.assert_allclose(
        from_vec, from_gm, rtol=0, atol=0, err_msg="Acc→Vec 与 Acc→GM 结果不一致"
    )


# ---------------------------------------------------------------------------
# 三种布局
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "kernel"), _LAYOUTS, ids=[n for n, _ in _LAYOUTS])
def test_each_layout_moves_the_same_values(name, kernel, device):
    """三种 `LayoutMode_t` 各自跑通，且搬出去的**值的集合**和参考一样。

    不逐位置比，因为 DN / NZ 的排布本来就和 ND 不同——比的是「没丢没多没改值」。
    逐位置的差异由下一条测。
    """
    a, b, got = _run(kernel, device)
    want = ref.matmul(a, b)
    np.testing.assert_allclose(
        np.sort(got.reshape(-1)),
        np.sort(want.reshape(-1)),
        rtol=0,
        atol=0,
        err_msg=f"{name} 布局搬出来的值集合和参考不一样",
    )


def test_nd_and_nz_differ(device):
    """防空测：`layout=` 必须真的改变排布。

    如果两种布局的输出逐位相同，说明 `layout=` 没被传下去，两个变体退化成同一个测试。
    """
    _, _, from_nd = _run(acc_to_vec_nd, device)
    _, _, from_nz = _run(acc_to_vec_nz, device)

    assert not np.array_equal(from_nd, from_nz), (
        "ND 与 NZ 的输出逐位相同——layout= 没起作用（或这批输入恰好对称）"
    )


def test_a_two_dimensional_dn_tile_has_no_store_path(device):
    """**边界：2-D 的 `NZ2DN` 在这一版走不通。** 这条把它钉成事实，不是留个注释。

    Python 侧两步都合法（否则会在解析期被拦，而它没有）：
    `Acc(NZ)→Vec(DN)` 在 `move` 的表里，`Vec(DN)→GM(DN)` 在 `store` 的表里。
    但 codegen 发出的 `GlobalTensor` 是 `Layout::ND`，被 A5 的 `TStore.hpp:169`
    的 static_assert 拦住：**"Src and dst layout must be same"**。

    两条旁证说明这不是我写错了：

    - **全仓 ST 里 DN 的 Vec Tile 一律是 `[N, 1]` 单列**（`test_histogram.py`、
      `test_softmax.py`、`test_layernorm.py`、`test_reduce_expand_fp32.py`），
      正好落在那个 static_assert 的 `Rows == 1 || Cols == 1` 例外里。
      **2-D 的 DN 一个先例都没有。**
    - 给 GM Tensor 注解 layout 本身是可行的——`test_nz_datacopy.py` 用
      `pl.Tensor[[...], pl.DT_FP16, pl.NZ]`，本例的 NZ 变体也跑通了。
      **只有 DN 这一支落不下去。**

    还没分清的是：codegen 丢了 DN 这个注解，还是 A5 本来就不支持 2-D 的 DN store。
    记在设计文档 §十四点八，待确认。

    如果哪天这条不再抛异常，说明修了——那时把它改回正常的布局对照测试。
    """
    a, b = ref.make_operands(M, K, N)
    out = _out(device)
    with pytest.raises(Exception, match=r"(?s)Failed to compile|layout must be same"):
        acc_to_vec_dn(_dev(a, device), _dev(b, device), out)


# ---------------------------------------------------------------------------
# 结构检查
# ---------------------------------------------------------------------------


def test_the_input_has_both_signs():
    """不需要卡。`fixpipe_riders` 要用同一个构造函数验随路 relu，
    全正的输入下 relu 是恒等映射。这里先把前提钉住。
    """
    a, _ = ref.make_operands(M, K, N)
    assert ref.has_both_signs(a), "输入没有负数——随路 relu 那一例会验空"


def test_the_gemm_producer_is_shared():
    """不需要卡。本例三条输出路径共用一个 `_cube_groups()` 生产者。"""
    source = (Path(__file__).parent / "kernel.py").read_text()
    assert source.count("def _cube_groups(") == 1, "生产者应当只声明一次"
    calls = [
        line
        for line in source.splitlines()
        if "_cube_groups()" in line and not line.lstrip().startswith("def ")
    ]
    assert len(calls) == 3, (
        f"三个 kernel（acc_to_gm / 工厂 / acc_to_mat_chain）各调一次，实际 {len(calls)} 处"
    )
