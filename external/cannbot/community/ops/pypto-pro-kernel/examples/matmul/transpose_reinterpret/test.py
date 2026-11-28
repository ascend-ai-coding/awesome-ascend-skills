# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""转置的第二条路：``pl.gemm(trans_a=/trans_b=)``，在调用点重解释。

**解决什么.** 和 ``l1_layout`` 样例同一个问题——操作数在 GM 里是转置存放的——但
换一条路解决。两条路的分工是：

| | 转置发生在 | L1 Tile 的 shape | L1 的 layout | load |
|---|---|---|---|---|
| ``l1_layout`` 的 TT | **搬运**（GM -> L1） | 逻辑形状 ``[M, K]`` | **ZN** | ``order=[1, 0]`` |
| 本样例 | **gemm 内部**（L1 -> L0） | GM 形状 ``[K, M]`` | 常规 NZ | 默认 |

两者都能得到正确结果，选哪条取决于别的约束（下面「为什么在调用点」一节）。

**物理访问.** ``trans_a=True`` 时，L1 上那块 Tile 原样装着 GM 里的 ``[K, M]``，
gemm 把它**重解释**成一块 ZN 的 ``[M, K]`` 视图，再搬进 L0A。重解释不动数据，
只换描述：NZ 的 ``[d0, d1]`` 和 ZN 的 ``[d1, d0]`` 在片上是同一串字节。

所以这条路**不产生额外搬运**。它不是「先转置一遍再算」，是「换个方向读同一片
字节」。

L0A / L0B / L0C 照旧：NZ / ZN / NZ。转置不影响它们，理由和 ``l1_layout`` 一样。

**为什么在调用点，不在声明上.** 重解释是在 ``pl.gemm`` 这一行构造的，不落到
``a_l1`` 的声明里。原因是**同一块 L1 Tile 可能同时喂给一个非转置的操作数**——
如果把 ZN 写进声明，那块 Tile 就再也不能当常规操作数用了。代价是 ``a_l1`` 必须
是一个解析期已知维度的具名 Tile，gemm 才能算出重解释后的形状。

**必须先绑到变量.** 两个 kernel 里都写成 ``cur_a = a_l1.current()`` 再传，而不是
``pl.gemm(ac, a_l1.current(), ...)``。gemm 降级时要读操作数的声明形状，**再求值
一次表达式会让 TileGroup 多转一格**，所以这类操作数必须先绑定。用到
``trans_a`` / ``trans_b`` 时是对应的 L1 操作数，用到 ``m`` / ``n`` / ``k`` 时三块
都要绑，用到 ``splitn`` 时是累加器。不绑会直接报错，不会静默出错。

**为何这样同步.** 五个显式地址组使用 ``mutex_ids="auto"``，由 ``auto_mutex`` 插入核内同步。重解释出来的
视图与原 Tile 是同一片字节，也共用同一个 mutex ID——这正是 mutex ID 表达的含义
（「这几个句柄指向同一片字节」），所以框架不需要为它额外插同步。

**如何独立验证.** 形状取 ``M=64, K=128, N=32`` 的非方阵，方阵会让 ``[K, M]`` 和
``[M, K]`` 长得一样。reference 把转置存放的张量转回来再按定义算。两个 case：
两边都转置、只转左边——后者确认 ``trans_a`` 不会顺带影响右操作数。

源于 ``pypto/python/tests/st/pypto_pro/frontend/matmul/test_gemm_matrix_st.py``
的 ``g_trans_both`` 与 ``g_trans_a_m``。

``make_rectangular_handover`` 独立验证另一条装配路径：两个 AIV 从 GM 的 A^T 各取
``[128, M/2]``，在 UB 转为 NZ 后完整写入 Mat ZN ``[M, 128]``，由 Cube 做
``A @ B``。每个 AIV 只写自己的列半区；B 层串行交接同步后才允许 Cube 读取。
测试取 M=32/64、FP16/BF16。A^T 的值随两轴变化，B 从 K 的前后半区各选一个值，
使漏写半区或转置错误可见；值与结果都能精确表示，参考在 CPU 按数学矩阵乘法计算。
"""

import pypto_pro.language as pl
import pytest
import torch

from .kernel import (
    K,
    M,
    N,
    gemm_trans_a_only,
    gemm_trans_both,
    make_rectangular_handover,
)
from .reference import reference_trans_a, reference_trans_both


def test_both_operands_transposed(device):
    torch.manual_seed(0)
    a_t = torch.randn([K, M], device=device, dtype=torch.float16)
    b_t = torch.randn([N, K], device=device, dtype=torch.float16)
    out = torch.zeros([M, N], device=device, dtype=torch.float32)

    gemm_trans_both(a_t, b_t, out)
    torch.npu.synchronize()

    torch.testing.assert_close(
        out, reference_trans_both(a_t, b_t), rtol=2e-2, atol=2e-2
    )


def test_transposing_the_left_operand_leaves_the_right_alone(device):
    """``trans_a`` 只管左操作数：右边仍按常规方向读。"""
    torch.manual_seed(1)
    a_t = torch.randn([K, M], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)
    out = torch.zeros([M, N], device=device, dtype=torch.float32)

    gemm_trans_a_only(a_t, b, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, reference_trans_a(a_t, b), rtol=2e-2, atol=2e-2)


def test_both_paths_to_a_transpose_agree(device):
    """重解释这条路与 ``l1_layout`` 的 load 转置那条路，结果必须一致。

    两条路是对同一个问题的两种答案，如果结论不同，说明至少有一条理解错了。
    """
    from ..fundamental_usage.l1_layout.kernel import matmul_tt

    torch.manual_seed(2)
    a_t = torch.randn([K, M], device=device, dtype=torch.float16)
    b_t = torch.randn([N, K], device=device, dtype=torch.float16)

    by_reinterpret = torch.zeros([M, N], device=device, dtype=torch.float32)
    gemm_trans_both(a_t, b_t, by_reinterpret)
    by_load_order = torch.zeros([M, N], device=device, dtype=torch.float32)
    matmul_tt(a_t, b_t, by_load_order)
    torch.npu.synchronize()

    torch.testing.assert_close(by_reinterpret, by_load_order, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("m", [32, 64])
@pytest.mark.parametrize(
    "torch_dtype,pl_dtype", [(torch.float16, pl.DT_FP16), (torch.bfloat16, pl.DT_BF16)]
)
def test_rectangular_nz_to_zn_handover(
    device, tmp_path, monkeypatch, m, torch_dtype, pl_dtype
):
    monkeypatch.setenv("ASCEND_WORK_PATH", str(tmp_path))
    k, n = 128, 64
    a_t = (
        ((torch.arange(k)[:, None] * 17 + torch.arange(m)[None, :] * 11) % 31 - 15) / 16
    ).to(torch_dtype)
    b = torch.zeros(k, n, dtype=torch_dtype)
    b[torch.arange(n), torch.arange(n)] = 1
    b[k - 1 - torch.arange(n), torch.arange(n)] = 2
    expected = reference_trans_a(a_t, b)
    out = torch.full((m, n), -999, dtype=torch.float32, device=device)

    kernel = make_rectangular_handover(m, pl_dtype)
    kernel[1](a_t.to(device), b.to(device), out)
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
