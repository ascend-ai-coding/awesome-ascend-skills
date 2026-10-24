# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Checks for the fixed single-head online-softmax kernel.

The independent FP64 reference and input-dependent error budget are defined in reference.py, including the hardware exp
assumption. Cases exercise normal/growing/first-max/last-max/negative scores, uniform and one-hot behavior, partial
launches, zero-tolerance preload/repeated-call consistency and five incorrect mathematical controls. Static checks
inspect native TileType shapes and GEMM/VF use. A numerical pass does not establish performance or prove all
synchronization lifetimes.

All resources have explicit addresses, automatic intra-core mutex IDs and per-slot forward/reverse cross-core events;
alpha's same-core ring has PRELOAD+1 slots. Device selection is inherited from examples/conftest.py.
"""

import ast
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from . import kernel as kernel_module
from .kernel import (
    BLOCK_KV,
    D,
    HALF,
    LAUNCH_BLOCK_DIM,
    M,
    N,
    NUM_BLOCKS,
    NUM_CORES,
    ROWS_PER_CORE,
    flash_attention,
    flash_attention_p0,
)
from .reference import (
    KINDS,
    attention_reference,
    blockwise_softmax_reference,
    error_bound,
    make_inputs,
    no_l_rescale_reference,
    no_o_rescale_reference,
    spike_inputs,
    uniform_inputs,
    unscaled_reference,
    wrong_scale_reference,
)


@lru_cache(maxsize=None)
def _case(kind: str):
    """一类输入的 (Q, K, V, 数学真值, 先验界)——CPU 侧，只算一次。"""
    q, k, v = make_inputs(0, kind, M, N, D, BLOCK_KV)
    return q, k, v, attention_reference(q, k, v), error_bound(q, k, v, BLOCK_KV)


def _run(kernel, q, k, v, device, groups=LAUNCH_BLOCK_DIM):
    """显式 ``kernel[None, groups]`` 启动（mix kernel 的方括号数是执行组数）；输出先填 NaN，漏写的行一眼可见。"""
    out = torch.full([M, D], float("nan"), dtype=torch.float32, device=device)
    kernel[None, groups](q.to(device), k.to(device), v.to(device), out)
    torch.npu.synchronize()
    return out.cpu()


def _where(row: int, col: int) -> str:
    return (
        f"行 {row}（核 {row // ROWS_PER_CORE}、AIV {(row % ROWS_PER_CORE) // HALF}、"
        f"AIV 内第 {row % HALF} 行）列 {col}"
    )


def _assert_within(
    got: torch.Tensor, want: torch.Tensor, bound: torch.Tensor, what: str
) -> float:
    """逐元素 ``|got - want| <= bound``；失败信息直接指出最坏元素落在哪个核 / AIV。返回最大 err/bound 供打印。"""
    got = got.to(torch.float64)
    nan = int(torch.isnan(got).sum())
    assert nan == 0, f"{what}：输出含 {nan} 个 NaN（未写到的位置保持哨兵值）"
    err = (got - want).abs()
    ratio = err / bound
    worst = int(ratio.argmax())
    row, col = divmod(worst, want.shape[1])
    n_bad = int((err > bound).sum())
    assert n_bad == 0, (
        f"{what}：{n_bad} 个元素超界；最坏 err/bound = {ratio.flatten()[worst]:.3g}，在 {_where(row, col)}："
        f"got {got[row, col].item():.6g}，want {want[row, col].item():.6g}，bound {bound[row, col].item():.3g}"
    )
    return float(ratio.max())


@pytest.mark.parametrize("kind", KINDS)
def test_matches_math_reference_within_derived_bound(device, kind):
    q, k, v, want, bound = _case(kind)
    out = _run(flash_attention, q, k, v, device)
    worst = _assert_within(out, want, bound, f"输入类 {kind}")
    print(
        f"[{kind}] max|err| = {(out.to(torch.float64) - want).abs().max().item():.3e}，max err/bound = {worst:.3f}"
    )


def test_serial_and_pipelined_agree_bit_for_bit(device):
    """Require identical output for preload=0 and the main schedule on three inputs.

    Both entries retain the same physical allocation. This checks sensitivity to scheduling for these cases; equality
    does not prove every buffer lifetime.
    """
    for kind in ("normal", "growing", "last_block_max"):
        q, k, v, _, _ = _case(kind)
        serial = _run(flash_attention_p0, q, k, v, device)
        pipelined = _run(flash_attention, q, k, v, device)
        torch.testing.assert_close(
            serial, pipelined, rtol=0, atol=0, msg=lambda m, kind=kind: f"[{kind}] {m}"
        )


def test_two_runs_agree_bit_for_bit(device):
    q, k, v, _, _ = _case("normal")
    first = _run(flash_attention, q, k, v, device)
    second = _run(flash_attention, q, k, v, device)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_uniform_scores_give_mean_of_v(device):
    """``Q = 0``：所有分数为 0，``alpha`` 恒为 1，``l`` 恰为 1024，``O[i] = mean_j V[j]``。

    测的是「每个 key 恰被计入一次」：漏块、重复计块、行和累加错，都会让整体偏离均值一个 ``V`` 块的量级（~1e-1）。
    """
    q, k, v = uniform_inputs(0, M, N, D)
    out = _run(flash_attention, q, k, v, device)
    want = v.to(torch.float64).mean(dim=0, keepdim=True).expand(M, D)
    _assert_within(out, want, error_bound(q, k, v, BLOCK_KV), "均匀分数")


def test_one_hot_scores_select_the_target_key(device):
    """每行恰命中一个 key（``target`` 是 ``[0, N)`` 的置换）：``O[i] = V[target[i]]``。

    覆盖每个块内位置、每个核、每个 AIV 半区：K/V 块下标错位、P 转置错（行列颠倒）、AIV 半区偏移错、核行偏移错，
    都会让某些行拿到别的 key 的 V（差 ~1）。除逐元素界外再要求「每行的输出离自己的目标 V 行最近」，失败时直接给出
    哪些行拿到了哪一行。

    这条输入的块间分数落差在指数域达 ~90，同时守着 running max 的**上溢安全**：若 ``m`` 只取块内最大（不是
    running max），数学上仍等价、随机输入上不可见，但目标块之后的块会算出 ``alpha = exp(+90)`` 而上溢。
    """
    q, k, v, target = spike_inputs(0, M, N, D)
    out = _run(flash_attention, q, k, v, device)
    want = v.to(torch.float64)[target]
    _assert_within(out, want, error_bound(q, k, v, BLOCK_KV), "one-hot 分数")
    picked = torch.cdist(out.float(), v.float()).argmin(dim=1)
    wrong = (picked != target).nonzero().flatten().tolist()
    assert not wrong, (
        f"{len(wrong)} 行输出最接近的不是目标 key 的 V；前几行 (行, 目标, 实得)："
        f"{[(r, int(target[r]), int(picked[r])) for r in wrong[:6]]}"
    )


def test_fewer_groups_touch_only_their_rows(device):
    """只启动 4 个组：核 0..3 的行段正确，核 4..7 的行段保持 NaN 哨兵——各核只写自己的行，且互不依赖。"""
    q, k, v, want, bound = _case("normal")
    groups = NUM_CORES // 2
    out = _run(flash_attention, q, k, v, device, groups=groups)
    rows = groups * ROWS_PER_CORE
    _assert_within(out[:rows], want[:rows], bound[:rows], f"前 {groups} 个核的行段")
    assert bool(torch.isnan(out[rows:]).all()), (
        "未启动的核对应的行段被写入（越界写或行偏移错）"
    )


def test_kernel_never_materializes_a_full_score_row():
    """静态：kernel 里没有任何缓冲的某一维等于 N，KV 块数 > 1（在线形态的结构性前提）。"""
    assert NUM_BLOCKS > 1 and BLOCK_KV < N
    tree = ast.parse(Path(kernel_module.__file__).read_text())
    shapes = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "TileType"
        ):
            shape_node = next(kw.value for kw in node.keywords if kw.arg == "shape")
            shape = eval(
                compile(ast.Expression(shape_node), "<shape>", "eval"),
                vars(kernel_module),
            )
            shapes.append(tuple(shape))
    assert shapes, "no TileType declarations found"
    assert all(max(s) < N for s in shapes), (
        f"存在含 N={N} 维的缓冲：{[s for s in shapes if max(s) >= N]}"
    )


TILE_COMPUTE_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "exp",
    "sum",
    "maximum",
    "minimum",
    "muls",
    "adds",
    "softmax",
    "dequant",
    "astype",
    "sqrt",
    "rsqrt",
    "log",
    "relu",
    "expand_sub",
    "expand_div",
}


def test_kernel_uses_vf_for_vector_compute_and_gemm_for_cube():
    """静态：出货形态的 vector 计算一律走 VF（无 tile 级计算算子），cube matmul 走 ``pl.gemm`` 糖。"""
    tree = ast.parse(Path(kernel_module.__file__).read_text())
    tile_ops, gemms, vfs = [], 0, 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pl"
        ):
            if node.func.attr in TILE_COMPUTE_OPS:
                tile_ops.append(node.func.attr)
            if node.func.attr == "gemm":
                gemms += 1
        if isinstance(node, ast.FunctionDef):
            vfs += any(
                isinstance(d, ast.Attribute) and d.attr == "vector_function"
                for d in node.decorator_list
            )
    assert not tile_ops, f"kernel 里出现 tile 级计算算子：{tile_ops}"
    assert gemms == 2 and vfs >= 3


# ---- 负对照（纯 CPU）：判据必须拒绝各类「错结构」实现 -------------------------------------------------

WRONG_STRUCTURES = {
    "unscaled": ("normal", lambda q, k, v: unscaled_reference(q, k, v)),
    "wrong_scale_1_over_d": ("normal", lambda q, k, v: wrong_scale_reference(q, k, v)),
    "blockwise_softmax_no_cross_block_rescale": (
        "normal",
        lambda q, k, v: blockwise_softmax_reference(q, k, v, BLOCK_KV),
    ),
    "o_accumulation_without_alpha": (
        "growing",
        lambda q, k, v: no_o_rescale_reference(q, k, v, BLOCK_KV),
    ),
    "row_sum_without_alpha": (
        "growing",
        lambda q, k, v: no_l_rescale_reference(q, k, v, BLOCK_KV),
    ),
}


@pytest.mark.parametrize("name", list(WRONG_STRUCTURES))
def test_bound_rejects_wrong_structure(name):
    """同一先验界对错结构参考的判定：超界元素占多数、最坏超界数量级以上——界分得清结构性错误。

    ``first_block_max`` 类只走「max 不更新」分支，漏 alpha 的两类错误在其上与正确实现数学等价（不可见，是定义）；
    所以这两类用 ``growing`` 输入（max 每块都更新）。
    """
    kind, fn = WRONG_STRUCTURES[name]
    q, k, v, want, bound = _case(kind)
    ratio = (fn(q, k, v) - want).abs() / bound
    frac = float((ratio > 1).double().mean())
    assert frac > 0.5 and float(ratio.max()) > 10, (
        f"{name}：超界占比 {frac:.3f}，最坏 err/bound {float(ratio.max()):.3g}"
    )
