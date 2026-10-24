# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Checks for both fixed GQA decode routes and their explicit resource contracts.

The FP64 reference and input-dependent error budget are assembled per (batch,KV-head) group. Both routes cover normal,
growing, extremal and group-distinct inputs, one-hot selection, partial launches, poisoned output/workspace and
zero-tolerance preload/repeat consistency. Incorrect mathematical controls retain the same rejection criterion. Device
selection is inherited from examples/conftest.py.

Static checks inspect native TileType shapes, each route's actual group depths, prefetch guards, memory capacity and
work partitioning. Numerical agreement alone does not prove all synchronization lifetimes or the safety of unused reads.
"""

import ast
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from . import kernel as kernel_module
from .kernel import (
    B,
    BKV,
    CUTS,
    D,
    G,
    HKV,
    HQ,
    ITEMS,
    LAUNCH_BLOCK_DIM,
    MAX_CORES,
    NB,
    NUM_CORES,
    P_SLOTS,
    PRELOAD,
    S,
    ST_SLOTS,
    UCUTS,
    UNITS,
    WS_COLS,
    decode_attention_gqa,
    decode_attention_gqa_fd,
    decode_attention_gqa_fd_p0,
    decode_attention_gqa_p0,
)
from . import reference as ref
from .reference import (
    KINDS,
    attention_reference,
    blockwise_softmax_reference,
    error_bound,
    group_onehot_inputs,
    group_shifted_reference,
    make_inputs,
    no_l_rescale_reference,
    no_o_rescale_reference,
    uniform_inputs,
    unscaled_reference,
    v_head_shifted_reference,
    wrong_scale_reference,
)


def test_reference_constants_match_kernel():
    assert (ref.B, ref.HQ, ref.HKV, ref.S, ref.BLOCK_KV) == (B, HQ, HKV, S, BKV)


@lru_cache(maxsize=None)
def _case(kind: str):
    q, k, v = make_inputs(0, kind)
    return q, k, v, attention_reference(q, k, v), error_bound(q, k, v)


FD_KERNELS = (decode_attention_gqa_fd, decode_attention_gqa_fd_p0)


def _run(kernel, q, k, v, device, groups=LAUNCH_BLOCK_DIM):
    """输出 NaN 毒化（漏写行一眼可见）。fd 形态的 ``ws`` 同样 NaN 毒化——部分状态记录必须
    先写后读（phase 1 落盘、屏障、phase 2 才读），毒化让任何「读了没写的记录」直接显形。"""
    out = torch.full([B, HQ, 1, D], float("nan"), dtype=torch.float32, device=device)
    if kernel in FD_KERNELS:
        ws = torch.full(
            [ITEMS, 2, G, WS_COLS], float("nan"), dtype=torch.float32, device=device
        )
        kernel[None, groups](q.to(device), k.to(device), v.to(device), ws, out)
    else:
        kernel[None, groups](q.to(device), k.to(device), v.to(device), out)
    torch.npu.synchronize()
    return out.cpu()


def _where(b: int, hq: int, col: int) -> str:
    item = b * HKV + hq // G
    core = next(c for c in range(NUM_CORES) if CUTS[c] <= item < CUTS[c + 1])
    return f"(b={b}, hq={hq}, d={col})（item {item}、核 {core}、组内行 {hq % G}）"


def _assert_within(got, want, bound, what: str) -> float:
    got = got.to(torch.float64)
    nan = int(torch.isnan(got).sum())
    assert nan == 0, f"{what}：输出含 {nan} 个 NaN（未写到的位置保持哨兵值）"
    err = (got - want).abs()
    ratio = err / bound
    worst = int(ratio.argmax())
    b, r = divmod(worst, HQ * D)
    hq, col = divmod(r, D)
    n_bad = int((err > bound).sum())
    assert n_bad == 0, (
        f"{what}：{n_bad} 个元素超界；最坏 err/bound = {ratio.flatten()[worst]:.3g} 在 "
        f"{_where(b, hq, col)}：got {got.flatten()[worst].item():.6g}，"
        f"want {want.flatten()[worst].item():.6g}，bound {bound.flatten()[worst].item():.3g}"
    )
    return float(ratio.max())


ROUTES = {"a": decode_attention_gqa, "fd": decode_attention_gqa_fd}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("route", list(ROUTES))
def test_matches_math_reference_within_derived_bound(device, route, kind):
    q, k, v, want, bound = _case(kind)
    out = _run(ROUTES[route], q, k, v, device)
    worst = _assert_within(out, want, bound, f"[{route}] 输入类 {kind}")
    print(
        f"[{route}/{kind}] max|err| = "
        f"{(out.to(torch.float64) - want).abs().max().item():.3e}，"
        f"max err/bound = {worst:.3f}"
    )


@pytest.mark.parametrize("route", list(ROUTES))
def test_serial_and_pipelined_agree_bit_for_bit(device, route):
    """Require identical output for serial and main schedules on three inputs per route."""
    serial_k = {"a": decode_attention_gqa_p0, "fd": decode_attention_gqa_fd_p0}[route]
    for kind in ("normal", "growing", "last_block_max"):
        q, k, v, _, _ = _case(kind)
        serial = _run(serial_k, q, k, v, device)
        pipelined = _run(ROUTES[route], q, k, v, device)
        torch.testing.assert_close(
            serial,
            pipelined,
            rtol=0,
            atol=0,
            msg=lambda m, kind=kind: f"[{route}/{kind}] {m}",
        )


@pytest.mark.parametrize("route", list(ROUTES))
def test_two_runs_agree_bit_for_bit(device, route):
    q, k, v, _, _ = _case("normal")
    first = _run(ROUTES[route], q, k, v, device)
    second = _run(ROUTES[route], q, k, v, device)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("route", list(ROUTES))
def test_uniform_scores_give_mean_of_own_head_v(device, route):
    """``q = 0``：每组输出 = **自己 KV 头** V 的按 S 均值——漏块 / 重块 / 组映射错都会偏离。"""
    q, k, v = uniform_inputs(0)
    out = _run(ROUTES[route], q, k, v, device)
    want = (
        v.to(torch.float64)
        .mean(dim=2, keepdim=True)  # [B, HKV, 1, D]
        .repeat_interleave(G, dim=1)
    )  # [B, HQ, 1, D]
    _assert_within(out, want, error_bound(q, k, v), f"[{route}] 均匀分数")


@pytest.mark.parametrize("route", list(ROUTES))
def test_group_onehot_selects_target_key_in_own_head(device, route):
    """每个 q 行恰命中自己组 KV 头里的一个 key：``O[b,hq] = V[b, hq//G, target]``。

    组映射（本题最可能的静默缺陷）、KV 块下标、AIV 半区、核 item/单元区间中的任何错位都会让
    某些行拿到别的 key / 别的头的 V。除逐元素界外，再要求每行输出在自己头的全部 V 行里最近邻
    恰为 target。fd 形态下这条同时压过部分收尾与 merge（one-hot 命中块随机落在两分部两侧）。
    """
    q, k, v, target = group_onehot_inputs(0)
    out = _run(ROUTES[route], q, k, v, device)
    want = torch.zeros([B, HQ, 1, D], dtype=torch.float64)
    for b in range(B):
        for hq in range(HQ):
            want[b, hq, 0, :] = v[b, hq // G, target[b, hq]].to(torch.float64)
    _assert_within(out, want, error_bound(q, k, v), "组内 one-hot")
    wrong = []
    for b in range(B):
        for hq in range(HQ):
            d2 = torch.cdist(out[b, hq].float(), v[b, hq // G].float()).argmin()
            if int(d2) != int(target[b, hq]):
                wrong.append((b, hq, int(target[b, hq]), int(d2)))
    assert not wrong, (
        f"{len(wrong)} 行最近邻不是目标 key；前几行 (b, hq, 目标, 实得)：{wrong[:6]}"
    )


@pytest.mark.parametrize("route", list(ROUTES))
def test_fewer_groups_touch_only_their_item_range(device, route):
    """只启动 14 个组：前一半 item 覆盖的行正确，其余保持 NaN 哨兵。

    两条 route 在 14 核处的覆盖恰好相同：CUTS[14] = 64 个 item；UCUTS[14] = 1024 单元恰是
    item 边界（1024 % NB == 0，import 期断言），无跨半截断，item 64 起整段未写。
    """
    assert (
        UCUTS[NUM_CORES // 2] % NB == 0
        and UCUTS[NUM_CORES // 2] // NB == CUTS[NUM_CORES // 2]
    )
    q, k, v, want, bound = _case("normal")
    groups = NUM_CORES // 2
    covered = CUTS[groups]  # 64 个 item = b 0..7 的全部组
    out = _run(ROUTES[route], q, k, v, device, groups=groups)
    rows_b = covered // HKV
    _assert_within(
        out[:rows_b], want[:rows_b], bound[:rows_b], f"[{route}] 前 {groups} 核的段"
    )
    assert bool(torch.isnan(out[rows_b:]).all()), (
        "未启动核的 item 段被写入（切分或行偏移错）"
    )


# ---- 静态结构测试（无卡） -----------------------------------------------------------------------


def _kernel_tree() -> ast.Module:
    return ast.parse(Path(kernel_module.__file__).read_text())


def test_kernel_never_materializes_a_full_kv_matrix():
    assert NB > 1 and BKV < S
    shapes = []
    for node in ast.walk(_kernel_tree()):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "TileType"
        ):
            shape_node = next(k.value for k in node.keywords if k.arg == "shape")
            shape = eval(
                compile(ast.Expression(shape_node), "<shape>", "eval"),
                vars(kernel_module),
            )
            shapes.append(tuple(shape))
    assert shapes and all(max(s) < S for s in shapes), (
        f"存在含 S={S} 维的缓冲：{[s for s in shapes if max(s) >= S]}"
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
    tile_ops, gemms, vfs = [], 0, 0
    for node in ast.walk(_kernel_tree()):
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
    assert gemms == 2 and vfs >= 3  # 共享 stage：QK 与 PV 各一处 gemm；VF 含 merge


def test_slot_counts_match_beat_distance_formula():
    """Check actual group depths in both routes against their stage distances."""
    assert ST_SLOTS == PRELOAD + 1
    assert P_SLOTS == max(PRELOAD - 1, 0) + 1
    kernels = [
        node
        for node in ast.walk(_kernel_tree())
        if isinstance(node, ast.FunctionDef) and node.name == "kernel"
    ]
    assert len(kernels) == 2
    for kernel in kernels:
        depths = {}
        for node in ast.walk(kernel):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "make_tile_group"
                and isinstance(node.targets[0], ast.Name)
            ):
                kw = {k.arg: k.value for k in node.value.keywords}
                depths[node.targets[0].id] = eval(
                    compile(ast.Expression(kw["depth"]), "<depth>", "eval"),
                    vars(kernel_module),
                )
        assert depths.get("shared_s") == 2
        assert depths.get("shared_pv") == 2
        assert depths.get("shared_p") == P_SLOTS
        assert depths.get("st_a") == ST_SLOTS
        assert depths.get("st_l") == ST_SLOTS
        assert depths.get("st_m") == ST_SLOTS


def test_next_item_prefetches_are_guarded_by_range_checks():
    """Require range guards on cross-item prefetches, including reads whose result may be unused."""
    src = Path(kernel_module.__file__).read_text()
    tree = ast.parse(src)

    def guarded_loads(fn, guard_test):
        hits = []
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and guard_test(ast.unparse(node.test)):
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "load"
                    ):
                        hits.append(sub)
        return hits

    def named(name):
        return [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        ]

    assert guarded_loads(named("qk_block")[0], lambda t: t == "has_next == 1"), (
        "qk 预取无守卫"
    )
    assert guarded_loads(named("pv_block")[0], lambda t: t == "has_next == 1"), (
        "pv 预取无守卫"
    )
    kernels = named("kernel")
    assert len(kernels) == 2, "应有 route A 与 fd 两个 kernel 定义"
    q_guards = {"it + 1 < end", "it + 1 <= last_item"}
    for fn in kernels:
        assert guarded_loads(fn, lambda t: t in q_guards), (
            f"{fn.lineno} 行 kernel 的 Q 预取无守卫"
        )


def test_l1_budget_fits():
    """Mat 用量 <= 524288B：Q 槽 + K/V 流环 + P 跨核槽。"""
    from .kernel import KV_SLOTS, MP, Q_SLOTS

    q_b = Q_SLOTS * MP * D * 2
    kv_b = 2 * KV_SLOTS * BKV * D * 2
    p_b = P_SLOTS * MP * BKV * 2
    assert q_b + kv_b + p_b <= 524288, (q_b, kv_b, p_b)


def test_partition_covers_items_contiguously_and_evenly():
    assert len(CUTS) == MAX_CORES + 1 and CUTS[0] == 0 and CUTS[-1] == ITEMS
    sizes = [CUTS[c + 1] - CUTS[c] for c in range(NUM_CORES)]
    assert all(s >= 1 for s in sizes)
    assert max(sizes) - min(sizes) <= 1, sizes


def test_fd_unit_partition_invariants():
    """fd 切分的结构不变量：连续覆盖、近均衡、每核 >= 2*NB 单元（stage 逻辑依赖
    first_item != last_item，即「首块即末块」只出现在部分 segment），item 至多跨 2 核。
    """
    assert len(UCUTS) == MAX_CORES + 1 and UCUTS[0] == 0 and UCUTS[-1] == UNITS
    sizes = [UCUTS[c + 1] - UCUTS[c] for c in range(NUM_CORES)]
    assert max(sizes) - min(sizes) <= 1, sizes
    assert min(sizes) >= 2 * NB, sizes  # 每 item 至多两分部 + first!=last
    straddles = sum(1 for c in range(1, NUM_CORES) if UCUTS[c] % NB != 0)
    assert straddles <= NUM_CORES - 1


# ---- 负对照（纯 CPU）：判据必须拒绝各类「错结构」实现 -------------------------------------------


WRONG_STRUCTURES = {
    "unscaled": ("normal", lambda q, k, v: unscaled_reference(q, k, v)),
    "wrong_scale_1_over_d": ("normal", lambda q, k, v: wrong_scale_reference(q, k, v)),
    "blockwise_softmax_no_rescale": (
        "normal",
        lambda q, k, v: blockwise_softmax_reference(q, k, v),
    ),
    "o_accumulation_without_alpha": (
        "growing",
        lambda q, k, v: no_o_rescale_reference(q, k, v),
    ),
    "row_sum_without_alpha": (
        "growing",
        lambda q, k, v: no_l_rescale_reference(q, k, v),
    ),
    "group_mapping_shifted": (
        "group_distinct",
        lambda q, k, v: group_shifted_reference(q, k, v),
    ),
    "v_head_shifted": (
        "group_distinct",
        lambda q, k, v: v_head_shifted_reference(q, k, v),
    ),
}


@pytest.mark.parametrize("name", list(WRONG_STRUCTURES))
def test_bound_rejects_wrong_structure(name):
    kind, fn = WRONG_STRUCTURES[name]
    q, k, v, want, bound = _case(kind)
    ratio = (fn(q, k, v) - want).abs() / bound
    frac = float((ratio > 1).double().mean())
    assert frac > 0.3 and float(ratio.max()) > 10, (
        f"{name}：超界占比 {frac:.3f}，最坏 err/bound {float(ratio.max()):.3g}"
    )
