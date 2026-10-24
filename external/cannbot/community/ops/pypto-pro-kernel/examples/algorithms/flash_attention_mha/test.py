# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Checks for the fixed BSND causal-MHA kernel.

The independent FP64 causal reference and input-dependent error budget live in reference.py. Cases cover
staircase/first-row/poison boundaries, per-owner payload identity, growing and extremal scores, uniform prefixes,
one-hot selection, partial launches and zero-tolerance preload/repeat consistency. Nine incorrect mathematical controls
exercise the comparator; poison checks select only rows with finite visible inputs.

All shared edges declare per-slot forward/reverse events. The m/l/alpha ring has PRELOAD+1 slots, distinct from
cross-core depths. Static checks cover native TileType shapes, Mat capacity, owner partitioning and each
next-item/next-block prefetch guard. Numerical agreement cannot establish safety of an unused prefetch or prove all
synchronization lifetimes. Device selection is inherited from examples/conftest.py.
"""

import ast
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from . import kernel as kernel_module
from .kernel import (
    BATCH,
    BKV,
    CUTS,
    D,
    HEADS,
    ITEMS,
    LAUNCH_BLOCK_DIM,
    NUM_CORES,
    OWNERS,
    QB,
    SEQ,
    flash_attention_mha,
    flash_attention_mha_p0,
    weighted_partition,
)
from .reference import (
    KINDS,
    attention_reference,
    batch_shifted_reference,
    block_causal_reference,
    checked_rows,
    error_bound,
    head_shifted_reference,
    make_inputs,
    no_l_rescale_reference,
    no_o_rescale_reference,
    noncausal_reference,
    spike_inputs,
    strict_causal_reference,
    unscaled_reference,
    wrong_scale_reference,
)


@lru_cache(maxsize=None)
def _case(kind: str):
    """一类输入的 (Q, K, V, 数学真值, 先验界, 该检查哪些行)——CPU 侧，只算一次。"""
    q, k, v = make_inputs(0, kind, BATCH, SEQ, HEADS, D, BKV)
    want = attention_reference(q, k, v)
    bound = error_bound(q, k, v, BKV)
    rows = checked_rows(kind, BATCH, SEQ, HEADS, BKV)
    return q, k, v, want, bound, rows


def _run(kernel, q, k, v, device, groups=LAUNCH_BLOCK_DIM):
    """显式 ``kernel[None, groups]`` 启动；输出先填 NaN，漏写的位置一眼可见。"""
    out = torch.full(
        [BATCH, SEQ, HEADS, D], float("nan"), dtype=torch.float32, device=device
    )
    kernel[None, groups](q.to(device), k.to(device), v.to(device), out)
    torch.npu.synchronize()
    return out.cpu()


def _where(b: int, i: int, h: int) -> str:
    item = (b * HEADS + h) * QB + i // (SEQ // QB)
    core = next((c for c in range(NUM_CORES) if CUTS[c] <= item < CUTS[c + 1]), -1)
    return (
        f"batch {b} 行 {i}（q 块 {i // (SEQ // QB)}）head {h}，item {item}，核 {core}"
    )


def _assert_within(
    got: torch.Tensor,
    want: torch.Tensor,
    bound: torch.Tensor,
    rows: torch.Tensor,
    what: str,
) -> float:
    """``rows`` 内逐元素 ``|got-want| <= bound``；失败信息指出最坏元素落在哪个 item/核。返回最大 err/bound。"""
    got = got.to(torch.float64)
    m = rows[..., None].expand_as(got)
    nan = int(torch.isnan(got[m]).sum())
    assert nan == 0, f"{what}：应检查的位置里有 {nan} 个 NaN（未写到的位置保持哨兵值）"
    err = torch.where(m, (got - want).abs(), torch.zeros_like(got))
    ratio = torch.where(m, err / bound, torch.zeros_like(err))
    n_bad = int((ratio > 1).sum())
    worst = int(ratio.flatten().argmax())
    b_, i_, h_, d_ = (
        int(x) for x in torch.unravel_index(torch.tensor(worst), ratio.shape)
    )
    assert n_bad == 0, (
        f"{what}：{n_bad} 个元素超界；最坏 err/bound = {ratio.flatten()[worst]:.3g}，在 {_where(b_, i_, h_)} 列 {d_}："
        f"got {got[b_, i_, h_, d_].item():.6g}，want {want[b_, i_, h_, d_].item():.6g}"
    )
    return float(ratio.max())


@pytest.mark.parametrize("kind", KINDS)
def test_matches_math_reference_within_derived_bound(device, kind):
    q, k, v, want, bound, rows = _case(kind)
    out = _run(flash_attention_mha, q, k, v, device)
    worst = _assert_within(out, want, bound, rows, f"输入类 {kind}")
    print(f"[{kind}] max err/bound = {worst:.3f}")


def test_serial_and_pipelined_agree_bit_for_bit(device):
    """Require identical output for preload=0 and the main schedule on three inputs.

    Both entries retain the same state-ring and cross-core depths. This checks sensitivity to scheduling for these
    cases; equality does not prove every buffer lifetime.
    """
    for kind in ("normal", "growing", "staircase"):
        q, k, v, _, _, _ = _case(kind)
        serial = _run(flash_attention_mha_p0, q, k, v, device)
        pipelined = _run(flash_attention_mha, q, k, v, device)
        torch.testing.assert_close(
            serial, pipelined, rtol=0, atol=0, msg=lambda m, kind=kind: f"[{kind}] {m}"
        )


def test_two_runs_agree_bit_for_bit(device):
    q, k, v, _, _, _ = _case("normal")
    first = _run(flash_attention_mha, q, k, v, device)
    second = _run(flash_attention_mha, q, k, v, device)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_causal_first_row_sees_only_itself(device):
    """每个 (batch, head) 的第 0 行只有 1 个可见 key（自己）：softmax 权重恒为 1，``O[b,0,h,:] == V[b,0,h,:]``。

    覆盖「因果掩码在最短可见集上的边界」：漏掉因果、掩码差一、块级掩码在这一行都会读到别的 key，误差远超
    FP16 舍入量级。
    """
    q, k, v, _, bound, _ = _case("normal")
    out = _run(flash_attention_mha, q, k, v, device)
    want_row0 = v[:, 0:1, :, :].to(torch.float64)
    got_row0 = out[:, 0:1, :, :].to(torch.float64)
    err = (got_row0 - want_row0).abs()
    assert bool(torch.all(err <= bound[:, 0:1, :, :])), (
        f"第 0 行 max|err| {err.max().item():.3e} 超出先验界"
    )


def test_diagonal_block_leak_is_caught_by_staircase(device):
    """``staircase`` kind：对角块内被掩的 key 全是「热」的（分数比可见 key 高 ~45 个指数域单位），泄漏一个就是
    O(1) 误差——专门盖住「转置布局下阶梯偏置 tri 的行/列搞反」这类错误（``tri`` 用错半区会让上半 AIV 的泄漏
    从第 0 块开始，下半 AIV 从第二个对角块开始，签名不同，方便定位）。
    """
    q, k, v, want, bound, rows = _case("staircase")
    out = _run(flash_attention_mha, q, k, v, device)
    _assert_within(out, want, bound, rows, "staircase 对角块泄漏探测")


@pytest.mark.parametrize("kind", ["poison_future"])
def test_poison_future_rows_never_touch_masked_keys(device, kind):
    """每个 owner 在自己对角块之后的 K/V 是 NaN；只检查该 owner 对角块及之前的行——它们的因果可见集完全不含
    NaN，读到就会得 NaN，被 ``checked_rows`` 排除在外的行不作要求（owner 缓存的槽位复用如果读错块号，最常见
    的后果正是把还没轮到、已被 NaN 污染的块读进可见集）。
    """
    q, k, v, want, bound, rows = _case(kind)
    out = _run(flash_attention_mha, q, k, v, device)
    _assert_within(out, want, bound, rows, "poison_future")


def test_uniform_scores_give_causal_prefix_mean(device):
    """``Q=0``：所有可见分数为 0，``alpha`` 恒为 1，``O[b,i,h,:] = mean_{j<=i} V[b,j,h,:]``（因果前缀均值，
    随 i 增长而变，与单头样例的全局均值不同）。漏块、重复计块、行和累加错都会让整体偏离该行专属的前缀均值。
    """
    q, k, v, want, bound, rows = _case("uniform")
    out = _run(flash_attention_mha, q, k, v, device)
    _assert_within(out, want, bound, rows, "均匀分数（因果前缀均值）")


def test_one_hot_scores_select_the_causal_target_key(device):
    """每行恰命中一个**可见**（``target <= i``）key：``O[b,i,h,:] = V[b,target,h,:]``。覆盖每个块内位置、
    每个核、每个 AIV 半区、每个 K/V 缓存槽：块下标错位、P 转置错、AIV 半区偏移错、核行偏移错、owner 缓存槽位
    串号，都会让某些行拿到别的 key 的 V。除逐元素界外再要求「每行输出离自己的目标 V 行最近」。
    """
    q, k, v, target = spike_inputs(0, BATCH, SEQ, HEADS, D)
    out = _run(flash_attention_mha, q, k, v, device)
    want = v.to(torch.float64)[
        torch.arange(BATCH)[:, None, None], target, torch.arange(HEADS)[None, None, :]
    ]
    bound = error_bound(q, k, v, BKV)
    _assert_within(
        out, want, bound, torch.ones_like(target, dtype=torch.bool), "one-hot 因果分数"
    )
    v_f = v.float()
    picked = torch.empty_like(target)
    for b in range(BATCH):
        for h in range(HEADS):
            picked[b, :, h] = torch.cdist(out[b, :, h].float(), v_f[b, :, h]).argmin(
                dim=1
            )
    wrong = (picked != target).nonzero().tolist()
    assert not wrong, (
        f"{len(wrong)} 行输出最接近的不是目标 key 的 V；前几个 (b, i, h)：{wrong[:6]}"
    )


def test_fewer_groups_touch_only_their_item_range(device):
    """只启动 14 个组：核 0..13 覆盖的 item 对应的行段正确，其余行段保持 NaN 哨兵——各核只写自己的 item 区间。

    用 item -> (b, q 块, h) 的精确映射逐 item 标出「该被写」的位置，而不是猜一个整体行切点——item 化之后
    切点不再是行号的简单线性函数（owner 顺序穿插了 batch 与 head 两个维度）。
    """
    q, k, v, want, bound, rows = _case("normal")
    groups = NUM_CORES // 2
    out = _run(flash_attention_mha, q, k, v, device, groups=groups)
    boundary_item = CUTS[groups]
    bq = SEQ // QB
    started = torch.zeros(BATCH, SEQ, HEADS, dtype=torch.bool)
    for it in range(boundary_item):
        owner, g = divmod(it, QB)
        b, h = divmod(owner, HEADS)
        started[b, slice(g * bq, (g + 1) * bq), h] = True
    worst = _assert_within(
        out, want, bound, rows & started, f"前 {groups} 个核覆盖的 item"
    )
    untouched = ~started
    assert bool(torch.isnan(out[untouched]).all()), (
        "未启动的核对应的 item 段被写入（越界写或 item 映射算错）"
    )
    print(f"[fewer_groups] max err/bound over started items = {worst:.3f}")


def test_partition_covers_items_contiguously_and_is_balanced():
    """``CUTS`` 是 host 侧算好的字面量表：覆盖 ``[0, ITEMS)`` 无缝无重叠，且 28 段的权重（步数）方差很小
    （权重切分不是摆设——注释掉它、按 item 数等分会让最重的核比最轻的核多 ~40% 步数，见 kernel docstring）。
    """
    assert CUTS[0] == 0 and CUTS[NUM_CORES] == ITEMS
    assert all(CUTS[c] <= CUTS[c + 1] for c in range(NUM_CORES)), (
        "存在切点倒退（某核区间为负长度以外的非法状态）"
    )
    steps = [
        sum(i % QB + 1 for i in range(CUTS[c], CUTS[c + 1])) for c in range(NUM_CORES)
    ]
    assert all(s > 0 for s in steps), f"存在空核（0 步）：{steps}"
    spread = (max(steps) - min(steps)) / (sum(steps) / NUM_CORES)
    assert spread < 0.10, f"28 核步数极差 {spread:.1%} 超过 10%，权重切分失衡：{steps}"


def test_weighted_partition_beats_equal_split_on_balance():
    """Compare weighted and equal-item partitions under the same causal block-count model.

    This is a host workload-balance check, not a device-time measurement.
    """
    weighted_cuts = weighted_partition(ITEMS, NUM_CORES)
    per_core = ITEMS // NUM_CORES
    equal_cuts = tuple(c * per_core for c in range(NUM_CORES)) + (ITEMS,)

    def spread(cuts):
        steps = [
            sum(i % QB + 1 for i in range(cuts[c], cuts[c + 1]))
            for c in range(NUM_CORES)
        ]
        return (max(steps) - min(steps)) / (sum(steps) / NUM_CORES)

    w_spread, e_spread = spread(weighted_cuts), spread(equal_cuts)
    assert w_spread < e_spread / 2, (
        f"权重切分极差 {w_spread:.1%} 未明显小于等分极差 {e_spread:.1%}"
    )


def test_kernel_never_materializes_a_full_kv_matrix():
    """静态：kernel 里没有任何缓冲的某一维等于 SEQ（在线形态的结构性前提，KV 块数 > 1）。"""
    assert SEQ // BKV > 1
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
    assert all(max(s) < SEQ for s in shapes), (
        f"存在含 SEQ={SEQ} 维的缓冲：{[s for s in shapes if max(s) >= SEQ]}"
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


def test_owner_cache_slot_count_fits_l1_budget():
    """Check the current Q, cached/streaming K/V and P depths against the 512 KiB Mat budget."""
    from .kernel import KV_SLOTS, P_SLOTS, Q_SLOTS

    l1_total = 512 * 1024
    tile = 128 * 128 * 2  # 每个 [128,128] FP16 槽
    used = Q_SLOTS * tile  # q_g
    used += KV_SLOTS * tile * 2  # k_g + v_g
    used += P_SLOTS * tile  # shared_p
    assert used <= l1_total, f"L1(Mat) 用量 {used}B 超过总容量 {l1_total}B"


def test_prefetch_slot_constants_are_consistent():
    """静态：预取方案对槽数的结构性要求。块内预取在第 j 步发射第 j+1 块的载入，流式环至少 2 槽，否则预取会把
    当前步正在读的块覆盖掉；Q 要双槽才能在 item 顶部预取下一个 item 的 Q；同核跨 stage 的状态环、跨核 P 交接槽
    的深度各自随 ``PRELOAD`` 走（框架只校验后者，前者由用户负责——见 ``test_serial_and_pipelined_agree_bit_for_bit``）。
    """
    from .kernel import KV_SLOTS, NK, NSK, P_SLOTS, PRELOAD, Q_SLOTS, ST_SLOTS

    assert KV_SLOTS == NK + NSK and NK >= 1
    assert NSK >= 2, "流式环不足 2 槽：块内预取会覆盖当前步还在读的块"
    assert Q_SLOTS >= 2, "Q 单槽无法在 item 顶部预取下一个 item 的 Q"
    assert ST_SLOTS == PRELOAD + 1
    assert P_SLOTS >= max(PRELOAD, 1), "P 交接槽数低于 拍距+1（拍距 = PRELOAD-1）"


def _calls_with_guards(fn_node):
    """``fn_node`` 里每个 Call 连同「包住它的 if 条件」（只算 if 的主体分支，条件文本用 ``ast.unparse``）。"""
    found = []

    def walk(node, guards):
        if isinstance(node, ast.Call):
            found.append((node, tuple(guards)))
        if isinstance(node, ast.If):
            walk(node.test, guards)
            for child in node.body:
                walk(child, guards + [ast.unparse(node.test)])
            for child in node.orelse:
                walk(child, guards)
            return
        for child in ast.iter_child_nodes(node):
            walk(child, guards)

    walk(fn_node, [])
    return found


def test_next_item_prefetches_are_guarded_by_range_checks():
    """Require guards on Q next-item, K/V next-owner and K/V next-block prefetches.

    An unused out-of-range read may not affect compared output. These static conditions therefore remain separate from
    numerical acceptance.
    """
    tree = ast.parse(Path(kernel_module.__file__).read_text())
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    def guards_of_loads(fn_name, dst_is):
        fn = funcs[fn_name]
        hits = [
            g
            for call, g in _calls_with_guards(fn)
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "load"
            and call.args
            and dst_is(ast.unparse(call.args[0]))
        ]
        assert hits, (
            f"{fn_name} 里没有找到目标匹配的 pl.load（预取写法改了？同步改这条测试）"
        )
        return hits

    cases = [
        ("kernel", lambda dst: "nxt" in dst, "it + 1 < end", "Q 的下一 item 预取"),
        ("qk_block", lambda dst: dst == "x_tile", "has_next == 1", "K 的跨 item 预取"),
        ("pv_block", lambda dst: dst == "x_tile", "has_next == 1", "V 的跨 item 预取"),
        ("qk_block", lambda dst: dst == "nk_tile", "j < g", "K 的块内预取"),
        ("pv_block", lambda dst: dst == "nv_tile", "j < g", "V 的块内预取"),
    ]
    for fn_name, dst_is, guard, what in cases:
        for guards in guards_of_loads(fn_name, dst_is):
            assert guard in guards, (
                f"{what}没有守在 `if {guard}` 里（实际外层条件：{list(guards)}）"
            )


# ---- 负对照（纯 CPU）：判据必须拒绝各类「错结构」实现 -------------------------------------------------

WRONG_STRUCTURES = {
    "noncausal": ("normal", lambda q, k, v: noncausal_reference(q, k, v)),
    "strict_causal_off_by_one": (
        "normal",
        lambda q, k, v: strict_causal_reference(q, k, v),
    ),
    "block_causal_no_diag_mask": (
        "staircase",
        lambda q, k, v: block_causal_reference(q, k, v, BKV),
    ),
    "unscaled": ("normal", lambda q, k, v: unscaled_reference(q, k, v)),
    "wrong_scale_1_over_d": ("normal", lambda q, k, v: wrong_scale_reference(q, k, v)),
    "head_shifted": ("normal", lambda q, k, v: head_shifted_reference(q, k, v)),
    "batch_shifted": ("normal", lambda q, k, v: batch_shifted_reference(q, k, v)),
    "o_accumulation_without_alpha": (
        "growing",
        lambda q, k, v: no_o_rescale_reference(q, k, v, BKV),
    ),
    "row_sum_without_alpha": (
        "growing",
        lambda q, k, v: no_l_rescale_reference(q, k, v, BKV),
    ),
}


@pytest.mark.parametrize("name", list(WRONG_STRUCTURES))
def test_bound_rejects_wrong_structure(name):
    """同一先验界对错结构参考的判定：超界元素占多数、最坏超界数量级以上——界分得清结构性错误，不是形同虚设。

    ``block_causal`` 用 ``staircase``（对角块整块可见 vs 半掩，staircase 的块内落差达 O(1) 放大这个差别）；
    漏 alpha 的两类错误在「max 不更新」的输入上与正确实现数学等价（不可见，是定义），所以用 ``growing``。
    """
    kind, fn = WRONG_STRUCTURES[name]
    q, k, v, want, bound, rows = _case(kind)
    wrong = fn(q, k, v)
    m = rows[..., None].expand_as(bound)
    ratio = torch.where(m, (wrong - want).abs() / bound, torch.zeros_like(bound))
    frac = float((ratio > 1).double().mean())
    assert frac > 0.5 and float(ratio.max()) > 10, (
        f"{name}：超界占比 {frac:.3f}，最坏 err/bound {float(ratio.max()):.3g}"
    )


def test_owners_span_multiple_cores_and_cores_span_multiple_owners():
    """切分健康性的最后一条：确认这个 shape 下真的存在「一个核跨多个 owner」与「一个 owner 跨切点被截断」两种
    情形——如果 128 个 owner 恰好整除到 28 核、每核只服务整数个 owner，owner 缓存的 ``fo`` 边界条件就测不到。
    """
    owners_per_core = [
        {it // QB for it in range(CUTS[c], CUTS[c + 1])} for c in range(NUM_CORES)
    ]
    assert any(len(s) > 1 for s in owners_per_core), (
        "没有任何核跨越了多个 owner——fo 边界条件未被覆盖"
    )
    split_owners = set()
    for c in range(NUM_CORES - 1):
        boundary = CUTS[c + 1]
        if boundary < ITEMS and boundary % QB != 0:
            split_owners.add(boundary // QB)
    assert split_owners, (
        "没有任何 owner 被核边界从中间切开——核边界重载分支（fo=1 且 j<g）未被覆盖"
    )
    assert OWNERS * QB == ITEMS
