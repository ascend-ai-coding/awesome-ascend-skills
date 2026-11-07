# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Board checks for M=1000 across eight groups, two bands and two AIVs.

Every group owns 125 rows with 64/61-row bands; the final AIV windows are [939,971) and [971,1000). Tests cover all
rows, every tail band, group seams, individual boundary rows and failure messages naming the responsible group/band/AIV.

The independent reference computes RMSNorm in float64, narrows to FP16 once, then uses float64 matmul. The numerical
gate remains rtol=atol=2e-2. A second criterion fits each row's error to at most three adjacent-FP16 changes in the
intermediate P, each contributing delta * W[k,:], with residual noise at most 1e-4. This supplements the numerical gate;
it is not a proof that every possible structural defect is excluded.

preload=0/2, repeated calls, guarded output, poisoned input padding and sanitizer output must agree bit for bit. Output
guards detect writes past M and unwritten rows; poisoning checks whether invalid input values reach useful output;
sanitizer checks logical access windows. Their visibility differs, as described in
references/diagnostics/check-coverage.md.
"""

from __future__ import annotations

import pytest
import torch

from .reference import assert_rows, owner, unexplained_rows, rmsnorm_gemm_reference
from .kernel import (
    BAND_ROWS,
    HALF_BAND,
    K,
    LAUNCH_BLOCK_DIM,
    M,
    N,
    NUM_BANDS_PER_CORE,
    NUM_CORES,
    ROWS_PER_CORE,
    TAIL_BAND_ROWS,
    rmsnorm_gemm_tail,
    rmsnorm_gemm_tail_p2,
    rmsnorm_gemm_tail_san,
)

SEED = 0


@pytest.fixture(scope="session")
def host_inputs():
    """CPU 生成、固定种子：同一份输入在任何机器上可复现（离线能重算容差账）。"""
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn([M, K], generator=g).to(torch.float16)
    gamma = torch.randn([K], generator=g).to(torch.float16)
    w = torch.randn([K, N], generator=g).to(torch.float16)
    return x, gamma, w


@pytest.fixture(scope="session")
def dev_inputs(device, host_inputs):
    return tuple(t.to(device) for t in host_inputs)


@pytest.fixture(scope="session")
def golden(host_inputs):
    return rmsnorm_gemm_reference(*host_inputs)


RTOL = ATOL = 2e-2
NOISE = 1e-4  # FP32 累加噪声底的判据阈值（见文件头「容差依据」）
GUARD_ROWS = BAND_ROWS  # X / O 之后留的守卫行，大于任何合理越界量
SENTINEL_BITS = 0x7FC0DEAD  # 带载荷的 quiet NaN：kernel 的算术不会产出这个位型
PLAN = dict(
    rows_per_core=ROWS_PER_CORE,
    band_rows=BAND_ROWS,
    half_band=HALF_BAND,
    num_bands=NUM_BANDS_PER_CORE,
)

LAST_CORE = NUM_CORES - 1
TAIL_BAND_START = (
    LAST_CORE * ROWS_PER_CORE + (NUM_BANDS_PER_CORE - 1) * BAND_ROWS
)  # 939


# --------------------------------------------------------------------------------------
# device 来自 examples/conftest.py；输入与参考 fixture 定义在本文件
# --------------------------------------------------------------------------------------
def _launch(kernel, x, gamma, w, out):
    """显式 ``kernel[None, LAUNCH_BLOCK_DIM]``：mix kernel 的方括号数是执行组数（1c+2v/组），
    不是 AIV 线程数；不依赖 eager 直调的 auto 哨兵（用满 stream 预算 = 幽灵组越界写）。
    """
    kernel[None, LAUNCH_BLOCK_DIM](x, gamma, w, out)
    torch.npu.synchronize()


def _nan_out(device: str) -> torch.Tensor:
    """NaN 预填充：没被写到的行会以 NaN 现形，而不是碰巧等于上一次的残留。"""
    return torch.full([M, N], float("nan"), device=device, dtype=torch.float32)


@pytest.fixture(scope="session")
def first_run(device, dev_inputs) -> torch.Tensor:
    """出货主形态（preload=0）跑一次，全部数值 / 尾块定位测试共用。"""
    out = _nan_out(device)
    _launch(rmsnorm_gemm_tail, *dev_inputs, out)
    return out.cpu()


# --------------------------------------------------------------------------------------
# 失败信息：直接指认尾块
# --------------------------------------------------------------------------------------
def _assert_rows(out, golden, r0, r1, what):
    assert_rows(out, golden, r0, r1, what, plan=PLAN, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------------------
# 0. 行切分模型（无设备）：与 kernel.py 的标量算式同形，证明覆盖与不越界
# --------------------------------------------------------------------------------------
def _plan():
    for core in range(NUM_CORES):
        for band in range(NUM_BANDS_PER_CORE):
            band_off = band * BAND_ROWS
            band_rows = min(BAND_ROWS, ROWS_PER_CORE - band_off)
            for aiv in (0, 1):
                off = aiv * HALF_BAND
                rows = min(HALF_BAND, band_rows - off)
                yield (
                    core,
                    band,
                    aiv,
                    core * ROWS_PER_CORE + band_off + off,
                    rows,
                    off,
                    band_rows,
                )


def test_row_plan_partitions_rows_exactly_and_stays_in_bounds():
    """每个 (核, 带, AIV) 的 X 读窗口 / O 写窗口：不空、在 [0, M) 内、恰好不重不漏地铺满。"""
    seen = torch.zeros(M, dtype=torch.int32)
    for core, band, aiv, row0, rows, off, band_rows in _plan():
        assert rows >= 1, (
            f"核{core} 带{band} AIV{aiv} 行数 {rows} < 1（0 行 DMA/VF 循环未验过）"
        )
        assert 0 <= row0 and row0 + rows <= M, (
            f"核{core} 带{band} AIV{aiv} 窗口 [{row0},{row0 + rows}) 越出 [0,{M})"
        )
        assert off % 16 == 0, (
            f"AIV{aiv} 的 NZ insert 行偏移 {off} 不是 16 行分形的整数倍"
        )
        seen[slice(row0, row0 + rows)] += 1
    assert bool((seen == 1).all()), (
        f"行覆盖不是恰好一次：漏 {int((seen == 0).sum())} 行，重复 {int((seen > 1).sum())} 行"
    )
    assert TAIL_BAND_ROWS == ROWS_PER_CORE - (NUM_BANDS_PER_CORE - 1) * BAND_ROWS == 61
    # 最后一个窗口恰好收在第 M 行：不越界也没有多余的一行
    last = max(row0 + rows for _, _, _, row0, rows, _, _ in _plan())
    assert last == M


# --------------------------------------------------------------------------------------
# 1. 数值
# --------------------------------------------------------------------------------------
def test_matches_reference(first_run, golden):
    _assert_rows(first_run, golden, 0, M, "全部 1000 行")


def test_error_is_only_fp16_rounding_flips(first_run, golden, host_inputs):
    """更尖的判据：每一个超过噪声底的行，都能被「至多 3 个 P 元素翻到相邻 FP16」解释掉。

    结构性错误（尾块的行错位 / 漏写 / 旧带残留）解释不了；门槛 2e-2 抓不到的小扰动它抓得到。
    """
    x, gamma, w = host_inputs
    bad = unexplained_rows(first_run, golden, x, gamma, w, noise=NOISE)
    if bad:
        rows = [b["row"] for b in bad]
        first = owner(rows[0], **PLAN)
        raise AssertionError(
            f"{len(bad)} 行的误差不能用 FP16 舍入边界翻转解释（结构性错误）：行 {rows[:12]}，"
            f"首个属于核 {first['core']} / 带 {first['band']} / AIV{first['aiv']}，"
            f"其残差 {bad[0]['residual']:.3e}（噪声阈值 {NOISE:g}）"
        )


def test_preload0_and_preload2_agree_bit_for_bit(device, dev_inputs, first_run):
    """1+1 拓扑下 preload 是死参数；这条测跨核同步本身对不对（同步不足 = 数值错误或不稳定）。"""
    out = _nan_out(device)
    _launch(rmsnorm_gemm_tail_p2, *dev_inputs, out)
    assert torch.equal(out.cpu().view(torch.int32), first_run.view(torch.int32))


def test_repeat_runs_agree_bit_for_bit(device, dev_inputs, first_run):
    for i in range(9):
        out = _nan_out(device)
        _launch(rmsnorm_gemm_tail, *dev_inputs, out)
        assert torch.equal(out.cpu().view(torch.int32), first_run.view(torch.int32)), (
            f"第 {i + 2} 次运行与首次不逐位一致"
        )


# --------------------------------------------------------------------------------------
# 2. 尾块定位：失败时能直接指认是尾块处理错，而不是整体数值错
# --------------------------------------------------------------------------------------
def test_full_bands_match_reference(first_run, golden):
    """所有整带（每核的带 0）：与尾块无关的对照。它绿而下面的尾块测试红 = 只有尾块处理错。"""
    for core in range(NUM_CORES):
        r0 = core * ROWS_PER_CORE
        _assert_rows(
            first_run,
            golden,
            r0,
            r0 + BAND_ROWS,
            f"核 {core} 的整带（带 0，{BAND_ROWS} 行）",
        )


def test_tail_last_core_last_band(first_run, golden):
    """最后一个核的最后一带：X 读窗口与 O 写窗口都恰好收在第 1000 行的那一块。"""
    _assert_rows(
        first_run,
        golden,
        TAIL_BAND_START,
        M,
        f"最后一核 {LAST_CORE} 的尾带 [{TAIL_BAND_START},{M})",
    )


@pytest.mark.parametrize("core", range(NUM_CORES), ids=lambda c: f"core{c}")
def test_tail_band_of_every_core(first_run, golden, core):
    """每个核的尾带（61 行）：尾块逻辑在每个核上都跑（均分策略下不是只有最后一个核有尾）。"""
    r0 = core * ROWS_PER_CORE + (NUM_BANDS_PER_CORE - 1) * BAND_ROWS
    _assert_rows(
        first_run, golden, r0, core * ROWS_PER_CORE + ROWS_PER_CORE, f"核 {core} 的尾带"
    )


SEAM_ROWS = BAND_ROWS - TAIL_BAND_ROWS  # 3：尾带若按整带 store，会越过自己份额的行数


@pytest.mark.parametrize(
    "core", range(1, NUM_CORES), ids=lambda c: f"core{c - 1}_tail_to_core{c}_head"
)
def test_core_seam_not_overwritten_by_previous_cores_tail(first_run, golden, core):
    """核 c 的头 ``SEAM_ROWS`` 行。上一核 c-1 的尾带若 store 了整带（gemm 没收窄 ``m=`` / acc 有效行偏大），
    会越过自己的 125 行份额、把这里盖掉——症状出现在**下一个核的整带**里，单看「哪行错」会指错人。
    这条单独验交界，失败信息里写明嫌疑是上一核的尾带 store。

    **它只是「症状可见时的指认」，不是越界写的判据**：数值症状取决于相邻两个核谁后写，实测同一个缺陷变体有时盖坏
    18~21 行、有时 0 行（时序相关）。越界写的确定性判据是守卫行与 sanitizer 两条测试。
    """
    r0 = core * ROWS_PER_CORE
    _assert_rows(
        first_run,
        golden,
        r0,
        r0 + SEAM_ROWS,
        f"核 {core - 1} 尾带 与 核 {core} 头部的交界；嫌疑：核 {core - 1} 尾带的 store 越过了自己的份额",
    )


def test_tail_last_band_split_by_aiv(first_run, golden):
    """尾带内两个 AIV 的份额分开验：AIV1 只有 29 行（不满半带），出错时能指认是「不满半带」。"""
    aiv1_start = TAIL_BAND_START + HALF_BAND
    _assert_rows(
        first_run,
        golden,
        TAIL_BAND_START,
        aiv1_start,
        f"最后一核尾带 AIV0 的满半带 [{TAIL_BAND_START},{aiv1_start})",
    )
    _assert_rows(
        first_run,
        golden,
        aiv1_start,
        M,
        f"最后一核尾带 AIV1 的不满半带（{M - aiv1_start} 行）[{aiv1_start},{M})",
    )


@pytest.mark.parametrize(
    "row,what",
    [
        pytest.param(
            TAIL_BAND_START - 1,
            "尾带前一行（整带最后一行）",
            id="row938_last_of_full_band",
        ),
        pytest.param(TAIL_BAND_START, "尾带第一行", id="row939_first_of_tail_band"),
        pytest.param(
            TAIL_BAND_START + HALF_BAND - 1, "尾带 AIV0 最后一行", id="row970_aiv0_last"
        ),
        pytest.param(
            TAIL_BAND_START + HALF_BAND,
            "尾带 AIV1 第一行（半带交界）",
            id="row971_aiv1_first",
        ),
        pytest.param(M - 3, "倒数第 3 行", id="row997"),
        pytest.param(M - 2, "倒数第 2 行", id="row998"),
        pytest.param(M - 1, "最后一行（第 999 行）", id="row999_last"),
    ],
)
def test_boundary_rows(first_run, golden, row, what):
    """单行：边界行最容易差一。整段比较里这些行淹没在 1000 行中，单独验才能直接指认。"""
    _assert_rows(first_run, golden, row, row + 1, what)


def test_locating_messages_name_the_right_owner(golden):
    """判据的自检（无设备）：往 golden 的副本里注入单行错，失败信息必须指认到正确的核 / 带 / AIV。"""
    cases = {
        TAIL_BAND_START: (7, 1, 0),
        TAIL_BAND_START + HALF_BAND: (7, 1, 1),
        M - 1: (7, 1, 1),
        ROWS_PER_CORE + 10: (1, 0, 0),
    }
    for row, (core, band, aiv) in cases.items():
        out = golden.clone()
        out[row] += 1.0
        with pytest.raises(AssertionError) as info:
            _assert_rows(out, golden, 0, M, "自检")
        msg = str(info.value)
        assert (
            f"首个坏行 {row}" in msg
            and f"核 {core} / 带 {band}" in msg
            and f"AIV{aiv}" in msg
        ), msg
        assert ("尾带" in msg) == (band == NUM_BANDS_PER_CORE - 1), msg


# --------------------------------------------------------------------------------------
# 3. 不越界读写 GM
# --------------------------------------------------------------------------------------
def test_no_write_past_end_of_output_and_every_row_written(
    device, dev_inputs, first_run
):
    """O 放进 ``[M + GUARD, N]`` 的缓冲，传前 M 行的视图：守卫行必须逐位保持哨兵；前 M 行没有一行
    还是哨兵（每行都被写到）。越界写 / 漏写数值对照看不见，这条看得见。
    """
    obuf = torch.full(
        [M + GUARD_ROWS, N], SENTINEL_BITS, device=device, dtype=torch.int32
    ).view(torch.float32)
    _launch(rmsnorm_gemm_tail, *dev_inputs, obuf[:M])
    bits = obuf.cpu().view(torch.int32)
    dirty = (bits[M:] != SENTINEL_BITS).any(dim=1).nonzero().flatten() + M
    assert dirty.numel() == 0, (
        f"越界写：守卫行 {dirty[:8].tolist()} 被写脏（O 只有 {M} 行）"
    )
    unwritten = (bits[:M] == SENTINEL_BITS).all(dim=1).nonzero().flatten()
    assert unwritten.numel() == 0, f"漏写：行 {unwritten[:8].tolist()} 仍是哨兵"
    assert torch.equal(bits[:M], first_run.view(torch.int32)), (
        "缓冲视图运行与首次运行不逐位一致"
    )


def test_poisoned_rows_past_end_of_input_do_not_leak(device, dev_inputs, first_run):
    """X 放进 ``[M + GUARD, K]`` 的缓冲，第 M 行起全是 NaN：结果必须与干净输入逐位一致。
    只能抓到「越界行被算进了有效行」；纯读越界（读了又丢）由 sanitizer 那条负责。
    """
    x, gamma, w = dev_inputs
    xbuf = torch.full(
        [M + GUARD_ROWS, K], float("nan"), device=x.device, dtype=torch.float16
    )
    xbuf[:M] = x
    out = _nan_out(x.device)
    _launch(rmsnorm_gemm_tail, xbuf[:M], gamma, w, out)
    assert torch.equal(out.cpu().view(torch.int32), first_run.view(torch.int32))


def test_sanitizer_reports_no_gm_or_tile_out_of_bounds(device, dev_inputs, first_run):
    """插桩变体（``sanitizer=True``）：每次 GM 访问的 ``偏移 + 有效窗口`` 对着张量逻辑 shape 检查，
    另查 tile 越界 / 重叠 / mutex 配对。有检出会在 launch 后抛 ``SanitizerReplayError``。
    另验：插桩不改变结果（逐位）。
    """
    out = _nan_out(device)
    _launch(rmsnorm_gemm_tail_san, *dev_inputs, out)
    assert torch.equal(out.cpu().view(torch.int32), first_run.view(torch.int32))
