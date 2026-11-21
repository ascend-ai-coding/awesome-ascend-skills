# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""W8A16 checks for the native kernel, its reference layer and coordinate mapping.

The numerical gate is abs(got-fixpipe_rne_reference) <= TOL_S*S, where S=sum(abs(X)*abs(W_q))*abs(scale) and
TOL_S=K*2^-24. The mathematical-reference check separately allows SCALE_REL*abs(exact)+TOL_S*S with SCALE_REL=2^-11.
Integer/power-of-two patterns require exact equality, including zeros outside isolated row/column/K regions.

CPU checks compare scale rounding with independent frexp/rint and sample the FP64 result against rational arithmetic.
Device checks cover preload=0/2, repeated and fresh-input launches, signed/zero scales, INT8 endpoints,
core/strip/K-half/split-K boundaries and a one-hot permutation. Four wrong-layer references must fail the own-layer
gate. The underlaunch case verifies intentionally unwritten rows; it is not full-output acceptance.
"""

import ast
import importlib.util
import os
import pathlib
from fractions import Fraction

import numpy as np
import pytest
import torch

pytest.importorskip("pypto_pro")
pytest.importorskip("torch_npu")

from . import reference as ref  # noqa: E402

HERE = pathlib.Path(__file__).parent


def _load_kernel_module():
    """Load kernel.py, or a QGEMM_KERNEL override for explicit negative controls."""
    override = os.environ.get("QGEMM_KERNEL")
    path = (HERE / override) if override else HERE / "kernel.py"
    spec = importlib.util.spec_from_file_location("kernel_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


kmod, KERNEL_SRC = _load_kernel_module()

M, K, N = kmod.M, kmod.K, kmod.N
ROWS, STRIP, HALF, CHUNK = kmod.ROWS_PER_CORE, kmod.STRIP, kmod.K_HALF, kmod.SK
NUM_CORES = kmod.NUM_CORES
TOL_S = K * 2.0**-24
SCALE_REL = 2.0**-11


# ---------------------------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------------------------


def launch(device, kern, x, w_q, scale, groups=kmod.LAUNCH_BLOCK_DIM):
    """预填 NaN 的输出，启动，同步，返回 CPU FP32（NaN 残留 = 有元素没被写）。"""
    out = torch.full([M, N], float("nan"), dtype=torch.float32, device=device)
    kern[None, groups](x.to(device), w_q.to(device), scale.to(device), out)
    torch.npu.synchronize()
    return out.cpu()


def fixpipe_ref(x, w_q, scale):
    return ref.fixpipe_rne_reference(x, w_q, scale)


def within_tol(got, want, x, w_q, scale):
    """逐元素 ``|got - want| <= TOL_S * S``（S = 0 的位置要求逐位相等）。返回 (是否全过, 最大 err/S)。"""
    diff = (got.to(torch.float64) - want).abs()
    s = ref.error_scale(x, w_q, scale)
    ok = bool(torch.all(diff <= TOL_S * s))
    ratio = torch.where(s > 0, diff / s.clamp_min(1e-300), diff).max().item()
    return ok, ratio


def exact_inputs(seed=0):
    """所有中间量精确的输入：X 是 [-8, 8] 的整数（FP16 精确），W_q 取满 [-128, 127]（含 -128），scale = 2 的幂。

    输出 = 整数 x 2^e，绝对值 < 2^18 个 2^-8 单位——FP32 精确，与求和顺序无关；float64 参考同样精确。
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-8, 9, [M, K], generator=g).to(torch.float16)
    w_q = torch.randint(-128, 128, [K, N], generator=g, dtype=torch.int64).to(
        torch.int8
    )
    e = torch.randint(-8, 1, [N], generator=g)
    scale = torch.pow(2.0, e.to(torch.float32))
    return x, w_q, scale


def assert_bitwise_exact(got, x, w_q, scale, what):
    want = ref.exact_reference(x, w_q, scale).to(torch.float32)
    assert not torch.isnan(got).any(), f"{what}: 输出有 NaN（未写元素）"
    bad = got != want
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} 个元素与精确参考不逐位相等；首个不等位置 "
        f"{tuple(torch.nonzero(bad)[0].tolist())}，got {got[bad][0].item()} want {want[bad][0].item()}"
    )


# ---------------------------------------------------------------------------------------------
# CPU 测试：参考本身、kernel 源的原生性
# ---------------------------------------------------------------------------------------------


def test_rne19_matches_independent_frexp_implementation():
    """reference.round_scale_fp19_rne（整数加法实现）对照独立的 frexp / rint 实现（含平局与进位）。"""
    g = np.random.default_rng(3)
    s = np.concatenate(
        [
            g.uniform(1e-6, 100.0, 200_000),
            -g.uniform(1e-6, 100.0, 1_000),
            [0.0, 1.0, 2.0, 0.5, 3.0e-4],
        ]
    ).astype(np.float32)
    # 显式平局：低 13 位恰为 0x1000（半个 ulp），奇 / 偶两种下一位；以及尾数全 1 会进位的情形
    base = (
        np.uint32(0x3F800000) + (np.arange(64, dtype=np.uint32) << np.uint32(13))
    ) | np.uint32(0x1000)
    ties = base.view(np.float32)
    all_s = np.concatenate(
        [s, ties, np.array([np.float32(np.nextafter(np.float32(2.0), np.float32(0)))])]
    )
    got = ref.round_scale_fp19_rne(torch.from_numpy(all_s)).numpy()
    m, e = np.frexp(
        all_s.astype(np.float64)
    )  # m in [0.5, 1)（含符号），11 位有效数字 = 隐含位 + 10 位显式
    want = np.ldexp(np.rint(m * 2048.0) / 2048.0, e).astype(np.float32)
    assert np.array_equal(got, want), f"{int((got != want).sum())} 个值与独立实现不同"
    nz = all_s != 0
    a, r = all_s[nz].astype(np.float64), got[nz].astype(np.float64)
    assert np.all(np.abs(r - a) <= SCALE_REL * np.abs(a))


def test_float64_reference_error_bound_against_exact_rational_arithmetic():
    """float64 参考与精确有理数算术的差 <= (K-1) * 2^-53 * S（reference.py 的误差口径）——抽样 300 个元素。"""
    x, w_q, scale = ref.make_inputs(5, M, K, N)
    got = ref.exact_reference(x, w_q, scale)
    s = ref.error_scale(x, w_q, scale)
    g = np.random.default_rng(1)
    for _ in range(300):
        m, n = int(g.integers(0, M)), int(g.integers(0, N))
        exact = sum(
            Fraction(float(x[m, k])) * int(w_q[k, n]) * Fraction(float(scale[n]))
            for k in range(K)
        )
        err = abs(Fraction(float(got[m, n])) - exact)
        assert err <= Fraction((K - 1) * 2.0**-53) * Fraction(float(s[m, n])), (m, n)


def test_kernel_module_is_a_plain_pypto_kernel():
    """kernel.py 只 import pypto_pro（去掉 agent 系统后它仍是普通 PyPTO kernel；[skill
    constraints](../../../SKILL.md#constraints-and-workspace-policy) 原生优先）。
    """
    tree = ast.parse(KERNEL_SRC.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots == {"pypto_pro"}, roots


# ---------------------------------------------------------------------------------------------
# 板上：数值
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def case0(device):
    x, w_q, scale = ref.make_inputs(0, M, K, N)
    return (
        x,
        w_q,
        scale,
        launch(device, kmod.quant_gemm_w8a16, x, w_q, scale),
        launch(device, kmod.quant_gemm_w8a16_p2, x, w_q, scale),
    )


def test_full_numeric_against_own_layer_reference(case0):
    x, w_q, scale, out, _ = case0
    assert not torch.isnan(out).any()
    ok, ratio = within_tol(out, fixpipe_ref(x, w_q, scale), x, w_q, scale)
    print(f"\nerr/S vs fixpipe_rne reference: max {ratio:.3e} (TOL_S {TOL_S:.3e})")
    assert ok, f"max err/S {ratio:.3e} > {TOL_S:.3e}"


def test_against_math_definition_a_priori_bound(case0):
    """相对数学定义（FP32 scale，不量化任何中间量）：``|out - exact| <= 2^-11 * |exact| + TOL_S * S``，逐元素。"""
    x, w_q, scale, out, _ = case0
    exact = ref.exact_reference(x, w_q, scale)
    diff = (out.to(torch.float64) - exact).abs()
    bound = SCALE_REL * exact.abs() + TOL_S * ref.error_scale(x, w_q, scale)
    mask = exact.abs() > 1.0
    rel = (diff / exact.abs())[mask]
    print(
        f"\nvs exact math: max rel err (|O|>1) {rel.max().item():.3e}, median {rel.median().item():.3e}; "
        f"max err/S {(diff / ref.error_scale(x, w_q, scale)).max().item():.3e}"
    )
    assert bool(torch.all(diff <= bound)), (
        f"{int((diff > bound).sum())} 个元素超出先验上界"
    )


def test_preload_tiers_bitwise_identical(case0):
    _, _, _, out0, out2 = case0
    assert torch.equal(out0, out2), "preload=0 与 preload=2 输出不逐位相同"


def test_repeat_runs_bitwise_identical(device, case0):
    x, w_q, scale, out0, _ = case0
    for i in range(3):
        again = launch(device, kmod.quant_gemm_w8a16, x, w_q, scale)
        assert torch.equal(out0, again), f"第 {i + 2} 次 launch 与第 1 次不逐位相同"


def test_wrong_layer_references_are_rejected(case0):
    """负对照：其余各层的参考必须**通不过**同一个判据——证明判据分得清反量化层（放宽它们才会变绿）。"""
    x, w_q, scale, out, _ = case0
    verdicts = {}
    for name, fn in (
        ("exact_math", ref.exact_reference),
        ("fixpipe_truncate", ref.fixpipe_reference),
        ("prescale_fp16_wdq", ref.prescale_reference),
        ("unscaled", ref.unscaled_reference),
    ):
        ok, ratio = within_tol(out, fn(x, w_q, scale), x, w_q, scale)
        verdicts[name] = (ok, ratio)
    print(
        "\nwrong-layer references, max err/S:",
        {k: f"{v[1]:.2e}" for k, v in verdicts.items()},
    )
    survivors = [k for k, (ok, _) in verdicts.items() if ok]
    assert not survivors, f"这些错误层的参考也通过了判据（判据分不清层）：{survivors}"


def test_fresh_inputs_every_launch_no_stale_payload(device):
    """同一进程连续 launch，X / W_q / scale 每次全换：载荷只能来自本次 launch（陈旧载荷 = 上一次的 scale，会立刻爆）。"""
    worst = 0.0
    for seed in range(20, 26):
        x, w_q, scale = ref.make_inputs(seed, M, K, N)
        out = launch(device, kmod.quant_gemm_w8a16, x, w_q, scale)
        ok, ratio = within_tol(out, fixpipe_ref(x, w_q, scale), x, w_q, scale)
        worst = max(worst, ratio)
        assert ok and not torch.isnan(out).any(), f"seed {seed}: err/S {ratio:.3e}"
    print(f"\nworst err/S over 6 fresh launches: {worst:.3e}")


def _scale_dists():
    g = torch.Generator().manual_seed(7)
    return {
        "log_uniform_1e-4_to_1": torch.exp(torch.rand([N], generator=g) * np.log(1e4))
        * 1e-4,
        "all_equal": torch.full([N], 0.0123),
        "power_of_two": torch.pow(
            2.0, torch.randint(-10, 3, [N], generator=g).to(torch.float32)
        ),
        "mixed_sign_with_zeros": torch.where(
            torch.rand([N], generator=g) < 0.1,
            torch.zeros([N]),
            (torch.rand([N], generator=g) - 0.5) * 0.1,
        ),
        "large_magnitude": torch.rand([N], generator=g) * 1000.0 + 1.0,
    }


@pytest.mark.parametrize("dist", list(_scale_dists()))
def test_scale_distributions(device, dist):
    x, w_q, _ = ref.make_inputs(30, M, K, N)
    scale = _scale_dists()[dist].to(torch.float32)
    out = launch(device, kmod.quant_gemm_w8a16, x, w_q, scale)
    ok, ratio = within_tol(out, fixpipe_ref(x, w_q, scale), x, w_q, scale)
    assert not torch.isnan(out).any()
    assert ok, f"{dist}: max err/S {ratio:.3e} > {TOL_S:.3e}"
    exact = ref.exact_reference(x, w_q, scale)
    diff = (out.to(torch.float64) - exact).abs()
    assert bool(
        torch.all(
            diff <= SCALE_REL * exact.abs() + TOL_S * ref.error_scale(x, w_q, scale)
        )
    ), f"{dist}: 超出相对数学定义的先验上界"
    if dist == "mixed_sign_with_zeros":
        zero_cols = scale == 0
        assert bool(torch.all(out[:, zero_cols] == 0)), "scale = 0 的列必须输出 0"


@pytest.mark.parametrize("wq_fill", [-128, 127, "alternating"])
def test_int8_extremes(device, wq_fill):
    x, _, scale = ref.make_inputs(31, M, K, N)
    if wq_fill == "alternating":
        w_q = (
            torch.where(torch.arange(K)[:, None] % 2 == 0, -128, 127)
            .expand(K, N)
            .to(torch.int8)
            .contiguous()
        )
    else:
        w_q = torch.full([K, N], wq_fill, dtype=torch.int8)
    out = launch(device, kmod.quant_gemm_w8a16, x, w_q, scale)
    ok, ratio = within_tol(out, fixpipe_ref(x, w_q, scale), x, w_q, scale)
    assert not torch.isnan(out).any()
    assert ok, f"W_q = {wq_fill}: max err/S {ratio:.3e}"


# ---------------------------------------------------------------------------------------------
# 板上：结构化边界定位（exact_inputs：逐位相等，无容差）
# ---------------------------------------------------------------------------------------------


def test_exact_inputs_full_bitwise(device):
    x, w_q, scale = exact_inputs(0)
    assert_bitwise_exact(
        launch(device, kmod.quant_gemm_w8a16, x, w_q, scale),
        x,
        w_q,
        scale,
        "exact_inputs 全量",
    )


@pytest.mark.parametrize("core", range(NUM_CORES))
def test_core_row_block_isolation(device, core):
    """X 只在第 core 个核的行块非零：别的行块必须严格为 0，本块逐位精确。定位「行块 <-> 核」映射与越界写。"""
    x, w_q, scale = exact_inputs(1)
    keep = torch.zeros(M, dtype=torch.bool)
    keep[slice(core * ROWS, (core + 1) * ROWS)] = True
    x = torch.where(keep[:, None], x, torch.zeros_like(x))
    got = launch(device, kmod.quant_gemm_w8a16, x, w_q, scale)
    assert_bitwise_exact(got, x, w_q, scale, f"core {core} 行块")
    assert bool(torch.all(got[~keep] == 0))


@pytest.mark.parametrize("row", [0, ROWS - 1, ROWS, 4 * ROWS - 1, 4 * ROWS, M - 1])
def test_single_row_at_core_boundaries(device, row):
    """X 只有一行非零，取核边界两侧的行（127 / 128、511 / 512、0、1023）：只有那一行有输出。"""
    x, w_q, scale = exact_inputs(2)
    x = torch.zeros_like(x).index_copy_(0, torch.tensor([row]), x[slice(row, row + 1)])
    got = launch(device, kmod.quant_gemm_w8a16, x, w_q, scale)
    assert_bitwise_exact(got, x, w_q, scale, f"单行 {row}")
    assert int((got != 0).any(dim=1).sum()) <= 1


@pytest.mark.parametrize("strip", range(kmod.NUM_STRIPS))
def test_scale_strip_isolation(device, strip):
    """Isolate one scale strip: the other strip must be zero, distinguishing the two payload slots."""
    x, w_q, scale = exact_inputs(3)
    keep = torch.zeros(N, dtype=torch.bool)
    keep[slice(strip * STRIP, (strip + 1) * STRIP)] = True
    scale = torch.where(keep, scale, torch.zeros_like(scale))
    got = launch(device, kmod.quant_gemm_w8a16, x, w_q, scale)
    assert_bitwise_exact(got, x, w_q, scale, f"scale 只在 strip {strip}")
    assert bool(torch.all(got[:, ~keep] == 0))


@pytest.mark.parametrize("strip", range(kmod.NUM_STRIPS))
def test_wq_strip_isolation(device, strip):
    x, w_q, scale = exact_inputs(4)
    keep = torch.zeros(N, dtype=torch.bool)
    keep[slice(strip * STRIP, (strip + 1) * STRIP)] = True
    w_q = torch.where(keep[None, :], w_q, torch.zeros_like(w_q))
    got = launch(device, kmod.quant_gemm_w8a16, x, w_q, scale)
    assert_bitwise_exact(got, x, w_q, scale, f"W_q 只在 strip {strip}")
    assert bool(torch.all(got[:, ~keep] == 0))


@pytest.mark.parametrize("half", range(2))
def test_wq_k_half_isolation(device, half):
    """W_q 只在一半 K 行非零：定位「AIV 下标 <-> K 行半」映射与 NZ 散写（每个 AIV 只写自己那一半）。"""
    x, w_q, scale = exact_inputs(5)
    keep = torch.zeros(K, dtype=torch.bool)
    keep[slice(half * HALF, (half + 1) * HALF)] = True
    w_q = torch.where(keep[:, None], w_q, torch.zeros_like(w_q))
    assert_bitwise_exact(
        launch(device, kmod.quant_gemm_w8a16, x, w_q, scale),
        x,
        w_q,
        scale,
        f"W_q 只在 K 半 {half}",
    )


@pytest.mark.parametrize("chunk", range(K // CHUNK))
def test_wq_k_chunk_isolation(device, chunk):
    """W_q 只在一个 splitk 步长（64 行）内非零：定位 gemm 的 K 分步与 L1 -> L0B 窗口偏移。"""
    x, w_q, scale = exact_inputs(6)
    keep = torch.zeros(K, dtype=torch.bool)
    keep[slice(chunk * CHUNK, (chunk + 1) * CHUNK)] = True
    w_q = torch.where(keep[:, None], w_q, torch.zeros_like(w_q))
    assert_bitwise_exact(
        launch(device, kmod.quant_gemm_w8a16, x, w_q, scale),
        x,
        w_q,
        scale,
        f"W_q 只在 K 步 {chunk}",
    )


def test_one_hot_permutation_places_every_k_row_and_column(device):
    """每列只有一个非零 W_q（行 = (37 n + 11) mod 256，是 K 上的一个置换）：输出列 n = X[:, k(n)] * q_n * scale_n。

    逐个 (k, n) 位置校验 NZ 散写与 splitk 窗口——任何一行或一列错位都会让对应输出列变成别的 X 列。
    """
    x, _, scale = exact_inputs(7)
    g = torch.Generator().manual_seed(9)
    q = torch.randint(-128, 128, [N], generator=g, dtype=torch.int64)
    q[q == 0] = 1
    k_of_n = (37 * torch.arange(N) + 11) % K
    assert len(set(k_of_n.tolist())) == K
    w_q = torch.zeros([K, N], dtype=torch.int8)
    w_q[k_of_n, torch.arange(N)] = q.to(torch.int8)
    assert_bitwise_exact(
        launch(device, kmod.quant_gemm_w8a16, x, w_q, scale),
        x,
        w_q,
        scale,
        "one-hot 置换",
    )


def test_fewer_groups_leave_trailing_row_blocks_unwritten(device):
    """启动语义（kernel.py docstring）：方括号里是 execution group 数。少于 8 时尾部行块不被写、前面的行块照常正确。"""
    x, w_q, scale = ref.make_inputs(8, M, K, N)
    got = launch(device, kmod.quant_gemm_w8a16, x, w_q, scale, groups=4)
    head = 4 * ROWS
    ok, ratio = within_tol(
        got[:head], fixpipe_ref(x, w_q, scale)[:head], x[:head], w_q, scale
    )
    assert ok, f"前 4 个行块 err/S {ratio:.3e}"
    assert bool(torch.isnan(got[head:]).all()), (
        "少于 8 个 group 时，尾部行块应保持预填值（NaN）"
    )
