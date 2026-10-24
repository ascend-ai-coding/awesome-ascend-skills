# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fixpipe quantization across Acc→Vec, Acc→Mat and Acc→GM.

The Vec matrix tests ten accepted dtype combinations with scalar and per-channel scales. Scaled FP32→FP32 has dedicated
parse-only rejection tests for both encodings and three scale values; it is not skipped or launched. GM tests all eleven
combinations in ND/NZ layouts. Mat tests seven q@q numerical readbacks, two UINT8 operand rejections and one HF8
execution-only case.

Small integer inputs and power-of-two scales preserve the existing rtol=atol=0 comparisons for exact/rounded cases and
decoded FP8. ND uses coordinate comparisons; NZ GM output uses the existing value-multiset criterion, not a proof of
every physical address. FP8's parameterized checks cover scalar, per-channel and GM outputs through one comparison
helper.

HF8 byte-carrier and Mat checks establish nontrivial execution only. A separate VF conversion compares 64 stride-4
values with HF8_MAX_REL=0.125; both hardware units interpret the same bytes, so this is cross-unit consistency, not an
independent format oracle. The dtype/layout domain is fixed by the case tables.

Additional controls cover scale-free routes, integer saturation and unsigned clamping, per-channel bit46, FP8
decoder/quantizer consistency, fp19 scale truncation, signed nine-bit offsets, and explicit quant_offset versus packed
runtime payloads. Source hashes and actual toolchain identity are captured outside the repository by the source-gated
runner.
"""

import numpy as np
import pytest
import torch
from pypto_pro._errors import NotSupported

from . import reference as ref
from ..fixpipe_scaling_buffer.reference import encode_scales
from .kernel import (
    K,
    M,
    N,
    SCALE,
    make_per_channel_move,
    make_quant_move,
    make_quant_move_to_mat,
    make_move_with_quant_offset,
    make_quant_store,
    hf8_move_and_decode,
    store_with_runtime_payload_int8,
    store_with_runtime_scale,
)

_DTYPE_OF = {
    "int8": torch.int8,
    "uint8": torch.uint8,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int32": torch.int32,
}

_PL = {}


def _pl_dtype(name):
    """Resolve the framework dtype by its public name."""
    if not _PL:
        import pypto_pro.language as pl

        for n in ("FP32", "INT32", "INT8", "UINT8", "HF8", "FP16", "BF16", "FP8E4M3FN"):
            _PL[n] = getattr(pl, f"DT_{n}")
    return _PL[name]


def _run(src, dst, carrier, device, *, with_scale=True, scale=SCALE):
    a, b = ref.make_inputs(M, K, N, src)
    out = torch.zeros((M, N), dtype=_DTYPE_OF[carrier], device=device)
    kernel = make_quant_move(
        _pl_dtype(src), _pl_dtype(dst), with_scale=with_scale, scale=scale
    )
    kernel(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device), out)
    torch.npu.synchronize()
    return a, b, out.cpu()


_MOVE_CELLS = [row for row in ref.MATRIX if row[3] != "rejected"]
_IDS = [f"{s}-{d}" for s, d, _, _ in _MOVE_CELLS]


# ---------------------------------------------------------------------------
# 矩阵本体
# ---------------------------------------------------------------------------


def _check_outcome(a, b, dst, got, outcome, scale, label):
    """Apply the case-specific exact, saturated or byte-carrier check; FP8 is decoded before numerical comparison."""
    if outcome == "exact":
        np.testing.assert_allclose(
            got.to(torch.float32).numpy(),
            ref.scaled_exact(a, b, scale),
            rtol=0,
            atol=0,
            err_msg=f"{label} 不是逐位的 scale*acc",
        )
    elif outcome == "rounded":
        np.testing.assert_allclose(
            got.to(torch.float32).numpy(),
            ref.scaled_rounded(a, b, scale, dst),
            rtol=0,
            atol=0,
            err_msg=f"{label} 不是就近取偶后饱和的结果",
        )
    elif outcome == "opaque":
        raw = got.to(torch.uint8).numpy().reshape(-1)
        assert (raw != 0).any(), f"{label} 输出全是 0——搬运没生效"
        if dst == "FP8E4M3FN":
            decoded = ref.decode_fp8_e4m3(got.numpy())
            expected = ref.quantize_to_fp8_e4m3(ref.scaled_exact(a, b, scale))
            np.testing.assert_allclose(
                decoded,
                expected,
                rtol=0,
                atol=0,
                err_msg=f"{label} 解码值不等于 FP8 量化参考",
            )
    elif outcome == "rejected":
        pytest.fail(
            f"{label} 的 outcome 是 rejected，不该跑到这里——"
            "该组合由独立解析期拒绝测试覆盖"
        )
    else:
        pytest.fail(f"未知的 outcome：{outcome}")


@pytest.mark.parametrize("src,dst,carrier,outcome", _MOVE_CELLS, ids=_IDS)
def test_each_cell_matches_the_measured_outcome(src, dst, carrier, outcome, device):
    """Each cell matches the measured outcome."""
    a, b, got = _run(src, dst, carrier, device)
    _check_outcome(a, b, dst, got, outcome, SCALE, f"{src}→{dst}")


def _run_per_channel(src, dst, carrier, device, *, scales=None, signed_flag=None):
    """Run a Vec case with an encoded per-column scale table."""
    if scales is None:
        scales = ref.per_channel_scales(N)
    if signed_flag is None:
        signed_flag = ref.SIGNED_FLAG_FOR.get(dst, True)
    a, b = ref.make_inputs(M, K, N, src)
    enc = encode_scales(scales.reshape(-1), signed_int8_flag=signed_flag)
    out = torch.zeros((M, N), dtype=_DTYPE_OF[carrier], device=device)
    kernel = make_per_channel_move(_pl_dtype(src), _pl_dtype(dst))
    kernel(
        torch.from_numpy(a).to(device),
        torch.from_numpy(b).to(device),
        torch.from_numpy(enc).to(device),
        out,
    )
    torch.npu.synchronize()
    return a, b, out.cpu()


@pytest.mark.parametrize("src,dst,carrier,outcome", _MOVE_CELLS, ids=_IDS)
def test_each_cell_behaves_the_same_with_a_per_channel_table(
    src, dst, carrier, outcome, device
):
    """Each cell behaves the same with a per channel table."""
    a, b, got = _run_per_channel(src, dst, carrier, device)
    _check_outcome(
        a,
        b,
        dst,
        got,
        outcome,
        ref.per_channel_scales(N),
        f"{src}→{dst}（per-channel）",
    )


def test_bit46_beats_the_destination_dtype(device):
    """Bit46 beats the destination dtype."""
    scales = ref.per_channel_scales(N)
    a, b, got = _run_per_channel("FP32", "UINT8", "uint8", device, signed_flag=True)

    signed_then_read_as_uint8 = ref.scaled_rounded(a, b, scales, "INT8") % 256
    np.testing.assert_allclose(
        got.to(torch.float32).numpy(),
        signed_then_read_as_uint8,
        rtol=0,
        atol=0,
        err_msg="UINT8 目的 + bit46 的输出不是「按 int8 饱和后当 uint8 读」",
    )

    unsigned = ref.scaled_rounded(a, b, scales, "UINT8")
    assert not np.array_equal(got.to(torch.float32).numpy(), unsigned), (
        "置位与不置位的结果相同——这条验空了，说明 bit46 对 UINT8 目的没作用"
    )


def test_the_fp8_encoder_is_built_from_the_decoder(device):
    """The fp8 encoder is built from the decoder."""
    values = ref.fp8_e4m3_values()
    assert len(values) == 253, f"E4M3FN 的有限值个数应当是 253，实际 {len(values)}"
    assert values[0] == -448.0 and values[-1] == 448.0, (
        f"E4M3FN 的范围应当是 ±448，实际 [{values[0]}, {values[-1]}]"
    )
    np.testing.assert_array_equal(
        ref.quantize_to_fp8_e4m3(values), values, err_msg="编码器对可表示值不是恒等映射"
    )


# ---------------------------------------------------------------------------
# scale 自己的精度：它不是 float32
# ---------------------------------------------------------------------------


def _scale_bits(value):
    """Encode an FP32 scale as an integer bit pattern."""
    import struct

    return struct.unpack("!I", struct.pack("!f", float(value)))[0]


def _hardware_scale(value, device):
    """Use a unit-input GEMM and FP32 GM output to recover the effective scale."""
    a = torch.ones((M, K), dtype=torch.float16, device=device)
    b = torch.ones((K, N), dtype=torch.float16, device=device)
    out = torch.zeros((M, N), dtype=torch.float32, device=device)
    store_with_runtime_scale(a, b, _scale_bits(value), out)
    torch.npu.synchronize()
    got = out.cpu().numpy().astype(np.float64)
    assert got.min() == got.max(), (
        f"scale={value} 的输出不一致：{got.min()}..{got.max()}"
    )
    return got[0, 0] / float(K)


def test_the_scale_keeps_only_ten_mantissa_bits(device):
    """The scale keeps only ten mantissa bits."""
    for k in range(1, 11):
        value = 1.0 + 2.0**-k
        assert _hardware_scale(value, device) == value, (
            f"scale = 1 + 2^-{k} 应当逐位生效，却被改了"
        )

    for k in range(11, 14):
        value = 1.0 + 2.0**-k
        assert _hardware_scale(value, device) == 1.0, (
            f"scale = 1 + 2^-{k} 应当被截成 1.0（尾数只有 "
            f"{ref.SCALE_MANTISSA_BITS} 位），却没有"
        )


def test_the_scale_truncates_toward_zero(device):
    """The scale truncates toward zero."""
    for value in (1.00146484375, -1.00146484375):
        got = _hardware_scale(value, device)
        want = ref.truncate_scale_to_hardware(value)
        assert got == want, f"scale={value} 硬件给 {got}，向零截断应当是 {want}"
        assert abs(got) < abs(value), (
            f"scale={value} 截断后绝对值没变小（{got}）——那不是向零截断"
        )


@pytest.mark.parametrize("exponent", [-100, -60, -1, 1, 60, 100])
def test_the_scale_exponent_is_at_least_eight_bits(exponent, device):
    """The scale exponent is at least eight bits."""
    value = 2.0**exponent
    assert _hardware_scale(value, device) == value, (
        f"scale = 2^{exponent} 没有准确生效——指数位宽比 8 位窄？"
    )


def test_a_runtime_scalar_scale_agrees_with_a_compile_time_one(device):
    """A runtime scalar scale agrees with a compile time one."""
    runtime = _hardware_scale(SCALE, device)
    assert runtime == SCALE, f"运行期标量 scale 给出 {runtime}，不等于 {SCALE}"

    a = torch.ones((M, K), dtype=torch.float16, device=device)
    b = torch.ones((K, N), dtype=torch.float16, device=device)
    out = torch.zeros((M, N), dtype=torch.float32, device=device)
    import pypto_pro.language as pl

    make_quant_store(_pl_dtype("FP32"), _pl_dtype("FP32"), pl.TensorLayout.ND)(
        a, b, out
    )
    torch.npu.synchronize()
    compile_time = out.cpu().numpy().astype(np.float64)[0, 0] / float(K)
    assert compile_time == runtime, (
        f"编译期 scale 给 {compile_time}，运行期给 {runtime}——两条路不一致"
    )


# ---------------------------------------------------------------------------
# 那个 uint64 里还藏着 offset
# ---------------------------------------------------------------------------


def _run_payload_int8(payload, device, *, sign=1):
    """Read back uniform INT8 output for one packed runtime payload."""
    a = torch.full((M, K), float(sign), dtype=torch.float16, device=device)
    b = torch.ones((K, N), dtype=torch.float16, device=device)
    out = torch.zeros((M, N), dtype=torch.int8, device=device)
    store_with_runtime_payload_int8(a, b, payload, out)
    torch.npu.synchronize()
    got = out.cpu().numpy().astype(np.int64)
    assert got.min() == got.max(), f"输出不一致：{got.min()}..{got.max()}"
    return int(got[0, 0])


@pytest.mark.parametrize(
    "offset,want", [(0, 32), (5, 37), (100, 127), (0x100, -128), (0x1FF, 31)]
)
def test_the_payload_carries_a_nine_bit_signed_offset(offset, want, device):
    """The payload carries a nine bit signed offset."""
    payload = ref.encode_quant_payload(0.5, offset=offset)
    got = _run_payload_int8(payload, device)
    assert got == want, f"offset={offset:#x} 期望 {want}，实测 {got}"
    assert (
        got == ref.quantized_with_offset(np.array([[float(K)]]), 0.5, offset)[0, 0]
    ), "实测与 `quantized_with_offset` 不一致——参考模型该改了"


def test_the_offset_does_nothing_for_a_float_destination(device):
    """The offset does nothing for a float destination."""
    payload = ref.encode_quant_payload(0.5, offset=5)
    a = torch.ones((M, K), dtype=torch.float16, device=device)
    b = torch.ones((K, N), dtype=torch.float16, device=device)
    out = torch.zeros((M, N), dtype=torch.float32, device=device)
    store_with_runtime_scale(a, b, payload, out)
    torch.npu.synchronize()
    got = out.cpu().numpy().astype(np.float64)

    assert got.min() == got.max(), f"输出不一致：{got.min()}..{got.max()}"
    assert got[0, 0] == 0.5 * K, (
        f"FP32 目的下 offset=5 改变了结果（{got[0, 0]}，期望 {0.5 * K}）"
        "——那 offset 就不是只对整数目的生效"
    )


def test_bit46_is_ignored_on_the_scalar_path(device):
    """Bit46 is ignored on the scalar path."""
    without = _run_payload_int8(
        ref.encode_quant_payload(0.5, is_signed=False), device, sign=-1
    )
    with_flag = _run_payload_int8(
        ref.encode_quant_payload(0.5, is_signed=True), device, sign=-1
    )

    assert without == with_flag == -32, (
        f"bit46 置位 {with_flag} / 不置位 {without}，期望两个都是 -32"
    )


def test_the_payload_layout_agrees_with_the_measured_mantissa_width():
    """The payload layout agrees with the measured mantissa width."""
    lowest_kept = 0xFFFFE000 & -0xFFFFE000  # 掩码里最低的那个 1
    dropped = lowest_kept.bit_length() - 1
    assert dropped == 13, f"掩码丢掉的位数算成了 {dropped}"
    assert 23 - dropped == ref.SCALE_MANTISSA_BITS, (
        f"掩码给 {23 - dropped} 位尾数，实测是 {ref.SCALE_MANTISSA_BITS} 位"
    )
    assert ref.encode_quant_payload(0.5) == 0x3F000000, (
        "scale=0.5 的位模式本来就没有低位，编码器不该改它"
    )


@pytest.mark.parametrize(
    "quant_offset,want", [(0, 32), (5, 37), (100, 127), (-1, 31), (-256, -128)]
)
def test_the_quant_offset_parameter_works_on_the_move_path(quant_offset, want, device):
    """The quant offset parameter works on the move path."""
    a = torch.ones((M, K), dtype=torch.float16, device=device)
    b = torch.ones((K, N), dtype=torch.float16, device=device)
    out = torch.zeros((M, N), dtype=torch.int8, device=device)
    make_move_with_quant_offset(quant_offset)(a, b, out)
    torch.npu.synchronize()
    got = out.cpu().numpy().astype(np.int64)

    assert got.min() == got.max() == want, (
        f"move 路 quant_offset={quant_offset}：期望整块都是 {want}，"
        f"实测 [{got.min()}, {got.max()}]"
    )


def test_the_quant_offset_parameter_agrees_with_the_hand_packed_payload(device):
    """The quant offset parameter agrees with the hand packed payload."""
    a = torch.ones((M, K), dtype=torch.float16, device=device)
    b = torch.ones((K, N), dtype=torch.float16, device=device)

    via_param = torch.zeros((M, N), dtype=torch.int8, device=device)
    make_move_with_quant_offset(5)(a, b, via_param)
    torch.npu.synchronize()

    via_payload = _run_payload_int8(ref.encode_quant_payload(0.5, offset=5), device)

    assert int(via_param.cpu().numpy()[0, 0]) == via_payload, (
        f"参数给 {int(via_param.cpu().numpy()[0, 0])}，手搓负载给 {via_payload}——两条路不一致"
    )


# ---------------------------------------------------------------------------
# Acc→GM：同一批 dtype × 两种目的 layout，读回最省事的一条路
# ---------------------------------------------------------------------------

_STORE_CELLS = [(s, d, c, ln) for s, d, c, _ in ref.MATRIX for ln in ("ND", "NZ")]
_STORE_IDS = [f"{s}-{d}-{ln}" for s, d, _, ln in _STORE_CELLS]


def _run_store(src, dst, carrier, layout_name, device):
    """Run a direct GM output case in the selected physical layout."""
    import pypto_pro.language as pl

    layout = pl.TensorLayout.ND if layout_name == "ND" else pl.TensorLayout.NZ
    a, b = ref.make_inputs(M, K, N, src)
    out = torch.zeros((M, N), dtype=_DTYPE_OF[carrier], device=device)
    kernel = make_quant_store(_pl_dtype(src), _pl_dtype(dst), layout)
    kernel(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device), out)
    torch.npu.synchronize()
    return a, b, out.cpu()


@pytest.mark.parametrize(
    ("src", "dst", "carrier", "layout_name"), _STORE_CELLS, ids=_STORE_IDS
)
def test_each_acc_to_gm_cell_matches(src, dst, carrier, layout_name, device):
    """Each acc to gm cell matches."""
    a, b, got = _run_store(src, dst, carrier, layout_name, device)
    outcome = ref.store_outcome(src, dst)
    label = f"{src}→{dst}（Acc→GM, {layout_name}）"

    if layout_name == "ND":
        _check_outcome(a, b, dst, got, outcome, SCALE, label)
        return

    # NZ：比值的集合
    f = got.to(torch.float32).numpy().astype(np.float64).reshape(-1)
    if outcome == "exact":
        want = ref.scaled_exact(a, b, SCALE).reshape(-1)
    elif outcome == "rounded":
        want = ref.scaled_rounded(a, b, SCALE, dst).reshape(-1)
    else:  # opaque
        raw = got.to(torch.uint8).numpy().reshape(-1)
        assert (raw != 0).any(), f"{label} 输出全是 0"
        if dst == "FP8E4M3FN":
            decoded = ref.decode_fp8_e4m3(got.numpy()).reshape(-1)
            expected = ref.quantize_to_fp8_e4m3(ref.scaled_exact(a, b, SCALE)).reshape(
                -1
            )
            np.testing.assert_allclose(
                np.sort(decoded),
                np.sort(expected),
                rtol=0,
                atol=0,
                err_msg=f"{label} 的 FP8 解码值集合与参考不同",
            )
        return
    np.testing.assert_allclose(
        np.sort(f),
        np.sort(want),
        rtol=0,
        atol=0,
        err_msg=f"{label} 搬出来的值集合和参考不一样",
    )


def test_the_store_path_keeps_the_scale_that_the_move_path_cannot(device):
    """The store path keeps the scale that the move path cannot."""
    a, b, stored = _run_store("FP32", "FP32", "float32", "ND", device)
    want = ref.scaled_exact(a, b, SCALE)
    np.testing.assert_allclose(
        stored.numpy(),
        want,
        rtol=0,
        atol=0,
        err_msg="store 那条路的 FP32→FP32 也丢 scale 了——那 §八 的覆盖表要改",
    )


def test_the_overrides_table_lists_exactly_the_cells_that_differ():
    """The overrides table lists exactly the cells that differ."""
    for (src, dst), outcome in ref.STORE_OUTCOME_OVERRIDES.items():
        vec = [cell for s, d, _, cell in ref.MATRIX if (s, d) == (src, dst)]
        assert vec, f"覆盖表里的 {src}→{dst} 不在 MATRIX 里"
        assert vec[0] != outcome, (
            f"{src}→{dst} 的覆盖值 {outcome} 和 Acc→Vec 的 {vec[0]} 相同——那就不该列在覆盖表里"
        )


def test_nd_and_nz_stores_differ(device):
    """Nd and nz stores differ."""
    _, _, nd = _run_store("FP32", "FP16", "float16", "ND", device)
    _, _, nz = _run_store("FP32", "FP16", "float16", "NZ", device)
    assert not np.array_equal(nd.numpy(), nz.numpy()), (
        "ND 与 NZ 的输出逐位相同——layout 没起作用"
    )


# ---------------------------------------------------------------------------
# Acc→Mat：同一批 dtype，换一条路，读回方式完全不同
# ---------------------------------------------------------------------------

_MAT_IDS = [f"{s}-{d}" for s, d, _ in ref.MAT_MATRIX]


def _run_to_mat(src, dst, device):
    """Run a quantized L1 result through q@q readback."""
    acc2 = torch.int32 if dst in ref.SATURATION else torch.float32
    a, b = ref.make_inputs(M, K, N, src)
    out = torch.zeros((M, N), dtype=acc2, device=device)
    kernel = make_quant_move_to_mat(_pl_dtype(src), _pl_dtype(dst))
    kernel(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device), out)
    torch.npu.synchronize()
    return a, b, out.cpu()


@pytest.mark.parametrize(
    ("src", "dst", "verify"),
    [row for row in ref.MAT_MATRIX if row[2] == "q_at_q"],
    ids=[f"{s}-{d}" for s, d, v in ref.MAT_MATRIX if v == "q_at_q"],
)
def test_each_acc_to_mat_cell_reads_back_bit_exact(src, dst, verify, device):
    """Each acc to mat cell reads back bit exact."""
    a, b, got = _run_to_mat(src, dst, device)
    q = ref.quantized_in_l1(a, b, SCALE, dst)

    np.testing.assert_allclose(
        got.to(torch.float64).numpy(),
        q @ q,
        rtol=0,
        atol=0,
        err_msg=f"{src}→{dst}（Acc→Mat）的 q@q 不等于参考",
    )


def test_the_acc_to_mat_readback_is_not_a_no_op(device):
    """The acc to mat readback is not a no op."""
    a, b, got = _run_to_mat("FP32", "FP16", device)
    acc = ref.accumulate(a, b)
    unscaled = (acc.astype(np.float16).astype(np.float64)) @ (
        acc.astype(np.float16).astype(np.float64)
    )

    assert not np.array_equal(got.to(torch.float64).numpy(), unscaled), (
        "Acc→Mat 的 q@q 等于未缩放的 acc@acc——scale 在这条路上也被丢了"
    )


@pytest.mark.parametrize(
    ("src", "dst"),
    [(s, d) for s, d, v in ref.MAT_MATRIX if v == "no_readback"],
    ids=[f"{s}-{d}" for s, d, v in ref.MAT_MATRIX if v == "no_readback"],
)
def test_a_uint8_tile_in_l1_cannot_be_a_matmul_operand(src, dst, device):
    """A uint8 tile in l1 cannot be a matmul operand."""
    from pypto_pro._errors import InvalidType

    a, b = ref.make_inputs(M, K, N, src)
    kernel = make_quant_move_to_mat(_pl_dtype(src), _pl_dtype(dst))
    with pytest.raises(InvalidType, match="unsupported layout/dtype combination"):
        kernel(
            torch.from_numpy(a),
            torch.from_numpy(b),
            torch.zeros(M, N, dtype=torch.int32),
        )


def test_the_acc_to_mat_hf8_cell_runs(device):
    """The acc to mat hf8 cell runs."""
    _, _, got = _run_to_mat("FP32", "HF8", device)
    values = got.to(torch.float64).numpy()
    assert np.isfinite(values).all(), "HF8 的 q@q 产生了非有限值"
    assert (values != 0).any(), "HF8 的 q@q 全是 0——量化或读回没生效"
    assert len(np.unique(values)) > 4, (
        f"HF8 的 q@q 只有 {len(np.unique(values))} 个不同值——不像真的乘法结果"
    )


def test_the_two_matrices_cover_the_same_dtypes_minus_one_cell():
    """The two matrices cover the same dtypes minus one cell."""
    vec_cells = {(s, d) for s, d, _, _ in ref.MATRIX}
    mat_cells = {(s, d) for s, d, _ in ref.MAT_MATRIX}
    assert vec_cells - mat_cells == {("FP32", "FP32")}, (
        f"Acc→Vec 比 Acc→Mat 多出来的不只是 FP32→FP32：{vec_cells - mat_cells}"
    )
    assert not mat_cells - vec_cells, (
        f"Acc→Mat 有 Acc→Vec 没有的格：{mat_cells - vec_cells}"
    )


def test_int32_to_int32_rejects_a_scale(device):
    """Int32 to int32 rejects a scale."""
    from pypto_pro._errors import InvalidType

    a, b = ref.make_inputs(M, K, N, "INT32")
    kernel = make_quant_move(_pl_dtype("INT32"), _pl_dtype("INT32"), with_scale=True)
    with pytest.raises(
        InvalidType, match="unsupported quantized layout/dtype combination"
    ):
        kernel(
            torch.from_numpy(a),
            torch.from_numpy(b),
            torch.zeros(M, N, dtype=torch.int32),
        )


def test_int32_to_int32_works_without_a_scale(device):
    """Int32 to int32 works without a scale."""
    a, b, got = _run("INT32", "INT32", "int32", device, with_scale=False)
    np.testing.assert_allclose(
        got.to(torch.float64).numpy(),
        ref.accumulate(a, b),
        rtol=0,
        atol=0,
        err_msg="不带 scale 的 INT32→INT32 不等于未缩放的 acc",
    )


#: HF8 的最坏相对舍入误差（实测上界）。hifloat8 是**渐变格式**——指数越大尾数越少,
#: 所以这个数比 `1/16` 大。实测最大 0.111，取 `1/8` 当界。
HF8_MAX_REL = 0.125


def test_hf8_decodes_to_the_scaled_result_within_its_precision(device):
    """Hf8 decodes to the scaled result within its precision."""
    a, b = ref.make_inputs(M, K, N, "FP32")
    out = torch.zeros((1, 64), dtype=torch.float32, device=device)
    hf8_move_and_decode(
        torch.from_numpy(a).to(device), torch.from_numpy(b).to(device), out
    )
    torch.npu.synchronize()

    got = out.cpu().numpy().reshape(-1).astype(np.float64)
    # astype 的 CastLayout.ZERO 在 4 倍扩宽下取的是 stride 4 相位 0
    want = ref.scaled_exact(a, b, SCALE).reshape(-1)[0::4][:64]

    rel = np.abs(got - want) / np.maximum(np.abs(want), 1.0)
    assert rel.max() <= HF8_MAX_REL, (
        f"HF8 解码值与 scale*acc 的最大相对差 {rel.max():.4f} 超过 {HF8_MAX_REL}——"
        "要么 HF8 的值不对，要么 CastLayout.ZERO 取的不是 stride 4 相位 0"
    )
    assert (got == want).any(), "没有一项逐位相等——像是系统性偏移，不是舍入"
    assert len(np.unique(got)) > 4, (
        f"解码结果只有 {len(np.unique(got))} 个不同值——不像真的解码"
    )


def test_hf8_byte_payload_is_nontrivial(device):
    """Hf8 byte payload is nontrivial."""
    _, _, got = _run("FP32", "HF8", "uint8", device)
    raw = got.numpy().reshape(-1)
    assert (raw != 0).any(), "HF8 输出全是 0——搬运没生效"
    assert len(np.unique(raw)) > 4, (
        f"HF8 输出只有 {len(np.unique(raw))} 个不同字节——不像真的转换结果"
    )


# ---------------------------------------------------------------------------
# 两个和直觉相反的事实
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src", "dst", "carrier"),
    [(s, d, c) for s, d, c, _ in ref.MATRIX if d not in ref.NO_SCALE_DESTINATIONS[s]],
    ids=[
        f"{s}-{d}" for s, d, _, _ in ref.MATRIX if d not in ref.NO_SCALE_DESTINATIONS[s]
    ],
)
def test_these_destinations_are_unreachable_without_a_scale(src, dst, carrier, device):
    """These destinations are unreachable without a scale."""
    a, b = ref.make_inputs(M, K, N, src)
    out = torch.zeros((M, N), dtype=_DTYPE_OF[carrier], device=device)
    kernel = make_quant_move(_pl_dtype(src), _pl_dtype(dst), with_scale=False)
    with pytest.raises(
        Exception, match=r"(?s)unsupported layout|InvalidType|Failed to compile"
    ):
        kernel(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device), out)


@pytest.mark.parametrize("scale", [0.5, 2.0, 0.125])
@pytest.mark.parametrize("per_channel", [False, True], ids=["scalar", "per-channel"])
def test_fp32_to_fp32_is_rejected_on_the_move_path(scale, per_channel, device):
    """Fp32 to fp32 is rejected on the move path."""
    with pytest.raises(
        NotSupported, match="scale is silently dropped for Acc-to-Vec FP32 -> FP32"
    ):
        if per_channel:
            _run_per_channel(
                "FP32",
                "FP32",
                "float32",
                device,
                scales=np.full((1, N), scale, dtype=np.float64),
            )
        else:
            _run("FP32", "FP32", "float32", device, scale=scale)


# ---------------------------------------------------------------------------
# 饱和
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("dst", "carrier"), [("INT8", "int8"), ("UINT8", "uint8")])
def test_integer_destinations_saturate_at_the_measured_range(dst, carrier, device):
    """Integer destinations saturate at the measured range."""
    lo, hi = ref.SATURATION[dst]
    a, b, got = _run("FP32", dst, carrier, device, scale=16.0)
    g = got.to(torch.float32).numpy()

    assert g.min() >= lo and g.max() <= hi, (
        f"{dst} 的输出 [{g.min()}, {g.max()}] 超出饱和区间 [{lo}, {hi}]"
    )
    assert (g == hi).any(), f"{dst} 没有一个值饱和到 {hi}——scale 放得不够大，验不到饱和"
    np.testing.assert_allclose(
        g,
        ref.scaled_rounded(a, b, 16.0, dst),
        rtol=0,
        atol=0,
        err_msg=f"{dst} 饱和后的值与参考不符",
    )


def test_negatives_clamp_to_zero_for_unsigned(device):
    """Negatives clamp to zero for unsigned."""
    a, b, from_u8 = _run("FP32", "UINT8", "uint8", device)
    _, _, from_i8 = _run("FP32", "INT8", "int8", device)
    acc = ref.accumulate(a, b)
    n_negative = int((acc < 0).sum())

    assert n_negative > 0, "输入里没有负数，这条验空了"
    assert int((from_u8.numpy() == 0).sum()) >= n_negative, (
        f"UINT8 的 0 只有 {int((from_u8.numpy() == 0).sum())} 个，"
        f"少于参考里的 {n_negative} 个负值——负数没被钳到 0"
    )
    assert (from_i8.numpy() < 0).any(), "INT8 里没有负值——符号丢了"


# ---------------------------------------------------------------------------
# 结构检查
# ---------------------------------------------------------------------------


def test_the_matrix_covers_the_documented_combos():
    """The matrix covers the documented combos."""
    documented = {
        ("FP32", "INT8"),
        ("FP32", "UINT8"),
        ("FP32", "HF8"),
        ("FP32", "FP16"),
        ("FP32", "BF16"),
        ("FP32", "FP8E4M3FN"),
        ("FP32", "FP32"),
        ("INT32", "INT8"),
        ("INT32", "UINT8"),
        ("INT32", "FP16"),
        ("INT32", "BF16"),
    }
    measured = {(s, d) for s, d, _, _ in ref.MATRIX}
    assert measured == documented, (
        f"缺 {sorted(documented - measured)}，多 {sorted(measured - documented)}"
    )
    rejected = {(s, d) for s, d, _, outcome in ref.MATRIX if outcome == "rejected"}
    assert rejected == {("FP32", "FP32")}, f"未由数值矩阵覆盖的组合发生变化：{rejected}"
