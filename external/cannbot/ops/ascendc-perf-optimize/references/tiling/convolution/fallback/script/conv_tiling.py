#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Conv2D Formula-Based Tiling Script.

Replicates the tiling algorithm from:
  Conv2dBaseTiling::DoOpTiling()
  -> GetTilingFromFastTiling()
    -> Conv2dOpTilingSetShape()    -- sets up shape/attr/type, decides M/HW-split
    -> NumBlocksDecision()          -- computes multi-core split
    -> GetConv2dOpsTiling()         -- fills runInfo, calls GetConv2dApiTiling()
      -> Conv2dApiTilingSetShape()  -- single-core shapes from numBlocks
      -> conv2dApiTiling_.GetTiling() -- formula-based API tiling
  -> DoLibApiTiling()
    -> SetTilingKey()               -- computes tiling key from result

Usage:
  python conv_tiling.py -n 1 -c 64 -hi 56 -wi 56 -co 64 -kh 3 -kw 3
  python conv_tiling.py -n 1 -c 64 -hi 56 -wi 56 -co 64 -kh 3 -kw 3 -g 64  # depthwise
"""

import argparse
import itertools
import logging
import math
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


DOUBLE_BUFFER_NUM = 2
C0_SIZE = 32
C0_BYTE_SIZE = 32
MIN_BURST_SIZE = 128

M0 = 16
N0 = 16
B16_K0 = 16
B32_K0 = 8
B8_K0 = 32

MAX_16_BIT_NUM = 65535
LOAD3D_M_START_POS_LIMIT = 32767
POSTK_LIMIT = 65535

C04_CIN_SIZE = 4
VGATHER_REGISTER_SIZE = 256
BAND_WIDTH_COEFF = 4

FP16_DTYPE_SIZE = 2
DATACOPYPARAMS_BURSTLEN_MAX = 65535

MKN_M_INDEX = 0
MKN_K_INDEX = 1
MKN_N_INDEX = 2

MAX_OUT_TYPE_SIZE = 4
TOTAL_SCALE_BIAS_16_TYPE_SIZE = 10
TOTAL_SCALE_BIAS_32_TYPE_SIZE = 8


class ConvDtype(IntEnum):
    UNDEFINED = 0
    FLOAT16 = 1
    FLOAT32 = 2
    BFLOAT16 = 3
    INT4 = 4
    INT8 = 5
    UINT8 = 6
    INT16 = 7
    UINT16 = 8
    INT32 = 9
    UINT32 = 10
    INT64 = 11
    UINT64 = 12
    HIFLOAT8 = 13
    FLOAT8_E4M3FN = 14
    DOUBLE = 15


class OutputOrder(IntEnum):
    HW = 0
    M = 1


class IterateMNOrder(IntEnum):
    ITER_M_FST = 0
    ITER_N_FST = 1


DTYPE_SIZE_TAB: Dict[ConvDtype, int] = {
    ConvDtype.FLOAT16: 2, ConvDtype.FLOAT32: 4, ConvDtype.BFLOAT16: 2,
    ConvDtype.INT4: 1, ConvDtype.INT8: 1, ConvDtype.INT32: 4,
    ConvDtype.INT64: 8, ConvDtype.UINT64: 8,
    ConvDtype.HIFLOAT8: 1, ConvDtype.FLOAT8_E4M3FN: 1,
}

MKN_TABLE: Dict[ConvDtype, Tuple[int, int, int]] = {}
for _dt in [ConvDtype.FLOAT16, ConvDtype.BFLOAT16]:
    MKN_TABLE[_dt] = (16, 16, 16)
MKN_TABLE[ConvDtype.FLOAT32] = (16, 8, 16)
for _dt in [ConvDtype.INT8, ConvDtype.HIFLOAT8, ConvDtype.FLOAT8_E4M3FN]:
    MKN_TABLE[_dt] = (16, 32, 16)


def align_b(a: int, b: int) -> int:
    if b == 0:
        return 0
    return ((a + b - 1) // b) * b


def ceil_div(a: int, b: int) -> int:
    if b == 0:
        return 0
    return (a + b - 1) // b


def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return (a * b) // math.gcd(a, b)


def calc_comm_factor(num: int, num_max: int) -> List[int]:
    if num_max == 0 or num == 0:
        return [1]
    res = set()
    for i in range(1, int(math.isqrt(num)) + 1):
        if num % i == 0:
            if i <= num_max:
                res.add(i)
            right = num // i
            if right != i and right <= num_max:
                res.add(right)
    result = sorted(res)
    return result if result else [1]


def calc_comm_factor_with_power_of_two(num: int, num_max: int) -> List[int]:
    res = set(calc_comm_factor(num, num_max))
    i = 2
    while i <= min(num, num_max):
        res.add(i)
        i *= 2
    return sorted(res)


def calc_comm_factor_of_two_num(num1: int, num2: int) -> List[int]:
    res = []
    for i in range(1, min(num1, num2) + 1):
        if num1 % i == 0 and num2 % i == 0:
            res.append(i)
    return sorted(res)


def infer_hi_l1(ho_l1: int, cfg: 'Conv2DConfig') -> int:
    kh_dilated = (cfg.kh - 1) * cfg.dilation_h + 1
    return min((ho_l1 - 1) * cfg.stride_h + kh_dilated, cfg.hi)


def infer_wi_l1(wo_l1: int, cfg: 'Conv2DConfig') -> int:
    kw_dilated = (cfg.kw - 1) * cfg.dilation_w + 1
    return min((wo_l1 - 1) * cfg.stride_w + kw_dilated, cfg.wi)


def dtype_from_str(s: str) -> ConvDtype:
    m = {
        "fp16": ConvDtype.FLOAT16, "float16": ConvDtype.FLOAT16,
        "fp32": ConvDtype.FLOAT32, "float32": ConvDtype.FLOAT32,
        "bf16": ConvDtype.BFLOAT16, "bfloat16": ConvDtype.BFLOAT16,
        "int8": ConvDtype.INT8, "int32": ConvDtype.INT32,
        "hf8": ConvDtype.HIFLOAT8, "hifloat8": ConvDtype.HIFLOAT8,
    }
    if s.lower() in m:
        return m[s.lower()]
    logger.warning("unknown dtype '%s', defaulting to fp16", s)
    return ConvDtype.FLOAT16


def _make_sized_range(values: List[int], multiplier: int) -> List[int]:
    return [v * multiplier for v in values]


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Conv2DConfig:
    batch: int = 1
    ci: int = 64
    hi: int = 56
    wi: int = 56
    co: int = 64
    kh: int = 3
    kw: int = 3
    stride_h: int = 1
    stride_w: int = 1
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    groups: int = 1

    @property
    def ho(self) -> int:
        return self.calc_ho()

    @property
    def wo(self) -> int:
        return self.calc_wo()

    def calc_ho(self) -> int:
        return (self.hi + self.pad_top + self.pad_bottom -
                self.dilation_h * (self.kh - 1) - 1) // self.stride_h + 1

    def calc_wo(self) -> int:
        return (self.wi + self.pad_left + self.pad_right -
                self.dilation_w * (self.kw - 1) - 1) // self.stride_w + 1


@dataclass
class PlatformInfo:
    l1_size: int = 524288
    l0_a_size: int = 65536
    l0_b_size: int = 65536
    l0_c_size: int = 262144
    ub_size: int = 262144
    bt_size: int = 4096
    fb_size: int = 4096
    aiv_per_aic: int = 2
    ab_l1_mte2_bw_cof: int = 4


@dataclass
class MKNConfig:
    m0: int = 16
    k0: int = 16
    n0: int = 16
    fmap_dtype_size: int = 2
    weight_dtype_size: int = 2


@dataclass
class TilingData:
    org_hi: int = 0
    org_wi: int = 0
    org_ho: int = 0
    org_wo: int = 0
    org_hix_wi: int = 0
    org_ci: int = 0
    org_co: int = 0

    single_core_batch: int = 0
    single_core_ho: int = 0
    single_core_wo: int = 0
    single_core_ci: int = 0
    single_core_co: int = 0

    ho_l1: int = 0
    wo_l1: int = 0
    k_al1: int = 0
    k_bl1: int = 0
    kh_l1: int = 0
    kw_l1: int = 0
    n_bl1: int = 0

    ho_l0: int = 0
    wo_l0: int = 0
    k_l0: int = 0
    n_l0: int = 0

    p_buffer_flag: int = 0
    a_l1_space_size: int = 0
    multi_n_bl1: int = 0

    groups: int = 1
    enlarge: int = 0
    single_core_groups: int = 0
    single_core_group_opt: int = 0

    b_ub_n_step: int = 0
    b_ub_k_step: int = 0
    kh_ub: int = 0
    kw_ub: int = 0

    kernel_h: int = 0
    kernel_w: int = 0
    kernel_hx_kernel_w: int = 0
    kernel_hx_kernel_wx_kernel_d: int = 0

    stride_h: int = 1
    stride_w: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0

    m_step: int = 0
    k_step: int = 0
    n_step: int = 0
    fmap_k_stride: int = 0
    weight_k_stride: int = 0
    cin_a_in_core: int = 0
    cin_a_tail_in_core: int = 0
    cin_b_in_core: int = 0
    cin_b_tail_in_core: int = 0
    cin_offset_block_in_gm: int = 0
    cout_offset_block: int = 0
    n_l1_div_block_size: int = 0

    iterate_mn_order: int = 0
    bias_full_load_flag: int = 0
    fixp_params_full_load_flag: int = 0
    hf32_enable: int = 0
    hf32_trans_mode: int = 0
    has_bias: int = 0
    has_scale: int = 0
    dual_output: int = 0
    quant_mode0: int = 0
    relu_mode0: int = 0
    clip_mode0: int = 0
    quant_mode1: int = 0
    relu_mode1: int = 0
    clip_mode1: int = 0
    offsetx: int = 0
    round_mode: int = 0
    union_data_xt: int = 0
    inner_batch: int = 1


@dataclass
class NumBlocksResult:
    batch_dim: int = 1
    m_dim: int = 1
    n_dim: int = 1
    ho_dim: int = 1
    wo_dim: int = 1
    group_dim: int = 1


@dataclass
class TilingKeyParams:
    fmp_tiling: int = 2
    weight_tiling: int = 2
    l1_ping_pong: int = 0
    l0_ping_pong: int = 0
    output_order: int = 0
    iter_order: int = 0
    group_type: int = 0
    enable_small_channel: int = 0
    weight_ub_trans: int = 0
    fmap_copy_mode: int = 0
    inner_batch: int = 0


@dataclass
class L1TilingResult:
    k_al1: int = 0
    k_bl1: int = 0
    m_al1: int = 0
    wo_al1: int = 0
    n_bl1: int = 0
    iterate_mn_order: int = IterateMNOrder.ITER_M_FST
    bias_full_load: bool = False
    fixp_full_load: bool = False
    pb_al1: int = DOUBLE_BUFFER_NUM
    pb_bl1: int = DOUBLE_BUFFER_NUM


# --- Input bundling dataclasses (reduce function argument counts) ---

@dataclass
class L0TilingCtx:
    """Context for L0 tiling functions."""
    mkn: MKNConfig
    platform: PlatformInfo
    cfg: Conv2DConfig
    is_c04: bool = False


@dataclass
class L0Result:
    """L0 tiling result for M-mode or HW-mode."""
    m_l0: int = 0
    k_l0: int = 0
    n_l0: int = 0
    inner_batch: int = 1
    ho_l0: int = 0
    wo_l0: int = 0


@dataclass
class L1MmodeSizes:
    """Pre-computed sizes for L1 M-mode tiling."""
    ci0_h_kw: int = 0
    hi_l1_min: int = 0
    ho_l1_min: int = 0
    k_al1_full_size: int = 0
    k_bl1_full_size: int = 0
    fmap_full_l1: int = 0
    weight_full_l1: int = 0
    fmap_min_l1: int = 0
    weight_min_l1: int = 0
    bias_min_l1: int = 0
    k_al1_min: int = 0
    k_bl1_min: int = 0


@dataclass
class L1MmodeCtx:
    """Context for L1 M-mode strategy helpers."""
    s: L1MmodeSizes
    mkn: MKNConfig
    platform: PlatformInfo
    cfg: Conv2DConfig
    single_m: int
    single_ci1: int
    single_co1: int
    m_l0: int
    n_l0: int
    inner_batch: int


@dataclass
class L1MmodeInput:
    """All inputs for l1_tiling_mmode."""
    single_m: int
    single_ci: int
    single_ci1: int
    single_co1: int
    single_batch: int
    mkn: MKNConfig
    platform: PlatformInfo
    cfg: Conv2DConfig
    m_l0: int
    n_l0: int
    k_l0: int  # unused but kept for signature compatibility
    inner_batch: int
    is_c04: bool = False
    is_dma: bool = False
    has_bias: bool = False
    bias_dtype_size: int = 0


@dataclass
class L1HwmodeCtx:
    """Context for L1 HW-mode strategy helpers."""
    mkn: MKNConfig
    platform: PlatformInfo
    cfg: Conv2DConfig
    result: L1TilingResult
    ho_l0: int
    wo_l0: int
    n_l0: int
    single_ci1: int
    single_co1: int
    is_c04: bool = False


@dataclass
class L1HwmodeFullSizes:
    """Max/min sizes for L1 HW-mode tiling."""
    k_al1_max: int = 0
    k_bl1_max: int = 0
    n_bl1_max: int = 0
    ho_max: int = 0
    wo_max: int = 0
    a_l1_max: int = 0
    b_l1_max: int = 0
    bias_max: int = 0
    a_l1_min: int = 0
    b_l1_min: int = 0
    bias_min: int = 0


@dataclass
class L1HwmodeInput:
    """All inputs for l1_hw_tiling."""
    single_ho: int
    single_wo: int
    single_ci: int
    single_ci1: int
    single_co1: int
    mkn: MKNConfig
    platform: PlatformInfo
    cfg: Conv2DConfig
    ho_l0: int
    wo_l0: int
    n_l0: int
    is_c04: bool = False
    is_dma: bool = False
    has_bias: bool = False
    bias_dtype_size: int = 0


@dataclass
class _NumBlocksBase:
    """Shared fields for M-split and HW-split numBlocks inputs."""
    aicore_num: int
    batch: int
    ho: int
    wo: int
    cur_ci: int
    cur_co: int
    cur_groups: int
    kh: int
    kw: int
    n0: int
    opt_group: bool


@dataclass
class NumBlocksMInput(_NumBlocksBase):
    """Inputs for num_blocks_m_split."""
    m0: int = 16
    k0: int = 16


@dataclass
class NumBlocksHwInput(_NumBlocksBase):
    """Inputs for num_blocks_hw_split."""
    pass


@dataclass
class CostCalcInput:
    """Inputs for calc_total_cost_m_split."""
    dims: Tuple[int, int, int, int]  # batch_dim, m_dim, n_dim, group_dim
    batch: int
    ho: int
    wo: int
    cur_ci: int
    cur_co: int
    cur_groups: int
    kh: int
    kw: int
    m0: int
    k0: int
    n0: int
    opt_group: bool


@dataclass
class TilingKeyCtx:
    """Inputs for compute_tiling_key."""
    t: TilingData
    nb: NumBlocksResult
    ci: int
    cur_co: int
    kh: int
    kw: int
    dtype: ConvDtype
    groups_active: bool
    opt_group: bool
    m_split: bool


@dataclass
class ComputeTilingInput:
    """All inputs for compute_tiling."""
    cfg: Conv2DConfig
    dtype_str: str = "fp16"
    output_order: int = 1
    aicore_num: int = 32
    platform: PlatformInfo = field(default_factory=PlatformInfo)


@dataclass
class RunL0L1Ctx:
    """Context for _run_l0_l1_tiling."""
    t: TilingData
    cfg: Conv2DConfig
    platform: PlatformInfo
    mkn: MKNConfig
    dtype: ConvDtype
    sc: 'SingleCoreShapes'
    m_split: bool
    ho: int
    wo: int


@dataclass
class SingleCoreShapes:
    """Single-core shape parameters."""
    batch: int = 0
    co: int = 0
    ho: int = 0
    wo: int = 0
    m: int = 0
    ci: int = 0


@dataclass
class ScalarParamsCtx:
    """Inputs for _compute_scalar_params."""
    t: TilingData
    single_core_ci: int
    ci: int
    groups: int
    m0: int
    n0: int
    m_split: bool


@dataclass
class InitTilingDataCtx:
    """Inputs for _init_tiling_data."""
    cfg: Conv2DConfig
    ho: int
    wo: int
    sc: SingleCoreShapes
    groups_active: bool
    enlarge: int
    s_groups: int
    s_group_opt: int


@dataclass
class OptGroupCtx:
    """Inputs for _compute_opt_group."""
    groups: int
    ci: int
    co: int
    mkn: MKNConfig
    kh: int
    kw: int
    dtype: ConvDtype
    platform: PlatformInfo


# ============================================================================
# L0 Tiling -- common helpers
# ============================================================================

def _calc_n_l0_range(single_co1: int, ctx: L0TilingCtx) -> List[int]:
    n_l0_max = min(
        ctx.platform.l0_b_size // (ctx.mkn.k0 * DOUBLE_BUFFER_NUM * ctx.mkn.weight_dtype_size),
        ctx.platform.l0_c_size // (ctx.mkn.m0 * 4),
    )
    return _make_sized_range(
        calc_comm_factor_with_power_of_two(single_co1, n_l0_max // ctx.mkn.n0), ctx.mkn.n0)


# ============================================================================
# L0 Tiling (M-mode)
# ============================================================================

def _check_l0_buffer_mmode(ml0: int, kl0: int, nl0: int, ctx: L0TilingCtx) -> bool:
    m = ctx.mkn
    a_size = align_b(ml0, m.m0) * align_b(kl0, m.k0) * DOUBLE_BUFFER_NUM * m.fmap_dtype_size
    b_size = align_b(kl0, m.k0) * align_b(nl0, m.n0) * DOUBLE_BUFFER_NUM * m.weight_dtype_size
    c_size = align_b(ml0, m.m0) * align_b(nl0, m.n0) * 4
    return (a_size <= ctx.platform.l0_a_size and
            b_size <= ctx.platform.l0_b_size and
            c_size <= ctx.platform.l0_c_size)


def _calc_l1_estimate_mmode(ml0: int, nl0: int, ctx: L0TilingCtx) -> int:
    m = ctx.mkn
    ho_l1_min = ml0 // ctx.cfg.wi + 2
    hi = infer_hi_l1(ho_l1_min, ctx.cfg)
    k_al1_min = C04_CIN_SIZE if ctx.is_c04 else m.k0
    a_size = hi * ctx.cfg.wi * k_al1_min * m.fmap_dtype_size * DOUBLE_BUFFER_NUM
    k_bl1_min = (C04_CIN_SIZE * ctx.cfg.kh * ctx.cfg.kw) if ctx.is_c04 else ctx.cfg.kh * ctx.cfg.kw * m.k0
    w_size = nl0 * k_bl1_min * m.weight_dtype_size * DOUBLE_BUFFER_NUM
    return a_size + w_size


def _expand_l0_mmode(state: Tuple[int, int, int, int],
                     m_l0_range: List[int], n_l0_range: List[int],
                     ctx: L0TilingCtx) -> Tuple[int, int, int, int]:
    m_l0, n_l0, m_idx, n_idx = state
    k_l0 = ctx.mkn.k0
    while (_check_l0_buffer_mmode(m_l0, k_l0, n_l0, ctx) and
           _calc_l1_estimate_mmode(m_l0, n_l0, ctx) <= ctx.platform.l1_size):
        if m_l0 <= n_l0:
            if m_idx < len(m_l0_range) - 1:
                m_idx += 1
            elif n_idx < len(n_l0_range) - 1:
                n_idx += 1
            else:
                break
        else:
            if n_idx < len(n_l0_range) - 1:
                n_idx += 1
            elif m_idx < len(m_l0_range) - 1:
                m_idx += 1
            else:
                break
        m_l0 = m_l0_range[m_idx]
        n_l0 = n_l0_range[n_idx]

    if not (_check_l0_buffer_mmode(m_l0, k_l0, n_l0, ctx) and
            _calc_l1_estimate_mmode(m_l0, n_l0, ctx) <= ctx.platform.l1_size):
        if m_l0 <= n_l0:
            m_idx = max(0, m_idx - 1)
        else:
            n_idx = max(0, n_idx - 1)
        m_l0 = m_l0_range[m_idx]
        n_l0 = n_l0_range[n_idx]
    return m_l0, n_l0, m_idx, n_idx


def l0_tiling_mmode(single_m: int, single_ci1: int, single_co1: int,
                    ctx: L0TilingCtx) -> L0Result:
    """Compute L0 tiling (mL0, kL0, nL0) for M-mode."""
    m = ctx.mkn
    n_l0_range = _calc_n_l0_range(single_co1, ctx)

    m_l0_max = min(
        ctx.platform.l0_a_size // (m.k0 * DOUBLE_BUFFER_NUM * m.fmap_dtype_size),
        ctx.platform.l0_c_size // (m.n0 * 4),
    )
    m_l0_range = _make_sized_range(
        calc_comm_factor_with_power_of_two(ceil_div(single_m, m.m0), m_l0_max // m.m0), m.m0)

    m_l0, n_l0, _, _ = _expand_l0_mmode(
        (m_l0_range[0], n_l0_range[0], 0, 0), m_l0_range, n_l0_range, ctx)

    inner_batch = 1
    if single_m <= m_l0:
        ib1 = ctx.platform.l0_a_size // (2 * m.fmap_dtype_size * m_l0 * m.k0)
        ib2 = ctx.platform.l0_c_size // (m_l0 * n_l0 * 4 * 2)
        if ib1 > 1 and ib2 > 1:
            inner_batch = min(ib1, ib2)

    return L0Result(m_l0=m_l0, k_l0=m.k0, n_l0=n_l0, inner_batch=inner_batch)


# ============================================================================
# L0 Tiling (HW-mode)
# ============================================================================

def _check_l0_buffer_hw(hl0: int, wl0: int, kl0: int, nl0: int,
                        ctx: L0TilingCtx) -> bool:
    m = ctx.mkn
    a_size = align_b(hl0 * wl0, m.m0) * align_b(kl0, m.k0) * DOUBLE_BUFFER_NUM * m.fmap_dtype_size
    b_size = align_b(kl0, m.k0) * align_b(nl0, m.n0) * DOUBLE_BUFFER_NUM * m.weight_dtype_size
    c_size = align_b(hl0 * wl0, m.m0) * align_b(nl0, m.n0) * 4
    return (a_size <= ctx.platform.l0_a_size and
            b_size <= ctx.platform.l0_b_size and
            c_size <= ctx.platform.l0_c_size)


def _calc_l1_estimate_hw(hl0: int, wl0: int, nl0: int, ctx: L0TilingCtx) -> int:
    m = ctx.mkn
    hi = infer_hi_l1(hl0, ctx.cfg)
    wi = infer_wi_l1(wl0, ctx.cfg)
    a_size = hi * wi * m.k0 * m.fmap_dtype_size * DOUBLE_BUFFER_NUM
    w_size = ctx.cfg.kh * ctx.cfg.kw * m.k0 * nl0 * m.weight_dtype_size * DOUBLE_BUFFER_NUM
    return a_size + w_size


def _try_expand_l0_hw(tiles: Tuple[int, int, int], ctx: L0TilingCtx,
                      ranges: Tuple[List[int], List[int], List[int]]
                      ) -> Tuple[int, int, int]:
    hl0, wl0, nl0 = tiles
    n_l0_range, wo_l0_range, ho_l0_range = ranges
    k_l0 = ctx.mkn.k0
    while (_check_l0_buffer_hw(hl0, wl0, k_l0, nl0, ctx) and
           _calc_l1_estimate_hw(hl0, wl0, nl0, ctx) <= ctx.platform.l1_size):
        expanded = False
        if nl0 < hl0 * wl0:
            ni = n_l0_range.index(nl0)
            if ni < len(n_l0_range) - 1:
                nl0 = n_l0_range[ni + 1]
                expanded = True
        if not expanded:
            wi = wo_l0_range.index(wl0)
            if wi < len(wo_l0_range) - 1:
                wl0 = wo_l0_range[wi + 1]
                expanded = True
        if not expanded:
            hi = ho_l0_range.index(hl0)
            if hi < len(ho_l0_range) - 1:
                hl0 = ho_l0_range[hi + 1]
                expanded = True
        if not expanded:
            break
    ho_l0, wo_l0, n_l0 = hl0, wl0, nl0

    if not (_check_l0_buffer_hw(ho_l0, wo_l0, k_l0, n_l0, ctx) and
            _calc_l1_estimate_hw(ho_l0, wo_l0, n_l0, ctx) <= ctx.platform.l1_size):
        if n_l0 < ho_l0 * wo_l0:
            n_l0 = n_l0_range[max(0, n_l0_range.index(n_l0) - 1)]
        else:
            wo_l0 = wo_l0_range[max(0, wo_l0_range.index(wo_l0) - 1)]
    return ho_l0, wo_l0, n_l0


def l0_hw_tiling(single_ho: int, single_wo: int, single_co1: int,
                 ctx: L0TilingCtx) -> L0Result:
    """Compute L0 tiling (hoL0, woL0, kL0, nL0) for HW-mode."""
    m = ctx.mkn
    n_l0_range = _calc_n_l0_range(single_co1, ctx)

    m_l0_max = min(
        ctx.platform.l0_a_size // (m.k0 * DOUBLE_BUFFER_NUM * m.fmap_dtype_size),
        ctx.platform.l0_c_size // (m.n0 * 4),
    )
    wo_l0_max = min(m_l0_max, single_wo)
    wo_l0_range = _make_sized_range(
        calc_comm_factor_with_power_of_two(ceil_div(single_wo, m.m0),
                                           ceil_div(wo_l0_max, m.m0)), m.m0)
    ho_l0_max = max(min(m_l0_max // single_wo, single_ho), 1)
    ho_l0_range = calc_comm_factor_with_power_of_two(single_ho, ho_l0_max)

    ho_l0, wo_l0, n_l0 = _try_expand_l0_hw(
        (ho_l0_range[0], wo_l0_range[0], n_l0_range[0]), ctx,
        (n_l0_range, wo_l0_range, ho_l0_range))

    return L0Result(ho_l0=ho_l0, wo_l0=wo_l0, k_l0=m.k0, n_l0=n_l0)


# ============================================================================
# L1 Tiling (M-mode) -- strategy-based helpers
# ============================================================================

def _l1_mmode_sizes(ctx: L1MmodeCtx, has_bias: bool, bias_dtype_size: int,
                    is_c04: bool) -> L1MmodeSizes:
    """Compute full-load and min-load sizes for L1 M-mode tiling."""
    m = ctx.mkn
    ci0_h_kw = ctx.cfg.kh * ctx.cfg.kw * m.k0

    ho_l1_full = min((ctx.single_m // ctx.cfg.wo) + 2, ctx.cfg.ho)
    hi_l1_full = infer_hi_l1(ho_l1_full, ctx.cfg)

    if is_c04:
        k_al1_full_size = C04_CIN_SIZE
        k_bl1_full_size = align_b(C04_CIN_SIZE * ctx.cfg.kh * ctx.cfg.kw, m.k0)
        fmap_full_l1 = align_b(k_al1_full_size * hi_l1_full * ctx.cfg.wi * m.fmap_dtype_size,
                               C0_SIZE) * ctx.inner_batch
    else:
        k_al1_full_size = ctx.single_ci1 * ci0_h_kw
        k_bl1_full_size = ctx.single_ci1 * ci0_h_kw
        k_ci_full = k_al1_full_size // (ctx.cfg.kh * ctx.cfg.kw)
        fmap_full_l1 = k_ci_full * hi_l1_full * ctx.cfg.wi * m.fmap_dtype_size * ctx.inner_batch

    weight_full_l1 = k_bl1_full_size * ctx.single_co1 * m.n0 * m.weight_dtype_size

    ho_l1_min = min((ctx.m_l0 // ctx.cfg.wo) + 2, ctx.cfg.ho)
    hi_l1_min = infer_hi_l1(ho_l1_min, ctx.cfg)
    k_al1_min = C04_CIN_SIZE if is_c04 else m.k0
    k_bl1_min = (C04_CIN_SIZE * ctx.cfg.kh * ctx.cfg.kw) if is_c04 else ci0_h_kw
    if is_c04:
        k_bl1_min = align_b(k_bl1_min, m.k0)

    fmap_min_l1 = k_al1_min * hi_l1_min * ctx.cfg.wi * m.fmap_dtype_size * ctx.inner_batch
    weight_min_l1 = k_bl1_min * ctx.n_l0 * m.weight_dtype_size
    bias_min_l1 = ctx.n_l0 * bias_dtype_size if has_bias else 0

    return L1MmodeSizes(
        ci0_h_kw=ci0_h_kw, hi_l1_min=hi_l1_min, ho_l1_min=ho_l1_min,
        k_al1_full_size=k_al1_full_size, k_bl1_full_size=k_bl1_full_size,
        fmap_full_l1=fmap_full_l1, weight_full_l1=weight_full_l1,
        fmap_min_l1=fmap_min_l1, weight_min_l1=weight_min_l1,
        bias_min_l1=bias_min_l1, k_al1_min=k_al1_min, k_bl1_min=k_bl1_min)


def _fm_l1(k_ci: int, hi: int, wi: int, mkn: MKNConfig, inner_batch: int) -> int:
    return k_ci * hi * wi * mkn.fmap_dtype_size * DOUBLE_BUFFER_NUM * inner_batch


def _l1_mmode_all_full(ctx: L1MmodeCtx) -> L1TilingResult:
    s = ctx.s
    return L1TilingResult(k_al1=s.k_al1_full_size, k_bl1=s.k_bl1_full_size,
                          m_al1=align_b(ctx.single_m, ctx.mkn.m0),
                          n_bl1=ctx.mkn.n0 * ctx.single_co1, pb_al1=1, pb_bl1=1)


def _l1_mmode_bl1_full(ctx: L1MmodeCtx) -> L1TilingResult:
    s, m, p = ctx.s, ctx.mkn, ctx.platform
    result = L1TilingResult(k_bl1=s.k_bl1_full_size, n_bl1=m.n0 * ctx.single_co1,
                            pb_bl1=1, iterate_mn_order=IterateMNOrder.ITER_M_FST)
    bias = s.bias_min_l1

    k_al1_range = _make_sized_range(calc_comm_factor(ctx.single_ci1, ctx.single_ci1), s.ci0_h_kw)
    k_al1 = k_al1_range[0]
    for v in k_al1_range:
        k_ci = v // (ctx.cfg.kh * ctx.cfg.kw)
        fmap = _fm_l1(k_ci, s.hi_l1_min, ctx.cfg.wi, m, ctx.inner_batch)
        if fmap + result.k_bl1 * result.n_bl1 * m.weight_dtype_size + bias <= p.l1_size:
            k_al1 = v
        else:
            break
    result.k_al1 = k_al1

    multi_m_max = ceil_div(align_b(ctx.single_m, m.m0), ctx.m_l0)
    m_al1_range = _make_sized_range(calc_comm_factor(multi_m_max, multi_m_max), ctx.m_l0)
    m_al1 = m_al1_range[0]
    for v in m_al1_range:
        ho_l1_cur = min((v // ctx.cfg.wo) + 2, ctx.cfg.ho)
        hi = infer_hi_l1(ho_l1_cur, ctx.cfg)
        k_ci = result.k_al1 // (ctx.cfg.kh * ctx.cfg.kw)
        fmap = _fm_l1(k_ci, hi, ctx.cfg.wi, m, ctx.inner_batch)
        if fmap + result.k_bl1 * result.n_bl1 * m.weight_dtype_size + bias <= p.l1_size:
            m_al1 = v
        else:
            break
    result.m_al1 = m_al1
    return result


def _l1_mmode_al1_full(ctx: L1MmodeCtx) -> L1TilingResult:
    s, m, p = ctx.s, ctx.mkn, ctx.platform
    result = L1TilingResult(k_al1=s.k_al1_full_size,
                            m_al1=align_b(ctx.single_m, m.m0),
                            pb_al1=1, iterate_mn_order=IterateMNOrder.ITER_N_FST)
    bias = s.bias_min_l1

    k_bl1_range = _make_sized_range(calc_comm_factor(ctx.single_ci1, ctx.single_ci1), s.ci0_h_kw)
    k_bl1 = k_bl1_range[0]
    for v in k_bl1_range:
        w_size = v * ctx.n_l0 * m.weight_dtype_size * DOUBLE_BUFFER_NUM
        if s.fmap_full_l1 + w_size + bias <= p.l1_size:
            k_bl1 = v
        else:
            break
    result.k_bl1 = k_bl1

    multi_n_max = ceil_div(ctx.single_co1 * m.n0, ctx.n_l0)
    n_bl1_range = _make_sized_range(calc_comm_factor(multi_n_max, multi_n_max), ctx.n_l0)
    n_bl1 = n_bl1_range[0]
    for v in n_bl1_range:
        w_size = result.k_bl1 * v * m.weight_dtype_size * DOUBLE_BUFFER_NUM
        if s.fmap_full_l1 + w_size + bias <= p.l1_size:
            n_bl1 = v
        else:
            break
    result.n_bl1 = n_bl1
    return result


def _l1_mmode_kal1_full(ctx: L1MmodeCtx) -> L1TilingResult:
    s, m, p = ctx.s, ctx.mkn, ctx.platform
    result = L1TilingResult(k_al1=s.k_al1_full_size,
                            iterate_mn_order=IterateMNOrder.ITER_N_FST)
    bias = s.bias_min_l1

    for v in _make_sized_range(calc_comm_factor(ctx.single_ci1, ctx.single_ci1), s.ci0_h_kw):
        k_ci = s.k_al1_full_size // (ctx.cfg.kh * ctx.cfg.kw)
        fmap = _fm_l1(k_ci, s.hi_l1_min, ctx.cfg.wi, m, ctx.inner_batch)
        if fmap + v * ctx.n_l0 * m.weight_dtype_size * DOUBLE_BUFFER_NUM + bias <= p.l1_size:
            result.k_bl1 = v
        else:
            break
    if result.k_bl1 == 0:
        result.k_bl1 = s.k_bl1_min

    multi_m_max = ceil_div(align_b(ctx.single_m, m.m0), ctx.m_l0)
    for v in _make_sized_range(calc_comm_factor(multi_m_max, multi_m_max), ctx.m_l0):
        ho_l1_cur = min((v // ctx.cfg.wo) + 2, ctx.cfg.ho)
        hi = infer_hi_l1(ho_l1_cur, ctx.cfg)
        k_ci = s.k_al1_full_size // (ctx.cfg.kh * ctx.cfg.kw)
        fmap = _fm_l1(k_ci, hi, ctx.cfg.wi, m, ctx.inner_batch)
        if fmap + result.k_bl1 * ctx.n_l0 * m.weight_dtype_size * DOUBLE_BUFFER_NUM + bias <= p.l1_size:
            result.m_al1 = v
        else:
            break
    if result.m_al1 == 0:
        result.m_al1 = ctx.m_l0
    result.n_bl1 = ctx.n_l0
    return result


def _l1_mmode_kbl1_full(ctx: L1MmodeCtx) -> L1TilingResult:
    s, m, p = ctx.s, ctx.mkn, ctx.platform
    result = L1TilingResult(k_bl1=s.k_bl1_full_size,
                            iterate_mn_order=IterateMNOrder.ITER_M_FST)

    for v in _make_sized_range(calc_comm_factor(ctx.single_ci1, ctx.single_ci1), s.ci0_h_kw):
        k_ci = v // (ctx.cfg.kh * ctx.cfg.kw)
        fmap = _fm_l1(k_ci, s.hi_l1_min, ctx.cfg.wi, m, ctx.inner_batch)
        if fmap + result.k_bl1 * ctx.n_l0 * m.weight_dtype_size + s.bias_min_l1 <= p.l1_size:
            result.k_al1 = v
        else:
            break
    if result.k_al1 == 0:
        result.k_al1 = s.ci0_h_kw

    multi_n_max = ceil_div(ctx.single_co1 * m.n0, ctx.n_l0)
    for v in _make_sized_range(calc_comm_factor(multi_n_max, multi_n_max), ctx.n_l0):
        if (s.fmap_min_l1 + result.k_bl1 * v * m.weight_dtype_size + s.bias_min_l1
                <= p.l1_size):
            result.n_bl1 = v
        else:
            break
    if result.n_bl1 == 0:
        result.n_bl1 = ctx.n_l0
    result.m_al1 = ctx.m_l0
    return result


def _l1_mmode_both_k_full(ctx: L1MmodeCtx) -> L1TilingResult:
    s, m, p = ctx.s, ctx.mkn, ctx.platform
    result = L1TilingResult(k_al1=s.k_al1_full_size, k_bl1=s.k_bl1_full_size,
                            iterate_mn_order=IterateMNOrder.ITER_N_FST, m_al1=ctx.m_l0)

    n_bl1_max = ceil_div(ctx.single_co1 * m.n0, ctx.n_l0)
    for v in _make_sized_range(calc_comm_factor(n_bl1_max, n_bl1_max), ctx.n_l0):
        k_ci = s.k_al1_full_size // (ctx.cfg.kh * ctx.cfg.kw)
        fmap = _fm_l1(k_ci, s.hi_l1_min, ctx.cfg.wi, m, ctx.inner_batch)
        if fmap + result.k_bl1 * v * m.weight_dtype_size + s.bias_min_l1 <= p.l1_size:
            result.n_bl1 = v
        else:
            break
    if result.n_bl1 == 0:
        result.n_bl1 = ctx.n_l0
    return result


def _l1_mmode_none_full(ctx: L1MmodeCtx) -> L1TilingResult:
    s, m, p = ctx.s, ctx.mkn, ctx.platform
    result = L1TilingResult(iterate_mn_order=IterateMNOrder.ITER_M_FST,
                            m_al1=ctx.m_l0, n_bl1=ctx.n_l0)

    ka_range = _make_sized_range(calc_comm_factor(ctx.single_ci1, ctx.single_ci1), s.ci0_h_kw)
    kb_range = _make_sized_range(calc_comm_factor(ctx.single_ci1, ctx.single_ci1), s.ci0_h_kw)
    for i in range(min(len(ka_range), len(kb_range))):
        ka, kb = ka_range[i], kb_range[i]
        k_ci = ka // (ctx.cfg.kh * ctx.cfg.kw)
        fmap = _fm_l1(k_ci, s.hi_l1_min, ctx.cfg.wi, m, ctx.inner_batch)
        wt = kb * ctx.n_l0 * m.weight_dtype_size * DOUBLE_BUFFER_NUM
        if fmap + wt + s.bias_min_l1 <= p.l1_size:
            result.k_al1, result.k_bl1 = ka, kb
        else:
            break

    if result.k_al1 != s.k_al1_full_size and result.k_bl1 != s.k_bl1_full_size:
        result.pb_bl1 = DOUBLE_BUFFER_NUM
        for v in _make_sized_range(calc_comm_factor(ctx.single_ci1, ctx.single_ci1), s.ci0_h_kw):
            k_ci = v // (ctx.cfg.kh * ctx.cfg.kw)
            fmap = _fm_l1(k_ci, s.hi_l1_min, ctx.cfg.wi, m, ctx.inner_batch)
            wt = result.k_bl1 * ctx.n_l0 * m.weight_dtype_size * DOUBLE_BUFFER_NUM
            if fmap + wt + s.bias_min_l1 > p.l1_size:
                result.pb_bl1 = 1
                break
    return result


def l1_tiling_mmode(inp: L1MmodeInput) -> L1TilingResult:
    """Compute L1 tiling for M-mode. Dispatches to strategy-specific helpers."""
    m, p, cfg = inp.mkn, inp.platform, inp.cfg

    ctx = L1MmodeCtx(s=L1MmodeSizes(), mkn=m, platform=p, cfg=cfg,
                     single_m=inp.single_m, single_ci1=inp.single_ci1,
                     single_co1=inp.single_co1, m_l0=inp.m_l0,
                     n_l0=inp.n_l0, inner_batch=inp.inner_batch)
    ctx.s = _l1_mmode_sizes(ctx, inp.has_bias, inp.bias_dtype_size, inp.is_c04)
    s = ctx.s

    bias = s.bias_min_l1
    all_full = (s.fmap_full_l1 + s.weight_full_l1 + bias <= p.l1_size and
                inp.single_m <= LOAD3D_M_START_POS_LIMIT)
    bl1_possible = (s.weight_full_l1 + s.fmap_min_l1 + bias <= p.l1_size)
    al1_possible = (s.fmap_full_l1 + s.weight_min_l1 + bias <= p.l1_size and
                    inp.single_m <= LOAD3D_M_START_POS_LIMIT)
    w_dominant = s.fmap_full_l1 <= s.weight_full_l1 * p.ab_l1_mte2_bw_cof

    if all_full:
        return _l1_mmode_all_full(ctx)
    if w_dominant and bl1_possible:
        return _l1_mmode_bl1_full(ctx)
    if (not w_dominant and al1_possible) or al1_possible:
        return _l1_mmode_al1_full(ctx)

    k_al1_full_l1 = s.k_al1_full_size * s.hi_l1_min * cfg.wi * m.fmap_dtype_size * inp.inner_batch
    k_bl1_full_l1 = s.k_bl1_full_size * inp.n_l0 * m.weight_dtype_size
    kal1_ok = (k_al1_full_l1 + s.weight_min_l1 + bias <= p.l1_size)
    kbl1_ok = (s.fmap_min_l1 + k_bl1_full_l1 + bias <= p.l1_size)

    if kal1_ok and not kbl1_ok:
        return _l1_mmode_kal1_full(ctx)
    if kbl1_ok and not kal1_ok:
        return _l1_mmode_kbl1_full(ctx)
    if kal1_ok and kbl1_ok:
        return _l1_mmode_both_k_full(ctx)
    return _l1_mmode_none_full(ctx)


# ============================================================================
# L1 Tiling (HW-mode) -- strategy-based helpers
# ============================================================================

def _l1_hwmode_all_full(ctx: L1HwmodeCtx, max_sizes: L1HwmodeFullSizes) -> L1TilingResult:
    r = ctx.result
    r.k_al1 = max_sizes.k_al1_max
    r.k_bl1 = max_sizes.k_bl1_max
    r.m_al1 = max_sizes.ho_max
    r.wo_al1 = max_sizes.wo_max
    r.n_bl1 = max_sizes.n_bl1_max
    r.bias_full_load = True
    r.pb_al1 = 1
    r.pb_bl1 = 1
    return r


def _l1_hwmode_bl1_full(ctx: L1HwmodeCtx, max_sizes: L1HwmodeFullSizes) -> L1TilingResult:
    m, p, cfg = ctx.mkn, ctx.platform, ctx.cfg
    r = ctx.result
    r.k_bl1 = max_sizes.k_bl1_max
    r.n_bl1 = max_sizes.n_bl1_max
    r.bias_full_load = True
    r.m_al1 = ctx.ho_l0
    r.wo_al1 = ctx.wo_l0
    for v in calc_comm_factor(ctx.single_ci1, ctx.single_ci1):
        ka = v * m.k0
        hi = infer_hi_l1(ctx.ho_l0, cfg)
        wi = infer_wi_l1(ctx.wo_l0, cfg)
        a_size = hi * wi * ka * m.fmap_dtype_size * DOUBLE_BUFFER_NUM
        if a_size + max_sizes.b_l1_max + max_sizes.bias_max <= p.l1_size:
            r.k_al1 = ka
        else:
            break
    if r.k_al1 == 0:
        r.k_al1 = m.k0
    r.pb_bl1 = 1
    r.iterate_mn_order = IterateMNOrder.ITER_M_FST
    return r


def _l1_hwmode_al1_full(ctx: L1HwmodeCtx, max_sizes: L1HwmodeFullSizes) -> L1TilingResult:
    m, p, cfg = ctx.mkn, ctx.platform, ctx.cfg
    ci0_h_kw = cfg.kh * cfg.kw * m.k0
    r = ctx.result
    r.k_al1 = max_sizes.k_al1_max
    r.m_al1 = max_sizes.ho_max
    r.wo_al1 = max_sizes.wo_max
    for v in calc_comm_factor(ctx.single_ci1, ctx.single_ci1):
        kb = v * ci0_h_kw
        if ctx.is_c04:
            kb = align_b(C04_CIN_SIZE * cfg.kh * cfg.kw, m.k0)
        if (max_sizes.a_l1_max + kb * ctx.n_l0 * m.weight_dtype_size * DOUBLE_BUFFER_NUM
                + max_sizes.bias_min <= p.l1_size):
            r.k_bl1 = kb
        else:
            break
    if r.k_bl1 == 0:
        r.k_bl1 = ci0_h_kw
    n_max = ceil_div(ctx.single_co1 * m.n0, ctx.n_l0)
    for v in calc_comm_factor(n_max, n_max):
        w_size = r.k_bl1 * (v * ctx.n_l0) * m.weight_dtype_size * DOUBLE_BUFFER_NUM
        if max_sizes.a_l1_max + w_size + max_sizes.bias_min <= p.l1_size:
            r.n_bl1 = v * ctx.n_l0
        else:
            break
    if r.n_bl1 == 0:
        r.n_bl1 = ctx.n_l0
    r.pb_al1 = 1
    r.iterate_mn_order = IterateMNOrder.ITER_N_FST
    return r


def _l1_hwmode_best_effort(ctx: L1HwmodeCtx, max_sizes: L1HwmodeFullSizes) -> L1TilingResult:
    m, p, cfg = ctx.mkn, ctx.platform, ctx.cfg
    ci0_h_kw = cfg.kh * cfg.kw * m.k0
    r = ctx.result
    r.k_al1 = m.k0
    r.k_bl1 = ci0_h_kw
    r.m_al1 = ctx.ho_l0
    r.wo_al1 = ctx.wo_l0
    r.n_bl1 = ctx.n_l0
    r.pb_al1 = DOUBLE_BUFFER_NUM
    r.pb_bl1 = DOUBLE_BUFFER_NUM
    for v in calc_comm_factor(ctx.single_ci1, ctx.single_ci1):
        ka, kb = v * m.k0, v * ci0_h_kw
        hi = infer_hi_l1(ctx.ho_l0, cfg)
        wi = infer_wi_l1(ctx.wo_l0, cfg)
        a_size = hi * wi * ka * m.fmap_dtype_size * DOUBLE_BUFFER_NUM
        w_size = kb * ctx.n_l0 * m.weight_dtype_size * DOUBLE_BUFFER_NUM
        if a_size + w_size + max_sizes.bias_min <= p.l1_size:
            r.k_al1, r.k_bl1 = ka, kb
        else:
            break
    return r


def _l1_hwmode_compute_sizes(inp: L1HwmodeInput) -> Tuple[L1HwmodeFullSizes, int]:
    """Compute max/min sizes for HW-mode L1 tiling."""
    m, cfg = inp.mkn, inp.cfg
    ci0_h_kw = cfg.kh * cfg.kw * m.k0

    if inp.is_c04:
        k_al1_max = 1
        k_bl1_max = align_b(C04_CIN_SIZE * cfg.kh * cfg.kw, m.k0)
    else:
        k_al1_max = inp.single_ci1 * m.k0
        k_bl1_max = inp.single_ci1 * ci0_h_kw

    n_bl1_max = inp.single_co1 * m.n0
    ho_max = inp.single_ho
    wo_max = align_b(inp.single_wo, m.m0)

    hi_max = infer_hi_l1(ho_max, cfg)
    wi_max = infer_wi_l1(wo_max, cfg)
    a_l1_max = hi_max * wi_max * k_al1_max * m.fmap_dtype_size * DOUBLE_BUFFER_NUM
    b_l1_max = k_bl1_max * n_bl1_max * m.weight_dtype_size * DOUBLE_BUFFER_NUM
    bias_max = inp.single_co1 * m.n0 * inp.bias_dtype_size if inp.has_bias else 0

    ho_l1_min = min(inp.single_ho, inp.ho_l0)
    wo_l1_min = min(align_b(inp.single_wo, m.m0), inp.wo_l0)
    hi_min = infer_hi_l1(ho_l1_min, cfg)
    wi_min = infer_wi_l1(wo_l1_min, cfg)
    a_l1_min = hi_min * wi_min * m.k0 * m.fmap_dtype_size * DOUBLE_BUFFER_NUM
    b_l1_min = ci0_h_kw * inp.n_l0 * m.weight_dtype_size * DOUBLE_BUFFER_NUM
    bias_min = inp.n_l0 * inp.bias_dtype_size if inp.has_bias else 0

    sizes = L1HwmodeFullSizes(k_al1_max=k_al1_max, k_bl1_max=k_bl1_max,
                              n_bl1_max=n_bl1_max, ho_max=ho_max, wo_max=wo_max,
                              a_l1_max=a_l1_max, b_l1_max=b_l1_max, bias_max=bias_max,
                              a_l1_min=a_l1_min, b_l1_min=b_l1_min, bias_min=bias_min)
    return sizes, ci0_h_kw


def l1_hw_tiling(inp: L1HwmodeInput) -> L1TilingResult:
    """Compute L1 tiling for HW-mode. Dispatches to strategy-specific helpers."""
    sizes, _ = _l1_hwmode_compute_sizes(inp)
    p = inp.platform

    result = L1TilingResult()
    ctx = L1HwmodeCtx(mkn=inp.mkn, platform=p, cfg=inp.cfg, result=result,
                      ho_l0=inp.ho_l0, wo_l0=inp.wo_l0, n_l0=inp.n_l0,
                      single_ci1=inp.single_ci1, single_co1=inp.single_co1,
                      is_c04=inp.is_c04)

    if sizes.a_l1_max + sizes.b_l1_max + sizes.bias_min <= p.l1_size:
        return _l1_hwmode_all_full(ctx, sizes)
    if sizes.a_l1_min + sizes.b_l1_max + sizes.bias_max <= p.l1_size:
        return _l1_hwmode_bl1_full(ctx, sizes)
    if sizes.a_l1_max + sizes.b_l1_min + sizes.bias_min <= p.l1_size:
        return _l1_hwmode_al1_full(ctx, sizes)
    return _l1_hwmode_best_effort(ctx, sizes)


# ============================================================================
# NumBlocks Decision (M-split / HW-split)
# ============================================================================

def conv_num_blocks_factor_mix(org_dim: int, input_range: List[int],
                                mix_range: List[int]) -> List[int]:
    tmp = [v for v in mix_range if v <= org_dim]
    return sorted(set(input_range) | set(tmp))


def calc_total_cost_m_split(inp: CostCalcInput) -> int:
    bd, md, nd, gd = inp.dims
    ci1 = ceil_div(inp.cur_ci, inp.k0)
    co1 = ceil_div(inp.cur_co, inp.n0)
    m1 = ceil_div(align_b(inp.ho * inp.wo, inp.m0), md)

    load_fm = ceil_div(inp.batch, bd) * ceil_div(inp.cur_groups, gd) * m1 * ci1 * inp.k0
    load_wt = ceil_div(inp.cur_groups, gd) * ci1 * inp.kh * inp.kw * inp.k0 * ceil_div(inp.batch, bd)
    if not inp.opt_group:
        load_wt *= ceil_div(co1 * inp.n0, nd)
    load_out = (ceil_div(inp.batch, bd) * ceil_div(inp.cur_groups, gd) *
                ceil_div(co1 * inp.n0, nd) * m1)
    cube_cost = (ceil_div(inp.batch, bd) * ceil_div(inp.cur_groups, gd) *
                 ceil_div(co1, nd) * ci1 * inp.kh * inp.kw * m1)
    bw = 1 if (inp.opt_group or inp.cur_groups > 1) else 4
    return (load_fm + load_wt * bw + load_out) // 128 + cube_cost


def _num_blocks_ranges(aicore_num: int, batch: int, cur_co: int,
                        cur_groups: int, n0: int) -> Tuple[List[int], List[int], List[int]]:
    batch_range = calc_comm_factor(batch, aicore_num)
    if batch >= 2 * aicore_num:
        batch_range = calc_comm_factor(aicore_num, aicore_num)
    else:
        batch_range = conv_num_blocks_factor_mix(
            batch, batch_range, calc_comm_factor(aicore_num, aicore_num))

    n_range = calc_comm_factor(ceil_div(cur_co, n0), aicore_num)
    n_range.append(1)

    group_range = calc_comm_factor(cur_groups, aicore_num)
    if cur_groups >= 2 * aicore_num:
        group_range = calc_comm_factor(aicore_num, aicore_num)
    else:
        group_range = conv_num_blocks_factor_mix(
            cur_groups, group_range, calc_comm_factor(aicore_num, aicore_num))
    group_range.append(1)

    return batch_range, n_range, group_range


def num_blocks_m_split(inp: NumBlocksMInput) -> NumBlocksResult:
    """Exact backtracking numBlocks for M-split."""
    aicore_num = inp.aicore_num or 32
    batch_range, n_range, group_range = _num_blocks_ranges(
        aicore_num, inp.batch, inp.cur_co, inp.cur_groups, inp.n0)

    m1 = ceil_div(align_b(inp.ho * inp.wo, inp.m0), inp.m0)
    m_range = calc_comm_factor(m1, aicore_num)
    m_range = conv_num_blocks_factor_mix(m1, m_range, calc_comm_factor(aicore_num, aicore_num))

    nb = NumBlocksResult()
    cost_inp = CostCalcInput(dims=(1, 1, 1, 1), batch=inp.batch, ho=inp.ho, wo=inp.wo,
                             cur_ci=inp.cur_ci, cur_co=inp.cur_co, cur_groups=inp.cur_groups,
                             kh=inp.kh, kw=inp.kw, m0=inp.m0, k0=inp.k0, n0=inp.n0,
                             opt_group=inp.opt_group)
    min_cost = calc_total_cost_m_split(cost_inp)

    for b, m, n, g in itertools.product(batch_range, m_range, n_range, group_range):
        total = b * m * n * g
        if total > aicore_num:
            continue
        cost_inp.dims = (b, m, n, g)
        cost = calc_total_cost_m_split(cost_inp)
        if cost < min_cost:
            min_cost = cost
            nb.batch_dim, nb.m_dim, nb.n_dim, nb.group_dim = b, m, n, g
    return nb


def num_blocks_hw_split(inp: NumBlocksHwInput) -> NumBlocksResult:
    """NumBlocks for HW-split."""
    aicore_num = inp.aicore_num or 32
    batch_range, n_range, group_range = _num_blocks_ranges(
        aicore_num, inp.batch, inp.cur_co, inp.cur_groups, inp.n0)

    ho_range = calc_comm_factor(inp.ho, aicore_num)
    ho_range = conv_num_blocks_factor_mix(inp.ho, ho_range, calc_comm_factor(aicore_num, aicore_num))
    wo_range = calc_comm_factor(inp.wo, aicore_num)
    wo_range = conv_num_blocks_factor_mix(inp.wo, wo_range, calc_comm_factor(aicore_num, aicore_num))

    nb = NumBlocksResult()
    best_total, best_waste = 1, float('inf')
    for b, h, w, n, g in itertools.product(batch_range, ho_range, wo_range, n_range, group_range):
        total = b * h * w * n * g
        if total > aicore_num:
            continue
        waste = aicore_num - total
        if total > best_total or (total == best_total and waste < best_waste):
            best_total, best_waste = total, waste
            nb.batch_dim, nb.ho_dim, nb.wo_dim, nb.n_dim, nb.group_dim = b, h, w, n, g
    return nb


# ============================================================================
# TilingKey Computation
# ============================================================================

FMP_TILING_NAMES = {0: "FULLLOAD_AL1", 1: "ONLY_M_FULLLOAD", 2: "OTHER"}
WT_TILING_NAMES = {0: "FULLLOAD_BL1", 1: "ONLY_N_FULLLOAD", 2: "OTHER"}
L1_PP_NAMES = {0: "ALL_CLOSE", 1: "AL1/BL0_OPEN", 2: "BL1/AL0_OPEN", 3: "ALL_OPEN"}
L0_PP_NAMES = {0: "ALL_CLOSE", 1: "AL0_OPEN", 2: "BL0_OPEN", 3: "ALL_OPEN"}


def _compute_fmp_tiling_mmode(kp: TilingKeyParams, t: TilingData,
                               k0: int, ci_kh_kw: int) -> None:
    k_al1_full = (t.k_al1 == ci_kh_kw)
    m1_full = ((t.inner_batch == 1 and t.single_core_ho <= t.ho_l1) or
               (t.inner_batch == t.single_core_batch))
    if k_al1_full and m1_full:
        kp.fmp_tiling = 0
    elif not k_al1_full and m1_full and (t.ho_l1 == t.ho_l0):
        kp.fmp_tiling = 1
    else:
        kp.fmp_tiling = 2


def _compute_fmp_tiling_hwmode(kp: TilingKeyParams, t: TilingData,
                                m0: int, ci_kh_kw: int) -> None:
    k_al1_full = (t.k_al1 == ci_kh_kw)
    ho_full = (t.single_core_ho <= t.ho_l1)
    wo_full = (t.single_core_wo <= t.wo_l1 and
               not (ceil_div(t.single_core_wo, t.wo_l0) > 1 and
                    t.single_core_wo % m0 > 0 and t.ho_l0 > 1))
    l1_full = ho_full and wo_full
    l0_full = (t.ho_l1 == t.ho_l0) and (t.wo_l1 == t.wo_l0)
    if k_al1_full and l1_full:
        kp.fmp_tiling = 0
    elif not k_al1_full and l1_full and l0_full:
        kp.fmp_tiling = 1
    else:
        kp.fmp_tiling = 2


def _compute_weight_tiling(kp: TilingKeyParams, td: Tuple[TilingData, NumBlocksResult],
                            cur_co: int, n0: int, ci_kh_kw: int) -> None:
    t, nb = td
    sc_n_size = align_b(ceil_div(align_b(cur_co, n0), nb.n_dim), n0)
    if (t.k_bl1 == ci_kh_kw) and (t.n_bl1 == sc_n_size):
        kp.weight_tiling = 0
    elif t.k_bl1 != ci_kh_kw and t.n_l0 == sc_n_size:
        kp.weight_tiling = 1
    else:
        kp.weight_tiling = 2


def _compute_pingpong_and_flags(kp: TilingKeyParams, t: TilingData,
                                 groups_active: bool, m_split: bool) -> None:
    l1_pb = (t.p_buffer_flag & 0x18) >> 3
    kp.l1_ping_pong = 3 if (groups_active and l1_pb in [2, 3]) else (0 if groups_active else l1_pb)
    kp.l0_ping_pong = t.p_buffer_flag & 0x03
    kp.output_order = 1 if m_split else 0
    kp.iter_order = t.iterate_mn_order
    if t.inner_batch > 1:
        kp.inner_batch = 1 if (t.kernel_h == 1 and t.kernel_w == 1) else 2
    else:
        kp.inner_batch = 0
    kp.weight_ub_trans = 1 if (t.b_ub_n_step > 0 and t.b_ub_k_step > 0) else 0
    kp.fmap_copy_mode = 1 if (t.kh_ub > 0 and t.kw_ub > 0) else 0


def compute_tiling_key(ctx: TilingKeyCtx) -> TilingKeyParams:
    """Compute tiling key from single-core tiling result."""
    kp = TilingKeyParams()
    n0, k0 = 16, MKN_TABLE.get(ctx.dtype, (16, 16, 16))[1]
    m0 = MKN_TABLE.get(ctx.dtype, (16, 16, 16))[0]
    ci_kh_kw = ceil_div(ctx.ci, k0) * ctx.kh * ctx.kw * k0

    if not ctx.groups_active:
        if ctx.m_split:
            _compute_fmp_tiling_mmode(kp, ctx.t, k0, ci_kh_kw)
        else:
            _compute_fmp_tiling_hwmode(kp, ctx.t, m0, ci_kh_kw)
        _compute_weight_tiling(kp, (ctx.t, ctx.nb), ctx.cur_co, n0, ci_kh_kw)

    _compute_pingpong_and_flags(kp, ctx.t, ctx.groups_active, ctx.m_split)
    kp.group_type = 2 if ctx.opt_group else (1 if ctx.groups_active else 0)
    return kp


# ============================================================================
# Core tiling engine
# ============================================================================

def _compute_opt_group(ctx: OptGroupCtx) -> Tuple[bool, int, int, int, int]:
    """Compute optimal group parameters."""
    if ctx.groups <= 1:
        return False, ctx.ci, ctx.co, 0, 0
    m = ctx.mkn
    ci_pg, co_pg = ctx.ci // ctx.groups, ctx.co // ctx.groups
    enlarge_val = min(lcm(lcm(ci_pg, m.k0) // ci_pg,
                          lcm(co_pg, m.n0) // co_pg), ctx.groups)
    group_opt_val = ceil_div(ctx.groups, enlarge_val)
    if enlarge_val > 1:
        wdt_size = DTYPE_SIZE_TAB.get(ctx.dtype, 2)
        ub_required = (align_b(ci_pg * enlarge_val, m.k0) * ctx.kh * ctx.kw *
                       align_b(co_pg * enlarge_val, m.n0) * 2 * wdt_size)
        if ub_required <= ctx.platform.ub_size - 256:
            return True, ci_pg * enlarge_val, co_pg * enlarge_val, int(enlarge_val), group_opt_val
    return False, ci_pg, co_pg, 0, ctx.groups


@dataclass
class SingleCoreInput:
    nb: NumBlocksResult
    batch: int
    ho: int
    wo: int
    m_size: int
    m0: int
    n0: int
    cur_co: int
    m_split: bool


def _compute_single_core_shapes(inp: SingleCoreInput) -> SingleCoreShapes:
    sc = SingleCoreShapes()
    sc.co = align_b(ceil_div(align_b(inp.cur_co, inp.n0), inp.nb.n_dim), inp.n0)
    sc.batch = ceil_div(inp.batch, inp.nb.batch_dim)
    if inp.m_split:
        sc.m = ceil_div(align_b(inp.m_size, inp.m0), inp.nb.m_dim)
    else:
        sc.ho = ceil_div(inp.ho, inp.nb.ho_dim)
        sc.wo = ceil_div(inp.wo, inp.nb.wo_dim)
    return sc


@dataclass
class GroupParamsInput:
    opt_group: bool
    groups_active: bool
    enlarge_val: int
    cur_co: int
    group_opt_val: int
    nb: NumBlocksResult


def _compute_group_params(inp: GroupParamsInput) -> Tuple[int, int, int]:
    if inp.opt_group:
        return int(inp.enlarge_val), inp.enlarge_val, ceil_div(inp.group_opt_val, inp.nb.group_dim)
    if inp.groups_active:
        return 1, 1, ceil_div(inp.cur_co, inp.nb.group_dim)
    return 0, 0, 0


def _init_tiling_data(ctx: InitTilingDataCtx) -> TilingData:
    t = TilingData()
    t.org_hi = ctx.cfg.hi
    t.org_wi = ctx.cfg.wi
    t.org_ho = ctx.ho
    t.org_wo = ctx.wo
    t.org_hix_wi = ctx.cfg.hi * ctx.cfg.wi
    t.org_ci = ctx.cfg.ci
    t.org_co = ctx.cfg.co
    t.kernel_h = ctx.cfg.kh
    t.kernel_w = ctx.cfg.kw
    t.stride_h = ctx.cfg.stride_h
    t.stride_w = ctx.cfg.stride_w
    t.dilation_h = ctx.cfg.dilation_h
    t.dilation_w = ctx.cfg.dilation_w
    t.pad_top = ctx.cfg.pad_top
    t.pad_bottom = ctx.cfg.pad_bottom
    t.pad_left = ctx.cfg.pad_left
    t.pad_right = ctx.cfg.pad_right
    t.groups = ctx.cfg.groups
    t.has_bias = 1
    t.single_core_batch = ctx.sc.batch
    t.single_core_co = ctx.sc.co
    t.single_core_ci = ctx.sc.ci
    t.single_core_ho = ctx.sc.ho
    t.single_core_wo = ctx.sc.wo
    if ctx.groups_active:
        t.enlarge = ctx.enlarge
        t.single_core_groups = ctx.s_groups
        t.single_core_group_opt = ctx.s_group_opt
    return t


def _fill_tiling_data_mmode(t: TilingData, l1_res: L1TilingResult, l0: L0Result) -> None:
    t.ho_l1 = l1_res.m_al1
    t.wo_l1 = 0
    t.k_al1 = l1_res.k_al1
    t.k_bl1 = l1_res.k_bl1
    t.n_bl1 = l1_res.n_bl1
    t.ho_l0 = l0.m_l0
    t.wo_l0 = 0
    t.k_l0 = l0.k_l0
    t.n_l0 = l0.n_l0
    t.inner_batch = l0.inner_batch
    t.iterate_mn_order = int(l1_res.iterate_mn_order)
    t.bias_full_load_flag = 1 if l1_res.bias_full_load else 0
    t.fixp_params_full_load_flag = 1 if l1_res.fixp_full_load else 0


def _fill_tiling_data_hwmode(t: TilingData, l1_res: L1TilingResult, l0: L0Result) -> None:
    t.ho_l1 = l1_res.m_al1
    t.wo_l1 = l1_res.wo_al1
    t.k_al1 = l1_res.k_al1
    t.k_bl1 = l1_res.k_bl1
    t.n_bl1 = l1_res.n_bl1
    t.ho_l0 = l0.ho_l0
    t.wo_l0 = l0.wo_l0
    t.k_l0 = l0.k_l0
    t.n_l0 = l0.n_l0
    t.inner_batch = 1
    t.iterate_mn_order = int(l1_res.iterate_mn_order)
    t.bias_full_load_flag = 1 if l1_res.bias_full_load else 0
    t.fixp_params_full_load_flag = 1 if l1_res.fixp_full_load else 0


def _safe_div(a: int, b: int) -> int:
    return a // b if b > 0 else 0


def _calc_tail(val: int, divisor: int) -> int:
    r = val % divisor
    return r if r != 0 else divisor


def _compute_cin_params(t: TilingData, khw: int, single_core_ci: int) -> None:
    t.cin_a_in_core = _safe_div(t.k_al1, khw)
    t.cin_a_tail_in_core = _safe_div(_calc_tail(single_core_ci * khw, t.k_al1), khw)
    t.cin_b_in_core = _safe_div(t.k_bl1, khw)
    t.cin_b_tail_in_core = _safe_div(_calc_tail(single_core_ci * khw, t.k_bl1), khw)


def _compute_scalar_params(ctx: ScalarParamsCtx) -> None:
    t = ctx.t
    khw = t.kernel_h * t.kernel_w
    t.kernel_hx_kernel_w = khw
    t.kernel_hx_kernel_wx_kernel_d = khw
    _compute_cin_params(t, khw, ctx.single_core_ci)

    if ctx.m_split:
        t.m_step = align_b(t.ho_l0, ctx.m0)
    else:
        t.m_step = align_b(t.ho_l0 * t.wo_l0, ctx.m0)
    t.fmap_k_stride = _safe_div(t.m_step, ctx.m0)

    t.n_step = ceil_div(t.n_l0, ctx.n0)
    t.k_step = _safe_div(t.k_l0, ctx.n0)
    t.weight_k_stride = ceil_div(t.n_bl1, ctx.n0)
    t.cin_offset_block_in_gm = t.cin_a_in_core * t.org_hix_wi
    t.cout_offset_block = (ctx.ci // ctx.groups) * khw if ctx.groups > 0 else ctx.ci * khw
    t.n_l1_div_block_size = _safe_div(t.n_bl1, ctx.n0)
    t.multi_n_bl1 = ceil_div(t.n_bl1, t.n_l0)


def _compute_buffer_flag(pb_al1: int, pb_bl1: int) -> int:
    pb_al0 = DOUBLE_BUFFER_NUM
    pb_bl0 = DOUBLE_BUFFER_NUM
    flag = 0
    flag |= (pb_bl0 == DOUBLE_BUFFER_NUM and 1 or 0)
    flag = (flag << 1) | (pb_al0 == DOUBLE_BUFFER_NUM and 1 or 0)
    flag = (flag << 1) | (pb_bl1 == DOUBLE_BUFFER_NUM and 1 or 0)
    flag = (flag << 1) | (pb_al1 == DOUBLE_BUFFER_NUM and 1 or 0)
    flag = (flag << 1) | 0
    return flag


def _compute_al1_space(t: TilingData, cfg: Conv2DConfig,
                        fmap_dtype_size: int, m_split: bool) -> int:
    if m_split:
        a_l1_space = t.cin_a_in_core * cfg.hi * cfg.wi * fmap_dtype_size
    else:
        hi_max = min((t.ho_l1 - 1) * cfg.stride_h + (cfg.kh - 1) * cfg.dilation_h + 1, cfg.hi)
        wi_max = min((t.wo_l1 - 1) * cfg.stride_w + (cfg.kw - 1) * cfg.dilation_w + 1, cfg.wi)
        a_l1_space = t.cin_a_in_core * hi_max * wi_max * fmap_dtype_size
    return align_b(a_l1_space, C0_SIZE) * t.inner_batch


def _run_l0_l1_tiling(ctx: RunL0L1Ctx) -> Tuple[int, int]:
    """Run L0 and L1 tiling and fill TilingData."""
    m = ctx.mkn
    bias_ds = 4 if ctx.dtype in [ConvDtype.INT8, ConvDtype.HIFLOAT8] else m.fmap_dtype_size
    sc_ci1 = ceil_div(ctx.sc.ci, m.k0)
    sc_co1 = ceil_div(ctx.sc.co, m.n0)
    l0_ctx = L0TilingCtx(mkn=m, platform=ctx.platform, cfg=ctx.cfg)

    if ctx.m_split:
        l0 = l0_tiling_mmode(ctx.sc.m, sc_ci1, sc_co1, l0_ctx)
        l1_inp = L1MmodeInput(single_m=ctx.sc.m, single_ci=ctx.sc.ci,
                              single_ci1=sc_ci1, single_co1=sc_co1,
                              single_batch=ctx.sc.batch, mkn=m, platform=ctx.platform,
                              cfg=ctx.cfg, m_l0=l0.m_l0, n_l0=l0.n_l0, k_l0=l0.k_l0,
                              inner_batch=l0.inner_batch,
                              has_bias=True, bias_dtype_size=bias_ds)
        l1_res = l1_tiling_mmode(l1_inp)
        _fill_tiling_data_mmode(ctx.t, l1_res, l0)
        return l1_res.pb_al1, l1_res.pb_bl1
    else:
        l0 = l0_hw_tiling(ctx.sc.ho, ctx.sc.wo, sc_co1, l0_ctx)
        l1_inp = L1HwmodeInput(single_ho=ctx.sc.ho, single_wo=ctx.sc.wo,
                               single_ci=ctx.sc.ci, single_ci1=sc_ci1, single_co1=sc_co1,
                               mkn=m, platform=ctx.platform, cfg=ctx.cfg,
                               ho_l0=l0.ho_l0, wo_l0=l0.wo_l0, n_l0=l0.n_l0,
                               has_bias=True, bias_dtype_size=bias_ds)
        l1_res = l1_hw_tiling(l1_inp)
        _fill_tiling_data_hwmode(ctx.t, l1_res, l0)
        return l1_res.pb_al1, l1_res.pb_bl1


@dataclass
class _SetupResult:
    """Return values from _setup_tiling."""
    mkn: MKNConfig
    opt_group: bool
    cur_co: int
    m_split: bool
    groups_active: bool
    nb: NumBlocksResult
    sc: SingleCoreShapes
    enlarge: int
    s_groups: int
    s_group_opt: int


def _setup_tiling(inp: ComputeTilingInput, out_dims: Tuple[int, int],
                  dtype: ConvDtype, mkn_dims: Tuple[int, int, int]) -> _SetupResult:
    """Setup shapes, groups, numBlocks, and single-core params for tiling."""
    cfg = inp.cfg
    ho, wo = out_dims
    m0, k0, n0 = mkn_dims
    m_size = ho * wo
    groups_active = cfg.groups > 1
    mkn = MKNConfig(m0=m0, k0=k0, n0=n0,
                    fmap_dtype_size=DTYPE_SIZE_TAB.get(dtype, 2),
                    weight_dtype_size=DTYPE_SIZE_TAB.get(dtype, 2))

    opt_ctx = OptGroupCtx(groups=cfg.groups, ci=cfg.ci, co=cfg.co, mkn=mkn,
                          kh=cfg.kh, kw=cfg.kw, dtype=dtype, platform=inp.platform)
    opt_group, cur_ci, cur_co, enlarge_val, group_opt_val = _compute_opt_group(opt_ctx)

    m_split = inp.output_order == OutputOrder.M
    nb_groups = group_opt_val if opt_group else (cfg.groups if groups_active else 1)
    nb_inp = dict(aicore_num=inp.aicore_num, batch=cfg.batch, ho=ho, wo=wo,
                  cur_ci=cur_ci, cur_co=cur_co, cur_groups=nb_groups,
                  kh=cfg.kh, kw=cfg.kw, n0=n0, opt_group=opt_group)
    if m_split:
        nb = num_blocks_m_split(NumBlocksMInput(m0=m0, k0=k0, **nb_inp))
    else:
        nb = num_blocks_hw_split(NumBlocksHwInput(**nb_inp))

    sc = _compute_single_core_shapes(SingleCoreInput(
        nb=nb, batch=cfg.batch, ho=ho, wo=wo, m_size=m_size,
        m0=m0, n0=n0, cur_co=cur_co, m_split=m_split))
    sc.ci = cur_ci
    sc.ho, sc.wo = (sc.m, 0) if m_split else (sc.ho, sc.wo)
    sc.m = 0 if not m_split else sc.m

    enlarge, s_groups, s_group_opt = _compute_group_params(GroupParamsInput(
        opt_group=opt_group, groups_active=groups_active, enlarge_val=enlarge_val,
        cur_co=cur_co, group_opt_val=group_opt_val, nb=nb))
    return _SetupResult(mkn=mkn, opt_group=opt_group, cur_co=cur_co,
                        m_split=m_split, groups_active=groups_active, nb=nb,
                        sc=sc, enlarge=enlarge, s_groups=s_groups,
                        s_group_opt=s_group_opt)


def compute_tiling(inp: ComputeTilingInput) -> Tuple[TilingData, NumBlocksResult,
                                                      TilingKeyParams, bool, str]:
    """Main tiling computation entry point."""
    cfg = inp.cfg
    ho, wo = cfg.calc_ho(), cfg.calc_wo()
    if ho <= 0 or wo <= 0:
        return None, None, None, False, f"Invalid output size: ho={ho}, wo={wo}"

    dtype = dtype_from_str(inp.dtype_str)
    m0, k0, n0 = MKN_TABLE.get(dtype, (16, 16, 16))
    s = _setup_tiling(inp, (ho, wo), dtype, (m0, k0, n0))

    t = _init_tiling_data(InitTilingDataCtx(
        cfg=cfg, ho=ho, wo=wo, sc=s.sc, groups_active=s.groups_active,
        enlarge=s.enlarge, s_groups=s.s_groups, s_group_opt=s.s_group_opt))

    run_ctx = RunL0L1Ctx(t=t, cfg=cfg, platform=inp.platform, mkn=s.mkn,
                         dtype=dtype, sc=s.sc, m_split=s.m_split, ho=ho, wo=wo)
    pb_al1, pb_bl1 = _run_l0_l1_tiling(run_ctx)

    _compute_scalar_params(ScalarParamsCtx(
        t=t, single_core_ci=s.sc.ci, ci=cfg.ci, groups=cfg.groups,
        m0=m0, n0=n0, m_split=s.m_split))

    t.p_buffer_flag = _compute_buffer_flag(pb_al1, pb_bl1)
    t.a_l1_space_size = _compute_al1_space(t, cfg, s.mkn.fmap_dtype_size, s.m_split)

    key_ctx = TilingKeyCtx(t=t, nb=s.nb, ci=cfg.ci, cur_co=s.cur_co,
                            kh=cfg.kh, kw=cfg.kw,
                           dtype=dtype, groups_active=s.groups_active,
                           opt_group=s.opt_group, m_split=s.m_split)
    kp = compute_tiling_key(key_ctx)
    return t, s.nb, kp, s.m_split, ""


# ============================================================================
# Output formatting
# ============================================================================

def _log_numblocks(nb: NumBlocksResult, m_split: bool) -> int:
    logger.info(f"\n  --- Multi-Core NumBlocks ---")
    if m_split:
        logger.info(f"  batchDim={nb.batch_dim}  mDim={nb.m_dim}  "
                    f"nDim={nb.n_dim}  groupDim={nb.group_dim}")
        total = nb.batch_dim * nb.m_dim * nb.n_dim * nb.group_dim
    else:
        logger.info(f"  batchDim={nb.batch_dim}  hoDim={nb.ho_dim}  "
                    f"woDim={nb.wo_dim}  nDim={nb.n_dim}  groupDim={nb.group_dim}")
        total = nb.batch_dim * nb.ho_dim * nb.wo_dim * nb.n_dim * nb.group_dim
    logger.info(f"  Total blocks: {total}")
    return total


def _log_single_core_shapes(t: TilingData, m_split: bool) -> None:
    logger.info(f"\n  --- Single-Core Shapes ---")
    if m_split:
        logger.info(f"  singleCoreM={t.single_core_ho}  singleCoreCo={t.single_core_co}  "
                    f"singleCoreBatch={t.single_core_batch}  singleCoreCi={t.single_core_ci}")
    else:
        logger.info(f"  singleCoreHo={t.single_core_ho}  singleCoreWo={t.single_core_wo}  "
                    f"singleCoreCo={t.single_core_co}  singleCoreBatch={t.single_core_batch}  "
                    f"singleCoreCi={t.single_core_ci}")


def _log_tiling_details(t: TilingData, groups_active: bool) -> None:
    logger.info(f"\n  --- Single-Core Tiling ---")
    logger.info(f"  L1:  hoL1={t.ho_l1}  woL1={t.wo_l1}  kAL1={t.k_al1}  "
                f"kBL1={t.k_bl1}  nBL1={t.n_bl1}")
    logger.info(f"  L0:  hoL0={t.ho_l0}  woL0={t.wo_l0}  kL0={t.k_l0}  nL0={t.n_l0}")
    logger.info(f"  M/K/N step:  mStep={t.m_step}  kStep={t.k_step}  "
                f"nStep={t.n_step}  innerBatch={t.inner_batch}")
    logger.info(f"  Buffer:  pBufferFlag={t.p_buffer_flag:#x}  aL1Space={t.a_l1_space_size}")
    logger.info(f"  Scalar:  cinA={t.cin_a_in_core}  cinATail={t.cin_a_tail_in_core}  "
                f"cinB={t.cin_b_in_core}  fmapKStr={t.fmap_k_stride}  "
                f"wtKStr={t.weight_k_stride}")
    if groups_active:
        logger.info(f"  Groups:  groups={t.groups}  enlarge={t.enlarge}  "
                    f"singleGroups={t.single_core_groups}  "
                    f"singleGroupOpt={t.single_core_group_opt}")


def _log_tiling_key(kp: TilingKeyParams) -> None:
    logger.info(f"\n  --- TilingKey ---")
    logger.info(f"  fmpTiling      ={kp.fmp_tiling} "
                f"({FMP_TILING_NAMES.get(kp.fmp_tiling, '?')})")
    logger.info(f"  weightTiling   ={kp.weight_tiling} "
                f"({WT_TILING_NAMES.get(kp.weight_tiling, '?')})")
    logger.info(f"  l1PingPong     ={kp.l1_ping_pong} "
                f"({L1_PP_NAMES.get(kp.l1_ping_pong, '?')})")
    logger.info(f"  l0PingPong     ={kp.l0_ping_pong} "
                f"({L0_PP_NAMES.get(kp.l0_ping_pong, '?')})")
    logger.info(f"  outputOrder    ={kp.output_order} "
                f"({'M-mode' if kp.output_order else 'HW-mode'})")
    logger.info(f"  iterOrder      ={kp.iter_order} "
                f"({'N_FIRST' if kp.iter_order else 'M_FIRST'})")
    group_names = {0: "NORMAL", 1: "ORI_GROUP", 2: "OPT_GROUP"}
    logger.info(f"  groupType      ={kp.group_type} "
                f"({group_names.get(kp.group_type, '?')})")
    logger.info(f"  weightUbTrans  ={kp.weight_ub_trans}  "
                f"fmapCopyMode={kp.fmap_copy_mode}")
    ib_names = {0: "SINGLE", 1: "1x1_MULTI", 2: "MULTI"}
    logger.info(f"  innerBatch     ={kp.inner_batch} "
                f"({ib_names.get(kp.inner_batch, '?')})")


@dataclass
class PrintTilingInput:
    t: TilingData
    nb: NumBlocksResult
    kp: TilingKeyParams
    m_split: bool
    cfg: Conv2DConfig
    ho: int
    wo: int
    dtype_str: str
    aicore_num: int
    opt_group: bool


def print_tiling_result(inp: PrintTilingInput):
    """Pretty-print tiling results."""
    cfg = inp.cfg
    dtype = dtype_from_str(inp.dtype_str)
    m0, k0, n0 = MKN_TABLE.get(dtype, (16, 16, 16))
    groups_active = cfg.groups > 1

    logger.info("=" * 70)
    logger.info("  Conv2D Formula-Based Tiling Result")
    logger.info("=" * 70)
    logger.info(f"  Shape:  N={cfg.batch} C={cfg.ci} H={cfg.hi} W={cfg.wi}  "
                f"Co={cfg.co} Kh={cfg.kh} Kw={cfg.kw}  Ho={inp.ho} Wo={inp.wo}")
    logger.info(f"  Attrs:  stride={cfg.stride_h}x{cfg.stride_w} dil=1x1 "
                f"groups={cfg.groups}  dtype={inp.dtype_str}  cores={inp.aicore_num}")
    group_label = 'OPT_GROUP' if inp.opt_group else ('ORI_GROUP' if groups_active else 'NORMAL')
    logger.info(f"  Group:  {group_label}")
    logger.info(f"  Split:  {'M-split' if inp.m_split else 'HW-split'}  "
                f"m0={m0} n0={n0} k0={k0}")

    total = _log_numblocks(inp.nb, inp.m_split)
    _log_single_core_shapes(inp.t, inp.m_split)
    _log_tiling_details(inp.t, groups_active)
    _log_tiling_key(inp.kp)

    logger.info(f"\n  BlockDim (for kernel dispatch): {total}")
    logger.info("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conv2D Formula-Based Tiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal convolution (ResNet50 first layer)
  python conv_tiling.py -n 1 -c 64 -hi 56 -wi 56 -co 64 -kh 3 -kw 3

  # Depthwise convolution
  python conv_tiling.py -n 1 -c 64 -hi 56 -wi 56 -co 64 -kh 3 -kw 3 -g 64

  # Different data types
  python conv_tiling.py -n 1 -c 64 -h 56 -w 56 -co 128 -kh 3 -kw 3 -t fp16
  python conv_tiling.py -n 1 -c 64 -h 56 -w 56 -co 128 -kh 3 -kw 3 -t fp32

  # HW-mode
  python conv_tiling.py -n 1 -c 64 -hi 56 -wi 56 -co 64 -kh 3 -kw 3 -m 0
        """,
    )
    parser.add_argument("-n", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("-c", type=int, default=64, help="Input channels (default: 64)")
    parser.add_argument("-hi", type=int, default=56, help="Input height (default: 56)")
    parser.add_argument("-wi", type=int, default=56, help="Input width (default: 56)")
    parser.add_argument("-co", type=int, default=64, help="Output channels (default: 64)")
    parser.add_argument("-kh", type=int, default=3, help="Kernel height (default: 3)")
    parser.add_argument("-kw", type=int, default=3, help="Kernel width (default: 3)")
    parser.add_argument("-sh", type=int, default=1, help="Stride H (default: 1)")
    parser.add_argument("-sw", type=int, default=1, help="Stride W (default: 1)")
    parser.add_argument("-pt", type=int, default=1, help="Pad top (default: 1)")
    parser.add_argument("-pb", type=int, default=1, help="Pad bottom (default: 1)")
    parser.add_argument("-pl", type=int, default=1, help="Pad left (default: 1)")
    parser.add_argument("-pr", type=int, default=1, help="Pad right (default: 1)")
    parser.add_argument("-dh", type=int, default=1, help="Dilation H (default: 1)")
    parser.add_argument("-dw", type=int, default=1, help="Dilation W (default: 1)")
    parser.add_argument("-g", type=int, default=1, help="Groups (default: 1)")
    parser.add_argument("-t", type=str, default="fp16",
                        help="Data type: fp16/fp32/bf16/int8 (default: fp16)")
    parser.add_argument("-m", type=int, default=1,
                        help="Output order: 0=HW-mode, 1=M-mode (default: 1)")
    parser.add_argument("--cores", type=int, default=32,
                        help="Number of AI cores (default: 32)")
    return parser


def _validate_tiling(t: TilingData) -> int:
    checks = [
        (t.n_l0 == 0, "nL0 is 0"), (t.k_l0 == 0, "kL0 is 0"),
        (t.n_bl1 == 0, "nBL1 is 0"), (t.k_al1 == 0, "kAL1 is 0"),
        (t.k_bl1 == 0, "kBL1 is 0"),
    ]
    errors = [msg for cond, msg in checks if cond]
    if errors:
        logger.warning("Tiling issues detected: %s", ", ".join(errors))
        return 1
    return 0


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = Conv2DConfig(batch=args.n, ci=args.c, hi=args.hi, wi=args.wi, co=args.co,
                       kh=args.kh, kw=args.kw,
                       stride_h=args.sh, stride_w=args.sw,
                       pad_top=args.pt, pad_bottom=args.pb,
                       pad_left=args.pl, pad_right=args.pr,
                       dilation_h=args.dh, dilation_w=args.dw,
                       groups=args.g)
    ho, wo = cfg.calc_ho(), cfg.calc_wo()

    if ho <= 0 or wo <= 0:
        logger.error("Invalid output size ho=%d, wo=%d", ho, wo)
        sys.exit(1)

    inp = ComputeTilingInput(cfg=cfg, dtype_str=args.t,
                             output_order=args.m, aicore_num=args.cores)
    t, nb, kp, m_split, err = compute_tiling(inp)

    if err:
        logger.error("%s", err)
        sys.exit(1)

    opt_group = (kp.group_type == 2)
    print_tiling_result(PrintTilingInput(t=t, nb=nb, kp=kp, m_split=m_split,
                                          cfg=cfg, ho=ho, wo=wo, dtype_str=args.t,
                                          aicore_num=args.cores, opt_group=opt_group))
    return _validate_tiling(t)


if __name__ == "__main__":
    sys.exit(main())
