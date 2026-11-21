# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU references for W8A16 dequantization placements.

exact_reference computes the mathematical FP64 result. prescale_reference consumes FP16-rounded dequantized weights;
fixpipe_reference uses truncated fp19 scales; fixpipe_rne_reference uses scales rounded to fp19 with ties-to-even;
postscale_reference and unscaled_reference are distinct comparison controls. Wrong-layer references are rejected by the
same numerical criterion in test.py.

error_scale returns S=sum(abs(X)*abs(W_q))*abs(scale). The tests use TOL_S=K*2^-24 against the selected own-layer
reference and a separate scale-rounding bound against exact_reference. Structured integer/power-of-two inputs require
exact equality. CPU tests independently check scale RNE with frexp/rint and sample the FP64 reference against rational
arithmetic.
"""

from __future__ import annotations

import numpy as np
import torch


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().to(torch.float64)


def exact_reference(
    x: torch.Tensor, w_q: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """数学真值 O = X @ (W_q * scale)，float64。"""
    w_dq = _f64(w_q) * _f64(scale)[None, :]
    return _f64(x) @ w_dq


def dequant_fp16(w_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """cube 输入前反量化：``fp16(fp32(W_q) * fp32(scale))``，两步各舍入一次（FP32 乘，再 RNE 到 FP16）。

    与 kernel 的 VF 序列逐步对应：INT8 -> INT32 -> FP32（精确）、FP32 乘 scale（IEEE RNE）、FP32 -> FP16（RNE，
    ``VFRoundMode.CAST_RINT`` 默认）。
    """
    w32 = (
        w_q.detach().cpu().to(torch.float32)
        * scale.detach().cpu().to(torch.float32)[None, :]
    )
    return w32.to(torch.float16)


def prescale_reference(
    x: torch.Tensor, w_q: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """反量化在 cube 输入前：吃 FP16 舍入后的 W_dq；float64 累加（cube 是 FP32 累加，差异在容差账里）。"""
    return _f64(x) @ _f64(dequant_fp16(w_q, scale))


def trunc_scale_fp19(scale: torch.Tensor) -> torch.Tensor:
    """fixpipe 实际使用的 scale：fp32 位型低 13 位清零（10 位显式尾数，向零截断）。"""
    bits = scale.detach().cpu().to(torch.float32).numpy().view(np.uint32) & np.uint32(
        0xFFFFE000
    )
    return torch.from_numpy(bits.view(np.float32).copy())


def round_scale_fp19_rne(scale: torch.Tensor) -> torch.Tensor:
    """scale 就近（ties-to-even）舍入到 10 位显式尾数：与 kernel 里 ``bits + 0x0FFF + ((bits >> 13) & 1)`` 再清低 13 位一致。"""
    bits = (
        scale.detach().cpu().to(torch.float32).numpy().view(np.uint32).astype(np.uint64)
    )
    bits = (
        bits + np.uint64(0x0FFF) + ((bits >> np.uint64(13)) & np.uint64(1))
    ) & np.uint64(0xFFFFE000)
    return torch.from_numpy(bits.astype(np.uint32).view(np.float32).copy())


def fixpipe_rne_reference(
    x: torch.Tensor, w_q: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """反量化在累加器出口，scale 先在设备上 RNE 到 fp19：``(X @ W_q) * rne19(scale)``。"""
    return (_f64(x) @ _f64(w_q)) * _f64(round_scale_fp19_rne(scale))[None, :]


def fixpipe_reference(
    x: torch.Tensor, w_q: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """反量化在累加器出口（fixpipe）：``(X @ W_q) * fp19(scale)``。"""
    return (_f64(x) @ _f64(w_q)) * _f64(trunc_scale_fp19(scale))[None, :]


def postscale_reference(
    x: torch.Tensor, w_q: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """反量化在累加器出口之后（Vector 侧 FP32 乘）：scale 全精度，``(X @ W_q) * scale``。"""
    return (_f64(x) @ _f64(w_q)) * _f64(scale)[None, :]


def unscaled_reference(
    x: torch.Tensor, w_q: torch.Tensor, scale: torch.Tensor | None = None
) -> torch.Tensor:
    """骨架变体（只拓宽、不乘 scale）用：``X @ W_q``。"""
    return _f64(x) @ _f64(w_q)


def error_scale(
    x: torch.Tensor, w_q: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """逐元素「项和尺度」``S[m, n] = sum_k |X[m, k]| * |W_q[k, n]| * |scale[n]|``。

    FP32 累加的舍入误差上界正比于它（Higham：``|err| <= gamma_{K-1} * S``，``gamma_{K-1} ~ (K-1) * 2^-24``），
    而**不是**正比于结果的绝对值——结果落在 0 附近（抵消）的位置相对误差会大。判据用它作尺度，见 test.py。
    """
    return (_f64(x).abs() @ _f64(w_q).abs()) * _f64(scale).abs()[None, :]


def make_inputs(seed: int = 0, m: int = 1024, k: int = 256, n: int = 256):
    """典型分布：X ~ N(0,1) 的 FP16，W_q 均匀取满 [-127, 127]（不含 -128，对称量化的常规做法），
    scale 取 ``max|w|/127`` 量级的 FP32（[0.001, 0.05)，低 13 位非零——会暴露 fp19 截断）。
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn([m, k], generator=g).to(torch.float16)
    w_q = torch.randint(-127, 128, [k, n], generator=g, dtype=torch.int64).to(
        torch.int8
    )
    scale = (torch.rand([n], generator=g) * 0.049 + 0.001).to(torch.float32)
    return x, w_q, scale
