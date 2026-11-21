# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。Direct-GM grouped dilated 1-D cross-correlation.

example-query: {"aliases": ["grouped dilated convolution", "grouped conv1d", "分组空洞卷积", "一维分组卷积"]}

Inputs are contiguous x[B,L,Cin] and w[G,3,Cin/G,Cout/G] in FP16, plus
FP32 bias[Cout]. Output is contiguous FP32 [B,ceil(L/stride),Cout]. Tap r
reads t*stride+(r-1)*dilation; invalid source positions contribute zero.
Each SIMT thread owns an output scalar and traverses its group's input channels.
Both FP16 operands are promoted before FP32 FMA; a FP32 accumulator alone
would not specify the product precision. The host handles metadata/allocation.

The domain is B=1..3, L=1..4097, Cin/Cout multiples of 16 in [16,128],
G in {1,2,4,8} dividing both channel counts, and stride/dilation in {1,2}.
Inputs/weights are bounded by 1 and bias by 2 in absolute value. The reference
uses the actual quantized inputs in FP64 with atol=rtol=3e-4.
Physical SIMT Tensor shapes are static on the supported frontend, so the factory
specializes them per call. Grid-stride ownership covers the automatic grid.
There is no UB workspace or inter-thread communication. This is a correctness
anchor; repeated input reads and scalar reduction work need their own timing.
The verification toolchain is PyPTO cbd3c4a778, CANN 9.2.0 and Ascend950PR;
current execution evidence must identify the selected source and installation.
"""

from functools import lru_cache

import pypto_pro.language as pl

FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
THREADS = 128


@lru_cache(maxsize=None)
def _make_kernel(batch, length, cin, groups, cout, stride, dilation):
    """按静态物理形状和采样间距生成原生入口。"""
    cin_group = cin // groups
    cout_group = cout // groups
    out_length = (length + stride - 1) // stride
    output_count = batch * out_length * cout

    @pl.vector_function(mode="simt", max_threads=THREADS)
    def conv_body(x, w, bias, y):
        # 一个线程独占一个 y 标量，再以 grid-stride 覆盖全部输出。
        # 线程保留一个 FP32 累加器，直接读取 3*Cin/G 个输入/权重值；无片上暂存或跨线程同步。
        start = pl.simt.block_idx().x * THREADS + pl.simt.linear_thread_idx()
        step = pl.simt.grid_dim().x * pl.simt.block_dim().x
        for linear in pl.range(start, output_count, step):
            out_c = linear % cout
            out_t = (linear // cout) % out_length
            out_b = linear // (cout * out_length)
            group = out_c // cout_group
            out_local = out_c % cout_group
            acc = bias[out_c]
            for tap in pl.range(0, 3, 1):
                source_t = out_t * stride + (tap - 1) * dilation
                if source_t >= 0 and source_t < length:
                    for in_local in pl.range(0, cin_group, 1):
                        in_c = group * cin_group + in_local
                        xv = pl.simt.cast(x[out_b, source_t, in_c], FP32)
                        wv = pl.simt.cast(w[group, tap, in_local, out_local], FP32)
                        acc = pl.simt.fma(xv, wv, acc)
            y[out_b, out_t, out_c] = acc

    @pl.jit()
    def kernel(
        x: pl.Tensor[[batch, length, cin], FP16],
        w: pl.Tensor[[groups, 3, cin_group, cout_group], FP16],
        bias: pl.Tensor[[cout], FP32],
        y: pl.Tensor[[batch, out_length, cout], FP32],
    ):
        with pl.section_vector():
            conv_body[THREADS](x, w, bias, y)

    return kernel


def run(x, w, bias, stride=1, dilation=1):
    """Run the native kernel; input contents remain on device and are never read by Host."""
    import torch

    if x.ndim != 3 or w.ndim != 4 or bias.ndim != 1:
        raise ValueError("expected x[B,L,Cin], w[G,3,Cin/G,Cout/G], bias[Cout]")
    batch, length, cin = map(int, x.shape)
    groups, kernel_width, cin_group, cout_group = map(int, w.shape)
    cout = groups * cout_group
    if kernel_width != 3 or cin_group * groups != cin or int(bias.shape[0]) != cout:
        raise ValueError("incompatible grouped-convolution shapes")
    if not 1 <= batch <= 3 or not 1 <= length <= 4097:
        raise ValueError("B and L are outside the supported domain")
    if not 16 <= cin <= 128 or cin % 16 or not 16 <= cout <= 128 or cout % 16:
        raise ValueError("Cin and Cout must be 16..128 in multiples of 16")
    if groups not in (1, 2, 4, 8) or cin % groups or cout % groups:
        raise ValueError("G must be one of 1, 2, 4, 8 and divide Cin/Cout")
    if stride not in (1, 2) or dilation not in (1, 2):
        raise ValueError("stride and dilation must be 1 or 2")
    if (
        x.dtype != torch.float16
        or w.dtype != torch.float16
        or bias.dtype != torch.float32
    ):
        raise TypeError("x and w must be FP16; bias must be FP32")
    if not x.is_contiguous() or not w.is_contiguous() or not bias.is_contiguous():
        raise ValueError("inputs must be contiguous")
    if x.device != w.device or x.device != bias.device:
        raise ValueError("inputs must be on the same device")

    out_length = (length + stride - 1) // stride
    y = torch.empty((batch, out_length, cout), dtype=torch.float32, device=x.device)
    kernel = _make_kernel(batch, length, cin, groups, cout, stride, dilation)
    kernel(x, w, bias, y)
    return y
