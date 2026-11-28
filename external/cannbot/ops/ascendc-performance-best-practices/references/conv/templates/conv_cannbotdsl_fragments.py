# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""卷积 DSL 关键片段（annotated_excerpt），供目标 Kernel 裁剪接入。

整理自用户提供的窗口归约、逐点 Cube、混合核拆分与原生 Vector 实现。
形状、分片数与 scale 由调用方按编译期参数确定；片段采用双缓冲，不含 Host/launch。
寄存器函数在 cbd.vf 内调用；缓冲函数在任务循环外调用。
"""

import cannbotdsl as cbd
from cannbotdsl import MemLoc, dtypes, make_layout, reg
from cannbotdsl.ring_buffer import RingBuffer


@cbd.jit
def cube_window_buffers(dtype, mt, nt, kmax, step_k, xcap):
    """dtype 为 FP16/BF16；xcap 已含 NZ 对齐和尾部，核对 L1/L0 物理容量。"""
    a1 = RingBuffer(MemLoc.L1, dtype, make_layout((mt, kmax)), depth=2)
    b1 = RingBuffer(MemLoc.L1, dtype, make_layout((xcap,)), data_format="nd", depth=2)
    a0 = RingBuffer(MemLoc.L0A, dtype, make_layout((mt, step_k)), depth=2)
    b0 = RingBuffer(MemLoc.L0B, dtype, make_layout((nt, step_k)), depth=2)
    c0 = cbd.Buffer(MemLoc.L0C, dtypes.float32, make_layout((mt, nt)))
    return a1, b1, a0, b0, c0


@cbd.jit
def reduction_range(total, parts, part):
    """parts > 0，0 <= part < parts；空区间由调用方写零贡献。"""
    return total * part // parts, total * (part + 1) // parts


@cbd.jit
def pointwise_cube_reduce(
    grad, x, dtype, mt, nt, step_k, co, ci, first, last, grad_stride, x_stride
):
    """逐点权重梯度：输入与梯度采样位置一致，grad/x 已定位到同一 batch/组。

    输入为 FP16/BF16，mt/nt/step_k 对齐 Cube；[first,last) 非空且按 step_k 对齐。
    循环先预取下一块，再消费上一块；返回 FP32 L0C，由外层写 partial。
    """
    a1 = RingBuffer(MemLoc.L1, dtype, make_layout((mt, step_k)), depth=2)
    b1 = RingBuffer(MemLoc.L1, dtype, make_layout((nt, step_k)), depth=2)
    a0 = RingBuffer(MemLoc.L0A, dtype, make_layout((mt, step_k)), depth=2)
    b0 = RingBuffer(MemLoc.L0B, dtype, make_layout((nt, step_k)), depth=2)
    c0 = cbd.Buffer(MemLoc.L0C, dtypes.float32, make_layout((mt, nt)))
    for pos in range(first, last + step_k, step_k):
        if pos < last:
            ga = cbd.from_tensor(
                grad,
                layout=make_layout((mt, step_k), stride=(grad_stride, 1)),
                offset=(co * grad_stride + pos) * 2,
            )
            xb = cbd.from_tensor(
                x,
                layout=make_layout((nt, step_k), stride=(x_stride, 1)),
                offset=(ci * x_stride + pos) * 2,
            )
            cbd.mem_copy(a1.next_write(), ga)
            cbd.mem_copy(b1.next_write(), xb)
        if pos > first:
            cbd.mem_copy(a0.next_write(), a1.next_read())
            cbd.mem_copy(b0.next_write(), b1.next_read())
            cbd.matmul(c0, a0.next_read(), b0.next_read(), init=(pos == first + step_k))
    return c0


@cbd.jit
def spatial_window_mmad(
    c0,
    grad_l1,
    fmap_l1,
    a0,
    b0,
    mt,
    nt,
    pixels,
    step_k,
    feature,
    kernel_size,
    stride,
    dilation,
    initialize,
):
    """空间权重梯度：grad_l1 为 Co×pixels 的 NZ，fmap_l1 已含输入窗口。

    mt/nt/pixels/step_k 为 16 的倍数，nt=CiTile×Kh×Kw；边界在 feature 中描述。
    feature 为 Load3DFeatureMap；外层按窗口预取 L1，首窗口 initialize=True。
    """
    first = initialize
    for ks in range(0, pixels, step_k):
        kl = min(step_k, pixels - ks)
        aa, bb = a0.next_write(), b0.next_write()
        source = cbd.from_tensor(
            grad_l1,
            layout=make_layout((mt, kl)),
            data_format="nz",
            physical_layout=make_layout((kl // 16, mt // 16, 16, 16)),
            offset=ks * mt * 2,
        )
        cbd.mem_copy(aa[None, 0:kl], source)
        cbd.load3d(
            bb,
            fmap_l1,
            feature=feature,
            window=cbd.Load3DWindow(m_start=ks, m_extent=kl, k_start=0, k_extent=nt),
            filter=kernel_size,
            stride=stride,
            dilation=dilation,
            config_bank="b",
            destination=cbd.Load3DDestination(k_stride_fractals=nt // 16),
        )
        cbd.matmul(
            c0, a0.next_read()[None, 0:kl], b0.next_read()[None, 0:kl], init=first
        )
        first = False
    return first


@cbd.jit
def reduce_fp32_parts(raw, summed, elements, part_stride, parts):
    """parts > 0；raw 为 FP32，分片 stride 向 64 元素对齐且尾部已初始化。"""
    with cbd.vf():
        for offset in range(0, elements, 64):
            mask, _ = reg.update_mask(elements - offset)
            value = reg.vload(raw, offset)
            for part in range(1, parts):
                value = reg.vadd(
                    value, reg.vload(raw, part * part_stride + offset), mask=mask
                )
            reg.vstore(summed, offset, value, mask)


@cbd.jit
def publish_aiv_l1_slice(packed, l1_slot, sub_elements):
    """两个 AIV 各写一个不重叠的 16 位切片；在 cbd.is_aiv() 分支调用。

    packed 为当前 AIV 的 UB，l1_slot 至少含 2×sub_elements 个元素。
    每个逻辑生产步，两个 AIV 各 next_write 一次，AIC 对应 next_read 一次。
    """
    dst = cbd.from_tensor(
        l1_slot,
        layout=make_layout((sub_elements,)),
        data_format="nd",
        offset=cbd.get_subblock_id() * sub_elements * 2,
    )
    src = cbd.from_tensor(packed, layout=make_layout((sub_elements,)), data_format="nd")
    cbd.mem_copy(dst, src)


@cbd.jit
def split_fp32(value, scale, mask):
    """scale 为正的 2 的幂；调用方约束缩放后值域和残差误差。"""
    scaled = reg.vmuls(value, scale, mask=mask)
    high = reg.vcast(scaled, dtypes.float16, mask=mask, rounding=reg.RoundingMode.RN)
    residual = reg.vsub(scaled, reg.vcast(high, dtypes.float32, mask=mask), mask=mask)
    low = reg.vcast(residual, dtypes.float16, mask=mask, rounding=reg.RoundingMode.RN)
    return high, low


@cbd.jit
def three_product_mmad(c0, a_packed, b_packed, mt, nt, k, initialize):
    """L0 NZ 沿 K 打包：A=[Ah,Al]，B=[Bl,Bh]；mt/nt/k 对齐 Cube。

    HH 单独计算；拼接乘积一次计算 HL+LH，三个乘积均累加到 FP32 c0。
    """
    ah = cbd.from_tensor(a_packed, layout=make_layout((mt, k)), data_format="nz")
    bh = cbd.from_tensor(
        b_packed, layout=make_layout((nt, k)), data_format="nz", offset=nt * k * 2
    )
    cbd.matmul(c0, ah, bh, init=initialize)
    a_both = cbd.from_tensor(
        a_packed, layout=make_layout((mt, 2 * k)), data_format="nz"
    )
    b_both = cbd.from_tensor(
        b_packed, layout=make_layout((nt, 2 * k)), data_format="nz"
    )
    cbd.matmul(c0, a_both, b_both, init=False)


@cbd.jit
def restore_scale_and_bias(value, bias, inverse_product_scale, mask):
    """VF 内：全部归约结束后恢复 sx×sw，最后加入 FP32 bias。"""
    restored = reg.vmuls(value, inverse_product_scale, mask=mask)
    return reg.vadd(restored, bias, mask=mask)


@cbd.jit
def native_fma(raw, offset, coefficient, accumulator, mask):
    """raw 为 FP16/BF16 UB；coefficient 是 FP32 标量，accumulator 是 FP32 向量。

    完整向量读取范围须有效且初始化；计算 mask 不限制底层 UB 加载范围。
    """
    native = reg.vload_unpack(raw, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
    value = reg.vcast(native, dtypes.float32, mask=mask)
    return reg.vaxpy(accumulator, value, coefficient, mask=mask)


@cbd.jit
def load_fp32_phases(raw, first_offset, second_offset):
    """VF 内分离两个 FP32 向量的偶/奇相位；两段读取均须完整有效。"""
    return reg.vdeinterleave(
        reg.vload(raw, first_offset), reg.vload(raw, second_offset)
    )


@cbd.jit
def phase_fma(accumulator, phase, indices, valid, coefficient, mask):
    """VF 内：标量 coefficient；indices 约束到寄存器范围，valid 屏蔽边界。"""
    shifted = reg.vgather_reg(phase, indices)
    zero = reg.vdups(0.0, dtypes.float32)
    selected = reg.vselect(shifted, zero, cond_mask=valid)
    return reg.vaxpy(accumulator, selected, coefficient, mask=mask)


@cbd.jit
def row_pair_fma(first_sum, second_sum, first_row, second_row, coefficient, mask):
    """VF 内共享同一 FP32 标量系数，交错推进两个输出行并保持各自累加顺序。"""
    first_sum = reg.vaxpy(first_sum, first_row, coefficient, mask=mask)
    second_sum = reg.vaxpy(second_sum, second_row, coefficient, mask=mask)
    return first_sum, second_sum
