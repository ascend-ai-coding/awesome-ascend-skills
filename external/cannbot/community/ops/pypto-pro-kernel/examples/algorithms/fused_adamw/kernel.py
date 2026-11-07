# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。原生 VF 融合 AdamW，连续状态向量原地更新。

契约：互不重叠的连续 param/m/v[N] 为 FP32，grad[N] 为 FP16。m'=b1*m+(1-b1)*float32(grad)，v'=b2*v+(1-b2)*float32(grad)^2，
param'=param*(1-lr*wd)-lr*(m'/(1-b1^step))/(sqrt(v'/(1-b2^step))+eps)。grad 不变；三个状态数组完整更新，不分配额外 GM 工作区。每次参考从实际调用前的状态计算
FP64 公式，三个输出均使用 atol=rtol=2e-5，不在当前参考公式中额外舍入 m'/v'。

N=1..1048576，workers=1/4/16；param/m/grad 在[-10,10]，v 在[0,100]。lr 在[1e-5,1e-2]，b1=.9，b2=.999，eps 为1e-8或1e-6，wd 在[0,.1]，
step=1..10000。输入范围按每次调用前的状态约束；Host 仅检查元数据、标量并计算 bias correction，逐元素更新在 VF 中执行。

每个 worker 独占不相交的 GM 区间。默认搬运块16384元素、单槽；四组 Vec tile 的地址与 mutex ID 独立，auto_mutex 保证最后的 MTE3 写回完成后才复用缓冲。片上高水位为
14*16384=229376字节；普通 VF 的 UB tile 容量为262144字节，容量口径见 ``references/api/memory-limits.md``。VF 每次处理64个FP32容器；FP16经 UNPK 放入低半
部，ZERO转换与后续FP32运算共用代表位4*i的掩码，尾部沿同一容器位置裁剪。

m'、v' 的精确数学范围分别为[-10,10]、[0,100]；bias correction 后 |m_hat|≤100、v_hat≤100000，epsilon 保持分母非零，最大更新量级为1e8，FP32可表示。精度由独立参考及微小
梯度、抵消和连续调用测试验证，不只由有限范围推断。

默认配置是当前工具链和已测工作负载下的取舍。单槽搜索比较了相邻块宽并在容量边界停止；没有穷举较小块与多槽的组合，不宣称全局最优。大输入的workers=1/4设备时间改善，workers=16未显示明确设备收益；Host提交和同步时间需另行
配对测量。内部配置入口用于独立实验，其他配置须重新检查容量、槽复用和数值。

验证身份：PyPTO行为基线872df763f，CANN9.2.0，Ascend950PR，torch_npu2.10.0。测试覆盖寄存器与搬运边界、联合上限、偏移守卫、逐位不变的梯度以及变化的梯度/标量和连续更新。
"""

from functools import lru_cache
import math
from numbers import Integral, Real

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf


FP32 = pl.DT_FP32
FP16 = pl.DT_FP16
VL = 64
_FINAL_TRANSFER_ELEMENTS = 16384
_FINAL_SLOT_DEPTH = 1


@pl.vector_function
def _adamw_update(
    param_tile,
    grad_tile,
    m_tile,
    v_tile,
    extent,
    lr: pl.DT_FP32,
    beta1: pl.DT_FP32,
    beta2: pl.DT_FP32,
    eps: pl.DT_FP32,
    wd: pl.DT_FP32,
    one_minus_beta1: pl.DT_FP32,
    one_minus_beta2: pl.DT_FP32,
    inv_bias1: pl.DT_FP32,
    inv_bias2: pl.DT_FP32,
):
    """Update one UB chunk; each FP16 source group is widened without reordering."""
    for col in pl.range(0, extent, VL):
        active = pl.min(VL, extent - col)
        mask = vf.update_mask(active, dtype=FP32)

        p = vf.load_align(param_tile, [0, col])
        m = vf.load_align(m_tile, [0, col])
        v = vf.load_align(v_tile, [0, col])
        grad_packed = vf.load_align(grad_tile, [0, col], dist=pl.LoadDist.UNPK)
        grad32 = vf.astype(grad_packed, mask, dtype=FP32, layout=pl.CastLayout.ZERO)

        m_decay = vf.muls(m, beta1, mask)
        m_grad = vf.muls(grad32, one_minus_beta1, mask)
        m_new = vf.add(m_decay, m_grad, mask)

        grad_square = vf.mul(grad32, grad32, mask)
        v_decay = vf.muls(v, beta2, mask)
        v_grad = vf.muls(grad_square, one_minus_beta2, mask)
        v_new = vf.add(v_decay, v_grad, mask)

        m_hat = vf.muls(m_new, inv_bias1, mask)
        v_hat = vf.muls(v_new, inv_bias2, mask)
        denominator = vf.sqrt(v_hat, mask)
        denominator = vf.adds(denominator, eps, mask)
        normalized_m = vf.div(m_hat, denominator, mask)

        decay_factor = 1.0 - lr * wd
        decayed_param = vf.muls(p, decay_factor, mask)
        scaled_update = vf.muls(normalized_m, lr, mask)
        param_new = vf.sub(decayed_param, scaled_update, mask)

        vf.store_align(param_tile, param_new, mask, [0, col])
        vf.store_align(m_tile, m_new, mask, [0, col])
        vf.store_align(v_tile, v_new, mask, [0, col])


@lru_cache(maxsize=None)
def _make_kernel(transfer_elements: int, slot_depth: int):
    """Build one static transfer-size/slot-depth variant for measurement."""
    if transfer_elements < VL or transfer_elements % VL:
        raise ValueError("transfer_elements must be a positive multiple of 64")
    if slot_depth not in (1, 2, 4, 8):
        raise ValueError("slot_depth must be 1, 2, 4, or 8")

    bytes_per_tile = {
        "param": transfer_elements * 4,
        "grad": transfer_elements * 2,
        "m": transfer_elements * 4,
        "v": transfer_elements * 4,
    }
    base = 0
    group_addrs = []
    group_ids = []
    for group_index, name in enumerate(("param", "grad", "m", "v")):
        group_addrs.append(
            [base + slot * bytes_per_tile[name] for slot in range(slot_depth)]
        )
        group_ids.append(
            [group_index * slot_depth + slot for slot in range(slot_depth)]
        )
        base += bytes_per_tile[name] * slot_depth
    param_addrs, grad_addrs, m_addrs, v_addrs = group_addrs
    param_ids, grad_ids, m_ids, v_ids = group_ids

    @pl.jit(auto_mutex=True, name=f"adamw_c{transfer_elements}_d{slot_depth}")
    def kernel(
        param: pl.Tensor[[pl.DYNAMIC], FP32],
        grad: pl.Tensor[[pl.DYNAMIC], FP16],
        m: pl.Tensor[[pl.DYNAMIC], FP32],
        v: pl.Tensor[[pl.DYNAMIC], FP32],
        lr: pl.DT_FP32,
        beta1: pl.DT_FP32,
        beta2: pl.DT_FP32,
        eps: pl.DT_FP32,
        wd: pl.DT_FP32,
        one_minus_beta1: pl.DT_FP32,
        one_minus_beta2: pl.DT_FP32,
        inv_bias1: pl.DT_FP32,
        inv_bias2: pl.DT_FP32,
    ):
        param_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, transfer_elements],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=param_addrs,
            mutex_ids=param_ids,
        )
        grad_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, transfer_elements],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=grad_addrs,
            mutex_ids=grad_ids,
        )
        m_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, transfer_elements],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=m_addrs,
            mutex_ids=m_ids,
        )
        v_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, transfer_elements],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=v_addrs,
            mutex_ids=v_ids,
        )
        with pl.section_vector():
            chunk_count = (param.shape[0] + transfer_elements - 1) // transfer_elements
            for chunk in pl.range(pl.get_block_idx(), chunk_count, pl.get_block_num()):
                origin = chunk * transfer_elements
                extent = pl.min(transfer_elements, param.shape[0] - origin)
                param_tile = param_group.next()
                grad_tile = grad_group.next()
                m_tile = m_group.next()
                v_tile = v_group.next()
                pl.set_validshape(param_tile, [1, extent])
                pl.set_validshape(grad_tile, [1, extent])
                pl.set_validshape(m_tile, [1, extent])
                pl.set_validshape(v_tile, [1, extent])
                pl.load(param_tile, param, [origin])
                pl.load(grad_tile, grad, [origin])
                pl.load(m_tile, m, [origin])
                pl.load(v_tile, v, [origin])
                _adamw_update(
                    param_tile,
                    grad_tile,
                    m_tile,
                    v_tile,
                    extent,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    wd,
                    one_minus_beta1,
                    one_minus_beta2,
                    inv_bias1,
                    inv_bias2,
                )
                pl.store(param, param_tile, [origin])
                pl.store(m, m_tile, [origin])
                pl.store(v, v_tile, [origin])

    return kernel


def _validate_inputs(param, grad, m, v, lr, b1, b2, eps, wd, step, workers):
    if workers not in (1, 4, 16):
        raise ValueError("workers must be 1, 4, or 16")
    tensors = (param, grad, m, v)
    if any(len(t.shape) != 1 for t in tensors):
        raise ValueError("param, grad, m, and v must be one-dimensional")
    shapes = tuple(tuple(t.shape) for t in tensors)
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError("param, grad, m, and v must have identical shapes")
    n = int(param.shape[0])
    if not 1 <= n <= 1_048_576:
        raise ValueError("N must be in [1, 1048576]")
    expected_dtypes = (
        "torch.float32",
        "torch.float16",
        "torch.float32",
        "torch.float32",
    )
    if tuple(str(t.dtype) for t in tensors) != expected_dtypes:
        raise TypeError("param/m/v must be FP32 and grad must be FP16")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("param, grad, m, and v must be contiguous")
    if any(t.device != param.device for t in tensors[1:]):
        raise ValueError("param, grad, m, and v must use the same device")
    if param.device.type != "npu":
        raise ValueError("tensors must be on an NPU")

    if not all(isinstance(x, Real) for x in (lr, b1, b2, eps, wd)):
        raise TypeError("lr, b1, b2, eps, and wd must be real scalar values")
    lr, b1, b2, eps, wd = map(float, (lr, b1, b2, eps, wd))
    if not 1e-5 <= lr <= 1e-2:
        raise ValueError("lr must be in [1e-5, 1e-2]")
    if not math.isclose(b1, 0.9, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("b1 must be 0.9")
    if not math.isclose(b2, 0.999, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("b2 must be 0.999")
    if not (
        math.isclose(eps, 1e-8, rel_tol=1e-6) or math.isclose(eps, 1e-6, rel_tol=1e-6)
    ):
        raise ValueError("eps must be 1e-8 or 1e-6")
    if not 0.0 <= wd <= 0.1:
        raise ValueError("wd must be in [0, 0.1]")
    if not isinstance(step, Integral) or not 1 <= int(step) <= 10_000:
        raise ValueError("step must be an integer in [1, 10000]")
    return n, lr, b1, b2, eps, wd, int(step)


def _run_config(
    param, grad, m, v, lr, b1, b2, eps, wd, step, workers, transfer_elements, slot_depth
):
    n, lr, b1, b2, eps, wd, step = _validate_inputs(
        param, grad, m, v, lr, b1, b2, eps, wd, step, workers
    )
    one_minus_beta1 = 1.0 - b1
    one_minus_beta2 = 1.0 - b2
    inv_bias1 = 1.0 / (1.0 - b1**step)
    inv_bias2 = 1.0 / (1.0 - b2**step)
    _make_kernel(transfer_elements, slot_depth)[None, workers](
        param,
        grad,
        m,
        v,
        lr,
        b1,
        b2,
        eps,
        wd,
        one_minus_beta1,
        one_minus_beta2,
        inv_bias1,
        inv_bias2,
    )


def run(param, grad, m, v, lr, b1, b2, eps, wd, step, workers=4):
    """Update param, m, and v in place; grad remains read-only."""
    _run_config(
        param,
        grad,
        m,
        v,
        lr,
        b1,
        b2,
        eps,
        wd,
        step,
        workers,
        _FINAL_TRANSFER_ELEMENTS,
        _FINAL_SLOT_DEPTH,
    )
