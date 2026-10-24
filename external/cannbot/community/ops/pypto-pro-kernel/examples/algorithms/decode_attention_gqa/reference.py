# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU 参考：GQA decode ``O[b,hq,0,:] = softmax(q[b,hq,0,:] K[b,hq//G]^T / sqrt(d)) V[b,hq//G]``。

数学定义按 (b, h_kv) 组分解：组 ``(b, h)`` 是一个单头注意力实例 ``Q_grp [G, D] x K/V [S, D]``，
与 ``examples/algorithms/flash_attention`` 的单头形状（m=G）逐字相同，所以**参考与先验误差界整体
复用那份实现**（``fa.attention_reference`` / ``fa.error_bound``——推导见其 ``error_terms``
docstring；decode 的舍入源同款：P->FP16 量化主导 + score FP32 累加 + exp 自变量 + 硬件 exp +
PV/l 累加，块数 nb = S/BLOCK_KV 代入即可）。本文件只负责 GQA 形状的组装、GQA 特有的输入类
（组映射 one-hot）与「错结构」负对照（错组映射、V 头错位等）。

误差界项数与本 kernel 的对应（为什么单头的界对 decode 成立）：``quant_p`` 与布局无关；``score``
的 Higham 界对任意求和顺序成立（本 kernel splitk=64 两步累加 <= 任意顺序上界）；``exp_arg`` 的
分段论证只依赖 running max 单调；``accum`` 取 ``gamma(block_kv + 2nb + 2)``——本 kernel 的 O 更新
是单条 FMA（每块 1 次舍入 < 2 次）、末块除法 1 次，实际舍入次数少于上界的计数，上界仍然成立。
"""

import torch

from ..flash_attention import reference as fa

B, HQ, HKV, D = 16, 64, 8, 128
G = HQ // HKV
S = 4096
BLOCK_KV = 256  # 与 kernel.BKV 一致；test 里有一致性断言
NB = S // BLOCK_KV


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().to(torch.float64)


def _per_group(fn, q, k, v, *args, **kw):
    """对每个 (b, h_kv) 组调用单头函数 ``fn(qg [G,D], kh [S,D], vh [S,D], ...)``，拼回 [B, HQ, 1, D]。"""
    out = torch.zeros([B, HQ, 1, D], dtype=torch.float64)
    for b in range(B):
        for h in range(HKV):
            qg = q[b, slice(h * G, (h + 1) * G), 0, :]
            r = fn(qg, k[b, h], v[b, h], *args, **kw)
            out[b, slice(h * G, (h + 1) * G), 0, :] = r
    return out


def attention_reference(q, k, v):
    """数学真值（float64），与 kernel 的分块 / 在线 rescale / 舍入无关。"""
    return _per_group(fa.attention_reference, q, k, v)


def error_bound(q, k, v, block_kv: int = BLOCK_KV):
    """逐元素先验上界，逐组复用 flash_attention 的推导（见文件头）。"""
    return _per_group(fa.error_bound, q, k, v, block_kv)


# ---------------------------------------------------------------------------------------------
# 「错结构」参考：必须通不过同一判据（含 GQA 特有的映射错误）
# ---------------------------------------------------------------------------------------------


def unscaled_reference(q, k, v):
    return _per_group(lambda qg, kh, vh: fa.unscaled_reference(qg, kh, vh), q, k, v)


def wrong_scale_reference(q, k, v):
    return _per_group(lambda qg, kh, vh: fa.wrong_scale_reference(qg, kh, vh), q, k, v)


def blockwise_softmax_reference(q, k, v, block_kv: int = BLOCK_KV):
    return _per_group(
        lambda qg, kh, vh: fa.blockwise_softmax_reference(qg, kh, vh, block_kv), q, k, v
    )


def no_o_rescale_reference(q, k, v, block_kv: int = BLOCK_KV):
    return _per_group(
        lambda qg, kh, vh: fa.no_o_rescale_reference(qg, kh, vh, block_kv), q, k, v
    )


def no_l_rescale_reference(q, k, v, block_kv: int = BLOCK_KV):
    return _per_group(
        lambda qg, kh, vh: fa.no_l_rescale_reference(qg, kh, vh, block_kv), q, k, v
    )


def group_shifted_reference(q, k, v):
    """GQA 映射错误 #1：q 组 ``h`` 错用相邻 KV 头 ``(h+1) % HKV`` 的 K 与 V——本题最可能的静默缺陷。"""
    out = torch.zeros([B, HQ, 1, D], dtype=torch.float64)
    for b in range(B):
        for h in range(HKV):
            w = (h + 1) % HKV
            qg = q[b, slice(h * G, (h + 1) * G), 0, :]
            out[b, slice(h * G, (h + 1) * G), 0, :] = fa.attention_reference(
                qg, k[b, w], v[b, w]
            )
    return out


def v_head_shifted_reference(q, k, v):
    """GQA 映射错误 #2：分数用对了 K 头，PV 却读了相邻头的 V（K/V 独立映射，可只错一半）。"""
    out = torch.zeros([B, HQ, 1, D], dtype=torch.float64)
    for b in range(B):
        for h in range(HKV):
            w = (h + 1) % HKV
            qg = q[b, slice(h * G, (h + 1) * G), 0, :]
            p = fa.softmax_weights(qg, k[b, h])
            out[b, slice(h * G, (h + 1) * G), 0, :] = p @ _f64(v[b, w])
    return out


# ---------------------------------------------------------------------------------------------
# 输入构造（kinds 语义沿用 flash_attention.make_inputs，按 KV 块作用在每个头的 S 轴上）
# ---------------------------------------------------------------------------------------------

KINDS = (
    "normal",
    "sharp",
    "growing",
    "first_block_max",
    "last_block_max",
    "large",
    "all_negative",
    "group_distinct",
)


def make_inputs(seed: int = 0, kind: str = "normal"):
    """FP16 的 ``(q [B,HQ,1,D], K [B,HKV,S,D], V [B,HKV,S,D])``。

    前七类语义同 flash_attention.make_inputs（rescale 路径覆盖：growing 每块更新 max、
    first/last_block_max 只在首/末块更新、all_negative 守 m 初值、large 守上/下溢）；
    ``group_distinct``：每个 KV 头的 K/V 带头编码的直流偏置、每组 q 对齐自己头的编码——
    组映射错一头，分数与 V 都错，输出可见地偏离（ascriptor head_distinct 的 GQA 版）。
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn([B, HQ, 1, D], generator=g)
    k = torch.randn([B, HKV, S, D], generator=g)
    v = torch.randn([B, HKV, S, D], generator=g)
    if kind == "sharp":
        q = q * 4.0
    elif kind == "growing":
        w = torch.repeat_interleave(
            1.0 + torch.arange(NB, dtype=torch.float32), BLOCK_KV
        )
        k = k * w.view(1, 1, S, 1)
    elif kind == "first_block_max":
        k = k.clone()
        k[:, :, :BLOCK_KV, :] *= 4.0
    elif kind == "last_block_max":
        k = k.clone()
        k[:, :, slice((NB - 1) * BLOCK_KV, None), :] *= 4.0
    elif kind == "large":
        q = (torch.rand([B, HQ, 1, D], generator=g) * 2 - 1) * 4.0
        k = (torch.rand([B, HKV, S, D], generator=g) * 2 - 1) * 4.0
    elif kind == "all_negative":
        q = torch.full([B, HQ, 1, D], 4.0)
        k = -(1.0 + 0.25 * k)
    elif kind == "group_distinct":
        code = (torch.arange(HKV, dtype=torch.float32) / HKV - 0.5).view(1, HKV, 1, 1)
        k = k * 0.25 + code
        v = v * 0.25 - code
        qcode = torch.repeat_interleave(code.view(HKV), G).view(1, HQ, 1, 1)
        q = q * 0.25 + 4.0 * qcode
    elif kind != "normal":
        raise ValueError(kind)
    return q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)


def uniform_inputs(seed: int = 0):
    """``q = 0``：每组的 softmax 是均匀分布，``O[b,hq] = mean_s V[b, hq//G, s]``。"""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn([B, HKV, S, D], generator=g)
    v = torch.randn([B, HKV, S, D], generator=g)
    return (
        torch.zeros([B, HQ, 1, D], dtype=torch.float16),
        k.to(torch.float16),
        v.to(torch.float16),
    )


def group_onehot_inputs(seed: int = 0, amplitude: float = 512.0):
    """每个 q 行恰命中**自己组的 KV 头**里的一个 key：返回 ``(q, K, V, target [B, HQ])``。

    K 的每一行是双热编码（自内积 2、任意两行内积 <= 1），编码在头间也不重复（全局取
    ``HKV*S`` 个不同 (a,b) 对；C(128,2)=8128 >= 8*4096 需要 b>a 全枚举——8128 < 32768，
    所以按 (头, 行) 用「双热 + 头相位的第三热」编码：``K[h,s] = e_a + e_b + 0.5*e_{ph}``，
    ``ph = h`` 固定在前 HKV 维——同头内 (a,b) 互不相同（a,b 从 HKV..D-1 取，
    C(120,2)=7140 >= 4096），跨头分数再差 ``0.25*amplitude/sqrt(d)``。
    ``q[b,hq] = amplitude * K[b, hq//G, t]``，命中分同头内其他 key 高 ``amplitude/sqrt(d)``、
    比错头最高分更高，``O[b,hq] ≈ V[b, hq//G, t]``。覆盖每个 KV 块内位置、每个核、组映射。
    """
    g = torch.Generator().manual_seed(seed)
    pairs = [(a, b) for a in range(HKV, D) for b in range(a + 1, D)]
    assert len(pairs) >= S
    k = torch.zeros([B, HKV, S, D])
    order = torch.randperm(len(pairs), generator=g)[:S].tolist()
    for h in range(HKV):
        for s, idx in enumerate(order):
            a, b = pairs[(idx + h * 977) % len(pairs)]
            k[:, h, s, a] = 1.0
            k[:, h, s, b] = 1.0
        k[:, h, :, h] = 0.5
    target = torch.randint(0, S, [B, HQ], generator=g)
    q = torch.zeros([B, HQ, 1, D])
    for b in range(B):
        for hq in range(HQ):
            q[b, hq, 0, :] = amplitude * k[b, hq // G, target[b, hq]]
    v = torch.randn([B, HKV, S, D], generator=g)
    return q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), target
