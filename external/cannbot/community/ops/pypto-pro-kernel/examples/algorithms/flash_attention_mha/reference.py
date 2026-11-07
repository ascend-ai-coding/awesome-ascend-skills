# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU 参考：多头多 batch 因果注意力 ``O[b, i, h, :] = softmax_{j <= i}(Q[b, i, h, :] . K[b, j, h, :] / sqrt(d)) V[b, :, h, :]``，
BSND 布局，从数学定义写起（float64）。

数学定义（与 kernel 的分块 / 在线 rescale / 精度 / 切分细节无关）::

    S[i, j] = (sum_d Q[i, d] * K[j, d]) / sqrt(d)                     每个 (batch, head) 各算各的
    P[i, j] = exp(S[i, j]) / sum_{j' <= i} exp(S[i, j'])   (j <= i)   否则 0        （因果：query i 只看 key <= i）
    O[i, e] = sum_{j <= i} P[i, j] * V[j, e]

本文件的口径
------------
- 输入本来就是 FP16（算子的真实输入），提升到 float64 是精确的。**参考不量化任何中间量**；减行最大值只为不溢出（softmax 平移不变）。
- ``error_bound`` 是逐元素的**先验**误差上界（只看输入）。各项来源与假设写在 ``error_terms``；判据尺度是项和
  ``S = sum_j P_ij |V_je|`` 而不是 ``|O|``（O 是带符号加权和，抵消位置 ``|O|`` 远小于 ``S``，api/numerics.md 一）。
  因果只改变「哪些 j 参与」：被掩的位置权重恰为 0，所有含 ``j`` 的求和都只跑 ``j <= i``，KV 块数 ``nb`` 变成逐行的
  ``i // block_kv + 1``。
- **因果边界的输入构造**（``make_inputs`` 的 ``KINDS``）：对角块半掩（``staircase``：块内阶梯，掩码泄漏 1 个 key 就是 O(1) 误差）、
  首行只看 1 个 key（所有 kind 的第 0 行；``uniform`` / ``staircase`` 上尤其敏感）、块级泄漏（``block_ramp``：分数只随 key 所在块
  递增，泄漏下一块就被放大到主导）、被掩位置写 NaN（``poison_future``：每个 (batch, head) 在各自的对角块之后把 K/V 写成 NaN，
  只检查不该碰到它们的行）。
- 后半部分是「错结构」参考（漏缩放、漏 alpha、无因果、掩码差一、块级掩码、串头等）：它们必须**通不过**同一个判据。
"""

import math

import torch

# ---- 数值常数（每个的来源见 error_terms 的 docstring）---------------------------------------------------
U16 = 2.0**-11  # FP16 就近舍入的单位舍入误差（P 转 FP16 喂 cube）
U32 = 2.0**-24  # FP32 就近舍入的单位舍入误差
FP16_SUBNORMAL_HALF_ULP = (
    2.0**-25
)  # FP16 次正规区的绝对舍入误差上界（次正规间距 2^-24 的一半）
EPS_EXP = (
    2.0**-22
)  # 硬件 exp 相对误差的先验假设（约 2.4e-7）；该项占总界不到 1%，放宽 10 倍也不改变判据松紧
HEAD_CHUNK = 16  # 逐块处理的 (batch, head) 数，限制 [n, S, S] 中间量的内存


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().to(torch.float64)


def _to_heads(t: torch.Tensor) -> torch.Tensor:
    """``[B, S, H, D]`` -> ``[B*H, S, D]``（float64，NaN 保留）。"""
    b, s, h, d = t.shape
    return _f64(t).permute(0, 2, 1, 3).reshape(b * h, s, d)


def _from_heads(t: torch.Tensor, batch: int, heads: int) -> torch.Tensor:
    """``[B*H, S, D]`` -> ``[B, S, H, D]``。"""
    bh, s, d = t.shape
    return t.reshape(batch, heads, s, d).permute(0, 2, 1, 3).contiguous()


def _causal_valid(seq: int) -> torch.Tensor:
    """``[S, S]`` 下三角（含对角）：query i 只看 key <= i。"""
    return torch.ones(seq, seq, dtype=torch.bool).tril()


def _gamma(count: torch.Tensor | float) -> torch.Tensor | float:
    """Higham 的 gamma_n = n u / (1 - n u)，u = 2^-24；对任意求和顺序成立。"""
    return count * U32 / (1.0 - count * U32)


def _chunks(bh: int):
    for lo in range(0, bh, HEAD_CHUNK):
        yield slice(lo, min(lo + HEAD_CHUNK, bh))


def _shape(q: torch.Tensor) -> tuple[int, int, int, int]:
    b, s, h, d = q.shape
    return b, s, h, d


def _sanitized(t: torch.Tensor) -> torch.Tensor:
    """被掩位置的 poison（NaN）只可能出现在从不参与被检查行的 key 上；参考里置 0 等价（那些位置权重恒为 0）。"""
    return torch.nan_to_num(t, nan=0.0)


def _softmax_causal(qh, kh, valid):
    """一块 head 的 ``(s, m, e, l, w)``：分数（缩放后）、行最大（只看可见 key）、exp(s - m)（被掩处恰为 0）、行和、权重。"""
    s = torch.matmul(qh, kh.transpose(1, 2)) / math.sqrt(qh.shape[-1])
    s_masked = s.masked_fill(~valid, float("-inf"))
    m = s_masked.amax(dim=-1, keepdim=True)
    e = torch.exp(s_masked - m)
    rowsum = e.sum(dim=-1, keepdim=True)
    return s, m, e, rowsum, e / rowsum


def attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """数学真值 ``[B, S, H, D]`` float64。**不是**任何 kernel 的分块顺序。"""
    b, s, h, d = _shape(q)
    valid = _causal_valid(s)
    qh, kh, vh = _to_heads(q), _sanitized(_to_heads(k)), _sanitized(_to_heads(v))
    out = torch.empty_like(qh)
    for sl in _chunks(qh.shape[0]):
        w = _softmax_causal(qh[sl], kh[sl], valid)[4]
        out[sl] = torch.matmul(w, vh[sl])
    return _from_heads(out, b, h)


def error_scale(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """项和尺度 ``S[i, e] = sum_{j <= i} P[i, j] * |V[j, e]|``（``[B, S, H, D]``）。所有误差项都正比于它。"""
    b, s, h, d = _shape(q)
    valid = _causal_valid(s)
    qh, kh, vh = _to_heads(q), _sanitized(_to_heads(k)), _sanitized(_to_heads(v))
    out = torch.empty_like(qh)
    for sl in _chunks(qh.shape[0]):
        w = _softmax_causal(qh[sl], kh[sl], valid)[4]
        out[sl] = torch.matmul(w, vh[sl].abs())
    return _from_heads(out, b, h)


def error_terms(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_kv: int,
    out_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """先验误差上界的各项（都是 ``[B, S, H, D]`` 的逐元素上界，单位与 ``O`` 相同），逐项分开返回。

    记 ``c = 1/sqrt(d)``，``P`` 为精确因果 softmax 权重，``S = P @ |V|``（``error_scale``），行 ``i`` 处理的 KV 块数
    ``nb_i = i // block_kv + 1``（对角块之后的块整块跳过，不产生任何舍入）。kernel 的舍入来源与对应的项：

    ``quant_p``  —— P 转 FP16 喂第二个 matmul（cube 的输入精度约束，不是可选项）。每个元素就近舍入，
        相对误差 <= 2^-11（正常区）；P <= 1 的次正规区（P < 2^-14）改为绝对误差 <= 2^-25。二者取并集：
        ``|dP| <= 2^-11 * P + 2^-25``。归一化用的行和 ``l`` 由**未量化**的 FP32 P 求和，所以只有分子被量化：
        ``|dO| <= 2^-11 * S + 2^-25 * sum_{j <= i} |V[j, e]| / l``，其中 ``l >= 1`` 是以行最大为基准的 exp 和
        （最大元素贡献 exp(0) = 1）。是**主导项**。被掩位置的 P 恰为 0（exp 对 NEG_BIG 精确下溢为 0，api/vf.md
        二点十五），量化误差为 0。
    ``score``    —— Q K^T 由 cube 以 FP32 累加（FP16 x FP16 乘积在 FP32 里精确，只有加法舍入）。Higham 的 gamma 界对
        任意求和顺序成立：``|dS_ij| <= c * gamma_d * T_ij``，``T_ij = sum_e |Q_ie||K_je|``，
        ``gamma_d = d * 2^-24 / (1 - d * 2^-24)``。行内所有可见 score 共同带来的权重相对变化 <= ``exp(2 * delta_i) - 1``，
        ``delta_i = c * gamma_d * max_{j <= i} T_ij``（softmax 对 score 的偏移取差，最坏两端各 delta）。
    ``exp_arg``  —— 指数的自变量 ``(s - m) * c`` 由 FP32 的「减、乘」得到，每次相对舍入 <= 2^-24；自变量绝对误差
        <= 3 * 2^-24 * |a|（``a = s - m_final`` 的缩放值）。在线 rescale 使一个元素的自变量分成若干段（块内
        ``s - m_j`` 与各次 ``m_{j-1} - m_j``），各段同号（running max 单调不降），绝对值之和恰为 ``|a|``，所以总自变量
        误差仍 <= ``3 * 2^-24 * |a|``。对 O 的影响：``3 * 2^-24 * (A + Abar * S)``，``A = sum_j P_j |a_j| |V_je|``，
        ``Abar = sum_j P_j |a_j|``（分母的加权平均误差）。
    ``exp_fn``   —— 硬件 exp 函数本身的相对误差，先验假设 <= 2^-22（``EPS_EXP``），每次 exp 调用一次；一个元素最多
        经历 ``nb_i`` 次（本块的 p 与至多 nb_i-1 次 alpha），分子分母各一份：``2 * nb_i * 2^-22 * S``。
    ``accum``    —— 分子 ``P V`` 的 FP32 累加（cube 块内 K = block_kv 项 + 每块「乘 alpha、加」的舍入 + 末尾除法的
        舍入），分母 ``l`` 的累加（寄存器内逐元素相加 block_kv 项 + 每块 2 次）：
        ``(gamma_{block_kv + 2 nb_i + 2} + gamma_{block_kv + 2 nb_i}) * S``（Higham gamma，任意求和顺序；两个
        gamma 各取一个宽松的项数上界，不依赖具体归约树形状）。对角块里被掩的项是精确的 0，加 0 不产生舍入。

    ``out_round`` —— 仅 ``out_dtype == torch.float16`` 时出现：最终结果由 FP32 就近舍入到 FP16 写出。就近舍入的误差
        ``<= 2^-11 |x|``（正常区）或 ``<= 2^-25``（次正规区），并集 ``<= 2^-11 |x| + 2^-25``；``x`` 是舍入前的 FP32 值，
        ``|x| <= |O_exact| + 其余五项之和 <= S + 其余五项之和``，故 ``out_round = 2^-11 (S + sum(其余)) + 2^-25``。

    Each term is input-dependent under the assumptions above; the comparator checks their sum.
    """
    b, seq, h, d = _shape(q)
    c = 1.0 / math.sqrt(d)
    valid = _causal_valid(seq)
    nb = (torch.arange(seq, dtype=torch.float64) // block_kv + 1.0).reshape(1, seq, 1)
    qh, kh, vh = _to_heads(q), _sanitized(_to_heads(k)), _sanitized(_to_heads(v))
    q_abs, k_abs, v_abs = qh.abs(), kh.abs(), vh.abs()
    names = ["quant_p", "score", "exp_arg", "exp_fn", "accum"] + (
        ["out_round"] if out_dtype == torch.float16 else []
    )
    terms = {name: torch.empty_like(qh) for name in names}
    for sl in _chunks(qh.shape[0]):
        s, m, _e, rowsum, w = _softmax_causal(qh[sl], kh[sl], valid)
        scale_s = torch.matmul(w, v_abs[sl])  # S[i, e]
        t = torch.matmul(q_abs[sl], k_abs[sl].transpose(1, 2)).masked_fill(
            ~valid, 0.0
        )  # T_ij，只看可见 key
        delta = c * _gamma(float(d)) * t.amax(dim=-1, keepdim=True)  # [n, S, 1]
        a_abs = (m - s).abs().masked_fill(~valid, 0.0)  # |a_ij|（已含 1/sqrt(d)）
        wa = w * a_abs
        cum_v = v_abs[sl].cumsum(dim=1)  # sum_{j <= i} |V[j, e]|
        terms["quant_p"][sl] = U16 * scale_s + FP16_SUBNORMAL_HALF_ULP * cum_v / rowsum
        terms["score"][sl] = torch.expm1(2.0 * delta) * scale_s
        terms["exp_arg"][sl] = (
            3.0
            * U32
            * (torch.matmul(wa, v_abs[sl]) + wa.sum(dim=-1, keepdim=True) * scale_s)
        )
        terms["exp_fn"][sl] = 2.0 * nb * EPS_EXP * scale_s
        terms["accum"][sl] = (
            _gamma(block_kv + 2 * nb + 2) + _gamma(block_kv + 2 * nb)
        ) * scale_s
        if "out_round" in terms:
            core = sum(
                terms[name][sl]
                for name in ("quant_p", "score", "exp_arg", "exp_fn", "accum")
            )
            terms["out_round"][sl] = U16 * (scale_s + core) + FP16_SUBNORMAL_HALF_ULP
    return {name: _from_heads(t, b, h) for name, t in terms.items()}


def error_bound(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_kv: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``error_terms`` 的和：逐元素先验上界 ``|O_kernel - O_exact| <= bound``。"""
    return sum(error_terms(q, k, v, block_kv, out_dtype).values())


# ---------------------------------------------------------------------------------------------
# 「错结构」参考：每一个都是一类真实的实现错误，必须通不过 error_bound 的判据
# ---------------------------------------------------------------------------------------------


def _attention_with(q, k, v, mask_fn=None, scale=None, k_of=None, v_of=None):
    """通用的 float64 注意力：``mask_fn(seq) -> [S, S] bool`` 选可见位置（缺省因果），``k_of/v_of`` 可换成别的 head 的 K/V。"""
    b, s, h, d = _shape(q)
    valid = (mask_fn or _causal_valid)(s)
    qh = _to_heads(q)
    kh = _sanitized(_to_heads(k if k_of is None else k_of(k)))
    vh = _sanitized(_to_heads(v if v_of is None else v_of(v)))
    scale = (1.0 / math.sqrt(d)) if scale is None else scale
    out = torch.empty_like(qh)
    for sl in _chunks(qh.shape[0]):
        sc = torch.matmul(qh[sl], kh[sl].transpose(1, 2)) * scale
        sc = sc.masked_fill(~valid, float("-inf"))
        out[sl] = torch.matmul(torch.softmax(sc, dim=-1), vh[sl])
    return _from_heads(out, b, h)


def noncausal_reference(q, k, v):
    """漏掉因果掩码（全可见）。"""
    return _attention_with(
        q, k, v, mask_fn=lambda s: torch.ones(s, s, dtype=torch.bool)
    )


def strict_causal_reference(q, k, v):
    """掩码差一：query i 只看 key < i（漏掉对角元素）；第 0 行没有可见 key，按 0 输出。"""
    out = _attention_with(
        q, k, v, mask_fn=lambda s: torch.ones(s, s, dtype=torch.bool).tril(-1)
    )
    out[:, 0] = 0.0
    return out


def block_causal_reference(q, k, v, block: int):
    """块级掩码：对角块整块可见（漏掉对角块内的半掩）。"""

    def blocked(s):
        i = torch.arange(s)[:, None]
        j = torch.arange(s)[None, :]
        return j < (i // block + 1) * block

    return _attention_with(q, k, v, mask_fn=blocked)


def unscaled_reference(q, k, v):
    """漏掉 1/sqrt(d) 缩放。"""
    return _attention_with(q, k, v, scale=1.0)


def wrong_scale_reference(q, k, v):
    """缩放系数写成 1/d（把 sqrt 漏了）。"""
    return _attention_with(q, k, v, scale=1.0 / q.shape[-1])


def head_shifted_reference(q, k, v):
    """串头：head h 的输出用了 head h+1 的 K 和 V（BSND 里 head 是倒数第二维，下标算错一格的典型后果）。"""
    return _attention_with(
        q, k, v, k_of=lambda t: t.roll(-1, dims=2), v_of=lambda t: t.roll(-1, dims=2)
    )


def batch_shifted_reference(q, k, v):
    """串 batch：batch b 的输出用了 batch b+1 的 K 和 V。"""
    return _attention_with(
        q, k, v, k_of=lambda t: t.roll(-1, dims=0), v_of=lambda t: t.roll(-1, dims=0)
    )


def _blockwise_online(q, k, v, block_kv: int, rescale_o: bool, rescale_l: bool):
    """在线 softmax 的两类漏 alpha 错误（float64，因果，逐行处理的块数 ``i // block_kv + 1``）。"""
    b, s, h, d = _shape(q)
    c = 1.0 / math.sqrt(d)
    qh, kh, vh = _to_heads(q), _sanitized(_to_heads(k)), _sanitized(_to_heads(v))
    out = torch.empty_like(qh)
    valid = _causal_valid(s)
    nb = s // block_kv
    for sl in _chunks(qh.shape[0]):
        sc = (torch.matmul(qh[sl], kh[sl].transpose(1, 2)) * c).masked_fill(
            ~valid, float("-inf")
        )
        m = torch.full(sc.shape[:2] + (1,), float("-inf"), dtype=torch.float64)
        rowsum = torch.zeros_like(m)
        out_acc = torch.zeros(sc.shape[:2] + (d,), dtype=torch.float64)
        for j in range(nb):
            sj = sc[:, :, slice(j * block_kv, (j + 1) * block_kv)]
            m_new = torch.maximum(m, sj.amax(dim=-1, keepdim=True))
            seen = torch.isfinite(m_new)  # 该行在这个块里已有可见 key
            m_safe = torch.where(seen, m_new, torch.zeros_like(m_new))
            pj = torch.exp(sj - m_safe)
            alpha = torch.where(
                torch.isfinite(m), torch.exp(m - m_safe), torch.zeros_like(m)
            )
            rowsum = (alpha if rescale_l else 1.0) * rowsum + pj.sum(
                dim=-1, keepdim=True
            )
            out_acc = (alpha if rescale_o else 1.0) * out_acc + torch.matmul(
                pj, vh[sl][:, slice(j * block_kv, (j + 1) * block_kv)]
            )
            m = m_new
        out[sl] = out_acc / rowsum
    return _from_heads(out, b, h)


def no_o_rescale_reference(q, k, v, block_kv: int):
    """l 正确 rescale、O 累加漏掉 alpha。"""
    return _blockwise_online(q, k, v, block_kv, rescale_o=False, rescale_l=True)


def no_l_rescale_reference(q, k, v, block_kv: int):
    """O 正确 rescale、行和 l 漏掉 alpha。"""
    return _blockwise_online(q, k, v, block_kv, rescale_o=True, rescale_l=False)


# ---------------------------------------------------------------------------------------------
# 输入构造
# ---------------------------------------------------------------------------------------------

KINDS = (
    "normal",
    "sharp",
    "growing",
    "first_block_max",
    "large",
    "all_negative",
    "staircase",
    "block_ramp",
    "uniform",
    "poison_future",
)


def _randn(g, shape):
    return torch.randn(shape, generator=g)


def make_inputs(
    seed: int = 0,
    kind: str = "normal",
    batch: int = 4,
    seq: int = 1024,
    heads: int = 32,
    d: int = 128,
    block_kv: int = 128,
):
    """FP16 的 ``(Q, K, V)``，形状 ``[B, S, H, D]``。``kind``（每一类覆盖因果在线 softmax 的一条不同路径）：

    - ``normal``：全部 N(0, 1)。
    - ``sharp``：Q 放大 4 倍（softmax 尖）。
    - ``growing``：第 j 个 KV 块的 K 放大 ``1 + j``——后面的块分数更大，running max 每块都更新，且每行的最大值总在它的
      对角块（最后一个被处理的块）里（因果下「末块最大」的对应物）。
    - ``first_block_max``：只有第 0 块的 K 放大 4 倍——max 在第一块之后不再变（alpha 恒为 1，只走「不更新」分支）。
    - ``large``：Q、K 取 [-4, 4] 均匀（分数到几百，不减最大值 exp 会上溢）。
    - ``all_negative``：``Q = 4``、``K = -(1 + 0.25 randn)``——所有分数约 -45：running max 初值若误取 0，输出整体为 0。
    - ``staircase``：块内阶梯。``p = i % block_kv``，``Q[i, d] = 512 * [d > p_i]``，``K[j] = onehot(p_j)``：分数只取决于块内位置，
      ``p_j > p_i`` 的 key 比其余高 45（指数域）。第 0 块的行里被掩的 key 全是「热」的，泄漏一个就是 O(1) 误差；后面的块里
      泄漏对角块的被掩 key 会把 O 拉向别的 V（误差 O(1/块数)，远大于界）。
    - ``block_ramp``：分数只随 key 所在**块**递增（每块高 45）：``Q[i, 0] = 64``、``K[j, 0] = 8 * (j // block_kv)``，其余维为 0。
      每个 q 块只该由自己的对角块主导，O[i] = 对角块内 key <= i 的 V 的平均；泄漏下一块（更热）会把 O 拉到别的块。每一块
      running max 都跳 45（alpha ~ 1e-20），rescale 路径被推到极端。
    - ``uniform``：``Q = 0``，所有可见分数为 0，O[i] = V[0..i] 的前缀平均：每个可见 key 恰被计入一次。
    - ``poison_future``：随机输入，但第 ``o`` 个 (batch, head) 在 ``G_o = (3 o) % (S / block)`` 号块之后的 K、V 全写成 NaN。只有
      ``checked_rows`` 里的行（``G_o`` 号块及之前）该被检查：它们永远不该读到那些 key——读到就会得 NaN。
    """
    g = torch.Generator().manual_seed(seed)
    shape = [batch, seq, heads, d]
    q, k, v = _randn(g, shape), _randn(g, shape), _randn(g, shape)
    nb = seq // block_kv
    blk = (torch.arange(seq) // block_kv).float()[None, :, None, None]
    if kind == "sharp":
        q = q * 4.0
    elif kind == "growing":
        k = k * (1.0 + blk)
    elif kind == "first_block_max":
        k = torch.where(blk == 0, k * 4.0, k)
    elif kind == "large":
        q = (torch.rand(shape, generator=g) * 2 - 1) * 4.0
        k = (torch.rand(shape, generator=g) * 2 - 1) * 4.0
    elif kind == "all_negative":
        q = torch.full(shape, 4.0)
        k = -(1.0 + 0.25 * k)
    elif kind == "staircase":
        p = torch.arange(seq) % block_kv
        dim = torch.arange(d)
        q = (
            (512.0 * (dim[None, :] > p[:, None]).float())[None, :, None, :]
            .expand(shape)
            .clone()
        )
        k = (dim[None, :] == p[:, None]).float()[None, :, None, :].expand(shape).clone()
    elif kind == "block_ramp":
        q = torch.zeros(shape)
        k = torch.zeros(shape)
        q[..., 0] = 64.0
        k[..., 0] = 8.0 * blk[..., 0]
    elif kind == "uniform":
        q = torch.zeros(shape)
    elif kind == "poison_future":
        owner = torch.arange(batch * heads).reshape(batch, 1, heads)  # [B, 1, H]
        last_block = (3 * owner) % nb
        future = (
            torch.arange(seq)[None, :, None] // block_kv
        ) > last_block  # [B, S, H]
        k = k.masked_fill(future[..., None], float("nan"))
        v = v.masked_fill(future[..., None], float("nan"))
    elif kind != "normal":
        raise ValueError(kind)
    return q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)


def checked_rows(
    kind: str, batch: int = 4, seq: int = 1024, heads: int = 32, block_kv: int = 128
) -> torch.Tensor:
    """``[B, S, H]`` bool：该 kind 下哪些输出行该被检查（其余行读了 NaN，内容不作要求）。"""
    if kind != "poison_future":
        return torch.ones(batch, seq, heads, dtype=torch.bool)
    nb = seq // block_kv
    owner = torch.arange(batch * heads).reshape(batch, 1, heads)
    return (torch.arange(seq)[None, :, None] // block_kv) <= (3 * owner) % nb


def spike_inputs(
    seed: int = 0,
    batch: int = 4,
    seq: int = 1024,
    heads: int = 32,
    d: int = 128,
    amplitude: float = 512.0,
):
    """每个 q 行恰好命中一个**可见** key（因果 one-hot softmax）：返回 ``(Q, K, V, target)``，``target[b, i, h] <= i``。

    K 的每一行是「双热」编码（两个不同维度各为 1，其余为 0；各行编码互不相同，每个 (batch, head) 各自随机分配）——
    自内积 2，任意两行内积 <= 1。``Q[i] = amplitude * K[target[i]]``，命中 key 的分数比任何其他 key 高
    ``amplitude / sqrt(d)``（默认 45.25），其余权重 <= exp(-45) ≈ 3e-20，输出 ``O[i] = V[target[i]]``（到 FP32 舍入）。
    ``target`` 在 ``[0, i]`` 里均匀随机：覆盖每个块内位置、每个核、每个 AIV 半区、每个 K/V 缓存槽。
    """
    g = torch.Generator().manual_seed(seed)
    pairs = torch.tensor([(a, b) for a in range(d) for b in range(a + 1, d)])
    k = torch.zeros(batch, seq, heads, d)
    target = torch.empty(batch, seq, heads, dtype=torch.long)
    for bi in range(batch):
        for hi in range(heads):
            order = torch.randperm(len(pairs), generator=g)[:seq]
            ab = pairs[order]
            rows = torch.arange(seq)
            k[bi, rows, hi, ab[:, 0]] = 1.0
            k[bi, rows, hi, ab[:, 1]] = 1.0
            target[bi, :, hi] = (
                (torch.rand(seq, generator=g) * (rows + 1))
                .long()
                .clamp(max=seq - 1)
                .minimum(rows)
            )
    idx = target[..., None].expand(batch, seq, heads, d)
    q = amplitude * torch.gather(k, 1, idx)
    v = torch.randn(batch, seq, heads, d, generator=g)
    return q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), target
