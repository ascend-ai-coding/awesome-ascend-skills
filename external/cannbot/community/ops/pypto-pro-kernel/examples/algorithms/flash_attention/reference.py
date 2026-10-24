# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU 参考：单头注意力 ``O = softmax(Q K^T / sqrt(d)) V``，从数学定义写起（float64）。

数学定义（与 kernel 的分块 / 在线 rescale / 精度细节无关）::

    S[i, j] = (sum_d Q[i, d] * K[j, d]) / sqrt(d)
    P[i, j] = exp(S[i, j]) / sum_j' exp(S[i, j'])            （沿 KV 维 j 的行 softmax）
    O[i, e] = sum_j P[i, j] * V[j, e]

本文件的口径
------------
- 输入 ``Q`` ``K`` ``V`` 本来就是 FP16（算子的真实输入），提升到 float64 是精确的。**参考不量化任何中间量**：
  softmax 用 float64 全精度，减行最大值只为不溢出（softmax 平移不变，数学等价）。kernel 自己做了哪些舍入（P 转 FP16
  喂 cube、FP32 累加、在线 rescale），由 ``error_terms`` 逐项记账，不靠调出来的阈值。
- ``error_bound`` 给出逐元素的**先验**误差上界（只看输入，不看 kernel 输出）。各项来源与假设写在 ``error_terms`` 里。
  判据尺度是项和 ``S = sum_j P_ij |V_je|``，不是 ``|O|``：``O`` 是带符号加权和，抵消位置 ``|O|`` 远小于 ``S``
  （api/numerics.md 一）。
- 后半部分是「错结构」参考（漏 rescale、漏缩放系数等）：它们必须**通不过**同一个判据——证明判据分得清结构性错误，
  而不是宽到什么都能过。最后是各类输入的构造（``make_inputs`` 等）。
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


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().to(torch.float64)


def scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """``S = Q K^T / sqrt(d)``，float64。"""
    return (_f64(q) @ _f64(k).T) / math.sqrt(q.shape[-1])


def softmax_weights(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """行 softmax 权重 ``P``，float64；减行最大值只为数值上不溢出。"""
    s = scores(q, k)
    e = torch.exp(s - s.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)


def attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """数学真值 ``O = softmax(Q K^T / sqrt(d)) V``，float64。**不是**任何 kernel 的分块顺序。"""
    return softmax_weights(q, k) @ _f64(v)


def error_scale(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """项和尺度 ``S[i, e] = sum_j P[i, j] * |V[j, e]|``（``[M, d]``）。所有误差项都正比于它。"""
    return softmax_weights(q, k) @ _f64(v).abs()


def error_terms(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_kv: int
) -> dict[str, torch.Tensor]:
    """先验误差上界的各项（都是 ``[M, d]`` 的逐元素上界，单位与 ``O`` 相同），逐项分开返回。

    记 ``c = 1/sqrt(d)``，``P`` 为精确 softmax 权重，``S = P @ |V|``（``error_scale``），``nb`` 为 KV 块数。
    kernel 的舍入来源与对应的项：

    ``quant_p``  —— P 转 FP16 喂第二个 matmul（cube 的输入精度约束，不是可选项）。每个元素就近舍入，
        相对误差 <= 2^-11（正常区）；P <= 1 的次正规区（P < 2^-14）改为绝对误差 <= 2^-25。二者取并集：
        ``|dP| <= 2^-11 * P + 2^-25``。归一化用的行和 ``l`` 由**未量化**的 FP32 P 求和，所以只有分子被量化：
        ``|dO| <= 2^-11 * S + 2^-25 * sum_j |V[j, e]| / l``，其中 ``l >= 1`` 是以行最大为基准的 exp 和
        （最大元素贡献 exp(0) = 1）。是**主导项**。
    ``score``    —— Q K^T 由 cube 以 FP32 累加（FP16 x FP16 乘积在 FP32 里精确，只有加法舍入）。Higham 的 gamma 界对
        任意求和顺序成立：``|dS_ij| <= c * gamma_d * T_ij``，``T_ij = sum_e |Q_ie||K_je|``，
        ``gamma_d = d * 2^-24 / (1 - d * 2^-24)``。行内所有 score 共同带来的权重相对变化 <= ``exp(2 * delta_i) - 1``，
        ``delta_i = c * gamma_d * max_j T_ij``（softmax 对 score 的偏移取差，最坏两端各 delta）。
    ``exp_arg``  —— 指数的自变量 ``(s - m) * c`` 由 FP32 的「减、乘」得到，每次相对舍入 <= 2^-24；自变量绝对误差
        <= 3 * 2^-24 * |a|（``a = s - m_final`` 的缩放值）。在线 rescale 使一个元素的自变量分成若干段（块内
        ``s - m_j`` 与各次 ``m_{j-1} - m_j``），各段同号（running max 单调不降），绝对值之和恰为 ``|a|``，所以总自变量
        误差仍 <= ``3 * 2^-24 * |a|``。对 O 的影响：``3 * 2^-24 * (A + Abar * S)``，``A = sum_j P_j |a_j| |V_je|``，
        ``Abar = sum_j P_j |a_j|``（分母的加权平均误差）。
    ``exp_fn``   —— 硬件 exp 函数本身的相对误差，先验假设 <= 2^-22（``EPS_EXP``），每次 exp 调用一次；一个元素最多
        经历 ``nb`` 次（本块的 p 与至多 nb-1 次 alpha），分子分母各一份：``2 * nb * 2^-22 * S``。
    ``accum``    —— 分子 ``P V`` 的 FP32 累加（cube 块内 K = block_kv 项 + 每块「乘 alpha、加」2 次 + 末尾除法 2 次的
        舍入），分母 ``l`` 的累加（寄存器内逐元素相加 block_kv 项 + 每块 2 次）：
        ``(gamma_{block_kv + 2 nb + 2} + gamma_{block_kv + 2 nb}) * S``（Higham gamma，任意求和顺序；两个
        gamma 各取一个宽松的项数上界，不依赖具体归约树形状）。

    Each term is input-dependent under the assumptions above; the comparator checks their sum.
    """
    d = q.shape[-1]
    n = k.shape[0]
    nb = n // block_kv
    c = 1.0 / math.sqrt(d)

    s = scores(q, k)
    m = s.max(dim=-1, keepdim=True).values
    e = torch.exp(s - m)
    l_max_rel = e.sum(dim=-1, keepdim=True)  # 以行最大为基准的 exp 和，>= 1
    w = e / l_max_rel  # 精确 softmax 权重
    v_abs = _f64(v).abs()
    scale_s = w @ v_abs  # S[i, e]

    def gamma(count: int) -> float:
        return count * U32 / (1.0 - count * U32)

    t = _f64(q).abs() @ _f64(k).abs().T  # T_ij
    delta = c * gamma(d) * t.max(dim=-1, keepdim=True).values  # [M, 1]
    a_abs = (m - s).abs()  # |a_ij| = |s_ij - m_i|（已含 1/sqrt(d)）
    a_weighted = (w * a_abs) @ v_abs  # A[i, e]
    a_bar = (w * a_abs).sum(dim=-1, keepdim=True)  # Abar[i, 1]

    return {
        "quant_p": U16 * scale_s
        + FP16_SUBNORMAL_HALF_ULP * v_abs.sum(dim=0)[None, :] / l_max_rel,
        "score": torch.expm1(2.0 * delta) * scale_s,
        "exp_arg": 3.0 * U32 * (a_weighted + a_bar * scale_s),
        "exp_fn": 2.0 * nb * EPS_EXP * scale_s,
        "accum": (gamma(block_kv + 2 * nb + 2) + gamma(block_kv + 2 * nb)) * scale_s,
    }


def error_bound(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_kv: int
) -> torch.Tensor:
    """``error_terms`` 的和：逐元素先验上界 ``|O_kernel - O_exact| <= bound``。"""
    return sum(error_terms(q, k, v, block_kv).values())


# ---------------------------------------------------------------------------------------------
# 「错结构」参考：每一个都是一类真实的在线 softmax 实现错误，必须通不过 error_bound 的判据
# ---------------------------------------------------------------------------------------------


def unscaled_reference(q, k, v):
    """漏掉 1/sqrt(d) 缩放。"""
    s = _f64(q) @ _f64(k).T
    e = torch.exp(s - s.max(dim=-1, keepdim=True).values)
    return (e / e.sum(dim=-1, keepdim=True)) @ _f64(v)


def wrong_scale_reference(q, k, v):
    """缩放系数写成 1/d（把 sqrt 漏了）。"""
    s = (_f64(q) @ _f64(k).T) / q.shape[-1]
    e = torch.exp(s - s.max(dim=-1, keepdim=True).values)
    return (e / e.sum(dim=-1, keepdim=True)) @ _f64(v)


def blockwise_softmax_reference(q, k, v, block_kv: int):
    """每个 KV 块各自做 softmax（块内最大、块内和）再平均——漏掉跨块 rescale 的经典错误。"""
    s = scores(q, k)
    out = torch.zeros(q.shape[0], v.shape[1], dtype=torch.float64)
    nb = k.shape[0] // block_kv
    for j in range(nb):
        sj = s[:, slice(j * block_kv, (j + 1) * block_kv)]
        pj = torch.exp(sj - sj.max(dim=-1, keepdim=True).values)
        pj = pj / pj.sum(dim=-1, keepdim=True)
        out += pj @ _f64(v)[slice(j * block_kv, (j + 1) * block_kv)]
    return out / nb


def no_o_rescale_reference(q, k, v, block_kv: int):
    """在线 softmax 里 l 正确 rescale、但 O 累加漏掉 alpha（``O += P_j V_j`` 而不是 ``O = alpha O + P_j V_j``）。"""
    s = scores(q, k)
    nb = k.shape[0] // block_kv
    m = torch.full([q.shape[0], 1], float("-inf"), dtype=torch.float64)
    rowsum = torch.zeros(q.shape[0], 1, dtype=torch.float64)
    out_acc = torch.zeros(q.shape[0], v.shape[1], dtype=torch.float64)
    for j in range(nb):
        sj = s[:, slice(j * block_kv, (j + 1) * block_kv)]
        m_new = torch.maximum(m, sj.max(dim=-1, keepdim=True).values)
        pj = torch.exp(sj - m_new)
        rowsum = torch.exp(m - m_new) * rowsum + pj.sum(dim=-1, keepdim=True)
        out_acc = (
            out_acc + pj @ _f64(v)[slice(j * block_kv, (j + 1) * block_kv)]
        )  # 漏 alpha
        m = m_new
    return out_acc / rowsum


def no_l_rescale_reference(q, k, v, block_kv: int):
    """O 正确 rescale、但行和 l 漏掉 alpha（``l += rowsum(P_j)``）。"""
    s = scores(q, k)
    nb = k.shape[0] // block_kv
    m = torch.full([q.shape[0], 1], float("-inf"), dtype=torch.float64)
    rowsum = torch.zeros(q.shape[0], 1, dtype=torch.float64)
    out_acc = torch.zeros(q.shape[0], v.shape[1], dtype=torch.float64)
    for j in range(nb):
        sj = s[:, slice(j * block_kv, (j + 1) * block_kv)]
        m_new = torch.maximum(m, sj.max(dim=-1, keepdim=True).values)
        pj = torch.exp(sj - m_new)
        alpha = torch.exp(m - m_new)
        rowsum = rowsum + pj.sum(dim=-1, keepdim=True)  # 漏 alpha
        out_acc = (
            alpha * out_acc + pj @ _f64(v)[slice(j * block_kv, (j + 1) * block_kv)]
        )
        m = m_new
    return out_acc / rowsum


# ---------------------------------------------------------------------------------------------
# 输入构造
# ---------------------------------------------------------------------------------------------

KINDS = (
    "normal",
    "sharp",
    "growing",
    "first_block_max",
    "last_block_max",
    "large",
    "all_negative",
)


def make_inputs(
    seed: int = 0,
    kind: str = "normal",
    m: int = 1024,
    n: int = 1024,
    d: int = 128,
    block_kv: int = 128,
):
    """FP16 的 ``(Q, K, V)``。``kind``（每一类覆盖在线 softmax 的一条不同路径）：

    - ``normal``：全部 N(0, 1)（缩放后 logits 标准差约 1，softmax 较平）。
    - ``sharp``：Q 放大 4 倍（logits 标准差约 4，softmax 尖）。
    - ``growing``：第 j 个 KV 块的 K 放大 ``1 + j``——后面的块分数更大，running max 每块都更新（alpha 远离 1）。
    - ``first_block_max``：只有第 0 块的 K 放大 4 倍——max 在第一块之后不再变（alpha 恒为 1，只走「不更新」分支；
      漏 rescale 类错误在这一类上不可见，那是它的定义，不是缺陷）。
    - ``last_block_max``：只有最后一块的 K 放大 4 倍——前面所有块的累积量在最后一块被 alpha 压到接近 0。
    - ``large``：Q、K 取 [-4, 4] 均匀（分数到几百，不减最大值 exp 会上溢，减了之后大部分 P 下溢为 0）。
    - ``all_negative``：``Q = 4``、``K = -(1 + 0.25 randn)``——所有分数约 -45：running max 初值若误取 0，
      ``exp(-45)`` 在 FP16 里全为 0，输出整体为 0。
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn([m, d], generator=g)
    k = torch.randn([n, d], generator=g)
    v = torch.randn([n, d], generator=g)
    nb = n // block_kv
    if kind == "sharp":
        q = q * 4.0
    elif kind == "growing":
        k = (
            k
            * torch.repeat_interleave(
                1.0 + torch.arange(nb, dtype=torch.float32), block_kv
            )[:, None]
        )
    elif kind == "first_block_max":
        k = k.clone()
        k[:block_kv] *= 4.0
    elif kind == "last_block_max":
        k = k.clone()
        k[slice((nb - 1) * block_kv, None)] *= 4.0
    elif kind == "large":
        q = (torch.rand([m, d], generator=g) * 2 - 1) * 4.0
        k = (torch.rand([n, d], generator=g) * 2 - 1) * 4.0
    elif kind == "all_negative":
        q = torch.full([m, d], 4.0)
        k = -(1.0 + 0.25 * k)
    elif kind != "normal":
        raise ValueError(kind)
    return q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)


def uniform_inputs(seed: int = 0, m: int = 1024, n: int = 1024, d: int = 128):
    """``Q = 0``：所有分数恰为 0，softmax 是均匀分布，``O[i] = mean_j V[j]``（对所有 i）。"""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn([n, d], generator=g)
    v = torch.randn([n, d], generator=g)
    return (
        torch.zeros([m, d], dtype=torch.float16),
        k.to(torch.float16),
        v.to(torch.float16),
    )


def spike_inputs(
    seed: int = 0, m: int = 1024, n: int = 1024, d: int = 128, amplitude: float = 512.0
):
    """每个 q 行恰好命中一个 key（one-hot softmax）：返回 ``(Q, K, V, target)``，``target[i]`` 是行 ``i`` 命中的 key。

    K 的每一行是「双热」编码（两个不同维度各为 1，其余为 0；各行编码互不相同）——自内积 2，任意两行内积 <= 1。
    ``Q[i] = amplitude * K[target[i]]``，于是命中 key 的分数比任何其他 key 高 ``amplitude / sqrt(d)``（默认 45.25），
    其余权重 <= exp(-45) ≈ 3e-20，输出 ``O[i] = V[target[i]]``（到 FP32 舍入）。``target`` 是 ``[0, n)`` 的置换
    （``m == n``）：每个 key 恰被一行命中，所以每个块内位置、每个核、每个 AIV 半区都被覆盖。
    """
    assert m == n, "spike_inputs 要求 m == n（target 是置换）"
    g = torch.Generator().manual_seed(seed)
    pairs = [(a, b) for a in range(d) for b in range(a + 1, d)]
    order = torch.randperm(len(pairs), generator=g)[:n].tolist()
    k = torch.zeros([n, d])
    for row, idx in enumerate(order):
        a, b = pairs[idx]
        k[row, a] = 1.0
        k[row, b] = 1.0
    target = torch.randperm(n, generator=g)
    q = amplitude * k[target]
    v = torch.randn([n, d], generator=g)
    return q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), target
