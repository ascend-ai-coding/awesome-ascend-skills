# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""参考实现：``O = rmsnorm(X, gamma) @ W``，从数学定义写起，CPU，与行数无关。

数学定义（逐行，沿最后一维）::

    r_i     = sqrt( mean_j(x_ij^2) + eps )
    p_ij    = x_ij / r_i * gamma_j
    O_ij    = sum_k fp16(p_ik) * w_kj

量化边界与库内整除版 ``examples/algorithms/rmsnorm_gemm/reference.py`` 同规则：

- ``x`` / ``gamma`` / ``w`` 本来就是 FP16（算子的真实输入精度），不再量化；
- rmsnorm 内部（均方、``+eps``、``sqrt``、``/rms``、``*gamma``）用 float64 中间精度；
- **只有一处量化**：``p`` 转 FP16——那是 kernel 真实存在的步骤（cube 输入精度），
  参考必须吃同一个量化后的值，否则误差会被错记成「kernel 算错」
  （references/tooling/examples.md「低精度 dtype 的参考必须吃量化后的值」）；
- 矩阵乘吃 FP16 值、累加精度高于硬件的 FP32（这里用 float64，再落回 float32），
  不是又一次量化步骤。整除版用 float32 CPU matmul；这里换 float64 只是让参考自己的
  累加误差远小于被测对象（~1e-16 相对 vs FP32 累加 ~1e-7 相对），容差账里就不用再
  给参考留份额。

不复现 kernel 的分核 / 分带 / 半带 / splitk / UNPK 顺序——参考对行数没有任何假设，
尾块、行归属、分核偏移是否算对只能由它与 kernel 输出的逐行比较来验，而不是抄一遍分块。
"""

import torch

EPS = 1e-6  # 必须与 kernel.py 的编译期常量一致


def rmsnorm_gemm_reference(
    x: torch.Tensor, gamma: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """``x``: ``[rows, K]`` FP16；``gamma``: ``[K]`` 或可广播到 ``[1, K]``；``w``: ``[K, N]`` FP16。

    返回 ``[rows, N]`` float32。
    """
    x64 = x.double()
    gamma64 = gamma.double().reshape(1, -1)
    mean_sq = (x64 * x64).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + EPS)
    normed = (x64 / rms) * gamma64
    normed_fp16 = normed.to(torch.float16)  # kernel 实际喂给 cube 的精度
    return torch.matmul(normed_fp16.double(), w.double()).float()


def owner(
    row: int, *, rows_per_core: int, band_rows: int, half_band: int, num_bands: int
) -> dict:
    """行 ``row`` 由谁算：与 kernel.py 的标量算式同形（core / band / AIV / 是否尾带）。"""
    core = row // rows_per_core
    local = row - core * rows_per_core
    band = local // band_rows
    within = local - band * band_rows
    this_band_rows = min(band_rows, rows_per_core - band * band_rows)
    return {
        "core": core,
        "band": band,
        "aiv": 0 if within < half_band else 1,
        "tail_band": band == num_bands - 1,
        "band_rows": this_band_rows,
    }


def bad_mask(
    out: torch.Tensor, golden: torch.Tensor, rtol: float, atol: float
) -> torch.Tensor:
    """逐元素判据，与 ``torch.testing.assert_close`` 同式：``|a-b| <= atol + rtol*|b|``。

    NaN 一律算不合格（没被写到的行是 NaN 预填充，会在这里现形）。
    """
    ok = (out - golden).abs() <= atol + rtol * golden.abs()
    return ~ok


def _fp16_neighbors(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """P 元素往上 / 往下一个 FP16 的差（float64）。"""
    inf = torch.full_like(p, float("inf"))
    up = torch.nextafter(p, inf).double() - p.double()
    dn = torch.nextafter(p, -inf).double() - p.double()
    return up, dn


def quantized_p(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """参考实现里喂给 cube 的 FP16 值 P（与 reference.py 同式）。"""
    x64 = x.double()
    normed = (
        x64
        / torch.sqrt((x64 * x64).mean(dim=-1, keepdim=True) + EPS)
        * gamma.double().reshape(1, -1)
    )
    return normed.to(torch.float16)


def unexplained_rows(
    out, golden, x, gamma, w, *, noise: float, max_flips: int = 3
) -> list[dict]:
    """返回**不能**用「至多 ``max_flips`` 个 P 元素翻转到相邻 FP16」解释的行。

    背景：kernel 在 FP32 里算 ``x / rms * gamma`` 再窄化 FP16，参考在 float64 里算再窄化。
    两者只在结果恰好落在 FP16 舍入边界附近时差一个 ulp——这一个元素 ``P[i,k]`` 的翻转让
    ``O[i,:]`` 差 ``delta * W[k,:]``，误差向量是 W 的一行的倍数。逐行贪心：找最能吃掉误差向量
    的（k, 上/下）翻转，减掉，重复；剩下的最大绝对值 ≤ ``noise``（FP32 累加噪声）即算解释了。
    任何结构性错误（行错位、用了旧带的 P、漏写）都不可能被这样解释。
    """
    p = quantized_p(x, gamma)
    w64 = w.double()
    err = out.double() - golden.double()
    bad: list[dict] = []
    for i in (
        (torch.nan_to_num(err.abs(), nan=1e9).amax(dim=1) > noise)
        .nonzero()
        .flatten()
        .tolist()
    ):
        e = err[i].clone()
        up, dn = _fp16_neighbors(p[i])
        flips = 0
        while flips < max_flips and (
            not torch.isfinite(e).all() or e.abs().max() > noise
        ):
            best = None
            for delta in (up, dn):
                cand = delta.reshape(-1, 1) * w64  # [K, N]
                res = (cand - e.reshape(1, -1)).abs().amax(dim=1)  # [K]
                k = int(res.argmin())
                if best is None or float(res[k]) < best[0]:
                    best = (float(res[k]), k, float(delta[k]))
            _, k, d = best
            e = e - d * w64[k]
            flips += 1
        residual = float(torch.nan_to_num(e.abs(), nan=1e9).max())
        if residual > noise:
            bad.append(
                {
                    "row": i,
                    "residual": residual,
                    "max_err": float(torch.nan_to_num(err[i].abs(), nan=1e9).max()),
                }
            )
    return bad


def assert_rows(
    out: torch.Tensor,
    golden: torch.Tensor,
    r0: int,
    r1: int,
    what: str,
    *,
    plan: dict,
    rtol: float,
    atol: float,
) -> None:
    """比较行区间 [r0, r1)；不合格时抛 AssertionError，信息直接指认首个坏行属于哪个核 / 带 / AIV。"""
    bad = (
        bad_mask(out[r0:r1], golden[r0:r1], rtol, atol).any(dim=1).nonzero().flatten()
        + r0
    )
    if bad.numel() == 0:
        return
    first = int(bad[0])
    own = owner(first, **plan)
    nan_rows = int(torch.isnan(out[r0:r1]).any(dim=1).sum())
    finite = torch.nan_to_num((out[r0:r1] - golden[r0:r1]).abs(), nan=0.0)
    where = (
        f"核 {own['core']} / 带 {own['band']}（{'尾带，' if own['tail_band'] else '整带，'}"
        f"{own['band_rows']} 行）/ AIV{own['aiv']}"
    )
    if nan_rows:
        hint = (
            "有行仍是 NaN 预填充——这些行根本没被写：查 store 的有效行数（gemm m= / acc 收窄）、"
            "半带 load / insert 的行数是否偏小"
        )
    else:
        hint = (
            "行都写了但值不对：查行归属算式 core_id * ROWS_PER_CORE + band_off + off、"
            "AIV1 的半带偏移，或读到了上一带残留的 P / X 行"
        )
    raise AssertionError(
        f"尾块处理错误 [{what}]：行 [{r0},{r1}) 中 {bad.numel()} 行不合格，首个坏行 {first}"
        f"（属于 {where}）；其中 NaN（未写）{nan_rows} 行；有限值最大绝对误差 {float(finite.max()):.3e}。{hint}"
    )
