# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""cancellation_stress input generation for test_inputs.py.

Copy this into ``custom/<op>/eval/test_inputs.py`` (or import it as a helper). It
implements the contract from verification.md §B.6: given a case dict with a
"cancellation_stress" knob, engineer inputs such that a named subtractive pair
(A - B) inside the module computes two operands that are within
relative_gap = |A - B| / max(|A|, |B|) of each other.

Strategy: random-restart over a per-op knob set (gate magnitude, input scale,
dht scale, etc.). Deterministic under the provided seed. No external deps beyond
torch. ~50 lines total; tune the probe callable per op.

HOW IT FITS INTO make_inputs::

    def make_inputs(case: dict) -> dict:
        inputs = _canonical_inputs(case)         # your existing path
        if "cancellation_stress" in case:
            inputs = apply_cancellation_stress(case, inputs, probe=_probe_ds_new)
        return inputs

where ``_probe_ds_new(inputs)`` returns (A, B) for the expression you care about.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_MAX_RESTARTS = 64          # random restarts per case
_KNOB_SEARCH_SCALES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]  # multiplicative scan over scale knobs


def apply_cancellation_stress(
    case: dict,
    inputs: dict,
    probe,                  # callable: inputs_dict -> (A: Tensor, B: Tensor)
    knobs: tuple[str, ...] = ("gate_scale", "weight_scale", "dht_scale"),
) -> dict:
    """
    Mutate `inputs` so the pair returned by `probe(inputs)` has
        |A - B| / max(|A|, |B|)  <=  case["cancellation_stress"]["relative_gap"]
    via random restart over the listed knobs. Deterministic under
    case["cancellation_stress"]["seed"].

    If the search budget is exhausted without hitting the target gap, the inputs
    that achieved the *closest* approach are returned and the caller should tag
    the case id with "_bestfound" so @pypto-op-debugger sees it did not fully converge.
    """
    cfg = case["cancellation_stress"]
    target_gap = float(cfg["relative_gap"])
    seed = int(cfg.get("seed", 0))

    g = torch.Generator().manual_seed(seed)
    best_gap = float("inf")
    best_inputs = inputs

    for _ in range(_MAX_RESTARTS):
        trial = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}

        # 1. Pick a scale value per knob uniformly at random from the scan grid.
        for knob in knobs:
            if knob not in trial:
                continue   # op doesn't expose this knob — ignore
            scale_idx = torch.randint(0, len(_KNOB_SEARCH_SCALES), (1,), generator=g).item()
            scale = _KNOB_SEARCH_SCALES[scale_idx]
            if torch.is_tensor(trial[knob]):
                trial[knob] = trial[knob] * scale
            else:
                trial[knob] = trial[knob] * scale

        # 2. Probe the subtractive pair under these trial inputs.
        try:
            a, b = probe(trial)
        except Exception:
            continue   # trial inputs degenerate — skip

        # 3. Compute the relative gap (worst-cell over the pair).
        denom = torch.maximum(torch.abs(a), torch.abs(b)).clamp_min(1e-30)
        gap_tensor = torch.abs(a - b) / denom
        gap = float(gap_tensor.max().item())

        if gap < best_gap:
            best_gap = gap
            best_inputs = trial

        if gap <= target_gap:
            return trial

    # Budget exhausted — return best and signal via case id suffix if caller wants.
    # The caller is responsible for renaming case["id"] with "_bestfound_<gap>" if
    # best_gap > target_gap; we don't mutate case here.
    return best_inputs


# ---------------------------------------------------------------------------
# Example probe (gated delta rule backward — d_s_new = A + B - C, stress A vs C)
# ---------------------------------------------------------------------------

def _probe_ds_new_example(inputs: dict):
    """Example probe for gated_delta_rule_backward's M2 d_s_new subtraction.

    Computes partial golden to obtain the (q_eff @ doc * scale) and
    (w^T @ dv_total) tensors, returns them as the (a, c) pair whose
    cancellation we want to stress.

    Replace this per op — the probe only needs to return *the two operands of
    the subtraction you care about*, computed with pure torch ops.
    """
    import math
    q_norm = inputs["q_norm"]  # [b, t, h, k] per op's existing contract
    k_norm = inputs["k_norm"]
    do = inputs["do"]
    w = inputs["w"]
    dht = inputs["dht"]
    g_raw = inputs["g_raw"]
    b, t, h, k = q_norm.shape
    v = do.shape[-1]
    bt = inputs.get("BT", 64)
    nt = t // bt
    scale = 1.0 / math.sqrt(k)

    # Take the last chunk only — cancellation in the last chunk is sufficient
    # for stress testing and avoids re-running the full nt loop in the probe.
    chunk = nt - 1
    t0 = chunk * bt
    qc = q_norm[:, t0:t0 + bt].transpose(1, 2).reshape(b * h, bt, k)
    doc = do[:, t0:t0 + bt].transpose(1, 2).reshape(b * h, bt, v)

    g_cum = torch.cumsum(g_raw[:, t0:t0 + bt].transpose(1, 2), dim=-1).reshape(b * h, bt)
    eg = torch.exp(g_cum)
    q_eff = qc * eg.unsqueeze(-1)
    a = torch.matmul(q_eff.transpose(-1, -2), doc) * scale   # [b*h, k, v]

    # For the w^T @ dv_total side, take dht directly as a stand-in for d_s
    # (last-chunk analytical approximation; close enough for gap-search purposes).
    w_chunk = w[:, :, chunk].reshape(b * h, bt, k)
    dv_total_proxy = torch.matmul(
        k_norm[:, t0:t0 + bt].transpose(1, 2).reshape(b * h, bt, k),
        dht.reshape(b * h, k, v),
    )
    c = torch.matmul(w_chunk.transpose(-1, -2), dv_total_proxy)  # [b*h, k, v]

    return a, c


# ---------------------------------------------------------------------------
# Minimal self-test (run:  python cancellation_stress_template.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)
    fake_inputs = {
        "q_norm": torch.randn(1, 128, 4, 128),
        "k_norm": torch.randn(1, 128, 4, 128),
        "do": torch.randn(1, 128, 4, 128),
        "w": torch.randn(1, 4, 2, 64, 128),
        "dht": torch.randn(1, 4, 128, 128),
        "g_raw": torch.randn(1, 128, 4) * 0.1,
        "gate_scale": torch.tensor(1.0),
        "weight_scale": torch.tensor(1.0),
        "dht_scale": torch.tensor(1.0),
        "BT": 64,
    }
    case = {"cancellation_stress": {"target": "d_s_new", "expression": "A - C", "relative_gap": 1e-4, "seed": 1}}
    a, c = _probe_ds_new_example(fake_inputs)
    gap0 = float((torch.abs(a - c) / torch.maximum(torch.abs(a), torch.abs(c)).clamp_min(1e-30)).max())
    logger.info("Before stress:  max relative gap = %.4e", gap0)

    stressed = apply_cancellation_stress(case, fake_inputs, probe=_probe_ds_new_example)
    a2, c2 = _probe_ds_new_example(stressed)
    gap1 = float((torch.abs(a2 - c2) / torch.maximum(torch.abs(a2), torch.abs(c2)).clamp_min(1e-30)).max())
    logger.info(
        "After stress:   max relative gap = %.4e  (target %.1e)",
        gap1,
        case["cancellation_stress"]["relative_gap"],
    )
