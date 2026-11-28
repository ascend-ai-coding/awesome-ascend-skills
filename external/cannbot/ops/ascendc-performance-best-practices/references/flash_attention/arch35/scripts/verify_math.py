# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU float64 controls: prove reduction identities and detect broken alternatives.
This validates mathematical necessity, not Ascend instruction behavior or speed.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def full(s, v):
    m = s.max(axis=1)
    p = np.exp(s - m[:, None])
    lsum = p.sum(axis=1)
    return p @ v / lsum[:, None], m + np.log(lsum)


def online(s, v, chunks, mutation=None):
    m = np.full(s.shape[0], -np.inf)
    lsum = np.zeros(s.shape[0])
    a = np.zeros((s.shape[0], v.shape[1]))
    for ids in chunks:
        score = s[:, ids]
        new = np.maximum(m, score.max(axis=1))
        alpha = np.exp(m - new)
        p = np.exp(score - new[:, None])
        a = a * (1 if mutation == "drop_output_rescale" else alpha[:, None]) + p @ v[ids]
        lsum = lsum * (1 if mutation == "drop_sum_rescale" else alpha) + p.sum(axis=1)
        m = new
    return a if mutation == "drop_final_divide" else a / lsum[:, None], m + np.log(lsum)


def partial(s, v, ids):
    ss = s[:, ids]
    m = ss.max(axis=1)
    p = np.exp(ss - m[:, None])
    lsum = p.sum(axis=1)
    return m, lsum, p @ v[ids] / lsum[:, None]


def merge(parts, mutation=None):
    ms = np.stack([p[0] for p in parts])
    ls = np.stack([p[1] for p in parts])
    os = np.stack([p[2] for p in parts])
    m = ms.max(axis=0)
    e = np.exp(ms - m)
    mass = e * ls
    lsum = mass.sum(axis=0)
    if mutation == "mean_partials":
        out = os.mean(axis=0)
    elif mutation == "ignore_local_sum":
        out = (e[:, :, None] * os).sum(axis=0) / e.sum(axis=0)[:, None]
    elif mutation == "treat_normalized_as_numerator":
        out = (e[:, :, None] * os).sum(axis=0) / lsum[:, None]
    elif mutation == "skip_merge":
        out = os[0]
    else:
        out = (mass[:, :, None] * os).sum(axis=0) / lsum[:, None]
    return out, m + np.log(lsum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    lengths = [112, 112, 112, 17]
    bounds = np.cumsum([0] + lengths)
    chunks = [np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
    s = rng.normal(size=(3, sum(lengths)))
    for ids, shift in zip(chunks, [0, 12, -7, 4]):
        s[:, ids] += shift
    # Values have different means per partition; equal averaging must be observable.
    v = rng.normal(size=(s.shape[1], 8))
    for ids, shift in zip(chunks, [4, -3, 8, 1]):
        v[ids] += shift
    ref, lref = full(s, v)
    tests = []

    def record(name, result, should_pass):
        out, lsum = result
        err = float(np.max(np.abs(out - ref)))
        le = float(np.max(np.abs(lsum - lref)))
        matched = bool(np.isfinite(out).all() and np.isfinite(lsum).all() and err < 1e-10 and le < 1e-10)
        tests.append(
            dict(
                name=name,
                max_abs_output_error=err,
                max_abs_lse_error=le,
                expected_match=should_pass,
                matched=matched,
                experiment_pass=matched == should_pass,
            )
        )

    record("online_112_112_112_17", online(s, v, chunks), True)
    record("online_different_order", online(s, v, [chunks[i] for i in [2, 0, 3, 1]]), True)
    parts = [partial(s, v, x) for x in chunks]
    record("weighted_flashdecode", merge(parts), True)
    record("weighted_flashdecode_reversed", merge(parts[::-1]), True)
    for bad in ["drop_output_rescale", "drop_sum_rescale", "drop_final_divide"]:
        record(bad, online(s, v, chunks, bad), False)
    for bad in ["mean_partials", "ignore_local_sum", "treat_normalized_as_numerator", "skip_merge"]:
        record(bad, merge(parts, bad), False)
    # Masked padding must contribute exp(-inf)=0, not exp(0) or max=0.
    pad = 31
    padded_s = np.pad(s, ((0, 0), (0, pad)), constant_values=-np.inf)
    padded_v = np.pad(v, ((0, pad), (0, 0)), constant_values=19)
    record("masked_tail_neutral", full(padded_s, padded_v), True)
    broken_s = padded_s.copy()
    broken_s[:, -pad:] = 12
    record("padding_included_in_reduce", full(broken_s, padded_v), False)
    stress = np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32)
    vv = np.array([[1.0], [3.0], [-1.0]], dtype=np.float32)
    stable = full(stress, vv)[0]
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.exp(stress)
        unstable = (raw @ vv) / raw.sum(axis=1)[:, None]
    stability = dict(
        stable_finite=bool(np.isfinite(stable).all()),
        unshifted_exp_finite=bool(np.isfinite(unstable).all()),
        stable_output=stable.tolist(),
    )
    tiny_s = np.array([[0.0, 0.0, np.log(3.0)]])
    tiny_v = np.array([[1.0], [1.0], [10.0]])
    tiny_parts = [partial(tiny_s, tiny_v, np.array([0, 1])), partial(tiny_s, tiny_v, np.array([2]))]
    hand_example = {
        n: float(merge(tiny_parts, mode)[0][0, 0])
        for n, mode in [("correct", None), ("mean", "mean_partials"), ("without_local_sum", "ignore_local_sum")]
    }
    assert np.allclose(list(hand_example.values()), [6.4, 5.5, 7.75])
    assert all(t["experiment_pass"] for t in tests)
    assert stability["stable_finite"] and not stability["unshifted_exp_finite"]
    result = dict(
        scope="CPU mathematical necessity controls, not kernel ablations",
        seed=42,
        shape=[3, 353, 8],
        chunks=lengths,
        tests=tests,
        overflow_control=stability,
        hand_example=hand_example,
        all_expected_outcomes_observed=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
