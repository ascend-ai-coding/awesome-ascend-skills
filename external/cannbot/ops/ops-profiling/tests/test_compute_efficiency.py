# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Validate compute_efficiency + cost models against the mla_prolog measured baseline."""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import cost_models  # noqa: E402
import compute_efficiency as ce  # noqa: E402

PEAK_BW = 1.9e12
PEAK_FLOPS = 400e12


def mla_cost():
    # Explicit fixture inputs: the production model no longer has an operator registry.
    shapes = [(7168, 1536), (7168, 576), (1536, 18432), (96, 128, 512)]
    return cost_models.compute_cost({"inputs": [
        {"shape": list(shape), "dtype": "bf16"} for shape in shapes
    ]}, flops=3.183e9)


def vector_cost(n, outputs, *, inputs=1, ops_per_elem=1):
    return cost_models.compute_cost({
        "inputs": [{"shape": [n], "dtype": "bf16"}] * inputs,
        "outputs": [{"shape": [outputs], "dtype": "bf16"}],
    }, flops=n * ops_per_elem, unit="vector")


def test_mla_prolog_cost_intrinsics():
    c = mla_cost()  # reference decode defaults
    assert abs(c.total_flops - 3.183e9) / 3.183e9 < 0.01          # 3.18 GFLOP
    weights = (7168 * 1536 + 7168 * 576 + 1536 * 18432 + 96 * 128 * 512) * 2
    assert abs(weights - 99.5e6) / 99.5e6 < 0.01                   # 99.5 MB weights


def test_mla_prolog_mbu_mfu_matches_hand_calc():
    c = mla_cost()
    r = ce.analyze(57.5, c.total_flops, c.total_bytes, PEAK_BW, PEAK_FLOPS)
    assert abs(r["MFU"] - 0.138) < 0.005          # hand calc: 13.8%
    assert r["MBU"] > 0.90                          # near HBM ceiling
    assert r["cost_bound"] == "Memory-Bound"
    assert r["MBU"] >= 0.90 and "fewer bytes" in r["verdict"]


def test_selfdev_c1_matches_prior_numbers():
    c = mla_cost()
    r = ce.analyze(85.72, c.total_flops, c.total_bytes, PEAK_BW, PEAK_FLOPS)
    assert abs(r["MFU"] - 0.093) < 0.005          # prior: ~9%
    assert abs(r["MBU"] - 0.635) < 0.03           # prior wall-MBU ~61%


def test_matmul_mm3_slice():
    c = cost_models.compute_cost({
        "inputs": [{"shape": [32, 1536], "dtype": "bf16"},
                   {"shape": [1536, 18432], "dtype": "bf16"}],
        "outputs": [{"shape": [32, 18432], "dtype": "bf16"}],
    }, flops=2 * 32 * 18432 * 1536)
    assert abs(c.total_flops - 1.812e9) / 1.812e9 < 0.01
    assert abs(c.total_bytes - 57.9e6) / 57.9e6 < 0.02


def test_explicit_flops_bytes_path():
    r = ce.analyze(20.0, 3.18e9, 99.5e6, PEAK_BW, PEAK_FLOPS)
    assert r["MBU"] > 0 and r["MFU"] > 0
    assert r["ceiling_us"] > 0


def test_pure_vector_ops_declare_vector_unit():
    ew = vector_cost(1_000_000, 1_000_000, ops_per_elem=2)
    assert ew.unit == "vector"
    assert ew.total_flops == 2_000_000
    assert ew.total_bytes == (1 + 1) * 1_000_000 * 2      # 1 in + 1 out, bf16
    rd = vector_cost(1_000_000, 1024)
    assert rd.unit == "vector"
    assert rd.total_bytes == (1_000_000 + 1024) * 2


def test_vector_op_mfu_uses_vector_peak_and_is_memory_bound():
    # elementwise add on 8M bf16 elems: tiny compute, dominated by IO -> memory-bound.
    ew = vector_cost(8_000_000, 8_000_000, inputs=2)
    peak_vec = 40e12   # example vector-unit peak (TFLOPS-scale), NOT the cube peak
    r = ce.analyze(30.0, ew.total_flops, ew.total_bytes, PEAK_BW, peak_vec,
                   compute_unit="vector")
    assert r["compute_unit"] == "vector"
    assert r["cost_bound"] == "Memory-Bound"        # vector op: bandwidth binds, not compute
    assert r["MFU"] < 0.05                            # vector-FLOP utilization is tiny
    assert r["MBU"] > 0                               # MBU is the meaningful metric


def test_cube_op_default_unit_unchanged():
    c = mla_cost()
    assert c.unit == "cube"
    r = ce.analyze(57.5, c.total_flops, c.total_bytes, PEAK_BW, PEAK_FLOPS)  # default cube
    assert r["compute_unit"] == "cube"
    assert abs(r["MFU"] - 0.138) < 0.005


def test_low_mbu_but_mte2_at_floor_is_pipeline_trap():
    # self-dev norope C1: 85.72us, MTE2=51.5us at HBM floor, wall-MBU ~64%.
    # Memory-bound with LOW wall-MBU -> would slip the >=90% MBU gate, but the
    # load pipe is already at the bandwidth floor -> PIPELINE-TRAP catches it.
    c = mla_cost()
    r = ce.analyze(85.72, c.total_flops, c.total_bytes, PEAK_BW, PEAK_FLOPS, mte2_us=51.5)
    assert r["cost_bound"] == "Memory-Bound"
    assert r["MBU"] < 0.90                 # slips the plain bandwidth-ceiling gate
    assert r["mte2_util"] >= 0.90          # but load pipe is at the floor
    assert r["mte2_util"] >= 0.90 and "non-overlap" in r["verdict"]


def test_mte2_util_derived_from_pipe_ratio(tmp_path):
    p = tmp_path / "PipeUtilization.csv"
    p.write_text("aic_cube_ratio,aic_mte2_ratio,aiv_vec_ratio\n0.36,0.60,0.01\n")
    c = mla_cost()
    r = ce.analyze(85.72, c.total_flops, c.total_bytes, PEAK_BW, PEAK_FLOPS, pipe_util=str(p))
    assert r["mte2_util"] is not None       # derived from 0.60 * 85.72us
    assert r["mte2_util"] >= 0.90 and "non-overlap" in r["verdict"]


def test_high_mbu_still_bandwidth_ceiling_not_trap():
    c = mla_cost()
    r = ce.analyze(57.5, c.total_flops, c.total_bytes, PEAK_BW, PEAK_FLOPS, mte2_us=34.0)
    assert r["MBU"] >= 0.90 and "fewer bytes" in r["verdict"]   # >=90% MBU branch wins
    assert "non-overlap" not in r["verdict"]


def test_pipe_util_reads_opmode_cube_column(tmp_path):
    # op-mode PipeUtilization.csv uses aic_cube_ratio (not aic_mac_ratio).
    p = tmp_path / "PipeUtilization.csv"
    p.write_text("aic_cube_ratio,aic_mte2_ratio,aiv_vec_ratio\n0.95,0.60,0.01\n0.94,0.61,0.00\n")
    bound, pipes = ce._pipe_bound(str(p))
    assert bound == "Compute Bound (Cube)"
    assert pipes["Cube"] > 0.9        # would be 0 if the column name were wrong


def test_zero_or_negative_inputs_raise_not_crash():
    import pytest
    c = mla_cost()
    for bad in [dict(duration_us=0), dict(peak_bw=0), dict(peak_flops=0), dict(duration_us=-1)]:
        kw = dict(duration_us=57.5, flops=c.total_flops, hbm_bytes=c.total_bytes,
                  peak_bw=PEAK_BW, peak_flops=PEAK_FLOPS)
        kw.update(bad)
        with pytest.raises(ValueError):
            ce.analyze(**kw)


def test_perf_csv_reader(tmp_path):
    p = tmp_path / "performance.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "total_us"])
        w.writeheader()
        w.writerow({"case_id": "caseA", "total_us": "57.5"})
        w.writerow({"case_id": "caseB", "total_us": "63.2"})
    assert ce._read_perf_csv(str(p), "caseA") == 57.5
    assert ce._read_perf_csv(str(p), "caseB") == 63.2


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
