# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Preparation launches are mapped, never mistaken for steady-state samples."""
import pytest

from test_collect_perf_application import MODULE, execution_manifest, write_op_summary


def test_prime_with_same_kernel_name_is_excluded_from_samples(tmp_path):
    manifest = execution_manifest(["network"])
    run = manifest["case_runs"][0]
    run["preparation_kernel_sequence"] = ["target_a", "target_b"]
    launches = [("target_a", 900), ("target_b", 100)]
    for _ in range(10):
        launches.extend([("ArgMax", 300), ("target_a", 3), ("target_b", 2)])
    path = tmp_path / "summary.csv"
    write_op_summary(path, launches)
    rows, validation = MODULE.parse_profile(path, manifest)
    assert rows[0]["total_us"] == 5
    assert validation["expected_relevant_launches"] == 32
    assert [item["role"] for item in validation["mapped_launches"][:2]] == ["preparation", "preparation"]
    assert MODULE.actual_max_launches_per_case(manifest) == 32


def test_undeclared_or_missing_preparation_is_rejected(tmp_path):
    manifest = execution_manifest(["network"])
    path = tmp_path / "summary.csv"
    launches = [("target_a", 900)] + [(name, 1) for _ in range(10) for name in ["ArgMax", "target_a", "target_b"]]
    write_op_summary(path, launches)
    with pytest.raises(ValueError, match="topology"):
        MODULE.parse_profile(path, manifest)
    manifest["case_runs"][0]["preparation_kernel_sequence"] = ["target_a", "target_b"]
    with pytest.raises(ValueError, match="topology"):
        MODULE.parse_profile(path, manifest)
