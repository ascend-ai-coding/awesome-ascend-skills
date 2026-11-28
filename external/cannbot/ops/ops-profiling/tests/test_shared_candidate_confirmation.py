# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from copy import deepcopy
from itertools import permutations

import pytest

from test_collect_perf_application import MODULE


def shared_plan():
    runs = []
    for i, order in enumerate(permutations(("original", "current", "candidate"))):
        for j, variant in enumerate(order):
            runs.append({"case_id": f"r{i}-{variant}", "confirmation": {
                "round_id": i, "position": j, "variant": variant,
                "order": list(order), "input_sha256": "a" * 64}})
    return {"paired_confirmation": {"kind": "shared_candidate_paired_v1", "round_count": 6,
            "baseline_variants": ["original", "current"], "candidate_variant": "candidate"},
            "case_runs": runs}


def test_three_variant_complete_balanced_plan_and_legacy_pair_are_accepted():
    plan = shared_plan()
    MODULE.validate_confirmation_labels(plan, [r["case_id"] for r in plan["case_runs"]])
    legacy = {"paired_confirmation": {"round_count": 2}, "case_runs": []}
    for i, order in enumerate(("AB", "BA")):
        variants = ("baseline", "candidate") if order == "AB" else ("candidate", "baseline")
        for j, variant in enumerate(variants):
            legacy["case_runs"].append({"case_id": f"l{i}-{variant}", "confirmation": {
                "round_id": i, "position": j, "variant": variant,
                "order": order, "input_sha256": "b" * 64}})
    MODULE.validate_confirmation_labels(legacy, [r["case_id"] for r in legacy["case_runs"]])


@pytest.mark.parametrize("mutation", ["missing", "other-input", "wrong-position", "same-order",
                                      "duplicate-round", "wrong-case-order", "wrong-role"])
def test_three_variant_bad_topology_cannot_pass(mutation):
    plan = shared_plan()
    case_ids = [r["case_id"] for r in plan["case_runs"]]
    if mutation == "missing":
        plan["case_runs"].pop()
    elif mutation == "other-input":
        plan["case_runs"][0]["confirmation"]["input_sha256"] = "b" * 64
    elif mutation == "wrong-position":
        plan["case_runs"][0]["confirmation"]["position"] = False
    elif mutation == "same-order":
        for run, old in zip(plan["case_runs"][3:6], plan["case_runs"][:3]):
            label = deepcopy(old["confirmation"])
            label["round_id"] = 1
            run["confirmation"] = label
    elif mutation == "duplicate-round":
        for run in plan["case_runs"][3:6]:
            run["confirmation"]["round_id"] = 0
    elif mutation == "wrong-case-order":
        case_ids[0], case_ids[1] = case_ids[1], case_ids[0]
    else:
        plan["paired_confirmation"]["baseline_variants"] = ["candidate", "current"]
    with pytest.raises(ValueError):
        MODULE.validate_confirmation_labels(plan, case_ids)
