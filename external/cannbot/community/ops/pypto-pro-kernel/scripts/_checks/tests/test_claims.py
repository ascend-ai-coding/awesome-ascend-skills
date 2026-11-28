# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fictional verifier attestations exercise scope gates, not hardware performance."""

import json
import subprocess
import sys

import pytest

from _checks import claims, experiment, feedback
from _checks.paths import SKILL_ROOT
from _checks.tests.test_experiment import timing


@pytest.fixture
def claim(tmp_path):
    for name, text in (
        ("baseline.py", "value=1\n"),
        ("candidate.py", "value=2\n"),
        ("toolchain.json", "{}"),
    ):
        (tmp_path / name).write_text(text)
    base = {
        "root": ".",
        "files": ["baseline.py"],
        "entry": "baseline.py:kernel",
        "configuration": {},
    }
    candidate = {
        "root": ".",
        "files": ["candidate.py"],
        "entry": "candidate.py:kernel",
        "configuration": {},
    }
    toolchain = feedback.file_identity(tmp_path / "toolchain.json")
    spec = {
        "schema": "pypto-performance-claim/1",
        "baseline": base,
        "candidate": candidate,
        "toolchain": toolchain,
        "claim": {
            "speedup_established": True,
            "workload": {"rows": 8, "cols": 256},
            "metric": "synchronized_wall_us",
            "changes_wrapper": False,
        },
    }
    scope = {
        "cases": {"measured": {"rows": 8, "cols": 256}},
        "timing_cases": ["measured"],
        "metric": "synchronized_wall_us",
        "numerical_contract": {"oracle_sha256": "f" * 64},
    }

    def numerical(variant):
        return {
            "variant_sha256": experiment.variant(variant, tmp_path)["sha256"],
            "correctness": {
                "scope": "complete_call",
                "oracle_sha256": "f" * 64,
                "cases": {"measured": "passed"},
            },
        }

    observed = {
        "schema": "pypto-experiment-observation/1",
        "evidence_kind": "native_measurement",
        "scope": scope,
        "toolchain_sha256": toolchain["sha256"],
        "baseline": numerical(base),
        "candidate": numerical(candidate),
        "comparisons": [
            {
                "case": "measured",
                "scope": "complete_call",
                "matched_inputs": True,
                "timing": timing(metric="synchronized_wall_us"),
            }
        ],
    }
    return tmp_path, spec, observed


def check(trial):
    root, spec, observed = trial
    path = root / "claim.json"
    evidence = root / "observation.json"
    if observed is not None:
        evidence.write_text(json.dumps(observed))
        spec["observation"] = feedback.file_identity(evidence)
    path.write_text(json.dumps(spec))
    return claims.report(path)


def test_matching_attestation_supports_only_the_exact_claim_without_execution(
    claim, monkeypatch
):
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: pytest.fail("must not execute")
    )
    result = check(claim)
    assert result["status"] == "passed" and result["claim_check"] == "supported"
    assert result["supported_speedup"] is True and result["matched_cases"] == [
        "measured"
    ]
    assert set(result["validation"].values()) == {"not_independently_established"}


@pytest.mark.parametrize(
    "mutation, reason",
    [
        ("shape", "workload_not_observed"),
        ("metric", "metric_mismatch"),
        ("numeric_shape_type", "workload_not_observed"),
        ("not_timed", "workload_not_timed"),
        ("synthetic", "native_measurement_not_attested"),
        ("statement", "native_measurement_not_attested"),
        ("kind_missing", "native_measurement_not_attested"),
    ],
)
def test_scope_or_provenance_does_not_become_a_native_speedup(claim, mutation, reason):
    _, spec, observed = claim
    if mutation == "shape":
        spec["claim"]["workload"]["rows"] = 1
    if mutation == "numeric_shape_type":
        spec["claim"]["workload"]["rows"] = 8.0
    if mutation == "metric":
        spec["claim"]["metric"] = "native_task_sum_us"
    if mutation == "not_timed":
        observed["scope"]["cases"]["other"] = {"rows": 17}
        observed["scope"]["timing_cases"] = ["other"]
    if mutation == "synthetic":
        observed["evidence_kind"] = "synthetic_fixture"
    if mutation == "statement":
        observed["evidence_kind"] = "user_statement"
    if mutation == "kind_missing":
        observed.pop("evidence_kind")
    result = check(claim)
    assert result["status"] == "unverified" and result["supported_speedup"] is None
    assert result["reason"] == reason


@pytest.mark.parametrize(
    "mutation",
    [
        "candidate",
        "baseline",
        "toolchain",
        "missing_ba",
        "oracle",
        "partial_correctness",
    ],
)
def test_existing_experiment_identity_and_acceptance_gates_are_reused(claim, mutation):
    _, _, observed = claim
    if mutation in {"candidate", "baseline"}:
        observed[mutation]["variant_sha256"] = "a" * 64
    if mutation == "toolchain":
        observed["toolchain_sha256"] = "a" * 64
    if mutation == "missing_ba":
        observed["comparisons"][0]["timing"]["orders"].pop()
    if mutation == "oracle":
        observed["candidate"]["correctness"]["oracle_sha256"] = "a" * 64
    if mutation == "partial_correctness":
        observed["candidate"]["correctness"]["cases"] = {}
    result = check(claim)
    assert result["status"] == "unverified" and result["supported_speedup"] is None


@pytest.mark.parametrize(
    "mutation", ["numerical_failure", "regression", "order_conflict"]
)
def test_negative_or_inconclusive_evidence_is_not_promoted(claim, mutation):
    _, _, observed = claim
    if mutation == "numerical_failure":
        observed["candidate"]["correctness"]["cases"]["measured"] = "failed"
    if mutation == "regression":
        observed["comparisons"][0]["timing"] = timing(
            metric="synchronized_wall_us", ab=1.2, ba=1.2
        )
    if mutation == "order_conflict":
        observed["comparisons"][0]["timing"] = timing(
            metric="synchronized_wall_us", ab=0.8, ba=1.2
        )
    result = check(claim)
    assert result["supported_speedup"] is (
        None if mutation == "order_conflict" else False
    )
    assert result["status"] == (
        "unverified" if mutation == "order_conflict" else "failed"
    )


def test_no_claim_does_not_mean_no_possible_improvement(claim):
    root, spec, _ = claim
    spec["claim"]["speedup_established"] = False
    result = check((root, spec, None))
    assert result["status"] == "passed" and result["claim_check"] == "not_claimed"
    assert (
        result["supported_speedup"] is None
        and result["evidence_outcome"] == "unverified"
    )


def test_changed_observation_or_source_cannot_reuse_old_receipt(claim):
    root, _, _ = claim
    check(claim)
    (root / "candidate.py").write_text("value=3\n")
    result = claims.report(root / "claim.json")
    assert result["status"] == "unverified" and result["supported_speedup"] is None
    (root / "observation.json").write_text("{}")
    assert (
        claims.report(root / "claim.json")["reason"] == "invalid_or_mismatched_evidence"
    )


def test_source_mutation_during_shared_gate_invalidates_result(claim, monkeypatch):
    original = experiment.observe

    def mutate(state, observed, evidence):
        original(state, observed, evidence)
        (claim[0] / "candidate.py").write_text("value=3\n")

    monkeypatch.setattr(experiment, "observe", mutate)
    result = check(claim)
    assert result["status"] == "unverified" and result["supported_speedup"] is None


def test_cli_preserves_unsupported_and_contradicted_status(claim):
    root, spec, observed = claim
    check(claim)
    command = [
        sys.executable,
        str(SKILL_ROOT / "scripts/check.py"),
        "claim",
        "--spec",
        str(root / "claim.json"),
    ]
    assert subprocess.run(command, capture_output=True).returncode == 0
    spec["claim"]["workload"]["rows"] = 1
    check(claim)
    assert subprocess.run(command, capture_output=True).returncode == 2
    spec["claim"]["workload"]["rows"] = 8
    observed["candidate"]["correctness"]["cases"]["measured"] = "failed"
    check(claim)
    assert subprocess.run(command, capture_output=True).returncode == 1
