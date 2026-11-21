# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Verifier publication binds actual bytes and preserves negative/unknown results."""

import json

import pytest

from _checks import experiment, feedback, observations
from .test_experiment import write, timing


def setup(trial):
    oracle = trial["tmp"] / "oracle.py"
    oracle.write_text("def reference(value): return value + 1\n")
    trial["plan"]["scope"]["numerical_contract"]["oracle_sha256"] = (
        feedback.file_identity(oracle)["sha256"]
    )
    trial["path"].unlink()
    trial["result"] = experiment.run(
        trial["path"], init=write(trial["tmp"] / "plan.json", trial["plan"])
    )
    return feedback.file_identity(oracle)


def capture(trial, oracle, verify, **kwargs):
    return observations.capture(
        trial["path"],
        trial["tmp"] / "observation.json",
        baseline=trial["variant"],
        verify=verify,
        oracle=oracle,
        evidence_kind="synthetic_fixture",
        **kwargs,
    )


def test_baseline_capture_is_read_only_until_explicitly_observed(trial):
    oracle = setup(trial)
    before = trial["path"].read_bytes()
    result = capture(trial, oracle, lambda *args: "passed")
    assert (
        result["phase_if_observed"] == "ready" and trial["path"].read_bytes() == before
    )
    data = json.loads((trial["tmp"] / "observation.json").read_text())
    assert data["baseline"]["correctness"]["cases"] == {
        "small": "passed",
        "tail": "passed",
    }
    assert experiment.run(trial["path"], observation=trial["tmp"] / "observation.json")[
        "best"
    ]["verified"]


@pytest.mark.parametrize(
    "outcome,phase",
    [("failed", "baseline_unverified"), ("not_run", "baseline_unverified")],
)
def test_failed_or_unknown_callbacks_are_not_promoted(trial, outcome, phase):
    oracle = setup(trial)
    assert capture(trial, oracle, lambda *args: outcome)["phase_if_observed"] == phase


def test_source_mutation_prevents_publication(trial):
    oracle = setup(trial)

    def changed(*args):
        (trial["root"] / "helper.py").write_text("OFFSET = 4\n")
        return "passed"

    with pytest.raises(ValueError, match="source changed"):
        capture(trial, oracle, changed)
    assert not (trial["tmp"] / "observation.json").exists()


def test_callback_failure_keeps_original_exception_and_no_success_record(trial):
    oracle = setup(trial)

    def failed(*args):
        raise RuntimeError("native verifier failed")

    with pytest.raises(RuntimeError, match="native verifier failed"):
        capture(trial, oracle, failed)
    assert not (trial["tmp"] / "observation.json").exists()


def test_changed_oracle_or_existing_output_is_rejected(trial):
    oracle = setup(trial)

    def changed(*args):
        (trial["tmp"] / "oracle.py").write_text("different oracle\n")
        return "passed"

    with pytest.raises(ValueError, match="Evidence changed"):
        capture(trial, oracle, changed)
    (trial["tmp"] / "observation.json").write_text("preserve existing evidence\n")
    with pytest.raises(ValueError, match="new file"):
        capture(trial, oracle, lambda *args: "passed")


def test_source_hash_cannot_replace_the_variant_digest(trial):
    oracle = setup(trial)
    trial["variant"]["configuration"]["workers"] = 2
    with pytest.raises(ValueError, match="differs from its selected snapshot"):
        capture(trial, oracle, lambda *args: "passed")


@pytest.mark.parametrize("failed", [False, True])
def test_candidate_observation_reuses_retention_and_suppresses_failed_timing(
    trial, failed
):
    import shutil

    oracle = setup(trial)
    capture(trial, oracle, lambda *args: "passed")
    experiment.run(trial["path"], observation=trial["tmp"] / "observation.json")
    candidate_root = trial["tmp"] / "candidate"
    shutil.copytree(trial["root"], candidate_root)
    (candidate_root / "kernel.py").write_text("def run(): return 2\n")
    candidate = {**trial["variant"], "root": str(candidate_root)}
    selection = {
        "schema": "pypto-experiment-selection/1",
        "variant": candidate,
        "changes_wrapper": False,
        "hypothesis": {
            "observation": "One intermediate",
            "change": "Direct output",
            "prediction": "Less work",
            "reject_if": "Numerical or timing failure",
        },
    }
    experiment.run(
        trial["path"], selection=write(trial["tmp"] / "selection.json", selection)
    )
    measured = []

    def measure(*args):
        measured.append(args)
        return {"timing": timing()}

    result = observations.capture(
        trial["path"],
        trial["tmp"] / "candidate-observation.json",
        baseline=trial["variant"],
        candidate=candidate,
        oracle=oracle,
        evidence_kind="synthetic_fixture",
        verify=lambda side, *args: "failed"
        if failed and side == "candidate"
        else "passed",
        measure=measure,
    )
    assert result["phase_if_observed"] == ("rejected" if failed else "retained")
    assert bool(measured) is not failed
    observed = experiment.run(
        trial["path"], observation=trial["tmp"] / "candidate-observation.json"
    )
    assert observed["phase"] == result["phase_if_observed"]
    if failed:
        assert observed["action_guidance"]["goal"] == "validate_numerics"
        assert (
            observed["action_guidance"]["next_action"]
            == "restore_verified_best_and_resolve_numerical_failure"
        )
