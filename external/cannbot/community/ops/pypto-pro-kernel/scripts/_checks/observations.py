# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build existing experiment observations inside an explicitly controlled verifier.

Callbacks own actual execution and the oracle. This binds their attestations to
declared bytes and scope; it does not authenticate a callback or replay a proof.
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

from . import experiment, feedback


def capture(
    state_path,
    output,
    *,
    baseline,
    verify,
    oracle,
    evidence_kind,
    candidate=None,
    measure=None,
    pypto_root=experiment.PYPTO_ROOT,
):
    """Verify each case, optionally measure, then publish one source-bound record.

    verify(side, case, workload) returns passed/failed/not_run. measure(case,
    workload) returns the unchanged comparison object with timing and optional
    wall_timing; case/scope/matched_inputs are supplied by this controller.
    """
    state_path, output = Path(state_path).resolve(), Path(output).absolute()
    experiment.require(
        not output.exists() and output.parent.is_dir(),
        "Observation output must be a new file in an existing directory",
    )
    experiment.require(
        evidence_kind in {"native_measurement", "synthetic_fixture"},
        "Declare the verifier evidence kind",
    )
    before = feedback.file_identity(state_path)
    state = experiment.run(state_path, pypto_root=pypto_root)
    original_oracle = experiment.artifact(oracle, state_path.parent)
    experiment.require(
        original_oracle["sha256"]
        == state["scope"]["numerical_contract"]["oracle_sha256"],
        "Verifier oracle differs from the frozen numerical contract",
    )
    experiment.require(
        (candidate is not None) == (state["current"] is not None),
        "Declare the current candidate exactly when one is selected",
    )
    specs = {
        "baseline": baseline,
        **({"candidate": candidate} if candidate is not None else {}),
    }
    identities = {
        side: experiment.variant(spec, state_path.parent)["sha256"]
        for side, spec in specs.items()
    }
    for side, identity in identities.items():
        expected = state["best"] if side == "baseline" else state["current"]
        experiment.require(
            identity == expected["variant"]["sha256"],
            f"Executed {side} differs from its selected snapshot",
        )
    data = {
        "schema": "pypto-experiment-observation/1",
        "evidence_kind": evidence_kind,
        "scope": copy.deepcopy(state["scope"]),
        "toolchain_sha256": state["toolchain"]["sha256"],
    }
    complete = True
    for side, identity in identities.items():
        cases = {}
        for case, workload in state["scope"]["cases"].items():
            outcome = verify(side, case, copy.deepcopy(workload))
            experiment.require(
                outcome in {"passed", "failed", "not_run"},
                "Verifier must return an explicit case outcome",
            )
            cases[case] = outcome
            complete = complete and outcome == "passed"
        data[side] = {
            "variant_sha256": identity,
            "correctness": {
                "scope": "complete_call",
                "oracle_sha256": original_oracle["sha256"],
                "cases": cases,
            },
        }
    if candidate is not None:
        data["comparisons"] = []
        if complete and measure is not None:
            for case in state["scope"]["timing_cases"]:
                measured = measure(case, copy.deepcopy(state["scope"]["cases"][case]))
                experiment.require(
                    isinstance(measured, dict)
                    and set(measured) <= {"timing", "wall_timing"}
                    and "timing" in measured,
                    "Return the original timing result; comparison identity is controller-owned",
                )
                data["comparisons"].append(
                    {
                        "case": case,
                        "scope": "complete_call",
                        "matched_inputs": True,
                        **measured,
                    }
                )
    experiment.require(
        feedback.file_identity(state_path) == before,
        "Experiment state changed during verification",
    )
    experiment.artifact(state["toolchain"], state_path.parent)
    experiment.artifact(oracle, state_path.parent)
    for side, spec in specs.items():
        experiment.require(
            experiment.variant(spec, state_path.parent)["sha256"] == identities[side],
            "Execution source changed during verification",
        )
    payload = experiment.canonical(data) + b"\n"
    experiment.require(
        len(payload) <= 512 * 1024, "Observation exceeds the existing input limit"
    )
    evidence = {"path": str(output), "sha256": feedback.digest(payload)}
    projected = copy.deepcopy(state)
    experiment.observe(projected, data, evidence)
    descriptor, temporary = tempfile.mkstemp(prefix=".observation-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return {
        "observation": evidence,
        "phase_if_observed": projected["phase"],
        "executed_variants": identities,
        "evidence_limit": (
            "Controlled-verifier attestation over declared inputs, not authentication or independent replay."
        ),
    }
