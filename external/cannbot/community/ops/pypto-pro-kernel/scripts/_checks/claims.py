# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check a scoped performance assertion against explicit verifier observations.

This checks consistency of supplied attestations, not their authenticity, and
reuses experiment retention gates without writing state or executing a kernel.
"""

from __future__ import annotations

import copy
from pathlib import Path

from . import experiment, feedback


def report(path):
    path = Path(path).resolve()
    spec = experiment.read_json(path)
    experiment.require(
        spec.get("schema") == "pypto-performance-claim/1",
        "Unsupported performance claim schema",
    )
    claim = spec.get("claim")
    experiment.require(
        isinstance(claim, dict) and isinstance(claim.get("speedup_established"), bool),
        "claim.speedup_established must be boolean",
    )
    workload, metric = claim.get("workload"), claim.get("metric")
    experiment.require(
        isinstance(workload, dict) and workload, "Declare the exact claimed workload"
    )
    experiment.require(
        len(experiment.canonical(workload)) <= 4096
        and not set(claim)
        - {"speedup_established", "workload", "metric", "changes_wrapper"},
        "Claim fields or workload exceed the bounded contract",
    )
    experiment.require(
        metric in {"native_task_sum_us", "synchronized_wall_us"},
        "Declare the claimed metric",
    )
    experiment.require(
        isinstance(claim.get("changes_wrapper"), bool), "Declare changes_wrapper"
    )
    result = {
        "schema": "pypto-performance-claim-check/1",
        "status": "unverified",
        "input": feedback.file_identity(path),
        "claim": claim,
        "claim_check": "unsupported",
        "supported_speedup": None,
        "evidence_outcome": "unverified",
        "reason": "missing_observation",
        "validation": {
            "native": "not_independently_established",
            "numerical": "not_independently_established",
            "performance": "not_independently_established",
        },
        "evidence_limit": (
            "Consistency over declared source snapshots and verifier attestations, not"
            " authentication, independent replay or generalized workload support."
        ),
    }
    try:
        if spec.get("observation") is not None:
            evidence = experiment.artifact(spec["observation"], path.parent)
            result["observation"] = evidence
            observed = experiment.read_json(evidence["path"])
            if feedback.file_identity(evidence["path"]) != evidence:
                raise ValueError("Observation changed during inspection")
            if observed.get("evidence_kind") != "native_measurement":
                result["reason"] = "native_measurement_not_attested"
            else:
                scope = experiment._scope(observed.get("scope"))
                matches = [
                    name
                    for name, value in scope["cases"].items()
                    if experiment.canonical(value) == experiment.canonical(workload)
                ]
                if not matches:
                    result["reason"] = "workload_not_observed"
                elif metric != scope["metric"]:
                    result["reason"] = "metric_mismatch"
                elif not set(matches).intersection(scope["timing_cases"]):
                    result["reason"] = "workload_not_timed"
                elif not all(
                    key in spec for key in ("baseline", "candidate", "toolchain")
                ):
                    result["reason"] = "missing_execution_identity"
                else:
                    toolchain = experiment.artifact(spec["toolchain"], path.parent)
                    baseline = experiment.variant(spec["baseline"], path.parent)
                    candidate = experiment.variant(spec["candidate"], path.parent)
                    experiment.require(
                        baseline["sha256"] != candidate["sha256"],
                        "Candidate and baseline are identical",
                    )
                    result["execution_identity"] = {
                        "baseline": baseline["sha256"],
                        "candidate": candidate["sha256"],
                        "toolchain": toolchain["sha256"],
                    }
                    state = {
                        "scope": copy.deepcopy(scope),
                        "policy": experiment._policy(spec.get("policy", {})),
                        "toolchain": toolchain,
                        "best": {
                            "variant": baseline,
                            "verified": False,
                            "evidence": None,
                            "correctness": None,
                            "performance": None,
                        },
                        "current": {
                            "variant": candidate,
                            "changes_wrapper": claim["changes_wrapper"],
                            "decision": "awaiting_evidence",
                            "evidence": None,
                            "correctness": None,
                            "comparisons": [],
                        },
                    }
                    experiment.observe(state, observed, evidence)
                    result["evidence_outcome"] = state["phase"]
                    result["reason"] = "experiment_" + state["phase"]
                    result["matched_cases"] = sorted(
                        set(matches).intersection(scope["timing_cases"])
                    )
                    if state["phase"] == "retained":
                        result["supported_speedup"] = True
                    elif state["phase"] == "rejected":
                        result["supported_speedup"] = False
            if feedback.file_identity(evidence["path"]) != evidence:
                raise ValueError("Observation changed during claim validation")
            if "execution_identity" in result:
                for side in ("baseline", "candidate"):
                    if (
                        experiment.variant(spec[side], path.parent)["sha256"]
                        != result["execution_identity"][side]
                    ):
                        raise ValueError(
                            "Execution source changed during claim validation"
                        )
                experiment.artifact(spec["toolchain"], path.parent)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(supported_speedup=None, evidence_outcome="unverified")
        result["reason"] = "invalid_or_mismatched_evidence"
        result["detail"] = str(exc)[:400]
    if feedback.file_identity(path) != result["input"]:
        raise ValueError("Claim input changed during inspection")
    if not claim["speedup_established"]:
        result.update(status="passed", claim_check="not_claimed")
    elif result["supported_speedup"] is True:
        result.update(status="passed", claim_check="supported")
    elif result["supported_speedup"] is False:
        result.update(status="failed", claim_check="contradicted")
    result["next_action"] = (
        "Keep the assertion within these exact snapshots, workload, metric and supplied evidence."
        if result["claim_check"] == "supported"
        else (
            "Do not promote an unsupported assertion. Obtain matching original"
            " numerical and complete-call timing evidence; unknown evidence does not"
            " end optimization or authorize replay."
        )
    )
    return result
