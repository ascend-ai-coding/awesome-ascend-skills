# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Retention requires complete numerical evidence and matched timing identities."""

import copy
import json
import subprocess
import sys

import pytest

from _checks import experiment, feedback
from _checks.paths import PYPTO_ROOT, SKILL_ROOT


def write(path, data):
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def trial(tmp_path):
    root = tmp_path / "code"
    root.mkdir()
    (root / "kernel.py").write_text("def run(): return 1\n")
    (root / "helper.py").write_text("OFFSET = 3\n")
    toolchain = write(
        tmp_path / "toolchain.json", {"compiler": "test fixture", "device": "no device"}
    )
    spec = {
        "root": str(root),
        "files": ["kernel.py", "helper.py"],
        "entry": "kernel.py:run",
        "configuration": {"workers": 1},
    }
    plan = {
        "schema": "pypto-experiment-plan/1",
        "toolchain": feedback.file_identity(toolchain),
        "baseline": spec,
        "max_trials": 3,
        "scope": {
            "cases": {"small": {"rows": 8}, "tail": {"rows": 17}},
            "timing_cases": ["small"],
            "metric": "native_task_sum_us",
            "numerical_contract": {"atol": 0, "rtol": 0, "oracle_sha256": "f" * 64},
        },
    }
    state = tmp_path / "current.json"
    result = experiment.run(state, init=write(tmp_path / "plan.json", plan))
    return {
        "path": state,
        "plan": plan,
        "variant": spec,
        "result": result,
        "root": root,
        "tmp": tmp_path,
    }


def numerical(digest, cases=None):
    return {
        "variant_sha256": digest,
        "correctness": {
            "scope": "complete_call",
            "oracle_sha256": "f" * 64,
            "cases": {"small": "passed", "tail": "passed"} if cases is None else cases,
        },
    }


def observation(trial):
    state = trial["result"]
    data = {
        "schema": "pypto-experiment-observation/1",
        "scope": state["scope"],
        "toolchain_sha256": state["toolchain"]["sha256"],
        "baseline": numerical(state["best"]["variant"]["sha256"]),
    }
    if state["current"]:
        data["candidate"] = numerical(state["current"]["variant"]["sha256"])
    return data


def observe(trial, data):
    trial["result"] = experiment.run(
        trial["path"], observation=write(trial["tmp"] / "observation.json", data)
    )
    return trial["result"]


def selected(trial, *, wrapper=False):
    observe(trial, observation(trial))
    (trial["root"] / "kernel.py").write_text("def run(): return 2\n")
    select = {
        "schema": "pypto-experiment-selection/1",
        "variant": trial["variant"],
        "changes_wrapper": wrapper,
        "hypothesis": {
            "observation": "One extra transfer",
            "change": "Write the required format directly",
            "prediction": "Reduce conversion work",
            "reject_if": "Numerical failure or full-call regression",
        },
    }
    trial["result"] = experiment.run(
        trial["path"], selection=write(trial["tmp"] / "select.json", select)
    )
    return trial


def timing(ab=0.8, ba=0.8, *, pairs=8, counts=(1, 1), metric="native_task_sum_us"):
    orders = []
    for order, ratio in (("AB", ab), ("BA", ba)):

        def duration(v):
            return {"median_us": v, "min_us": v, "max_us": v}

        wins = pairs if ratio < 1 else 0
        orders.append(
            {
                "order": order,
                "pairs": pairs,
                "baseline": duration(10),
                "candidate": duration(10 * ratio),
                "median_ratio": ratio,
                "candidate_wins": wins,
                "win_fraction": wins / pairs,
            }
        )
    return {
        "metric": metric,
        "orders": orders,
        "warmup": 3,
        "pairs_per_order": pairs,
        "tasks_per_call": {"baseline": counts[0], "candidate": counts[1]},
    }


def measured(trial, **kwargs):
    data = observation(trial)
    data["comparisons"] = [
        {
            "case": "small",
            "scope": "complete_call",
            "matched_inputs": True,
            "timing": timing(**kwargs),
        }
    ]
    return data


def test_baseline_must_be_verified_before_selection(trial):
    spec = {"schema": "pypto-experiment-selection/1"}
    with pytest.raises(ValueError, match="Verify the baseline"):
        experiment.run(
            trial["path"], selection=write(trial["tmp"] / "select.json", spec)
        )
    assert not trial["result"]["best"]["verified"]


def test_failed_or_partial_baseline_cannot_start_search(trial):
    data = observation(trial)
    data["baseline"]["correctness"]["cases"] = {"small": "passed"}
    assert observe(trial, data)["phase"] == "baseline_unverified"
    data["baseline"]["correctness"]["cases"]["tail"] = "failed"
    assert not observe(trial, data)["best"]["verified"]


def test_missing_correctness_is_not_an_exit_zero_pass(trial):
    data = observation(trial)
    del data["baseline"]["correctness"]
    data["exit_code"] = 0
    assert observe(trial, data)["phase"] == "baseline_unverified"


@pytest.mark.parametrize("kind", ["bytes", "toolchain", "workload", "oracle"])
def test_mismatched_observation_never_mutates_best(trial, kind):
    selected(trial)
    data = measured(trial)
    if kind == "bytes":
        data["candidate"]["variant_sha256"] = "e" * 64
    elif kind == "toolchain":
        data["toolchain_sha256"] = "e" * 64
    elif kind == "workload":
        data["scope"] = copy.deepcopy(data["scope"])
        data["scope"]["cases"]["small"]["rows"] = 16
    else:
        data["candidate"]["correctness"]["oracle_sha256"] = "e" * 64
    before = trial["path"].read_bytes()
    with pytest.raises(ValueError):
        observe(trial, data)
    assert trial["path"].read_bytes() == before


def test_rejects_failed_candidate_even_when_reported_fast(trial):
    selected(trial)
    before = trial["result"]["best"]
    data = measured(trial)
    data["candidate"]["correctness"]["cases"]["tail"] = "failed"
    result = observe(trial, data)
    assert result["phase"] == "rejected" and result["best"] == before


def test_partial_correctness_does_not_replace_best(trial):
    selected(trial)
    data = measured(trial)
    del data["candidate"]["correctness"]["cases"]["tail"]
    assert observe(trial, data)["phase"] == "awaiting_correctness"


@pytest.mark.parametrize(
    "ab,ba,pairs,expected",
    [
        (0.8, 1.2, 8, "inconclusive"),
        (0.8, 0.8, 2, "inconclusive"),
        (1, 1, 8, "inconclusive"),
        (1.2, 1.2, 8, "rejected"),
    ],
)
def test_noise_regression_and_order_dependence_preserve_best(
    trial, ab, ba, pairs, expected
):
    selected(trial)
    before = trial["result"]["best"]
    result = observe(trial, measured(trial, ab=ab, ba=ba, pairs=pairs))
    assert result["phase"] == expected and result["best"] == before


def test_every_frozen_timing_case_is_required(trial):
    trial["plan"]["scope"]["timing_cases"].append("tail")
    trial["path"].unlink()
    trial["result"] = experiment.run(
        trial["path"], init=write(trial["tmp"] / "plan.json", trial["plan"])
    )
    selected(trial)
    assert observe(trial, measured(trial))["phase"] == "awaiting_timing"


@pytest.mark.parametrize("wrapper,counts", [(True, (1, 1)), (False, (1, 2))])
def test_added_wrapper_or_tasks_require_paired_wall_evidence(trial, wrapper, counts):
    selected(trial, wrapper=wrapper)
    data = measured(trial, counts=counts)
    assert observe(trial, data)["phase"] == "awaiting_timing"
    data["comparisons"][0]["wall_timing"] = timing(
        metric="synchronized_wall_us", ab=1.1, ba=1.1
    )
    assert observe(trial, data)["phase"] == "inconclusive"
    data["comparisons"][0]["wall_timing"] = timing(metric="synchronized_wall_us")
    assert observe(trial, data)["phase"] == "retained"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_ba",
        "nan",
        "zero",
        "metric",
        "wins",
        "task_count",
        "scope",
        "ratio",
        "pair_count",
    ],
)
def test_invalid_timing_cannot_promote_or_corrupt_state(trial, mutation):
    selected(trial)
    data = measured(trial)
    comparison = data["comparisons"][0]
    stats = comparison["timing"]
    if mutation == "missing_ba":
        stats["orders"].pop()
    elif mutation in {"nan", "zero"}:
        stats["orders"][0]["candidate"]["median_us"] = (
            float("nan") if mutation == "nan" else 0
        )
    elif mutation == "metric":
        stats["metric"] = "event_interval_us"
    elif mutation == "wins":
        stats["orders"][0]["win_fraction"] = 0
    elif mutation == "task_count":
        stats["tasks_per_call"]["candidate"] = 0
    elif mutation == "scope":
        comparison["scope"] = "isolated_stage"
    elif mutation == "ratio":
        stats["orders"][0]["median_ratio"] = 0.2
    else:
        stats["orders"][0]["pairs"] = 12
    before = trial["path"].read_bytes()
    with pytest.raises(ValueError):
        observe(trial, data)
    assert trial["path"].read_bytes() == before


def test_winner_is_restorable_after_editing_source_dependencies(trial):
    selected(trial)
    candidate_sha = trial["result"]["current"]["variant"]["sha256"]
    result = observe(trial, measured(trial))
    assert (
        result["phase"] == "retained"
        and result["best"]["variant"]["sha256"] == candidate_sha
    )
    (trial["root"] / "kernel.py").write_text("broken source")
    (trial["root"] / "helper.py").unlink()
    out = trial["tmp"] / "restore"
    experiment.run(trial["path"], export_best=out)
    assert (out / "kernel.py").read_text() == "def run(): return 2\n"
    assert (out / "helper.py").read_text() == "OFFSET = 3\n"
    assert all("content" not in f for f in result["best"]["variant"]["files"].values())
    assert result["best"]["correctness"]["cases"] == {
        "small": "passed",
        "tail": "passed",
    }
    assert result["best"]["performance"]["comparisons"][0]["outcome"] == "improved"


def test_baseline_failure_revokes_acceptance_and_can_be_revalidated(trial):
    selected(trial)
    data = measured(trial)
    data["baseline"]["correctness"]["cases"]["tail"] = "failed"
    result = observe(trial, data)
    assert (
        not result["best"]["verified"]
        and result["best"]["correctness"]["cases"]["tail"] == "failed"
    )
    with pytest.raises(ValueError, match="not passed"):
        experiment.run(trial["path"], export_best=trial["tmp"] / "bad-export")
    result = observe(trial, measured(trial, ab=1.2, ba=1.2))
    assert result["phase"] == "rejected" and result["best"]["verified"]
    assert result["best"]["correctness"]["cases"]["tail"] == "passed"


def test_dependency_and_configuration_are_part_of_identity(trial):
    initial = experiment.variant(trial["variant"])["sha256"]
    (trial["root"] / "helper.py").write_text("OFFSET = 4\n")
    changed = experiment.variant(trial["variant"])["sha256"]
    assert changed != initial
    trial["variant"]["configuration"]["workers"] = 2
    assert experiment.variant(trial["variant"])["sha256"] != changed


def test_replacing_trials_is_bounded_and_preserves_stronger_baseline(trial):
    selected(trial)
    observe(trial, measured(trial))
    best = trial["result"]["best"]
    selection = json.loads((trial["tmp"] / "select.json").read_text())
    sizes = []
    for value in (3, 4):
        (trial["root"] / "kernel.py").write_text(f"def run(): return {value}\n")
        trial["result"] = experiment.run(
            trial["path"], selection=write(trial["tmp"] / "select.json", selection)
        )
        observe(trial, measured(trial, ab=1.2, ba=1.2))
        assert trial["result"]["best"] == best
        sizes.append(trial["path"].stat().st_size)
    assert abs(sizes[1] - sizes[0]) < 20
    assert trial["result"]["remaining_trials"] == 0
    with pytest.raises(ValueError, match="budget exhausted"):
        experiment.run(trial["path"], selection=trial["tmp"] / "select.json")
    assert "history" not in json.loads(trial["path"].read_text())


def test_show_is_read_only_and_state_integrity_is_checked(trial):
    before = trial["path"].read_bytes()
    experiment.run(trial["path"])
    assert trial["path"].read_bytes() == before
    data = json.loads(before)
    data["best"]["verified"] = True
    write(trial["path"], data)
    with pytest.raises(ValueError, match="integrity"):
        experiment.run(trial["path"])


def test_environment_change_invalidates_evidence(trial):
    (trial["tmp"] / "toolchain.json").write_text('{"compiler": "changed"}')
    with pytest.raises(ValueError, match="Evidence changed"):
        experiment.run(trial["path"])


def test_framework_drift_invalidates_state(trial, tmp_path):
    framework = tmp_path / "framework"
    for name in feedback.FRAMEWORK_FILES:
        target = framework / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PYPTO_ROOT / name).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Framework source changed"):
        experiment.run(trial["path"], pypto_root=framework)


def test_snapshot_path_traversal_is_rejected(trial):
    trial["variant"]["files"].append("../toolchain.json")
    with pytest.raises(ValueError, match="relative file"):
        experiment.variant(trial["variant"])


def test_cli_show_is_compact_json_from_any_directory(trial):
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts/check.py"),
            "experiment",
            "--state",
            str(trial["path"]),
        ],
        cwd=trial["root"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["schema"] == experiment.SCHEMA and data["remaining_trials"] == 3
    assert "content" not in data["best"]["variant"]["files"]["kernel.py"]
