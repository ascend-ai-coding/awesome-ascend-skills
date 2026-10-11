
# Prepare Optimize Baseline

## Goal

Establish a reusable canonical `baseline/` before any optimize round begins.

Use this skill when optimize work cannot yet start because baseline artifacts are missing, invalid, or no longer match the current operator workspace.

## Outputs

- reusable correctness and benchmark harnesses
- `baseline/`
- `baseline/state.json`
- `baseline/<operator>_perf.txt`

## Workflow

### 1. Inspect And Reuse

- Inspect the operator workspace before generating anything new.
- Reuse existing correctness and benchmark harnesses when they already validate the current operator workspace.
- If a usable correctness harness is missing, use the [gen-test](../triton-npu-gen-test/gen-test.md) reference.
- If a usable benchmark harness is missing, use the [gen-bench](../triton-npu-gen-bench/gen-bench.md) reference.

### 2. Reach A Benchmarkable Start

- Use the [run-eval](../triton-npu-run-eval/run-eval.md) reference for correctness validation and benchmark validation.
- If the current operator or harnesses need repair before they validate cleanly, do only the minimum repair needed to reach a correct, benchmarkable starting point.
- Treat this phase as baseline repair, not as an optimization round.

### 3. Write Canonical Baseline Artifacts

- Read the `<Language>-npu-optimize` (where `<Language>` is `triton` or `tilelang`) skill's `../artifacts.md` before writing `baseline/state.json`.
- Create `baseline/`.
- Write `baseline/state.json`.
- Write `baseline/<operator>_perf.txt`.
- Keep the canonical baseline artifacts anchored to the operator state that just passed correctness and benchmark validation.

### 4. Gate The Baseline

- Use the [optimize-state](../triton-npu-optimize-state/optimize-state.md) reference's `submit-baseline` subcommand to submit the baseline and validate it.
- Keep repairing baseline state until the baseline submission passes.
- Stop once the workspace has a reusable canonical baseline.

## Completion Condition

This skill is complete only when:

- the workspace has reusable correctness and benchmark harnesses
- `baseline/` exists
- `baseline/state.json` exists and matches the optimize artifact contract
- `baseline/<operator>_perf.txt` exists
- `submit-baseline` validation passes (see [optimize-state](../triton-npu-optimize-state/optimize-state.md))

## Hard Rules

- Do not start `opt-round-N/` from this skill.
- Do not do open-ended optimization work here.
- Do not skip benchmark validation.
- Do not treat a partially repaired workspace as a reusable baseline.
