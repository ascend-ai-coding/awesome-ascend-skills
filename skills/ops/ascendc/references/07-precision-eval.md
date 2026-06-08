# Phase 7 — Precision evaluation

Run a comprehensive precision suite on the built operator and produce a structured report.

Precondition: operator compiled/installed and passing the basic functional test; the
unified test-case doc `csrc/ops/<op>/test/<op>-test-cases.md` exists (Phase 3).

## Flow

```
Load test-case doc + info → adapt cases → generate scripts → run → report
```

## Step 1 — Load inputs

Read `<op>-test-cases.md` for `SUPPORTED_DTYPES`, `TEST_SHAPES`, `BOUNDARY_VALUES`,
`NPU_CALL`, `CPU_REF`. Cross-check with code: NPU call vs `register.cpp`; supported dtypes
vs `op_host` `TORCH_CHECK`; input domain vs `design.md`. If the doc is missing, design
cases yourself and note it in the report.

## Step 2 — Adapt cases

- Reuse `TEST_SHAPES` / `BOUNDARY_VALUES` from the doc directly when present.
- Every shape and every boundary value iterates **all** supported dtypes.
- Total `(len(TEST_SHAPES) + len(BOUNDARY_VALUES)) x len(SUPPORTED_DTYPES) >= 30`.
- Keep shapes reasonable (<= ~200K elements; fp16 max ~65504).

## Step 3 — Generate scripts (from templates)

Read templates under [`../templates/precision/`](../templates/precision/), replace
placeholders, write to `csrc/ops/<op>/test/`:

| Template | Output |
|---|---|
| `test_op_precision_template.py` | `test_<op>_precision.py` |
| `run_precision_report_template.py` | `run_<op>_precision_report.py` |

Placeholders: `{{OP_NAME}}`, `{{NPU_CALL}}`, `{{CPU_REF}}`, `{{SUPPORTED_DTYPES}}`,
`{{INPUT_LOW}}`, `{{INPUT_HIGH}}`, `{{TEST_SHAPES}}`, `{{BOUNDARY_VALUES}}`.

## Metrics

Decision metrics (MERE/MARE, ecosystem open-source standard — see
[`07a-precision-standards.md`](07a-precision-standards.md)):

```
rel_err = abs(npu - ref) / (abs(ref) + 1e-7)
MERE = mean(rel_err);  MARE = max(rel_err)
```

**Pass rule: `MERE < Threshold` AND `MARE < 10 x Threshold`.**

Auxiliary (analysis only, not pass/fail): MaxAbsErr, MeanAbsErr, CosineSim.

## Step 4 — Run

```bash
source ${CANN_PATH}/*/set_env.sh && conda activate <env>
cd <project_root>
python3 -m pytest csrc/ops/<op>/test/test_<op>_precision.py -v --tb=short
python3 csrc/ops/<op>/test/run_<op>_precision_report.py
```

Failure handling:

| Failure | Direction |
|---|---|
| RuntimeError (kernel) | input out of domain / dtype unsupported |
| AssertionError (precision) | MERE/MARE slightly over Threshold → boundary effect? |
| single-dtype fails | check that dtype's Threshold; MARE concentrated in few points? |
| many fails | bug in Compute logic |

When precision fails and thresholds cannot fix it, run root-cause debugging:
[`07b-precision-debug.md`](07b-precision-debug.md). Only loosen thresholds for genuine
hardware-precision edge cases, and document it in the report.

## Step 5 — Report

Write `csrc/ops/<op>/test/<op>_precision_report.md` (and `*.json` if the script emits it)
with: overview table (total/pass/fail/pass-rate), threshold table, regular-shape results
(grouped by category), boundary-value results, per-dtype summary, and **>= 3 key
findings**.

## In-chat display (MANDATORY)

After running, in the **current reply** you MUST show: overview (totals + pass rate);
failures (case/shape/dtype + key error metric) or an explicit "all passed"; **>= 3 key
findings**; one line on the standard used (MERE/MARE + per-dtype Thresholds). Put the
report path **after** the readable results — never reply with only a path.

## Gate (before Phase 8)

- [ ] pytest green; JSON + Markdown report written.
- [ ] `(shapes + boundary) x dtypes >= 30`; every dtype tested.
- [ ] Results displayed in chat (overview, failures, >= 3 findings).

## Anti-patterns (NEVER)

- NEVER generate only the report file without showing the overview + conclusions in chat.
- NEVER hide failed-case counts behind a path.
- NEVER loosen thresholds without documenting why.
