# Evaluation evidence contract

`scripts/evaluation_contract.py` is the shared parser used by `wm_ops.py refine`,
`check_round_artifacts.py`, profiling/precision checks and `wm_ledger.py`.
Read this reference when adding an evaluator or investigating disagreement between
an evaluation result and the world model.

## Input formats

| Format | Result fields | Effective reported speedup |
|---|---|---|
| Flat (lingxi) | Top-level `compilation_success`, `precision_passed`, `speedup` | Top-level speedup |
| Comparison | Fields under `comparison`; missing precision may come from `evolved.precision_passed` | Comparison speedup, falling back to top-level only when missing/null |
| Multi-shape | Nonempty `shape_results.target`, optional `generalization`, optional compatibility `comparison` | Minimum target speedup, also bounded by a supplied `aggregate.target_min_speedup` |

Multi-shape takes precedence. It checks every supplied target and generalization
result for compilation/precision; an explicit false comparison flag remains false.
Generalization performance regression remains subject to the existing gating and
supervisor policy. Geometric means cannot hide a slower target.

The flags must be JSON booleans. Passing evidence needs a finite, positive numeric
speedup; strings, booleans, NaN and infinity are not timings. Failed evaluations
may omit timing. Unrecognized or malformed evidence fails only its own variant,
so other parallel slots can finish.

An `evolved.pipeline` field alone does not select the comparison format. Pipeline
comes from `evolved.pipeline` when that key exists, otherwise top-level `pipeline`.
An explicit nested null records unavailable profiling and is not replaced by old
top-level data. Profiling absence is reported separately from compilation/precision.

## Measurement quality and search decisions

An explicit `measurement_quality` of `good`, `acceptable`, `noisy` or `unknown`
takes precedence. Otherwise quality is derived from finite, non-negative `cv_pct`:

| CV (%) | Quality | Substantial improvement over parent | Global stagnation threshold |
|---|---|---|---|
| `[0, 5)` | good | Existing good threshold: 5% | 2% |
| `[5, 15)` | acceptable | 8% | 3% |
| `>= 15` | noisy | 15% | 5% |
| Missing/invalid | unknown | 15% | 5% |

An invalid explicit quality stays unknown; it is not silently upgraded by CV.
Comparison quality fields take precedence over top-level fields. Multi-shape uses
the worst available target quality; aggregate quality can cover targets without
their own quality fields, but cannot override an explicitly noisier target.

Unknown quality uses the existing conservative noisy thresholds. For example,
`speedup=1.03, cv_pct=20` counts as global stagnation rather than clearing it.
Quality controls child expansion, global stagnation and stale-direction demotion;
the existing separate “beat parent at all” counter retains its semantics.

This is a deterministic quality policy, not a statistical significance test or
automatic remeasurement. `best_score` remains the best observed accepted score.

## Accepted scores and recorded evidence

Parsing does not replace the existing full-case checks, multi-shape parent
eligibility or same-run canonical timing rule. A positive result rejected by a
case-coverage check is still a failed node.

For single-shape refinement, the variant's own `performance.json` canonical timing
can supersede reported speedup. Reference time from a different baseline run is
never used to recalculate it. Multi-shape retains the minimum-target score.

Nodes record `compile_success`, `precision_passed`, `measurement_quality` and
`evaluation_source` when normalized evidence is available. These describe the
measurement; node `status` and `score` describe the accepted decision.

Lineage entries keep `speedup` aligned with the accepted node score, and record
`reported_speedup`, `status`, `measurement_quality` and `evaluation_source`
separately. Thus an evaluator's 1.49 value corrected to 0.68 by same-run canonical
timing appears as reported=1.49, accepted=0.68. A failed decision is not appended
as a passed lineage entry merely because its raw precision flag was true.

## CPU regression

```bash
EVOLUTION_TEST_PYTHON=python3.10 bash tests/run-tests.sh \
    --platform none --test unit/test-evolution-evaluation-contract.sh
```

The suite covers equivalent result formats, unknown/noisy quality, malformed
parallel slots, case coverage, canonical timing, real multi-shape aggregation
helpers and CLI round replay into world-model/ledger artifacts. These tests do
not claim NPU performance gains.
