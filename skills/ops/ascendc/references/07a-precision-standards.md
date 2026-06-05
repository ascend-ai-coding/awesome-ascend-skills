# Phase 7a — Precision standards (ecosystem open-source standard)

Used to judge whether a computation-class operator is accurate enough, via **MERE**
(mean relative error) and **MARE** (max relative error).

```
rel_err = abs(actual - golden) / (abs(golden) + 1e-7)   # 1e-7 avoids divide-by-zero
MERE = mean(rel_err)
MARE = max(rel_err)
```

Baseline: a single higher-precision reference (CPU, GPU, or an Ascend small-op
composition).

## Pass rule

**`MERE < Threshold` AND `MARE < 10 x Threshold`.**

## Threshold by dtype

| dtype | Threshold | ≈ value | MERE limit | MARE limit (10x) |
|---|---|---|---|---|
| FLOAT16 | 2⁻¹⁰ | 9.77e-4 | 9.77e-4 | 9.77e-3 |
| BFLOAT16 | 2⁻⁷ | 7.81e-3 | 7.81e-3 | 7.81e-2 |
| FLOAT32 | 2⁻¹³ | 1.22e-4 | 1.22e-4 | 1.22e-3 |
| HiFLOAT32 | 2⁻¹¹ | 4.88e-4 | 4.88e-4 | 4.88e-3 |
| FLOAT8 E4M3 | 2⁻³ | 1.25e-1 | 1.25e-1 | 1.25e+0 |
| FLOAT8 E5M2 | 2⁻² | 2.5e-1 | 2.5e-1 | 2.5e+0 |

## Notes

- Denominator is `abs(golden) + 1e-7` (not a clamp).
- Do not loosen thresholds casually. If genuinely required for a hardware-precision edge
  case, document the reason in the precision report.
- Auxiliary metrics (MaxAbsErr, MeanAbsErr, CosineSim) help analysis but do not decide
  pass/fail. CosineSim is 0/NaN for all-zero outputs — annotate rather than fail.
