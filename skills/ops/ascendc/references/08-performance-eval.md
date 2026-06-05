# Phase 8 — Performance evaluation (torch_npu.profiler)

Benchmark the custom operator against a **baseline** and produce a dual-path Markdown
report. Collection always uses `torch_npu.profiler`.

## Two hard constraints

1. **Always a dual-path comparison** (custom vs baseline) — never a single-path report.
   The baseline **must run on NPU** (use a small-op composition if no equivalent API
   exists; tensor ops only, no Python scalar loops).
2. **Always read `design.md` first** before generating any JSONL case, to extract param
   constraints, typical shapes, dtypes, and execution modes.

## Case source

1. Read `csrc/ops/<op>/test/<op>-test-cases.md` (Phase 3): `SUPPORTED_DTYPES`,
   `TEST_SHAPES`, `GENERAL_SHAPES`, `NPU_CALL`, `CPU_REF`.
2. Read `csrc/ops/<op>/design.md`: dtypes, param constraints, typical shapes, execution
   modes, perf-sensitive branches.

Convert to JSONL: pick representative shapes (small/medium/large), iterate all dtypes,
fill attributes from `design.md` constraints, total **>= 8 cases**. Cover every execution
mode `design.md` defines (e.g. transpose vs non-transpose). JSONL spec:
[`08b-perf-case-jsonl.md`](08b-perf-case-jsonl.md).

## Baseline path decision tree

```
Equivalent baseline API exists (torch.nn.functional.* / torch_npu builtin)?
  ├─ yes → use it as the baseline path
  └─ no  → implement a small-op composition baseline (mandatory)
            from design.md reference impl, using PyTorch tensor ops
            (torch.zeros, slicing, .permute(), torch.cat, ...) — runnable on NPU
```

Mark the baseline type in the report header. Never degrade to single-path because no
equivalent API exists.

## Fixed profiler schedule

| Param | Value |
|---|---|
| `warmup` | **5** (do not change) |
| `active` | **5** (do not change) |
| `wait` | 0 (default) |
| `repeat` | 1 (default; if >1, document CSV selection) |

Call `prof.step()` at the end of every step; total steps = `repeat*(wait+warmup+active)`.

## File layout (all under `csrc/ops/<op>/test/`)

| Artifact | Name |
|---|---|
| cases (JSONL only) | `<op>_perf_cases.jsonl` (no `.json`) |
| Markdown report | `<op>_torch_npu_profiler_report.md` (no `_results.json`) |
| profiler export root | `test/profiler_trace/` |

Copy [`../examples/layer_norm_profiler_reference/`](../examples/layer_norm_profiler_reference/)
into `test/`, then replace op name, forward call, `build_inputs`, and trace subdir.

## Metrics

See [`08a-profiler-and-metrics.md`](08a-profiler-and-metrics.md). In short: sum `Total
Time(us)` over operator rows in each run's `op_statistic.csv`, divide by `active*repeat`
(here `active=5`, `repeat=1` → divisor `5`).

## Report structure (`<op>_torch_npu_profiler_report.md`)

1. Title.
2. **Unified comparison table** (single table, all dtypes):
   `Case | Shape | DType | Custom(us) | Baseline(us) | Speedup`.
3. **Overall summary** (key/value: case count, mean speedup, custom-better count,
   baseline-better count) + per-dtype summary table.
4. **Short analysis**: >= 3 bullets (overall trend, dtype/shape-size differences,
   memory- vs compute-bound characteristics).

## In-chat display (MANDATORY)

In the current reply MUST show: the unified comparison table (with DType column;
truncate with a note if long), the overall + per-dtype summary, and >= 3 short
conclusions. Put the report path after the data — never reply with only a path.

## Gate (before optional Phase 9)

- [ ] design.md + test-case doc read; >= 8 JSONL cases covering all modes/dtypes.
- [ ] baseline runs on NPU (API or small-op composition).
- [ ] `torch_npu.profiler` used; `warmup=5, active=5` unchanged.
- [ ] dual-path report written; displayed in chat with table + summary + >= 3 conclusions.

## Anti-patterns (NEVER)

- NEVER produce a single-path report.
- NEVER change warmup/active away from 5.
- NEVER use a non-profiler timing as the conclusion.
- NEVER write a Python scalar-loop baseline (profiles CPU, not NPU).
- NEVER reply with only the report path.
