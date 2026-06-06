# Phase 8a — Profiler and metrics

How `torch_npu.profiler` exports data and how to aggregate the timing metric.

## Profiler usage

- Each `with torch_npu.profiler.profile(...)` produces an export directory suffixed
  `_ascend_pt` under the handler directory.
- CSV path: `.../*_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv`.
- Run **one independent `with`** per case and per implementation (`custom` / `baseline`).
  Suggested subpaths: `{trace_root}/{op_tag}/{custom|baseline}/case_XXX/`; clear
  `case_XXX` before each run.
- Schedule fixed at `warmup=5, active=5`; call `prof.step()` each step.

Minimal shape:

```python
import torch_npu
sched = torch_npu.profiler.schedule(wait=0, warmup=5, active=5, repeat=1)
with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=sched,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir)) as prof:
    for _ in range(1 * (0 + 5 + 5)):
        out = forward(*inputs)
        prof.step()
```

## Metric aggregation

1. For a single `with`'s CSV: **sum `Total Time(us)` across all operator rows**.
2. Divide by `active*repeat` (`divisor_mode=active_steps`) or by `active` only
   (`active_only`). With `active=5, repeat=1` → **divisor = 5**.
3. Speedup = `baseline_us / custom_us` (>1 means the custom operator is faster).

## Robustness notes

- The CSV header may carry a BOM or vary; match the **`Total Time(us)`** column tolerantly.
- `repeat > 1` may emit multiple `*_ascend_pt` exports; if selecting by mtime, document
  the selection semantics.
- If the custom operator is not registered/loaded, only the baseline path runs — load the
  custom library before comparing.

## Common pitfalls

- `warmup`/`active` changed away from 5.
- Not using `torch_npu.profiler`, or `prof.step()` inconsistent with the schedule.
- Reporting a path without showing the table/summary in chat.
