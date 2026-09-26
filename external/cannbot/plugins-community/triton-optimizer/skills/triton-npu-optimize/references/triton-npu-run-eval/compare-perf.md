# `compare-perf`

Use `compare-perf` after you already have two perf artifacts for the same benchmark cases, typically:

- after `run-bench` on a baseline operator and an optimized operator
- during optimize workflows when you want both per-case deltas and a headline speed summary

Run:

**Kernel-target optimize rounds** (kernel latency is the primary optimization target):
```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py compare-perf \
  --baseline baseline/<operator>_perf.txt \
  --compare opt-round-N/opt_<operator>_perf.txt \
  --metric-source kernel
```

**Operator-target optimize rounds** (both kernel and total-op views are needed):
```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py compare-perf \
  --baseline baseline/<operator>_perf.txt \
  --compare opt-round-N/opt_<operator>_perf.txt \
  --metric-source all
```
Record `effective_metric_source: total-op` for the canonical round conclusion.

Rules:

- Keep the baseline file in the standard `latency-<id>: <float>` format.
- The compare-side file may include extra summary lines such as `mean_ms: ...`; the helper ignores them unless they replace a required latency entry.
- `--metric-source auto|kernel|total-op|all` selects how `compare-perf` derives each case's timing: `auto` preserves the current kernel-first fallback behavior, `kernel` requires kernel latency, `total-op` requires total-op aggregation (from raw op statistics in profiler modes, or derived from `kernel_avg_time_us` in `perf-counter` mode), and `all` prints both kernel and total-op comparison sections.
- Cross-mode comparison is rejected with an error. Perf-counter results support all metric sources: `kernel`, `total-op`, and `all` produce the same value; `auto` resolves to the same result via kernel-first fallback.
- By default, non-recoverable `# latency-error-<id>:` markers fail the comparison immediately. Add `--skip-latency-errors` to keep comparing valid cases and report skipped-case errors at the end.
- The command prints per-case deltas plus `Avg improvement` and `Geomean speedup`.
- During optimize workflows, treat this command as the authority for claimed benchmark deltas and speedups.
- For kernel-target optimize rounds, prefer the kernel-oriented view, but record the resolved `effective_metric_source` when fallback changes the real basis.
- For operator-target optimize rounds, use `--metric-source all` so both kernel and total-op views are visible, then treat the total-op section as the canonical round conclusion.
