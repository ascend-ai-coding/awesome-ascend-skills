# `probe-bench`

Use `probe-bench` when you need a cheap directional screen for one candidate operator against one required baseline operator.

Treat it as screening only:

- use it to reject clearly bad directions early
- use it to keep clearly promising directions long enough to justify a canonical benchmark
- do not use it as the official source for round speedups or benchmark deltas
- do not write probe artifacts into canonical round perf paths or treat them as `submit-round` evidence

Run with the `probe-bench` surface available in the current workspace:

```bash
triton-agent probe-bench \
  --bench-file bench_<operator>.py \
  --operator-file opt_<operator>.py \
  --baseline-operator-file baseline/<operator>.py
```

If the staged helper script in this skill already exposes the subcommand, the equivalent helper form is:

```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py probe-bench \
  --bench-file bench_<operator>.py \
  --operator-file opt_<operator>.py \
  --baseline-operator-file baseline/<operator>.py
```

Rules:

- Always pass `--bench-file`, `--operator-file`, and `--baseline-operator-file`.
- `--metric-source auto|kernel|total-op` controls which timing basis the screen uses. Do not expect `all` here.
- Default output is intentionally short and decision-oriented. Treat `Probe classification` and the final `Summary:` line as the screening result.
- Advisory geomean, average improvement, and improved or regressed case counts are non-authoritative diagnostics.
- Use `--verbose` only when you need cache or artifact-path diagnostics. Baseline cache hit or miss details and hidden probe perf paths are verbose-only.
- The helper caches the baseline-side probe perf artifact under `.triton-agent/` so repeated screening against the same baseline is cheaper.
- If the selected bench mode cannot apply the fast probe warmup and repeat caps, the command warns and falls back to canonical benchmark execution for that run. Treat that probe result as less useful for cost-saving decisions.

After a promising or inconclusive probe result, still run canonical `run-bench` and then `compare-perf` before recording any official optimization conclusion.
