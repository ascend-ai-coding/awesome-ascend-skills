# `profile-report`

Summarize an existing `PROF_*` profiling directory without re-running the benchmark:

```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py profile-report --profile-dir PROF_000001_.../ --target-op matmul_kernel
python3 <triton-npu-optimize>/scripts/run-eval/cli.py profile-report --profile-dir . --target-op MatMul --format json
```

## When to use

- Re-summarize profiling data with different `--target-op` values without re-profiling
- Inspect historical `PROF_*` directories from past runs
- Emit structured JSON for downstream round-analysis workflows when the profile run was already completed
- Share profiling reports without access to the original operator or benchmark files

## Mode auto-detection

`profile-report` automatically detects the profile mode from the directory structure:

- A `PROF_*/mindstudio_profiler_output/` directory → **msprof** mode
- A `*/ASCEND_PROFILER_OUTPUT/` directory → **standalone** mode
- When both exist (standalone `torch_npu.profiler` also emits a `PROF_*` subdirectory), standalone mode is preferred because `ASCEND_PROFILER_OUTPUT/kernel_details.csv` provides richer pipeline ratio data

The detected mode determines which files are parsed:

| Mode | Parsed files |
|---|---|
| `msprof` | `op_statistic_*.csv`, `op_summary_*.csv`, `task_time_*.csv`, `api_statistic_*.csv`, `msprof_*.json` |
| `standalone` | `op_statistic.csv`, `kernel_details.csv`, `operator_details.csv`, `step_trace_time.csv`, `api_statistic.csv`, `trace_view.json` |

The common data is normalized into the same output format. Mode-specific data (e.g. `task_time` for msprof, `step_trace` for standalone) is included only when available in the source mode.

## Arguments

| Argument | Required | Default | Purpose |
|---|---|---|---|
| `--profile-dir` | yes | — | Path to a `PROF_*` directory, an `ASCEND_PROFILER_OUTPUT` directory, or a parent directory containing one |
| `--target-op` | no | (hottest) | Operator name to summarize; inferred from `op_statistic` by total time when omitted |
| `--format` | no | `markdown` | Output format: `markdown` or `json` |
| `--top` | no | `5` | Number of top operators to include in the hotspot table |

## Selection rules

- When `--target-op` is provided, the report targets the named operator and fails if it is not found in `op_statistic`
- When `--target-op` is omitted, the script selects the operator with the highest `Total Time(us)` in `op_statistic` and marks the selection as inferred

## Output

### Markdown

Sections include:
- **Operator timing**: core type, invocation count, total/avg/min/max time, runtime ratio
- **op_summary cross-check**: matched invocation rows with duration statistics
- **Core type totals**: aggregated by normalized core type bucket (cube/vector/scalar/other)
- **Data movement hotspots**: transfer-like operators sorted by total time
- **Layered profiler signals**: task timeline, host API, msprof tracks, binary availability
- **Top operators by total time**: hotspot table

### JSON

All structured data from the parsed artifacts, plus classification results (operator type guess, bound analysis). Intended for machine consumption by downstream workflows such as `triton-npu-analyze-round-performance`.

## Relationship to `profile-bench`

- `profile-bench`: runs the benchmark with a profiler AND prints a summary
- `profile-report`: only prints a summary from existing profiling data — no benchmark execution

Use `profile-report` when the profiling run has already completed and you only need to view or export the data.
