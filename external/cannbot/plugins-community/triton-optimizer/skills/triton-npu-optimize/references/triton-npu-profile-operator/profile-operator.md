
# Ascend NPU Operator Profiler

Profile Ascend NPU operators and summarize the resulting timing data.

## Default workflow

### Profiling + summary (profile-bench)

1. Profile benchmark harnesses through the [run-eval](../triton-npu-run-eval/run-eval.md) reference's `profile-bench` helper.
   - For the exact wrapper invocation, read the [run-eval](../triton-npu-run-eval/run-eval.md) reference's `profile-bench.md`.
   - The standard argument shape is:

   ```text
   profile-bench --bench-file bench_matmul.py --operator-file matmul.py
   ```

2. The helper runs the profiler, copies back the generated local `PROF_*` directory, and prints a summary inline.

3. Review the inline summary in the conversation. Call out:
   - which operator was analyzed
   - whether that operator was explicit or inferred
   - total, average, min, and max time
   - runtime ratio and top hotspots

### Re-reporting without re-profiling (profile-report)

When the profiling data already exists (e.g. from a previous `profile-bench` run), use `profile-report` to generate a new summary without re-running the benchmark:

```text
profile-report --profile-dir PROF_000001_.../ --target-op matmul_kernel
```

For round-analysis workflows that need structured signals, use JSON mode:

```text
profile-report --profile-dir PROF_000001_.../ --target-op matmul_kernel --format json
```

This is useful when you want to:
- Inspect the same `PROF_*` data with different `--target-op` values
- Re-summarize historical profiling data
- Emit structured JSON for downstream workflows without re-profiling

## Working rules

- Prefer the `triton-npu-run-eval` skill's `profile-bench` helper for benchmark profiling, especially when the workflow is remote-aware.
- profile one selected `--case-id <id>` case with `torch-npu-profiler`; benchmark profiling must not receive `--bench` or `--num-bench`.
- If the benchmark file declares exactly one case, the helper may auto-select it; otherwise provide `--case-id`.
- When the outer task is remote-aware, pass the same `--remote` and `--remote-workdir` settings through `profile-bench` so profiling runs on the remote machine while the resulting `PROF_*` directory is copied back locally.
- Treat `op_summary_*.csv` as potentially large. Do not dump it into the conversation or read it naively line-by-line into memory if a streaming pass is enough.
- Prefer the bundled summary script over ad hoc shell parsing so the report stays consistent.
- Prefer the bundled summary script in JSON mode when the downstream task is [analyze-round-performance](../triton-npu-analyze-round-performance/analyze-round-performance.md) or any profiler-first layered analysis workflow.
- If the user names the operator, pass it through `--target-op`.
- If the user does not name the operator, let the script infer the hottest operator from `op_statistic` and state that this is an inference.
- Always show the final Markdown summary in the conversation instead of only mentioning generated files.

## Bench mode contract

- `torch-npu-profiler`
  - Use:
    ```text
    profile-bench --bench-file bench_<operator>.py --operator-file <operator>.py --case-id <id>
    ```
  - Runtime command shape inside the helper:
    ```bash
    python3 bench_runtime.py profile-one --bench-file bench_<operator>.py --operator-file <operator>.py --case-id <id>
    ```
  - Do not pass `--bench` or `--num-bench`.
  - The helper profiles one selected `--case-id <id>` case with `torch_npu.profiler`.

## Validation checks

- Accept any of these input shapes for CSV summarization: a `PROF_*` directory, a `mindstudio_profiler_output/` directory, or a parent directory that contains one of them.
- Before summarizing CSV output, confirm `mindstudio_profiler_output/` exists and contains at least one `op_statistic_*.csv`.
- When `op_summary_*.csv` exists, use it as a cross-check for the target operator but keep the main summary anchored on `op_statistic`.
- Before running binary parsing, confirm the `.bin` file exists and contains `ZZ{` markers.

## Failure handling

- If no profile directory can be resolved, re-run the profiling command and check the working directory.
- If `mindstudio_profiler_output/` or `op_statistic_*.csv` is missing, stop and report that the profiler output is incomplete.
- If the requested operator is not present in `op_statistic`, list the available operators and ask the user to confirm the target.
- If `op_summary` uses unrecognized columns, fall back to `op_statistic` and mention that the `op_summary` aggregation was unavailable.
- If `.bin` parsing finds no `ZZ{` markers, treat the file as unsupported profiler data instead of guessing.

## Bundled resources

- `<triton-npu-optimize>/scripts/profile-operator/reporter.py`
  Unified profile reporter used by the `profile-report` subcommand. Loads profiles from either mode, runs classification (operator type, bound analysis), and renders Markdown or JSON output.

- `<triton-npu-optimize>/scripts/profile-operator/models.py`
  Typed data model for parsed profile data (`OperatorStats`, `KernelInvocation`, `PipelineStage`, `ParsedProfile`). Shared by `reporter.py` and all parsers.

- `<triton-npu-optimize>/scripts/profile-operator/parser_base.py`
  Shared parsing utilities: mode auto-detection, `op_statistic` CSV parser, `api_statistic` CSV parser. Used by both mode-specific parsers.

- `<triton-npu-optimize>/scripts/profile-operator/msprof_parser.py`
  Parses `PROF_*/mindstudio_profiler_output/` artifacts: `op_statistic`, `op_summary`, `task_time`, `api_statistic`, `msprof` JSON.

- `<triton-npu-optimize>/scripts/profile-operator/torch_npu_profiler_parser.py`
  Parses `ASCEND_PROFILER_OUTPUT/` artifacts: `op_statistic`, `kernel_details`, `operator_details`, `step_trace_time`, `api_statistic`, `trace_view`.

- `<triton-npu-optimize>/scripts/profile-operator/parse_bin.py`
  Keep this for raw profiler binary inspection when the user provides files such as `visualize_data.bin`.
  It also owns the structured deep-signal parsing that the summary script can reuse.

- `profiler-csv-spec.md`
  Read this when you need the expected CSV layout or column names for `op_statistic` and `op_summary`.

- `references/binary-format-spec.md`
  Read this only when the user needs `.bin` parsing details.

## Examples

Profile a benchmark and summarize the hottest operator:

```text
profile-bench --bench-file bench_matmul.py --operator-file matmul.py --case-id fp16_1024
```

Profile a benchmark and summarize a known operator:

```text
profile-bench --bench-file bench_matmul.py --operator-file matmul.py --case-id fp16_1024 --target-op MatMul
```

Profile one benchmark case on a remote machine and keep the remote workspace:

```text
profile-bench --bench-file bench_matmul.py --operator-file opt_matmul.py --case-id fp16_2048 --remote user@host:2222 --remote-workdir /tmp/triton-agent --keep-remote-workdir
```

Then re-summarize the copied-back profile through the [run-eval](../triton-npu-run-eval/run-eval.md) reference's `profile-report` helper, for example `profile-report --profile-dir PROF_000001_.../ --target-op MatMul`.

Inspect a raw profiler binary block:

```bash
python3 <triton-npu-optimize>/scripts/profile-operator/parse_bin.py visualize_data.bin --block-id 0
```
