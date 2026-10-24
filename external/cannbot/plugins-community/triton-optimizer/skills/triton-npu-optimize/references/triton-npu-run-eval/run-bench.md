# `run-bench`

Run a generated benchmark with:

```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --baseline-operator-file <operator>.py
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --baseline-operator-file <operator>.py --metric-source all --skip-latency-errors
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-bench --bench-file bench_a.py --operator-file a.py --output ./artifacts/a_perf.txt
```

Rules:

- Always pass both `--bench-file` and `--operator-file`.
- If `--bench-mode` is omitted, defaults to `torch-npu-profiler`.
- Use `--bench-mode torch-npu-profiler` or `--bench-mode perf-counter` only when you need to override the default.
- Use `--output <path>` when you need the perf artifact at a specific location.
- Use `--baseline-operator-file <path>` when you want `run-bench` to reuse or generate the baseline perf artifact automatically and then compare it against the candidate perf artifact.
- Use `--skip-latency-errors` when you want automatic baseline comparison to continue past recoverable latency-error entries.
- Use `--metric-source auto|kernel|total-op|all` when you want automatic baseline comparison to use a specific comparison basis.
- On success without `--baseline-operator-file`, `run-bench` prints `Perf file: <path>` and a short hint to use `compare-perf` instead of reading perf files directly.
- On success with `--baseline-operator-file`, `run-bench` prints `Baseline perf file: <path>`, `Perf file: <path>`, and the automatic comparison output.
- On failure, `run-bench` prints the captured benchmark output so the error remains diagnosable.

Baseline compare notes:

- The baseline perf artifact path is derived from `--baseline-operator-file` using the normal `<operator-stem>_perf.txt` naming rule beside that file.
- If that baseline perf artifact already exists, `run-bench` reuses it and does not rerun the baseline benchmark.
- If that baseline perf artifact does not exist, `run-bench` benchmarks the baseline operator first to create it, then benchmarks the candidate operator from `--operator-file`, then compares the two perf files automatically.

Mode notes:

- In both modes, the benchmark file is import-only. `run-bench` imports the module, calls `build_operator_api(operator_module)`, reads the declared cases from `build_bench_cases()`, and constructs each executable case via `build_bench_case_fn(operator_api, case)`.
- In `torch-npu-profiler` mode, the runner profiles each declared case with `torch_npu.profiler` and writes `latency-<case-id>` perf entries.
- In `perf-counter` mode, the runner measures per-iteration wall-clock time with `time.perf_counter()` instead of using NPU profiling. Results include `kernel_avg_time_us` (per-iteration average) but `ops` is empty. `total_op_avg_time_us` equals `kernel_avg_time_us`. Results are only comparable within the same mode; `compare-perf` rejects cross-mode comparisons.

Examples:

```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py --bench-mode torch-npu-profiler
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode perf-counter
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --baseline-operator-file <operator>.py
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --baseline-operator-file <operator>.py --metric-source kernel
```

Remote examples:

```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py --remote user@host:2222
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --bench-mode perf-counter --remote user@host:2222 --remote-workdir /tmp/triton-agent
```
