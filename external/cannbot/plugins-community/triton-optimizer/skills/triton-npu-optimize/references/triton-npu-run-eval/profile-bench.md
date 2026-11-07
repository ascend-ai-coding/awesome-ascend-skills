# `profile-bench`

Profile a generated benchmark and summarize the copied-back `PROF_*` output with:

```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py profile-bench --bench-file bench_<operator>.py --operator-file <operator>.py
python3 <triton-npu-optimize>/scripts/run-eval/cli.py profile-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --target-op MatMul
```

Rules:

- Always pass both `--bench-file` and `--operator-file`.
- `profile-bench` always uses `torch_npu.profiler`. It does not accept `--bench-mode` and does not support `perf-counter` (which produces no profiler traces).

Remote examples:

```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py profile-bench --bench-file bench_<operator>.py --operator-file <operator>.py --case-id <id> --remote user@host:2222
python3 <triton-npu-optimize>/scripts/run-eval/cli.py profile-bench --bench-file bench_<operator>.py --operator-file opt_<operator>.py --case-id <id> --remote user@host:2222 --remote-workdir /tmp/triton-agent --keep-remote-workdir
```

Use `profile-report` to re-summarize an existing `PROF_*` directory without re-running the benchmark:

```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py profile-report --profile-dir PROF_000001_.../ --target-op MatMul
python3 <triton-npu-optimize>/scripts/run-eval/cli.py profile-report --profile-dir PROF_000001_.../ --target-op MatMul --format json
```

If the inline `profile-bench` summary is not enough, rerun `profile-report` on the copied-back directory or inspect the raw files inside that `PROF_*` directory directly.
