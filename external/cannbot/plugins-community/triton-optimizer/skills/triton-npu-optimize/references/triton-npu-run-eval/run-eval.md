
# Run-Eval Router

Use the bundled helper script at `<triton-npu-optimize>/scripts/run-eval/cli.py`:

```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py <subcommand> ...
```

For `probe-bench`, use the surface that actually exposes that subcommand in the current workspace. If the staged `cli.py` in this skill has not been updated yet, use the workspace's public `triton-agent probe-bench` command instead of guessing helper internals.

Read only the focused guide for the subcommand you are about to run:

- `run-test-baseline` / `run-test-optimize`: [
run-test.md](
run-test.md)
- `run-bench`: [
run-bench.md](
run-bench.md)
- `probe-bench`: [
probe-bench.md](
probe-bench.md)
- `profile-bench`: [
profile-bench.md](
profile-bench.md)
- `profile-report`: [
profile-report.md](
profile-report.md)
- `compare-perf`: [
compare-perf.md](
compare-perf.md)

During normal use:

- call `python3 <triton-npu-optimize>/scripts/run-eval/cli.py <subcommand> ...` directly
- do not read unrelated command guides
- do not reread Python files under `<triton-npu-optimize>/scripts/run-eval/` unless you need to debug, patch, or verify helper behavior
