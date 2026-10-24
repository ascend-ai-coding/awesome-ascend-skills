# Evaluation and harness reuse

Reuse reduces repeated initialization, compilation and data preparation during a
Stage-7 search. The correctness command, lint and profiling method remain those
defined in [evaluator.md](evaluator.md).

## Select an evaluator mode

| Mode | Behavior |
|---|---|
| `subprocess` | Separate correctness and profiling processes; Python API and BO CLI default. |
| `resident` | One worker; correctness and profiling share the candidate module and JIT objects. |
| `cached` | Resident execution plus complete results and exact-source syntax/compiler-capacity failures. |
| `prepared` | Cached execution plus guarded deterministic inputs and pure golden outputs. |

The standalone launcher defaults to `resident`. Pass the mode explicitly and use
the same operator, paths, device and mode for INIT/STEP and BO evaluations.

From the repository root, in the operator's existing Ascend Python environment:

```sh
SKILL_ROOT=cannbot-skills/ops/pypto-op-perf-panko-manual
LINT_ROOT=cannbot-skills/plugins-official/pypto-op-orchestrator/hooks/pypto-op-lint
OP_DIR=/absolute/path/to/GELU
MODE=cached
python "$SKILL_ROOT/scripts/evaluation_session.py" \
  --op-dir "$OP_DIR" --op GELU --op-file "$OP_DIR/GELU_impl.py" \
  --device 0 --test-command "python \"$OP_DIR/test_GELU.py\"" \
  --lint-root "$LINT_ROOT" --mode "$MODE"
```

Keep the operator's original test command and lint configuration. Resident tests
must use the evaluation Python interpreter; provide its full path when needed.
Append `--evaluation-mode "$MODE"` to the existing `panko_harness.py block`
command to use the same mode for BO.

## Cache validity and preparation

A cache hit returns the original correctness, lint and profiler evidence and
sets `evaluation_cache_hit`; it does not take a new latency measurement. Reuse is
invalidated by source, operator files (including binary and extensionless
fixtures), SPEC, lint, configuration, device, interpreter, package versions,
host, environment or search generation changes. Changes during evaluation
reject the result.

Generated `tuning_notes.md` and the `optimization`, `output`, `prof` and
`__pycache__` directories are excluded. Keep test inputs outside these locations.
Set `PANKO_RUNTIME_SIGNATURE` when external dependencies or runtime binaries
change without a corresponding package version change. Lint failures, transient
device faults and timeouts are retried. Compiler-capacity failures close the
worker before the next candidate.

Prepared mode requires deterministic fixtures and a pure `<op>_golden` function
in `<op>_golden.py`. Tensor contents and versions are checked; mutations trigger
recomputation. Unsupported fixtures use fresh preparation. Content checks and
retained tensors add time and memory cost, so choose the mode by workload.

## Harness reuse and lifecycle

Replace `python "$SKILL_ROOT/scripts/panko_harness.py" <original arguments>` with
`python "$SKILL_ROOT/scripts/harness_session.py" --op-dir "$OP_DIR" -- <original arguments>`.
The wrapper retains imports and API metadata while preserving the original
arguments, output and stop/report routines. `--timeout` bounds a complete harness
command (default 3600 seconds); match it to the command's BO budget.

Before a new search's first evaluation, reset once; do not reset a resumed STEP:

```sh
python "$SKILL_ROOT/scripts/evaluation_session.py" --op-dir "$OP_DIR" --reset
```

On completion, abort, or when disabling reuse, close both workers:

```sh
python "$SKILL_ROOT/scripts/harness_session.py" --op-dir "$OP_DIR" --close
```

Return to the original harness command and `subprocess` evaluation to disable
reuse. Optimizer conversation reuse is separate: resume only within the same
operator, role and configuration, reading persisted state each cycle. Start a
fresh conversation when that context changes.
