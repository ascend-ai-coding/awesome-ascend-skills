# Device runs and measurement

Before remote operations, read the target project’s external environment and execution configuration. It must identify machines, paths, device selection, locks and environment scripts. Keep these machine-specific settings outside the skill.

## Execution and evidence capture

1. Pin source and installation identity; confirm CANN/compiler versions and [actual imports](#verify-imports-in-the-execution-process). Python-only changes can use the documented single-file synchronization flow. For C++ changes, confirm the new library is installed and actually loaded: a successful build does not prove that it took effect.
2. Run one board job at a time and respect device locks. A failed lock with empty logs may mean the test never started. Select idle resources through the environment configuration; do not terminate others' jobs.
3. Capture full stdout/stderr and retain the real exit code and completion marker before extracting a summary. Numerical harnesses must assert or exit nonzero on mismatches and caught exceptions; printing `FAIL` or a failure list still leaves an ordinary script's exit status at 0. Do not use the exit status of `pytest | tail` as a test result or interpret a missing summary as absence of failures.
4. Estimate the number of compilation variants first. Start with a probe answering one question; batch related targets to amortize initialization, then expand coverage experiments.

Use [check.py run](../tooling/checks.md#run-a-command-after-source-checks) in the execution environment, under the configured device lock and timeout. Include the submitted kernel, host wrapper and test in `--source`; run the actual assertion-based test or evaluator as the child command. It preserves child output/status and reports source SHA256, and rejects a successful command whose selected sources changed during the run. It cannot make a print-only harness assert, infer numerical success, inspect another host's imports, or recover a status already swallowed by a child shell pipeline.

Calibrate the test's failure path once with a deliberately corrupted result at the original tolerance: it must fail before any timing callback. Assert public output shape/dtype, finiteness where required, guards and the actual evaluator's numerical criterion. A reported error norm or `ok: false` followed by process exit 0 is an invalid acceptance harness. Use the reference's required precision and output dtype for each branch, including BF16; keep correctness and timing behind the same assertions.

## Integrate a supplied evaluator

Inspect how the evaluator discovers the public callable and loads the submission before invoking its build/install path. A source-directory option can build a wheel and install it into the current interpreter; it does not imply source-only loading. For a read-only shared installation, verify the evaluator's supported source-loading mode or use a trial-owned installation environment, preserving the pinned framework dependencies. Source loading still uses the original golden, checker and performance protocol.

Inspect internal device-worker defaults as well as the parent command: an outer device lock does not serialize a worker pool. Match the environment's per-device concurrency policy and isolate compilation roots for any permitted concurrent workers. Set the evaluator's report, child-capture and temporary directories to trial-owned locations through its supported configuration before launching. A top-level output option need not control child reports or `TMPDIR`. Verify the [actual imports](#verify-imports-in-the-execution-process) in a fresh process matching the evaluator's execution context before launching the device evaluator.

For a score target, complete the [capture preflight](native-timing.md#verify-capture-before-expanding-performance-coverage) as part of this first integration probe. A correctness-only invocation does not establish the evaluator's performance path.

## Verify imports in the execution process

Probes must not insert a source tree into `sys.path`. Preserve existing `PYTHONPATH` entries in the parent and child processes: sourcing a pinned environment does not protect it from a later path replacement. Verify imports after the final path edits.

For a locally overlaid evaluation package, copy its package initializers and assert that the exact public attribute discovered by the evaluator is callable. Importing a submodule can succeed while exporting a module instead of its function. Verify that both the exported entry and implementation module resolve to the intended trial root in the evaluator's import context. Prepending a directory containing only a subpackage can still select an installed regular package. Confirm the actual import before interpreting correctness or timing; hashing an unimported candidate file does not identify the code that ran.

For an evaluator that launches the current Python without rewriting paths, preserve the environment's installation entries when adding its packages:

```bash
source "$ENV_SCRIPT"
export PYTHONPATH="$SUBMISSION_ROOT:$BENCH_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import sys, pypto, pypto_pro; print(sys.executable); print(pypto.__file__); print(pypto_pro.__file__)'
```

Compare these paths with the pinned installation before launching the evaluator in the same shell. If it changes interpreter, working directory or import paths, repeat the identity check inside that child context; a parent-only check is insufficient.

After copying a fix into a submission package or remote trial, verify those actual imported files against the checked candidate. A local source SHA, a copied package and a generated CCE file may identify different revisions. Any relevant source/configuration change invalidates the previous candidate's acceptance; retain old results only as explicitly scoped controls. Avoid editing an executing candidate in place.

## Diagnostic probes

Before interpreting a new probe's numbers, state which boundary it tests and how each observed value reaches GM. Initialize every byte it will read: UB is not zero-initialized, and a nonzero value does not prove the current operation wrote it. A smaller kernel can change execution mode, synchronization and ownership; passing one region does not validate an unwritten region.

| Probe change | Required control |
| --- | --- |
| Extract Vector work from a MIX kernel | Check the probe's actual block/subblock mapping. Pure Vector has `get_subblock_num()==1` and `get_subblock_idx()==0`; explicitly assign the original two halves to probe workers or retain MIX execution. Use the [indexing contract](../api/support.md#5-get_subblock_num-depends-on-whether-the-kernel-has-a-cube-section), not the original kernel's launch geometry |
| Retain Cube→Vector or Vector→Cube transport | Keep a complete handover. For a one-shot automatic probe, use `pipeline=PipelineConfig(preload=0)` and producer/consumer stages called from the same `pl.range(1)` driver loop; the [minimal banded GEMM anchor](../../examples/algorithms/gemm_bias_silu_banded/kernel.py) supplies runnable declarations. Follow the [driver-loop contract](../api/crosscore-sync.md#layer-b-still-needs-a-driver-loop-even-for-one-handover), and verify emitted set/wait on the actual writing/reading pipes. Group event metadata requires the pipeline transform to insert synchronization; manual synchronization requires its own proof |
| Add score/state dumps inside tiled or multi-worker execution | Define a unique logical address per record. With a fixed `NB` blocks per item, use `slot = item * NB + j` and rows `sub * R + r`, allocating for all items. A worker-local `tick` collides across workers; `slot=item` is overwritten by later blocks. Pass the logical item/block scalars into the producing stage so delayed data and its labels refer to the same iteration |
| Add a read/store of a shared stage buffer | Dump inside the owning stage, or prove completion and retained storage before a later read. Recheck reader pipes and release lifetime: an added MTE3 read can change synchronization requirements; a store outside the pipeline stage may observe reused data |
| Simplify dimensions or input values | Use distinguishable nonzero batches, heads, rows and key blocks for the axes under investigation. Single-head tests cannot expose a missing head offset; zero or uniform controls may hide a broken producer or recurrence. Initialize outputs/dumps with a sentinel, check every expected region and guards, and distinguish intentional padding from missing writes |
| Add a standalone numerical harness | Use the [actual reference/evaluator path](../workflows/authoring.md#reference-and-evaluator), assert values and full write coverage, and enforce the [failure-status contract](#execution-and-evidence-capture) |

First run with one worker and the smallest legal complete slice, then extend the dimensions needed to test ownership or reuse. A partial match can locate a working region, but keep the boundary unresolved until all expected regions pass. If the probe violates these controls, repair it before changing the kernel or attributing the result to unsupported hardware behavior.

## Isolating device faults

- Run traps, illegal hardware combinations and potentially hanging cases in separate processes with an external timeout. Afterwards, launch a trivial kernel in a fresh process to verify recovery.
- Launch failures following the first AICore fault in one process are usually cascades, not dozens of independent defects. See [regression leads](../diagnostics/regressions.md) for selection and known observations.
- Device printf output may not be flushed when synchronization finishes; use a controllable subprocess when capturing it.
- Report invalid logs, timeouts and unexecuted cases explicitly; none counts as verified.

## Numerics and races

- If isolated cases pass but combined cases fail, first inspect synchronization and shared state, especially whether `wait_cross_core` waits on the first consuming pipe. This is a diagnostic priority, not proof of a unique cause.
- Inspect generated IR/CCE for difficult reproductions instead of indefinitely repeating one shape. Increasing transfer volume can expose a race window while keeping addresses and synchronization fixed; identify the changed variable.
- Capture mismatch locations and a concise summary when an assertion fails. A reverse control should show that removing the fix makes the criterion fail.
- For cache/SPR probes, vary call order and process boundaries. Repetition in the same state is not independent evidence.
- Explain whether `atol=0` relies on deterministic semantics or carefully chosen inputs. Post-accumulation HF8 rounding boundaries are unsuitable for general bitwise acceptance; relaxing a tolerance also needs evidence.

## Performance

[Native-task timing](native-timing.md) owns callbacks, task counts, clock meanings, noise checks and same-process AB/BA comparisons. Absolute times from separate sessions provide scale estimates, not sufficient evidence of a change's benefit; model diagnostics also cannot replace hardware duration. Use [optimization](../workflows/optimization.md) to choose experiments and input regimes, and [performance diagnosis](../diagnostics/performance.md) only when finer counters or timelines are needed.

Do not reuse timings from a numerically failing variant as a verified transfer cost or architectural lower bound. A later correct probe of a different path does not validate that earlier timer. For a multi-kernel wrapper, measure every native task in the complete call; a synchronized host average must not be labeled a native-task median.
