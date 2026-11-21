# Device runs and measurement

Before remote operations, read the target project’s external environment and execution configuration. It must identify machines, paths, device selection and environment scripts. Keep these machine-specific settings outside the skill. NPU resources may be shared; this skill does not require device locks or exclusive allocation for Host work, device validation or timing.

After sourcing an SDK environment, verify that the resolved library and profiler paths exist and that the actual interpreter can import its runtime. A copied setup script can retain an unavailable installation prefix. Correct that prefix in trial-local environment configuration before diagnosing a kernel failure; preserve the selected runtime and matching framework source.

## Execution and evidence capture

1. Pin source and installation identity; confirm CANN/compiler versions and [actual imports](#verify-imports-in-the-execution-process). Python-only changes can use the documented single-file synchronization flow. For C++ changes, confirm the new library is installed and actually loaded: a successful build does not prove that it took effect.
2. Select available resources through the environment configuration; independent jobs may share a device. Choose concurrency within actual Host/device memory capacity and evaluator constraints, and do not terminate others' jobs. For timing, use [device identity and shared occupancy](#device-identity-and-shared-occupancy) to assess comparable load, including after an intermittent capture failure or unexplained latency change. Host preparation and compilation need no device reservation.
3. Capture full stdout/stderr and retain the real exit code and completion marker before extracting a summary. Numerical harnesses must assert or exit nonzero on mismatches and caught exceptions; printing `FAIL` or a failure list still leaves an ordinary script's exit status at 0. Do not use the exit status of `pytest | tail` as a test result or interpret a missing summary as absence of failures.
4. Estimate the number of compilation variants first. Start with a probe answering one question; batch related targets to amortize initialization, then expand coverage experiments.

Use [check.py run](../tooling/checks.md#run-a-command-after-source-checks) in the execution environment with the configured timeout. Include the submitted kernel, host wrapper and test in `--source`; run the actual assertion-based test or evaluator as the child command. It preserves child output/status and reports source SHA256, and rejects a successful command whose selected sources changed during the run. It cannot make a print-only harness assert, infer numerical success, inspect another host's imports, or recover a status already swallowed by a child shell pipeline.

Calibrate the test's failure path once with a deliberately corrupted result at the original tolerance: it must fail before any timing callback. Assert public output shape/dtype, finiteness where required, guards and the actual evaluator's numerical criterion. A reported error norm or `ok: false` followed by process exit 0 is an invalid acceptance harness. Use the reference's required precision and output dtype for each branch, including BF16; keep correctness and timing behind the same assertions.

### Device identity and shared occupancy

Keep the runtime's logical device index separate from the management CLI's device identifier. Remapping a physical card to `npu:0` does not show that `npu-smi -i 0` queries that card. Use the physical selection in the external environment and the inventory's mapping; retain the actual query command and returned device ID. Identify the measuring process and its children by PID before labeling a listed context as external.

A foreign process holding a device context does not by itself prove overlapping kernels or an active collector, and one zero-utilization snapshot does not prove an idle measurement window. Correlate process ownership, commands and observed activity with the capture interval when interpreting contention. Shared-load measurements can support a comparison under that stated workload when paired AB/BA samples have comparable load. Report variability and limit the conclusion to the measured conditions; if changing load confounds the ranking, keep it unresolved and repeat only the affected comparison under comparable conditions. Coordinate a capture window when an observed collector conflict or unstable load requires it, without reserving the NPU throughout preparation and compilation.

### Track remote-job completion

A controller timeout or lost SSH/tool connection does not prove that a remote job stopped. Use the environment's supported managed-job or detached execution mechanism when needed, retain its job/process identity and logs, and query or reattach before submitting a replacement. Distinguish queued, running, completed and unknown states; a missing local waiter is not remote completion. Machine-specific process-control commands and recovery policies stay in the external execution configuration.

Stop requests must use a recorded job identity and verified process descendants. A global process-name or shared-workspace-prefix match does not establish ownership; broad `pkill`, `killall` or `ps | grep | xargs kill` can terminate other sessions. In supervised runs, keep process control in the execution controller and let kernel sessions request status or a scoped stop.

## Integrate a supplied evaluator

Inspect how the evaluator discovers the public callable and loads the submission before invoking its build/install path. A source-directory option can build a wheel and install it into the current interpreter; it does not imply source-only loading. For a read-only shared installation, verify the evaluator's supported source-loading mode or use a trial-owned installation environment, preserving the pinned framework dependencies. Source loading still uses the original golden, checker and performance protocol. Inspect the active compute guard before rejecting an interface adapter: a compute block mode can still permit same-device layout/dtype transport. Keep the mathematics native and include permitted preparation in the original timing.

Inspect internal device-worker defaults as well as the parent command. Include concurrent workers in Host/device memory estimates, and isolate their compilation roots and output paths. Set the evaluator's report, child-capture and temporary directories to trial-owned locations through its supported configuration before launching. A top-level output option need not control child reports or `TMPDIR`. Verify the [actual imports](#verify-imports-in-the-execution-process) in a fresh process matching the evaluator's execution context before launching the device evaluator.

For a score target, resolve the executing evaluator's task-set version, active score weights/formula, detected physical hardware, selected baseline metadata and complete-call metric. Framework/package version and task-set version may differ; an example report or a default-platform metadata file does not select the current score configuration. Prefer the evaluator's resolved configuration and original report; inspect its implementation only when that resolution is unclear. Carry the resolved choices into score estimates and final aggregation. Fewer native launches or a sum-of-kernel-times metric alone does not prove that a fused path is faster.

When comparing a bundled example with a repository operator, prefer an explicit public-callable override in the existing harness. Keep the original input generator, independent golden and comparator; first run the repository entry, then the candidate on their common supported domain. Private workspace, tiling or cache tests may remain specific to the repository implementation: select public-contract tests explicitly instead of treating their pass as candidate acceptance. Isolate same-named example modules, retain package-relative imports, report the loaded entry's path/hash and fail on loading errors rather than silently selecting the default. Include the loader and every executable local dependency in the checked source set. Metadata checks or candidate collection alone do not establish device execution.

For a score target, complete the [capture preflight](native-timing.md#verify-capture-before-expanding-performance-coverage) as part of this first integration probe. A correctness-only invocation does not establish the evaluator's performance path.

## Verify imports in the execution process

Probes must not insert a source tree into `sys.path`. Preserve existing `PYTHONPATH` entries in the parent and child processes: sourcing a pinned environment does not protect it from a later path replacement. Verify imports after the final path edits.

Standalone example drivers need the original [device/runtime fixture setup](../tooling/examples.md#admission-rules). Perform that initialization in the executing child before allocating NPU tensors; importing the runtime in a separate environment probe does not register the child's device backend.

For a locally overlaid evaluation package, copy its package initializers and assert that the exact public attribute discovered by the evaluator is callable. Importing a submodule can succeed while exporting a module instead of its function. Verify that both the exported entry and implementation module resolve to the intended trial root in the evaluator's import context. Prepending a directory containing only a subpackage can still select an installed regular package. Confirm the actual import before interpreting correctness or timing; hashing an unimported candidate file does not identify the code that ran.

For configurable wrappers, bind the selected algorithm and [realized configuration](../workflows/optimization.md#compare-the-realized-configurations) as well as the file path. A debug script and evaluator importing the same package can execute different paths because their environment flags or metadata differ. Confirm that the representative actually enters the changed implementation before using it to diagnose that implementation or expanding its coverage.

A trial-local public entry can bind directly to the candidate under the original signature and mathematical contract, retaining the control for the remaining domain. This makes the tested path part of the frozen source rather than relying on remembered environment overrides. Merge the measured selection into the final wrapper afterward and verify that actual entry again.

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

The negative control must reach the condition it is meant to expose, and its expected result must be independent of the implementation switch. Assert the relevant geometry (such as an actual ring wrap or misaligned boundary), and confirm a source perturbation changed the intended bytes before interpreting the verdict. A no-op edit or a checker that changes its expectation with the same switch is not a failure-path test. Count the comparisons actually executed, not only test functions that returned successfully.

## Isolating device faults

- Run traps, illegal hardware combinations and potentially hanging cases in separate processes with an external timeout. Afterwards, launch a trivial kernel in a fresh process to verify recovery.
- Launch failures following the first AICore fault in one process are usually cascades, not dozens of independent defects. See [regression leads](../diagnostics/regressions.md) for selection and known observations.
- Device printf output may not be flushed when synchronization finishes; use a controllable subprocess when capturing it.
- Report invalid logs, timeouts and unexecuted cases explicitly; none counts as verified.

### Localize a stall from the executed binary

When a fault dump supplies stuck PCs, preserve it and the actual device binary before removing synchronization edges. A watchdog timeout alone does not distinguish an event wait, local pipe/buffer acquisition, an invalid access or another stalled instruction. A mapped PC narrows the investigation; an asynchronous fault's reported location need not be its root cause.

1. Identify the loaded device image and its source, specialization and toolchain. Some builds embed it in a shared-library section such as `.aicore_binary`; inspect that build's section table before extracting it. [llvm-objcopy](https://llvm.org/docs/CommandGuide/llvm-objcopy.html) provides section extraction, but the container and section name are runtime-specific.
2. Establish the dump's address convention and load base independently. For an absolute PC, derive the object address from the verified load base and check an independently reported entry address against its symbol offset. Do not subtract a base from an already-relative PC or apply another binary's entry offsets.
3. With matching debug line information, [llvm-symbolizer](https://llvm.org/docs/CommandGuide/llvm-symbolizer.html) or `llvm-dwarfdump --lookup` can map an object address to generated-source lines even when an ISA disassembler is unavailable. If a rebuild adds debug information, require identical device text bytes, entry offsets and corresponding generated source before using it to map the failed executable. Otherwise the new line table describes a different program.
4. Inspect the mapped operation and its producers, consumers and resource ownership, then choose a discriminating probe under the existing isolation rules. Do not infer that every hang is a flag mismatch, that a source-level buffer-acquisition call proves a unique hardware cause, or that a smaller shape produces the same binary without checking its identity. Missing dump/debug information leaves this localization unavailable; continue with generated-code and controlled dataflow checks.

## Numerics and races

- If isolated cases pass but combined cases fail, first inspect synchronization and shared state, especially whether `wait_cross_core` waits on the first consuming pipe. This is a diagnostic priority, not proof of a unique cause.
- Inspect generated IR/CCE for difficult reproductions instead of indefinitely repeating one shape. Increasing transfer volume can expose a race window while keeping addresses and synchronization fixed; identify the changed variable.
- Capture mismatch locations and a concise summary when an assertion fails. A reverse control should show that removing the fix makes the criterion fail.
- For cache/SPR probes, vary call order and process boundaries. Repetition in the same state is not independent evidence.
- Explain whether `atol=0` relies on deterministic semantics or carefully chosen inputs. Post-accumulation HF8 rounding boundaries are unsuitable for general bitwise acceptance; relaxing a tolerance also needs evidence.

## Performance

[Native-task timing](native-timing.md) owns callbacks, task counts, clock meanings, noise checks and same-process AB/BA comparisons. Absolute times from separate sessions provide scale estimates, not sufficient evidence of a change's benefit; model diagnostics also cannot replace hardware duration. Use [optimization](../workflows/optimization.md) to choose experiments and input regimes, and [performance diagnosis](../diagnostics/performance.md) only when finer counters or timelines are needed.

Do not reuse timings from a numerically failing variant as a verified transfer cost or architectural lower bound. A later correct probe of a different path does not validate that earlier timer. For a multi-kernel wrapper, measure every native task in the complete call; a synchronized host average must not be labeled a native-task median.
