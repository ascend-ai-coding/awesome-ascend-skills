# Native-task timing

Use this after numerical validation when the task asks for device time or an implementation comparison. The bundled [profile_kernel.py](../../scripts/profile_kernel.py) owns warmup, capture validation, task grouping and temporary-capture cleanup. Device locks, process timeouts and input setup belong to the caller; follow [board runs](board-runs.md).

Validation is scoped to the exact shape, dtype, runtime workload and launch configuration being timed. An early [representative computation-path comparison](../workflows/authoring.md#compare-computation-paths-before-expanding-coverage) can precede full-matrix correctness. Final acceptance still needs the complete numerical matrix; it does not require profiling every case. Keep representative timing coverage explicit and do not infer times for unmeasured cases.

Before collecting samples, define the [performance workload and locality regimes](../workflows/optimization.md#data-dependent-work). A passing correctness generator does not establish a representative timing workload; name segment distribution, value distribution and address reuse independently of shape.

## Choose and name the metric

The workspace's default speed claim uses profiler device-task duration. Keep one source throughout a comparison; these metrics are not interchangeable:

| Metric | Meaning |
| --- | --- |
| Profiler `kernel_details.csv` → `Duration(us)` | Native task duration; the helper sums all tasks belonging to each complete call |
| msprof application → `OpBasicInfo.csv` → `Task Duration` | A different device clock with potentially different per-task overhead |
| msprof operator replay | Useful for per-core decomposition; replay overhead can change task duration |
| `torch.npu.Event` interval | Stream interval, potentially including submission gaps; first-use measurements may include compilation delays |
| End-to-end wall time | Host dispatch, execution and synchronization; label explicitly |

Historical cross-collector comparisons are scoped to their kernels/toolchains, not conversion factors. Counter and replay details belong to [performance diagnosis](../diagnostics/performance.md).

For a supplied benchmark baseline, establish its cache policy, warmup, input distribution, task aggregation and included wrapper work before quoting a speedup. Repeated calls on resident inputs from this helper do not reproduce an evaluator that clears L2 before each measured call, even when both read `Duration(us)`. Use the supplied evaluator for the final baseline comparison; retain helper timings as labeled diagnostic or paired-variant measurements. If the original protocol is unavailable, report the two measurements separately without claiming a baseline speedup.

## Verify capture before expanding performance coverage

After one representative passes numerically, exercise the requested collector, export, report parsing and scoring path on that case. For a score target, do this before scheduling any full matrix, including a numerical-only matrix. Choose a small case that exercises the newly mapped axes; a singleton axis can hide a layout or stride error. Use the observed compilation, reference and export cost to reserve time for the required complete score. When the supplied evaluator checks accuracy and performance together, that combined run can provide both final matrices; a separate full numerical pass needs a concrete diagnostic reason.

Require nonempty native-task data with the expected units and complete-call aggregation. Check the actual per-case candidate duration and required baseline/hardware anchors separately, then confirm a populated performance contribution. A combined missing-score warning does not identify which input is absent; inspect the raw fields and export status before diagnosing missing metadata or a slow kernel.

For collectors that invoke external exporters, verify executable lookup in the actual child environment. For example, a PATH-based `msprof` lookup can fail even when `ASCEND_HOME_PATH` identifies a complete CANN installation. Check `shutil.which("msprof")`, the resolved executable and the installed collector's lookup code; a dangling convenience symlink is not the executable. If the selected SDK provides the tool under `tools/profiler/bin/`, prepend that verified directory for the trial process while preserving the pinned library paths. Keep machine-specific paths in the external environment configuration. A successful help command establishes discovery, not a successful capture.

If the first capture or export fails, repair that path before expanding the performance matrix; focused numerical diagnostics may continue separately. Retain the requested metric and benchmark protocol from [Choose and name the metric](#choose-and-name-the-metric). A compiler success or numerical pass does not establish that profiling works.

## Resolve the helper on the execution host

Check an explicitly supplied absolute path first. A local checkout path or a failed search under an older remote root does not establish that the helper is missing. In a copied skill, `examples/` and `scripts/` share the skill root. Pass the actual `<skill-root>/scripts/profile_kernel.py`; a standalone copied helper is also supported. Verify the copied content hash and report the path used. Keep correctness runnable independently of the optional profiler path.

Define JIT variants in inspectable source files. Final handoffs use [module-level kernels without factories](../workflows/delivery.md#kernel-definitions-and-dispatch). For generated functions that cannot be inspected, use the [source-availability diagnosis](../api/constraints.md#kernels-created-with-exec-cannot-be-parsed-the-parser-needs-real-source).

## Verify the compiled variant

A source hash identifies the candidate text, not the executable that supplied a timing sample. Confirm the selected dtype/shape/configuration and that the intended change appears in the transformed/generated code, or establish that lowering removed it. Resolve artifacts from the executing process's actual working directory and environment, including evaluator child processes; deleting a similarly named directory elsewhere does not refresh that process's kernel.

Distinguish reuse of an already compiled JIT object/specialization from on-disk artifact naming. A shared name or output directory alone does not prove reuse of an old binary. Check the installed cache behavior and bind the launched kernel to its current generated code and compiled library before interpreting unexpectedly identical variants. Use fresh variant objects and, where needed, distinct supported names or isolated artifact roots. Keep candidate edits and compilation outside the timed callback; only invalidate artifacts owned by this experiment.

## Measure one complete call

Allocate/copy inputs, select the logical device and validate the exact shape/dtype/launch specialization before profiling. The callback must contain only the intended native work, in a fixed sequence on one stream; setup, poison fills and reference checks stay outside it.

```python
import runpy

helper = runpy.run_path(helper_path)  # absolute path on the execution host
stats = helper["measure_pipes"](
    lambda: run(x, out, workers=workers),
    tasks_per_call=TASKS_PER_CALL,
)
print(stats)
```

This returns durations and [pipe diagnostics](#collect-pipe-counters-for-diagnosis) in one capture. For a timing-only request or a later configuration sweep, `helper["measure_native"]` has the same call arguments and returns the duration summary. The [optimization workflow](../workflows/optimization.md#diagnose-before-changing-the-implementation) owns when to collect or revisit diagnostics.

Defaults are three warmups, twenty measured calls and one task per call. Set `tasks_per_call` to the known native task count for multi-kernel callables. Do not equate a Python function or logical stage with one profiler task: a kernel can emit several tasks, while internal stages of one task are not separated by this summary.

- The headline distribution comes from the sum of task durations **within each call**, excluding Host dispatch and task gaps.
- `per_task` gives one-based execution positions and median/min/max durations. Map positions to the actual launch sequence to locate the dominant work. Adding these medians does not give the median of complete-call sums.
- Missing/extra tasks, unknown CSV units, unavailable profiler capabilities or coincident start timestamps fail explicitly. Do not substitute an Event measurement for a failed native capture.
- The private profiler configuration is capability-checked. Capture/CSV behavior was verified with torch/torch_npu 2.10.0, CANN 9.2.0 and Ascend950PR; other versions need the same successful capability and capture checks.

## Collect pipe counters for diagnosis

The `measure_pipes` call above already supplies the initial timing baseline and counters; no preceding `measure_native` capture is required. It adds `pipe_counters`, grouped by one-based native task position even when symbols are identical. Each entry includes the exported name/type/core family/block count and counters with their original CSV names/units, median, valid sample count and missing sample count. Missing values remain `None`; `available=False` means no usable counters in that task, not zero utilization. Ignore exported zeros for an inactive AIC/AIV family.

This explicitly requests `AiCMetrics.PipeUtilization` and checks that capability. Its `pipe_scope="task_aggregate"` does not expose individual cores. For uneven work or an unexplained tail, take the exact symbol from `identity["Name"]` into a per-core `msprof op` capture using [performance diagnosis](../diagnostics/performance.md#collect-the-right-profile). Ratios from different pipes overlap and must not be summed; preserve the collector's clock and units. Counter interpretation does not replace the paired comparison below.

## Compare paired implementations

Validate both callbacks first. Use equivalent inputs, consistent allocations and the same shape, worker count and runtime data distribution. Each callback must fully define its outputs; mutating callbacks follow the [stateful contract](#in-place-and-stateful-callbacks) below. Keep the board lock for the whole comparison; do not infer an implementation gain from separate-session absolute timings.

```python
result = helper["compare_native"](
    lambda: baseline(x, out, workers=workers),
    lambda: candidate(x, out, workers=workers),
    tasks_per_call=(BASELINE_TASKS, CANDIDATE_TASKS),
    repeats=20,
)
print(result["orders"])
```

The helper alternates calls in both AB and BA order. Each `median_ratio` is the median of per-pair candidate/baseline ratios, using the complete task sum on each side. It is not a ratio of separately computed medians or a sum of `per_task` medians. Report ranges, pair counts and win fractions; compare against the best verified baseline, not just the latest slower prototype.

Establish the noise floor and verify that added, dependency-preserving work actually changes measured duration. Inputs/allocations and shared-board load can move absolute times enough to hide small effects. For wall/stream interval studies, batched launches plus one synchronization can reduce per-launch noise, but remain a separately labeled metric and cannot replace native-task evidence.

For synchronized complete-call latency, use `helper["compare_wall"](baseline, candidate, warmup=3, repeats=20)`. It runs both AB and BA orders in one process, synchronizes before the timer and after each callback, and returns the same paired `orders` summaries under `metric="synchronized_wall_us"`. Warm both variants before collection; input preparation stays outside callbacks. This helper uses no profiler and does not establish device-task time. Its callback count is also `warmup + 2*repeats` per side; allocate additional states if collecting both metrics.

A wall result collected in a separate process/session is a diagnostic lead, not a paired implementation comparison. When native and wall rankings disagree, preserve the faster candidate and remeasure wall with the paired helper before rejecting it, stopping a winning configuration search, or adding a shape/worker dispatch. Report unresolved order dependence or noise instead of converting it into a tuning rule. Keep the agreed target metric fixed through the search.

### In-place and stateful callbacks

For a single-update comparison, prepare independent state pools with identical initial values for each side and each invocation; advance a Host index in the callback. Allocate and initialize outside capture, include warmups and validation calls in the pool size, and check updated outputs afterward. `compare_native` invokes each callback `warmup + 2*repeats` times; any separate validation or wall measurements need additional states. Resetting tensors inside the callback adds native work to the captured metric, while reusing mutated tensors changes the next invocation's inputs.

A matched evolving trajectory is another workload: use separate identical starting states, the same parameter sequence and call counts on both sides, account for warmups, and verify the resulting states. Describe that trajectory and its value/address reuse instead of calling it a repeated fixed-input measurement. Include nonzero representative updates as well as zero/special cases; a shape alone does not specify the workload. [Reference and evaluator calibration](../workflows/authoring.md#reference-and-evaluator) owns per-call correctness and the runnable state-update example.

Choose active, skipped, boundary and locality regimes through [data-dependent work](../workflows/optimization.md#data-dependent-work). Return the source hash, toolchain identity, shapes/dtypes, workers, metric source and measured distribution in the result. Captures are temporary; do not add profiler dumps or experiment ledgers to the repository.
