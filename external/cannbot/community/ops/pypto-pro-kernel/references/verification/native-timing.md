# Native-task timing

Use this after numerical validation when the task asks for device time or an implementation comparison. The bundled [profile_kernel.py](../../scripts/profile_kernel.py) owns warmup, capture validation, task grouping and temporary-capture cleanup. For a configuration shortlist, use [batched configuration screening](#screen-configurations-in-two-captures) to share export cost across adjacent paired comparisons. Device selection, process timeouts and input setup belong to the caller; shared NPUs follow the [load-comparability guidance](board-runs.md#device-identity-and-shared-occupancy).

Validation is scoped to the exact shape, dtype, runtime workload and launch configuration being timed. An early [representative computation-path comparison](../workflows/authoring.md#compare-computation-paths-before-expanding-coverage) can precede full-matrix correctness. Final acceptance still needs the complete numerical matrix; it does not require profiling every case. Keep representative timing coverage explicit and do not infer times for unmeasured cases.

Before collecting samples, define the [performance workload and locality regimes](../workflows/optimization.md#data-dependent-work). A passing correctness generator does not establish a representative timing workload; name segment distribution, value distribution and address reuse independently of shape.

## Choose and name the metric

Choose the final performance protocol before the first complete timed case. Follow the user's supplied evaluator and measurement requirements, then applicable project performance rules. When none are specified, use CANN official `msprof`, with L2 clearing recommended. Keep that protocol through optimization and final acceptance. Disabling or replacing artifact organization does not change the selected collector, workload, cache policy or acceptance requirements. Performance validation does not require running the default artifact-organization stage first.

The bundled helper measures profiler device-task duration for diagnosis and paired experiments. Its results establish final performance only when the selected protocol uses the same collector, workload and cache policy. Keep one source throughout a comparison; these metrics are not interchangeable:

| Metric | Meaning |
| --- | --- |
| Profiler `kernel_details.csv` → `Duration(us)` | Native task duration; the helper sums all tasks belonging to each complete call |
| msprof application → `OpBasicInfo.csv` → `Task Duration` | A different device clock with potentially different per-task overhead |
| msprof operator replay | Useful for per-core decomposition; replay overhead can change task duration |
| `torch.npu.Event` interval | Stream interval, potentially including submission gaps; first-use measurements may include compilation delays |
| End-to-end wall time | Host dispatch, execution and synchronization; label explicitly |

Historical cross-collector comparisons are scoped to their kernels/toolchains, not conversion factors. Counter and replay details belong to [performance diagnosis](../diagnostics/performance.md).

For a supplied benchmark baseline, establish its cache policy, warmup, input distribution, task aggregation and included wrapper work before quoting a speedup. Repeated calls on resident inputs from this helper do not reproduce an evaluator that clears L2 before each measured call, even when both read `Duration(us)`. Use the supplied evaluator for the final baseline comparison; retain helper timings as labeled diagnostic or paired-variant measurements. If the original protocol is unavailable, report the two measurements separately without claiming a baseline speedup.

## External collectors

When the task needs per-core samples or the metric groups not exposed here, collect through
[performance collection](../tooling/performance-collection.md). Those captures are diagnostic:
`msprof` application timing, operator replay and `Duration(us)` are different clocks, so they
do not replace the paired comparison below. Per-core samples from `aicore.db` are process-level
and derived from an inferred frequency; label them as such.

## Verify capture before expanding performance coverage

After one representative passes numerically, exercise the requested collector, export, report parsing and scoring path on that case. For a score target, do this before scheduling any full matrix, including a numerical-only matrix. Choose a small case that exercises the newly mapped axes; a singleton axis can hide a layout or stride error. Use the observed compilation, reference and export cost to reserve time for the required complete score. When the supplied evaluator checks accuracy and performance together, that combined run can provide both final matrices; a separate full numerical pass needs a concrete diagnostic reason.

Require nonempty native-task data with the expected units and complete-call aggregation. Check the actual per-case candidate duration and required baseline/hardware anchors separately, then confirm a populated performance contribution. A combined missing-score warning does not identify which input is absent; inspect the raw fields and export status before diagnosing missing metadata or a slow kernel.

For collectors that invoke external exporters, verify executable lookup in the actual child environment. For example, a PATH-based `msprof` lookup can fail even when `ASCEND_HOME_PATH` identifies a complete CANN installation. Check `shutil.which("msprof")`, the resolved executable and the installed collector's lookup code; a dangling convenience symlink is not the executable. If the selected SDK provides the tool under `tools/profiler/bin/`, prepend that verified directory for the trial process while preserving the pinned library paths. Keep machine-specific paths in the external environment configuration. A successful help command establishes discovery, not a successful capture.

If the first capture or export fails, repair that path before expanding the performance matrix; focused numerical diagnostics may continue separately. Retain the requested metric and benchmark protocol from [Choose and name the metric](#choose-and-name-the-metric). A compiler success or numerical pass does not establish that profiling works.

The helper retains a failed capture in its temporary directory and prints its path while preserving the original exception. Inspect that capture's exporter logs and raw/exported files before another device request: collection, CANN export and the host parser are different failure boundaries. A missing `kernel_details.csv` does not establish that the device recorded no tasks. Use the installed collector's supported offline analysis/export on the retained capture when available; changing the temporary directory or repeating the same launch without new evidence does not diagnose the failing boundary. Keep these artifacts outside tracked source and remove them after retaining the needed evidence. Successful captures are removed automatically.

<!-- kernel-query: {"id":"evaluator-protocol-identity","aliases":["evaluator protocol identity","input pool memory budget","single versus double flush","actual profiler batch"],"when":"Comparing a development driver with a supplied evaluator, changing input pools/flush/capture mode, or planning a memory-heavy validation worker.","missing_facts":["Verify the resolved callable, executed pool and collector mode, per-sample cache boundary, complete-call aggregation and peak Host/device budgets; requested settings are not observations."]} -->
## Preserve evaluator and cache protocol identity

The same wheel does not prove that an explicit Python function and an evaluator's matcher execute the same callable. Resolve registration/dispatch, wrapper attributes, selected specialization and actual native tasks; reject an adapter shadowed by another registration. Keep evaluator/scorer versions, case domain, precision checker, runtime settings, input reuse/rotation and collector protocol bound to the result. Reusing a collector alone does not equate worker, allocator or JIT lifecycles.

When the prescribed protocol flushes cache before each sample, complete that flush and its required synchronization before the target invocation, outside the target duration. Record the actual method and addresses reused/rotated. A diagnostic flush before and after the call is a separate protocol, even if both flush kernels are excluded from timing. It does not inherit an official score interpretation. Filter only verified flush task identities; a broad name substring can erase a real candidate task. Sum all target task durations within each complete invocation, then aggregate invocations as prescribed: the sum of independently computed component medians need not equal the median of complete-call sums.

Observe the collector's actual batch mode/size and input-pool construction. A requested batch of 20, an empty collector-internal object or a successful configuration call does not prove shared capture occurred. Keep missing fields unknown rather than filling them from arguments. Correctness input reuse and timing address rotation can also differ; record both. Same-source or identical-device-code evidence may avoid a redundant compile, but cannot supply a numerical/performance result that was never measured for that artifact/protocol.

Budget peak memory per worker before expanding cases: live inputs, reference intermediates/outputs, native-reference copies, mutation replay, actual input-pool multiplicity, flush buffers, compilation and retained captures. Include simultaneous workers and the enforced cgroup/process limit rather than only host-wide free RAM; calculate Host and device budgets separately. Sum the actually overlapping lifetimes, release case-local data when possible, and distinguish unavailable capture from numerical failure. A large input allocation alone is not the full budget, while summing RSS across forked workers can double-count shared pages. Numerical output error and validation wall time remain different metrics.

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

Defaults are three warmups, twenty measured calls and one task per call. These are helper defaults; the workflow's [sampling policy](../workflows/optimization.md#screen-candidates-before-confirmation) governs sample counts, evidence reuse and justified retries. Set `tasks_per_call` to the known native task count for multi-kernel callables. Do not equate a Python function or logical stage with one profiler task: a kernel can emit several tasks, while internal stages of one task are not separated by this summary.

- The headline distribution comes from the sum of task durations **within each call**, excluding Host dispatch and task gaps.
- `per_task` gives one-based execution positions and median/min/max durations. Map positions to the actual launch sequence to locate the dominant work. Adding these medians does not give the median of complete-call sums.
- Missing/extra tasks, unknown CSV units, unavailable profiler capabilities or coincident start timestamps fail explicitly. Do not substitute an Event measurement for a failed native capture.
- The private profiler configuration is capability-checked. Capture/CSV behavior was verified with torch/torch_npu 2.10.0, CANN 9.2.0 and Ascend950PR; other versions need the same successful capability and capture checks.

## Collect pipe counters for diagnosis

The `measure_pipes` call above already supplies the initial timing baseline and counters; no preceding `measure_native` capture is required. It adds `pipe_counters`, grouped by one-based native task position even when symbols are identical. Each entry includes the exported name/type/core family/block count and counters with their original CSV names/units, median, valid sample count and missing sample count. Missing values remain `None`; `available=False` means no usable counters in that task, not zero utilization. Ignore exported zeros for an inactive AIC/AIV family.

This explicitly requests `AiCMetrics.PipeUtilization` and checks that capability. Its `pipe_scope="task_aggregate"` does not expose individual cores. For uneven work or an unexplained tail, take the exact symbol from `identity["Name"]` into a per-core `msprof op` capture using [performance collection](../tooling/performance-collection.md). Ratios from different pipes overlap and must not be summed; preserve the collector's clock and units. Counter interpretation does not replace the paired comparison below.

## Collect a memory metric group

Use `measure_metrics` for one explicit group: `Memory`, `MemoryUB`, `L2Cache`, `ResourceConflictRatio` or `PipeUtilization`. It uses the same warmup, strict task-count checks and native-task clock as `measure_pipes`:

```python
stats = helper["measure_metrics"](
    lambda: run(x, out, workers=workers),
    metric="Memory",
    tasks_per_call=TASKS_PER_CALL,
)
```

The result labels the requested `counter_group` and `counter_scope="task_aggregate"`; `counters` retains the exported names/units, valid zeros and missing sample counts by native task position. No usable counter values leaves that position's `available=False`, while task timing can remain valid. Unsupported groups/counts fail before runtime access; unavailable installed configuration capabilities fail without substituting another group. A successful configuration still needs a complete capture with relevant fields. This entry does not enable the separate `l2_cache` export switch or collect several groups in one capture. Interpret scope and cache evidence through [SIMT memory diagnosis](../diagnostics/performance.md#simt-memory-paths-and-cache-evidence).

## Compare paired implementations

Validate both callbacks first. Use equivalent inputs, consistent allocations and the same shape, worker count and runtime data distribution. Each callback must fully define its outputs; mutating callbacks follow the [stateful contract](#in-place-and-stateful-callbacks) below. Run the complete paired comparison in one process under comparable observed load, including when the NPU is shared; do not infer an implementation gain from separate-session absolute timings.

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

Use existing valid noise/capture calibration, or a focused control when the measurement path is uncertain and the sampling policy permits it. A control can test whether added dependency-preserving work changes measured duration. Inputs/allocations and shared-board load can hide small effects; near ties do not automatically require more samples. Batched launches plus one synchronization can reduce per-launch noise in wall/stream studies, but remain a separately labeled metric and cannot replace native-task evidence.

For synchronized complete-call latency, use `helper["compare_wall"](baseline, candidate, warmup=3, repeats=20)`. It runs both AB and BA orders in one process, synchronizes before the timer and after each callback, and returns the same paired `orders` summaries under `metric="synchronized_wall_us"`. Warm both variants before collection; input preparation stays outside callbacks. This helper uses no profiler and does not establish device-task time. Its callback count is also `warmup + 2*repeats` per side; allocate additional states if collecting both metrics.

A wall result collected in a separate process/session is a diagnostic lead, not a paired implementation comparison. If that unpaired result would change candidate selection or dispatch, obtain a matched comparison when the task protocol allows it; otherwise report the unresolved tradeoff. Valid paired native and wall rankings can differ: do not repeat them to force agreement or turn order dependence into a dispatch rule. Keep the agreed target metric fixed through the search.

### In-place and stateful callbacks

For a single-update comparison, prepare independent state pools with identical initial values for each side and each invocation; advance a Host index in the callback. Allocate and initialize outside capture, include warmups and validation calls in the pool size, and check updated outputs afterward. `compare_native` invokes each callback `warmup + 2*repeats` times; any separate validation or wall measurements need additional states. Resetting tensors inside the callback adds native work to the captured metric, while reusing mutated tensors changes the next invocation's inputs.

A matched evolving trajectory is another workload: use separate identical starting states, the same parameter sequence and call counts on both sides, account for warmups, and verify the resulting states. Describe that trajectory and its value/address reuse instead of calling it a repeated fixed-input measurement. Include nonzero representative updates as well as zero/special cases; a shape alone does not specify the workload. [Reference and evaluator calibration](../workflows/authoring.md#reference-and-evaluator) owns per-call correctness and the runnable state-update example.

Choose active, skipped, boundary and locality regimes through [data-dependent work](../workflows/optimization.md#data-dependent-work). Return the source hash, toolchain identity, shapes/dtypes, workers, metric source and measured distribution in the result. Failed captures remain at the reported temporary path for diagnosis; do not add profiler dumps or experiment ledgers to the repository.

## Screen configurations in two captures

For a fixed implementation family with the same native task count per call, `compare_configurations` screens a named shortlist with one AB capture and one BA capture. This shares collector/export overhead across configurations. Validate every callback first against the original numerical contract, bind its label to the exact parameter tuple and source, and prepare all inputs/outputs before calling:

```python
calls = {label: (lambda config=config: run_config(x, out, **config))
         for label, config in shortlist.items()}
for call in calls.values():
    call()
    validate_output(out)
screen = helper["compare_configurations"](
    baseline_call, calls, warmup=3, repeats=5, tasks_per_call=TASKS_PER_CALL,
)
for label, result in screen["candidates"].items():
    print(label, result["orders"])
```

Each repetition pairs a candidate with its own adjacent baseline call. AB follows the dictionary's candidate order; BA reverses it and invokes each candidate before its baseline. Results retain the same complete-call sums, per-pair ratios and win fractions as `compare_native`. Counts, timestamps and units receive the same strict capture checks. Missing tasks or a failed order raise an error; no partial ranking is returned. Configurations with different task counts use separate `compare_native` calls with explicit counts.

Callbacks use equivalent immutable inputs and overwrite their outputs. With `N` candidates, the baseline executes `warmup + 2*N*repeats` times, while each candidate executes `warmup + 2*repeats` times. Candidate creation, compilation warmup, numerical checking and result serialization stay outside capture. Keep the shortlist bounded by the caller's execution timeout, including compile time; save completed batches in ignored scratch and resume missing batches after interruption.

This screens one declared workload and cache protocol. Reverse order reduces fixed ordering bias, without eliminating drift or candidate-set interference. Apply the [sampling policy](../workflows/optimization.md#screen-candidates-before-confirmation) when deciding whether ordinary [paired comparisons](#compare-paired-implementations) are needed: reuse compatible valid evidence meeting acceptance requirements, and validate a changed public selector or additional regimes from the [configuration procedure](../workflows/optimization.md#choose-configuration-comparisons). A close ranking alone does not mandate recapture. Do not turn separately collected medians into paired evidence.
