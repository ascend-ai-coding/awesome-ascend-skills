# Optimization: measure → choose an experiment → compare

Use this for a kernel that is numerically correct on the workloads being compared. For a new kernel, start with [representative computation-path comparisons](authoring.md#compare-computation-paths-before-expanding-coverage); full-matrix correctness is a final acceptance requirement, not a prerequisite for those comparisons. A baseline-measurement request can finish with a verified metric. An optimization request also needs implementation comparisons. Keep the [authoring acceptance matrix](authoring.md#preflight-decisions) active throughout, and profile representative workloads unless full performance coverage is explicitly requested.

## Measure first

Use [native-task timing](../verification/native-timing.md) for the execution-host helper path, callback contract, task counts, metric definitions and AB/BA calls. Freeze the comparison inputs, shapes, worker counts, metric, baseline and stop condition before searching. Keep the best verified baseline available.

Inspect `per_task` and map execution positions to actual launches. Choose work that matters to the complete call, including initialization, movement, conversions and merges; a locally faster phase can lose once its new stages are included.

For a requested aggregate benchmark score, inspect its actual formula and estimate each workload's contribution from plausible latency reductions. Combine that sensitivity with measured headroom and implementation cost; the worst speedup ratio need not offer the largest score gain. Keep this prioritization subordinate to the user's chosen objective and the unchanged numerical contract.

If the goal is to exceed the supplied baseline, a gain over a slower prototype does not meet it. Validate the final library implementation through the original evaluator and cache protocol, including the selected dispatch on the complete case set. Report an aggregate score and mean speedup separately when both are requested; a nonlinear score is not interchangeable with an arithmetic mean. A mechanism example or isolated stage improvement is supporting evidence, not a replacement for that acceptance.

When a change adds native calls or wrapper work, report the task count and compare both [device-task and complete-call metrics](../verification/native-timing.md#choose-and-name-the-metric) before accepting it. A faster variant relative to its own new prototype does not supersede a stronger earlier implementation; unresolved numerical failures disqualify it regardless of timing.

## Diagnose before changing the implementation

Before selecting the first implementation experiment, collect [task-level pipe counters](../verification/native-timing.md#collect-pipe-counters-for-diagnosis) for the correct baseline. Use source, generated CCE (`kernel.cpp`) and, when needed, transformed IR to inspect work counts, buffer lifetimes and dependencies. Compare per-worker work with actual per-core counters when averages can hide a long tail. Record one observed cost, its source/core/pipe, the proposed change and the expected effect; [performance diagnosis](../diagnostics/performance.md) owns interpretation. Missing collector support is an explicit limitation, not a reason to abandon a valid native candidate or spend the whole budget on setup. Revisit diagnostics after a mapping change or unexplained regression; do not collect every expensive trace for every tuning point.

Development and optimization use code inspection and focused native experiments. Select each experiment for a named uncertainty and a bounded diagnostic budget.

## Budget and candidate identity

When the host exposes `experiment_plan`, call it on the correct baseline before editing. Declare the task's time/device-request budget and reserve device requests for repair and confirmation, in addition to time. The runtime supplies actual elapsed/remaining values each round; analysis length and a feeling of being stuck are not clocks. Use `task_start` for native execution, with `budget_phase=screen`, `fix` or `confirm`; `local` is for CPU-only work. A request counts one managed command, not one kernel launch or necessarily one test case. Outside this host, obtain the equivalent values from the execution controller; unknown budget usage must remain unknown.

Plan the discriminating implementation comparison before spending requests on final coverage. Read the remaining quota for its actual phase: unused protected repair/confirmation requests need not be available for a new candidate. If that admission rule stops exploration, report the budget constraint and the untested direction separately from any hardware hypothesis; do not relabel a new experiment as repair to bypass the reserve.

Bind the entry, wrappers and local imported helpers to each attempt. Explicitly include dynamic imports and non-Python dependencies; installed libraries belong to toolchain identity. Exact byte hashes bind test results. Semantic source comparison identifies rollback and ignores comments and unimported backups, but neither an AST change nor a different directory hash proves a structural implementation. Check the executed path: literal-only changes are configuration tuning; changed ownership, residency, reduction or dependencies are implementation candidates requiring their own correctness and performance evidence.

Use the [source-bound work comparison](../tooling/checks.md#source-bound-contract-and-work-inspection) before spending another device request. Resolve a cheap source contradiction first, screen one discriminating case next, then use reserved requests for the selected candidate's required coverage and matched timing. Do not rerun a full matrix merely because a backup file appeared. Never reuse an old numerical pass after an executable dependency changes. If every candidate loses, retaining the baseline is a valid result; report the negative comparisons and open directions instead of retaining a regression to satisfy a change check.

## Preserve a fast baseline during refactoring

Freeze the working callable, its generated code and the requested measurement protocol. Change interface adaptation, function/stage extraction, synchronization ownership, stage schedule, physical layout and computation mapping separately when they can be isolated. A from-scratch implementation may be the deliverable while the old implementation remains an experimental control.

For `auto_mutex`/pipeline migration, first inspect what is already enabled. Match physical aliases to mutex IDs; an enabled decorator alone is insufficient. Remove a hand-written event only after identifying the generated dependency that replaces it. Check `get_buf/rls_buf`, actual cross-core producer/consumer pipes and slot reuse, including first/last and empty iterations. `preload=0` can isolate generated cross-core synchronization from changed overlap; it does not represent a final pipelined performance result.

If extraction or rewriting loses performance, inspect newly runtime VF flags and [hardware-loop lowering](../api/vf.md#keep-runtime-dispatch-out-of-vf-loops) before blaming the pipeline. A neutral loop/unrolling experiment does not exclude a separate branch cost. Re-test previously losing algorithms after removing that cost; an old ranking can reflect a compiler-shape penalty rather than the algorithm's work or reuse.

## Choose the implementation experiment

| Observation in the dominant work | Next comparison or owner |
| --- | --- |
| Grouped work repeatedly starts from the same worker ID, or many launched workers have little work | [Across-group ownership](../api/gemm.md#distribute-tiles-across-runtime-groups): count every worker's full-call work, compare continuous numbering at fixed geometry, then retune blocks; inspect empty local consumers when ownership changes |
| An indexed copy/packing phase dominates, or payload elements repeat metadata | [SIMT transport mappings](../patterns/simt-payload.md#compare-flat-and-grouped-mappings): compare tuned flat and record/lane controls, including when continuing an already grouped kernel |
| Each channel repeats the same maximum, denominator or nonlinear coefficients | [Normalized contractions](../patterns/normalized-contractions.md): shared computed state and phase-specific Cube/VF/SIMT choices |
| Ordered selection repeatedly scans most retained candidates, or few output owners leave workers idle | [Construction, emission and worker coverage](../patterns/ordered-selection.md#separate-construction-emission-and-worker-coverage): compare the dominant selection phase or input partitioning while retaining validated phases |
| One output has conflicting updates while another has a long serial reduction or newly introduced atomic contention | [Multiple-output ownership](../patterns/multi-output-reductions.md): participants versus final writers, owned partials and full-call cost |
| Repeated tiny DMA/VF calls, spilled state or duplicated input across owners | [Batch independent retained rows](../patterns/vf-row-reduce.md#batch-independent-rows-even-when-each-row-fits) for row-wise work; [transfer blocks and state lifetime](../patterns/vf-outer-reduce.md#choose-transfer-and-register-blocks-separately) for outer-axis reductions |
| Load/store time remains visible after batching and local capacity is available | [Independent IDs and slot depth](../api/memory-allocation.md#independent-buffers-and-slot-depth): separate accidental scheduling coupling, then compare local buffering at fixed transfer width |
| A long row reduces each register to one scalar and serially updates the same state | [Horizontal-reduction frequency](../patterns/vf-row-reduce.md#choose-the-horizontal-reduction-frequency): lane-wise partials, bounded merge intervals and numerical revalidation |
| A busy or regressed VF loop tests first/last, mode or row-validity conditions | Compare [outer dispatch and static VF bodies](../api/vf.md#keep-runtime-dispatch-out-of-vf-loops), inspect native loop/address-generator instructions, then decompose the remaining arithmetic |
| Predicate construction remains expensive in a regular VF loop | [Runtime predicate alternatives](../api/vf.md#repeated-runtime-masks-in-vf-loops), separating bounds, mask construction and selection |
| Expanded VF code with repeated predicates, indices or many live temporaries | Compare the compact `pl.range` loop; expansion is not automatically faster. Preserve logical masks when considering [register-gather wrap](../api/vf.md) |
| Movement appears duplicated or materializes an avoidable intermediate | [Data movement](../api/datamove.md), [GEMM fusion](../patterns/gemm-desugar.md), [fixpipe riders](../api/fixpipe.md) and [CV handovers](../api/crosscore-sync.md) |
| Matrix-operand conversion dominates while Cube is lightly used or requested workers are idle | [Transformed operands](../api/gemm.md#transformed-operands-format-residency-and-repeated-work): repeated producer work, operand reuse and ownership across both output axes |
| Thin-row matrix work streams wide weights or represents one FP32 activation with several Cube products | [Thin-row contractions](../api/gemm.md#thin-row-contractions-compare-arithmetic-and-transport-paths): compare VF arithmetic and AIV-assisted Cube transport separately, with register/resource bounds and the original precision contract |
| Unequal work items or runtime segment lengths dominate the tail | Host-known weighted partition versus [device-side segment/partial work](../patterns/multi-output-reductions.md#partition-unequal-runtime-reductions); production Host must not read back tensor contents to choose a route |
| Task-level costs no longer explain the bottleneck | [Performance diagnosis](../diagnostics/performance.md): block counts, pipe counters, timelines, bandwidth and critical-path experiments |

A worker sweep changes the launch budget, not repeated arithmetic or logical ownership. First name the cost and the comparison that can test it. Source-level work counts locate candidates; they do not prove hardware gains. Pattern documents own the implementation-specific choices; this workflow owns the experiment sequence.

An optimization task needs at least one measured implementation comparison addressing the named cost when the budget permits; changing only worker count or transfer size while leaving the dominant computation unchanged is a configuration result. If the current implementation writes intermediates between dependent row-local phases, first revisit the [state-residency decision](../patterns/vf-row-reduce.md#choose-where-dependent-phases-share-state). Compare at the same worker count before reporting an implementation gain, then tune the accepted implementation's configuration.

## Compare the realized configurations

Before interpreting a parameter sweep, [verify the compiled variant](../verification/native-timing.md#verify-the-compiled-variant) and inspect the selected specialization: physical/valid tile sizes, slot depths, work-item counts, reduction panels, workers and generated compute calls. A capacity fallback can silently shrink another tile when a split size grows. Fewer inner splits per item may then be canceled by more items; changing only one source constant has not isolated one hardware cost.

Recompute work counts from executable expressions and the current input metadata. Inline comments, variant names and old summaries can disagree with the actual constants or loop bounds. For a wave-balance hypothesis, show the before/after tile count, worker count, `ceil(tiles / workers)` and the busiest worker's tile extents; a short host calculation is sufficient for known scalars. If that calculation contradicts the hypothesis, correct the prediction before spending a device request. A measured improvement remains evidence for the candidate, but cannot validate a work-count premise that was already false. Update affected geometry and slot comments with the retained implementation before freezing its final validation snapshot.

Resolve the Host dispatch predicate and selected specialization for every measured workload before calling its path unchanged. Check untested workloads against the same predicates: passing one shape does not validate all shapes using that branch, and an unmeasured shape may select a changed branch even when its broad label is “decode” or “prefill”. Use that map to choose confirmation and boundary cases; report the remaining coverage explicitly.

Count work over the complete workload and the busiest worker, including padded/boundary computation, handovers and repeated transfers. Where legal, hold the logical geometry fixed and adjust the relevant staging slots within their capacity and lifetime contract to isolate the compute-step comparison. Otherwise describe the result as a coupled configuration change. A full current allocation is a constraint on that layout, not proof that another tiling or state lifetime is infeasible.

## Data-dependent work

Equal shapes do not imply equal work for runtime masks, sparse routes, bounded gathers or interpolation. Uniform inputs over the legal domain may mostly skip costly loads. Separate execution density from locality using a small matrix: mostly active, mostly skipped, a boundary mixture, and repeated versus dispersed valid addresses where relevant.

Give performance inputs their own workload specification. Correctness fixtures may deliberately repeat a few addresses or use constant scores; reusing that generator can benchmark a tiny cached working set inside a large allocation. For indexed payloads, report the accessed-row count and approximate payload working-set bytes, and include a dispersed case spanning the legal table as well as high reuse. Keep segment-length distribution, score/value distribution and address locality as separate controls: changing a segment-distribution label must not silently change the scores or index stream too.

Keep each pair's exact inputs and worker count fixed. Tests may compute a valid fraction or workload count to explain results; production Host still handles metadata only. Correctness covers all contractual branches at the original tolerance.

For a changed SIMT decomposition, use the [launch contract](../patterns/simt-first.md#launch-variants) and [payload mapping comparison](../patterns/simt-payload.md#compare-flat-and-grouped-mappings) on these workload regimes. Retain a simple path or explain measured regressions when one configuration is not uniformly better.

## Change discipline

1. Change one coherent cause, pass correctness, measure the complete call, then keep or revert. Matching ownership, workspace and launch edits can belong to the same implementation choice.
2. Continue successive experiments when the requested goal and budget call for them. “One lever per step” does not mean one experiment per task. When local edits remain neutral, revisit the cost decomposition and implementation alternatives before more configuration tweaks. A successful thread-size comparison does not resolve a remaining serial-reduction or contention hypothesis.
3. Keep the current configuration fixed while comparing a new logical work unit; retune the accepted mapping's configuration in a separate step. Preserve both comparisons instead of attributing all gains to one edit.
4. Use the [paired timing procedure](../verification/native-timing.md#compare-paired-implementations) against the best verified baseline. Reject numerical regressions; retain performance tradeoffs only with measurements for the affected regimes, not an assertion that other shapes should improve.
5. Put a reusable conclusion in its pattern/API owner and example-specific evidence with the example. Keep transient results in the reply or ignored scratch; no per-step ledgers, JSON reports or manifests unless requested.

Scope a losing or neutral result to its tested implementation, realized geometry, inputs and toolchain. A slow per-row branch does not rule out a block-specialized or vector-predicate path. After removing a dominant cost, reconsider earlier candidates that now address the exposed bottleneck; an earlier negative result is not an unconditional prohibition on that direction.

## Screen candidates before confirmation

Separate screening from confirmation. Screen one representative workload and worker count with a small declared AB/BA sample budget; helper defaults are not required at every point, and close results need more samples. Freeze the validated baseline as a callable in the same driver. For transport, launch or batching changes that preserve arithmetic and indexing, first compare candidate outputs directly against that baseline, using bitwise equality where bits must be preserved. Add focused controls for changed boundaries, ownership and slot reuse. This differential screen does not replace the independent reference; changes in arithmetic order or dtype need the original numerical checker and semantic controls. Run the full supplied evaluator and acceptance matrix on the selected candidate, with profiling disabled when only numerical coverage is required. Confirm performance on the selected representatives. Rebuilding the full reference and collecting pipe counters at every configuration point can consume the search budget without resolving another uncertainty.

Keep one benchmark driver parameterized by the actual baseline/candidate and configuration; preserve the strongest correct implementation and make the retained entry reproduce the reported comparison. Follow the [stateful-callback contract](../verification/native-timing.md#in-place-and-stateful-callbacks) for mutating kernels. Resolve conflicting native/wall rankings through [paired timing](../verification/native-timing.md#compare-paired-implementations) before turning separate-session medians into a stopping rule or dispatch.

## Check the neighbors of a winning configuration

Treat an initial heuristic value or bounded sweep as an open search region. Compare the current winner with its legal neighbors at fixed inputs, worker count and metric. When an edge candidate wins, test the next legal neighbor in that direction while budget remains; the new winner needs its own neighborhood check. Close each explored direction with a measured losing neighbor or actual resource/capability bound, or report it open when the budget expires. State the tested range and untested directions; a local winner is not a global optimum.

For coupled parameters, change one coordinate around the winner, then recheck the other after an accepted change. In particular, holding transfer width fixed isolates a slot-depth comparison; afterward, recheck width under the new total capacity. Changes in ownership or buffering can invalidate an earlier configuration choice. Reserve time for this search and final validation rather than stopping at the first successful implementation experiment.

## Exit criteria

For measurement only, report the validated metric. For optimization, check the final configuration against the [neighborhood stop conditions](#check-the-neighbors-of-a-winning-configuration) and revisit pending implementation comparisons justified by the diagnosed cost. A completed numerical matrix or worker sweep does not close an unchecked direction. Continue feasible experiments while budget remains; at expiry, report the search as incomplete wherever evidence is missing. A necessary change in tiling, ownership or compute paradigm returns to [authoring](authoring.md) within the same task.

Report source/toolchain identity, shapes, workers, data regimes, metric, paired results against the best verified baseline, and the stopping evidence or unresolved directions. A failed configuration sweep or an estimated byte count does not establish a DSL or hardware ceiling. Report an aggregate score only from the complete prescribed aggregation; a subset-based estimate cannot establish that the target was met or missed. If a requested score is unmeasured or unmet, identify the tested configuration family and open structural comparisons; retain an unverified bottleneck as a hypothesis. Keep numerical acceptance, measured improvements and remaining headroom distinct. Tool/link checks do not establish hardware speed or agent effectiveness.
