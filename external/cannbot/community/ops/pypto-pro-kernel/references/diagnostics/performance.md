# Performance diagnosis

Use this after the first correct [complete-call timing](../verification/native-timing.md) baseline, before choosing an implementation experiment, and again when the bottleneck changes. The [optimization workflow](../workflows/optimization.md) owns that sequence. These techniques were exercised on the linked examples with the toolchain identities in their source. Their historical measurements are not universal limits or current acceptance results.

## Test a mechanism with source-bound evidence

Turn the next experiment into a short chain: **observed cost → source work → predicted change → competing explanation → discriminating comparison**. Keep it in the current working context; no separate history file is needed. The model chooses the hypothesis and implementation. Deterministic tools establish the source facts and execution evidence used by that choice.

| Reasoning step | Required evidence and useful next action |
| --- | --- |
| Locate the cost | Name the measured task, workload, source SHA and metric. Use `check.py explain --source candidate.py --baseline baseline.py --function stage` to inspect the relevant [work sites and declarations](../tooling/checks.md#source-bound-contract-and-work-inspection). Source sites are not dynamic invocation counts. |
| Count what changes | Expand loop repetitions and output ownership at the actual workload. Count padded transfer extents × element bytes × repetitions × owners, separately from useful bytes. Include initialization, format conversion, stores, reloads and merges. If an activation is already loaded outside the output-consumer loop, adding another consumer does not itself halve activation traffic. Leave unresolved bounds or physical traffic unknown. |
| Predict a result | State which task/pipe or byte term should decrease and which new term may offset it. A smaller norm stage can be canceled by buffer clearing or atomic contention; inspect the complete call. An allocation bound constrains the current layout, not every possible residency schedule. |
| Separate explanations | Write one plausible competing cause. Hold transfer width fixed when testing slot depth; otherwise call it a coupled configuration result. Use a shape/rank declaration check before a cache theory, and a layout probe before a synchronization theory when both explain the symptom. |
| Select the next observation | Use existing pipe/timeline collectors or a focused native control when timing alone cannot distinguish the causes. Total elapsed time cannot reveal L2 hit rate or prove serialization. If a collector is unavailable, keep the mechanism unresolved and use a bounded legal implementation comparison. |
| Update the conclusion | A neutral/losing candidate rejects that implementation under the tested controls. Reconcile the observed result with the prediction, then retain, repair, change the hypothesis or stop using actual remaining budget. Do not turn one failed layout or coupled sweep into a framework/hardware ceiling. |

Keep the comparison's scope explicit. For a claimed reduction in GM activation bytes, first compare that operand's producer loads, extents, repetitions and owners in both sources. L0 copies, accumulator capacity and measured speed are separate questions; leave them open if they are not needed to decide the stated byte prediction. Once the local question is resolved, report that result and select the next distinguishing measurement instead of expanding every referenced hardware topic.

## Collect the right profile

The official performance guide (`pypto/docs/zh/guide/programming_guide/pro/debug/performance_optimization.md`) describes available collectors. Confirm the installed capability before relying on its configuration. Keep profiler output in ignored scratch and remove captures after extracting the needed evidence.

| Collector | Useful output | Checks before interpretation |
| --- | --- | --- |
| `torch_npu.profiler` | `kernel_details.csv`: `Duration(us)`, `aic_mac_ratio`, `aic_mte1/2/3_ratio`, `aic_fixpipe_ratio`, `aiv_vec_ratio`, `Block Num` | Warm up and synchronize before capture; verify task/variant identity |
| `msprof op --aic-metrics=PipeUtilization python3 script.py` | Per-block/per-core `PipeUtilization.csv` and `OpBasicInfo.csv`, including active bandwidth | Select the native mangled symbol with `--kernel-name`; otherwise a framework fill may be captured first. Replay timing differs from application timing |
| PipeTimeline | Event ordering, steady-state bubbles, startup/tail dependencies and scalar work | More expensive collection; inspect the relevant work-item window rather than only whole-kernel ratios |

Use the [capture helper](../verification/native-timing.md#collect-pipe-counters-for-diagnosis) for the validated `torch_npu.profiler` configuration, capability checks and returned fields. Its task aggregate counters need the per-core collector above when averages hide a tail.

Two collector traps observed in that environment: trace directory names containing `=` were rejected, and CSV `Name`/`Type` used C++-mangled symbols rather than `pl.jit(name=)`. Factory variants sharing an inner function name can share a symbol. Use separate captures or unambiguous variant names; do not split rows by the requested JIT label. The shared [native-task timer](../verification/native-timing.md) captures complete calls with explicit task counts.

## Read work distribution before pipe ratios

Check `Block Num`, launch group semantics and output ownership first. Replicated work and same-address writes can make memory pipes look dominant while no useful resource is saturated. The [multicore GEMM example](../../examples/algorithms/gemm_bias_silu_multicore/) demonstrates disjoint ownership; the [cross-core contract](../api/crosscore-sync.md) owns mix-kernel block semantics.

Read two time ratios separately:

- **Duty:** pipe busy time divided by the whole-kernel span. Low duty may indicate waiting on other work or startup/tail cost.
- **Occupancy within its active window:** busy time divided by `last_active - first_active`. Near-continuous activity means additional buffering cannot remove nonexistent internal gaps; investigate the amount of critical work, contention and dependencies instead.

Neither ratio alone identifies the complete-call bottleneck. A device-task improvement can be hidden by Host launch cost in a wall-time study; keep [clock meanings](../verification/native-timing.md#choose-and-name-the-metric) explicit. [Banded GEMM](../../examples/algorithms/gemm_bias_silu_banded/) and [multicore GEMM](../../examples/algorithms/gemm_bias_silu_multicore/) provide runnable transfer and ownership controls.

Pipe busy times can overlap. Their sum being close to elapsed time does not prove a serial schedule, and `elapsed - sum(pipe_busy)` is not a measurement of waiting. Use timeline dependencies or controlled implementation comparisons for those conclusions. High activity alone also does not establish useful arithmetic throughput or the hardware limit.

## Distinguish bandwidth quantities

Keep the following quantities separate when diagnosing transfers or deciding whether to stop tuning:

- **Nominal device bandwidth:** the specification for the exact device variant. It is not a measurement of this kernel or a universal value for an architecture family.
- **Sustained device baseline:** a same-device transfer benchmark with its read/write mix, transfer path/layout, working-set size, cache policy and worker count stated. Compare the kernel with a relevant baseline under matched conditions; a previous application kernel's throughput does not establish this ceiling.
- **Kernel effective throughput:** explicitly counted useful bytes divided by the complete kernel's elapsed time, with the collector and decimal/binary units stated. Repeated transfers, padding and L2 hits separate this quantity from physical main-memory traffic. Repeated calls on the same buffers are not evidence of cold-cache bandwidth; control cache state outside the timed region and verify the control, or report cache state as unknown.
- **Per-core active bandwidth:** transferred bytes divided by that pipe's active time, as defined by the [profiler fields](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_performance_data.md). Summing rates with different active windows, or multiplying a median by core count, does not yield whole-device elapsed-time bandwidth. Where matching per-core bytes are available, sum those bytes and divide by a common task interval; this still measures the named transfer path, not automatically HBM traffic.

A nearly 100% MTE2 busy ratio can coexist with low delivered throughput. Check access granularity, layout conversion, repeated bytes, cache behavior and concurrent streams before declaring the device bandwidth saturated. A losing split or preload experiment rejects that candidate; it does not establish a universal contention ceiling or rule out all other transfer strategies. Without physical-memory traffic counters or a matched sustained-bandwidth benchmark, label an HBM-saturation explanation as a hypothesis and leave the hardware limit unmeasured.

Keep reused activations and streamed weights separate in the byte calculation. For example, a small A operand reread by many N owners contributes repeatedly to GM→L1 requests even when those reads may be served by L2. Adding those requests to a once-streamed B operand and dividing by whole-kernel time establishes effective transfer throughput, not a sustained HBM lower bound. A faster slot configuration verifies a performance gain; it does not by itself isolate exposed load latency, physical-memory saturation or the reason another shape gains less. Retain the gain while keeping those mechanisms unresolved until a distinguishing observation is available.

Check that a proposed lower bound is independent of the candidate time. Setting `bandwidth = bytes / elapsed` and then computing `bytes / bandwidth` returns `elapsed` by construction; it cannot establish saturation or remaining headroom. Several kernels with similar achieved rates do not remove this circularity. Without an independently justified limit, the original precision contract and legal alternative mappings still determine which comparison is worth testing. If the budget ends first, report an untested alternative rather than a proven floor.

## Decompose a busy VF stage

Inspect [VF runtime control and final loop form](../api/vf.md#keep-runtime-dispatch-out-of-vf-loops) before ranking the arithmetic. An invocation-uniform first/last flag can still prevent hardware-loop lowering when tested inside the row/chunk loop. Compare a static VF body selected outside the loop while holding arithmetic, geometry and synchronization fixed. Do not attribute a busy Vector pipe to useful vector throughput or missing overlap without this check.

Then count loads/stores, nonlinear operations, reductions, casts/packing and predicates at their actual row/chunk/block frequencies. [Runtime mask construction](../api/vf.md#repeated-runtime-masks-in-vf-loops) owns the remaining predicate alternatives. Mathematical complexity, source line count and a successful compile do not establish the dominant cost.

Use focused ablations to separate those costs, including a predicate's bounds calculation from mask generation and selection. Preserve initialized reads, ownership and the necessary handovers under the [probe controls](../verification/board-runs.md#diagnostic-probes), and label intentionally incorrect arithmetic as timing-only. Observe both the affected stage/pipe time and the complete-call time. A small edit with no speedup does not exclude the entire stage; compare a larger removal when needed to distinguish remaining work from slack. Once a cost is exposed, implement an equivalent path and restore numerical acceptance before treating its timing as a usable result.

## Correlate model and board observations

Start with source/generated-code work counts and board counters. Match the source, specialization, inputs, worker topology and native stage. Generated CCE or transformed IR can expose repeated transfers, slot selection and stage dependencies; per-core time and pipe activity test whether they matter on hardware. Whole-task average utilization can hide one busy core followed by idle peers. Check per-core span, work and transfers before interpreting a low compute ratio as a compute-engine tuning problem.

Before merging work items to reuse inputs, count both saved transfers and retained compute on the critical worker: VF reductions/nonlinear operations multiplied by their loop extents, serial dependencies, independent items and scheduling waves. Halving transfers while doubling sequential compute per item can lose, especially with unequal runtime lengths. A one-worker model can reveal reuse but cannot establish the parallel benefit of that grouping. A bottleneck observed for one paradigm or mapping is not a property of the task's name.

Optimize complete-call latency at the original numerical contract, not maximum utilization. Removing redundant work may lower a pipe's busy ratio while making the kernel faster. Simulated ticks, queue waits, task-level profiler ratios and operator-replay counters have different denominators and scopes; do not average them into one score. If they disagree, check model coverage, generated code, collector identity and a focused implementation comparison. Keep the disagreement visible instead of fitting model parameters to make this kernel look correct.

## Follow the symptom

| Observation | Next action and evidence owner |
| --- | --- |
| Vector is continuously busy, but reducing arithmetic does not explain total time | [Decompose the VF stage](#decompose-a-busy-vf-stage), including scalar predicate/control work, before another buffer-depth sweep |
| No compute/transfer overlap | Check whether stages actually alternate and buffers have valid lifetimes; use [slot depth](../api/memory-allocation.md#independent-buffers-and-slot-depth) and [cross-core sync](../api/crosscore-sync.md) |
| VF → Cube passes through ND staging and an extra move | Examine the [NZ handover and row+1 pitch](../api/crosscore-sync.md); [mm_softmax_mm](../../examples/algorithms/mm_softmax_mm/) is a runnable case |
| A separate activation/quantization pass dominates | Test an allowed [GEMM fusion](../patterns/gemm-desugar.md) or [fixpipe rider](../api/fixpipe.md), retaining the numerical contract |
| A preload or buffer-depth change does not help | First meet required slot lifetimes. Extra slots are not free, and preload interacts with the shared L1 budget; sweep a justified range after other layout changes rather than assuming monotonic benefit. See [flash attention](../../examples/algorithms/flash_attention/) and [MHA](../../examples/algorithms/flash_attention_mha/) |
| MTE2 is continuously busy | Ratios alone stop discriminating candidates. Distinguish [effective, sustained and active bandwidth](#distinguish-bandwidth-quantities) before comparing required bytes / device time, per-core counters and concurrent loading streams with a [per-core msprof capture](#collect-the-right-profile). Balanced core times can still lose; see [decode GQA](../../examples/algorithms/decode_attention_gqa/) |
| Several pipes are busy but local speedups do not move total time | Establish timing-only ablation floors: remove traffic or replace one computation station, clearly marking numerical results invalid. Overlapping ablations do not add linearly. See [MHA](../../examples/algorithms/flash_attention_mha/) |
| Ablation points to coupling between stations | For each MAC gap, find the predecessor track whose final event plus a consistent delay predicts MAC resumption. Compare steady-state windows and account for adjacent merged events before calling one beat unusually slow; see the MHA dependency analysis |
| Removing a station helps but shrinking it does not | Include both vector and serial scalar work, and compare the reduction with available slack. A cut below that slack can leave total time unchanged. A dependency model omitting scalar/slack terms can mispredict gains; use the removal-versus-shrink device comparison |
| No pipe is saturated and steady-state tuning has stalled | Inspect what must finish before the first MAC and after the last. Startup/tail dependencies may need a different band or preload placement, not deeper buffering; see [RMSNorm GEMM tails](../../examples/algorithms/rmsnorm_gemm_tail/) |
| Unequal work items create a long tail | For Host-known work, compare weighted partitions; for runtime lengths, use [device-side segments/partials](../patterns/multi-output-reductions.md#partition-unequal-runtime-reductions). Do not infer runtime work by Host readback |
| A moved prefetch is unexpectedly late | Inspect generated CCE (`kernel.cpp`) or IR from the [pipeline transform path](../api/crosscore-sync.md#slot-counts-identity-and-check-coverage). Stage-entry waits may precede independent loads inside that stage; an item-loop prefetch can have a different dependency. See [MHA](../../examples/algorithms/flash_attention_mha/) |
| Recompilation or dispatch dominates a wall-time study | Check [tiling keys](../../examples/matmul/tilingkey_specialize/) and [dynamic shape](../../examples/matmul/fundamental_usage/dynamic_shape/); distinguish this from device-task cost |

Any ablation or dependency simulation is diagnostic evidence, not a correct implementation or a hardware-performance substitute. Restore correctness before measuring a candidate for acceptance.

## Scoped mechanism references

- `unitFlag` / accumulation phase is a [performance mechanism](../../examples/matmul/fundamental_usage/acc_phase/), not a substitute for accumulation control.
- Tile-level versus VF softmax comparisons are [shape-specific](../api/vf.md); do not extrapolate one result to an entire API family or override workspace shipping policy.
- Changing preload must not change numerical results. Independent specializations and adequate steady-state reuse are covered by the [pipeline example checks](../tooling/examples.md#pipeline-examples).
- More conservative synchronization can cost time, but manual events change the correctness proof. Read the [cross-core contract](../api/crosscore-contract.md) before weakening synchronization.

The exact inputs, source hashes, toolchain identities and measured values remain with the linked examples. This page keeps reusable diagnostic actions rather than a second performance ledger.
