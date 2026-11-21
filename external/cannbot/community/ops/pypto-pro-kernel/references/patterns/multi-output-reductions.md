# Output ownership: writers, participants and stage boundaries

Use when outputs retain different axes, indexed reads/writes need an ownership proof, or unequal runtime segments need partitioning. For contiguous payloads, read the [SIMT record/lane mapping](simt-payload.md) as the implementation companion; this page owns reduction distribution, communication and final writes.

## Map each output independently

For each output, identify retained/reduced axes, destination multiplicity, initializer, participating lanes and final writer. A unique final writer need not compute the whole reduction.

Grouping input work by length or layout does not partition every output. An output without that group axis must combine contributions from all groups: reinitializing an accumulator and storing each group to the same GM region overwrites earlier contributions even on one Stream. Initialize once and carry the accumulation, or write group-private partials and merge them before the final commit.

Indexed input reads do not imply conflicting output writes. In a gather reduction, repeated source indices can still feed disjoint output rows. Start with owners of final elements or bounded reduction slices, accumulating locally and writing each partial once; compare an edge/record-scatter mapping separately. Counting each coefficient once is not enough to choose scatter: include payload atomics, contention, access coalescing, metadata/nonlinear recomputation, partial traffic and the merge. The cost of repeated arithmetic versus atomic updates needs a measured ownership comparison, not only a sweep of atomic bucket sizes.

| Structure | Candidate | Required proof |
| --- | --- | --- |
| Multiple records select one indexed destination | Atomic updates or owned partials plus merge | Initialization precedes updates; no lost or duplicated contribution |
| A record's statistic reduces its contiguous payload | [VF reduction](vf-row-reduce.md), cooperative lanes or a short serial owner | Every payload term participates once and every final statistic is written |
| Outputs share metadata and input values | A shared traversal where ownership permits it | Reuse does not introduce races or force incompatible work domains into one mapping |

A long per-record serial loop can dominate despite its tiny output. In a flattened element-per-thread kernel, placing that whole loop under a leader-only predicate also leaves neighboring lanes idle during the reduction. Let a leader finalize already-computed partials; compare that with taking over the entire scan merely to save a launch.

### Indexed-axis owners for local scatter

For a same-rank indexed write, factor the destination as `[outer,K,inner]`: an index changes only the `K` coordinate. One owner can therefore hold `(outer_coordinate,inner_coordinate)` and consume its update positions in the contract's order. Distinct owners cannot write the same destination element. This is a candidate decomposition, not validation of a new scatter implementation; [VF gather/scatter](../api/vf.md#31-gather-has-two-forms-out-of-bounds-behavior-is-defined-for-only-one) still owns address, dtype and uniqueness rules.

When the complete indexed axis fits in UB for the selected owners, two lane maps give a concrete uniqueness proof. With inner lanes `0<=l<W` and physical row pitch `P>=W`, offsets `index[l]*P+l` have distinct residues modulo `P`. For batched owners `r`, offsets `r*P+index[r]` are disjoint when `0<=index[r]<K<=P`. Use the actual physical pitch, including padding, and prove every offset fits both storage and the index dtype. Repeated destinations for one owner across update steps still require the correct combine/last-writer semantics and [memory-dependency ordering](../api/vf.md#212-multiple-memory-accesses-to-the-same-tile-inside-vf-have-no-ordering-guarantee); unique indices within one instruction do not prove those later dependencies.

For extrema updates with initialized destinations and unique instruction offsets, compare expressing the winning-update predicate directly as the scatter mask instead of explicitly selecting between the update and current value. This candidate requires supported compare and predicated-store operations for the actual dtype; it establishes neither current int64 support nor a speedup. Derive the predicate from the contract: a NaN-propagating maximum may require `isnan(update)` in addition to `update>current`, while ties, signed zero and an already-NaN destination can require further rules. Test those cases over repeated updates and retain the ordering proof above; a comparison-only mask can silently drop a required NaN update. No new target validation is claimed for this alternative.

A rank-one destination has no non-indexed owner axis to distribute. Mapping lanes to update positions then loses the uniqueness proof when indices repeat. Likewise, insufficient full-`K` residency or too few owners calls for a separately proved destination-slice, direct-GM or partial/merge candidate, including filtering, initialization and merge costs. Atomic addition cannot substitute for overwrite semantics; choose a unique winning update before writing when the contract requires deterministic last-write behavior.

<!-- kernel-query: {"id":"per-output-arithmetic","aliases":["per-output precision plan","multiple output numerical strategies","按输出分配精度","多输出误差链"],"when":"Outputs share inputs but differ in reduction length, cancellation, coefficient bounds, exactness or final storage dtype.","missing_facts":["Derive each output's arithmetic, coefficient bounds, rounding points, representation, participants and final writer at its original criterion."],"example":"algorithms/layernorm_backward/kernel.py","term_groups":[["output","outputs","多输出","逐输出"],["precision","arithmetic","cancellation","精度","误差","相消"]]} -->
## Choose arithmetic independently for each output

Extend the output ownership map with reduction length, coefficient signs/absolute-sum bounds, required rounding points, final dtype and near-zero/exact-output criteria. Shared input dtype does not imply a shared accumulation strategy. One output may permit bounded integer accumulation, another may need product residuals or compensated partials, while a bit-preserving output needs transport without numerical conversion. Keep these choices attached to their output consumers and live-resource costs.

For a weighted error `sum(c_i*e_i)`, the exact-arithmetic bound is `sum(abs(c_i)*abs(e_i))`. A uniform `abs(e_i)<=h/2` yields `h/2*sum(abs(c_i))`; term formation, coefficient representation, accumulation and final conversion still need their own bounds. Signed cancellation makes the absolute coefficient sum relevant even when the final result is small. Use [numerical budgets](../api/numerics.md#budget-residual-products-through-the-full-consumer) and [reproducible accumulation](../api/numerics.md#6-reproducible-reductions) rather than choosing a fixed-point scale from one sampled output. Preserve low components until the cancellation/rounding point that requires them.

[LayerNorm backward](../../examples/algorithms/layernorm_backward/) demonstrates independent row/column output mappings; [weighted combine](../../examples/algorithms/moe_weighted_combine/) demonstrates a bounded integer strategy, and [projection/normalization](../../examples/algorithms/mla_prolog/) demonstrates residual components. Their individual contracts do not establish one combined multi-output algorithm. When adapting, test cancellation, coherent small terms and range extremes for each affected output, keep unaffected outputs as controls and measure the complete shared traversal plus all merges.

## Reproducible indexed reductions

First choose a numerical strategy from [fixed-order and exact accumulation](../api/numerics.md#6-reproducible-reductions), then compare ownership. Reproducibility does not require one atomic update per payload element. Retain an output-owner control that gathers its contributors, accumulates locally and writes once. A direct scan of routing metadata is a useful small-domain control; count its repeated metadata work before extending it to many destinations.

When payload width amortizes preprocessing, compare constructing a device-side destination-to-input index with repeatedly scanning all routes or scattering the full payload. Account for counts, positions, index storage, initialization and the consumer in the complete call. An atomic counter can give each record a unique position without giving records a stable order: a floating reduction needing a fixed order must establish that order separately. An exact accumulator may permit arbitrary contributor order if its range and error proof cover it. Invalid records consume no contribution, empty destinations still get their defined output, and Host must not construct indices from tensor contents.

Privatizing scatter slots reduces collisions between owners but can leave many threads contending within each private slot. Count updates to the busiest slot at both levels. A shape bounds the number of destinations; it does not reveal how runtime routes populate them. Compare distributed, concentrated and skipped routes at the same shape before choosing a shape-only dispatch. Include the replicated workspace and its initialization/merge traffic; fewer collisions alone do not prove a complete-call gain.

[Weighted combine](../../examples/algorithms/moe_weighted_combine/) instantiates an error/range proof and worker/group private integer slots. Its source header owns the supported bounds and workspace cost; quantization parameters and dispatch choices are not defaults for other reductions.

## Distinguish two sources of atomic contention

An indexed scatter can collide because several records choose the same destination. A regular per-record statistic is different: using one global atomic per payload contribution introduces contention at every record's few statistic addresses, even when records are otherwise well distributed. Count updates per destination as well as total atomics. More workers do not reduce updates to one slot.

Use owned partials as a first cooperative baseline for the regular reduction:

1. Each participant handles its assigned payload slice, reusing only metadata independent of that slice's element index.
2. It normally writes `partial[record, participant, statistic]` once. A participant with no elements writes the neutral partial; every logical partial the consumer reads must be initialized.
3. A later native task reduces the participant axis and uniquely writes each statistic. Workspace scales with participants and statistic width, not every arithmetic intermediate. Consecutive native calls on one Stream supply the stage boundary; no in-kernel cross-group barrier is needed for that GM handover.

The producer may also perform genuinely conflicting scatter updates when safe input reuse makes that useful. Keep their cost separate from optional atomic consolidation of the statistic. Compare the serial owner, owned partials and any atomic alternative including all launches; accepting an atomic signature and passing correctness do not establish cheap contention. [Embedding-bag partial/merge](../../examples/algorithms/embedding_bag/) shows a native GM handover.

On-chip cooperation is another option, but block barriers must be reached uniformly. Group-local ownership does not prove every thread reaches a block barrier; see [SIMT synchronization](../api/simt.md#8-two-new-rules-both-cases-of-documented-but-not-enforced).

<!-- kernel-query: {"id":"simt-atomic-aggregation","aliases":["SIMT atomic return value","SIMT atomic aggregation","SIMT UB histogram","SIMT 原子返回值","SIMT 局部直方图","atomic 返回值优化"],"term_groups":[["atomic","atomics","原子"],["return value","unused result","aggregation","histogram","返回值","聚合","直方图"]],"when":"Reduce the cost of SIMT atomic updates without changing result, allocation or ordering semantics.","missing_facts":["Identify whether the old value is consumed, the target dtype/space, update multiplicity and arithmetic contract; include initialization and merge in any comparison."],"example":"algorithms/simt_histogram/kernel.py","symbols":["pl.simt.atomic_add"],"discovery_terms":[["histogram","counters","categorical","直方图"],["SIMT","UB","block","local","聚合"]]} -->
## Local aggregation and atomic instruction choices

For exact integer updates whose atomic old value is unused, also compare combining consecutive equal targets within each thread's traversal. Keep a typed key and accumulator, flush on a target change and at the end, and prove the accumulator range. This reduces update multiplicity without shared storage or extra barriers. A run is defined along that thread's actual grid-stride sequence, not adjacent source positions. Dispersed or alternating targets can retain every atomic and add branch/live-state cost; test them alongside long-run and concentrated inputs. Consumed reservation/ticket values cannot be replaced by this counting-only transformation. Floating updates need separate order/accuracy acceptance.

The native [histogram kernel](../../examples/algorithms/simt_histogram/kernel.py) exposes this comparison with `local=False, combine_runs=True`; it preserves global initialization and exact integer counts. A direct path is still a necessary control. Combining at thread scope, aggregating in shared UB, and spacing GM targets change different costs. For consumed returns with an unordered per-key ticket contract, consider [range reservation](#preserve-returns-through-range-reservation) instead of discarding those returns.

When a bounded set of reduction targets fits UB, compare per-block local aggregation followed by a small GM merge with direct GM updates. Initialize every local bin before updates, use unconditional block barriers between initialization/update/consumption, and establish global initialization before merging. All threads must reach the barriers even when some have no input work. Count both block-local hotspots and final GM updates; local storage can change contention without removing it. Include initialization, scratch, merge traffic and every launch at the original precision. A floating local reduction changes rounding order and needs its own accuracy/reproducibility acceptance.

The native [SIMT histogram](../../examples/algorithms/simt_histogram/) keeps direct-GM and block-local variants with the same two-stage initialization contract and exact integer reference. The existing [VF histogram](../../examples/algorithms/byte_histogram/) is a separate regular-vector implementation; neither default establishes a universal paradigm winner.

If the algorithm does not consume an atomic's old value, avoid adding a diagnostic output or dependency solely to retain it in a shipping path. Compare actual generated instructions and complete calls for the same dtype and memory space. The [upstream atomic guide](https://gitcode.com/cann/cannbot-knowledge/blob/01b5bf44246aa6f2ac04e4ee6dfe107bf30c2095/knowledge/ops/ascendc/guides/operator_practice/simt_operator_optimization/instruction_optimization/atomic_instruction_optimization.md) observes different unused-result behavior for INT32 and INT64; its gains are not transferable constants. An already-dead local assignment may be eliminated by the compiler without a source rewrite. Preserve old values used for ticket allocation, ownership, compare-and-swap or correctness tests; discarding them changes the algorithm.

<!-- kernel-query: {"id":"simt-ticket-ranges","aliases":["SIMT ticket ranges","SIMT batched reservation"],"term_groups":[["simt","atomic"],["ticket","reservation","\u7968\u636e"]],"example":"algorithms/simt_ticket_reservation/kernel.py","when":"A consumed atomic return allocates per-key tickets, with only final counts and unique per-key ticket permutations required.","missing_facts":["Establish the ticket ordering/visibility contract, integer bounds and global initialization; bulk reservation must retain the returned base and a distinct rank for every item."]} -->
### Preserve returns through range reservation

When the contract permits a whole equal-key run to linearize together, reserve its length with one atomic add and use the returned base plus each item's rank. A second traversal over that thread's saved start/end/stride can publish the tickets without a thread-private array. Ranges for one key are disjoint because their bases come from the atomic; merely writing a local count to each ticket is not equivalent. Keep the accumulator and base/rank arithmetic in their proven integer domain.

The [native ticket reservation](../../examples/algorithms/simt_ticket_reservation/kernel.py) retains both one-element and range-reservation paths, independent per-key permutation checks and initialization on the same Stream. This example promises final counts and a per-key ticket permutation, not input-order ranks, external per-item progress or intermediate count observations. Stronger ordering or allocation side effects require a separate proof. Compare no-run and concentrated traversals; a globally alternating key sequence may still be constant along an even grid stride.

<!-- kernel-query: {"id":"simt-atomic-target-layout","aliases":["SIMT atomic sector","SIMT atomic counter layout","atomic sector contention","SIMT 原子扇区","原子计数器布局"],"term_groups":[["atomic","原子"],["sector","target layout","counter layout","扇区","计数器布局"]],"when":"Distinct GM atomic targets may still share a memory transaction resource.","missing_facts":["Resolve target sector/alignment semantics for the installed platform and distinguish address collisions from shared-sector traffic; preserve the public output layout and count private-layout initialization/merge costs."]} -->
### Target address layout

Different GM addresses can share a sector and contend for resources even without same-address updates. Compare tightly packed versus suitably spaced **internal** counters, or owned local partials, when the target platform and profile support that hypothesis. Keep update count, dtype, actual thread geometry and return-value use fixed to isolate layout. Preserve dense public outputs by a verified final merge if necessary. Padding changes workspace and traffic; the 128-byte sector used by the upstream Ascend 950 experiment is a version/platform premise, not a universal pointer alignment requirement or an instruction to pad caller tensors. UB atomics require their own bank analysis rather than this GM-sector rule.

## Partition unequal runtime reductions

When an output scans a runtime segment of length `L_s`, estimate both total work and the busiest worker's work. One owner per output avoids conflicts, but outputs for a long segment may land on one worker while others finish. Grid-stride coverage and average segment length do not establish balanced runtimes.

Compare a direct owner for short segments with splitting long reductions into disjoint slices and merging their partials. Apply the owned-partial proof above, retaining scratch until the consumer completes. Include partial traffic, initialization and the added launch/merge in the comparison; a `segment × chunk × output` rectangle allocated from static upper bounds can contain many empty tasks. These criteria apply to VF, SIMT and other native phases, not only one API family.

Compute runtime validity on device, without Host readback. For a contiguous valid interval, cap the loop end before entering it; an inner predicate may prevent invalid reads while still traversing unused positions. Skip empty payload loops, but initialize every partial the consumer will read. Define the tail slice and final output owner explicitly. Consecutive native kernels on one Stream can supply completion ordering; a block barrier cannot synchronize independent workers.

Revalidate changed summation order at the original tolerance. Compare same-shape uniform, concentrated, mixed-length and empty segments, including segments with no valid indices. Choose chunk size from complete-call results for those distributions, then retune the launch for the new work units. [Embedding bag](../../examples/algorithms/embedding_bag/) demonstrates direct and partial/merge paths; [normalized contractions](normalized-contractions.md#merge-partitions-of-the-reduced-axis) owns the extra reference/rescaling rules for normalized aggregates.

## Fuse after checking work domains

Combining initialization with a different reduction saves a launch but couples iteration counts, access patterns and parallelism. A union loop over the larger domain may leave the important computation serial or sparsely assigned. Compare separate stages when they enable a better reduction or contiguous movement; conversely, combine consumers when a shared traversal safely avoids input work. Task count alone is not the objective.

Conflicting updates require completed initialization. The [scatter example](../../examples/algorithms/simt_scatter_add/) uses separate native kernels on one Stream; atomics and block barriers do not create a grid-wide initialization barrier. For floating sums, retain the [dtype-specific acceptance rule](../api/simt.md#7-order-independence-in-the-int32-example) and original tolerance.

## Derive backward boundary rules afresh

A value-preserving forward shortcut need not preserve derivatives. For `y=(1-t)*a+t*b`, `t=0` returns `a`, but the selected cell's derivative is still `b-a`. Zero interpolation weight can therefore have nonzero derivative. Derive validity, the selected piece at a boundary and every differentiated contribution independently before reusing a forward skip/direct-copy branch.

Test selected-piece boundaries, partially/fully invalid neighborhoods, repeated destinations, degenerate axes and tails. Check exact all-invalid outputs even within mixed inputs. Poison outputs before repeated calls, verify inputs/guards, and compare floating atomic results to the reference on each call rather than requiring schedule-independent bits.

## Compare and retain

Follow the [optimization experiment sequence](../workflows/optimization.md#change-discipline). If a dominant phase has a long serial scan, repeated payload metadata or avoidable atomic contention, a thread-count sweep does not test the ownership hypothesis. Compare the logical work unit, then retune its configuration. Include concentrated and distributed destinations and small workloads where added stages can lose.

Runnable examples: [layernorm backward](../../examples/algorithms/layernorm_backward/) uses different reduction axes; [bilinear sampling backward](../../examples/algorithms/bilinear_sampling_backward/) combines scatter with owned record partials and merge. The latter's fixed configuration gained on larger workloads and cost more on smaller ones; remeasure configuration and any simple-path dispatch for the new task.
