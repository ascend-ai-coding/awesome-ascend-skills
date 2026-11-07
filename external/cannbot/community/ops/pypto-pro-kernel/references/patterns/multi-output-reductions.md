# Output ownership: writers, participants and stage boundaries

Use when outputs retain different axes, indexed reads/writes need an ownership proof, or unequal runtime segments need partitioning. For contiguous payloads, read the [SIMT record/lane mapping](simt-payload.md) as the implementation companion; this page owns reduction distribution, communication and final writes.

## Map each output independently

For each output, identify retained/reduced axes, destination multiplicity, initializer, participating lanes and final writer. A unique final writer need not compute the whole reduction.

Indexed input reads do not imply conflicting output writes. In a gather reduction, repeated source indices can still feed disjoint output rows. Start with owners of final elements or bounded reduction slices, accumulating locally and writing each partial once; compare an edge/record-scatter mapping separately. Counting each coefficient once is not enough to choose scatter: include payload atomics, contention, access coalescing, metadata/nonlinear recomputation, partial traffic and the merge. The cost of repeated arithmetic versus atomic updates needs a measured ownership comparison, not only a sweep of atomic bucket sizes.

| Structure | Candidate | Required proof |
| --- | --- | --- |
| Multiple records select one indexed destination | Atomic updates or owned partials plus merge | Initialization precedes updates; no lost or duplicated contribution |
| A record's statistic reduces its contiguous payload | [VF reduction](vf-row-reduce.md), cooperative lanes or a short serial owner | Every payload term participates once and every final statistic is written |
| Outputs share metadata and input values | A shared traversal where ownership permits it | Reuse does not introduce races or force incompatible work domains into one mapping |

A long per-record serial loop can dominate despite its tiny output. In a flattened element-per-thread kernel, placing that whole loop under a leader-only predicate also leaves neighboring lanes idle during the reduction. Let a leader finalize already-computed partials; compare that with taking over the entire scan merely to save a launch.

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
