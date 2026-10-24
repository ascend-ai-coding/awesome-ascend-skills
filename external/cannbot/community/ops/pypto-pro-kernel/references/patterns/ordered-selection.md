# Stable selection and merging ordered candidates

Use this route for repeated extrema, Top-K, ranked candidate lists or a merge that carries values together with indices/tags. First define one total order for every stage: primary score, tie key, and validity. A sentinel may represent an exhausted stream only if no valid item can tie it. Keep the original payload separate from any transformed comparison key; numerical equality does not imply identical floating-point bits.

## Choose the candidate representation and count its work

Before fixing a partition, estimate per row the number of DMA transfers, VF calls, cross-lane reductions, GM intermediate bytes and native tasks. Capacity alone is not a cost model. Register width describes lane parallelism; it need not be the DMA chunk size or the lifetime of a local candidate list. Read [transfer chunks and persistent aggregates](vf-row-reduce.md#when-a-full-row-does-not-fit) when inputs are streamed.

Compare at least two representations before tuning a grouping constant. One option repeatedly reduces each short input fragment and emits its candidates; another lets each lane or worker update a bounded local summary over many input fragments, then combines those summaries. In the latter, per-lane updates can proceed in parallel, and cross-lane selection is postponed until the summaries must be combined. A bounded ordered prefix is a mergeable summary: it need not be materialized as a fresh GM array for every register block. Choose registers or an explicit UB tile for its lifetime, and apply the memory-ordering contract below when that tile is reused.

Local pruning can reduce the number of values yet still increase total cost: emitting one partial list per register creates many small loads, stores and handovers, followed by additional reads and launches. Grouping those lists later only reduces the later-stage work. Larger DMA chunks, retaining summaries across chunks, or reducing the number of global stages change different costs; identify which term each experiment changes. More partitions can expose parallelism, but their extra work must be included in the whole-call comparison. Treat these counts as hypotheses, then measure; do not assign unknown operations a guessed zero cost.

## Separate construction, emission and worker coverage

An existing small-prefix example is a correctness control, not a cost model for larger prefixes. Estimate the construction and emission phases separately. With N input items, L local lists and at most K_local retained items per list:

| Phase or representation | Comparison work to account for |
| --- | --- |
| Insert every input into a bounded local list | O(N K_local); cap K_local by the number of items that can reach that list |
| Emit K results by scanning every retained candidate each time | O(K L K_local), plus any repeated exclusion or candidate-clearing pass |
| Emit from the current head of each sorted list | O(K L), plus cursor updates and refreshing the changed head |

These are logical work counts, not hardware durations; lane parallelism, horizontal reductions and indirect UB loads have different costs. A useful implementation comparison can replace only emission, retaining local-list construction if it already establishes the new consumer's invariant. Final-output tests alone do not prove that internal lists are sorted by the complete comparator; verify that precondition before using the head/cursor proof below. It need not replace the entire sorting algorithm. When Vector computation dominates and GM movement is minor, a transfer-size sweep alone leaves those selection costs intact; compare a representation that changes the dominant work before spending the remaining budget on neighboring DMA sizes.

Count useful workers as well as per-worker work. If one worker owns each complete row, R rows expose at most min(R, W) active workers from a W-worker launch. For few long rows, compare splitting each row's input into ordered partitions with bounded local prefixes and a final merge, using the same pruning proof below. Preserve original indices and the total comparator across partitions. Include all partial traffic, merge work and native launches; sufficient row parallelism or short inputs can favor the simpler row-owned path. A fully busy Vector pipe on one owner does not establish device-wide parallelism.

## Reduce the candidate set with a proof

If only the first K items are needed, each partition may discard everything after its own first K under the **same order**: a discarded item already has K predecessors in its partition, hence cannot enter the global first K. The partition can be a lane, tile, input chunk or worker. This proof does not choose the partition size, the local selection algorithm or the physical layout; compare their transfer cost, live state and work per item.

Apply the secondary key during local insertion and displacement, not only during final selection. Arrival order alone does not make a bounded insertion list stable: a later stronger item can displace an older tied item into a slot occupied by a newer tie. Comparing only primary values can then reorder the ties or discard the older one at the retention boundary. A later stable merge cannot recover a candidate already pruned under the wrong order.

When these local lists are already sorted, the next global winner must be one of their current heads. Maintain a cursor and current key/payload for each nonempty list; choose among the heads, emit the winner and advance only its list. Compare this with scanning all retained candidates for every output. Rechecking every previously emitted item is unnecessary if advancing the winning stream itself proves uniqueness. For L lists of at most K retained items, a full candidate scan costs O(LK) comparisons per output; a scan of current heads costs O(L). Local list construction and irregular refresh loads may still dominate, so measure both small and long inputs.

The state transition is the contract: exhausted streams cannot win, cursors cannot leave allocated storage, and key, original index, payload and cursor use the same winner predicate. A gather mask is not an address proof; make every issued address legal, then exclude invalid candidates from comparison. Lane exchanges need the [coupled-state boundary rules](../api/vf.md#grouped-state-and-boundaries-in-lane-exchange).

## Place mutable state deliberately

Keep frequently updated state in registers when feasible; persist only the state that must cross VF calls. If a VF writes a candidate tile and then reloads it in a later iteration, reason about the actual load/store dependency, including the loop backedge. `auto_mutex` does not by itself order all accesses inside one VF. Use the [VF memory-ordering contract](../api/vf.md#212-multiple-memory-accesses-to-the-same-tile-inside-vf-have-no-ordering-guarantee) to choose a supported memory barrier, separate VF calls or a representation without that memory round trip. A repeated winner can indicate either a wrong exclusion/cursor transition or stale state; distinguish them with a small deterministic input before adding more exclusion tests.

Compact values and indices still need a [physical output plan](vf-small-results.md). All `vf.*` operations, including mask construction and initial register fills, execute inside `@pl.vector_function`; `pl.section_vector()` alone does not establish that scope. Check required positional arguments against the installed API declaration: implicit destination operands in lower-level errors are not extra Python arguments, and optional-looking masks may be required.

## Verify the order and the payload independently

Use an independent stable reference with exactly the specified comparator. Check indices, their uniqueness and order, then compare selected payloads against the original input at those indices. If the contract preserves floating-point bits, compare integer views as well as numeric values. Include all-equal inputs, ties spanning partition boundaries, both signs of zero, adjacent representable values, finite extrema, subnormals, exhausted lists and partial final chunks. Repeat after poisoning outputs and vary worker counts. Do not infer stability from random inputs or from a sorting API's default behavior.

Exercise tie order at a real pruning boundary: concentrate the winning candidates in one partition, fill its retention capacity with ties, then introduce stronger later keys that displace older entries. All-equal inputs do not exercise displacement, and diffuse ties may never fill any local list. Also check the new consumer's internal ordering requirement when replacing a full scan with head-only emission.

For tuning, profile the complete call with identical inputs and worker counts. A faster merge may matter most when K is large relative to the input length, while long-input performance may be governed by local insertion work and DMA granularity. Report both regimes.

Runnable reference: [row_topk](../../examples/algorithms/row_topk/) keeps lane-local candidate prefixes in bounded UB storage and orders their updates explicitly. It is a verified synchronization reference, not a claim that its final candidate scan is the best merge strategy. The tests compare complete finite FP32 payloads by integer bits, including signed zero, subnormals, boundary ties, extra shapes and compact output guards.

[row_topk_heads](../../examples/algorithms/row_topk_heads/) demonstrates head/cursor emission with a complete comparator in local insertion too. Its tests include displaced ties that actually reach local pruning capacity. The implementation keeps one owner per row; it does not demonstrate splitting a single row across workers, and its local insertion cost remains visible in full-call measurements.
