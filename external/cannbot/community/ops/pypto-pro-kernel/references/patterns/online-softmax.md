# Pattern: online softmax / flash-attention block recurrence

Use after [normalized contractions](normalized-contractions.md) has selected the score producer, validity domain and exponential reference. This page owns streamed recurrence, layout and state lifetime. Concrete block sizes, preload settings and historical measurements stay with the runnable examples below.

## Applies when

Stream score blocks when capacity or fusion makes materialization unattractive. Keep an offline baseline when the whole score row fits comfortably; sequence length alone does not establish a benefit from online rescaling. A bounded fixed reference may remove rescaling entirely, so first complete the [range proof](../api/numerics.md#4-fixed-references-in-bounded-exponential-normalization). Compare complete calls with the same valid domain.

## The recurrence, and where state lives

For each nonempty score block `S_j`, retain a running maximum `m`, denominator `l` and unnormalized weighted accumulator `U`:

```text
m_new = max(m, rowmax(S_j))
alpha = exp(m - m_new)
W_j = exp(S_j - m_new)
l_new = alpha * l + rowsum(W_j)
U_new = alpha * U + W_j @ V_j
```

Finalize `O = U/l` and, when required, `LSE = m + log(l)`. Define the first valid block and the all-empty output explicitly; avoid `-inf - -inf` and `0/0`. A finite initial maximum is usable only with a proved score-range bound; the sentinel in one example is not a general default.

A runnable placement keeps FP32 row statistics and accumulators in UB while each Cube payload product is drained to the appropriate AIV. Other placements need the same live-state and precision proof. A VF register persists only within its VF call; use [explicit state residency](vf-row-reduce.md#choose-where-dependent-phases-share-state) across calls and the [VF memory-ordering contract](../api/vf.md) for in-place state updates.

When a validated fixed reference removes blockwise rescaling, rederive the consumers and lifetimes of `l` and `U` before retaining the anchor's per-block drains. If `U` is only a sum of payload products until final normalization, one candidate is to retain it in Acc using [GEMM's in-place accumulation](../api/gemm.md#division-of-responsibilities-between-the-caller-and-the-sugar) and drain at finalization. Compare the complete call only after proving Acc capacity, first-block initialization, cross-item reuse, final handover synchronization and accumulation precision. This is a conditional residency choice, not an established speedup or a reason to remove required pipeline events. It does not apply unchanged when a data-dependent reference still rescales `U`.

State such as `alpha` that crosses stages needs slots for its full producer/consumer lifetime, including preload. Derive those slots and IDs using [cross-core synchronization](../api/crosscore-sync.md), and check each AIV's effective split ownership. Use a continuous per-worker logical tick for state and prefetch rings when items can contain odd numbers of panels. Resetting a slot selector at every item can overwrite a preceding item's deferred consumer even though each item works alone. Successful compilation alone does not prove safe reuse. For independent reduction partitions, use the [normalized-partial merge](normalized-contractions.md#merge-partitions-of-the-reduced-axis) instead of assuming that online state can be summed directly.

## Build the serial recurrence before overlap

Start from the [verified serial contraction chain](normalized-contractions.md#connect-the-bounded-chain). This section owns the additional recurrence controls before overlap; probe construction follows [board-runs](../verification/board-runs.md#diagnostic-probes). Keep the applicable controls in the kernel's existing acceptance tests:

| Control | What it must establish |
| --- | --- |
| One item, one score block | Physical/valid query rows, complete QK→weights→PV→output transport, coefficient narrowing and output coverage |
| The same item, at least two distinguishable score blocks | Carry `m`, `l` and `U` across the update branch. Use a later larger maximum so `alpha` is nontrivial, and nonzero payloads so a bad denominator affects the output; compare intermediate state if final normalization hides the defect. Load the old state into the [FMA's target name](../api/vf.md#destination-registers-used-as-inputs) |
| Multiple batches/heads with distinct values and selected keys | Correct batch, KV-head and grouped-query offsets in both loads and stores. Expand only legal contract dimensions; an explicitly single-head contract needs no invented multi-head case |
| Enough blocks to wrap the state slots, followed by another item with a different reduction length/ring phase on the same worker | Ring selectors, first-block reset, finalization and reuse across items; keep initialization and finalization inside their owning stages or establish an explicit completion boundary before reuse |
| Positive preload with enough tasks for fill, steady-state and drain | Compare against the accepted serial kernel, including multiple workers and repeated calls. For unchanged arithmetic order, require the [serial/shipping bitwise comparison](#numerical-acceptance-template) |

When one-block output passes but the two-block control fails, inspect old-state operands and persistence before tuning NZ packing or preload. When only positive preload fails, inspect the transformed schedule, slot distances and code outside stages. A single-block softmax probe never exercises the running denominator's update branch.

## Layout choice: transposed scores (lane = q) vs standard (lane = key)

| Layout | VF organization | Cost to check |
| --- | --- | --- |
| `S = Q K_j.T`, lane = key | Horizontal row reductions and broadcasts | Reduction frequency and the score/weight row layout |
| `S.T = K_j Q.T`, lane = query row | Elementwise accumulation across key registers | Active query lanes, longer key loops and transposed weight handback |

Count active lanes and instructions at the actual query/key block sizes. A layout that saves horizontal reductions for many query rows can waste most lanes in thin decode. Include coefficient narrowing, NZ/ZN transport and physical padding in the comparison.

Plan L1 and L0 blocks separately: retaining a large input block in L1 does not require the whole block to occupy L0 at once. Check transpose/slot forms through the [GEMM contract](../api/gemm.md), and resolve `pl.insert` coordinates through [data movement](../api/datamove.md). For a thin transposed panel, use the [rectangular NZ-to-ZN handover](../api/datamove.md#rectangular-nz-to-zn-handover) to distinguish the GEMM declaration from the producer's assembly frame and a split-write rejection from a dimension limit. Square tiles can hide an incorrect source/destination frame, so test nonsquare shapes and tails before scaling the schedule.

For causal/window masks, include [runtime predicate construction](../api/vf.md#repeated-runtime-masks-in-vf-loops) in the layout comparison before tuning preload. Derive full, empty and boundary regions from logical query/key coordinates and each AIV's valid rows, including any folded head/position axes. This route covers block specialization, vector-maintained boundaries and MHA's precomputed triangular bias; a non-causal anchor does not establish their cost or validity.

## Numerical acceptance (template)

Use the original mathematical reference and tolerance. If the task permits deriving a bound, include score accumulation, exponential approximation, coefficient narrowing, payload accumulation and finalization; the [flash-attention reference](../../examples/algorithms/flash_attention/) shows one such derivation. Do not make the reference adopt candidate-specific coefficient rounding without a contractual reason.

Exercise state transitions with rescaling on every block, a maximum only in the first or last block, all-negative scores, one-hot selection, uniform scores and empty blocks/rows. [Recurrence coverage](../diagnostics/check-coverage.md#6-random-inputs-are-blind-to-recurrent-state-defects-structured-inputs-are-what-catch-them-board-evidence-2026-09-30) explains why random inputs can miss a non-carried maximum; targeted negative controls should expose the intended defect.

When only preload or scheduling changes and arithmetic order is preserved, compare serial/shipped executions and repeated calls bitwise. When an accepted design changes reduction order, apply the original numerical contract and its required determinism instead.

## Tune buffering and the dependency chain

Choose L1 panels, L0 steps and VF state widths independently. For several matrix products in one loop, budget their [shared L0 groups](../api/memory-allocation.md) together: changing the first product's K step and the second product's N strip can permit double buffering without enlarging the resident inputs. Inspect the emitted extract/matmul/reuse sequence as well as MTE2 counters. Component axes shorter than a preferred step need their own matching [L0 view and step](../api/gemm.md#effective-dimensions-and-splitting); a successful full-width component does not verify the short component.

A scalar statistic per row need not occupy a full register-width broadcast row in UB. Compare a legally aligned compact record with [broadcast-on-load](vf-small-results.md), including every live state slot, while retaining the complete weighted accumulator at the required precision. The [streamed MLA tests](../../examples/algorithms/mla_bounded/test.py) exercise compact state, odd panel counts, different component widths and serial/overlapped agreement.

- Account for Q, K/V cache, coefficient handover and accumulator slots in one live-storage budget. More preload can require giving up cached inputs; more slots do not automatically improve overlap.
- Inspect the generated schedule for fill, steady-state and drain costs. A correct prefetch issued after a blocking stage-entry wait can be too late to hide the transfer; issue position matters as well as distance.
- Prove prefetch bounds even when the loaded values would be discarded. Numerical agreement cannot expose every unused out-of-bounds read; follow [access-check coverage](../diagnostics/check-coverage.md#5-three-distinct-criteria-for-out-of-bounds-access-each-with-its-own-blind-spot-board-evidence-2026-09-29).
- Use [model/board diagnosis](../diagnostics/performance.md#correlate-model-and-board-observations) to distinguish critical work from slack or coupled waits. A shorter VF phase may leave total latency unchanged; diagnostic ablations are not numerically valid candidates.

Choose and revisit configurations through the [optimization workflow](../workflows/optimization.md), rather than copying an example's preload range or block size.

## Scaling up to many heads and causal queries

Estimate work per `(batch, head, query block)` from its valid key range; equal item counts can leave unequal tails. Host-known geometry can support weighted partitions, while device-resident lengths need [device-side work assignment](multi-output-reductions.md#partition-unequal-runtime-reductions). Preserve reuse among consecutive items that consume the same K/V prefix.

Compare loop nesting and driver-loop lifetime: repeated pipeline fill/drain can dominate short items, while a longer shared schedule may retain more state. Keep deferred non-scalar stage arguments loop-invariant and pass changing slot selectors as scalars under the [stage contract](../api/crosscore-sync.md). Inspect compiled code when a proposed prefetch or overlap fails to appear.

## Decode / very thin m (GQA)

Classify the work by the number of query rows sharing one K/V operand, rather than by a prefill/decode name or the causal flag. If a block includes several query positions and several query heads per KV head, both are retained axes: their product is the logical row count before padding. A short multi-position query can therefore use the same thin-group mechanism as single-position decode.

Use [decode_attention_gqa](../../examples/algorithms/decode_attention_gqa/) as the first mechanism anchor when only a few query rows are active. Its standard score layout reduces over contiguous keys instead of spending most query-vector lanes on padding. Adapt its logical row map and [validity predicates](normalized-contractions.md#preserve-coordinates-when-folding-retained-axes) while retaining the verified conversion/packing and stage interfaces for the first probe. Its single-position, non-causal contract does not implement the new mask for you. The wide-query flash examples remain useful for recurrence or mask ideas, but their transposed score layout and physical register-sized query extents require a separate cost and capacity argument at thin M.

Make the initial adaptation preserve the anchor's arithmetic, score orientation and conversion interfaces while changing public indexing and useful regions. Establish a separate small product before replacing that computation with a new VF/Cube mapping. In particular, a VF alternative consuming a transposed operand needs its own [physical address-map probe](../api/datamove.md#verify-a-vf-consumers-address-map-after-reinterpretation) before a complete normalization/weighted-sum body.

Keep [useful rows, producer extent and receiving capacity](../api/crosscore-sync.md#logical-validity-is-not-transfer-capacity) separate when adapting padding: reducing a logical group does not reduce an unchanged padded producer's FIX writes. Derive each AIV's owned coordinates from the actual source extent. Also retain a logical-item loop over a legal worker budget using the [mixed-group driver](../api/crosscore-sync.md#a-mixed-kernels-block_dim-is-a-group-count-launch-semantics--index-semantics): more batch/head items do not imply that the device can launch that many execution groups at once.

At thin query groups, compare standard-layout row reduction with query-lane layouts using the actual active-lane count. When the matrix path pads query rows, distinguish physical from valid rows during loads, splitting, normalization and stores; a padded output row is harmless only while it remains isolated from valid reductions.

Use group-distinct inputs and different selected keys within one group to expose head-mapping errors. Retain first/last maxima and empty-row tests. A grouped Cube schedule can be bandwidth dominated while a VF or scalar implementation has a different critical cost; select the ruler from measured work, not the task name.

## Runnable references

| Example | Detail to inspect |
| --- | --- |
| [flash_attention](../../examples/algorithms/flash_attention/) | Block recurrence, state slots, numerical bound and serial/shipped comparison |
| [flash_attention_mha](../../examples/algorithms/flash_attention_mha/) | Causal work distribution, K/V reuse, prefetch placement and shared L1 budget |
| [decode_attention_gqa](../../examples/algorithms/decode_attention_gqa/) | Thin query groups, standard score layout and physical/valid row separation |
| [mla_bounded/_kernels/decode.py](../../examples/algorithms/mla_bounded/_kernels/decode.py) | Split score features, wide independent payload, compact row state and shared L0 buffering |
| [mm_softmax_mm](../../examples/algorithms/mm_softmax_mm/) | Whole-row on-chip counterpart, without blockwise online recurrence |

Each source header owns its tested shapes, toolchain identity and measurements. Historical examples establish mechanisms and comparisons within that scope, not current capability limits or universal dispatch choices.
