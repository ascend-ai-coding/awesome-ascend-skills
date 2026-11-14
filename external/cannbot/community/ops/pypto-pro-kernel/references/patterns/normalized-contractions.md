# Normalized contractions: shared row state and phase interfaces

Use for normalized weighted sums of dense or indexed payloads and their reverse/gradient passes, including segmented domains, extra row statistics and runtime masks. The matrix chain `S = X @ Y.T`, `P = normalize(S, mask)`, `O = P @ Z` is one instance; scores may instead be supplied directly. An irregular validity domain or gather does not make all phases irregular.

## Select a correctness anchor

For a new forward chain, choose an anchor by the retained/reduced dimensions and live storage, then establish one complete numerical slice before tuning. A permitted GM workspace does not require a materialized implementation. Follow [anchor adaptation](../workflows/authoring.md#selecting-an-example), including one relevant unchanged test on the installed toolchain.

### Starting path for runtime-selected rows

Choose the anchor by its value/index contract as well as its dimensions. The [sparse-attention kernel](../../examples/algorithms/sparse_attention/kernel.py) and [tests](../../examples/algorithms/sparse_attention/test.py) implement the CANN contract: FP16, distinct valid indices, V equal to a K prefix, BSND dense-mask traversal or BNSD selected-row traversal, and right-aligned causal masking. They show a staged online recurrence with static first/last VF bodies. They are not an independent-V, BF16 or duplicate-index contract; preserve those semantics through a compatible anchor or a separately verified adaptation.

For phase isolation or a materialized control, connect these interfaces:

1. **Selected rows → dense product.** Open the [indexed product kernel](../../examples/algorithms/indexed_batched_gemm/kernel.py) and [its test](../../examples/algorithms/indexed_batched_gemm/test.py). It demonstrates a native SIMT panel producer followed by Cube, including FP16/BF16 and short M. Name the selector group, retained vectors sharing its panel, and reduced feature axis. Retained vectors can be groups or heads; distinct selectors need distinct panels. Verify transport and one product on coordinate-distinct data before extending this edge.
2. **Scores → normalization → weighted product.** Open the [materialized chain kernel](../../examples/algorithms/mm_softmax_mm_materialized/kernel.py) and [its test](../../examples/algorithms/mm_softmax_mm_materialized/test.py). Connect its phases to the verified operand path, preserving the full coefficient conversion/store map. Implement the task's mask in original coordinates and its public layouts separately; the examples do not provide those semantics.
3. **One complete task slice.** Use the supplied evaluator for a nondegenerate case, following [reference calibration](../workflows/authoring.md#reference-and-evaluator); an isolated golden call may bypass its dtype promotion and comparison rules. Nonuniform scores and nonzero payloads must exercise every phase. On failure, isolate the first mismatching intermediate instead of launching the complete case matrix. For a performance target, use this passing slice for the [representative computation-path comparison](../workflows/authoring.md#compare-computation-paths-before-expanding-coverage) before expanding to the remaining layouts, dtypes and sizes.

Keep the verified control for later comparisons. A per-index scalar/VF dot can diagnose values, but thin retained axes or outer indirection do not establish that it should replace the regular product. Compare complete calls on representative geometry before expanding coverage: a resident score row, an online recurrence and GM materialization have different work and reuse costs. A fused kernel is not automatically faster, and a slow serial fusion does not establish that the matrix path is unavailable.

### Other starting contracts and later residency choices

| Contract or interface to inspect | Runnable entry and scope |
| --- | --- |
| Dense forward chain with inspectable GM scores and coefficients, including short or partial M tiles | [Materialized matrix-chain control](../../examples/algorithms/mm_softmax_mm_materialized/): independent phase grids, runtime prefix validity and FP16/BF16 conversion; no head mapping, positional mask or split features |
| Split score features and a single KV head (MLA) | [Materialized MLA](../../examples/algorithms/mla_bounded/kernel.py): BSND/BNSD, right-aligned causal visibility, independent V and explicit BF16 tiny-value preservation |
| Short resident key axis | [Fused MLA](../../examples/algorithms/mla_bounded/_kernels/fused.py): FP16, resident coefficients, serial payload-column consumption and causal query intervals; inspect this path before stopping at materialization |
| Single query with a long key axis and wide payload | [Streamed MLA](../../examples/algorithms/mla_bounded/_kernels/decode.py): FP16, independent V and local FP32 running state; its visibility rule does not implement a multi-query causal mask |

Each entry's header and tests own its shape, range, resource and toolchain limits. Transport mechanics do not transfer a numerical proof to another dtype; validate its range, coefficient conversion and final output at the original tolerance.

## Choose residency for the actual reduction length

The materialized anchor is a correctness control, not the end of an optimization task. When both products spend substantial time moving data, count their actual requested bytes separately: original operands, repeated operand loads, score writes/reads and coefficient writes/reads. Increasing query reuse may save only a small part of that total while leaving the large key/value streams and intermediates unchanged. High MTE2 activity does not establish that the current stage boundaries are optimal.

When a complete score row fits with the other live tiles, a resident-row path can normalize once and accumulate coefficient panels in Cube, avoiding per-panel Vector output updates. Compare it with the streamed path instead of choosing it from capacity alone: retaining all scores can constrain tile sizes, prolong operand lifetimes or serialize the two contractions. Conversely, an online implementation with slow runtime VF branches is not a fair measure of recurrence cost. Inspect [VF loop lowering](../api/vf.md#keep-runtime-dispatch-out-of-vf-loops), then compare complete calls. Use [NZ publication](../api/crosscore-sync.md) and preserve independent V unless the task explicitly guarantees the prefix relationship.

Capacity is a peak-liveness constraint. The first product's key staging may be dead before value staging is filled. A native `pl.reinterpret` alias can reuse that allocation with a different shape when its last reader has completed; do not sum sequential lifetimes as if all operands must coexist. Preserve physical capacity, layout, group identity and synchronization through [native allocation views](../api/memory-allocation.md), and inspect the actual stage order after enabling preload. Start with a correct serial producer→normalizer→consumer slice, then compare shallow overlap with the slot depths required by that schedule. A serial fused result does not close the neighborhood of the fused implementation.

For a long reduced axis, the complete score row may not fit with the other live state. Compare streaming bounded coefficient panels and retaining a weighted numerator/denominator locally against materialization. Choose panel-local normalization, the [exponential reference](#choose-the-exponential-reference-and-state), and any [partial merge](#merge-partitions-of-the-reduced-axis) together: casting tiny unnormalized exponentials to FP16 can erase them even though their normalized probabilities are representable. Include running-state updates, payload-width work, rereads and merge traffic in the comparison. A short-axis win does not establish the best long-axis mapping.

For the streamed path, [online softmax](online-softmax.md) owns recurrence, score orientation, shared L0 buffering and state-slot lifetime across work items.

## Factor through the consumer

Name retained/reduced axes and consumers for each intermediate. A maximum, denominator or normalized coefficient belongs to the score row and mask, not to its final payload channel. Sharing only the maximum and denominator still repeats nonlinear work if each channel reconstructs coefficients. Compare materializing shared coefficients with their traffic and launches; also check whether the shared producer has become one long serial thread per row.

For indexed or segmented payloads, apply [output ownership](multi-output-reductions.md#map-each-output-independently) before selecting the traversal. Input gather, coefficient reuse and output reduction are separate decisions. Preserve a measured owner-accumulation control when considering atomic scatter; sharing weights does not require sharing output writers. Split long reductions when needed, then [retune the launch](simt-first.md#launch-variants) for the resulting work units.

### Count work independently in each phase

For the dense chain, label each phase's matrix axes before choosing one ownership scheme for the whole call:

| Phase | Independent output axes | Reduction that must complete |
| --- | --- | --- |
| `S = Q @ K.T` | Query rows and key columns | Input feature dimension |
| `P = normalize(S)` | Score rows | Valid keys in each row |
| `O = P @ V` | Query rows and payload channels | Valid keys |

Count tiles across both retained axes using the [complete GEMM grid](../api/gemm.md#count-the-complete-output-grid-before-assigning-workers). Few batches, KV heads or query-row tiles do not imply few independent score or output tiles. A materialized score path can distribute key-column tiles in its first product, and a completed coefficient matrix can distribute payload-channel tiles in its last product. Neither product needs to inherit the normalizer's row-only grid.

Fusion changes these choices because coefficients and row state can be local to one execution group. Do not split a streamed key reduction as if it were an independent output axis: use [normalized partials and their merge](#merge-partitions-of-the-reduced-axis), including scratch and merge cost. Compare reuse, available work and full-call cost before fixing the fused or materialized mapping; a launch-count sweep cannot repair a grid with too few owners. The [bounded chain](#connect-the-bounded-chain) still establishes the local interfaces.

## Choose a native paradigm per phase

### Match indexed transfer granularity to the payload

Distinguish an index selecting a contiguous page/block from an index selecting just one short payload row. Count scalar index reads, DMA commands and local-buffer handovers as well as bytes. A serial loop issuing one tiny load/store per index can dominate a call whose subsequent dense products are already fast; high MTE activity does not by itself establish a bandwidth limit. Measure the gather phase separately and inspect the actual transfer width before tuning the contractions.

Keep the direct-transfer control while comparing [SIMT panel packing](simt-payload.md#compare-flat-and-grouped-mappings), including [wider transport words](simt-payload.md#use-wider-words-for-pure-transport) for pure copies with suitable alignment and extent. Establish the gathered values first, then connect the same verified dense consumer. Include packing launches, scratch writes/reads and repeated packing in complete-call time; bound scratch and amortize it across consumers sharing the selector. Packing can lose for small workloads or already-large contiguous blocks, so it is a measured alternative, not a required extra kernel.

Preserve the mapping from panel positions to original indices: positional masks and output ownership use the original coordinates. Selectors may differ across queries or heads, so equal shapes do not imply a reusable panel. Contractually equal operands or prefix views may share gathered storage; preserve the backing row stride when a consumer uses only a narrower feature prefix. Compare this byte-saving change separately from transfer granularity and ownership.

### Map the compute phases

| Phase | Route to evaluate | Remaining contract |
| --- | --- | --- |
| Regular dot products with shared input rows/columns | [Cube GEMM](gemm-desugar.md); [split reduction-axis inputs](../api/gemm.md#inputs-split-along-the-reduction-axis) when the reference concatenates feature components; [VF reduction](vf-row-reduce.md) as a control for thin groups | Group/head mapping, physical shapes, tiling and input reuse |
| Row normalization and statistics | [VF row reduction](vf-row-reduce.md) and [small-result transport](vf-small-results.md) | Runtime validity, empty rows, precision and output ownership |
| Weighted payload contraction | Cube if dtype/layout/error budget fit; otherwise a native consumer of shared coefficients | Coefficient rounding, neutral reduction padding and payload layout |

For a thin retained dimension, first read the [short-query/grouped geometry route](online-softmax.md#decode--very-thin-m-gqa). Match an anchor's computation shape and physical interfaces before its mask label: a non-causal thin-group implementation can be a better starting mechanism than a causal wide-query implementation, provided the new validity contract is implemented and tested separately.

Choose one repeated dense product as the first compute probe: map its retained/reduced axes to a bounded block, run the [short GEMM route](gemm-desugar.md#the-mapping) or [VF row reduction](vf-row-reduce.md), and compare it to the mathematical values before expanding the full kernel. Runtime indirection may select a contiguous physical block. Establish this path or a concrete failed capability probe before spending the optimization budget on scalar SIMT algebra, partition counts or launch sweeps. A working scalar kernel and a faster scalar variant do not complete that paradigm comparison.

Include physical-row padding, independent work-item counts and the consumer in the comparison. SIMT remains a candidate for small or irregular work; justify it with complete-call measurements or the specific installed-interface failure. Familiarity with direct-GM scalar code, outer indirection, thin groups and high SIMT pipe activity alone do not establish that it is efficient.

Check products separately: coefficient precision at one consumer does not rule out Cube in another phase. A GM cut point can help isolate a product's numerical result. Once the products work, follow the [complete on-chip slice](#connect-the-bounded-chain) before choosing the full call's kernel boundaries.

Keep representation and compute mapping as separate experiments. Hold the producer mapping fixed when testing an algebraic change such as a fixed exponential reference, then compare other mappings with the same validity and numerical contract. Unrolling and worker sweeps do not replace the regular-block comparison.

Plan local tiles rather than placing a whole logical axis in L0. A long output axis can be an outer tile loop. For an oppositely stored operand, resolve one [transpose view](../../examples/matmul/transpose_reinterpret/) with the [GEMM mapping](gemm-desugar.md), then extend the grid. Verify tail/transpose/split combinations instead of treating a failed combination as absence of matrix support.

## Connect the bounded chain

Computed normalization coefficients are [transformed Cube operands](../api/gemm.md#transformed-operands-format-residency-and-repeated-work), even when the original inputs needed no conversion. After choosing residency and phase ownership, connect the verified products and normalizer through [incremental construction](../workflows/authoring.md#build-an-unfamiliar-dataflow-incrementally). The complete slice must preserve the coefficients' validity domain, dtype/rounding and final public output.

Keep received scores [read-only in their consumer](../api/crosscore-sync.md#postprocess-a-shared-input-without-changing-its-writer), applying masks in VF registers; editing the shared tile changes the handover's writer topology.

For this chain, use `preload=0` with [visible stage slot selection](../api/crosscore-sync.md#keep-slot-selection-visible-to-the-stage-scanner) and [named GEMM operands](../api/gemm.md#division-of-responsibilities-between-the-caller-and-the-sugar). Both AIVs and their AIC must visit the same logical item sequence under the [mixed-group indexing contract](../api/crosscore-sync.md#a-mixed-kernels-block_dim-is-a-group-count-launch-semantics--index-semantics). Verify two groups with distinct data and more than one item per group before overlap; correct coefficient layout alone does not establish matching participation.

## Stream reverse contractions with bounded state

Derive the reverse dependency graph and label each product's retained axes, reduced axis and operand types before choosing writers. Gradients of different operands can reduce opposite axes of the same pairwise relation. Apply [independent output ownership](multi-output-reductions.md#map-each-output-independently): separate tile-owner traversals may recompute a shared block while avoiding conflicting writes; sharing one scalar traversal is only one candidate. Keep row statistics shared across output channels.

A ban on storing the complete pairwise matrix constrains residency, not the use of matrix instructions. Reconstruct one bounded coefficient tile from original operands and compact completed row statistics, consume it, then reuse its storage. Distinguish statistics that need a complete reduction from quantities that can be accumulated tile by tile; consumers must wait for the statistics they require. Plan persistent row state, local product tiles and any gradient partials by peak live bytes. [Streaming forward attention](../../examples/algorithms/flash_attention_mha/) demonstrates tile residency, while the reverse pass needs its own writer and dependency proof.

Evaluate raw-input products and derived-coefficient products independently. Their dtype and numerical requirements can differ. Preserve the task's mathematical reference when considering coefficient narrowing, and propagate the [exponential range proof](../api/numerics.md#4-fixed-references-in-bounded-exponential-normalization) through the actual consumer dtype. A reference safe for FP32 accumulation may overflow or lose significant values after narrowing. Keep [layout, extent and producer/consumer handovers](../api/gemm.md#transformed-operands-format-residency-and-repeated-work) explicit when replacing a scalar phase.

## Choose validity traversal and state lifetime

### Preserve coordinates when folding retained axes

Combining several retained axes into matrix rows can share a contraction operand. Keep an explicit inverse map from each packed row to the original coordinates, and use it for validity as well as input/output addressing. A row number that now includes a head, group or batch index is not automatically a query position. Derive each row's valid reduction domain in the original coordinates before translating it into block/lane predicates; layout conversion alone does not implement that predicate. Degenerate examples can hide this distinction: a one-position domain may make two different mask modes equivalent, while extending that axis activates a previously invisible contract branch.

Batch rows sharing a validity descriptor when the physical view permits it. Computing that descriptor once in the outer scalar kernel and loading along the corresponding axis can remove per-row quotient, remainder or wraparound state from VF. A strided head-axis batch can preserve a head-major layout without a GM transpose. Prove that the batch stays within one descriptor's domain; the [bounded MLA tests](../../examples/algorithms/mla_bounded/test.py) include non-power-of-two query lengths so a folded-row shortcut cannot pass only through degenerate masks.

### Visit the valid domain

If runtime predicates intersect contiguous intervals, derive lower/upper bounds once per row with consistent index types, handle the empty interval, then visit that interval. A full padded-axis scan with per-element predicates performs different work. Masks with holes still need their actual participation rule.

For a monotone causal prefix, propagate visibility into both contractions, not only the softmax mask. A row block needs keys only through its largest visible key, rounded up to the chosen key tile; retain row masks inside the final intersecting tile. Group rows with compatible bounds in original query coordinates. For static shapes, a small number of query intervals can select smaller native reduction shapes in separate launches; count every launch in the complete-call result and prove disjoint, complete output coverage. For runtime bounds, use a bounded device loop where supported. Skipping a score producer is insufficient if a later consumer still reads those uninitialized score lanes: either exclude them from every downstream read or initialize them explicitly and include that cost.

### Choose the exponential reference and state

For exponential normalization with bounded scores, first evaluate the [fixed-reference range proof](../api/numerics.md#4-fixed-references-in-bounded-exponential-normalization). A feasible common reference can remove maximum discovery and online rescaling while preserving the mathematical result. When a data-dependent reference is needed and scores are materialized, retain a two-pass normalization baseline before adding an [online recurrence](online-softmax.md): rescaling adds nonlinear work whose streaming benefit may be absent. When capacity or fusion calls for streamed state, compare that schedule with the same valid domain. Whole-block reuse also trades against independent work items: splitting a retained axis exposes parallelism but may reload shared inputs.

### Merge partitions of the reduced axis

When a few long rows leave workers idle, use the [runtime-segment partitioning criteria](multi-output-reductions.md#partition-unequal-runtime-reductions) for ownership, scratch, empty work and complete-call cost. The normalized aggregate determines the merge: partials with one fixed reference sum denominators and weighted numerators directly. For local references `(m_j, l_j, u_j)`, merge with `m=max(m_j)`, `l=sum(exp(m_j-m)*l_j)` and `u=sum(exp(m_j-m)*u_j)`, then finalize once. Empty partials need explicit neutral state; an all-empty row must bypass undefined `-inf - -inf`.

## Make producer/consumer interfaces explicit

- Record logical extent, physical shape/row pitch, dtype and rounding at every edge. One product's output axis becomes the next product's reduction axis; a harmless output tail can then contaminate every result.
- Pad physical intermediates when required by [GEMM extents](../api/gemm.md#effective-dimensions-and-splitting) or [data movement](../api/datamove.md), but retain logical validity and public tensor addressing. Every consumed padded reduction lane must be neutral or excluded; an output mask cannot repair an earlier contaminated sum.
- Account for narrower coefficient types against the original mathematical reference and tolerance. Do not make the reference adopt candidate-specific rounding unless that rounding is explicitly part of the task contract.
- Define empty aggregates explicitly. `-inf - -inf` and `0/0` are not empty-row semantics; retain validity or a safe internal normalization value without changing valid rows.
- Produce LSE or other row statistics from the completed shared aggregate and give each compact output a legal store path and unique owner.

## Separate validity errors from roundoff

Before blaming coefficient narrowing or changing a contraction's precision, run controls with analytically simple outputs through the current complete call:

- Use zero/equal scores with one nonzero payload at a validity boundary. Check a row that excludes it and a row that includes it; the latter must have a nonzero expected result. Include distinct original positions after row packing. This distinguishes a missing/shifted predicate from rounding without reproducing the implementation's reduction order.
- Change only an excluded key or payload within its permitted range, keeping the validity domain and visible inputs fixed, and check that the affected row's mathematical output is unchanged at the original tolerance. Other rows may legitimately include that position. Include an excluded score larger than every valid score and nonzero valid payloads: letting masked values set an exponential reference can erase valid coefficients when they narrow, even if the final weights are masked.

Trace the same valid domain through maximum/reference selection, denominator and weighted numerator. A small mismatch on a long reduction can be a missing or extra contribution; its magnitude alone does not identify finite precision as the cause. After the semantic controls pass, compare precision or summation choices with indices, predicates and layouts held fixed, following [numerical diagnosis](../api/numerics.md#1-cancellation-in-chained-matmul-golden-is-not-the-scale-for-rtol).

For a new boundary predicate, start with [register selection](../api/vf.md#25-combining-masks-with-existing-comparisons-and-selection) on the loaded score chunk before its reductions. A one-lane write into the score tile introduces separate alignment and memory-ordering obligations; a participation mask alone does not satisfy them. Keep a working contraction/packing path while introducing the predicate, and run the controls above before combining it with a new layout, precision or schedule.

Trace the mask through the exponential explicitly. Multiplying an invalid score by zero before `exp` produces weight `1`, not an excluded contribution. Select a safe score/reference using validity, and make invalid weights zero before both the denominator and weighted product. The [materialized control tests](../../examples/algorithms/mm_softmax_mm_materialized/test.py) perturb excluded payloads while retaining nonzero valid outputs, and test constant payloads at attainable positive/negative score endpoints. Reuse those controls with the task's actual mask and range; the [fixed-reference proof](../api/numerics.md#4-fixed-references-in-bounded-exponential-normalization) must include coefficient narrowing.

## Validation and comparisons

Test all-empty/mixed-empty rows, singleton valid sets, mask boundaries, distinct heads/groups, physical tile tails and poisoned repeats. Structured scores should expose wrong grouping and reductions. Keep guards around the original public layouts.

For a streamed MIX adaptation, add two reduction chunks and two distinguishable work items per execution group, then two groups, before overlap. Check state reset/carry and both AIV regions at each step. Include unequal score/output feature widths and every public layout/dtype branch through the wrapper; a single-item/single-chunk pass cannot cover those branches. Keep these cases in the existing test file.

For a reverse pass, check the reference with small directional finite differences and use mathematical invariances to design gradient cases. Constant payloads or identical operands can create zero derivatives despite nonzero internal terms; concentrated weights and interior-mask impulses distinguish numerical cancellation from omitted or misassigned contributions. Recheck such cases after algebraic factoring or coefficient narrowing at the original tolerance.

Use the [optimization workflow](../workflows/optimization.md) and [complete-call timer](../verification/native-timing.md). Compare shared-state factoring, matrix-phase alternatives and query partitioning with matched validity and all added stages included. Revisit a remaining SIMT phase's [launch configuration](simt-first.md#launch-variants) after the mapping changes.

| Interface to inspect | Runnable example |
| --- | --- |
| A whole score row resident on chip, with two products and VF normalization | [mm_softmax_mm](../../examples/algorithms/mm_softmax_mm/) |
| Reverse contractions with bounded row state and distinct output owners | [Causal attention backward](../../examples/algorithms/causal_attention_backward/) |
| Runtime intervals and shared row statistics | [window_gqa_attention](../../examples/algorithms/window_gqa_attention/) |
| Indexed segments, shared weights and owned partials | [CSR softmax aggregation](../../examples/algorithms/csr_softmax_aggregate/) |
| Indirect physical blocks and multi-page partials | [Paged GQA decode](../../examples/algorithms/paged_gqa_decode/) |

Their configurations and size limits are example boundaries, not general dispatch rules.
