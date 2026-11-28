# Pattern: VF row-wise reduce → scalar chain → broadcast

Use with the [VF API contracts](../api/vf.md); the runnable examples below define their own verified scope.

## Applies when

Each output row depends only on its own input row through one or more full-row reductions (max / sum / mean-of-squares) followed by a per-row scalar chain and a broadcast back over the row. FP32/FP16. Row width need not be a multiple of the register width. Rows split freely across cores.

**Register width scales with the tile dtype** (256-byte physical register): FP32 source = 64 lanes per `load_align`, FP16 source = **128** — the `VL=64` constant in the worked fillings is FP32-specific. For FP16 source data needing FP32 math, the order-preserving widening is **`load_align(..., dist=pl.LoadDist.UNPK)`** (64 contiguous FP16 into the low half of 64 b32 containers, source order kept) followed by `astype(dtype=FP32, layout=ZERO)`. Before applying the 64-lane computation, derive the widening predicate from the [physical-container mask contract](../api/vf.md#213-the-fp32-path-for-fp16-source-data-order-preserving-widening-with-unpk-required-reading): an FP16 prefix of the logical count masks off half the expanded values. Plain load + widening `astype` instead silently even/odd-subsamples the source. Worked example: `examples/algorithms/rmsnorm_gemm`.

<!-- kernel-query: {"id":"vf-row-reduction-costs","aliases":["row reduction tails","row normalization work","residual normalization tails","逐行归一化尾部"],"term_groups":[["rmsnorm","rms norm","row reduction","行归约"],["tail","padding","residual","尾","填充","残差"]],"example":"algorithms/layernorm/kernel.py","when":"Mapping independent row reductions, rectangular tails and dependent residual/normalization phases. The linked LayerNorm is a native API/dataflow anchor, not the RMSNorm formula.","missing_facts":["Preserve the actual formula/dtype, compare reduction frequency, padding and batching, trace input/state/output lifetimes, and measure the declared tail and aligned regimes."]} -->
## Plan the row work before choosing a template

Keep the task's formula and rounding: LayerNorm's mean subtraction and a quantizer's required FP16 intermediate do not belong to a plain FP32 residual RMSNorm. Use their native declarations as mechanism references only when those mathematical differences are explicit.

Inspect these sites in the first correct source. Choose the largest plausible removable cost, compare one structural control at fixed workers, then retune launch configuration. A worker sweep alone does not test these costs.

| Source structure | First comparison and boundary |
| --- | --- |
| Horizontal `reduce_sum` inside the feature-block loop | [Accumulate lanes across blocks, then reduce](#choose-the-horizontal-reduction-frequency). For `K` blocks this can replace `K` horizontal reductions by one. Preserve tail neutrality and validate the changed summation order with the original tolerance. |
| A separate load→VF→store sequence for every short row | [Batch independent rows](#batch-independent-rows-even-when-each-row-fits) in a tile/VF call. Include a modest coarser batch; account for capacity, partial batches and the busiest worker's actual rows. |
| One scalar `setval` per padded element | Compare [bulk `fillpad`](../api/datamove.md#3-fillpad-requires-three-pieces-in-three-places-omitting-one-leaves-valid-data-surrounded-by-garbage) at the original valid extent. Masked consumers may need no fill; the sentinel must remain neutral through the actual arithmetic. |
| The input tile also carries retained state and the final output | When capacity permits, compare [independent input and state/output buffers](../api/memory-allocation.md#independent-buffers-and-slot-depth) at the same batch and workers. In-place residual/output updates keep the input slot live through apply and GM store; its next load must wait. `auto_mutex` preserves that dependency. A larger batch or a fused VF does not remove it; account for the extra UB before selecting the smaller tile count. |
| The apply phase reloads or recomputes the same residual/intermediate | Compare [retaining it in UB](#choose-where-dependent-phases-share-state) at fixed batching. Separate producer/consumer VF calls can remain in one native task; fewer source calls do not establish lower cost. |
| A VF reloads bytes it previously stored | Resolve the [same-VF memory dependency](../api/vf.md#212-multiple-memory-accesses-to-the-same-tile-inside-vf-have-no-ordering-guarantee): use the required VF ordering or separate TileGroup-tracked VF calls. Proved disjoint physical accesses need no store-to-load barrier. Passing numerics and `auto_mutex` do not prove intra-VF ordering. |

Time the declared tail/small-work and aligned/large-work regimes before freezing a selector. Numerical coverage of a shape does not supply its timing. Keep the task's arithmetic, rounding and acceptance unchanged; a rejected structural control remains useful evidence, not a requirement to adopt it.

The sections below own the detailed ownership, capacity, padding and numerical conditions. The [LayerNorm example](../../examples/algorithms/layernorm/) supplies an executable row/padding API anchor; [residual RMS quantization](../../examples/algorithms/residual_rms_quant/) owns its different FP16-rounding and integer-output contract.

## Logical dataflow

GM row block → UB tile (padded width) → per row: reduce(s) → scalar chain (sqrt/log/div…) → broadcast-apply over the row → store back only the valid columns.

### Batch independent rows even when each row fits

A full reduction row can still be a small DMA. When many short rows produce repeated load→VF→store calls and transfer/reuse waits, compare loading a `[row_batch, padded_columns]` tile and looping over its rows **inside one VF call**. This changes transfer/call granularity without combining rows' reductions. Initialize maximum, sum and other state independently for each row; the [basic row-softmax example](../../examples/basic_api/vf/vf_softmax/kernel.py) demonstrates this row loop. A resident-row correctness anchor's batch of one is not a tuned default.

Keep metadata attached to its logical owner. Rows may reuse one loaded selector or mask only when their validity domains really match; stop a batch at such a domain boundary or load distinct metadata per row. Equal row widths do not establish equal selectors. Include coordinate-distinct rows, differing masks and partial final batches in validation.

Choose row batch size separately from column/register width and count the busiest worker's actual row batches: a larger batch can increase the last worker's work or leave workers idle. The [local buffering contract](../api/memory-allocation.md#independent-buffers-and-slot-depth) owns slot-depth comparisons, capacity accounting and ring-wrap validation; include every row-batched input, output and metadata tile in that plan.

Use the [configuration comparison procedure](../workflows/optimization.md#choose-configuration-comparisons) to test batch size together with launch count and validate a metadata selector on additional shapes. Work-item coverage is one screening quantity; retain legal batching candidates on both sides of the estimated concurrency/reuse tradeoff.

Batching can add metadata DMA and statistic reloads even while removing many row-sized calls. Keep a row-at-a-time control when those added costs are not amortized, and compare complete calls at matched workers before selecting a shape regime. The [grouped GEMM quantization example](../../examples/algorithms/grouped_gemm_swiglu_quant/) exposes row and batch controls, guards each worker's actual consumers, and uses a block-wide phase boundary. Its batch path normalizes by a bounded row maximum before the integer-range factor; the [normalized-ratio contract](../api/numerics.md#2-small-magnitudes-in-normalized-ratios) still requires independent numerical tests when that arithmetic order changes.

The [pairwise squared-distance example](../../examples/algorithms/pairwise_squared_distance/) compares one-row and batched transfers for independent reductions sharing an input. It packs scalar outputs before GM stores and checks available work groups before choosing the larger batch.

### Reconstruct very short rows in registers

When a row uses only a small part of a register, compare loading a contiguous flat span and restoring row/feature organization in registers. Choose a group of `R` rows of width `D`, with `R` fitting the compute register's lane capacity. Source element `(r,d)` has group-relative flat offset `r*D+d` for `0<=r<R, 0<=d<D`; construct feature registers whose lane `r` holds that row's feature `d`. The row reduction then combines feature registers lane-wise, and output reconstruction maps `(r,d)` back to `r*D+d`. This gives one independent row per lane, a different dataflow from the padded two-dimensional row loop above. Prove that the source/output views and actual strides permit those flat transfers; a valid shape does not remove physical padding.

For `D=2`, compare a supported load-port/deinterleave arrangement; for `D=3..8`, compare register gathers that collect feature columns from a flat group. These are design candidates, not fixed width bands or newly target-validated kernels. The [lane-layout contracts](../api/vf.md#28-vf_lane_layout-two-points-spanning-examples), [gather forms](../api/vf.md#31-gather-has-two-forms-out-of-bounds-behavior-is-defined-for-only-one) and [narrow-source widening](../api/vf.md#213-the-fp32-path-for-fp16-source-data-order-preserving-widening-with-unpk-required-reading) own supported operations and predicates; do not treat a widening cast as a portable deinterleave recipe.

For multiple source registers, derive both the source-register number and its lane from `r*D+d`: register-gather wrapping cannot select another register. Check rows crossing register boundaries, partial row groups, all feature masks and final output guards with coordinate-distinct inputs. Keep invalid lanes out of every row's arithmetic, and make all issued addresses legal before masking results. Compare complete calls with the padded-row control, including reconstruction, padding, register pressure, local slot depth and worker utilization; a smaller DMA count alone does not select the winner.

### Choose where dependent phases share state

The [gated outer-state example](../../examples/algorithms/gated_outer_state/) keeps a recurrent matrix in UB across a complete sequence, with only per-step outputs and the final state stored to GM. Its source declares the live storage, reset boundary and numerical scope; it is a correctness anchor without a speed claim.

A mathematical phase is not a native-kernel boundary. First identify the owner of each row and budget the peak simultaneously live tiles, including padding and slots. If that owner can perform all dependent phases within the capacity limit, compare a resident-row native kernel before choosing separate global stages. Values can stay in registers within one VF procedure, or in explicit UB tiles between VF procedures called by the same JIT kernel; [VF scope](../api/constraints.md#simd-vf-functions-communicate-through-tile-parameters) does not require GM intermediates.

Separate native kernels are justified when a consumer needs other workers' completed results, the phase mappings differ, capacity forces streaming/materialization, or a complete-call measurement favors that split. State the actual boundary reason. Count input reloads, intermediate GM writes/reads and task overhead in the comparison. Keep worker count fixed when comparing these dataflows; a worker sweep can improve a staged implementation while leaving its avoidable phase-boundary cost untouched. After choosing a dataflow, retune its worker count.

If a fused phase reads the tile's old contents, first inspect [same-VF memory dependencies](../api/vf.md#212-multiple-memory-accesses-to-the-same-tile-inside-vf-have-no-ordering-guarantee). `auto_mutex` orders calls through tile groups; it does not order a store and reload inside one VF procedure. Compare the appropriate VF barrier, or separate producer/consumer VF procedures using the same tile group within one JIT kernel, before attributing the problem to fusion or requiring a GM round trip. Changing to separate native kernels may hide the missing local ordering without establishing that the split is necessary.

For independent rows in one UB block, compare a block-wide phase boundary with a barrier inside every row: first write all rows' intermediates and statistics, issue the required store-to-load barrier, then consume the rows. Count the additional live statistic storage and keep the row batch fixed when attributing a gain to barrier placement. [Bounded MLA](../../examples/algorithms/mla_bounded/kernel.py) exposes this organization; removing a barrier without moving the dependent reads is not the same experiment.

When a row statistic becomes a divisor for a scale-normalized result, validate the permitted input magnitudes as well as shapes. Tiny numerator and denominator errors can become large ratio errors even if the statistic passes an absolute tolerance. Use the [normalized-ratio dynamic-range method](../api/numerics.md#2-small-magnitudes-in-normalized-ratios) when the contract permits very small values; preserve required intermediate rounding.

If any per-row output is a value, index, count or statistic, read [patterns/vf-small-results.md](vf-small-results.md) for its physical output shape, write path and dtype. This includes a compact side output alongside a full-sized tensor: each output needs its own transport plan. These decisions are independent of the reduction's tie rule or metadata representation.

For runtime row metadata (length, segment bound, scale), `pl.getval(tensor, linear_element_offset)` reads a scalar directly from GM in the outer SIMD kernel; pass that scalar as a VF argument. This is a simple starting point for isolated control values. For batches of reused metadata, compare it with the [UB batch-and-broadcast route](vf-outer-reduce.md#metadata-and-invariant-inputs). Signature and integer promotion: official getval contract (`pypto/docs/zh/api/pro_api/SIMD-API/transpose_and_element_access/getval.md`).

For uneven row counts, define ownership together with the host launch. In a pure vector kernel, `pl.range(pl.get_block_idx(), ROWS, pl.get_block_num())` gives every row one owner. Launch a bounded `WORKERS` with `kernel[None, WORKERS](...)`; pure Vector has `get_subblock_num()==1`, not two. Start with one worker, then test several workers and a row count that does not divide them. Mixed cube/vector kernels use a different index mapping ([primer](../workflows/getting-started.md)).

<!-- kernel-query: {"id":"small-matrix-lane-map","aliases":["small matrix lane mapping","paired column epsilon","register resident Sinkhorn"],"when":"A batch of small matrices repeatedly normalizes rows and columns or keeps dependent iterations in registers.","missing_facts":["Map matrix/row/column to register, lane and physical predicate; preserve reduction groups, combine paired partials before epsilon, and compare the complete iteration chain."]} -->
### Small matrices: map lanes and dependent normalizations together

For many independent small matrices, compare a lane-owned matrix with entries kept in separate registers against a layout placing a reduction axis across lane groups. In the first design, columns/rows become register choices and reductions become short register chains; in the second, they use explicit within-group exchanges/reductions. A third candidate can pair two columns in the low/high register halves. Write `(matrix, row, column) → (register, lane, physical predicate bit)` for every stage; changing the layout changes the reduction participants and the broadcast map, not the mathematical axes.

If two halves hold partial sums for the same denominator, combine them before applying its nonlinear step or adding epsilon. `(s0 + s1) + eps` differs from `(s0 + eps) + (s1 + eps)`. Enable both halves in a paired predicate, keep neighboring matrices outside the group, and define incomplete groups independently. The two-half trick is a mapping candidate, not a claim that a particular Pro reduction/gather API implements it automatically; check the selected port and exchange contracts.

Budget live registers and explicit UB scratch across all dependent iterations. A register-resident matrix can remove repeated UB traffic while an expanded implementation can spill or increase scheduling cost. Test nonsymmetric matrices, dimensions around the packing boundary, epsilon-dominated rows, tails and repeated calls; validate every normalization step and the final output. Compare the complete resident sequence with the existing tiled control rather than extrapolating one reduction's timing.

## When a full row does not fit

The resident-row skeleton below assumes all live tiles fit in UB. For a longer row, budget the input/output chunks, slot multiplicity and persistent state together, then stream GM chunks through a bounded workspace. Choose a **transfer chunk** separately from the VF register width: one DMA and VF call per register can create many small transfers and synchronization handovers. Process several register blocks inside each VF call when capacity permits; compare chunk sizes by measurement rather than treating a particular size as optimal.

Define the aggregate's initialization, per-chunk merge and finalization before allocating an array of partials. A mergeable reduction can carry a fixed-size aggregate instead; a VF register lives within its VF call, so state crossing calls must be stored in an explicit UB tile and protected by the tile-group synchronization. Decide which outputs require final aggregate values: those outputs need a later input pass or retained intermediates. Empty valid sets, tail masks and output guards must keep the same semantics across chunk sizes. Stable exponential normalization has a mergeable max/denominator state; see the [online recurrence](online-softmax.md#the-recurrence-and-where-state-lives), independently of its matmul consumer.

For a long row whose output is the full softmax probability vector, compare rereading the input with retaining provisional exponentials when the storage contract permits it. For each nonempty chunk b, store `e_b = exp(x_b - m_b)` and its local `(m_b, l_b = sum(e_b))`. After merging the global `(m, l)`, finalize that chunk as `e_b * exp(m_b - m) / l`. A global max/denominator alone cannot reconstruct all outputs without either input rereads or per-element state. Keep the chunk statistics with the stored values; handle masked/empty chunks using the [online empty-domain rules](online-softmax.md), rather than evaluating an undefined empty maximum difference.

This is a dataflow candidate, not a new validated kernel. Budget every provisional store/reload, statistic, final write, exponential and native launch. A single input read need not minimize total GM traffic. Using low-precision output storage as scratch introduces an additional rounding boundary, and aliasing it with an input needs its own lifetime proof. Compare complete calls against the reread control at the original numerical criterion.

### Choose the horizontal-reduction frequency

Reducing every loaded register to one lane creates a serial scalar dependency, especially when each partial also updates compensated state. Compare accumulating independent lane-wise partials over several register blocks, then reducing lanes at a bounded interval or finalization. Multiple independent accumulator registers can shorten dependency chains further, at a register-pressure cost. Transfer size and horizontal-reduction interval are separate choices: larger DMA chunks alone leave a per-register reduction chain intact.

This changes summation order and each lane's accumulation length. Bound intermediate magnitudes, choose reset/merge intervals and retain compensation where the error budget requires it; independent lanes are not a numerical proof. For centered statistics, consult [centering and accumulation order](../api/numerics.md#3-centering-and-accumulation-order). Compare the complete implementation at fixed workers and transfer size before retuning those parameters, using the same input and unchanged tolerance.

## Physical invariants

- **Pad the tile width to a multiple of 64 lanes**; the real row width rides in the valid shape. Two board-verified tail treatments:
  - *Sentinel fill* — declare the destination `TileType.pad`, set the source valid shape, then call `pl.fillpad(..., mode=...)` ([three-part contract](../api/datamove.md#3-fillpad-requires-three-pieces-in-three-places-omitting-one-leaves-valid-data-surrounded-by-garbage)). `pad` is not a `fillpad` call keyword. The sentinel must be **absorbing for every step** of the computation; a max sentinel alone does not establish an empty-set or later-sum contract.
  - *Mask* — `vf.create_mask`/`update_mask` limiting active lanes in the tail register
    block; official `reduce_sum` semantics guarantee unselected lanes are zeroed
    (RMSNorm form).
Pick sentinel when one value is absorbing for the whole chain; pick masks when no single sentinel survives every step.
- **Write back only the real columns**: choose a fixed `valid_shape=[r, c]` without later setters, or dynamic `valid_shape=[-1, -1]` with `pl.set_validshape` before the relevant load/store. Do not mix the two forms ([shape contract](../api/datamove.md#fixed-and-dynamic-valid-regions)).
- **Multi-core split by rows**: `pl.get_block_idx() * ROWS_PER_CORE` directly in the `pl.load`/`pl.store` offsets (ST precedent: `gather/test_gather_block_variables.py`, `multicore_add_kernel`). Launch with the grid form; even splits first — an uneven row count is a *contract* decision, not a loop afterthought.
- **Name every VF result before reuse**; give independent states separate registers, and use VF writes in every branch/loop arm, including `vf.move` for copies, rather than ordinary `dst = src` rebinding ([copy and alias contract](../api/vf.md#vf-register-aliasing-and-copying)). A result-producing VF call may not nest as another call's argument.
- Reduction/mask/squeeze constraint details (UnalignReg budget, mask state, squeeze alternation) stay owned by [api/vf.md](../api/vf.md) §1–§4 and are gated by `check.py rules`.

## Minimal skeleton

```python
@pl.jit(auto_mutex=True)
def row_op(x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]):
    with pl.section_vector():
        g = pl.make_tile_group(
            type=pl.TileType(shape=[ROWS_PER_CORE, PADDED_COLS], dtype=FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=..., mutex_ids=[...])
        tile = g.next()
        base = pl.get_block_idx() * ROWS_PER_CORE
        pl.load(tile, x, [base, 0])          # + fillpad / validshape per tail choice
        _rows(tile)                          # @pl.vector_function body below
        pl.set_validshape(tile, [ROWS_PER_CORE, COLS])
        pl.store(out, tile, [base, 0])

@pl.vector_function
def _rows(vf: pl.Vf, tile):
    for r in pl.range(ROWS_PER_CORE):        # row loop; register blocks unrolled inside
        ...  # vf.load_align → vf.reduce_* → scalar chain → vf.mul/sub … → vf.store_align
```

Board-verified fillings: log-softmax (max+sum, sentinel tail) and RMSNorm (sum-of-squares, mask tail) — see runnable references.

## Failure signatures

- Parser rejects a nested VF call as an argument → name the intermediate ([api/vf.md](../api/vf.md)).
- Tail lanes poison the reduce → the sentinel was not absorbing for *every* step of the chain, or the mask was not updated for the tail block — re-derive per invariant above.
- `squeeze`/`UnalignReg` violations → `python scripts/check.py rules` catches these before the board; run it first.

## Runnable references

- [examples/algorithms/streaming_moments](../../examples/algorithms/streaming_moments/) — separate transfer/reduction granularities, compensated lane and datablock partials, and whole-state masks across input tails; includes exact constant-row variance and long-row cancellation checks.
- [examples/algorithms/residual_rms_quant](../../examples/algorithms/residual_rms_quant/) — required FP16 intermediate rounding, normalized integer outputs across the full coefficient range and whole-row reuse; its two-stage task/GM costs are stated explicitly.
- [examples/algorithms/streaming_masked_softmax](../../examples/algorithms/streaming_masked_softmax/) — bounded UB chunks, persistent max/denominator state and a second input pass for long rows; includes compact logsumexp, empty prefixes and structured recurrence tests.
- [examples/algorithms/masked_softmax](../../examples/algorithms/masked_softmax/) — runtime valid prefixes, empty rows, padded physical width and uneven worker ownership; test the same boundary classes when your reduction has a data-dependent valid set.
- [examples/basic_api/vf/vf_softmax](../../examples/basic_api/vf/vf_softmax/) — the small resident-row VF max/sum/broadcast chain.
- Upstream STs: `pypto/python/tests/st/pypto_pro/frontend/vf_api/test_softmax_dn.py`, `datacopy/test_dual_mode_tail.py`, `gather/test_gather_block_variables.py`.
- Board-verified filling of this pattern: [examples/algorithms/layernorm](../../examples/algorithms/layernorm/) — multi-core, mask-tail, double reduction (the case where a zero sentinel breaks at the second reduce). Further fillings (log-softmax, RMSNorm) can be reproduced by a full workflow pass and admitted the same way.

## Do not use when

- The reduction crosses rows (column-wise / global) — use [outer-axis ownership and blocking](vf-outer-reduce.md); the per-row independence invariants here do not transfer.
- The op is matmul-shaped — cube sections, [api/gemm.md](../api/gemm.md).
- The access pattern is lane-scatter/gather driven — start from [examples/basic_api/vf/vf_gather_scatter](../../examples/basic_api/vf/vf_gather_scatter/) instead.

## Source escape

Signatures: `python scripts/check.py api vf`. Constraint details and measured limits: [api/vf.md](../api/vf.md). Official per-API pages: `../pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/`. If this page disagrees with any of those owners, the owner wins — fix this page.
