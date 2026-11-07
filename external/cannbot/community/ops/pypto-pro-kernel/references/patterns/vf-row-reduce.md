# Pattern: VF row-wise reduce → scalar chain → broadcast

Use with the [VF API contracts](../api/vf.md); the runnable examples below define their own verified scope.

## Applies when

Each output row depends only on its own input row through one or more full-row reductions (max / sum / mean-of-squares) followed by a per-row scalar chain and a broadcast back over the row. FP32/FP16. Row width need not be a multiple of the register width. Rows split freely across cores.

**Register width scales with the tile dtype** (256-byte physical register): FP32 source = 64 lanes per `load_align`, FP16 source = **128** — the `VL=64` constant in the worked fillings is FP32-specific. For FP16 source data needing FP32 math, the order-preserving widening is **`load_align(..., dist=pl.LoadDist.UNPK)`** (64 contiguous FP16 into the low half of 64 b32 containers, source order kept) followed by `astype(dtype=FP32, layout=ZERO)`. Before applying the 64-lane computation, derive the widening predicate from the [physical-container mask contract](../api/vf.md#213-the-fp32-path-for-fp16-source-data-order-preserving-widening-with-unpk-required-reading): an FP16 prefix of the logical count masks off half the expanded values. Plain load + widening `astype` instead silently even/odd-subsamples the source. Worked example: `examples/algorithms/rmsnorm_gemm`.

## Logical dataflow

GM row block → UB tile (padded width) → per row: reduce(s) → scalar chain (sqrt/log/div…) → broadcast-apply over the row → store back only the valid columns.

### Batch independent rows even when each row fits

A full reduction row can still be a small DMA. When many short rows produce repeated load→VF→store calls and transfer/reuse waits, compare loading a `[row_batch, padded_columns]` tile and looping over its rows **inside one VF call**. This changes transfer/call granularity without combining rows' reductions. Initialize maximum, sum and other state independently for each row; the [basic row-softmax example](../../examples/basic_api/vf/vf_softmax/kernel.py) demonstrates this row loop. A resident-row correctness anchor's batch of one is not a tuned default.

Keep metadata attached to its logical owner. Rows may reuse one loaded selector or mask only when their validity domains really match; stop a batch at such a domain boundary or load distinct metadata per row. Equal row widths do not establish equal selectors. Include coordinate-distinct rows, differing masks and partial final batches in validation.

Choose row batch size separately from column/register width and count the busiest worker's actual row batches: a larger batch can increase the last worker's work or leave workers idle. The [local buffering contract](../api/memory-allocation.md#independent-buffers-and-slot-depth) owns slot-depth comparisons, capacity accounting and ring-wrap validation; include every row-batched input, output and metadata tile in that plan.

Batching can add metadata DMA and statistic reloads even while removing many row-sized calls. Keep a row-at-a-time control when those added costs are not amortized, and compare complete calls at matched workers before selecting a shape regime. The [grouped GEMM quantization example](../../examples/algorithms/grouped_gemm_swiglu_quant/) exposes row and batch controls, guards each worker's actual consumers, and uses a block-wide phase boundary. Its batch path normalizes by a bounded row maximum before the integer-range factor; the [normalized-ratio contract](../api/numerics.md#2-small-magnitudes-in-normalized-ratios) still requires independent numerical tests when that arithmetic order changes.

### Choose where dependent phases share state

A mathematical phase is not a native-kernel boundary. First identify the owner of each row and budget the peak simultaneously live tiles, including padding and slots. If that owner can perform all dependent phases within the capacity limit, compare a resident-row native kernel before choosing separate global stages. Values can stay in registers within one VF procedure, or in explicit UB tiles between VF procedures called by the same JIT kernel; [VF scope](../api/constraints.md#simd-vf-functions-communicate-through-tile-parameters) does not require GM intermediates.

Separate native kernels are justified when a consumer needs other workers' completed results, the phase mappings differ, capacity forces streaming/materialization, or a complete-call measurement favors that split. State the actual boundary reason. Count input reloads, intermediate GM writes/reads and task overhead in the comparison. Keep worker count fixed when comparing these dataflows; a worker sweep can improve a staged implementation while leaving its avoidable phase-boundary cost untouched. After choosing a dataflow, retune its worker count.

If a fused phase reads the tile's old contents, first inspect [same-VF memory dependencies](../api/vf.md#212-multiple-memory-accesses-to-the-same-tile-inside-vf-have-no-ordering-guarantee). `auto_mutex` orders calls through tile groups; it does not order a store and reload inside one VF procedure. Compare the appropriate VF barrier, or separate producer/consumer VF procedures using the same tile group within one JIT kernel, before attributing the problem to fusion or requiring a GM round trip. Changing to separate native kernels may hide the missing local ordering without establishing that the split is necessary.

For independent rows in one UB block, compare a block-wide phase boundary with a barrier inside every row: first write all rows' intermediates and statistics, issue the required store-to-load barrier, then consume the rows. Count the additional live statistic storage and keep the row batch fixed when attributing a gain to barrier placement. [Bounded MLA](../../examples/algorithms/mla_bounded/kernel.py) exposes this organization; removing a barrier without moving the dependent reads is not the same experiment.

When a row statistic becomes a divisor for a scale-normalized result, validate the permitted input magnitudes as well as shapes. Tiny numerator and denominator errors can become large ratio errors even if the statistic passes an absolute tolerance. Use the [normalized-ratio dynamic-range method](../api/numerics.md#2-small-magnitudes-in-normalized-ratios) when the contract permits very small values; preserve required intermediate rounding.

If any per-row output is a value, index, count or statistic, read [patterns/vf-small-results.md](vf-small-results.md) for its physical output shape, write path and dtype. This includes a compact side output alongside a full-sized tensor: each output needs its own transport plan. These decisions are independent of the reduction's tie rule or metadata representation.

For runtime row metadata (length, segment bound, scale), `pl.getval(tensor, linear_element_offset)` reads a scalar directly from GM in the outer SIMD kernel; pass that scalar as a VF argument. This is a simple starting point for isolated control values. For batches of reused metadata, compare it with the [UB batch-and-broadcast route](vf-outer-reduce.md#metadata-and-invariant-inputs). Signature and integer promotion: official getval contract (`pypto/docs/zh/api/pro_api/SIMD-API/transpose_and_element_access/getval.md`).

For uneven row counts, define ownership together with the host launch. In a pure vector kernel, `pl.range(pl.get_block_idx(), ROWS, pl.get_block_num())` gives every row one owner. Launch a bounded `WORKERS` with `kernel[None, WORKERS](...)`; pure Vector has `get_subblock_num()==1`, not two. Start with one worker, then test several workers and a row count that does not divide them. Mixed cube/vector kernels use a different index mapping ([primer](../workflows/getting-started.md)).

## When a full row does not fit

The resident-row skeleton below assumes all live tiles fit in UB. For a longer row, budget the input/output chunks, slot multiplicity and persistent state together, then stream GM chunks through a bounded workspace. Choose a **transfer chunk** separately from the VF register width: one DMA and VF call per register can create many small transfers and synchronization handovers. Process several register blocks inside each VF call when capacity permits; compare chunk sizes by measurement rather than treating a particular size as optimal.

Define the aggregate's initialization, per-chunk merge and finalization before allocating an array of partials. A mergeable reduction can carry a fixed-size aggregate instead; a VF register lives within its VF call, so state crossing calls must be stored in an explicit UB tile and protected by the tile-group synchronization. Decide which outputs require final aggregate values: those outputs need a later input pass or retained intermediates. Empty valid sets, tail masks and output guards must keep the same semantics across chunk sizes. Stable exponential normalization has a mergeable max/denominator state; see the [online recurrence](online-softmax.md#the-recurrence-and-where-state-lives), independently of its matmul consumer.

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
