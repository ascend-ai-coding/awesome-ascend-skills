# VF outer-axis reduction: lanes retain outputs, loops reduce the outer axis

Use for outer-axis reduction of contiguous 2-D inputs, per-channel statistics and parameter gradients. Identify each output's free and reduced axes first; different outputs need not share a cross-core partition. This page owns organization, while [VF APIs](../api/vf.md) and [data movement](../api/datamove.md) own register signatures and physical constraints.

## Output ownership before blocking

| Output dependency | Lane meaning | Cross-core mapping |
| --- | --- | --- |
| Each row reduces columns to its own result | Different columns of one row, requiring cross-lane reduction | Workers own disjoint rows; use [row reduction](vf-row-reduce.md) |
| Each column accumulates across rows | Different output columns, accumulated independently; do not `reduce_sum` across them | Workers own disjoint column blocks and traverse all input rows |

For multiple reduction directions, separate native kernels can establish each output's ownership first. Do not let workers update a shared output without synchronization just to reuse one row loop, or introduce atomics/cross-core partials prematurely. If too few output blocks limit parallelism, evaluate splitting the reduction axis and the storage, synchronization and launch cost of a merge; measure whether it pays.

## Choose transfer and register blocks separately

This also applies to 1-D streaming statistics and other repeated VF-state updates; the reduced axis need not be a 2-D input's outer axis. Choose three granularities separately: the register block processed by one instruction, the GM→UB transfer block, and the safe accumulation extent before reset or widening. Different constraints govern them. Register width is not DMA size, and extending accumulation lifetime does not automatically reduce transfers.

One GM→UB transfer can load `[row_block, col_block]` while each `vf.load_align` handles one register segment. **One VF call after a DMA should consume as much of the transferred block as practical**, looping over rows and register blocks internally. Fixing row_block=1 and calling VF per row repeatedly pays DMA, call-boundary and UB-state-spill costs. It is a correctness starting point, not a block-size constraint derived from register width.

For each candidate, count GM transfers, repeated input bytes and UB state spills over a complete call before choosing the granularity to change. Workers that each scan all inputs while owning few outputs may preserve or amplify the main work. Compare this with input partitioning, independent partials and merge, including scratch and extra launches. Explicit owners must write outputs; increasing workers does not permit unsynchronized updates to one location.

1. Start with a 2-D UB block accommodating all live inputs, outputs, metadata, state and slots, with addresses derived from full physical capacity. If the whole reduced axis fits, load once and reduce inside VF; otherwise stream row blocks outside VF. Do not budget only one input tile.
2. Create independent accumulator registers per retained column block and loop over rows **inside VF**. Registers in one call need not spill to UB after each row. Initialize statistics separately: ordinary assignment does not copy registers; see [register aliases](../api/vf.md#vf-register-aliasing-and-copying).
3. Store cross-call state explicitly in UB and include it in tile-group synchronization: initialize the first block, merge intermediate blocks and write output after the last. VF-local registers are not persistent across calls, and every block must not reset the whole result.
4. Use actual valid row counts for row tails and correct masks for column tails. Check whether padding enters computation. In multiplication, mean subtraction and other chains, a final store mask does not make uninitialized lanes safe.
5. Preserve mathematical work and numerical contracts while comparing feasible row_block/col_block choices. More active registers, larger tiles or fewer calls alone do not guarantee performance.

Narrow integer state also has a numerical capacity: derive maximum accumulator values from worst-case input, not an average random distribution. Widen or emit partials before the safe bound; reset only after consumers have used all old state. The UB transfer block need not equal this extent: multiple VF loops or transfers may lie between resets. Test concentration into one output and measure the complete call, rather than changing reset frequency while missing a dominant per-register DMA cost.

## Metadata and invariant inputs

Plan transfers from the operands each output needs instead of making all output branches inherit the largest argument set. For loop-invariant vectors, check whether the owning worker can load once and reuse them; tile lifetime must cover every consumer.

`pl.getval` suits a few scalar reads in outer SIMD control flow. Repeated per-row metadata can instead be loaded into UB as a batch and broadcast for the current row inside VF. For FP32 metadata, `vf.load_align(metadata_tile, element_offset, dist=pl.LoadDist.BRC_B32)` reads and broadcasts one 32-bit element; offsets are in elements. This distribution requires 4-byte source alignment, not ordinary NORM load's 32 bytes. Physical tiles still require valid declaration alignment and capacity, so this does not permit arbitrary narrow `[rows, 1]` buffers. See load_align (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/data_movement/load_align.md`) and [flash_attention](../../examples/algorithms/flash_attention/). Supplied statistics are contractual inputs: both reference and kernel consume them rather than recomputing replacements.

## Verification and measurement

- In addition to random values, include constants, zero inputs eliminating one contribution, structured values across row-block boundaries, row/column tails, workers and repeated calls. Check every output with an independently derived mathematical reference.
- After numerical acceptance, use [native-task timing](../verification/native-timing.md) for all kernels in a complete call and inspect each task, so another fast phase does not hide an inefficient reduction. Share inputs and worker counts, alternate calls and reverse order.
- Runnable outer-axis anchor: [column_statistics](../../examples/algorithms/column_statistics/), demonstrating 2-D transfers, VF row loops and independent per-column state. Its whole-axis residency scope follows the actual UB budget.
- Different reduction axes for multiple outputs: [layernorm_backward](../../examples/algorithms/layernorm_backward/), including batched statistic broadcasts and complete-call validation.
