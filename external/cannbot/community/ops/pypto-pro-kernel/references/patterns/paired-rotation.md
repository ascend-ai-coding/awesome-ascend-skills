<!-- kernel-query: {"id":"paired-rotation","aliases":["paired-rotation","RoPE","rotary embedding","旋转位置编码","成对旋转"],"term_groups":[["rotation","rotary","rope","旋转"],["pair","table","position","配对","系数表","位置"]],"when":"Paired row elements rotate using a position-indexed coefficient table; determine pairing and table-access maps independently.","missing_facts":["Establish pairing, batch/position mapping, coefficient dtype and rounding, supported target transport and complete-call performance."]} -->
# Paired rotation: separate pairing from coefficient-table access

Use when an even-width row rotates pairs with coefficients from a smaller position-indexed table. If coefficients already arrive at full operand shape, this table-transport decision is absent. The formulas below are design candidates, not a newly validated kernel. [RoPE and KV scatter](../../examples/algorithms/rope_kv_scatter/) supplies a current direct-GM SIMT anchor for its explicit-position, adjacent-pair FP16 contract; it does not validate every layout or table form here.

## Fix the pair and position maps independently

For rotated width `D=2F`, pair `j` is `(j,j+F)` for split-half pairing or `(2j,2j+1)` for adjacent pairing. Both apply:

```text
out_a = a*c - b*s
out_b = b*c + a*s
```

An ordinary contiguous row load already contains both split halves. Adjacent pairing may use a supported port/register de-interleave or a pair-owned SIMT work unit; see [VF port distributions](../api/vf.md#214-loadstore-port-distribution-modes-prefer-built-in-port-rearrangement-over-compute-emulation) and [transport units](simt-payload.md#derive-ownership). Do not introduce Host copies to split or reassemble the data. Partial rotary domains need their own pair bounds and untouched-output contract.

For contiguous row-major operands with positive `B,S,N`, flatten all axes above `D` to row index `r`:

| Operand layout | Position `s` | Batch `b` |
| --- | --- | --- |
| `[B,S,N,D]` | `(r//N)%S` | `r//(S*N)` |
| `[B,N,S,D]` | `r%S` | `r//(S*N)` |

Equivalently, `s=(r//divisor)%S`, with divisor `N` or `1`. A flattened coefficient table has `table_row=b*batch_stride+s`, where `batch_stride=0` selects a shared `[S,F]` table and `batch_stride=S` selects `[B,S,F]`. Table column is pair index `j`. These formulas require the stated layout and dimensions; explicit runtime position indices, offsets or noncontiguous strides need their actual coordinate map instead.

## Batch by the coefficient-access regime

| Structural condition | Transfer candidate | Batch boundary |
| --- | --- | --- |
| `divisor==1` | Operand rows advance with consecutive table rows; load a contiguous `[row_batch,F]` coefficient slice | Stop before `s` wraps at `S` |
| `[B,S,N,D]` with `N>1` | Consecutive operand rows share one table row; retain/broadcast that row across a row batch | Stop before the repeated group ends; from arbitrary `r`, at most `N-(r%N)` rows remain |

Respect the batch/table allocation boundary as well. In `[B,S,N,D]` with `N=1`, the first regime applies: using a one-table-row broadcast loop would create one tiny transfer per operand row. With explicit positions, equal positions must be proved from the task contract or checked on device before using the repeated-row regime; equal shapes do not establish table reuse.

Choose row batch and local slot depth separately, accounting for all live pair values, coefficient fields and outputs. [Row batching](vf-row-reduce.md#batch-independent-rows-even-when-each-row-fits) owns the transfer/worker tradeoff, while [allocation](../api/memory-allocation.md) owns peak residency. Compare this DMA/VF candidate with the complete direct-GM SIMT call at matched geometry, including table reloads, pairing, output transport and any scatter. A small arithmetic body does not establish that its arithmetic is free or that every shape is bandwidth bound.

## Preserve arithmetic and verify the maps

Read the actual reference/evaluator's intermediate dtypes and rounding before replacing two products and an add with fused or compensated arithmetic. Cancellation can make a small result sensitive to product rounding; [numerical contracts](../api/numerics.md#1-cancellation-in-chained-matmul-golden-is-not-the-scale-for-rtol) owns that diagnosis. Widen inputs and coefficients together when the contract requires wide computation, then narrow at the prescribed boundary. An extra narrow published output must not silently become the internal computation source.

Test both pairing rules only when supported by the requested contract. For the table map, use distinct coefficients per batch, position and pair, both operand layouts, `N=1`, shared/per-batch tables, batches starting inside a repeated group, and batches ending at a position wrap. Add row tails, permitted dtypes/ranges, cancellation, workers, repeated calls and output guards. Check preserved regions by [bits](../api/numerics.md#5-representation-preserving-outputs). A CPU coordinate proof does not validate a VF conversion, DMA layout or device synchronization; retain a scope-matching target result before selecting a new implementation.
