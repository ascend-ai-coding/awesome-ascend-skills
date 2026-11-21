<!-- kernel-query: {"id":"indexed-axis-gather","aliases":["indexed-axis-gather","GatherElements","逐元素索引","按轴采集","逐元素 Gather"],"term_groups":[["gather","gatherelements","采集","索引"],["axis","element","逐元素","按轴"]],"when":"Each output element independently indexes one source axis; broadcasting and row/page payload transport need their own coordinate contracts.","missing_facts":["Establish original-coordinate maps, index bounds, physical strip pitch, pass/merge cost and target behavior before selecting the implementation."]} -->
# Indexed-axis gather: coordinates, residency and pass costs

Use when each output element has its own index along one source axis, as in `GatherElements`. An index selecting a whole row/page is a different contract; use [indexed payload transport](simt-payload.md) for that case. The mapping below is a design candidate, with no new target validation from its inclusion here. The [row-gather example](../../examples/algorithms/row_gather/) validates its own smaller, row-local contract.

## Preserve coordinates before flattening

For the **non-broadcasting, equal-rank GatherElements model**, factor source and index tensors about indexed axis `k`:

```text
x   -> [Px, X, W]       X = x.shape[k]
idx -> [P,  G, V]       y has the index tensor's shape
y[p,g,v] = x[premap(p), idx[p,g,v], wmap(v)]
```

`premap` decodes index-prefix coordinates and re-encodes them using source-prefix strides; `wmap` does the same for suffix coordinates. Matching non-indexed extents is a sufficient condition for identity; unequal extents need their own coordinate proof, even when flattened element counts agree. Follow the task's bounds and negative-index rules; flattening does not normalize an index.

For `take_along_dim` or another contract that broadcasts non-indexed axes, first derive the broadcast **output domain** and its coordinate map into each operand, including singleton-axis selection. The output may be larger than the original index tensor. Apply the strip analysis to those mapped coordinates; do not substitute the original index shape for the output domain or materialize a broadcast copy merely to recover this notation.

If differing suffix extents make `wmap(v)` discontinuous, keep those coordinates in the work-item map and find a genuinely contiguous inner run before choosing a strip. Host may derive shape/stride metadata; runtime indices and any content-dependent routing stay on device. For noncontiguous tensors, use the actual [physical stride contract](../api/datamove.md#host-tensor-strides-and-boundary-guards).

## Compare a resident source strip with direct GM indexing

For one mapped outer coordinate and contiguous source columns `[w0,w0+C)`, stage source rows `[pass_lo,pass_hi)` into UB. Let `Xs=pass_hi-pass_lo` and `PITCH` be the **physical** UB row width, including padding. A valid output lane with local column `c` reads:

```text
offset = (index - pass_lo) * PITCH + c
```

Logical width `C` and physical `PITCH` need not agree. `set_validshape` does not compact the row pitch. Budget source, index, output, temporary and rotating slots together using the [memory plan](../api/memory-allocation.md); capacity alone does not choose the strip width or number of passes.

For a dense logical walk over an output `[Gtile,C]` block, register step `r` and lane `l` have `q=r*L+l`, `g=q//C`, `c=q%C`. The actual output address still uses its own physical pitch. If the walk reaches `r=1`, hoisting `c=l%C` requires `L%C==0`; replacing modulo with `l & (C-1)` additionally requires positive, power-of-two `C`. With `L=32,C=64`, the next register starts at column 32: the hoisted expression silently selects the wrong half. A padded output may need row-wise stores rather than one flat contiguous store.

Keep short inner runs and partial final strips explicit. `floor(inner/C)==0` is not zero work when `0<inner<C`; choose a legal smaller/flat run or implement a genuine tail. Wider strips reduce narrow transfers but consume more UB and may increase source-axis passes. Compare nearby widths and full-call costs instead of treating a historical byte width as a platform threshold.

When row-strided indexed accesses dominate, compare legal padded physical pitches at fixed logical width, dtype and index distribution. Padding may change access conflicts as well as capacity and pass count; a pitch rule measured for another layout or device is a hypothesis to test, not a correctness constraint or a guaranteed speedup.

The [gather API owner](../api/vf.md#31-gather-has-two-forms-out-of-bounds-behavior-is-defined-for-only-one) distinguishes UB-source and register-source forms, address units and index dtypes. For a UB gather, prove every issued address legal **before** loading; a final result/store mask cannot repair an out-of-range read. For a source-axis pass, form a pass-membership predicate, use a safe initialized address for nonmembers, then update output only for members. Derive index-range capacity from the maximum physical offset, including `PITCH` and `c`, before narrowing an index register.

## Include the merge in a multi-pass comparison

Putting the source-axis pass outside output-row chunks can reuse one source strip across those chunks. Reversing the loops may reread that strip for every output chunk. Conversely, outputs span several passes: define initialization and retention of values already selected. A local masked VF update does not make a following `pl.store` an element-masked GM write. If the implementation reloads and rewrites output on later passes, include those reads/writes, index reloads, scratch and synchronization in the complete-call cost. Never read uninitialized output as the initial retained value.

Keep a [direct-GM SIMT control](simt-first.md) when the target supports it, and compare [flat/grouped transport](simt-payload.md#compare-flat-and-grouped-mappings) where several elements share metadata. The strip route's regular DMA is useful only if it offsets pass and merge work; a vector-only historical implementation does not rule out the current SIMT surface.

## Verify mapping and representation independently

Use a reference that indexes original tensor coordinates, including unequal non-indexed extents, singleton/rank-one inputs, source-axis passes, inner tails and register steps with `L%C!=0`. Coordinate-distinct data exposes a wrong `premap`, `wmap` or pitch. Cover first/last legal indices, duplicates, permitted negative indices, changed indices on the same storage, workers and guards. For pure movement, check [original bit patterns](../api/numerics.md#5-representation-preserving-outputs); the supplied evaluator remains the acceptance criterion. Shape arithmetic controls prove the mapping, while the actual installed API and target run establish the implementation's behavior and performance.
