# Pattern: VF reductions with compact outputs

Read when any output per input row is a value, index, count or a few statistics, including a compact side output accompanying a full-sized tensor. Input row independence does not by itself settle every output's physical write path; analyze the transports separately.

## Four decisions before the first run

1. **Keep physical tile shape separate from GM rank and valid shape.** Plain row-major Vec tiles are 2-D; every physical row, including a staging or output tile, needs a byte width divisible by 32. Use `[1, padded_cols]` for a 1-D GM vector and `[1, count]` for its effective region. Padding only the final work tile does not repair an unaligned intermediate tile. Prefer loading directly into an aligned tile with the true valid shape when an extra staging tile serves no purpose. Owners: TileType (`pypto/docs/zh/api/pro_api/SIMD-API/basic_data_structures/TileType.md`), [dynamic versus fixed valid shape](../api/datamove.md#fixed-and-dynamic-valid-regions).
2. **Choose the output transport explicitly.** For tightly packed outputs from multiple workers, form the result in UB and use `pl.store`. Replacing the DMA with adjacent GM `pl.setval` calls lost writes in current board probes. This is a transport-specific observation: it does not establish a spacing requirement for unrelated DMA outputs. Grouping more rows under one worker can hide the scalar-write conflict for some base addresses while reducing parallelism; changing a test guard's origin does not establish support for other contiguous views. Preserve the public tensor contract and compare the write paths before changing ownership. See the [scalar/DMA comparison](../api/support.md#7-small-gm-scalar-writes-from-multiple-blocks).
3. **Track the actual register dtype through the final store.** A rounded or truncated floating value is still floating: `vf.truncate` does not produce an INT32 register. Use `vf.astype(..., dtype=...)` for numerical conversion; use `vf.bit_cast` only when interpreting a bit representation (such as the index payload of an extrema reduction). Keep indices/counts integer where possible; if produced in floating point, check the representable range before conversion. Owners: [conversion distinction](../api/vf.md#numerical-truncation-is-not-integer-type-conversion), [extrema payload](../api/vf.md#21-two-facts-about-reduce_).
4. **Use VF operations to update a carried register.** In a runtime loop or branch, `state = next_register` is ordinary name rebinding. Write directly to the existing result with the VF operation, or use `vf.move`; [register identity and control-flow merging](../api/vf.md#vf-register-aliasing-and-copying).

## Check the actual UB address, not only the tile base

For ordinary b32 data-register `store_align` with `StoreDist.NORM`, every target address must be 32-byte aligned: check `tile_base + 4 * linear_offset`, including the runtime loop step. An aligned tile base and padded row width do not make every column address aligned. A one-lane predicate limits the bytes written; it does not relax the instruction's starting-address alignment. Treat address alignment, selected lanes and destination capacity as separate checks.

When collecting one b32 result per iteration, choose the transport before laying out the partials. `StoreDist.FIRST_ELEMENT_B32` writes exactly the first 4 bytes with 4-byte address alignment, **ignoring the predicate**; use it only when that unconditional single-element write is intended. Alternatively, give each result an aligned physical slot and use the same pitch when reading it back; initialize or mask padding before reducing the slots. For a mergeable reduction, an aligned fixed-size state tile can avoid a growing partial-result table altogether. The distribution-specific alignment and mask contracts belong to store_align (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/data_movement/store_align.md`); do not generalize the b32 rule to packed, b64 or MaskReg stores.

The output part, inside `@pl.jit`, has this shape; the reduction itself remains yours:

```python
group = pl.make_tile_group(
    type=pl.TileType(shape=[1, PAD_COLS], dtype=OUT_DTYPE,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
    addrs=[OUT_ADDR], mutex_ids=[OUT_ID])
tile = group.next()
# A VF writes count initialized OUT_DTYPE elements into tile.
pl.set_validshape(tile, [1, count])
pl.store(output, tile, [output_offset])
```

Run `check.py rules` before the board; the literal-rank gate catches `[N]`, but unknown dimensions and byte alignment still require reasoning or the framework. First verify one row and then a multi-worker compact output initialized to a sentinel. If only the latter fails, compare the write path before changing the reduction or core-index formula.

The physical output tile must also cover the VF store's actual width. GM `valid_shape=[1, 1]` does not shorten an all-lane register store into UB. Either mask the VF store to the intended lanes or reserve its full footprint. Runnable consumer: [row_argmax](../../examples/algorithms/row_argmax/), with exact ties and compact output guards.

For a compact side output alongside a full-sized result, [prefix_cross_entropy](../../examples/algorithms/prefix_cross_entropy/) carries bounded reduction state, drains one scalar via UB/DMA and streams the dense result separately. Its tests include nonzero storage offsets, so a public contiguous view is not silently replaced with a specially aligned allocation.
