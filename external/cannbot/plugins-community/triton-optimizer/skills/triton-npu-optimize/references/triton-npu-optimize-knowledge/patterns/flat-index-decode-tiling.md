# Flat Index Decode To Layout Tiling Pattern

## Summary

Replace scalar-heavy 1D linear-index traversal with layout-aware multidimensional tiles when the logical operation is an affine data movement.

The anti-pattern is a kernel that walks `numel(out)` as a flat stream, then reconstructs logical coordinates with `//` and `%` for every lane before computing source or destination offsets. For regular layout operations, this turns simple memory movement into repeated scalar index decoding. The repair is to make the Triton tile axes match the logical output layout: rows, columns, and rank axes are represented directly by `tl.arange` tensors, so pointer offsets are mostly base-plus-stride arithmetic over contiguous or low-stride dimensions.

This is broader than permute. It applies to materialized transpose/permute, reshape-copy, pad/crop/slice copies, bounded affine gathers, and other copy-like kernels where the mapping from output coordinates to input coordinates is static and affine.

## Use When

- The kernel is mostly data movement, not dense arithmetic or reduction.
- Work is launched over a flat `n_elements` or `out.numel()` stream.
- Each lane recovers coordinates with repeated `//`, `%`, or residual chains.
- The output-to-input mapping is affine: coordinates map through strides, axis reorder, fixed offsets, padding bounds, or simple slice windows.
- At least one logical dimension can be made contiguous or low-stride inside the tile.
- Shape/rank regimes are known enough to dispatch to specialized tile layouts or guarded fallbacks.

## Avoid When

- Indices are value-dependent or irregular enough that the hot path is true gather/scatter; use gather/discrete-access patterns instead.
- The tensor layout materialization can be removed by folding the layout into the next consumer.
- The operation is compute-heavy enough that index decode is not a meaningful bottleneck.
- Rank/stride/shape assumptions are not guarded and would silently change semantics.
- A multidimensional tile would exceed UB/register budget or create invalid grid dimensions.

## Signals

### Code

- `offsets = pid * BLOCK + tl.arange(...)` is the main work assignment.
- The kernel computes `coord0 = linear // stride0`, `residual = linear % stride0`, then repeats for more axes.
- A single flattened mask combines all rank or boundary conditions for every lane.
- Inner dimensions are looped or decoded even though they could be represented as tile axes.
- The same generic kernel handles simple cases such as identity copy, 2D transpose, last-dim slice, or regular pad.

### Profile

- Scalar/control overhead is high for a copy-like kernel.
- Changing only flat `BLOCK_SIZE` gives weak or unstable gains.
- Cases with more rank dimensions or larger flattened extents regress disproportionately.
- Specialized row-column or multidimensional tile variants improve regular shape regimes.

## Repairs

### Promote the effective contiguous dimension into the tile

Pick a logical inner dimension that gives contiguous or low-stride access on either load or store. Often this is the last non-unit output dimension, not necessarily the physical last axis.

```python
rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
mask = (rows[:, None] < num_rows) & (cols[None, :] < inner_size)

src = row_src_base[:, None] + cols[None, :] * src_inner_stride
dst = row_dst_base[:, None] + cols[None, :] * dst_inner_stride
vals = tl.load(x + src, mask=mask, other=0.0)
tl.store(out + dst, vals, mask=mask)
```

This changes coordinate work from per-element div/mod to per-row base setup plus vector stride increments.

### Use rank-aware tile axes instead of residual decode chains

For small fixed ranks, express each active coordinate as a tensor axis and compute offsets directly:

```python
d0 = d0_start + tl.arange(0, BLOCK_D0)[:, None, None, None]
d1 = d1_start + tl.arange(0, BLOCK_D1)[None, :, None, None]
d2 = d2_start + tl.arange(0, BLOCK_D2)[None, None, :, None]
d3 = d3_start + tl.arange(0, BLOCK_D3)[None, None, None, :]

src = d0 * src_s0 + d1 * src_s1 + d2 * src_s2 + d3 * src_s3
dst = d0 * dst_s0 + d1 * dst_s1 + d2 * dst_s2 + d3 * dst_s3
```

This is useful for generic rank-limited layout copies where a row-column collapse would hide important stride differences.

### Add guarded simple-case fast paths

Before launching the generic path, split cases with much simpler layout:

- identity or contiguous copy,
- 2D transpose,
- last-two-axis swap that can be reshaped into a 2D transpose,
- pure slice/crop with contiguous inner spans,
- pad regimes where interior columns need no input-boundary checks.

Fast paths should be shape/permutation/stride guarded and benchmarked against the generic parent.

### Choose tile sizes from stride locality and UB footprint

Prefer widening dimensions with stride `1` or small stride first. Bound the total tile product by dtype, live tensors, masks, and temporary values. A useful heuristic is to grow one axis at a time while the UB estimate stays below budget, then choose `num_warps` from the resulting tile size.

### Cap grid dimensions with grid-stride loops

If the backend or runtime benefits from capped grid dimensions, do not truncate work. Use `tl.num_programs(axis=*)` and loop with a grid stride:

```python
for row_start in tl.range(pid_m * BLOCK_M, rows_total, tl.num_programs(0) * BLOCK_M):
    ...
```

This lets a smaller physical grid cover a large logical tensor.

## Failure Modes And Anti-signals

- Replacing flat decode with a multidimensional tile but still doing per-lane div/mod inside the tile.
- Choosing the literal last dimension as the inner dimension when it is size `1` or high-stride.
- Adding a fast path without exact guards for rank, shape, stride, dtype, and output contract.
- Ignoring destination layout: contiguous loads with scattered stores can still bottleneck if store shape is poor.
- Grid caps without grid-stride loops cause incomplete writes.
- Over-widened tiles reduce scalar work but increase UB/register pressure enough to regress.

## What To Verify After Applying

- Correctness across full tiles, tails, empty tensors, unit dimensions, and every dispatched rank/shape path.
- Parent-vs-child performance for each specialized branch and the generic fallback.
- The optimized kernel has fewer or no hot-path `//` / `%` coordinate decode operations.
- Load/store offsets match the true source and destination strides.
- Tile product stays within practical UB/register pressure on largest representative cases.
- Grid caps plus grid-stride loops cover all logical elements exactly once.

## Related Patterns

- `layout-store-and-block-pointers`
- `padded_row_col_copy`
- `program-multiple-rows`
- `tiling`
- `scalar-latency-traps`
- `gather-load`
- `discrete_memory_access`
- `a5-simt-sliding-window-tuning` — outer dispatch and inner-path routing on A5 SIMT fixed-window reduces
- `simt-clip-window-closed-reduction` — closed normalizer inner window loop after outer tiling
