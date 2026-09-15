---
id: layout-materialization-elision
priority: normal
---
# Layout Materialization Elision

## Summary

Avoid materializing tensors whose only purpose is to change logical layout, such as `permute`, `transpose`, `movedim`, `reshape`, `squeeze`, or `unsqueeze`, when the next step immediately copies, stores, reduces, gathers, or otherwise consumes the data. Instead, express the desired logical layout in the consuming kernel's pointer math or block-pointer metadata and write directly to the final destination layout.

This pattern is most valuable when the materialized layout tensor is large enough that the extra global-memory pass dominates the useful work. The win comes from deleting a full layout-copy phase, not from making the remaining copy kernel magically compute less data.

## Use When

- The current implementation creates an intermediate tensor with `permute(...).contiguous()`, `transpose(...).contiguous()`, `movedim(...).contiguous()`, `clone()`, `copy_()`, or a Triton helper that exists only to produce a different physical layout.
- A later step immediately copies that intermediate into the final output, consumes it in a reduction, feeds it to a simple elementwise/gather/scatter kernel, or stores it in another layout.
- The layout transform is semantically just axis reordering, singleton-axis insertion/removal, reshape/view-compatible reindexing, or another affine mapping.
- The source and destination access pattern can be represented with explicit strides, 2D/3D tile offsets, or `tl.make_block_ptr`.
- Profiling shows `Transpose`, `Contiguous`, `DataCopy`, `Memcpy`, `copy_`, or a separate layout-conversion Triton kernel taking meaningful time.
- The output destination is known at dispatch time, so the optimized kernel can write the final layout directly.

## Avoid When

- The intermediate layout is reused by multiple later kernels and materializing it once is cheaper than repeating strided access.
- The transform is not a simple affine layout mapping, such as value-dependent indexing that needs a different gather/scatter strategy.
- The consumer truly requires a physical contiguous layout for a backend-specific fast path, and direct strided access is slower than the one-time layout copy.
- The layout copy intentionally converts a hot strided kernel access into contiguous time-axis loads, such as `[B, T, HV] -> [B, HV, T]` for repeated per-head chunk loads. In that case, treat the copy as an explicit layout optimization and verify it end-to-end instead of deleting it blindly.
- The tensor is tiny and the extra branch, specialization, or kernel complexity costs more than the removed copy.
- Non-contiguous inputs are allowed but the rewritten kernel only handles contiguous row-major strides.
- The final output has aliasing or in-place semantics that make direct writes unsafe.

## Signals

### Code

- A wrapper computes `tmp = x.permute(...).contiguous()` or `tmp = x.transpose(...).contiguous()` before launching another kernel.
- A dispatch branch squeezes a singleton dimension, calls a lower-rank helper, then performs `out.copy_(tmp.reshape(out_shape))`.
- A kernel writes an intermediate layout and a later copy writes the same number of elements to the final output.
- Flat offset code decodes output coordinates and maps them back to input strides, but only as a generic fallback for a small set of known permutations.
- A specialized tile kernel already exists for the reduced-rank or transposed case, but it writes to a temporary output instead of the final buffer.

### Profile

- End-to-end latency includes a separate layout-copy, transpose, or copy kernel before or after the main computation.
- The layout-copy time scales with full tensor size and is comparable to the useful compute/copy kernel.
- Removing a temporary should reduce global-memory traffic by roughly one full read plus one full write of the intermediate tensor.

### IR

- Lowered IR contains a temporary allocation or copy-like phase whose shape matches a permuted or squeezed view.
- The hot path shows a layout transform followed by another full read/write pass over the same logical data.

## Core Rewrite

Do not do this when the temporary is only a bridge to the final layout:

```python
tmp = triton_permute(x_squeezed, new_dims)
out.copy_(tmp.reshape(out_shape))
```

Instead, pass the final output buffer to the layout-aware kernel:

```python
_launch_rank3_perm_201(x_squeezed, out)
```

The same principle applies inside Triton kernels. Prefer a tile whose axes match one side of the transform, then store to the final layout with explicit offsets or a block pointer:

```python
values = tl.load(src_block)
tl.store(dst_block, tl.trans(values))
```

or, for non-square / non-block-pointer cases:

```python
values = tl.load(src + src_offsets, mask=src_mask)
tl.store(dst + dst_offsets, values.permute(1, 0), mask=dst_mask)
```

## Implementation Guidance

1. Identify whether the intermediate layout has any real consumer besides the next copy/store/compute step.
2. Write down the source logical coordinates and final destination coordinates.
3. Choose the tile orientation that keeps at least one side contiguous, usually the input load side for read bandwidth or the output store side for write coalescing.
4. Encode the mapping with explicit strides or `tl.make_block_ptr` instead of flattening all coordinates through div/mod.
5. If singleton dimensions are inserted or removed, check whether the flattened storage order is unchanged; when it is unchanged, a lower-rank specialized kernel can often write directly to the higher-rank final `out.reshape(-1)`.
6. Keep the generic fallback for unsupported permutations, non-contiguous inputs, or small unusual shapes.

## Example: Permute Copy

For `out = x.permute(0, 2, 1)` on contiguous `[B, M, N]`, a direct 2D tile transpose per batch avoids a generic flat coordinate decoder:

```python
x_block = tl.make_block_ptr(
    base=x + b * M * N,
    shape=(M, N),
    strides=(N, 1),
    offsets=(m0, n0),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0),
)
values = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")

out_block = tl.make_block_ptr(
    base=out + b * M * N,
    shape=(N, M),
    strides=(M, 1),
    offsets=(n0, m0),
    block_shape=(BLOCK_N, BLOCK_M),
    order=(1, 0),
)
tl.store(out_block, tl.trans(values), boundary_check=(0, 1))
```

For a leading singleton such as `[1, C, H, W]`, a reduced-rank specialized kernel can write the final output directly when the squeezed dimension only changes shape metadata. Avoid producing a squeezed temporary and then copying it back into the final 4D output.

## Evidence

Permute-style layout kernels commonly expose this. Specialized 2D/3D permute kernels already use block-pointer tile loads and `tl.trans(values)` to write final layouts directly. A remaining opportunity is the leading-singleton squeeze path: it can compute a lower-rank permutation and then call `out.copy_(result_squeezed.reshape(out_shape))`. For a large leading-singleton tensor, that final `copy_` is an extra full-size global-memory pass. Passing the final `out` buffer directly to the lower-rank specialized kernel should remove that copy kernel and its memory traffic.

The same structural idea also underlies non-last-dimension reductions that avoid `movedim(...).contiguous()` before reducing, but this card applies more broadly to copy-like, store-like, gather-like, and compute-consuming layout bridges.

## What To Verify After Applying

1. Correctness matches the original layout semantics for all dispatched shapes and permutations.
2. The optimized branch writes exactly the final output layout, not a temporary layout that merely has the same number of elements.
3. Profiles no longer show the removed `copy_`, `Transpose`, `Contiguous`, or intermediate layout-conversion kernel.
4. Total global-memory traffic decreases for the targeted branch.
5. Block-pointer `shape`, `strides`, `offsets`, `block_shape`, and `order` match the physical source and destination layouts.
6. Singleton-axis rewrites preserve flattened element order before writing directly to a reshaped final output.
7. Fallback paths still cover non-contiguous, unsupported-rank, or uncommon layout cases.

## Related Patterns

- `block-pointer-dimensionality`
- `reduce-avoid-transpose-copy`
- `remove-implicit-transpose`
- `padded_row_col_copy`
- `grid-flatten-and-ub-buffering`
