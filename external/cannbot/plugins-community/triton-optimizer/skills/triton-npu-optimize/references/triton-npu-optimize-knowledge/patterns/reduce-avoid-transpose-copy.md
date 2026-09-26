---
id: reduce-avoid-transpose-copy
priority: normal
---
# Reduce Avoid Transpose Copy for Non-Last-Dim Reduction

## Summary

Avoid implementing a non-last-dimension single-axis reduction by first doing `movedim(...).contiguous()` or an equivalent layout materialization. For contiguous row-major input, compute `[outer, reduce, inner]` from the original shape and reduce directly from the original layout with a strided/tiled kernel.

The goal is end-to-end latency, not making the new kernel faster than a pure contiguous last-dim reduction in isolation. This pattern wins when removing the copy costs more than the extra strided-access cost.

## Use When

- The operator reduces exactly one logical axis.
- The reduce dimension is not the last dimension.
- The input tensor is contiguous in its original row-major layout.
- The current implementation uses `movedim(...).contiguous()`, `transpose(...).contiguous()`, `permute(...).contiguous()`, or another full layout materialization before reduction.
- Profiling shows `Transpose`, `Memcpy`, `DataCopy`, `Contiguous`, or similar layout-conversion work before the reduction kernel.
- The copy time is comparable to or larger than the reduction-kernel time.
- The suffix dimension after the reduced axis is large enough to provide reasonably coalesced loads along `inner`.

## Avoid When

- The reduction dimension is already the last dimension.
- The reduction is a full-tensor reduction such as `dim=None`; use a flat parallel reduction instead.
- The input is not contiguous and the kernel does not explicitly handle real tensor strides.
- The reduction spans multiple axes. A generalized variant may cover adjacent reduced axes that form one contiguous block, but this card assumes exactly one logical reduced axis.
- The tensor is tiny and copy cost is negligible.
- `inner_size` is very small, so coalescing is poor. For example, reducing `dim=1` on `[B, M, 1]` gives `inner_size=1`.
- The backend cannot lower the needed strided or block-pointer loads efficiently.

## Signals

### Code

- Wrapper code rewrites non-last-dim reduction into last-dim reduction with a sequence like:
  - `x_contiguous = x.movedim(dim, -1).contiguous()`
  - `reduce_size = x_contiguous.shape[-1]`
  - `outer_size = x_contiguous.numel() // reduce_size`
  - `x_2d = x_contiguous.reshape(outer_size, reduce_size)`
- Equivalent anti-patterns appear as `transpose(...).contiguous()`, `permute(...).contiguous()`, `movedim(...).clone()`, or a manual copy into a temporary layout buffer.

### Profile

- A copy or layout-conversion op appears before the real reduction kernel.
- End-to-end time looks like `copy + reduction`, and the copy is a meaningful part of the total.

### IR

- Lowered IR shows a temporary buffer or copy step before reduction, such as `memref.alloc`, `memref.copy`, `DataCopy`, transpose materialization, or permute materialization.

## Key Rewrite

Do not do this:

```python
x = x.movedim(dim, -1).contiguous()
```

Instead compute directly from the original shape:

```python
outer = prod(shape[:dim])
reduce = shape[dim]
inner = prod(shape[dim + 1 :])
```

Then reinterpret the tensor logically as `[outer, reduce, inner]` and launch a 2D kernel over `[outer, inner]`:

- a simple implementation uses one program per `outer` index and `inner` tile
- a more optimized variant may tile both `outer` and `inner`
- the kernel loops over `reduce`
- loads come from the original tensor with explicit strides
- the result is written directly to `[outer, inner]` output

The flattened `[outer, inner]` output is naturally contiguous and can be reshaped to the framework output shape without another copy.

For contiguous row-major input, the logical address rule is:

```text
x[o, r, i] = base + o * reduce * inner + r * inner + i
```

so the logical strides are:

```text
outer stride  = reduce * inner
reduce stride = inner
inner stride  = 1
```

## Example Rewrite

For `x.shape = [B, M, N]` and `torch.sum(x, dim=1)`:

```python
# Before: move the reduced axis to the end, then materialize a copy.
x2 = x.movedim(1, -1).contiguous()   # [B, N, M]
reduce_size = x2.shape[-1]           # M
outer_size = x2.numel() // reduce_size
out = sum_lastdim_kernel(x2.reshape(outer_size, reduce_size))
out = out.reshape(B, N)
```

```python
# After: keep the original layout and reduce directly from it.
outer_size = B
reduce_size = M
inner_size = N

out_2d = torch.empty((outer_size, inner_size), device=x.device, dtype=x.dtype)
sum_nonlastdim_kernel(x, out_2d, reduce_size=reduce_size, inner_size=inner_size)
out = out_2d.reshape(B, N)
```

The key change is that the optimized path never materializes `[B, N, M]` as a temporary contiguous tensor.

## Implementation Notes

- `reduce_size` means the length of the reduced axis: `shape[dim]`.
- `inner_size` means the product of the suffix dimensions after the reduced axis: `prod(shape[dim + 1:])`.
- `TILE_REDUCE` and `TILE_INNER` are kernel tile sizes chosen by the implementation or autotuning.
- A simple implementation uses one program per `outer` index and `inner` tile; a more optimized one may tile both `outer` and `inner`.
- If you use a block pointer for tile layout `[TILE_REDUCE, TILE_INNER]`, use:
  - `shape = (reduce_size, inner_size)`
  - `strides = (inner_size, 1)`
  - `block_shape = (TILE_REDUCE, TILE_INNER)`
  - `order = (1, 0)`
  - `reduce axis = 0`
- If `dim` is already the last dimension, use the normal contiguous last-dim reduction path instead of this pattern.
- If `reduce_size` is very large, a split-reduction design may be better than one program looping across the full reduction axis.

## Pitfalls / Risks

- Do not use `movedim(...).reshape(...)` as a "check"; the copy may already have happened.
- Small `inner_size` can erase the coalescing benefit even if the copy disappears.
- Large tiles can create UB pressure.
- Non-contiguous inputs need a separate stride-aware kernel.
- Floating-point differences are expected because the summation order changes.
- For `fp16` / `bf16` inputs, accumulate in `fp32` when possible before casting back to the output dtype.

## What To Verify After Applying

1. The optimized path is used only for non-last-dim reduction.
2. Input-contiguous preconditions are enforced, or real tensor strides are handled explicitly.
3. Output shape matches the framework reference for both `keepdim=True` and `keepdim=False`.
4. Profiles no longer show the pre-reduction transpose/copy path.
5. End-to-end operator time decreases, not just kernel self time.
6. Numerical differences stay within dtype-appropriate tolerance.
7. Empty reduced dimensions return zeros with the correct output shape.

## Related Patterns

- `remove-implicit-transpose`
- `tiling`
- `software-pipeline`
- `autotune`
