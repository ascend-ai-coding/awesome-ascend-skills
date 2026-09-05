# Exact-Tile No-Boundary Fast Path

## Summary

Split exact-tile hot paths from generic masked kernels when dispatch-time shape guards can prove there are no tail tiles, so Ascend lowering can avoid boundary-only masks, padding values, block-pointer `boundary_check`, and related control branches.

## Use When

- A dominant benchmark shape is exactly tile-divisible, such as `M % BLOCK_M == 0` and `N % BLOCK_N == 0`.
- Python dispatch can guard the aligned branch before launch and keep the original masked kernel as fallback.
- MLIR, LLVM, or profiler traces still show boundary checks, masks, padding, or branch/control overhead on the exact-tile hot path.
- The kernel is already structurally reasonable, so a bounded control-overhead cleanup can matter.

## Avoid When

- The mask is algorithm semantics, not a boundary/tail guard.
- Exact-divisibility cannot be proven at dispatch.
- Tail-heavy or irregular shapes dominate the workload.
- The main bottleneck is clearly random global memory, atomics, or compute throughput and boundary control is negligible.
- The fast path would duplicate too much complex logic and drift from the fallback.

## Signals

### Code

- `tl.load(..., mask=tail_mask, other=...)` where the mask only protects block edges.
- `tl.store(..., mask=tail_mask)` on shapes known to be full-tile.
- `tl.make_block_ptr` loads or stores keep `boundary_check` for exact shapes.
- Removing the mask does not change address math for the guarded shape.

### Profile

- Parent kernel is close to the target but still shows scalar/control overhead.
- Expected gain is modest, often small single-digit percent to low double-digit percent.

## Optimization Strategy

1. Identify the hot exact-tile shape and tile divisibility guard.
2. Split a minimal aligned kernel from the generic masked kernel.
3. Remove only boundary/tail masks, padding, and `boundary_check` in the aligned kernel.
4. Keep the generic masked kernel for all non-exact cases.
5. Compare parent-vs-child performance on the exact case and verify fallback coverage.

### Variant: chunk recurrence tail peeling

For chunked recurrence kernels, the whole sequence may not be tile-divisible, but every chunk before the last one is still full. In that case, split the recurrence into a full-chunk hot loop plus one tail block:

- hot loop: `for i_t in range(NT - 1)` with no per-iteration `min`, tail mask, or boundary-only `tl.where`
- tail block: `i_t = NT - 1`, compute `last_idx = min(NT * BT, T) - 1`, and keep the masks needed for the partial final chunk

This is a "mostly exact tile" fast path. It avoids paying tail-control cost in every recurrence iteration while preserving the generic final chunk behavior.

```python
# Full chunks: no scalar min/mask/where in the hot loop.
for i_t in range(NT - 1):
    last_idx = (i_t + 1) * BT - 1
    b_g_last = tl.load(g_base + last_idx)
    b_g = tl.load(p_g_full_chunk, boundary_check=(0,))
    b_v = b_v * exp(b_g_last - b_g)[:, None]

# Tail chunk: may be partial.
i_t = NT - 1
last_idx = min(NT * BT, T) - 1
m_t = (i_t * BT + tl.arange(0, BT)) < T
b_g_last = tl.load(g_base + last_idx)
b_g = tl.load(p_g_tail, boundary_check=(0,))
b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
```

Use this when only the final chunk can be partial. Avoid it if many chunks are irregular, or if the mask is algorithm semantics rather than boundary protection.

## Example

```python
if M % BLOCK_M == 0 and N % BLOCK_N == 0:
    _kernel_aligned_no_boundary[grid](...)
else:
    _kernel_masked_fallback[grid](...)
```

Inside the aligned kernel:

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
value = tl.load(src + offs_m[:, None] * stride_m + offs_n[None, :])
tl.store(dst + offs_m[:, None] * out_m + offs_n[None, :], value)
```

## Evidence

Splitting an aligned/no-boundary kernel out of a masked kernel removes boundary-check control overhead. The gain is modest (roughly ~1.1x) when the dominant cost lies elsewhere — for example a rank-2 gather whose real bottleneck is random global-memory access. Treat this as control-overhead cleanup rather than an access-pattern fix.

## What To Verify After Applying

- Fast path and fallback produce identical values on representative exact-tile shapes.
- Fallback still handles non-divisible shapes.
- The aligned kernel IR no longer contains the targeted boundary checks or masks.
- Parent-vs-child benchmark improves on the targeted case without broad regressions.
- For chunk recurrence tail peeling, test `T < BT`, `T == BT`, `T % BT == 0`, `T % BT != 0`, and varlen branches if present.

## Related Patterns

- `compile_hint`
- `padded_row_col_copy`
- `block-pointer-dimensionality`
- `discrete_memory_access`
- `scalar-latency-traps`
