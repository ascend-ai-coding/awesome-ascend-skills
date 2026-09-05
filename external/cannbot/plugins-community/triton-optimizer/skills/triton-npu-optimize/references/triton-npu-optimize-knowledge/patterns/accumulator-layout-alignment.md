# Accumulator Layout Alignment Pattern

## Summary

Align accumulator shapes with output memory layout to avoid store-time transposes that degrade into scalar element writes on Ascend NPU.

## Use When

- `tl.store` writes a transposed logical tensor and profiling or code inspection suggests the write degraded into scalar element stores.
- The accumulator shape differs from the output memory layout, forcing an implicit store-time transpose.
- The kernel performs a reduction that naturally produces the "wrong" shape order, and a simple axis swap in the reduction logic would avoid the store-side transpose entirely.

## Signals

### Code

- The accumulator has shape `(N, M)` but the output tensor is `(M, N)` contiguously in memory.
- `tl.store` uses pointer expressions that stride across the accumulator's leading dimension in a way that mirrors a transpose.
- Reduction axes, mask broadcasts, and final pointer expressions all orbit around a shape convention that mismatches the output.

## What To Verify After Applying

- Re-check every reduction axis: changing accumulator shape may require changing the reduction dimension.
- Re-check mask broadcasts: the broadcast direction changes when the leading dimension changes.
- Re-check the final pointer expression: confirm it addresses the output in the correct contiguous order.
- Confirm correctness on tail shapes where partial blocks interact with the new axis ordering.

---

## Detail

### Before (accumulator shape mismatches output layout)

```python
# Accumulator carried as (N, M), output expects (M, N)
acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
# ... compute into acc ...
# Store-time transpose can degrade to scalar writes
tl.store(out + offs_n[:, None] * stride_n + offs_m[None, :] * stride_m, acc, mask=...)
```

### After (accumulator matches output layout)

```python
# Accumulator carried as (M, N) — same shape as output memory layout
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
# ... compute into acc with adjusted reduction axes ...
# Store is now a straight write along the output's contiguous direction
tl.store(out + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, acc, mask=...)
```
