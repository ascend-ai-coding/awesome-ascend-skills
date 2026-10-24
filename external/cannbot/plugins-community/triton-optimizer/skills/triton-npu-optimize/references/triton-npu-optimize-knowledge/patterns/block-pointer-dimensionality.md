# Block Pointer Dimensionality Pattern

## Summary

Use `tl.make_block_ptr` to model multidimensional contiguous tensor dimensions directly, enabling wider DMA transfers and reducing scalar address-generation overhead compared to flattened 1D offsets.

## Use When

- A high-dimensional contiguous tensor is accessed through flattened one-dimensional offsets that stride through an inner dimension.
- An inner dimension is processed by an explicit loop or decoded from `program_id` even though it could be included in the block shape.
- Profiling or IR suggests the 1D pointer path produces strided or non-coalesced loads across a dimension that is actually contiguous in memory.

## Signals

### Code

- Manual pointer arithmetic reconstructs multi-dimensional coordinates from a single flat `program_id`.

## What To Verify After Applying

- Confirm every field in `tl.make_block_ptr` — `shape`, `strides`, `offsets`, `block_shape`, and `order` — matches the actual tensor layout. One wrong field can silently benchmark a different access pattern.
- Verify that `boundary_check` and `padding_option` produce correct results on tail blocks.
- When an inner dimension was previously part of grid partitioning, verify the grid reduction is correct and that per-program work density improves.

---

## Detail

### Before (flattened 1D offset)

```python
pid = tl.program_id(0)
# Flattened offset strides through inner dimension — compiler sees one long strided access
offs = pid * BLOCK + tl.arange(0, BLOCK)
vals = tl.load(x + offs, mask=offs < total)
```

### After (multidimensional block pointer)

```python
pid_t = tl.program_id(0)
ptr = tl.make_block_ptr(
    base=x,
    shape=(T, H),
    strides=(stride_t, stride_h),
    offsets=(pid_t * BLOCK_T, 0),
    block_shape=(BLOCK_T, BLOCK_H),
    order=(1, 0),
)
tile = tl.load(ptr, boundary_check=(0, 1), padding_option="zero")
```

### Vectorizing an inner dimension

If an inner loop only walks a small dimension, include that dimension in the loaded tile and compute with an extra tensor axis:

```python
# Before: explicit inner loop over small dim
pid = tl.program_id(0)
for d in range(D):
    vals = tl.load(x + pid * stride_pid + d * stride_d + tl.arange(0, BLOCK))

# After: include D in the block shape
pid = tl.program_id(0)
ptr = tl.make_block_ptr(
    base=x, shape=(N, D), strides=(stride_n, stride_d),
    offsets=(pid * BLOCK_N, 0), block_shape=(BLOCK_N, D), order=(1, 0),
)
tile = tl.load(ptr, boundary_check=(0, 1))
```

Update broadcasting and grid mapping together; if the inner dimension was part of grid partitioning, removing that grid axis may be part of the optimization.
