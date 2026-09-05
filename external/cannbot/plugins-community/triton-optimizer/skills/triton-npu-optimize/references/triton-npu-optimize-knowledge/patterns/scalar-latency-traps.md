# Scalar Latency Trap Removal Pattern

## Summary

Remove scalarizing constructs that block vector hardware utilization on Ascend NPU, including unnecessary scalar control flow, loop-carried pointer recurrences, modulo addressing, narrow masks, and int64 arithmetic on vector paths.

## Use When

- Runtime values that are shape constants are passed as normal arguments instead of `tl.constexpr`.
- Pointer variables are updated with `+=` inside a loop, creating loop-carried address dependencies.
- Address expressions use modulo addressing (`%`) to wrap tail tiles or index boundaries.
- `tl.where` masks all lanes except a single special position, or has exactly one false lane in a vector.
- Integer elementwise arithmetic is done as scalar-looking `int64` work even though the value range is safely `int32`.
- `tl.cumsum` or `tl.associative_scan` runs on the last axis of a tensor and profiling or IR suggests scalar fallback instead of vector lowering.
- `tl.cumsum` runs on a long one-dimensional vector and profiling or IR suggests scalar degradation.
- A boundary-only mask repeats validity conditions that earlier `tl.load(..., boundary_check=...)` or safe zero-padding already handled.

## Repairs

### Static parameters

Make compile-time constants explicit:

```python
@triton.jit
def kernel(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
```

Prefer `tl.constexpr` for fixed sizes, strides, booleans, mode flags, and architecture-selected knobs. Do not make data-dependent runtime values constexpr.

### Loop pointer recurrences

Avoid pointer updates that depend on the previous iteration:

```python
# Prefer this shape.
for i in tl.range(0, K, BLOCK_K):
    ptrs = base + (i + offs_k) * stride_k + offs_n
    vals = tl.load(ptrs, mask=(i + offs_k) < K)
```

This keeps each iteration's address computation derived from a stable base plus an explicit offset. It is especially useful when loop trip count is large enough for scalar scheduling to matter.

### Modulo removal

Apply this repair only when `%` is being used as a boundary guard for tail or edge handling. If removing `%` would change the kernel's intended index mapping, this repair does not apply.

Problem shape:

```python
offs = (block_start + tl.arange(0, BLOCK)) % N
vals = tl.load(x + offs)
```

Preferred rewrite:

```python
offs = block_start + tl.arange(0, BLOCK)
mask = offs < N
vals = tl.load(x + offs, mask=mask, other=0.0)
```

### Single-position `tl.where`

When exactly one lane differs, consider replacing a whole-vector `tl.where` with a targeted extract/insert style repair. Only apply this when the one-position condition is proven by shape or index construction. If more than one lane can differ, keep the original vector conditional.

### Int32 vector arithmetic

If index or offset arithmetic is proven to stay within `[-2**31, 2**31 - 1]`, cast once near load or construction and keep the hot vector math in `int32`. Cast back only when the API or pointer expression truly requires it.

Do not use this for values that can overflow `int32`.

### Redundant boundary mask removal

If prior loads already zero-pad invalid rows or columns through `boundary_check`, later vector predicates may not need to repeat both row and column validity. Keep semantic masks, such as causal lower-triangle conditions, but remove boundary-only terms that no longer protect an unsafe value.

Example:

```python
# Before: row and column validity repeated in the final mask.
m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)

# After: row out-of-bounds is already zero from boundary-checked inputs.
m_A = (o_t[:, None] > o_t[None, :]) & m_t[None, :]
```

Only apply this when invalid lanes cannot reintroduce nonzero, NaN, or unsafe pointer behavior between the protected load and the final mask/store.

### Cumsum axis splitting

For a long one-dimensional `tl.cumsum`, consider reshaping to a two-dimensional tile so cumsum runs on shorter axes, then combine block-local prefix totals. Tune the split size because both axes trade off against each other and can affect UB pressure.

### Cumsum axis placement

If `tl.cumsum` or `tl.associative_scan` is on the last axis and the backend lowers it to scalar loops, transpose or swap axes so the cumulative axis is no longer last before scanning. If the output layout needs the original axis order, transpose back after the scan.

```python
# Before: last-axis cumsum can fall back to scalar lowering.
prefix = tl.cumsum(values, axis=1)

# After: move the cumulative axis off the last position to unlock vector lowering.
prefix = tl.trans(tl.cumsum(tl.trans(values), axis=0))
```

## Risks

- `tl.constexpr` changes specialization behavior and compile-cache cardinality.
- Removing `%` is only safe when masks preserve the original boundary semantics.
- Removing redundant boundary masks is only safe when earlier boundary-checked loads or stores still fully protect out-of-range access and invalid lanes have safe values.
- Int32 conversion is a semantic promise about value range.
- Cumsum decomposition must preserve prefix order exactly.
- Axis swaps for cumsum can materialize a transpose; keep the temporary footprint within UB limits.

## Related Patterns

- `reduce-avoid-transpose-copy`
- `tiling`

## What To Verify After Applying

- Record the trap and exact code location in `attempts.md`.
- Run correctness before trusting performance.
- Use the project benchmark and `compare-perf` authority for any claimed speedup.
- If the repair changes specialization keys or host call signatures, verify all call sites.
