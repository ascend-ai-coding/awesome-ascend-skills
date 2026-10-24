# Shift 2D Mask To 1D Index Stream

## Summary

When a hot shift or predecessor path is expressed as a 2D mask-and-reduce construction, rewrite it to a direct 1D index stream (`base + arange - 1`) with only boundary masking. This removes unnecessary 2D intermediates and keeps the shift path closer to one-dimensional vector loads and elementwise math on Ascend NPU; do not stop at replacing the reduce with an on-chip `tl.gather` if the final lane formula can be simplified further.

## Use When

- A shift relation is structurally "take previous element" or "take previous position in chunk", including cross-chunk lane-0 handling.
- Code uses 2D mask construction and reduction-like assembly for shifting, such as `arange[:, None]`, `arange[None, :]`, `tl.where`, and `tl.sum(..., axis=...)` over an extra axis.
- IR shows `tt.broadcast`, `tt.reduce`, helper outlined functions, or temporary mask tensors dedicated to shift assembly rather than the core math.
- Profiling indicates scalar/control overhead, UB pressure, vector-function fragmentation, or poor vector utilization around the shift path.

## Avoid When

- The dependency is not a simple predecessor relation and truly needs multi-source gather semantics.
- Boundary behavior depends on nonlocal logic that cannot be encoded as a simple masked predecessor load.
- The path is not hot enough for rewrite complexity to be worthwhile.
- The shifted intermediate is reused by multiple later expressions and direct predecessor reloads would add more traffic than the 2D form removes.
- The proposed rewrite only changes `tl.sum(tl.where(mask, ...))` into `tl.gather` over a previously computed shifted intermediate while preserving the same extra global loads and intermediate computation. That is usually an incomplete rewrite, not the intended pattern.

## Signals

### Code

- Shift path computes with 2D masks and reduction-like reconstruction.
- Lane 0 receives special handling via separate path while other lanes follow "previous lane" semantics.
- The same shift can be expressed with:
  - `idx = chunk_start + tl.arange(0, BLOCK) - 1`
  - `mask = idx >= 0`
  - masked 1D loads from source tensors.
- The square mask is fixed by lane coordinates rather than by data-dependent values.

### Profile

- Scalar/control overhead remains visible after basic tiling and load/store cleanup.
- Runtime improves when replacing 2D-shift assembly with direct predecessor loads.

### IR

- Presence of mask-heavy helper vector functions or 2D mask materialization around shift-only logic.
- After rewrite, shift path lowers to simpler index arithmetic and masked 1D loads.

## Optimization Strategy

1. Isolate the shift-only subexpression from the main compute path.
2. Derive the final per-output-lane formula before coding the replacement.
3. Build predecessor indices as a 1D vector (`base + arange - 1`).
4. Keep a single boundary mask for invalid predecessor positions.
5. Load only the operands that truly come from the predecessor lane.
6. Recompose outputs using the same math semantics as before.
7. Delete obsolete next-lane loads, shifted intermediates, and `tl.gather`/lane-reorder steps after the formula is simplified.

## Example

```python
# Before: conceptual 2D mask/reduce shift assembly (shape omitted)
shift_mask = offs_i == (offs_j + 1)
shifted = tl.sum(tl.where(shift_mask, x[None, :], 0.0), axis=1)

# After: direct 1D predecessor stream
idx = chunk_start + tl.arange(0, BLOCK) - 1
valid = idx >= 0
prev_x = tl.load(x_ptr + idx * stride, mask=valid, other=0.0)
shifted = prev_x
```

If the 2D path first computes `values[j]` from inputs at `j + 1` and then shifts `values[j]` to output `j + 1`, simplify the composition before loading. Often only the upstream gradient or source value is shifted, while destination-local operands remain at the current lane:

```python
# Conceptual example: values[j] = current_operand[j + 1] * upstream[j]
# shifted[i] = values[i - 1] = current_operand[i] * upstream[i - 1]
idx = chunk_start + tl.arange(0, BLOCK) - 1
prev_upstream = tl.load(upstream_ptr + idx * stride, mask=idx >= 0, other=0.0)
shifted = current_operand * prev_upstream
```

Do not treat this as complete if it still computes a full shifted intermediate and then gathers from it:

```python
# Incomplete: removes the explicit reduce but keeps the shifted intermediate.
next_operand = tl.load(ptr + (offs + 1) * stride, mask=(offs + 1) < n, other=0.0)
values = next_operand * upstream
shifted = tl.gather(values, tl.maximum(tl.arange(0, BLOCK) - 1, 0), 0)

# Preferred: compose the shift into the final lane formula.
prev_upstream = tl.load(
    upstream_ptr + (offs - 1) * stride,
    mask=offs > 0,
    other=0.0,
)
shifted = current_operand * prev_upstream
```

## What To Verify After Applying

- Exact correctness on lane-0 and cross-chunk boundaries.
- Equality against reference for both full chunks and edge chunks.
- IR no longer shows the targeted `tt.broadcast` / `tt.reduce` or 2D shift-mask materialization path.
- End-to-end benchmark improves on representative shapes, not only micro-cases.
- The rewrite did not blindly shift destination-local operands that should remain at the current lane.
- The generated code no longer keeps unnecessary `next` loads, shifted temporary vectors, or `tl.gather` just to emulate the old mask-reduce result.

## Related Patterns

- `scalar-latency-traps`
- `exact-tile-no-boundary-fast-path`
- `algebraic-optimization`
- `layout-materialization-elision`
