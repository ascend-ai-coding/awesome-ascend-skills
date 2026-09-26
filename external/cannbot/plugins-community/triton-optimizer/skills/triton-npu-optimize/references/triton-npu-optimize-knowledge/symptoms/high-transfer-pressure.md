# high-transfer-pressure

## Summary

The round looks dominated by data movement, staging cost, or transfer-heavy execution rather than useful cube or vector work.

## Evidence To Confirm

- Profile summaries show transfer-heavy ratios, low compute saturation, or wait tied to memory movement.
- Measured movement time is far above a rough moved-bytes / bandwidth lower bound, especially when the working set should fit on chip comfortably.
- IR summaries show many transfer-dense stages or repeated data reshaping around the hot path.
- Code structure repeatedly reloads tensors, stages many intermediates, or performs gather/scatter-like movement.

## Candidate Pattern Directions

- `tiling`
- `effective-extent-tiling`
- `cache-use`
- `algebraic-optimization`
- `discrete_memory_access`
- `gather-load`
- `sliding-window-inner-w-slab-gather`
- `slice-coalesce`
- `slice-intermediate`

## Common Non-Matches

- High transfer alone does not prove the best fix is software pipelining.
- A kernel can be transfer-heavy because its structure is still scalarized or under-batched, not because the transfer order itself is wrong.
