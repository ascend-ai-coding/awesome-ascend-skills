# Merge Adjacent Stores Pattern

## Summary

Combine separate small `tl.store` operations into one wider store when the destination
addresses form a continuous interval, so the NPU emits a single vector-friendly DMA write
instead of multiple tiny transactions.

## Use When

- Multiple stores target adjacent addresses but are emitted as separate small `tl.store` operations.
- The destination addresses are provably continuous and the per-element masks are compatible.
- Profiling or code inspection shows store granularity, not load or compute, is limiting throughput.

## Avoid When

- Destination addresses are not a continuous interval — merging would change semantics.
- Masks differ across the candidate stores in a way that changes correctness.
- The combined write would exceed UB capacity or register pressure for the active tile.

## Signals

### Code

- Multiple `tl.store` calls write to addresses that differ only by a small constant offset within a contiguous block.
- A loop emits one `tl.store` per iteration even though the outputs across iterations are adjacent.
- Store instructions outnumber load instructions significantly in a copy-heavy or epilogue-heavy kernel.

## What To Verify After Applying

- Confirm semantic equivalence: the merged store produces identical output values for all lanes.
- Check that the merged mask correctly handles tail and partial blocks.
- Benchmark to confirm the wider store reduces DMA transaction count without harming other pipeline stages.

---

## Detail

### Before

```python
# Multiple small stores to adjacent addresses
for i in range(N):
    tl.store(dst + i * stride, val[i])
```

### After

```python
# One wide store over the contiguous interval
offs = base + tl.arange(0, BLOCK)
vals = compute_contiguous_values(...)
tl.store(out + offs, vals, mask=offs < N)
```
