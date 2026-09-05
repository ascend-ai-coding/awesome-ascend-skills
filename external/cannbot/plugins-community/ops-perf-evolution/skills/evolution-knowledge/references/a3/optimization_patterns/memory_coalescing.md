# Memory Coalescing Pattern

## When to Apply
- Operators with strided memory access (e.g., transpose, gather, channel-last formats)
- DMA efficiency analysis shows many small transfers (< 256 bytes)
- Profiling shows low bandwidth utilization

## Impact
- 10-40% speedup for operators with non-contiguous access patterns

## Key Principle

DMA (MTE2/MTE3) achieves best throughput with large, contiguous transfers.
Small or strided transfers waste DMA setup cycles.

```
Good:  DataCopy(dst, src, count=4096)         → 1 DMA op, high throughput
Bad:   for i in 0..127: DataCopy(dst+i*S, src+i*S, count=32)  → 128 DMA ops, low throughput
```

## Techniques

### 1. Tile Along Contiguous Dimension
```
Memory layout: [B, N, S, D] row-major
  → D is the innermost (contiguous) dimension
  → Tile along D first for large contiguous transfers
  → Then tile along S

Bad:  tile along N first → each tile is D elements apart in memory
Good: tile along D first → each tile is contiguous in memory
```

### 2. Use DataCopyPad for Non-Aligned Access
```cpp
// When source data is not 32-byte aligned:
DataCopyParams params;
params.blockCount = 1;
params.blockLen = actualLen;
DataCopyPad(dst, src, params);  // Handles alignment internally
```

### 3. Reorganize Data Layout in UB
```cpp
// Load contiguous chunks, then reorganize in UB (fast SRAM operation)
// rather than doing strided loads from GM (slow HBM operation)

// Step 1: Contiguous load
DataCopy(ubBuf, gmSrc, contiguousChunkSize);

// Step 2: In-UB reorganization (vector scatter/gather)
// Much faster than strided GM access
Gather(ubDst, ubBuf, indices, count);
```

### 4. Fuse Multiple Small Accesses
```cpp
// Before: Multiple small reads
for (int i = 0; i < K; i++) {
    DataCopy(ubSlice, gmInput + offsets[i], smallSize);
    ProcessSlice(ubSlice);
}

// After: One large read + UB slicing
DataCopy(ubFull, gmInput, fullSize);
for (int i = 0; i < K; i++) {
    ProcessSlice(ubFull + localOffsets[i]);  // Already in UB
}
```

## DMA Efficiency Thresholds
```
Transfer size:
  < 32 bytes:   Extremely inefficient (alignment overhead dominates)
  32-256 bytes:  Poor (DMA setup cost significant)
  256-4096:      Moderate (acceptable for most cases)
  > 4096:        Good (bus bandwidth well utilized)
  > 65536:       Excellent (near peak throughput)
```

## Common Mistakes
1. **Transposing in GM**: Never read transposed data element by element from GM. Load a contiguous block and transpose in UB.
2. **Ignoring DataCopyPad cost**: DataCopyPad handles alignment but is slower than aligned DataCopy. If possible, ensure source alignment.
3. **Over-tiling**: Creating too many tiny tiles can make every DMA transfer small. Balance tile count vs. tile size.
