# Tiling Strategies

## When to Apply
- Input shapes are variable or large
- Need to distribute work across multiple cores
- Current tiling causes load imbalance or poor UB utilization

## Three Tiling Dimensions

### Split-N (Row-wise tiling)
```
Best for: Tall matrices (large N, small D)
Strategy: Each core processes a chunk of rows

tileN = ceil(N / blockDim)
tileD = D  (process full row width per tile)

Pros: Simple, good locality
Cons: Poor balance if N not divisible by blockDim
```

### Split-D (Column-wise tiling)
```
Best for: Wide matrices (small N, large D) or D > UB capacity
Strategy: Each tile processes a slice of columns, needs reduction across tiles

tileN = N  (or small chunk)
tileD = UB_capacity / sizeof(dtype) / BUFFER_NUM

Pros: Handles large D
Cons: May need cross-tile reduction (e.g., for sum/max), more complex
```

### Split-ND (2D tiling)
```
Best for: Both N and D are large
Strategy: Tile in both dimensions

tileN × tileD ≤ UB_capacity / sizeof(dtype) / BUFFER_NUM
Optimize: maximize tileD first (better vectorization), then adjust tileN

Pros: Flexible, good UB utilization
Cons: Most complex, 2D loop structure
```

## Multi-Core Distribution

### Round-Robin Assignment
```cpp
// Simple: core i processes tiles i, i+blockDim, i+2*blockDim, ...
uint32_t startTile = GetBlockIdx();
uint32_t step = GetBlockNum();
for (uint32_t t = startTile; t < totalTiles; t += step) {
    ProcessTile(t);
}
```

### Contiguous Assignment
```cpp
// Better locality: core i processes tiles [start_i, end_i)
uint32_t tilesPerCore = (totalTiles + GetBlockNum() - 1) / GetBlockNum();
uint32_t startTile = GetBlockIdx() * tilesPerCore;
uint32_t endTile = min(startTile + tilesPerCore, totalTiles);
for (uint32_t t = startTile; t < endTile; t++) {
    ProcessTile(t);
}
```

## Adaptive Selection (Host-side)
```cpp
// In tiling function
if (D <= UB_SINGLE_ROW_CAPACITY) {
    // Split-N: process full rows
    tiling.set_tileMode(SPLIT_N);
    tiling.set_tileN(ceil(N / blockDim));
    tiling.set_tileD(D);
} else {
    // Split-D: slice columns
    tiling.set_tileMode(SPLIT_D);
    tiling.set_tileN(1);
    tiling.set_tileD(computeMaxTileD(dtype, bufferNum));
}
```

## Load Balancing
```
Problem: N=100, blockDim=32 → 4 cores get 4 tiles, 28 cores idle
Solution 1: Use smaller blockDim (e.g., min(blockDim, N))
Solution 2: 2D tiling to create more tiles
Solution 3: Fine-grained tiles with round-robin assignment
```

## Common Mistakes
1. **Not aligning tile boundaries to 32 bytes**: `tileD = (rawTileD / alignment) * alignment`
2. **Forgetting tail tiles**: Last tile may be smaller than tileD, needs mask or padding
3. **Ignoring UB overhead**: Account for temporary buffers, not just input/output
