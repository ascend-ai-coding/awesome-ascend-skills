# Reduction Operations Optimization Insights

## Operator Types
- **Full reduction**: Sum, Mean, Max, Min (output is scalar or reduces entire dimension)
- **Partial reduction**: LayerNorm, RMSNorm, Softmax, BatchNorm (reduce along specific dims)
- **Segment reduction**: Segment sum/mean, grouped operations

## Key Algorithmic Optimizations

### 1. Two-Pass vs One-Pass Algorithms

#### Two-Pass (Standard)
```
Pass 1: Compute statistics (mean, variance)
Pass 2: Normalize using statistics

Problem: Reads input data TWICE from GM → 2× memory traffic
```

#### One-Pass (Welford's Algorithm)
```
Maintain running mean and variance:
  for each x:
    count += 1
    delta = x - mean
    mean += delta / count
    M2 += delta * (x - mean)
  variance = M2 / count

Benefit: Reads input data ONCE → 50% memory traffic reduction
Caveat: More compute per element, numerically less stable for large sequences
```

#### Decision Guide
```
If reduction_dim_size < UB_capacity:
  → Load entire reduction dim to UB, two-pass is fine (data stays in SRAM)

If reduction_dim_size > UB_capacity:
  → One-pass (Welford) saves significant GM traffic
  → OR: tile reduction dim with partial sums, then merge
```

### 2. Tree Reduction Pattern
```
Standard:  [a0, a1, a2, a3, a4, a5, a6, a7]
           → a0+a1+a2+a3+a4+a5+a6+a7  (sequential, 7 steps)

Tree:      [a0+a1, a2+a3, a4+a5, a6+a7]
           → [(a0+a1)+(a2+a3), (a4+a5)+(a6+a7)]
           → final  (log2(8)=3 steps, better numerical stability)

On Ascend: Use ReduceSum/ReduceMax API which implements tree reduction internally
```

### 3. Softmax Optimization
```
Standard softmax (numerically unstable):
  softmax(x_i) = exp(x_i) / sum(exp(x_j))

Safe softmax (subtract max):
  m = max(x)
  softmax(x_i) = exp(x_i - m) / sum(exp(x_j - m))

Online softmax (single pass, for tiled computation):
  Process tiles of x:
    For each tile:
      m_new = max(m_old, tile_max)
      sum_new = sum_old * exp(m_old - m_new) + tile_sum_exp(tile - m_new)
    Final: softmax = exp(x - m_final) / sum_final
```

### 4. LayerNorm Fusion
```
LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta

Fused implementation:
  Pass 1 (in UB): compute mean and var while loading x
  Pass 2 (in UB): normalize and apply affine, write output

If gamma/beta are small (D <= UB), keep them in UB across all row tiles.
```

## AscendC-Specific Patterns

### Small-D Multi-Row Merging (P3)
```
When D is small (e.g., D=64, hidden_size=128):
  Single row doesn't fill UB → poor utilization

Solution: Process multiple rows per tile
  mergedRows = min(N, UB_capacity / (D * sizeof(dtype) * BUFFER_NUM))

  Benefits:
  - Better UB utilization
  - Fewer DMA operations (larger transfers)
  - Better vector instruction utilization
```

### Cross-Core Reduction
```
When reduction spans more tiles than one core can handle:
  Option 1: Each core reduces its portion → write partial results to GM → final reduction
  Option 2: Use workspace for atomic accumulation (slower but simpler)

For LayerNorm/RMSNorm: Usually N (batch×seq) is large enough to assign
different rows to different cores, avoiding cross-core reduction entirely.
```
