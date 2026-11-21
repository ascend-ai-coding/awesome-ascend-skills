# Pipeline Overlap Optimization

## When to Apply
- Kernel already has double buffering but still shows pipeline stalls
- Profiling shows MTE2/MTE3 and VECTOR have poor overlap
- Want to squeeze more performance from an already-optimized kernel

## Key Concepts

### Overlap Types (from profiling)
```
VEC-MTE2 overlap: VECTOR computes while MTE2 loads next data
  → Primary goal of double buffering
  → Good: > 60% overlap

MTE2-MTE3 overlap: Load and store happening simultaneously
  → Can cause bus contention (shared HBM bus)
  → Moderate: 10-30% overlap is acceptable
  → High (>30%): May indicate bus contention, consider staggering

VEC-MTE3 overlap: VECTOR computes while MTE3 stores previous results
  → Secondary benefit of double buffering
  → Good: > 40% overlap
```

### Pipeline Bubble Categories
```
NORMAL:     ≤1ps issue gap, unavoidable
STRUCTURAL: Pipeline drain/barrier/icache miss
DATA_STALL: Waiting for data (MTE2/MTE3)
SCALAR:     Address computation/parameter loading blocking execution

Priority: DATA_STALL > SCALAR > STRUCTURAL > NORMAL
```

## Optimization Techniques

### 1. Reduce Pipeline Barriers
```cpp
// Before: Unnecessary barrier between independent operations
Compute(tile);
PipeBarrier();  // ← Forces ALL pipes to synchronize
CopyOut(tile);

// After: Use SetFlag/WaitFlag for fine-grained sync
Compute(tile);
SetFlag<HardEvent::V_MTE3>(pipe_flag);  // Signal VECTOR done
WaitFlag<HardEvent::V_MTE3>(pipe_flag); // MTE3 waits for VECTOR
CopyOut(tile);
```

### 2. Prefetch Next Tile
```cpp
// Start loading next tile before current compute finishes
for (int i = 0; i < tileNum; i++) {
    // These three can overlap due to double buffering:
    CopyIn(i);      // MTE2 loads tile i
    Compute(i);     // VECTOR processes tile i-1 (from previous iteration)
    CopyOut(i);     // MTE3 stores tile i-2 (from two iterations ago)
}
```

### 3. Balance MTE2 and MTE3 Sizes
```
If MTE2 transfer >> MTE3 transfer:
  → VECTOR waits for MTE2 (data stall)
  → Solution: Reduce input size or increase output processing

If MTE3 transfer >> MTE2 transfer:
  → MTE3 blocks next iteration's MTE2 (bus contention)
  → Solution: Reduce output size or merge store operations
```

## Profiling-Guided Optimization

### Reading Overlap Metrics
```json
{
  "overlap_status": "partial_overlap",
  "vec_mte2_overlap_pct": 35.2,
  "vec_mte3_overlap_pct": 22.1,
  "mte2_mte3_overlap_pct": 8.5
}
```

### Decision Table
| vec_mte2_overlap | Action |
|-----------------|--------|
| < 20% | Double buffering not working → check BUFFER_NUM and tile sizes |
| 20-50% | Partially working → increase tile size or reduce compute |
| > 50% | Good overlap → look for other bottlenecks |

## Common Mistakes
1. **Over-synchronizing**: Too many PipeBarrier() calls destroy overlap
2. **Ignoring bus contention**: Maximizing all overlaps simultaneously can cause resource conflicts
3. **Wrong BUFFER_NUM for Queue**: VECIN needs BUFFER_NUM=2 for double buffering, but VECCALC temp buffers can be 1
