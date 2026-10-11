# Causal Block Skipping Pattern

## When to Apply
- Attention operators with causal masking
- Matrix operations where a significant portion of output is zeroed by mask
- Any operator where entire blocks of computation can be proven unnecessary

## Impact
- Causal attention: up to 50% theoretical reduction (approaching (n-1)/2n for sequence length n)
- Measured: 45.5% speedup at sequence length 32K on Ascend 910B3

## Core Insight

In causal (lower-triangular) attention, for any S2 block starting at column `s2_start`:
```
If s2_start > s1_end (the last query row in the current S1 block):
    → ALL entries in this S2 block will be masked to -inf
    → softmax(-inf) = 0 → this block contributes ZERO to output
    → SKIP the entire block (no load, no compute, no store)
```

## Code Template

### Host-side: Compute valid S2 range per S1 block
```cpp
// For causal mask: query at row i can only attend to keys at columns [0, i]
// For S1 block [s1_start, s1_end], valid S2 blocks are those where s2_start <= s1_end

// In tiling:
for (int s1Block = 0; s1Block < numS1Blocks; s1Block++) {
    int s1End = (s1Block + 1) * blockSize - 1;
    int validS2Blocks = (s1End / blockSize) + 1;  // Only need S2 blocks 0..validS2Blocks-1
    tiling.set_validS2Count(s1Block, validS2Blocks);
}
```

### Kernel-side: Skip invalid blocks
```cpp
// In the S2 loop:
for (int s2Block = 0; s2Block < totalS2Blocks; s2Block++) {
    int s2Start = s2Block * blockSize;
    if (s2Start > s1End) {
        break;  // All remaining S2 blocks are fully masked → skip
    }
    // Process this S2 block (may still need partial masking within the block)
    ProcessBlock(s1Block, s2Block);
}
```

## Performance Model
```
Without skip: time = H × (heads/cores) × S2_blocks_per_head
With skip:    time = H × (heads/cores) × avg_valid_S2_blocks

For causal mask:
  avg_valid_S2_blocks ≈ (n+1)/2  where n = total S2 blocks
  Theoretical reduction = 1 - (n+1)/(2n) → approaches 50% for large n

Practical measurement at different sequence lengths:
  S=512:   ~22% reduction (n=4 blocks)
  S=2048:  ~37% reduction (n=16 blocks)
  S=8192:  ~43% reduction (n=64 blocks)
  S=32768: ~45.5% reduction (n=256 blocks)
```

## Generalization: Any Block-Sparse Pattern
```
The same principle applies to ANY block-level sparsity:
- Band attention: skip blocks outside the band
- Sliding window: skip blocks outside the window
- Custom masks: precompute block-level skip table on host

Key requirement: the skip decision must be computable at BLOCK granularity,
not requiring per-element evaluation.
```

## GQA (Grouped Query Attention) Considerations
```
When num_q_heads > num_kv_heads (GQA):
- Each KV head serves multiple Q heads
- The valid S2 range must account for the GROUP of Q heads sharing each KV head
- Token computation formula changes:

  Grouped layout: token = s1_row (within the S1 block for current Q head)
  NOT: token = group_offset + s1_row (incorrect for grouped KV sharing)

  The s1 row index determines the causal boundary,
  NOT the global head index.
```

## Common Mistakes

1. **Off-by-one in block boundary**: `s2Start > s1End` not `s2Start >= s1End` — the diagonal block is partially valid
2. **Forgetting partial mask within diagonal block**: Even with block-level skip, the block on the diagonal needs element-level masking
3. **GQA token formula**: Using global head index instead of per-group row index (see above)
4. **Not updating both cube and vector sides**: If skipping a cube block, must also skip the corresponding vector post-processing
