# Double Buffering Pattern

## When to Apply
- Kernel is memory-bound (MTE2 active >> VECTOR active)
- Current kernel uses single buffer (BUFFER_NUM=1)
- UB has enough space for 2× buffer allocation

## Impact
- Typical: 20-80% speedup for memory-bound kernels
- Overlaps data transfer with computation: while VECTOR processes buffer A, MTE2 loads buffer B

## Code Template

### Before (Single Buffer)
```cpp
// Tiling
TQue<QuePosition::VECIN, 1> inQueue;    // Single buffer
TQue<QuePosition::VECOUT, 1> outQueue;

// Process loop
for (int i = 0; i < tileNum; i++) {
    CopyIn(i);    // MTE2: GM → UB (VECTOR idle!)
    Compute(i);   // VECTOR: compute (MTE2 idle!)
    CopyOut(i);   // MTE3: UB → GM (VECTOR idle!)
}
```

### After (Double Buffer)
```cpp
// Tiling: halve tile size, double buffer count
TQue<QuePosition::VECIN, 2> inQueue;    // Double buffer
TQue<QuePosition::VECOUT, 2> outQueue;

// Process loop — pipeline fills naturally via EnQue/DeQue
for (int i = 0; i < tileNum; i++) {
    CopyIn(i);    // MTE2: loads tile i into buffer (i%2)
    Compute(i);   // VECTOR: processes tile (i-1) from buffer ((i-1)%2)
    CopyOut(i);   // MTE3: stores tile (i-2) from buffer ((i-2)%2)
}
// Note: EnQue/DeQue automatically manages producer-consumer synchronization
```

## Pipeline Timeline Comparison
```
Single buffer:
  MTE2: [LOAD0]          [LOAD1]          [LOAD2]
  VEC:         [COMP0]          [COMP1]          [COMP2]
  MTE3:               [STORE0]        [STORE1]        [STORE2]
  Time: =========================>

Double buffer:
  MTE2: [LOAD0][LOAD1][LOAD2][LOAD3]...
  VEC:         [COMP0][COMP1][COMP2]...
  MTE3:               [STORE0][STORE1]...
  Time: ============>  (significantly shorter)
```

## UB Memory Impact
```
Single buffer: tile_size × sizeof(dtype) × num_tensors
Double buffer: tile_size × sizeof(dtype) × num_tensors × 2

If UB is tight, halve tile_size when enabling double buffering:
  new_tile = old_tile / 2  (rounded down to 32-byte boundary)
```

## Common Mistakes

1. **Forgetting to halve tile size**: Double buffer needs 2× memory. If tile stays the same, UB overflows silently.
2. **Missing PipeBarrier removal**: After enabling double buffering, remove unnecessary PipeBarrier() calls between CopyIn/Compute/CopyOut — the EnQue/DeQue mechanism handles synchronization.
3. **Incorrect tail handling**: With halved tiles, the number of tiles doubles. Ensure tail tile handling is updated.
4. **Not adjusting tiling parameters**: Host-side tiling must also compute halved tile sizes and doubled tile counts.

## Feasibility Check
```
Required UB = tile_size × sizeof(dtype) × buffer_num × pipe_count
Available UB = 192KB (910B3)

Example: FP16 with 2 tensors (in+out), double buffer
  = 12288 × 2 × 2 × 2 = 98304 bytes = 96KB [PASS] (fits in 192KB)
```
