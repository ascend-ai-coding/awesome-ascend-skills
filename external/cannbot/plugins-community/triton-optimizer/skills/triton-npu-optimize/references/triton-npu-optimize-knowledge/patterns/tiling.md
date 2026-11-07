# Hierarchical Tiling Optimization Pattern (UB Overflow Prevention)

## Summary

Reduce per-program working-set size through hierarchical or sub-block tiling, keeping live data within UB capacity.

## Use When

- Block sizes, live intermediates, or multi-tensor loads risk UB overflow or poor locality.
- The main problem is working-set size and memory footprint, not the need for a completely different kernel structure.

## Signals

### Code

- Large `BLOCK_SIZE` values, multiple tensor loads, or heavy intermediates keep too much data live per program.
- The kernel already has a reasonable overall structure, but it still needs smaller sub-blocks to control UB usage.
- Runtime failures or memory access violations appear when block sizes increase on NPU.

## Problem Description

**Root Cause:**
- Ascend NPU has limited on-chip Unified Buffer (192KB on Atlas 800T A2/A3).
- When multiple tensors or complex operations are loaded simultaneously, UB usage can overflow.
- Large `BLOCK_SIZE` values amplify the issue by increasing the number of live elements held per program instance.

**Symptoms:**
- Runtime errors indicating UB overflow
- Kernel failures with large block sizes
- Memory access violations on NPU

## Optimization Strategy

Introduce **hierarchical tiling** (also called sub-blocking) to further subdivide large blocks:

Choose this pattern when the main problem is **working-set size**. The question it answers is:

- how should the kernel reduce per-program memory footprint so tiles and intermediates fit UB safely

If the kernel should first be re-expressed as a standard tiled matmul, prefer `classic-matmul`.
If the tiled loop already exists and the remaining problem is poor memory/compute overlap, prefer `software-pipeline`.

### Key Principles

1. **Two-level blocking**: Separate task scheduling from memory management
   - Keep main `BLOCK_SIZE` for task scheduling (coreDim compliance)
   - Introduce `BLOCK_SIZE_SUB` for processing data in smaller batches

2. **Process in loops**: Use inner loops to process sub-blocks sequentially
   - Reduce peak memory usage by processing data in smaller chunks
   - Maintain reasonable coreDim values for efficient task scheduling
   - Control UB usage through smaller batch sizes

3. **Balance performance and memory**:
   - Small enough to fit within UB capacity
   - Large enough to maintain reasonable performance
   - Aligned with memory access patterns (32-byte alignment)

## Detection Pattern

A kernel that loads multiple large tensors in a single flat block risks UB overflow:

### Problematic Code Patterns

```python
# Problem: Large block size + multiple loads = UB overflow
@triton.jit
def kernel(inp, mask1, mask2, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Loading multiple large tensors at once
    mask = offsets < N
    data1 = tl.load(inp + offsets, mask=mask)
    data2 = tl.load(mask1 + offsets, mask=mask)
    data3 = tl.load(mask2 + offsets, mask=mask)

    # UB overflow here!
    result = complex_operation(data1, data2, data3)
    tl.store(out + offsets, result, mask=mask)
```

## Code Example

### Before Optimization (UB Overflow)

```python
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Loading all data at once causes UB overflow
    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    cur_inp = tl.load(inp + offsets, mask=(~fill_mask) & mask, other=0)

    tl.store(out + offsets, cur_inp, (~fill_mask) & mask)
    tl.store(out + offsets, value, fill_mask & mask)
```

**Issues:**
- Single large block loads all data at once
- Multiple tensors occupy UB simultaneously
- Risk of overflow with large BLOCK_SIZE values

### After Optimization (Hierarchical Tiling)

```python
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N,
                      BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE

    # Calculate the number of sub-blocks to process
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

    # Process in blocks to avoid UB overflow
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N

        # Load and process data in batches (smaller UB footprint)
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        # First write the original data
        tl.store(out + offsets, input_vals, mask=mask)

        # Then overwrite the target value at positions that need filling
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        final_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, final_vals, mask=mask)
```

**Improvements:**
- Data processed in smaller sub-blocks
- Reduced peak UB usage
- Maintains task scheduling efficiency with large main BLOCK_SIZE

### Host Code Configuration

```python
def masked_fill(inp, mask, value):
    N = inp.numel()

    # Two-level blocking strategy
    MAIN_BLOCK_SIZE = 32768  # Ensure coreDim compliance (N / 32768 < 65535)
    SUB_BLOCK_SIZE = 1024    # Control UB usage (process in smaller chunks)

    grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)
    masked_fill_kernel[grid](inp, mask, value, out, N,
                           MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE)
    return out
```

## Guidelines for Sub-Block Size Selection

**SUB_BLOCK_SIZE should be:**

1. **UB-safe**: Small enough to fit within UB capacity
   - Simple element-wise operations: 1024-2048
   - Operations with multiple tensors: 512-1024
   - Complex reductions: 256-512

2. **Performance-aware**: Large enough to maintain reasonable performance
   - Avoid too-small blocks that increase loop overhead
   - Balance between memory usage and computation efficiency

3. **Alignment-aware**: Divisible by or aligned with memory access patterns
   - 32-byte alignment requirement
   - Vector width considerations (typically 128/256 elements)

## Avoid When

1. **Small BLOCK_SIZE** No significant memory pressure
2. **Simple operations** with single tensor - UB usage is minimal
3. **Already optimized** with sub-blocking present
4. **Structure is the real problem** - if the current kernel is really a manual matmul or reduction that should first become a regular tiled `tl.dot` loop

## What To Verify After Applying

- Verify the chosen `BLOCK_SIZE_SUB` fits the operation type and keeps the working set UB-safe.
- Verify the inner sub-block loop actually reduced peak live data instead of only adding loop overhead.
- Verify both kernel signature and host launch code pass the new block-size parameters consistently.

## Expected Performance Impact

**Memory Usage:**
- UB usage reduced by factor of `BLOCK_SIZE / BLOCK_SIZE_SUB`
- Enables processing larger arrays without overflow

**Performance Trade-offs:**
- **Pros**: Enables kernels that would otherwise overflow UB
- **Cons**: Small loop overhead for sub-block iteration (typically < 5%)
- **Net effect**: Enables functionality that was previously impossible

**Typical Results:**
- UB overflow kernels become functional
- Performance impact: -5% to +10% depending on operation complexity
- Enables larger batch sizes and more efficient task scheduling

## Related Patterns

- `classic-matmul`: use it first when the real problem is that a manual reduction should become a tiled matmul structure at all.
- `software-pipeline`: combine it only after the footprint already fits UB, because pipelining deliberately keeps multiple tiles live.

## Code Transformation Pattern

**Step 1: Add sub-block parameter**
```python
# Before
def kernel(..., BLOCK_SIZE: tl.constexpr):

# After
def kernel(..., BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
```

**Step 2: Calculate sub-block count**
```python
num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
```

**Step 3: Wrap computation in loop**
```python
for sub_block_idx in range(num_sub_blocks):
    # Compute sub-block offsets
    sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
    offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)

    # Load, compute, store for this sub-block
    ...
```

**Step 4: Update host code**
```python
# Before
kernel[grid](..., BLOCK_SIZE)

# After
kernel[grid](..., BLOCK_SIZE, BLOCK_SIZE_SUB)
```
