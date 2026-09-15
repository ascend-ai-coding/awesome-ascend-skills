# NPU Load Order Optimization Pattern

## Summary

Reorder independent loads so false sequencing does not block memory-level parallelism or create avoidable wait time in a memory-bound kernel.

## Problem Description

On Huawei Ascend NPU devices, the compiler preserves the exact execution order of load instructions as specified in the code. When load instructions are blocked by data dependencies from previous operations, independent load instructions cannot be issued in parallel, leading to suboptimal hardware utilization and reduced performance.

## Optimization Strategy

Reorganize load instruction sequences to maximize parallel execution by:
1. **Identifying independent loads**: Find load operations that have no data dependencies with preceding instructions
2. **Promoting early issuance**: Move independent load instructions as early as possible in the execution flow
3. **Breaking dependency chains**: Separate loads from the operations that create their dependencies

### Key Principles

1. **Preserve data dependencies**: Never change the semantic meaning or create race conditions
2. **Maximize instruction-level parallelism**: Enable concurrent execution of independent memory operations
3. **Leverage NPU memory hierarchy**: Utilize the full memory bandwidth by overlapping multiple load operations
4. **Consider loop-carried dependencies**: Pay special attention to dependencies across loop iterations

## Detection Pattern

Look for code patterns like:

```python
# Problematic: Dependent load blocking independent load
store B          # Cycle N
load B           # Cycle N+1 (waits for store B)
load A           # Cycle N+1 (cannot start until load B completes)

# Problematic: Serialized loads due to false dependencies
result = computation()  # Creates dependency
load X                  # Waits for computation
load Y                  # Waits for load X (but could start earlier)
```

## Optimization Example

### Before Optimization (Serialized Execution)

```python
@triton.jit
def processing_kernel(A_ptr, B_ptr, B_index_ptr, O_ptr,
                     B_DIM: tl.constexpr, HEAD_NUM: tl.constexpr, HEAD_DIM: tl.constexpr):
    i_n = tl.program_id(0)
    i_range = tl.arange(0, B_DIM)

    for i in range(HEAD_NUM):
        p_A = A_ptr + i * HEAD_DIM + i_n * B_DIM + i_range
        p_O = O_ptr + i * HEAD_DIM + i_n * B_DIM + i_range
        p_B_index = B_index_ptr + i

        # ❌ Problem: load B blocks load A due to loop-carried dependency
        idx_B = tl.load(p_B_index)      # Depends on previous store B
        p_B = B_ptr + idx_B
        b_B = tl.load(p_B)              # Memory access 1

        b_A = tl.load(p_A)               # Memory access 2 (serialized)

        # Calculation and storage
        b_B += tl.sum(b_A)
        b_O = b_A * b_B
        tl.store(p_O, b_O)
        tl.store(p_B, b_B)               # Creates dependency for next iteration
```

### After Optimization (Parallel Execution)

```python
@triton.jit
def processing_kernel(A_ptr, B_ptr, B_index_ptr, O_ptr,
                     B_DIM: tl.constexpr, HEAD_NUM: tl.constexpr, HEAD_DIM: tl.constexpr):
    i_n = tl.program_id(0)
    i_range = tl.arange(0, B_DIM)

    for i in range(HEAD_NUM):
        p_A = A_ptr + i * HEAD_DIM + i_n * B_DIM + i_range
        p_O = O_ptr + i * HEAD_DIM + i_n * B_DIM + i_range
        p_B_index = B_index_ptr + i

        # ✅ Optimization: Load independent data first
        b_A = tl.load(p_A)               # Memory access 1 (can start immediately)

        # Load dependent data (must wait for previous store)
        idx_B = tl.load(p_B_index)        # Depends on previous store B
        p_B = B_ptr + idx_B
        b_B = tl.load(p_B)                # Memory access 2

        # Calculation and storage
        b_B += tl.sum(b_A)
        b_O = b_A * b_B
        tl.store(p_O, b_O)
        tl.store(p_B, b_B)
```

## Performance Impact

**Before optimization:**
```
Cycle N:   store B (previous iteration)
Cycle N+1: load B → load A (serialized)
```

**After optimization:**
```
Cycle N:   store B (previous iteration)
Cycle N+1: load A → load B (parallelizable)
           ↑
           load A can overlap with store B completion
```

## Use When

1. **Loop-carried dependencies**: When current iteration depends on previous iteration's store
2. **Multiple independent loads**: When several load operations have no data dependencies
3. **Memory-bound kernels**: Where memory latency is the performance bottleneck
4. **NPU targets**: Particularly beneficial for NPU's memory execution model

## Signals

### Code

- Independent load operations are delayed behind unrelated computations or loop-carried dependencies.
- The hot loop contains several loads with no true dependency between them, but they are still issued serially.

### Profile

- The kernel behaves memory-bound, and reducing avoidable wait between independent loads looks more promising than changing arithmetic.

## Avoid When

1. **Actual data dependencies**: When the load order affects semantic correctness
2. **Very small kernels**: Where optimization overhead outweighs benefits
3. **CPU targets**: CPUs typically have out-of-order execution and hardware scheduling
4. **Complex dependency graphs**: Where reordering might create subtle race conditions

## What To Verify After Applying

- Verify each reordered load is independent from the operations it moves past.
- Verify loop-carried dependencies and semantic ordering remain correct after reordering.
- Verify the intended memory-level parallelism win actually appears in benchmark or profile evidence.
