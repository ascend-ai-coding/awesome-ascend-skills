# Software Pipelining and Block Pointer Optimization Pattern

## Summary

Improve overlap between memory movement and compute in a hot loop that is already structurally tiled, typically by combining block pointers, prefetching, and pipelined loop structure.

## Use When

- The hot loop already has a real tiled structure, but loads and computation still happen too serially.
- Profiling suggests wait-heavy or overlap-poor behavior, and the next question is pipeline quality rather than basic kernel structure.

## Signals

### Code

- The loop is already tiled, but each iteration still follows a mostly synchronous load-then-compute rhythm.
- Manual pointer arithmetic dominates the tiled loop, and block-pointer plus prefetch structure is still missing.

### Profile

- `msprof` timelines show Cube or Vector gaps while the MTE engines fetch the next tile.
- Wait-heavy behavior suggests insufficient memory/compute overlap rather than a missing tiled-kernel rewrite.

## Problem Description

Huawei Ascend NPUs (DaVinci architecture) use a **Decoupled Access-Compute (DAC)** design. This means memory movement (MTE engines) and computation (Cube/Vector cores) happen on different hardware units.

In standard Triton code, the kernel often executes synchronously: it waits for a data load to finish before starting the calculation. This results in "stalls" or "gaps" in the `msprof` timeline where the expensive Cube/Vector cores sit idle while waiting for the MTE engine to fetch data from Global Memory (HBM).

## Optimization Strategy

Implement **Software Pipelining (Double Buffering)** combined with **Block Pointers**. This allows the NPU to fetch the *next* tile of data from memory while simultaneously performing computation on the *current* tile.

Choose this pattern when the main problem is **overlap**, not basic kernel structure. The question it answers is:

- can an already tiled loop overlap memory transfer with compute more effectively

If the hot loop is still fundamentally manual reduction code and has not yet become a regular tiled `tl.dot` loop, prefer `classic-matmul` first.
If the main issue is UB overflow or an oversized working set rather than pipeline gaps, prefer `tiling`.

### Key Principles

1.  **Use `tl.make_block_ptr`**: Replaces manual pointer arithmetic. It allows the hardware to utilize specialized 2D DMA controllers, reducing Scalar Unit overhead.
2.  **Pre-fetching**: Load the first tile (or first few tiles) before entering the main loop.
3.  **Overlapped Loop**: Inside the loop, load data for iteration `i+1` before (or during) the computation of iteration `i`.
4.  **Pointer Advancement**: Use `tl.advance` to move pointers forward. This is more efficient for the Scalar unit than recalculating offsets.

### Important Notes

-   **UB Size Constraints**: Pipelining requires keeping multiple tiles in the Unified Buffer (UB) simultaneously. Ensure `2 * (Tile_M * Tile_K * dtype_size)` fits within the available UB (usually 192KB-256KB).
-   **Loop Latency**: Pipelining is most effective when the computation time and memory transfer time are roughly balanced.

## Detection Pattern

Look for code patterns where loads and math are interleaved sequentially inside a loop:

```python
# Problematic: Synchronous "Load-then-Compute"
for k in range(0, K, BLOCK_SIZE_K):
    # Scalar unit calculates offsets here
    a = tl.load(a_ptr + k * stride + offsets)
    b = tl.load(b_ptr + k * stride + offsets)
    # Cube core waits for the loads above to complete
    res = tl.dot(a, b, res)
```

## Optimization Example

### Before Optimization

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # ... (program ID and offset logic)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # Sequential Load (MTE2)
        a = tl.load(a_ptr + k_offsets)
        b = tl.load(b_ptr + k_offsets)
        # Sequential Compute (Cube) - Waits for MTE2
        accumulator = tl.dot(a, b, accumulator)
```

### After Optimization (Pipelined with Block Pointers)

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # 1. Initialize Block Pointers (Hardware optimized)
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(K, 1),
                                   offsets=(0, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(N, 1),
                                   offsets=(0, 0), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0))

    # 2. Prefetch first tile (Start MTE2 transfer early)
    a_tile = tl.load(a_block_ptr)
    b_tile = tl.load(b_block_ptr)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # 3. Advance pointers for the NEXT iteration
        a_block_ptr = tl.advance(a_block_ptr, [0, BLOCK_SIZE_K])
        b_block_ptr = tl.advance(b_block_ptr, [BLOCK_SIZE_K, 0])

        # 4. Compute CURRENT tile while loading NEXT tile
        # The Triton compiler maps this to overlapped Cube and MTE instructions
        accumulator = tl.dot(a_tile, b_tile, accumulator)

        # Load next tile (Async)
        a_tile = tl.load(a_block_ptr)
        b_tile = tl.load(b_block_ptr)
```

## Avoid When

1.  **Tiny Inner Loops**: If the loop only runs 1 or 2 times, the pre-fetch overhead might exceed the savings.
2.  **Extreme Memory Pressure**: If the tile size is so large that the Unified Buffer cannot hold two sets of tiles (current and next).
3.  **Dependency Chains**: If Tile `i+1` depends on the result of the computation of Tile `i`.
4.  **Pre-tiling rewrite still needed**: If the loop should first be rewritten into a regular tiled matmul or another clearer tile-based structure.

## What To Verify After Applying

- Verify `tl.make_block_ptr` and `tl.advance` really replaced the raw pointer arithmetic on the hot path.
- Verify the first tile is prefetched before the overlapped loop starts.
- Verify the loop computes on the previously loaded tile while fetching the next one.
- Verify the total memory used by all active tiles still fits NPU UB capacity.

## Related Patterns

- `classic-matmul`: use it first when the hot loop is still manual reduction code rather than a regular tiled `tl.dot` loop.
- `tiling`: use it first when overlap would require more live data than UB can hold, or when footprint reduction is still the main problem.
