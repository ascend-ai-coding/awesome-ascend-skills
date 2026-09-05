# Slice Coalescing Pattern

## Summary

When the kernel performs scatter/gather operations with non-contiguous memory access
patterns, such as token rearrangement in MOE layers, sparse data processing, or any operation involving
index-based data movement, use `extract_slice` or `insert_slice` to data reuse while minimizing expensive
global memory transactions.

## Use When

- Scatter or gather style data movement dominates, and batching work in UB could replace many random global accesses with fewer contiguous transfers.
- The kernel resembles token rearrangement, sparse reordering, or other index-based movement where access direction determines whether reads or writes should be coalesced.

## Detail

### Principle and Implementation
NPUs have significantly fewer cores than GPUs, making kernel launch overhead substantial. Unlike CUDA where launching thousands of small kernels is efficient due to massive parallelism, NPUs benefit from fewer, larger kernels that process more data per core. For scatter/gather operations, use `extract_slice` and `insert_slice` to batch process data within UB before performing coalesced global memory writes. This approach transforms multiple small, random memory accesses into fewer, larger, sequential accesses. The MOE token rearrangement example demonstrates two complementary patterns: using `extract_slice` to batch-load contiguous data then scatter-write individual tokens (token reverse), and using `insert_slice` to gather scattered data into a contiguous buffer for batch writing (token rearrangement).

```python
# Pattern A: MOE Token Reverse - Batch load, scatter write
def moe_token_reverse_npu(x_ptr, indices, output_ptr, BLOCK_SIZE, D):
    # Batch load contiguous data block into UB
    data = tl.load(x_ptr + block_start + data_offset)

    for i in range(BLOCK_SIZE):
        # Extract individual token from UB (cheap operation)
        token = tl.extract_slice(data, [i, 0], [1, D], [1, 1])

        # Calculate scattered write location
        output_offset = D * tl.get_element(indices, (i,)) + tl.arange(0, D)[None, :]

        # Store to scattered location
        tl.store(output_ptr + output_offset, token)

# Pattern B: MOE Token Rearrangement - Gather data, batch write
def moe_token_rearrangement_npu(x_ptr, indices, output_ptr, BLOCK_SIZE, D):
    # Create output buffer in UB
    output_buffer = tl.full((BLOCK_SIZE, D), 0, dtype=x_ptr.type.element_ty)

    for i in range(BLOCK_SIZE):
        # Load from scattered locations (expensive global memory access)
        token_idx = tl.get_element(indices, (i,))
        data_offset = token_idx * D + tl.arange(0, D)[None, :]
        token_data = tl.load(x_ptr + data_offset)

        # Assemble into contiguous buffer in UB
        output_buffer = tl.insert_slice(output_buffer, token_data, [i, 0], [1, D], [1, 1])

    # Single coalesced store to global memory
    tl.store(output_ptr + block_offset, output_buffer)

# Key insight: Choose pattern based on memory access characteristics:
# - If reads are contiguous but writes are scattered: Use Pattern A (extract_slice for reads)
# - If reads are scattered but writes are contiguous: Use Pattern B (insert_slice for writes)
# This minimizes the number of expensive random global memory accesses
```

**Combined Insight**: These slice operations enable you to batch process data according to its memory access pattern—either loading contiguous blocks and extracting scattered elements, or gathering scattered elements into contiguous blocks for efficient writing. This compensates for NPU's limited core count by maximizing data processed per kernel while respecting UB constraints and memory access efficiency.
