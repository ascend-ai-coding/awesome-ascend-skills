# Intermediate Slice Processing Pattern

## Summary

When the kernel computation creates intermediate tensors that, combined with
inputs and outputs, would exceed the Unified Buffer (UB) capacity (in attention mechanisms, batch
normalization, etc), divide computation into several steps, and use `extract_slice` and `insert_slice`
to read/write into UB.

## Use When

- Intermediate tensors, rather than just inputs or outputs, are the main source of UB pressure.
- The overall algorithm is still reasonable, but staged slice processing is needed to keep temporary values within on-chip memory limits.

## Detail

### Principle and Implementation
Ascend NPUs have fixed-size Unified Buffers (192KB per core), unlike CUDA's more flexible shared memory. When performing operations like `acc = acc * scale + update`, the system may need to store multiple full-sized tensors simultaneously: the original accumulator, the scaling factors broadcasted to the same shape, the update tensor, and intermediate results. Instead of letting the compiler fail with UB overflow, manually slice the computation using `extract_slice` to process smaller chunks that fit within UB constraints. The key insight is that `extract_slice` and `insert_slice` are low-overhead operations that create views rather than copies, allowing you to work on subsets of data while maintaining the overall computation.

```python
# Example: Safe accumulator update for large tensors
def safe_accumulator_update(acc, alpha, update, BLOCK_M, HEAD_DIM):
    # Without slicing (may overflow UB):
    # result = acc * alpha[:, None] + update

    # With slicing (UB-safe):
    num_slices = 4  # Adjust based on UB capacity
    slice_size = BLOCK_M // num_slices

    for i in range(num_slices):
        offset = i * slice_size

        # Extract slices (creates views, not copies)
        acc_slice = tl.extract_slice(acc, (offset, 0), (slice_size, HEAD_DIM), (1, 1))
        alpha_slice = tl.extract_slice(alpha, [offset], [slice_size], [1])
        update_slice = tl.extract_slice(update, (offset, 0), (slice_size, HEAD_DIM), (1, 1))

        # Compute on slice (fits in UB)
        result_slice = acc_slice * alpha_slice[:, None] + update_slice

        # Update in-place
        acc = tl.insert_slice(acc, result_slice, (offset, 0), (slice_size, HEAD_DIM), (1, 1))

    return acc
```
