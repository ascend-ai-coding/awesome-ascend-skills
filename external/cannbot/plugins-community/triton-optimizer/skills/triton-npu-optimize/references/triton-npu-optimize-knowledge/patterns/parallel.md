# Dual Vector-Core Parallel Pattern

## Summary

Use `tl.parallel` to run tasks in the two vector cores of an aicore at the same time.

## Use When

- Two independent vector-side computations happen in sequence and can be split across vector cores.
- The bottleneck is not primarily memory movement, so exposing more vector-core concurrency is more promising than reworking loads.

## Signals

### Code

- Independent type conversions, element-wise operations, or scaling steps already exist on the hot path.
- The candidate work split is compute-side and independent, rather than shared-bandwidth memory loading.

## Avoid When

- The candidate work items still share a real data dependency.
- The operation is mostly memory loading, where shared bandwidth is already the limiting factor.
- The operation is so small that `tl.parallel` overhead is likely larger than the gain.

## Detail

### Principle

Ascend NPU has 2 vector cores per AICore. Use `tl.parallel()` to utilize them.

**What to parallelize:**
- ✅ **Independent operations**: A and B scaling (best choice)
- ✅ Type conversions (int8→fp32, fp32→fp16)
- ✅ Element-wise operations (activations, multiplications)
- ✅ Output dtype conversion and stores

**What NOT to parallelize:**
- ❌ Memory loads (shared bandwidth, hardware already parallelizes)
- ❌ Very fast operations (overhead > benefit)
- ❌ Operations with data dependencies

**Key insight:**
```python
# A and B scaling are INDEPENDENT - perfect for parallelization!
for core_id in tl.parallel(0, 2, bind_sub_block=True):
    if core_id == 0:
        scaled_a = a.to(fp32) * a_scales  # Vector Core 0
    else:
        scaled_b = b.to(fp32) * b_scales  # Vector Core 1
```


### Example: Parallel Vector Cores (Quantized Matmul)

**Demonstrates:** Independent A/B scaling using `tl.parallel()`

```python
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    k_remaining = K - k * BLOCK_SIZE_K

    # Load quantized matrices (int8)
    a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
    b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)

    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Load scale factors
    k_group_idx = (k * BLOCK_SIZE_K) // group_k
    a_scale_indices = k_group_idx + k_group_offset[None, :]
    b_scale_indices = k_group_idx + k_group_offset[:, None]

    a_s = tl.load(
        As + offs_am[:, None] * stride_As_m + a_scale_indices * stride_As_k,
        mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
        other=1.0
    )
    b_s = tl.load(
        Bs + b_scale_indices * stride_Bs_k + offs_bsn[None, :] * stride_Bs_n,
        mask=(offs_k[:, None] < k_remaining) & (offs_bsn[None, :] < num_groups_n),
        other=1.0
    )

    # PARALLEL SCALING: A and B on separate vector cores
    scaled_a = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    scaled_b = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

    for core_id in tl.parallel(0, 2, bind_sub_block=True):
        if core_id == 0:
            # Vector Core 0: Scale A
            scaled_a = a_s * a.to(tl.float16)
        else:
            # Vector Core 1: Scale B (runs in parallel!)
            scaled_b = b_s * b.to(tl.float16)

    # Cube Unit performs matmul on scaled matrices
    accumulator = tl.dot(scaled_a, scaled_b, acc=accumulator, allow_tf32=False)

    # Advance pointers
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
```

**Why this parallelization works:**
- A and B scaling are **completely independent** (no data dependencies)
- Each vector core works on contiguous data (good cache locality)
- Zero overhead (no extract/insert slice operations needed)
- Natural work division (one matrix per core)


## What To Verify After Applying

- Verify the two parallel branches are actually independent and do not need cross-core ordering.
- Verify `tl.parallel` is applied to compute-side work rather than shared-bandwidth loads.
- Verify end-to-end timing improves instead of only making the structure look more parallel.

### Common pitfalls

#### ❌ Parallelizing Loads
```python
# BAD: Memory bandwidth is shared!
for core in tl.parallel(0, 2):
    if core == 0:
        a = tl.load(...)  # No benefit, just overhead
```

#### ✅ Parallelizing Compute
```python
# GOOD: Vector cores are independent
for core in tl.parallel(0, 2):
    if core == 0:
        scaled_a = a.to(fp32) * scales  # Parallel execution
```
