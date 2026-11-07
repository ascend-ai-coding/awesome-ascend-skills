# Loop-Invariant Hoisting Pattern

## Summary

Apply **Loop-Invariant Code Motion (LICM)** to Triton kernels: move computations that do **not** depend on the loop induction variable out of the loop, so each iteration performs only the minimal work that truly varies.

## Use When

- The kernel has a hot inner loop (often a K loop in GEMM-like kernels).
- Each loop iteration repeats substantial pointer math, mask construction, type casts, or shape bookkeeping.
- Profiling shows scalar/control work is disproportionately high relative to useful compute.

## Signals

### Code

- Inner loop recomputes expressions of the form:
  - `base(pid, offs) + delta(loop_var)`
  - e.g. `a_ptr + offs_m*stride_am + k*stride_ak`
- Masks are rebuilt each iteration even when parts are invariant:
  - e.g. `a_mask_m = offs_m < M` is invariant, but recomputed into `a_mask` each iter.

### IR

- Repeated arithmetic chains (`muli/addi/index_cast`) inside `scf.while` / `scf.for` bodies.
- Loop bodies contain repeated `subi/minsi/maxsi` patterns for bounds handling.

### Profile

- AIV scalar dominated by `LD_XD_XN_IMM`, `ST_XD_XN_IMM`, `ADD(_IMM)`, `CMP_IMM`.
- Timeline shows CUBE waiting on flags around the loop, while AIV performs control-heavy work.

## Optimization strategy

For any expression `E(loop_var)`:

1. Split it into **loop-invariant base** and **loop-varying delta**:
   - `E(loop_var) = BASE + DELTA(loop_var)`
2. Compute `BASE` once outside the loop.
3. Compute only `DELTA` inside the loop, and combine.

This pattern has several common specializations in Triton.

## Specialization A: Pointer address-generation hoisting (formerly “hoist-base-pointers”)

### Goal

Reduce per-iteration address-generation by hoisting invariant pointer bases.

### Before

```python
k = 0
while k < K:
    k_offs = k + offs_k
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak)
    b_ptrs = b_ptr + (k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    a = tl.load(a_ptrs, ...)
    b = tl.load(b_ptrs, ...)
    acc += tl.dot(a, b)
    k += BLOCK_K
```

### After

```python
a_base = a_ptr + (offs_m[:, None] * stride_am)
b_base = b_ptr + (offs_n[None, :] * stride_bn)

k = 0
while k < K:
    k_offs = k + offs_k
    a_ptrs = a_base + (k_offs[None, :] * stride_ak)
    b_ptrs = b_base + (k_offs[:, None] * stride_bk)
    a = tl.load(a_ptrs, ...)
    b = tl.load(b_ptrs, ...)
    acc += tl.dot(a, b)
    k += BLOCK_K
```

## Specialization B: Mask / bounds hoisting (partial LICM)

### Goal

Hoist invariant parts of masks and bounds checks outside the loop.

### Example

- Invariant:
  - `a_mask_m = offs_m < M`
  - `b_mask_n = offs_n < N`
- Varying with `k_offs`:
  - `k_mask = k_offs < K`

Inside the loop build masks from precomputed invariants:

```python
a_mask_m = offs_m < M
b_mask_n = offs_n < N

k = 0
while k < K:
    k_offs = k + offs_k
    k_mask_row = k_offs[None, :] < K
    k_mask_col = k_offs[:, None] < K
    a_mask = a_mask_m[:, None] & k_mask_row
    b_mask = k_mask_col & b_mask_n[None, :]
    ...
```

## Specialization C: Closed-form window divisor (pooling)

### Goal

Replace **per-tap counting** inside a fixed-kernel window loop with an **O(1) algebraic divisor** computed from the same clip bounds used for loads.

### Before

```python
for kd in range(KERNEL_D):
    for kh in range(KERNEL_H):
        for kw in range(KERNEL_W):
            if window_mask:
                acc += load(...)
                valid_count += 1
divisor = valid_count
```

### After

```python
tstart_d, ..., d_len, h_len, w_len = clip_window(...)
divisor = closed_form_volume(start_*, tend_*, COUNT_INCLUDE_PAD, ...)
for kd in range(KERNEL_D):
    ti = tstart_d + kd
    if kd < d_len:
        ...
        acc += load(x_base + ti * stride_d + ...)
```

This is not mere hoisting: the counting loop is **eliminated**, not moved. See `simt-clip-window-closed-reduction` for padding semantics and Triton/Ascend details.

## Performance impact expectations

- Lower AIV scalar/control overhead, especially on large-K loops.
- Cleaner loop bodies can improve backend scheduling and reduce flag/wait overhead.
- LICM is typically a **low-risk, incremental** optimization: does not change math, only where expressions are computed.

## Pitfalls / risks

- **Broadcast orientation mistakes**: `[:, None]` vs `[None, :]` must be preserved.
- **Over-hoisting**: do not hoist expressions that depend on `k_offs` or other loop-varying values.
- LICM does not eliminate transform costs (e.g. ND2NZ) by itself; treat layout issues as separate patterns.

## What To Verify After Applying

1. **Correctness**: compare against reference across boundary shapes (non-multiples of block sizes).
2. **Profiler**: reduced scalar instruction mix (`LD/ST/ADD/CMP`) and improved wall time.
3. **IR sanity**: fewer repeated arithmetic ops inside loop bodies (qualitative evidence).

## Related Patterns

- Complements **`compile-hint`**: after LICM, add alignment/contiguity hints.
- Complements **`software-pipeline`**: LICM simplifies loop bodies; pipeline overlaps remaining transfer/compute.
- Complements **`remove-implicit-transpose`**: layout fixes reduce transform work; LICM reduces residual loop control cost.
- `simt-clip-window-closed-reduction` — algebraic normalizer replacement plus clip-window loads for fixed-window reduces
