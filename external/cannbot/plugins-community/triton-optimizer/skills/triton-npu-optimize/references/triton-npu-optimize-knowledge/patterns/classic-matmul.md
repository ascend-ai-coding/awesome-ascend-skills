# Classic Tiled Matmul Rewrite Pattern

Use this reference when a Triton Ascend NPU kernel is logically matmul-like but the current hot loop is written as manual reduction code, row-wise multiply-plus-sum code, or scalar-heavy pointer math around the `K` axis.

## Summary

Rewrite a manual matmul or K-reduction hot loop into a regular tiled `tl.dot`-based matmul so the kernel structure matches what Ascend Triton lowers well.

## Use When

- the kernel computes an `M x N` output tile with a regular reduction over `K`
- the current implementation is effectively `sum_k A[..., k] * B[..., k]`
- profiling or IR suggests the hot loop is spending too much effort on scalar address generation or repeated reduction structure
- a block-pointer rewrite reduced one scalar chain but the full loop is still not a regular matmul
- dtype-specialized or shape-specialized paths are acceptable when one tiled regime is clearly better but a unified rewrite would change numerics too much

## Avoid When

- purely elementwise kernels
- gather/scatter dominated kernels
- tiny shapes where tile setup cost is unlikely to amortize

Choose this pattern when the main problem is **kernel structure**. The question it answers is:

- should this manual reduction loop become a regular tiled matmul at all

If the kernel is already a reasonable tiled matmul and the remaining problem is memory/compute overlap, use `software-pipeline` instead.
If the kernel is failing because tiles or intermediates are too large for UB, use `tiling` instead.

## Rewrite Goal

Turn the hot loop into the standard tiled form:

- `pid_m`, `pid_n`
- `offs_m`, `offs_n`, `offs_k`
- explicit masked `tl.load` for `A[BLOCK_M, BLOCK_K]`
- explicit masked `tl.load` for `B[BLOCK_K, BLOCK_N]`
- `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)`
- `acc += tl.dot(a, b)`
- fused bias/activation epilogue after the dot loop

## Skeleton

```python
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.static_range(0, K, BLOCK_K):
        k_idx = offs_k + k
        a_ptrs = A + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
        b_ptrs = B + k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
```

## Layout Rule

If a weight tensor is stored as `[N, K]` but the dot wants `[BK, BN]`, keep the source tensor in place and materialize the tile in the needed logical orientation through pointer math:

```python
w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_idx[:, None] * stride_wk
```

This is usually cleaner than forcing a transpose-like transform after load.

## Precision Rule

Separate these choices:

1. input tile dtype
2. accumulator dtype
3. final store dtype

On Ascend Triton, make the two `tl.dot` operand dtypes match, but do not lower them to `fp16` by default when the real operator inputs are already `fp32` or `fp16`. The default pattern is:

- keep `a` and `b` in the loaded input dtype when both sides already match
- use `acc`: `fp32`
- keep the fused epilogue in `fp32`
- store in the output tensor dtype

Only add an explicit cast when the two operands would otherwise differ in dtype and you have validated that the cast is both numerically acceptable and performance-positive.

When a single tiled rewrite is fast but fails the existing correctness contract for some dtypes or shape regimes, prefer **dtype-specialized or shape-specialized dispatch** over discarding the idea entirely. A common fallback is:

- `fp16` and sufficiently large `M`: tiled matmul path
- `fp32` or small shapes: baseline-style reduction path

Use this when the performance win is real for one operating regime, but a unified replacement changes accumulation order or precision behavior too much for another regime.

## Host Launch Template

```python
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

matmul_kernel[grid](
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M=BLOCK_M,
    BLOCK_N=BLOCK_N,
    BLOCK_K=BLOCK_K,
    num_warps=4,
    num_stages=1,
)
```

Treat those tile values as a starting point, not as a universal rule.

## Dispatch Rule

Do not assume the classic tiled path must replace every input regime.

Prefer a dispatched design when evidence shows:

- tiled matmul is clearly faster for `fp16` or larger shapes
- baseline-style reduction is more accurate or cheaper for `fp32` or smaller shapes
- the correctness gate rejects one unified implementation

Typical host-side decision pattern:

```python
if x.dtype == torch.float16 and M >= LARGE_M_THRESHOLD:
    launch_tiled_matmul(...)
else:
    launch_baseline_reduction(...)
```

The exact threshold should come from benchmark evidence, not from a fixed guess.

### K-dimension constexpr specialization

When common shapes concentrate around a few `K` values, split dispatch by `K` and compile specialized kernels instead of keeping runtime branch trees inside one generic kernel. This is useful when `K == 64` and `K == 128` have meaningfully different block-pointer, load, or dot structure.

```python
if K == 64:
    kernel_k64[grid](...)
elif K == 128:
    kernel_k128[grid](...)
else:
    kernel_generic[grid](...)
```

Use this when the specialized branch removes active `if K > ...` logic, dead pointer paths, or padded work from the hot loop. Avoid it when `K` has many active values, when compile-cache growth dominates, or when the generic kernel already receives `K` as `tl.constexpr` and the backend eliminates the branch cleanly.

Record which shapes justify the specialization. Delete or demote experimental branches that lose to the generic path.

## Expected Benefit

- more regular lowering of the main `K` loop
- better fit for `tl.dot`/matmul code generation
- less scalar-heavy reduction structure
- better amortization of fused epilogues on larger `M` and `N`

## Main Risks

- tile setup overhead can hurt small shapes
- larger tiles can increase register or UB pressure
- forcing `fp16` dot inputs can change `fp32` input semantics

## Related Patterns

- `software-pipeline`: use it after `classic-matmul` when the tiled structure already exists but profiling still shows load-then-compute stalls.
- `tiling`: use it when the kernel already has the right tiled structure, but block size or intermediate footprint is still too large for UB.

## What To Verify After Applying

- confirm logical `A`, `B`, and `C` shapes
- confirm the second operand really arrives as `[BK, BN]`
- confirm masks on `M`, `N`, and `K`
- confirm fused bias or activation broadcasting
- if you introduce dispatch, validate each dispatched regime explicitly
- if you specialize by `K`, validate each `K` branch and the generic fallback
- benchmark against the previous implementation
- record whether the win came from lower scalar pressure, better matmul lowering, or better epilogue amortization
