# Attention Cube-Vector Pipeline Pattern

## Summary

Reduce latency in Cube+Vector fused attention-like kernels by cutting vector-side instruction pressure, making mask/scale work cheaper, and using architecture-gated compile options only when the target device supports them.

## Use When

- A `tl.dot` loop is followed by substantial vector epilogue work such as scale, mask, softmax, dropout, or bias.
- Profiling suggests Cube and Vector work are close enough that vector-side overhead limits overlap.
- A loop repeatedly recomputes the same mask tensor from sequence lengths or causal indices.
- Scale and mask are separate operations before softmax.
- The code stores log-sum-exp state in a base-2 representation solely because the forward path uses `exp2`.
- The target is known to be an A5 device such as `ascend950PR` or `ascend950DT`.

## Avoid When

- The kernel is pure Vector work rather than Cube-plus-Vector fused work.
- Profiling shows memory transfer, not vector epilogue work, is the dominant bottleneck.
- Architecture-specific compile settings cannot be gated on verified target information.

## Repairs

### Cube/Vector pipeline scheduling

Move independent vector work away from the critical Cube path so loads, `tl.dot`, and epilogue work can overlap better. Prefer changes that reduce live vector temporaries and instruction count before adding buffering.

Do not use this pattern when the kernel is pure Vector or when profiling shows memory transfer, not vector epilogue work, is dominant.

### Precompute repeated masks

If mask construction is repeated inside a hot loop and depends only on host-known metadata, precompute the mask on the host and pass it as a tensor. In varlen cases, build each batch mask so invalid positions are already false, then use block pointer shapes that reflect the real `(q_len, k_len)` region.

This trades memory bandwidth for less vector control work. Validate the tradeoff with benchmark evidence.

### Fuse scale and mask

When softmax scores are scaled then masked, consider combining the operations into one expression that feeds softmax directly:

```python
scores = scores * scale + tl.where(mask, 0.0, -float("inf"))
```

Use a finite large negative value only when dtype and downstream numerics make that equivalent and safe.

### Use `exp` instead of `exp2` consistently

If `exp2(x * log2e)` is used only to approximate `exp(x)`, consider switching to `tl.exp(x)` and store matching log-sum-exp state. Update backward formulas together; forward-only changes can silently break gradients.

### Architecture-gated compile parameters

Some compile parameters are only appropriate for A5 targets. Gate them on the actual architecture and record the target evidence. Do not enable A5-only options on older Ascend devices.

## Risks

- Mask precomputation can change boundary behavior if block pointer shapes describe the padded max shape instead of real lengths.
- Scale-mask fusion changes where infinities and dtype conversions happen.
- `exp` versus `exp2` must be consistent across saved state and backward code.
- A5 compile flags are target-specific and must not become unconditional defaults.

## What To Verify After Applying

- Run correctness with mask-heavy, boundary, and varlen cases when available.
- Compare both latency and profiler balance; a vector-instruction reduction should show up in the evidence.
- Record any architecture gate in `attempts.md` and `summary.md`.
- If backward code exists, verify forward/backward state conventions together.
