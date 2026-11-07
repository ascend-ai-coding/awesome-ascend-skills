# Python Operators Inside PyPTO JIT (DEBUG_GUIDEBOOK.md §9.14)

*Agent-learned patterns from GDR kernel development.*

**Discovery:** Python operators **DO work** inside `@pypto.frontend.jit` decorated functions!

**Evidence from working examples:**

```python
# From pypto_l2norm_bwd
dot_q = (dyq * yq).sum(-1, keepdim=True)
d = dyq * rstd_yq - dot_q * yq * rstd_yq
```

**Recommendation:** Use Python operators for cleaner code:

```python
# VERBOSE (unnecessary)
result = pypto.mul(pypto.mul(a, b), pypto.add(c, d))
result = pypto.sum(x, dim=-1, keepdim=True)
result = pypto.rsqrt(pypto.add(sum_sq, eps))

# PREFERRED (clean and works!)
result = a * b * (c + d)
result = x.sum(-1, keepdim=True)
result = (sum_sq + eps).rsqrt()
```

**Supported Method Chains:**

```python
tensor.T              # transpose
tensor.exp()          # element-wise exp
tensor.reshape([...]) # reshape
tensor.sum(-1)        # sum along last dim
tensor.rsqrt()       # reciprocal square root
tensor.abs()         # absolute value
tensor.sqrt()        # square root
tensor.neg()         # negation
```

## Caveats

- `.sum()` requires the reduction axis to be 32-byte aligned. If alignment is off, the matmul-based reduction workaround applies — see `matmul.md` "Reduction axis needs 32-byte alignment".
- `.T` works on PyTorch-backed tensors but not on intermediate PyPTO tensors. For matmul, use `a_trans` / `b_trans` flags — see `matmul.md`.
