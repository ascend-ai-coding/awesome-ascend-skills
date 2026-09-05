# Tensor Operations & Transpose (DEBUG_GUIDEBOOK.md §9.7, §9.16)

*Agent-learned patterns from GDR kernel development.*

## §9.7 Tensor Operations

### pypto.matmul requires 2D+ tensors

**Error:**

```
RuntimeError: Tensor dimension mismatch. Expect input_dim == mat2_dim and both in [2, 3, 4], got input_dim: 2, mat2_dim: 1.
```

**Cause:** `pypto.matmul` requires both input tensors to have 2+ dimensions. 1D tensors must be reshaped to 2D.

**Common case - vector-matrix multiplication:**

```python
# c_cum is [bt, bt] (2D), gc_raw is [bt] (1D)
# WRONG
g_cum = pypto.matmul(c_cum, gc_raw, ...)

# CORRECT - reshape to 2D
gc_raw_2d = gc_raw.reshape([bt, 1])
g_cum = pypto.matmul(c_cum, gc_raw_2d, ...).reshape([bt])
```

**Pattern for 1D results from matmul:**

```python
# When result should be [bt] but matmul gives [bt, 1]
result = pypto.matmul(matrix, vector_2d, ...).reshape([bt])
```

For full `pypto.matmul` API (transpose flags, tile shapes), see `matmul.md`.

### Broadcasting

Use `pypto.reshape` to add/remove dimensions for broadcasting:

```python
# WRONG
result = tensor * scalar  # scalar needs explicit reshape

# CORRECT
scalar_reshaped = pypto.reshape(scalar, [bt, 1])
result = pypto.mul(tensor, scalar_reshaped)
```

### Zeros initialization

```python
zeros = pypto.zeros([M, N], pypto.DT_FP32)
```

### Element-wise operations

```python
negated = pypto.mul(tensor, -1.0)  # multiply by negative one
```

For Python-operator equivalents (`*`, `+`, `.exp()`) inside JIT, see `python-operators.md`.

---

## §9.16 Transpose Operations

Both approaches work:

```python
# Method 1: .T property (preferred - cleaner)
transposed = tensor.T

# Method 2: explicit function
transposed = pypto.transpose(tensor, dim0, dim1)
```

**Caveat:** `.T` works inside `@pypto.frontend.jit` on PyTorch-backed tensors but **does NOT work** on PyPTO tensors created by view/matmul intermediates (`AttributeError: 'Tensor' object has no attribute 'T'`). For matmul, prefer the `a_trans` / `b_trans` flags — see `matmul.md` "`.T` attribute doesn't exist on PyPTO tensors".
