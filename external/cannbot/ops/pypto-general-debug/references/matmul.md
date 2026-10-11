# matmul API and Tile Shapes (DEBUG_GUIDEBOOK.md §9.19)

*Agent-learned patterns from GDR kernel development.*

## Issue: Wrong matmul syntax causes errors

**Symptom:** matmul operations fail with cryptic errors.

**Root Cause:** Wrong API usage for `pypto.matmul`.

**Correct matmul syntax:**

```python
# WRONG - this syntax does not work
result = pypto.matmul(a, b.T, out_dtype=pypto.DT_FP32)
result = pypto.matmul(a, b, out_dtype=pypto.DT_FP32)

# CORRECT - use a_trans and b_trans parameters
result = pypto.matmul(a, b, pypto.DT_FP32, a_trans=False, b_trans=True)
result = pypto.matmul(a, b, pypto.DT_FP32)  # both False by default
```

**Transpose patterns:**

```python
# a @ b.T  →  a_trans=False, b_trans=True
result = pypto.matmul(a, b, dtype, a_trans=False, b_trans=True)

# a.T @ b  →  a_trans=True, b_trans=False
result = pypto.matmul(a, b, dtype, a_trans=True, b_trans=False)

# a @ b    →  both False (default)
result = pypto.matmul(a, b, dtype)

# a.T @ b.T  →  both True
result = pypto.matmul(a, b, dtype, a_trans=True, b_trans=True)
```

**Note:** Do NOT use `.T` on tensors before passing to matmul - use the transpose flags instead.

---

## Issue: Large K and the 65535 ND inner-axis limit

**Rule:** The 65535 bound applies to each ND operand's physical inner axis:
`operand.shape[-1] <= 65535`. It is not a limit on logical K.

For `A @ B`:

- `A` as `[M, K]`, `a_trans=False` → inner axis is K.
- `A` as `[K, M]`, `a_trans=True` → inner axis is M.
- `B` as `[K, N]`, `b_trans=False` → inner axis is N.
- `B` as `[N, K]`, `b_trans=True` → inner axis is K.

For large logical K, first choose the layout / `pypto.view` shape / `a_trans`
/ `b_trans` so K is a physical outer axis when possible, then call the full
`pypto.matmul`.
`enable_split_k=True` is a performance option for already-legal 2D operands;
it cannot make an illegal inner axis legal.

Do not hand-split K only because `K > 65535`. A manual K-loop is only a
fallback when layout and matmul parameters cannot express the operation.

---

## Issue: Both vec and cube tile shapes needed

**Symptom:** matmul or other ops fail with tiling errors.

**Root Cause:** Need to set BOTH `set_vec_tile_shapes` AND `set_cube_tile_shapes`.

**Solution:**

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(...):
    # BOTH are required for matmul to work
    pypto.set_vec_tile_shapes(TILE_B, TILE_H, TILE_T, TILE_K)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    # Now matmul operations will work
    result = pypto.matmul(a, b, pypto.DT_FP32)
    ...
```

**Common tile configurations:**

```python
# For forward kernels
TILE_B = 1
TILE_H = 2
TILE_T = 8
TILE_K = 32
pypto.set_vec_tile_shapes(TILE_B, TILE_H, TILE_T, TILE_K)
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# For backward kernels
TILE_BH = 16
TILE_NT = 4
TILE_BT = 8
TILE_KV = 32
pypto.set_vec_tile_shapes(TILE_BH, TILE_NT, TILE_BT, TILE_KV)
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
```

**Rule:** Always set BOTH tile shape configurations when using matmul or complex tensor operations.

---

## Issue: pypto.assemble shape mismatch

**Symptom:**

```
CHECK FAILED: dest.GetShape().size() == tensor.GetShape().size()
Assemble: src and dest requires same shape
```

**Root Cause:** The src tensor shape doesn't match the expected view dimensions for the destination.

**Solution:** Reshape the output tensor to match the expected view shape:

```python
# WRONG - out_chunk is [bt, V] but assemble expects [1, bt, 1, V]
pypto.assemble(out_chunk, [b, t0, h, 0], out_out)

# CORRECT - reshape to match view dimensions
pypto.assemble(out_chunk.reshape([1, bt, 1, V]), [b, t0, h, 0], out_out)
```

**Rule:** `pypto.assemble(tensor, offsets, dest)` requires tensor shape to have same number of dimensions as the view at those offsets.

---

## Issue: reshape([1]) on multi-element tensor

**Symptom:**

```
CHECK FAILED: capacity == 1
Shape size not match, func CheckAndInferShape
```

**Root Cause:** Trying to reshape a tensor with multiple elements to a single element shape.

**Solution:** Use `pypto.view` with offsets to extract single elements, then reshape:

```python
# WRONG - bt=4 tensor has capacity 4, can't reshape to [1]
gl = g_cum.reshape([1])[bt - 1:bt].reshape([1])

# CORRECT - use view to get last element, then reshape
gl = pypto.view(g_cum, [1], [bt - 1]).reshape([1])
```

**Common pattern for getting last element:**

```python
# For getting last element of a 1D tensor of size bt
last_elem = pypto.view(tensor, [1], [bt - 1]).reshape([1])
```

---

## Issue: `.T` attribute doesn't exist on PyPTO tensors

**Symptom:**

```
AttributeError: 'Tensor' object has no attribute 'T'
```

**Root Cause:** PyPTO tensors don't support the `.T` property like PyTorch tensors.

**Solution:** Use `pypto.transpose(tensor, 0, 1)` or use matmul's transpose flags:

```python
# WRONG - .T doesn't work on PyPTO tensors
kc_t = kc.T
result = pypto.matmul(qc, kc_t, dtype)

# CORRECT - use matmul transpose flags
result = pypto.matmul(qc, kc, dtype, a_trans=False, b_trans=True)

# OR use explicit transpose
kc_t = pypto.transpose(kc, 0, 1)
result = pypto.matmul(qc, kc_t, dtype)
```

**For 2D matrix transpose:** `pypto.transpose(tensor, 0, 1)` does NOT work on 2D tensors with the tiling system - it causes "TileShape dim num should same to input" error.

**Instead, use matmul transpose flags:**

```python
# Instead of transpose, use a_trans/b_trans in matmul:
kc_t = pypto.transpose(kc, 0, 1)  # WRONG - doesn't work!
qk = pypto.matmul(qc, kc, dtype, a_trans=False, b_trans=True)  # CORRECT

# For symmetric sum m_mat + m_mat.T, split into two matmuls:
dk_c = dk_c + pypto.matmul(m_mat, kc, dtype)  # m_mat @ kc
dk_c = dk_c + pypto.matmul(m_mat, kc, dtype, a_trans=True, b_trans=False)  # m_mat.T @ kc
```

---

## Issue: Reduction axis needs 32-byte alignment

**Symptom:**

```
Reduce op: the tileShape of last axis need to 32Byte align!
```

**Root Cause:** PyPTO reduction operations (`.sum()`) require tensor dimensions to be 32-byte aligned.

**Calculation:** For FP32 (4 bytes), dimension * 4 must be divisible by 32.
- `bt=4` → 4*4=16 bytes ❌ Not aligned
- `bt=8` → 8*4=32 bytes ✅ Aligned
- `V=16` → 16*4=64 bytes ✅ Aligned

**Solution:** Use dimensions that are multiples of 8 for reduction axes:

```python
# WRONG - bt=4 causes alignment error
bt = 4
result = tensor.sum(-1)

# CORRECT - bt=8 is 32-byte aligned
bt = 8
result = tensor.sum(-1)
```

**Rule:** Any tensor dimension involved in `.sum()`, `.mean()`, or other reduction operations must satisfy `(dim * bytes_per_element) % 32 == 0`. For FP32, use dimensions that are multiples of 8.

**Common alignment examples (FP32):**
- V=16 → 16*4=64 bytes ✅ Aligned
- V=32 → 32*4=128 bytes ✅ Aligned
- K=16 → 16*4=64 bytes ✅ Aligned
- K=32 → 32*4=128 bytes ✅ Aligned
- bt=8 → 8*4=32 bytes ✅ Aligned

---

## Issue: Sum reduction fails even with aligned dimensions

**Symptom:**

```
Reduce op: the tileShape of last axis need to 32Byte align!
```

**Problem:** Even when dimensions are theoretically aligned (e.g., V=32), `.sum(-1)` on the last axis may still fail due to how PyPTO tiles the tensor internally.

**Solution:** Replace `.sum()` with matmul-based reduction using a precomputed ones vector:

```python
# Create ones vectors on host (for any dimension, not just aligned)
ones_v = torch.ones(V, 1, device=device, dtype=torch.float32)  # [V, 1]
ones_k = torch.ones(K, 1, device=device, dtype=torch.float32)  # [K, 1]

# In kernel signature, add ones vectors as parameters:
@pypto.frontend.jit(...)
def kernel(..., ones_v: pypto.Tensor([], pypto.DT_FP32), ones_k: pypto.Tensor([], pypto.DT_FP32), ...):

    # Replace .sum(-1) with matmul:
    db_c = (dvb * vc).sum(-1)  # WRONG - may fail

    db_c = pypto.matmul(dvb * vc, ones_v, pypto.DT_FP32).reshape([bt])  # CORRECT

    # Replace .sum(0) and .sum(1) with matmul:
    d_l_l_mat_0 = (d_l * l_mat).sum(0)  # WRONG
    d_l_l_mat_0 = pypto.matmul(d_l * l_mat, ones_k, pypto.DT_FP32, a_trans=True, b_trans=False).reshape([bt])  # CORRECT

    d_l_l_mat_1 = (d_l * l_mat).sum(1)  # WRONG
    d_l_l_mat_1 = pypto.matmul(d_l * l_mat, ones_k, pypto.DT_FP32).reshape([bt])  # CORRECT
```

**Key insight:** Matmul-based reduction works regardless of alignment because it uses cube operations, while `.sum()` uses vector operations that require strict 32-byte alignment.

**Rule:** When in doubt, use matmul with ones vector for reduction instead of `.sum()` on any axis.

---

## Issue: K-dimension valid shape mismatch in matmul

**Symptom:**

```
RuntimeError: K-dimension valid shape mismatch. Got input valid shape: [SymbolicScalar(8), SymbolicScalar(8)], mat2 valid shape: [SymbolicScalar(32), SymbolicScalar(1)], a_trans: False, b_trans: False.
```

**Problem:** Using wrong ones vector for matmul reduction. The ones vector must match the dimension being reduced.

**Solution:** Use different ones vectors for different dimensions:

```python
# Create different ones vectors for different dimensions
ones_v = torch.ones(V, 1, device=device, dtype=torch.float32)   # [V, 1] - for reducing V dimension
ones_k = torch.ones(K, 1, device=device, dtype=torch.float32)  # [K, 1] - for reducing K dimension
ones_bt = torch.ones(bt, 1, device=device, dtype=torch.float32) # [bt, 1] - for reducing bt dimension

# In kernel signature, add all ones vectors:
def kernel(..., ones_v, ones_k, ones_bt, ...):

    # For [bt, V] tensor reducing V dim (sum over last axis):
    db_c = pypto.matmul(tensor, ones_v, dtype).reshape([bt])  # CORRECT

    # For [bt, bt] tensor reducing bt dim (sum over rows/cols):
    d_l_l_mat_0 = pypto.matmul(tensor, ones_bt, dtype, a_trans=True, b_trans=False).reshape([bt])  # row sums
    d_l_l_mat_1 = pypto.matmul(tensor, ones_bt, dtype).reshape([bt])  # col sums
```

**Rule:** The ones vector's first dimension must equal the dimension being reduced in the matmul operation.

---

## Issue: 5D tensor views with 4D vec tile shapes causes "Run pass failed"

**Symptom:**

```
Errcode: FFFFFF!
Run pass failed., func CompileFunction
```

**Root Cause:** Using 5D tensor views (e.g., `pypto.view(A_in, [1, 1, 1, bt, bt], [b, h, c, 0, 0])`) when `pypto.set_vec_tile_shapes` only sets 4D tile shapes. The framework cannot handle 5D operations with 4D tile configuration.

**Solution:** Reshape 5D tensors to 2D before passing to the kernel, then use matching view shape/offsets:

```python
# In host_wrapper, reshape before kernel call:
# Original: [B, H, nt, bt, bt] -> Reshape to 2D: [B*H*nt, bt*bt]
A_2d = A_5d.reshape([B * H * nt, bt * bt])
w_2d = w_4d.reshape([B * H * nt, bt * K])
S_before_2d = S_before_4d.reshape([B * H * nt, K * V])
v_new_2d = v_new_4d.reshape([B * H * nt, bt * V])
g_cum_2d = g_cum_3d.reshape([B * H * nt, bt])

# In kernel, use 2D views with 2 offsets (len(shape) == len(offsets)):
session_base = session * nt
for c in pypto.loop(0, nt, 1):
    cache_idx = session_base + c
    # For 2D tensor [N, M] with view [a, b]: offsets = [cache_idx, 0]
    a = pypto.view(A_2d, [bt, bt], [cache_idx, 0]).reshape([bt, bt])
    w = pypto.view(w_2d, [bt, K], [cache_idx, 0]).reshape([bt, K])
    s_before = pypto.view(S_before_2d, [K, V], [cache_idx, 0]).reshape([K, V])
    v_new = pypto.view(v_new_2d, [bt, V], [cache_idx, 0]).reshape([bt, V])
    # For 1D result from 2D tensor: shape = [1, size], offsets = [cache_idx, 0]
    g_cum = pypto.view(g_cum_2d, [1, bt], [cache_idx, 0]).reshape([bt])
```

**Rule:** `len(shape) == len(offsets)` is mandatory for pypto.view. Keep all tensors ≤4D and reshape to 2D before passing to kernel. For 1D views from 2D tensor, use shape [1, size] with 2 offsets.
