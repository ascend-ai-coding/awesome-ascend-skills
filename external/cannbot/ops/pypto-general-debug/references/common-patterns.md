# Common Patterns & Quick Reference (DEBUG_GUIDEBOOK.md §9.8, §9.17, §9.18)

*Agent-learned patterns from GDR kernel development.*

## §9.8 Common Patterns

### Pattern: Multi-session with state carry

```python
for session in pypto.loop(range(B * H), name="sessions"):
    b = session // H
    h = session % H

    state = pypto.view(initial_state, [K, V], [b, h, 0, 0])

    for c in pypto.loop(range(nt), name="chunks"):
        # process chunk
        ...
        state = updated_state

    output[b, h, :, :] = state
```

### Pattern: Constant tile shapes at module level

```python
# Module-level constants for tile shapes
TILE_SESSIONS = 16
TILE_CHUNKS = 4
TILE_BT = 16
TILE_KV = 64

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(...):
    pypto.set_vec_tile_shapes(TILE_SESSIONS, TILE_CHUNKS, TILE_BT, TILE_KV)
    ...
```

### Pattern: Precompute on host

For operations not supported by PyPTO (e.g., `torch.linalg.solve_triangular`):

```python
def make_host_constants(bt, k, g_raw, beta, device):
    # Precompute on CPU/GPU
    A = torch.linalg.solve_triangular(...)
    return A

# In kernel, receive as constant
@pypto.frontend.jit()
def kernel(A_in: pypto.Tensor(...)):
    A_c = pypto.view(A_in, [bt, bt], [b, h, c, 0, 0])
    ...
```

### Pattern: Reverse iteration for backward

```python
for i in pypto.loop(range(nt), name="chunks_reverse"):
    c = nt - 1 - i  # reverse chunk index
    ...
```

---

## §9.17 Pattern Quick Reference

| Operation | Verbose Form | Preferred Form |
|-----------|-------------|----------------|
| Tensor shape | `pypto.Tensor([pypto.DYNAMIC, ...], dtype)` | `pypto.Tensor([], dtype)` |
| Multiply | `pypto.mul(x, y)` | `x * y` |
| Square | `pypto.mul(x, x)` | `x * x` |
| Sum | `pypto.sum(x, dim=-1, keepdim=True)` | `x.sum(-1, keepdim=True)` |
| Rsqrt | `pypto.rsqrt(x)` | `x.rsqrt()` |
| Exp | `pypto.exp(x)` | `x.exp()` |
| Add scalar | `pypto.add(x, scalar)` | `x + scalar` |
| Sub scalar | `pypto.sub(x, scalar)` | `x - scalar` |
| Transpose | `pypto.transpose(t, 0, 1)` | `t.T` |
| Cast | `pypto.cast(x, pypto.DT_FP32)` | `x.float()` |
| Reshape | `pypto.reshape(t, [a, b])` | `t.reshape([a, b])` |
| Matmul | `pypto.matmul(a, b, ...)` | `pypto.matmul(a, b, ...)` (keep explicit) |

---

## §9.18 Key Takeaways

1. **Shape Inference**: Use `pypto.Tensor([], dtype)` for automatic shape inference
2. **Python Operators**: Use Python operators inside JIT (`*`, `+`, `-`, `/`)
3. **Method Chaining**: PyPTO tensors support method chaining (`.exp()`, `.T`, `.rsqrt()`)
4. **Tile Shapes**: Match tile shapes to actual tensor dimensions or use even divisors
5. **Keep matmul explicit**: `pypto.matmul()` is preferred over Python `@` operator
