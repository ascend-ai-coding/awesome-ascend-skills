# JIT signature & tensor type hints (DEBUG_GUIDEBOOK.md §9.1, §9.13)

*Agent-learned patterns from GDR kernel development. Add to this file when discovering new patterns.*

## §9.1 JIT Signature Parsing

### Issue: `from __future__ import annotations` breaks JIT

**Symptom:** `RuntimeError: Non-tensor parameter 'q_in' must not be a torch.Tensor. Use positional arguments for tensors.`

**Root Cause:** PEP 563 string annotations cause all type hints to be stored as strings instead of objects.

**Diagnosis:**

```python
# Check annotations - they should be pypto.Tensor objects, not strings
func = kernel._original_func
print(func.__annotations__)  # If strings, it's the import issue
```

**Solution:** Remove `from __future__ import annotations` from files using `@pypto.frontend.jit`.

```python
# WRONG - causes JIT to fail
from __future__ import annotations
@pypto.frontend.jit()
def kernel(x: pypto.Tensor(...)):
    pass

# CORRECT
@pypto.frontend.jit()
def kernel(x: pypto.Tensor(...)):
    pass
```

---

## §9.13 Tensor Shape Specifications for Dynamic Axes

### Issue: Shape Size Exceeds INT32_MAX

**Error:**

```
RuntimeError: Errcode: FFFFFF!
The shape size of tensor must less than or equal to INT32_MAX(2,147,483,647)
```

**Root Cause:** Feeding a `pypto.DYNAMIC` tensor *directly* into a compute
API (`pypto.sum`, elementwise ops, etc.) without first slicing it into a
concrete-shape tile. The compiler's workspace estimator multiplies the
DYNAMIC bound by the other dims, overflowing INT32.

```python
# WRONG — DYNAMIC tensor consumed directly by compute APIs (no view)
def kernel(
    x: pypto.Tensor([pypto.DYNAMIC, 16, 64], pypto.DT_FP32),
    ...
):
    pypto.set_vec_tile_shapes(1, 16, 64)
    sum_x = x.sum(-1, keepdim=True)   # workspace estimator: DYNAMIC dim unbounded → INT32 overflow / OOM
```

**Solution:** Keep `pypto.DYNAMIC` in the annotation, but slice it into a
concrete-shape tile **inside the JIT body** using `pypto.loop` +
`pypto.view`. The workspace estimator then sees only the concrete tile
shape, never the unbounded DYNAMIC dim. **Keep the view small** — a few
KB / FP32 is the right scale; do not view megabyte-sized blocks per
iteration (the framework would then have to spill / split the workspace
behind your back).

```python
# CORRECT — DYNAMIC in annotation, concrete slicing inside JIT body
@pypto.frontend.jit(...)
def kernel(
    x: pypto.Tensor([pypto.DYNAMIC, 16, 64], pypto.DT_FP32),
    y: pypto.Tensor([pypto.DYNAMIC, 16, 64], pypto.DT_FP32),
    ...
):
    pypto.set_vec_tile_shapes(1, 16, 64)        # 3D tile, matches view rank; batch=1 loop-collapsed
    B = x.shape[0]
    for b in pypto.loop(B, name="batch", unroll_list=[1]):  # single value before Stage 6
        # pypto.view shape is concrete Python ints; offsets may be SymbolicScalar.
        x_tile = pypto.view(x, [1, 16, 64], [b, 0, 0])    # 1024 elem = 4 KB FP32 per iter
        # ... compute on x_tile (concrete shape) ...
        pypto.assemble(result, [b, 0, 0], y)
```

If the static dims are larger than the safe view size, **nest a Python `for`
loop (or another `pypto.loop`) over the larger axis** and view a small
slice at each step instead of viewing the whole inner volume in one go.

**DO NOT use empty `[]` annotation as a workaround.** That escape route
defers shape inference to the caller, which only works for tiny fixed
shapes that fit entirely in UB (~32 KB). For any production input the
kernel will OOM or silently truncate at runtime. It also breaks DESIGN
`dynamic_axes` consistency and the DYNAMIC shape declaration.

**Required production pattern (4 elements together — none optional):**

1. Annotation: `pypto.Tensor([pypto.DYNAMIC, ...static dims...], dtype)`
2. Tile config: `pypto.set_vec_tile_shapes(...)` or `pypto.set_cube_tile_shapes(...)`
3. Loop: `for b in pypto.loop(B, name=..., unroll_list=[1])` (single value before
   Stage 6 — default `[1]`; multi-value tuning is Stage 7 only)
4. View: `pypto.view(x, [concrete shape], [b, ...])` inside the loop

Cross-reference: `pypto-op-design/SKILL.md` §2.4 "动态轴模式" and
`pypto-op-design/references/quick_ref.md` for the canonical production
pattern. These requirements apply to every `_impl.py` and `_module*_impl.py`
file — check them whenever you write or edit one.
