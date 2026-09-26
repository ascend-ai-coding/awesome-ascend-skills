# Debug checklist, testing strategy, error table, API notes (DEBUG_GUIDEBOOK.md §9.9–§9.12)

*Agent-learned patterns from GDR kernel development.*

## §9.9 Debug Checklist

| Check | Command/Method |
|-------|---------------|
| JIT signature | Check `_original_func.__annotations__` are not strings |
| Dynamic shapes | Verify tile shapes are concrete |
| SIM mode | Accept limitations, test on NPU |
| Imports | Ensure `pypto` is imported correctly |
| Type hints | No `from __future__ import annotations` |

---

## §9.10 Testing Strategy

1. **Syntax check:** `python -m py_compile module.py`
2. **Golden comparison:** Compare against PyTorch reference
3. **SIM mode:** For basic execution (not precision)
4. **NPU mode:** For actual precision validation

---

## §9.11 Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Non-tensor parameter 'x' must not be a torch.Tensor` | String annotations | Remove `from __future__ import annotations` |
| `Not concrete value` | Symbolic in tile shapes | Use concrete constants |
| `Cannot convert symbols to int` | Symbolic in loop/indexing | Use `pypto.view` with offsets |
| `ValueError: Invalid value type` | Symbolic in `pypto.loop` | Pass loop bounds as concrete int parameters |
| `Errcode: F21004 tile shape not set` | Missing `set_vec_tile_shapes` before ops | Call `set_vec_tile_shapes` first |
| `Errcode: F21004 Their size actually are X and Y` | `pypto.view` shape/offsets mismatch | Ensure `len(shape) == len(offsets)` |
| `operand1 dim[0] = -1` | SIM mode dynamic shape issue | Test on NPU |
| `Invalid tile values` | SIM mode tiling issue | Test on NPU |

---

## §9.12 Key PyPTO API Notes

- **`pypto.DYNAMIC`** - Dynamic dimension marker for tensor type hints
- **`pypto.DT_FP32`, `pypto.DT_BF16`, etc.** - Data type enums
- **`pypto.RunMode.NPU`** - Run on actual NPU hardware
- **`pypto.RunMode.SIM`** - Run in simulation mode (limited)
- **`pypto.loop(start, stop, step)`** - Kernel loop construct
- **`pypto.view(tensor, shape, offsets)`** - Tensor view/slice
- **`pypto.matmul(a, b, out_dtype=...)`** - Matrix multiplication
- **`pypto.exp`, `pypto.mul`, `pypto.add`, etc.** - Element-wise ops
- **`pypto.transpose(t, dim0, dim1)`** - Transpose
- **`pypto.reshape(t, shape)`** - Reshape for broadcasting
- **`pypto.zeros(shape, dtype)`** - Create zeros tensor
- **`pypto.set_vec_tile_shapes(...)`** - Set vector tile configuration
- **`pypto.set_cube_tile_shapes(...)`** - Set cube tile configuration
