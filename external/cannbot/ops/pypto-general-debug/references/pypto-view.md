# Comprehensive pypto.view Guide (DEBUG_GUIDEBOOK.md §9.4)

*Agent-learned patterns from GDR kernel development.*

## Signature

```python
pypto.view(
    input: pypto.Tensor,
    shape: List[int],           # Must be concrete integers
    offsets: List[Union[int, pypto.SymbolicScalar]],
    valid_shape: Optional[List[Union[int, pypto.SymbolicScalar]]] = None
) -> pypto.Tensor
```

## Golden Rule

**`len(shape) == len(offsets)`** - This is mandatory!

## Best Practices

**1. Always match dimensions:**

```python
# Tensor shape: [B, T, H, K] (4D)
# Offsets: [b, t0, h, 0] (4 elements)
# View shape must be 4D: [1, bt, 1, K]
qc = pypto.view(q_norm, [1, bt, 1, K], [b, t0, h, 0]).reshape([bt, K])
```

**2. Use 1s to pad unused dimensions:**

```python
# Tensor [B, H, K, V] → view at [b, h]
# Use [1, 1, K, V] to match 4-element offsets [b, h, 0, 0]
s = pypto.view(state, [1, 1, K, V], [b, h, 0, 0]).reshape([K, V])
```

**3. For 1D/2D tensors with multi-dim offsets:**

```python
# Tensor [B, T, H] with 3D offsets [b, t0, h]
# Use [1, bt, 1] to match 3 elements
betac = pypto.view(beta_in, [1, bt, 1], [b, t0, h]).reshape([bt])
```

**4. For 5D tensors:**

```python
# Tensor [B, H, nt, bt, bt] with 5D offsets [b, h, c, 0, 0]
# Use [1, 1, 1, bt, bt] to match 5 elements
A_c = pypto.view(A_in, [1, 1, 1, bt, bt], [b, h, c, 0, 0]).reshape([bt, bt])
```

**5. Assemble back with reshape:**

```python
# When assembling, reshape to match original tensor's view dimensions
pypto.assemble(s.reshape([1, 1, K, V]), [b, h, 0, 0], output)
```

## Common Mistakes

| Mistake | Error | Fix |
|---------|-------|-----|
| Shape dims ≠ offset dims | `Their size actually are X and Y` | Pad with 1s |
| Using `[bt, K]` with 4 offsets | Dimension mismatch | Use `[1, bt, 1, K]` |
| Forgetting reshape after view | Wrong shape in computation | Add `.reshape([bt, K])` |

## Dimension Padding Pattern

When the desired view has fewer dims than the offsets:

```
Original:  [K, V]  desired
Offsets:   [b, h, 0, 0]  has 4 elements
Solution:  Pad shape: [1, 1, K, V]
Result:    pypto.view(tensor, [1, 1, K, V], [b, h, 0, 0]).reshape([K, V])
```

## Related issues

- 5D tensor view + 4D vec tile shapes can trigger `Run pass failed` — see `matmul.md` (§9.19 last subsection).
- `pypto.assemble` expects matching view dimensions — see `matmul.md` (§9.19 "pypto.assemble shape mismatch").
