# Tile Shape Configuration & Debug (DEBUG_GUIDEBOOK.md §9.15)

This reference covers both **basic tile-shape configuration** and **diagnosis / patch
proposals** when `pypto.matmul` or any cube/vec op fails at tile-config or compile time.

The pypto-op-debugger owns patch proposals; the pypto-op-coder applies them. This file
produces patch *proposals*, not commits.

---

## 0. Basic configuration patterns (agent-learned from kernel development)

**Issue:** Fixed tile shapes may not match actual tensor dimensions.

**For simple kernels without loops:**

```python
B, T, H, K = x.shape
pypto.set_vec_tile_shapes(B, H, T, K)
```

**For complex kernels with loops:**

```python
TILE_0 = 16   # first tile dimension
TILE_1 = 4    # second tile dimension
TILE_2 = 8    # third tile dimension
TILE_3 = 32   # fourth tile dimension
pypto.set_vec_tile_shapes(TILE_0, TILE_1, TILE_2, TILE_3)
```

**Rule:** Tile shape values should divide evenly into tensor dimensions for best performance.

### See also

- For matmul, both `set_vec_tile_shapes` AND `set_cube_tile_shapes` are required — see `matmul.md` "Both vec and cube tile shapes needed".
- For `F21004` / `REGISTER_COPY` invalid vec tile errors, see `error-codes.md` §4.

---

## 1. API contract recap (must read first)

```python
pypto.set_cube_tile_shapes(
    m: List[int],          # [mL0, mL1]  — exactly 2 elements
    k: List[int],          # [kL0, kL1]
    n: List[int],          # [nL0, nL1]
    enable_split_k: bool = False,
)
```

Hard invariants (per axis X ∈ {m, k, n}):

- **Shape of the argument itself**: `len(X) == 2`. A 1-element list (`[16]`) or a 3+ list is a bug.
- **Ordering**: `0 < XL0 <= XL1`.
- **Divisibility**: `XL1 % XL0 == 0`.
- **Alignment** (non-FP32):
  - `kL0, kL1, nL0, nL1` must be **32-byte aligned**, i.e.
    `XLi * sizeof(dtype) % 32 == 0`.
  - FP16 / BF16: multiples of **16 elements**.
  - INT8: multiples of **32 elements**.
- **Alignment (FP32)**: replace "32-byte" with "**16-element**" for kL0/kL1/nL0/nL1.
- **A matrix ND-transposed**: mL0 must be 32-byte aligned too.
- **NZ format**: outer axis 16-element aligned, inner axis 32-byte aligned.
- **L0 buffer budget** (dtype FP16/BF16/FP32, cDtype = FP32):
  - `CeilAlign(mL0,16) * CeilAlign(kL0,16) * sizeof(aDtype) <= L0A_size`
  - `CeilAlign(nL0,16) * CeilAlign(kL0,16) * sizeof(bDtype) <= L0B_size`
  - `CeilAlign(mL0,16) * CeilAlign(nL0,16) * sizeof(FP32)  <= L0C_size`
- **L0 buffer budget** (dtype INT8, cDtype = INT32): same shape, align=32, sizeof per dtype.
- **L1 buffer budget**:
  - `CeilAlign(mL1,16) * CeilAlign(kL1,16) * sizeof(aDtype)`
    `+ CeilAlign(nL1,16) * CeilAlign(kL1,16) * sizeof(bDtype)`
    `<= L1_size`
  - Replace 16 with 32 for INT8.
- **Bias (BTBuffer = 1 KB, upcast to fp32)**: `nL0 * 4 <= 1024`  →  `nL0 <= 256`.
- **FixPipe (FixBuffer = 2 KB, scale uint64)**: `nL0 * 8 <= 2048`  →  `nL0 <= 256`.
- **`enable_split_k=True`**: only valid for **2D** inputs. 3D/4D must use `enable_split_k=False`.

`CeilAlign(v, a) = ((v + a - 1) // a) * a`.

Typical device budgets on Atlas A2/A3 (confirm per product in the docs site for your release):

| Buffer | Typical size |
|--------|--------------|
| L0A    | 64 KB        |
| L0B    | 64 KB        |
| L0C    | 128 KB       |
| L1     | 512 KB       |

Always verify the current device's exact budgets if the error message quotes a limit.

---

## 2. Symptom → Root cause routing table

Before proposing anything, route the failure by matching the error message or
lint OL48 output.

| Symptom (in error / CI output) | Category | Jump to |
|---|---|---|
| 2-element-list arity error (from lint OL48) | Arity  | §3 |
| `L1 % L0 == 0` / `requires L0 <= L1` (from lint OL48) | Divisibility / ordering | §4 |
| `tile shape not set` | Missing call | §5 |
| `L0A size exceeded` / `L0B size exceeded` / `L0C size exceeded` / "L0 ... overflow" | L0 budget | §6 |
| `L1 size exceeded` / "L1 buffer out of range" | L1 budget | §7 |
| `alignment` / `% 32 != 0` / `% 16 != 0` / "not aligned" | Alignment | §8 |
| `enable_split_k` with 3D/4D inputs | split_k misuse | §9 |
| `BTBuffer` / `FixBuffer` overflow | Bias / FixPipe | §10 |
| Runtime wrong result but no crash; matmul then vec op | Missing `set_vec_tile_shapes` | §11 |

If the error doesn't match any row, fall back to the systematic procedure in §12.

---

## 3. Arity fix — lists must be `[L0, L1]`

**Symptom**: lint **OL48 (S0)** reports that `m` (or `k` / `n`) must be a
2-element list `[L0, L1]` but got a 1-element (or 3+-element) list.

**Cause**: A code block like:

```python
pypto.set_cube_tile_shapes([16], [32], [64])      # BUG — L1 missing
```

**Patch proposal**: pick `L1 = L0 * k` for a small integer `k ≥ 1` that still fits L0/L1 budgets.
Safe default when you have no other constraint:

```python
pypto.set_cube_tile_shapes([16, 32], [32, 64], [64, 128])
```

If shape is known: target `mL1 ≈ M`, `nL1 ≈ N`, `kL1 ≈ K`, then pick `L0` as a clean divisor
(e.g. 64 or 128), confirming §6 and §7 still hold.

---

## 4. Divisibility / ordering fix

**Symptom**: validator flags `XL0 > XL1` or `XL1 % XL0 != 0`.

**Patch proposal**: round `L1` up to the nearest multiple of `L0`; if that blows the L1 budget,
shrink `L0` to a power-of-two divisor of `L1`.

```
# Given [L0, L1] = [96, 128]  (128 % 96 = 32, fails)
# Fix option A:  [64, 128]     (128 % 64 == 0)  ✅
# Fix option B:  [96, 192]     (192 % 96 == 0)  ✅ if L1 budget permits
```

---

## 5. "tile shape not set"

**Symptom**: `tile shape not set` at the first matmul.

**Cause**: `set_cube_tile_shapes` is never called, or is called after matmul, or is set in a
different function scope than the matmul call.

**Patch proposal**: add the call at the **top of the `@pypto.frontend.jit` function body**,
before any matmul, and re-set after any nested-scope change if the API requires it (see
`DEBUG_GUIDEBOOK.md §8`).

If mixing cube and vec ops, also call `set_vec_tile_shapes(...)`:

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(...):
    pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 256])
    pypto.set_vec_tile_shapes(TILE_B, TILE_H, TILE_T, TILE_K)
    ...
```

---

## 6. L0 buffer exceeded

**Symptom (examples)**:

- `L0A size exceeded`, `L0A buffer overflow`, or error quoting "L0A = 65536"
- Similarly for `L0B`, `L0C`

**Step 1 — identify which buffer**:

| Buffer | Formula that must hold                                                  |
|--------|-------------------------------------------------------------------------|
| L0A    | `CeilAlign(mL0, A_align) * CeilAlign(kL0, A_align) * sizeof(aDtype) <= L0A_size` |
| L0B    | `CeilAlign(nL0, A_align) * CeilAlign(kL0, A_align) * sizeof(bDtype) <= L0B_size` |
| L0C    | `CeilAlign(mL0, A_align) * CeilAlign(nL0, A_align) * sizeof(cDtype) <= L0C_size` |

`A_align = 16` for FP16/BF16/FP32, `32` for INT8. `cDtype = FP32` (or INT32 for INT8 path).

**Step 2 — shrink the offending `L0`**:

- `L0A` exceeded → halve `mL0` or `kL0`, whichever is larger.
- `L0B` exceeded → halve `nL0` or `kL0`.
- `L0C` exceeded → halve `mL0` or `nL0`.

Always preserve §4 after the change: after shrinking `XL0`, also shrink `XL1` if needed so
that `XL1 % XL0 == 0` still holds.

**Step 3 — sanity-check L1** (§7) because reducing `L0` may allow a larger `L1` safely.

### Worked example (FP16, L0A = 64 KB)

```
Fail:   mL0=256, kL0=256, sizeof(FP16)=2 → 256*256*2 = 131072 > 65536 ❌
Fix:    mL0=128, kL0=256                  → 128*256*2 = 65536  ≤ 65536 ✅
Or:     mL0=256, kL0=128                  → 256*128*2 = 65536  ≤ 65536 ✅
```

---

## 7. L1 buffer exceeded

**Symptom**: `L1 size exceeded`, or error quoting "L1 = 524288".

**Formula**:

```
CeilAlign(mL1, align) * CeilAlign(kL1, align) * sizeof(aDtype)
+ CeilAlign(nL1, align) * CeilAlign(kL1, align) * sizeof(bDtype)
<= L1_size
```

`align = 16` for FP16/BF16/FP32, `32` for INT8.

**Patch strategy** — in priority order:

1. **Halve `kL1`** first: it multiplies both A-term and B-term, so it gives ~2× the relief.
2. If not enough, halve whichever of `mL1` / `nL1` is largest.
3. Keep `L0 <= L1` and `L1 % L0 == 0` (§4).
4. If workload shape can't support reducing `L1`, consider enabling `enable_split_k=True`
   (only for 2D inputs — see §9) to distribute K across cores.

### Worked example (FP32, L1 = 512 KB)

```
Fail:   mL1=512, kL1=256, nL1=512, sizeof(FP32)=4
         A = 512*256*4 = 524288
         B = 512*256*4 = 524288
         total = 1048576 > 524288 ❌
Fix A:  halve kL1 → kL1=128
         A = 512*128*4 = 262144
         B = 512*128*4 = 262144
         total = 524288 ≤ 524288 ✅  (exactly at the limit)
Fix B:  halve nL1 → nL1=256, kL1=256
         A = 512*256*4 = 524288
         B = 256*256*4 = 262144
         total = 786432 > 524288 ❌  (not enough — halve kL1 as in A)
```

---

## 8. Alignment errors

**Symptom (examples)**: `not aligned`, `% 32 != 0`, `% 16 != 0`, "tile align".

**Rule recap**:

- FP16 / BF16 / FP32: `kL0, kL1, nL0, nL1` → **16-element aligned**
- INT8: 32-element aligned
- A transposed (ND, shape `[K, M]`): `mL0` 32-byte aligned
- NZ format: outer 16-element, inner 32-byte

**Patch**: round the offending value UP to the nearest multiple of the required alignment
unit. If that breaks §6 / §7 / §4, shrink a different axis to compensate.

```
Fail (FP32): kL0=12  → 12 not multiple of 16 ❌
Fix:         kL0=16                               ✅
```

---

## 9. `enable_split_k` misuse

**Symptom**: error mentions `enable_split_k` or "split k not supported for 3D/4D".

**Rule**: `enable_split_k=True` is only valid when inputs are **2D**. For 3D/4D inputs
(attention, 3D matmul), **must be False** (the default).

**Patch**: either reshape inputs to 2D in the host wrapper and issue matmul as 2D, or set
`enable_split_k=False`.

```python
# 3D/4D input
pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 256])  # no split_k

# 2D input, need more parallelism
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], enable_split_k=True)
```

---

## 10. Bias / FixPipe buffer overflow

- **BTBuffer (1 KB, bias upcast to FP32)**: `nL0 * 4 <= 1024` → `nL0 <= 256`.
- **FixBuffer (2 KB, scale uint64)**: `nL0 * 8 <= 2048` → `nL0 <= 256`.

**Patch**: cap `nL0` at 256 whenever bias or FixPipe is active; if more parallelism is
needed, raise `nL1` instead (subject to L1 budget, §7).

---

## 11. Wrong result (not a crash): matmul + vec without `set_vec_tile_shapes`

**Symptom**: numerical wrong answer from a kernel that mixes `matmul` with vector ops
(`mul`, `add`, element-wise). No explicit tile-config error, but precision check fails.

**Cause**: `set_cube_tile_shapes` alone is **not** enough — AIV ops (including compiler-
inserted `REGISTER_COPY`) still require `set_vec_tile_shapes`. See
`DEBUG_GUIDEBOOK.md §8` bullet "Do not rely only on `set_cube_tile_shapes`".

**Patch proposal**: call **both**:

```python
pypto.set_vec_tile_shapes(1, 1, 128, 128)
pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 256])
```

---

## 12. Systematic fallback procedure

If the symptom didn't match §3–§11:

1. **Find the call**: `grep -n "set_cube_tile_shapes\|set_vec_tile_shapes" <op_dir>`.
2. **Dump the argument values** (even symbolic): is each a 2-element list? do literal ints
   satisfy §4?
3. **Recompute each L0 / L1 formula** (§6, §7) by hand with the current dtype and device
   budgets from the docs site.
4. **Identify the first invariant violated**; propose the minimum change that restores it.
5. **Re-derive §4** (divisibility, ordering) after any change.
6. **Re-check §6 / §7 / §10** — changes can cascade.
7. **Write the proposal** to the operator memory file (e.g. `<op_dir>/MEMORY.md`) under
   `## tile-shape patch proposal`. Do NOT modify `_module*_impl.py` yourself; pypto-op-coder
   applies it.

---

## 13. Output format — patch proposal

Write the proposal to the operator's memory file (path supplied by the orchestrator):

```markdown
## tile-shape patch proposal (cycle N)

- File:   <op_dir>/modules/<op>_module<suffix>_impl.py
- Line:   <lineno of the failing set_cube_tile_shapes call>
- Before:
    pypto.set_cube_tile_shapes([256, 256], [256, 256], [256, 256])
- After:
    pypto.set_cube_tile_shapes([128, 256], [64, 128], [128, 256])
- Root cause: L1 budget exceeded (see §7). With FP32 and L1=512 KB,
    (256*256*4) + (256*256*4) = 524288 is at the limit but combined with
    kL1=256 halving needed ⇒ kL1=128.
- pypto-op-verifier step: pypto-op-coder re-runs verification; if still failing,
    return with the new error message for the next cycle.
```

Always include a reference to the project's `set_cube_tile_shapes` API doc in every proposal
so the pypto-op-coder has a single source of truth.

---

## 14. Budget constants cheat-sheet

| Symbol      | Typical value (A2/A3) | Override source            |
|-------------|----------------------|-----------------------------|
| `L0A_size`  | 65536 (64 KB)        | API config docs             |
| `L0B_size`  | 65536 (64 KB)        | API config docs             |
| `L0C_size`  | 131072 (128 KB)      | API config docs             |
| `L1_size`   | 524288 (512 KB)      | API config docs             |
| `BTBuffer`  | 1024 (1 KB)          | API config docs             |
| `FixBuffer` | 2048 (2 KB)          | API config docs             |

If the device's actual error text quotes a different number (e.g. `L1 = 786432`), trust the
error message and update your math accordingly.

---

## 15. Scope boundary (anti-pattern check)

This reference produces **patch proposals**, not commits. The pypto-op-debugger:

- MUST NOT open or modify `<op_dir>/modules/<op>_module*_impl.py`.
- MUST write the proposal to the operator's memory file (path supplied by the orchestrator).
- MUST cap retry cycles per the orchestrator's debug protocol; if still failing, escalate
  to pypto-op-orchestrator.

The pypto-op-coder applies the patch, then pypto-op-verifier re-judges.
