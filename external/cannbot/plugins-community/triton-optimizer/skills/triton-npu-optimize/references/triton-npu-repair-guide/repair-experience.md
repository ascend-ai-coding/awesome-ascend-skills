# Ascend Triton operator repair experience

Heuristic fixes for **Triton Ascend** kernels. These are **not guaranteed optimal**; match the compiler error or symptom, apply a **minimal** change, and re-run validation. Extend this list as new patterns appear.

---

## 1. Prefer `tl.math.*` over `libdevice.*` for elementwise math

The Ascend toolchain often routes “libdevice-style” math through symbols that are **unsupported or missing** compared to CUDA-oriented references. When the error names `libdevice.<name>`, try the **`tl.math`** equivalent first.

| Message / clue | Try |
|----------------|-----|
| `libdevice.tanh` (or similar) | `tl.math.tanh` |
| `libdevice.erf` (or similar) | `tl.math.erf` |
| Other `libdevice.<op>` | `tl.math.<op>` if it exists in your Triton version |

If `tl.math.erf` is still unavailable or lowering fails, a **small analytic approximation** (e.g. tanh-based) was used in one internal case—only when you can accept the numerical tradeoff and document it.

Keep semantics (dtype, masking) identical aside from the call target.

---

## 2. `allow_tf32` deprecated

Older flags such as **`allow_tf32`** may be rejected or deprecated on Ascend Triton.

- Prefer the documented replacement (e.g. **`input_precision='hf32'`** where the API supports it).
- Ensure operands that participate in that path use **float32** as expected by the compiler (cast or construct tensors/parameters in **fp32** so lowering matches intent).
- **TF32-style rounding is often meaningless on Ascend**: if the kernel mixes “TF32 dot” helpers with Ascend constraints, prefer a plain **fp32** dot path and remove TF32-only rounding that does not apply.

---

## 3. Infinity literals and `float32`

If lowering fails with **dtype / inference** issues around special values, ensure **infinity** is visible to the compiler as **float32** (e.g. scalars feeding `tl.full`, masks, or constants). Integer or ambiguous dtypes for `inf` can break inference.

---

## 4. UB overflow → reduce block size

Errors indicating **UB (unified buffer) overflow** or on-chip scratch exhaustion:

- **Reduce** `BLOCK_SIZE`, tile size, or other per-block parameters so each block uses less local/UB budget.
- Re-tune for performance only after correctness is stable.

Related log pattern: UB issues combined with **`hivm.hir.load` / `vcast` / `vsel`** in diagnostics—still try **smaller tiles** first before deeper IR work.

---

## 5. `TypeError: 0d block_type is forbidden` (Ascend)

The Ascend Triton compiler may reject certain **zero-dimensional `block_type`** forms.

**Symptom:** compile-time `TypeError` mentioning **`0d`** and **`block_type`**.

**Direction:**

- Avoid shapes that lower to a forbidden **scalar block** in that position. A pattern that has worked in practice is to materialize the value with an explicit dtype and a **defined** rank, for example:

  ```python
  sxn = tl.broadcast_to(tl.full((), sxn, tl.int64), ())
  ```

  Adapt the **variable name**, **dtype** (`tl.int64` vs `tl.float32`, etc.), and surrounding uses to your kernel. The intent is to replace a naïve 0-D form the compiler rejects with an explicit `tl.full` + `tl.broadcast_to` (or equivalent) so the block type is acceptable.

- This is **context-dependent**; if one reshape does not fix it, simplify how scalars enter loads/stores so they always pass through explicit `tl.full` / broadcasts with clear dtypes.

---

## 6. `while` loops vs bounded `for` + `range`

Many historical **general `while`** / IR **`scf.while`** failures are addressed by **CANN / bisheng upgrades**—prefer upgrading the stack before large kernel rewrites. If loop-related compiler crashes **persist**, try a bounded **`for k in range(...)`** with an explicit step; avoid **`tl.static_range`** that **fully unrolls** a huge trip count (IR blowup / compile timeout).

---

## 7. `tl.view` arguments (shape vs dtype)

**Symptom:** type/shape errors or bogus lowering when reshaping tensors in the kernel.

**Direction:** Confirm **`tl.view`** receives a **shape** (or correct tuple of extents), not a **dtype** where a shape was intended (internal case: `tl.view(a32, tl.int32)`-style mistakes). Match the Triton API for your version.

---

## 8. Chained boolean conditions

**Symptom:** odd failures on boolean-heavy expressions in `@triton.jit` code.

**Direction:** Split **compound boolean** conditions into intermediate masks or simpler tests so lowering is clearer for Ascend (internal case: “chained boolean” refactor in the kernel).

---

## 9. `tl.dot` and mixed dtypes

**Symptom:** compile or runtime issues when **`tl.dot`** mixes **fp16** and **fp32** operands in a path the stack handles poorly.

**Direction:** Align operands explicitly, e.g. **`.to(tl.float32)`** (or your target dtype) on both sides of **`tl.dot`** before the multiply-add, then cast back if needed.

---

## 10. `tl.trans` before `tl.dot` and AI Core alignment

**Symptom:** compile may succeed but execution hits **AI Core `ADDR_MISALIGN`** or similar when **`tl.trans`** participates in a **`tl.dot`** chain.

**Direction:** Change **load layout / pointer indexing** so the tile used in **`tl.dot`** already has the desired orientation—e.g. build **`[T, C]`** strides instead of **`[C, T]`** + **`tl.trans`**, and use **`tl.dot(x_tile, w_tile)`** without transposing one operand. Adjust **masks** to match the new row/column layout. Semantics should stay equivalent; this avoids some bad instruction sequences on NPU.

---

## 11. Numerical mismatch (1 ULP) between Triton and torch on Ascend

**Symptom:** differential test shows non-zero diffs, typically 1 ULP in fp32 scale or
±1 in int8 output, affecting a small percentage (<5%) of rows/elements. Reduction
ops (`tl.max`, `tl.sum`) are already confirmed bit-exact.

**Root cause on Ascend:** Triton keeps `x / CONST` as hardware **divide**
(`__hmf_div`), while torch CANN optimizes it to **multiply** by precomputed
reciprocal (`__hmf_mul`). Ascend's divider and multiplier have **different
rounding** → results differ by 1 ULP for 1-5% of values.

**How to confirm:**
1. Build a kernel with only the reduction (no division): compare `triton_max` vs `torch.abs().max()`. Should be bit-exact.
2. Build kernels for Triton DIV vs Triton MUL vs Torch DIV vs Torch MUL. If Triton MUL == Torch DIV == Torch MUL but Triton DIV differs, the diagnosis is confirmed.
3. Print hex of diff values — should be exactly 1 ULP.

**Fix:**

```python
# Before (differs from Torch):
scale = tl.maximum(row_max_abs / 127.0, 1e-10)

# After (matches Torch bit-exact):
INV_127: tl.constexpr = 1.0 / 127.0
scale = tl.maximum(row_max_abs * INV_127, 1e-10)
```

**Why:** Explicit `*(1.0/CONST)` uses Ascend's **multiplier** (same as Torch CANN),
producing bit-identical results. Also typically faster than division.

**Verify:** re-run differential test — should be 0 diffs.
