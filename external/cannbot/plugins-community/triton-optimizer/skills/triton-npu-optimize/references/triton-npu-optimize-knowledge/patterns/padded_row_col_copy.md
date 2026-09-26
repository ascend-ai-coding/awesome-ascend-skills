# Padded Last-Dim Row–Column Tiling

## Summary

Rewrite flat 1D copy kernels over `numel(out)` into row–column tiled form to reduce scalar overhead from per-element coordinate reconstruction on the last dimension.

## Use When

- The operator is **constant pad**, **slice + pad**, or another **per-axis bounds** elementwise map (not gather).
- The baseline uses **`pid * BLOCK + arange`** over **`numel(out)`** with **heavy div/mod** for **all** coordinates each iteration.
- Profiling shows **high scalar** or **`tl.load` / mask** cost on **last-dim** pad boundaries.

## Signals

### Code

- Single linear `offsets` and repeated `//` / `%` on **large strides** to recover the **last** coordinate.
- One **global `valid`** merging every dimension on each lane of a large flat block.
- **Multi-phase** column loops (left / interior / right) with **different** `tl.store` masks.

### Profile

- Scalar or control overhead out of proportion to copy bandwidth.
- Hot path dominated by **masked load** or **compare** chains for pad bounds.

## Avoid When

- The hot path is **gather/scatter** or **index-driven** discrete access (prefer `discrete_memory_access`).
- **Dynamic `if`/`elif` on tile kind** inside the column loop is required without proof the backend lowers it safely—prefer a **uniform** column loop on Ascend unless validated.
- **Interior-only** fast paths that **omit `col_mask`** on `tl.store` without proof for **tail** blocks (`out_dim_last % BLOCK_COLS != 0`).

## Optimization Strategy (reference)

1. **Structure:** `out_rows` = product of output dims except the last (after any fixed-rank padding your stack uses); **`out_dim_last`** = last extent; grid `ceil_div(out_rows, BLOCK_ROWS)` with hardware caps if needed; `tl.range` row stride with `num_programs * BLOCK_ROWS`; **`static_range(BLOCK_ROWS)`**; column loop `ceil_div(out_dim_last, BLOCK_COLS)`; **`store_edge = row_mask & col_mask`**, **`col_mask = cols < out_dim_last`**; compute **input row base** once per row lane before the column loop.
2. **`BLOCK_ROWS`:** use **4 or 8** when **`out_dim_last`** is small and **`out_rows`** is large (tune; UB/register limits).
3. **Masks:** `valid_row` for high dims; load mask includes last-dim pad range unless `NO_COL_PAD`.
4. **`NO_COL_PAD`:** when **`pad_left_last == 0`** and **`in_dim_last == out_dim_last`**, constexpr branch drops last-dim pad compares; **`in_cols = cols`**.
5. **Host `BLOCK_COLS`:** optionally skip refinement for a wide-last-dim regime; else require **interior column work × rows** ≥ threshold or **halve** `BLOCK_COLS` (floor ~16) with a cap on extra iterations (e.g. avoid **>4×** column blocks).
6. **`NATIVE_MASKED_LOAD`:** typical vs wide-last-dim paths are **backend- and dtype-dependent**; one useful split is narrow regime with `other=fill_value` vs wide with `other=0.0` + fp32 + `tl.where`—**profile**, do not assume “wide always uses fill_value”.

**Ascend note:** Prefer **one** column loop with **`col_mask` on every store**; recover perf via host **`BLOCK_COLS`** tuning, not weaker tail-unsafe store masks.

## What To Verify After Applying

- **Correctness:** **`out_dim_last % BLOCK_COLS != 0`**, asymmetric pad, supported ranks and dtypes.
- **Benchmark:** vs your baseline and framework reference under the same harness.
- **Profiler:** scalar / mask / load mix before changing refinement thresholds.
- **Edge:** zero output numel early exit if applicable.

## Related Patterns

- `tiling`
- `program-multiple-rows`
- `loop-invariant-hoisting`
- `autotune`
- `compile_hint`
- `vec-cmp`
