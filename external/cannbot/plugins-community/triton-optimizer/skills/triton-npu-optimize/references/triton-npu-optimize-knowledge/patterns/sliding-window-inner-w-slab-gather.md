---
id: sliding-window-inner-w-slab-gather
priority: normal
---

# Sliding-Window Inner-Dim Slab Plus Gather

## Summary

For **fixed-window reductions** along the **innermost contiguous spatial dimension** (NCHW-style: often **W**), load one **slab** of length **`W_SLAB_LEN = STRIDE_W * (BLOCK_OW - 1) + KERNEL_W`** at **`w_abs_min = ow_pid * BLOCK_OW * STRIDE_W - PAD_W`**, then **`tl.gather(slab, STRIDE_W * arange(BLOCK_OW) + kw)`** per **`kw`** instead of many **`start_w + kw`** masked loads. Higher rank differs only in outer loops over remaining spatial axes. **Adoption gate:** hot path must show **`W_SLAB_LEN` load + gather**.

**A5 SIMT (`force_simt_only=True`):** measured regression vs discrete SIMT flat/row×col on window-reduction harnesses — see **`a5-simt-sliding-window-tuning`** §8. Do not apply on A5 SIMT unless re-proven on your harness.

**Not op-specific** — any affine sliding-window reduce with overlapping taps along the innermost dim; pool ops are the canonical example.

## Use When

- The kernel is a **fixed `KERNEL_W` window reduction** (mean, max, etc., **values only**) along **W** on a **contiguous** NCHW (or 5D) tensor, with **`kw`** in a **`tl.constexpr`** loop and **`BLOCK_OW`** output columns per program.
- Profiling or IR shows **repeated narrow or predicate-heavy global loads** on **W** inside **`kw`**, while **`stride_w`** maps output columns to **regularly strided** input columns.
- **`out_w`** is large enough that **vectorizing along `ow`** matters, and **`triton.cmotion.cdiv(out_w, BLOCK_OW)`** (launch count along W) stays below a measured knee on the target NPU.
- You can prove **semantic equivalence** for the branches you enable (**padding**, **ceil**, **divisor** / **count_include_pad** for average, **`-inf` / dtype** rules for max).

## Detail

### Intuition

The problem this pattern targets is **overlap**. Adjacent output windows along **W** often reuse many of the same input columns, but a naive pooling loop still reloads those columns inside every **`kw`** step.

- Baseline mental model: for each **`kw`**, every lane does something like **`tl.load(input_ptr + start_w + ow * STRIDE_W + kw, mask=...)`**.
- Optimized mental model: first load the smallest contiguous W range that covers all **`BLOCK_OW`** output windows owned by the program, then reuse that staged slab for every **`kw`** with **`tl.gather`**.

The slab length is:

`W_SLAB_LEN = STRIDE_W * (BLOCK_OW - 1) + KERNEL_W`

That is exactly the span from the first input column needed by the tile to the last one.

### Example

Suppose:

- **`BLOCK_OW = 4`**
- **`STRIDE_W = 2`**
- **`KERNEL_W = 3`**

Then:

- **`W_SLAB_LEN = 2 * (4 - 1) + 3 = 9`**
- one output tile needs input columns **`[0..8]`**

The four output windows consume:

- **`ow = 0`** -> **`[0, 1, 2]`**
- **`ow = 1`** -> **`[2, 3, 4]`**
- **`ow = 2`** -> **`[4, 5, 6]`**
- **`ow = 3`** -> **`[6, 7, 8]`**

So the program can load one contiguous slab **`[0, 1, 2, 3, 4, 5, 6, 7, 8]`** once, then gather:

- **`kw = 0`** -> indices **`[0, 2, 4, 6]`**
- **`kw = 1`** -> indices **`[1, 3, 5, 7]`**
- **`kw = 2`** -> indices **`[2, 4, 6, 8]`**

The overlapping columns (**`2`**, **`4`**, **`6`**) are loaded once from global memory and then reused locally.

### Implementation Shape

- One program usually owns **`BLOCK_OW`** adjacent output columns.
- For each **`kh`** (and **`kd`** in 3D), load a contiguous slab along **W**.
- For each **`kw`**, gather with **`STRIDE_W * tl.arange(BLOCK_OW) + kw`** and reduce into a **`BLOCK_OW`**-wide accumulator vector.
- **2D pooling** uses **`kh` / `kw`** only; **3D** adds **`kd`** outside them. The slab logic itself does not change.

### Fast-Path Branches

- **`USE_W_SLAB_LOAD`**: use an unmasked slab load when padding is zero, the tile is a full **`out_w`** tile, and the entire slab is known in-bounds.
- **`USE_W_MASKED_SLAB`**: still use the slab rewrite when the tile is full but slab columns may cross padding or **`ceil_mode`** boundaries; load the slab with **`mask=`** / **`other=0`**, then gather the same way.
- Otherwise, fall back to a simpler branch such as **`NO_PADDING_FASTPATH`** or generic boundary handling.

### Launch Notes

- Host-side tuning usually picks **`BLOCK_OW`** from **`out_w`** using a bounded candidate set.
- A common **2D** grid is **`(batch * channels * out_h, cdiv(out_w, BLOCK_OW))`**.
- A common **3D** variant folds **`out_d * out_h`** into the row axis.
- Pair with **`program-multiple-rows`** when consecutive flat spatial rows can share enough setup to amortize slab overhead.

## Avoid When

- **A5 SIMT fixed-window reduce (`force_simt_only=True`)** — see **`a5-simt-sliding-window-tuning`** §8; W-slab regressed vs SIMT flat/row×col + coordinate-mask paths on multi-shape harnesses.
- **`W_SLAB_LEN`** exceeds **UB / compiler** or **gather** limits—**reduce `BLOCK_OW`** or use **`USE_W_MASKED_SLAB`** / non-slab branches for that shape.
- **Host predicates** for **`USE_W_SLAB_LOAD`** are wrong (windows not fully inside input, pad misclassified)—always **diff against the framework reference**.
- **Tail tiles:** **`out_w % BLOCK_OW != 0`** requires **`ow_mask`** on **store** and usually **disables full-tile slab branches** unless tail handling is explicit.
- **Max-like reductions** with **`return_indices`**, **dilation**, or extra validity state — W-slab may help **loads** only; compare/index update needs a separate design and full geomean proof.
- **Layout** is not **W-contiguous** per row (fix with **`contiguous()`** or a different pattern).
- **A5 SIMT discrete window reduce** — follow **`a5-simt-sliding-window-tuning`** (single launch, feature-derived dispatch/inner path) before W-slab; use **`simt-clip-window-closed-reduction`** only when normalizer uses **clipped volume only**, not as default for all pad cases.
- **One failed gather attempt** on device—treat as **implementation bug**, not “pattern invalid”; see **Failure playbook** below before abandoning gather.

## Signals

### Code

- **Baseline:** **`for kh`** (and **`for kd`** in 3D) with **`for kw`** doing **`tl.load(..., mask=window_mask)`** per **`kw`** on W.
- **Target:** per **`(kd, kh)`**, one **`tl.load`** of **`W_SLAB_LEN`** columns, then **`KERNEL_W`** **`gather`**s into **`BLOCK_OW`** lanes.
- **Host constexpr flags** (names illustrative): **`USE_W_SLAB_LOAD`**, **`USE_W_MASKED_SLAB`**, **`NO_PADDING_FASTPATH`**, **`FULL_W_TILE`**, **`BLOCK_OW`**, **`W_SLAB_LEN`**.

### Profile

- **`high-transfer-pressure`** or many **discrete / narrow LD** ops on the pooling kernel name; **Avg** improves when IR shows fewer per-`kw` GM touches.
- **High `aiv_scalar_ratio`** alone does **not** mean “skip slab”—often means **add boundary specialization**, not **drop gather**.

### IR

- **Before:** repeated **small loads** or **per-lane** predicates on W inside lowered **`scf.for`** over **`kw`**.
- **After:** **`tensor<W_SLAB_LEN x dtype>`** (or masked load of that rank), then **`hfusion.gather` / `triton_gather`** into **`BLOCK_OW`**, then **`arith.addf`** or **`maxnumf`**.

## Optimization Strategy

### 1. Indexing and tile geometry (must stay consistent)

| Symbol | Definition |
|--------|------------|
| **`BLOCK_OW`** | Output columns per program along W; prefer **`out_w % BLOCK_OW == 0`** on slab branches (**`FULL_W_TILE`**). |
| **`W_SLAB_LEN`** | **`STRIDE_W * (BLOCK_OW - 1) + KERNEL_W`** — span of input columns touched by the OW tile. |
| **`w_abs_min`** | **`ow_pid * BLOCK_OW * STRIDE_W - PAD_W`** (+ region W offset if pooling a subtensor). |
| **`lane_offsets`** | **`STRIDE_W * tl.arange(0, BLOCK_OW)`** — gather indices for each output lane at fixed **`kw`**. |

**Do not** mix ad-hoc “strip” lengths (e.g. **`block_ow - 1` without `STRIDE_W`**, or per-lane **`start_w + kw`** as gather base) with the table above—common cause of **UB OOB** on Ascend.

### 2. Host branch selection (constexpr)

| Branch | Enable when | Kernel behavior |
|--------|-------------|-----------------|
| **`USE_W_SLAB_LOAD`** | **Zero pad** (or pad only outside W), **`FULL_W_TILE`**, every window **fully inside** unpadded input (no partial D/H/W window) | Unmasked slab **`tl.load`**, then **`gather`**. |
| **`USE_W_MASKED_SLAB`** | **`FULL_W_TILE`**, but slab columns can be **OOB** (padding / ceil on W) | **`col = w_abs_min + arange(W_SLAB_LEN)`**, **`mask = (col >= 0) & (col < in_w)`**, **`load(..., mask=, other=0)`**, then **`gather`**. |
| **`NO_PADDING_FASTPATH`** | No pad, not eligible for slab, or **tail / partial window** | Per-**`kw`** vector **`tl.load`** with **`ow_mask`** only. |
| **Generic boundary** | **`ceil_mode`**, partial windows on D/H/W | Keep **safe indices + `window_mask`** per **`kw`**; **do not** force slab. |

Pick **`BLOCK_OW`** from **`out_w`** with a **small candidate list** (e.g. 128…1) favoring **divisors of `out_w`** when **`out_w`** is large; cap tile size for UB. **Do not** jump tile sizes arbitrarily (e.g. 8→32) without checking **`W_SLAB_LEN`** and **`FULL_W_TILE`**.

### 3. Grid (workload-specific)

- **2D:** **`(batch * channels * out_h, triton.cmotion.cdiv(out_w, BLOCK_OW))`** — one logical **`oh`** per **`program_id(0)`** row bundle.
- **3D:** often **`(batch * channels * out_d * out_h, triton.cdiv(out_w, BLOCK_OW))`** — fold depth×height into the row axis; **W slab is unchanged**.

**Avoid** folding **many `oh` (or `od×oh`) into one program** while also expanding an **H-slab** loop unless IR/profile proves benefit—launch count drops but **per-program GM traffic** can explode.

### 4. Orthogonal optimizations (combine, do not replace)

- **Interior / boundary output splitting** (separate launches for full-window vs edge tiles) can combine with slab on **non-SIMT** paths. On **A5 SIMT**, global region multi-launch is an anti-pattern — see **`a5-simt-sliding-window-tuning`** §8; prefer single-kernel `constexpr` branches instead of stacking splits + slab.
- **`program-multiple-rows`**: amortize setup when batching **multiple flat spatial rows** per program—profile launch vs reuse.
- **Do not** mark this pattern “done” if the final kernel has **zero `tl.gather`** on the hot path.

### 5. Ascend / dtype notes

- If **`tl.gather`** on **`f16`/`bf16`** faults, try **`slab = load(...).to(tl.float32)`** before **`gather`**, then accumulate in **`f32`**—validate numerics for **avg** and **max**.
- After edit, confirm IR shows **`gather`** on the **slab tensor**, not only **`memref.copy`** loops without consumption—or **repeated per-`kw` loads**.

## Detection Pattern

**Problematic (conceptual baseline)**

```python
@triton.jit
def pool_w_baseline(..., BLOCK_OW: tl.constexpr, KERNEL_W: tl.constexpr, STRIDE_W: tl.constexpr):
    ow_pid = tl.program_id(1)
    ow = ow_pid * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow < out_w
    start_w = ow * STRIDE_W - PAD_W  # per-lane window start along W
    acc = tl.zeros([BLOCK_OW], dtype=tl.float32)
    for kh in range(KERNEL_H):
        h_base = row_base + (start_h + kh) * stride_h
        for kw in range(KERNEL_W):
            # Many predicate-heavy narrow loads — poor on NPU when W is contiguous
            v = tl.load(x_ptr + h_base + (start_w + kw) * stride_w, mask=ow_mask, other=0.0)
            acc += v.to(tl.float32)
```

## Code Transformation

**Target (masked slab branch — illustrative)**

```python
@triton.jit
def pool_w_slab_gather(
    ...,
    BLOCK_OW: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_W: tl.constexpr,
    W_SLAB_LEN: tl.constexpr,
    PAD_W: tl.constexpr,
    USE_W_MASKED_SLAB: tl.constexpr,
):
    ow_pid = tl.program_id(1)
    ow = ow_pid * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow < out_w
    w_abs_min = ow_pid * BLOCK_OW * STRIDE_W - PAD_W
    lane_offsets = STRIDE_W * tl.arange(0, BLOCK_OW)
    acc = tl.zeros([BLOCK_OW], dtype=tl.float32)

    for kh in range(KERNEL_H):
        h_base = row_base + (start_h + kh) * stride_h
        if USE_W_MASKED_SLAB:
            j = tl.arange(0, W_SLAB_LEN)
            col = w_abs_min + j
            slab_mask = (col >= 0) & (col < in_w)
            safe_col = tl.where(slab_mask, col, 0)
            slab = tl.load(
                x_ptr + h_base + safe_col * stride_w,
                mask=slab_mask,
                other=0.0,
            )
        else:
            slab = tl.load(
                x_ptr + h_base + (w_abs_min + tl.arange(0, W_SLAB_LEN)) * stride_w
            )
        for kw in range(KERNEL_W):
            values = tl.gather(slab, lane_offsets + kw, axis=0)
            acc += values.to(tl.float32)

    # average: acc / divisor; max: tl.maximum(acc, values) with appropriate init
    tl.store(out_ptr + out_base + ow * out_stride_w, acc / divisor, mask=ow_mask)
```

**Host launcher sketch (no operator names)**

```python
def pick_block_ow(out_w: int, candidates: tuple[int, ...], max_single: int) -> int:
    if out_w <= max_single:
        return out_w
    for b in candidates:
        if b <= out_w and out_w % b == 0:
            return b
    return 1

def windows_fully_inside(in_w, out_w, kw, sw, pad_w) -> bool:
    return (out_w - 1) * sw + kw <= in_w  # extend with D/H for 3D

block_ow = pick_block_ow(out_w, (128, 64, 32, 16, 8, 4, 2, 1), 128)
full_w_tile = (out_w % block_ow) == 0
w_slab_len = sw * (block_ow - 1) + kw
use_w_slab = zero_pad and full_w_tile and windows_fully_inside(...)
use_w_masked_slab = full_w_tile and not use_w_slab

kernel[grid](..., BLOCK_OW=block_ow, W_SLAB_LEN=w_slab_len,
             USE_W_SLAB_LOAD=use_w_slab, USE_W_MASKED_SLAB=use_w_masked_slab, ...)
```

## Failure Playbook

| Symptom | Likely cause | Action (before abandoning gather) |
|---------|--------------|----------------------------------|
| **UB / vec address OOB** | **`W_SLAB_LEN`** mismatch; **`gather` index ≥ slab length**; wrong **`w_abs_min`**; **`BLOCK_OW`** vs tail mask | Recompute table in §1; enforce **`FULL_W_TILE`** on slab branches; shrink **`BLOCK_OW`**. |
| **Sync / illegal instruction** | **`gather`** on unsupported dtype | **`extf` slab to `f32`** before **`gather`**. |
| **Correctness drift** | Slab branch enabled with **partial D/H/W** windows | Disable **`USE_W_SLAB_LOAD`**; use **masked slab** or **generic** path. |
| **No perf gain** | Hot path still **per-`kw` load** (gather optimized away) | Inspect IR; fix branches so **slab path actually runs**. |
| **Regression on deep shapes** | Extra **region launches** without slab on interior | **Gate splits**; keep **slab on large interior regions**. |

**Do not** conclude “slab-gather invalid on Ascend” from a single failed attempt without completing this checklist and verifying IR on a failing case.

## Related Patterns

- `gather-load`
- `discrete_memory_access`
- `constexpr-tile-discrete-access`
- `program-multiple-rows`
- `scalar-latency-traps` (supporting lens for boundary math—not a substitute for W staging)
- `a5-simt-sliding-window-tuning` — A5 SIMT fixed-window tuning (use before W-slab on `force_simt_only` paths)
- `simt-clip-window-closed-reduction` — closed normalizer inner loop; not the default include-pad path on A5 SIMT

## What To Verify After Applying

- **Correctness:** framework **`avg_pool{2,3}d` / `max_pool{2,3}d` (values)** — **padding**, **ceil**, **divisor_override**, **count_include_pad**, **tail `out_w`**.
- **Pattern applied:** IR or source shows **`W_SLAB_LEN` load + `gather`** on the profiled hot case—not only region splits or fastpaths.
- **Benchmark:** **Avg / geomean** vs baseline; watch shapes with **many sub-launches** (depth × boundary bands).
- **Numerics:** **fp16/bf16** with **`f32` gather staging** if used.
