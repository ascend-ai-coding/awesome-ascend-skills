---
id: simt-clip-window-closed-reduction
priority: normal
---

# SIMT Clip-Window And Closed-Form Normalizer

## Summary

**Inner-loop structural repair** for **fixed-window reductions** over **affine layouts** on **SIMT paths**: compute each output window's **clipped input bounds once**, use a **closed-form window volume** as normalizer (e.g. mean divisor), and iterate **absolute coordinates inside the clip** instead of scanning the full `KERNEL_*` cube with per-tap validity masks and runtime counting.

**Pattern signature:** constexpr `KERNEL_D/H/W` loops + per-tap `valid_*` / `safe_*` masks + **`count += tl.where(mask)`** inside the nest for normalization.

**Not op-specific** — applies whenever normalization uses **clipped tap volume** (exclude virtual pad from the normalizer). Pool-style `count_include_pad=False` is the canonical example.

**Relationship to `a5-simt-sliding-window-tuning`:** that card owns outer dispatch, interior fast path, and mask vs clip routing. Use **this card** when normalizer semantics are **closed volume** (`d_len × h_len × w_len` from clip bounds). When normalizer must **include virtual pad extent**, prefer coordinate-mask + one-shot normalizer in the tuning card — A/B on harness before assuming clip is faster.

## SIMT Precondition

Treat an active SIMT execution path as a **required precondition**, not something to infer from mask-heavy window code alone.

Accept any of these as SIMT-in-use evidence:

- The kernel launch already passes **`force_simt_only=True`** (or an equivalent backend flag that forces SIMT-only execution).
- The current optimize round is applying **`a5-force-simt-only-discrete-access`** and SIMT-only launch is already enabled and compiling successfully.
- The user explicitly states the kernel runs on a **SIMT-only** discrete-access path.

If SIMT is not already in use, route to **`a5-force-simt-only-discrete-access`** first; do not expect the same benefit on non-SIMT launches.

## Structural Repair Precedence

1. **Outer structure**: flat `numel(out)` + hot-path coordinate decode → `flat-index-decode-tiling`.
2. **Launch mode**: A5 scalar/index dominated → `a5-force-simt-only-discrete-access`.
3. **Dispatch / inner routing**: `a5-simt-sliding-window-tuning` — single launch, flat/row×col, interior path, mask vs clip by semantics and stack_pressure.
4. **Inner structure (this card)**: **closed normalizer** — clip bounds + closed-form volume; no per-tap counting.
5. **Inner-W slab**: `sliding-window-inner-w-slab-gather` on **non-SIMT** paths only when profile proves slab+gather wins.

Closed-form divisor is **related to but not the same as** generic loop-invariant hoisting (`loop-invariant-hoisting`): hoisting moves unchanged expressions out of a loop, while this pattern **replaces an O(window_volume) accumulation** (`valid_count += 1`) with an **O(1) algebraic formula** derived from the same clip bounds used for loads.

## Use When

- **SIMT active** (`force_simt_only=True` or documented SIMT round).
- **Fixed affine window** over N-D layout; output→input map is static per lane.
- Normalizer = **clipped tap count** (exclude virtual pad from volume), not include-pad extent.
- Hot path: **`for kd/kh/kw`** + per-tap **`valid_*` / `safe_*`** and/or **`count += tl.where(...)`** for normalizer.
- Padding / ceil / edge partial windows; mapping still affine.
- Correctness checkable vs framework reference on boundary shapes.

## Avoid When

- **SIMT is not enabled yet** — apply `a5-force-simt-only-discrete-access` first on A5 instead of using this pattern as a pre-SIMT cleanup.
- The kernel is on an **HIVM W-slab** or other non-SIMT compile path where `force_simt_only` is off or incompatible.
- Indices are value-dependent or the window is not a fixed affine span (true gather/scatter with irregular offsets).
- The kernel is already dominated by Cube/matmul work; scalar window bookkeeping is not the bottleneck.
- **Include-pad normalizer** is required — use coordinate-mask + one-shot normalizer (`a5-simt-sliding-window-tuning` §4) unless harness A/B proves clip wins.
- Applying clip-window would require dynamic loop trip counts per lane that the backend cannot lower safely, and the fallback kernel-index path is already fast enough on the current SIMT launch.
- You are about to abandon or have not finished validating the SIMT-only launch (compile `507035` repair, correctness) — stabilize SIMT first.

## Signals

### Code

- Innermost work is **`for kw in range(KERNEL_W):`** with **`in_w = start_w + kw`**, then **`valid_w = (in_w >= 0) & (in_w < in_w)`** and **`safe_w = tl.where(valid_w, in_w, 0)`**.
- Average pooling uses **`valid_count += tl.where(window_mask, 1, 0)`** or **`padded_count += ...`** inside the `kd/kh/kw` loops.
- **`start_* = o* * STRIDE_* - PAD_*`** is computed once per output lane, but clip bounds are **not** reused for divisor and loads consistently.
- **`x_base = n * stride_n + c * stride_c`** is recomputed inside every kernel tap instead of before the inner loops.

### Profile

- Profile row is for a kernel already launched with **`force_simt_only=True`** (or documented SIMT-only round).
- High **`aiv_scalar_ratio`** on the SIMT kernel; math is mostly load + reduce + normalize.
- Many compare/mask/where operations relative to useful loads, especially on padded or `count_include_pad=False` cases.
- Gains from SIMT-only launch and block tuning have plateaued while the inner loop body remains mask-heavy.

### IR

- Repeated **`subi/minsi/maxsi`** and **`select`** inside lowered loops over **`kd/kh/kw`**.
- Integer increment chains for divisor counting inside the same loops as loads.

## Optimization Strategy

### Step 1 — Compute clip bounds once per output lane

Mirror reference forward semantics (e.g. PyTorch/CUDA sliding-window reduce). For each output coordinate, compute the **padded logical window**, then clip to the input tensor:

```python
start_d = od * STRIDE_D - PAD_D
start_h = oh * STRIDE_H - PAD_H
start_w = ow * STRIDE_W - PAD_W

tend_d = tl.minimum(start_d + KERNEL_D, in_d + PAD_D)
tend_h = tl.minimum(start_h + KERNEL_H, in_h + PAD_H)
tend_w = tl.minimum(start_w + KERNEL_W, in_w + PAD_W)

tstart_d = tl.maximum(start_d, 0)
tstart_h = tl.maximum(start_h, 0)
tstart_w = tl.maximum(start_w, 0)

tend_d = tl.minimum(tend_d, in_d)
tend_h = tl.minimum(tend_h, in_h)
tend_w = tl.minimum(tend_w, in_w)

d_len = tl.maximum(tend_d - tstart_d, 0)
h_len = tl.maximum(tend_h - tstart_h, 0)
w_len = tl.maximum(tend_w - tstart_w, 0)
```

Empty window (`d_len == 0` or `h_len == 0` or `w_len == 0`): leave accumulator at zero; set divisor fallback so `0 / 1 = 0`, matching reference early-return behavior.

### Step 2 — Closed-form divisor before the load loops

Do **not** count taps inside `kd/kh/kw`.

```python
if HAS_DIVISOR_OVERRIDE:
    divisor = DIVISOR_OVERRIDE
elif COUNT_INCLUDE_PAD:
    # use pre-clip padded spans
    tend_d_p = tl.minimum(start_d + KERNEL_D, in_d + PAD_D)
    tend_h_p = tl.minimum(start_h + KERNEL_H, in_h + PAD_H)
    tend_w_p = tl.minimum(start_w + KERNEL_W, in_w + PAD_W)
    d_len_div = tl.maximum(tend_d_p - start_d, 0)
    h_len_div = tl.maximum(tend_h_p - start_h, 0)
    w_len_div = tl.maximum(tend_w_p - start_w, 0)
else:
    d_len_div, h_len_div, w_len_div = d_len, h_len, w_len

divisor = (d_len_div * h_len_div * w_len_div).to(tl.float32)
divisor = tl.where(divisor > 0, divisor, 1.0)
```

For interior tiles with **`padding=0`** and fully in-bounds windows, this collapses to the constant **`KERNEL_D * KERNEL_H * KERNEL_W`** — a host-dispatch fast path is optional.

### Step 3 — CUDA-style clip-window load loop

Replace kernel-offset iteration with **absolute clipped input indices**:

```python
x_base = n * stride_n + c * stride_c  # hoist before loops
acc = tl.zeros(..., dtype=tl.float32)

for kd in range(KERNEL_D):
    ti = tstart_d + kd
    d_ok = kd < d_len
    for kh in range(KERNEL_H):
        hi = tstart_h + kh
        h_ok = kh < h_len
        for kw in range(KERNEL_W):
            wi = tstart_w + kw
            w_ok = kw < w_len
            load_mask = lane_mask & d_ok & h_ok & w_ok
            off = x_base + ti * stride_d + hi * stride_h + wi * stride_w
            acc += tl.load(x_ptr + off, mask=load_mask, other=0.0).to(tl.float32)

tl.store(..., acc / divisor, mask=lane_mask)
```

Notes for Triton/Ascend:

- Keep **`range(KERNEL_D)`** as a **constexpr upper bound**; use **`kd < d_len`** to skip invalid taps when per-lane clip lengths differ. This is the practical equivalent of CUDA's `for (ti = tstart; ti < tend; ++ti)` under SIMT divergence.
- Remove **`valid_*` / `safe_*`** when loads use **`tstart + kd`** and **`kd < d_len`**.
- On Ascend, prefer **`tl.zeros([BLOCK], dtype=tl.float32)`** over **`tl.zeros_like(..., dtype=...)`** when creating divisor tensors.

### Step 4 — Optional host fast paths (CUDA-inspired)

| Fast path | Guard | Effect |
|-----------|-------|--------|
| Constant divisor | `padding=0`, window fully inside input, `count_include_pad=True`, no override | `divisor = KERNEL_D*KERNEL_H*KERNEL_W` |
| No length masks | same as above | drop `d_ok/h_ok/w_ok`; inner loops become unmasked loads |
| `KERNEL_W` specialization | `kernel_w in 1..7` | separate `@triton.jit` instances; helps unrolling on CUDA-like backends |
| Interior / boundary split | mixed tiles | **Avoid global multi-launch on A5 SIMT** — see `a5-simt-sliding-window-tuning` §8; use single-kernel `constexpr` branches |

### Step 5 — Combine with outer tiling on an established SIMT launch

- Row-column tiling: compute clip bounds on **`(row, col)`** tiles; reuse the same formulas with broadcast shapes `(BLOCK_ROWS, 1)` and `(1, BLOCK_SIZE)`.
- Keep **`force_simt_only=True`** unchanged through this inner repair unless compile or correctness forces a rollback; after the structural change, retune **`BLOCK_*`**, grid decomposition, and **`num_warps`** on the same SIMT launch.

## Code Transformation

### Anti-pattern (kernel-index + per-tap count)

```python
for kd in range(KERNEL_D):
    in_d_index = start_d + kd
    valid_d = (in_d_index >= 0) & (in_d_index < in_d)
    safe_d = tl.where(valid_d, in_d_index, 0)
    for kh in range(KERNEL_H):
        ...
        for kw in range(KERNEL_W):
            window_mask = valid_d & valid_h & valid_w
            acc += tl.load(x_ptr + off(safe_d, safe_h, safe_w), mask=window_mask, other=0.0)
            if not HAS_DIVISOR_OVERRIDE:
                valid_count += tl.where(window_mask, 1, 0)
divisor = valid_count.to(tl.float32)
```

### Target (clip bounds + closed-form divisor + clip-window loads)

```python
tstart_d, tstart_h, tstart_w, d_len, h_len, w_len = clip_window(...)
divisor = closed_divisor(start_*, ..., COUNT_INCLUDE_PAD, ...)
x_base = n * stride_n + c * stride_c
for kd in range(KERNEL_D):
    ti = tstart_d + kd
    for kh in range(KERNEL_H):
        hi = tstart_h + kh
        for kw in range(KERNEL_W):
            wi = tstart_w + kw
            if kd < d_len and kh < h_len and kw < w_len:  # use vector masks in Triton
                acc += tl.load(x_ptr + x_base + ti*stride_d + hi*stride_h + wi*stride_w)
```

## Failure Modes And Anti-signals

- Using **post-clip lengths** for normalizer when **include-pad semantics** apply — wrong; use one-shot normalizer from padded extent.
- Applying clip for **heavy compare + half pad** without geomean proof — often loses to coordinate mask on A5 SIMT.
- Applying only closed-form divisor but leaving kernel-index + `valid_*` loops — partial gain only.
- Assuming clip-window removes all masks on **border outputs**; edge lanes still need `kd < d_len` guards.
- Applying this pattern on a **non-SIMT** launch or replacing the SIMT path with W-slab without measurement — may regress or conflict on compile path (HIVM slab vs SIMT-only).
- Nested `@triton.jit` helpers that use unsupported APIs (`tl.zeros_like(..., dtype=...)`) on Ascend.

## What To Verify After Applying

- Correctness vs framework reference: padding, closed vs include-pad normalizer, ceil edges, normalizer override, empty windows.
- Inner hot path no longer contains **`valid_count +=`** / **`padded_count +=`** in `kd/kh/kw` loops.
- Clip bounds match CUDA reference on sampled outputs (compare against manual golden for one lane).
- Benchmark on representative JSON cases: interior-heavy shapes vs padding-heavy shapes separately.
- Round record states SIMT was already enabled (`force_simt_only=True` or equivalent) before this inner repair.
- Retune `num_warps` / block sizes on the **same** SIMT launch after the structural change.
- Document whether W-slab was considered and rejected or deferred (only relevant if SIMT is abandoned).

## Related Patterns

- `flat-index-decode-tiling` — outer output traversal / row-column tile before SIMT inner-window repair
- `loop-invariant-hoisting` — hoist `x_base`, masks, and bounds; closed-form divisor is algebraic LICM plus counting elimination
- `a5-force-simt-only-discrete-access` — **required predecessor** on A5: enable and validate SIMT-only launch before inner-window repair
- `a5-simt-sliding-window-tuning` — dispatch, interior path, mask vs clip routing
- `sliding-window-inner-w-slab-gather` — inner-W slab on **non-SIMT** paths
- `exact-tile-no-boundary-fast-path` — drop `d_ok/h_ok/w_ok` when tile is fully interior
- `scalar-latency-traps` — remove redundant `safe_*`, narrow masks, and unsupported APIs
- `algebraic-optimization` — closed-form divisor is an algebraic replacement for incremental counting
