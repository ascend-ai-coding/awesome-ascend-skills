# Stencil Resize: GM→UB Slab Staging and IR Quality

## Summary

For 2D sampling kernels (resize, gather-stencil, pooling-like windows) whose hot path reads **multiple overlapping input samples per output point**, the core memory strategy is: **stage one contiguous input slab from global memory (GM) into on-chip buffer (UB) per program**, flatten it, then serve stencil reads via **UB-resident gather** and vector math. Secondary wins — often larger than tuning gather itself — are **eliminating UB internal densify copies**, **aligning slab row stride with gather indexing at allocation time**, **tightening slab bounds to the true input window**, and **enlarging the output tile to cut program count**. Prefer **2D output tiling** `(BLOCK_H, BLOCK_W)` over flattening the output plane onto a single 1D program axis.

## Use When

- Kernel is memory-bound; input read count scales with `(output pixels × stencil footprint)`.
- IR shows `{DiscreteMemAccess}` / `{ExtractedLoadOrStore}` on per-lane GM loads in the stencil hot path.
- IR shows **`hivm.hir.copy` UB strided → UB dense** between slab load and gather.
- Source maps a **2D output tile** through a **1D linear program layout**, or uses **dynamic-index global loads** per stencil sample instead of a staged slab.
- msprof shows high **ST/LD** or inner **call_count** while **VGATHER share stays ~1%** — layout/launch issue, not gather-compute bound.

## Avoid When

- One input sample per output and access is already row-coalesced — slab may not amortize.
- Input layout is non-contiguous; rectangular GM→UB copy is invalid without reorder.
- Memory IR is clean but scalar control (`arith.select` chains) dominates — see weight-precompute / scalar patterns.
- Larger tile exceeds UB budget or grows slab faster than it saves programs — re-score candidates.

## Overview: Before vs After

| Dimension | Before (typical early path) | After (target path) |
|-----------|----------------------------|---------------------|
| **Parallelism** | 1D flattened output indexing | 2D tile `(BLOCK_H, BLOCK_W)` per program |
| **Read GM** | Discrete load **per sample × per output** | **One rectangular `memref.copy`** per program |
| **Read UB** | None; or strided slab + **densify copy** | Aligned slab; **direct reshape + gather** |
| **UB layout** | Logical size ≠ physical stride | `slab_h × align_up(logical_w, 8)`; index `y_off * IN_W + x_off` |
| **Write GM** | May be OK; read dominates | Block `tl.store` over full tile |
| **Launch** | Small tile → many programs | Scored picker: fewer programs when slab×grid wins |
| **Warps** | Default 8 | `tile_elems ≥ 128` → 16 |
| **IR** | GM scalar loops; UB densify `hivm.hir.copy` | GM copy + aligned UB + `hivm.hir.vgather` + vector FMA |
| **msprof** | High ST/LD, high call_count, M-scale cycles | Lower cycles/call_count; VGATHER still ~1% |

## Generic Optimization Directions (Prioritized)

Apply in order. Items ①–⑤ are **mode-agnostic**; item ⑥ is compute-path specific but listed for completeness when profiling shows scalar dominance after ①–⑤.

| # | Direction | Host / Triton lever | IR / msprof signal |
|---|-----------|---------------------|-------------------|
| ① | **UB row stride alignment** | `IN_W = align_up(logical_slab_w, 8)`; gather `y_off * IN_W + x_off` | Index row-step constant == `IN_W`; **zero** UB densify copy |
| ② | **Eliminate UB densify copy** | Allocate UB with **final aligned stride** before first use | `hivm.hir.copy` strided→dense: tens → **0** |
| ③ | **Tight slab bounds** | `slab = f(block, scale, stencil_radius)` from block corner — not oversized padding | Flat slab elements/program drop 10×–100×; GM→UB bytes ↓ |
| ④ | **Large tile, fewer programs** | Score `(BLOCK_H, BLOCK_W)` by **grid count + slab footprint** | Programs ÷~4 when tile doubles; gather width ×4, count/program often unchanged |
| ⑤ | **Warp heuristic** | `num_warps = 16 if BLOCK_H*BLOCK_W >= 128 else 8` | Better vector path occupancy |
| ⑥ | **Precompute repeated scalars** | Host tables / constexpr for periodic weights | `arith.select` / polynomial chains shrink (after memory clean) |

**Generic thesis:** Match UB physical layout to gather indices, **remove UB densify**, **minimize GM→UB volume**, **cut program count** — msprof cycles and ST fall; VGATHER share often unchanged (~1%).

## IR: Bad Signals

| IR feature | Meaning | Typical source |
|------------|---------|----------------|
| `{ExtractedLoadOrStore}` + `{DiscreteMemAccess}` | Scalar **GM** load per lane | Dynamic-index `tl.load` from global |
| **`hivm.hir.copy` UB strided → UB dense** | Densify tax before gather | Logical width ≠ allocated row stride |
| Index row step ≠ `IN_W` | Layout / gather mismatch | `y*18` into stride-24 buffer |
| `scf.for` trip = large `BLOCK` + heavy mask | Wasted scalar iterations | 1D block on 2D output |
| Many `arith.select` on periodic index | Scalar hot path | Max-table select for small period |
| Tens of **internal** `hivm.hir.copy` | UB layout bug | Fix ①② first |

**Two copy kinds — do not conflate:**

| Copy | Keep? | Role |
|------|-------|------|
| `memref.copy` GM → UB | Yes | Intended slab staging |
| `hivm.hir.copy` UB → UB densify | **No** | Accidental; eliminate |

**Bad IR (remove):**

```mlir
memref.copy %gm, %ub_strided : ... strided<[24, 1]>    // logical 18, stride 24
hivm.hir.copy ins(%ub_strided) outs(%ub_dense)       // densify
%flat = collapse_shape %ub_dense -> 324 elements
// gather row step c18 — inconsistent with stride 24
```

## IR: Good Signals

| IR feature | Meaning |
|------------|---------|
| One GM **`memref.copy`** / program | Staging |
| UB **`memref<H×IN_W>`**, `IN_W = align_up(w,8)` | Physical = logical for gather |
| **`reshape` / `collapse_shape` on aligned UB** without densify | Direct flat buffer |
| Gather row multiplier == **`IN_W` constant** | Host/device agreement |
| **`hivm.hir.vgather ins(ub_flat)`** | Stencil from UB |
| Wide gather × fewer programs | e.g. 256-wide × 16 calls/program |
| Block vector **store** to GM | Coalesced write |

**Good IR (target):**

```mlir
memref.copy %gm_subview, %ub : ... strided<[IN_W, 1], ub>
// no densify
%flat = tensor.reshape %ub : tensor<H×IN_W> -> tensor<H*IN_W>
%samples = hivm.hir.vgather ins(%flat) indices(%idx) -> tensor<WIDE>
```

## Tile, Gather, and Launch Geometry (Generic)

When output tile area increases (e.g. 64 → 256 points/program):

| Metric | Smaller tile | Larger tile | Implication |
|--------|--------------|-------------|-------------|
| Programs (fixed output) | More | Fewer (~÷4 when area ×4) | Launch/sync ↓ |
| Gather ops / program | Often ~stencil_rank² for separable | Same order | **Not** reduced by tile alone |
| Indices / gather | Narrower | Wider | Per-program gather work ×4 |
| **Total gather over grid** | programs × narrow | (programs/4) × wide | **≈ constant** |
| Slab / program (if formula tied to block) | May be loose | Tighter formula helps | GM→UB often **main win** |

**Do not** expect VGATHER msprof share to collapse when only enlarging tile. Watch **program count**, **GM→UB bytes**, **densify/ST**.

**Launch picker (generic):**

```python
def pick_tile(candidates, score_fn):
    # score_fn penalizes grid_count * slab_elems (and UB overflow)
    return min(candidates, key=score_fn)
```

## msprof Interpretation (Generic)

Compare **same kernel**, before/after IR fix, single core (`instr_exe` / `code_exe`).

| Metric | Unoptimized class | Optimized class | Read as |
|--------|-------------------|-----------------|---------|
| **code_exe cycles** | ~M | ~M/2 or better | Primary signal |
| **Hotspot call_count** | 100k+ | 10k–30k | Fewer programs / less glue |
| **VGATHER share** | ~0.7–1.5% | ~1% | **Usually not cap** |
| **BAR share** | moderate | **higher %** | Total cycles fell — relative, not regression |
| **ST/LD in Top** | yes | often absent | Densify / extra traffic gone |

Bench (end-to-end) and msprof (micro) **trends** align; absolute µs differ.

## Good Triton Patterns (Generic)

```python
# A — Aligned slab + consistent index stride
IN_W: tl.constexpr  # host: align_up(logical_w, 8)
iy = in_y0 + tl.arange(0, IN_H)[:, None]
ix = in_x0 + tl.arange(0, IN_W)[None, :]
tile = tl.load(gm_ptr + ..., mask=mask_in, other=0.0)
flat = tl.reshape(tile, (IN_H * IN_W,))
off = (y - in_y0) * IN_W + (x - in_x0)

# B — 2D grid + warps
num_warps = 16 if BLOCK_H * BLOCK_W >= 128 else 8

# C — Stencil from flat UB only (not GM)
val = tl.load(flat + off)  # → UB vgather in IR

# D — Host stride helper
def aligned_stride_w(w, align=8):
    return (w + align - 1) // align * align
```

## Anti-Patterns (Generic)

- Index stride uses **logical width** while tile storage uses **different padded width** → densify copy.
- Slab size from **fixed multiplier of block** unrelated to scale window → GM bloat.
- Default **small tile** without scoring → excess programs.
- Tuning **VGATHER** when msprof shows ~1% and ST/copy dominate.

## Generic Workflow

1. GM staging: one slab load / program; kill GM `ExtractedLoadOrStore` on stencil body.
2. Align `IN_W`; gather row step == `IN_W`.
3. IR check: UB densify `hivm.hir.copy` → **0**.
4. Tighten slab from `{block, scale, radius}`.
5. Launch picker: tile vs grid vs slab; set warps.
6. If memory IR clean but slow: scalar precompute (periodic weights).
7. msprof: cycles ↓, ST ↓; ignore VGATHER % unless it rises.

## What To Verify (Generic)

- GM `memref.copy` yes; UB densify copy **no**.
- Gather row constant == `IN_W`; flat size == `IN_H * IN_W`.
- Program count vs gather width tradeoff documented in IR.
- msprof cycles/call_count down; VGATHER may stay ~1%.
- Reference match on corners, up/down, all dispatched branches.

## Related Patterns

- [discrete_memory_access.md](discrete_memory_access.md)
- [padded_row_col_copy.md](padded_row_col_copy.md)
- [tiling.md](tiling.md)
- [scalar-latency-traps.md](scalar-latency-traps.md)

## Expected Performance Impact (Generic)

| Change | Signal |
|--------|--------|
| GM staging | IR: no DiscreteMemAccess on stencil |
| Remove densify | IR: internal `hivm.hir.copy` → 0; ST leaves Top |
| Tight slab | IR: flat elems/program ↓10×+; msprof ST ↓ |
| Larger tile | Grid ↓; call_count ↓; gather width ↑ |
| Scalar precompute | IR: shorter select/polynomial chains |
| VGATHER-only tune | Flat when share ~1% |

---

## Mode Notes (Specific → Generalization)

The following summarizes **mode-specific** choices observed during one multi-mode resize implementation. Each subsection states what differs, then whether it **generalizes**.

### Nearest

**Mode-specific behavior**

- One input sample per output; stencil radius = 0.
- Coordinate map is integer indexing (floor / round per convention).
- Often no separable gather chain; may use direct row load or minimal slab.

**Optimizations applied**

- 2D output tile instead of 1D flattened program axis.
- Optional small slab when a block spans multiple input rows (non-trivial scale).

**Generalization**

| Item | Promote to generic? |
|------|---------------------|
| 2D tile + launch scoring | **Yes** — applies to all modes |
| Full GM→UB slab + flatten gather | **Partial** — only if block reads ≥2 non-contiguous rows; nearest often skips |
| UB stride align / densify elimination | **Yes** when slab path is used |
| Tight slab formula | **Yes** when slab used |
| Weight precompute | **No** — N/A |

---

### Bilinear

**Mode-specific behavior**

- 2×2 stencil (4 samples); separable 2-tap weights.
- Lower gather count per program than wider stencils.

**Optimizations applied**

- Same slab pipeline as wider stencils but smaller `IN_H×IN_W`.
- ①②④⑤ generic stack applies when bilinear is hot path.

**Generalization**

| Item | Promote to generic? |
|------|---------------------|
| GM→UB slab once / program | **Yes** — canonical 2D stencil pattern |
| Aligned IN_W, no densify | **Yes** |
| Launch picker | **Yes** |
| Integer-ratio dispatch | **Optional** — only if scale is exact N× and map simplifies |
| Phase weight tables | **Rare** — only 2 taps; often cheaper to compute inline |

---

### Bicubic (generic coordinate path)

**Mode-specific behavior**

- 4×4 stencil; separable 4-tap Keys weights.
- Per-pixel float coordinate → floor → polynomial (8 evals/pixel before precompute).
- ~16 UB gathers / program (separable horizontal + vertical passes).

**Optimizations applied**

- Slab + flat UB + vgather (eliminate GM discrete loads).
- ①②③④⑤ full stack; evidenced IR: densify 39→0, 8×8→16×16 programs, msprof cycles ~÷2.

**Generalization**

| Item | Promote to generic? |
|------|---------------------|
| ①–⑤ entire memory/launch stack | **Yes** — this mode **proved** the generic directions |
| Separable gather structure | **Template** for any rank-4 separable filter on UB |
| msprof “VGATHER ~1%, optimize ST/slab” | **Yes** — diagnostic rule for all stencil-resize |
| Keys polynomial on device | **Mode-specific** unless other modes share same kernel body |

---

### Bicubic (exact integer scale, corners not aligned)

**Mode-specific behavior**

- Dispatch when `in/out` or `out/in` is exact integer `N≥2` (equal H/W ratio).
- Downsample: fixed fractional offset → **constexpr taps**.
- Upsample: phase `index % N` → **small `[4,N]` weight table**.
- Slab laws: down ≈ `N×block+margin`; up ≈ `(block+N−1)//N+margin` (not loose `block×4+3`).

**Optimizations applied**

- Dedicated kernel with `RATIO_N`, `IS_UP` constexpr.
- Host phase table exact size N (no pad-to-max + deep `tl.where`).
- Tight up-slab: e.g. flat 2520→192 (~13×/program), total GM→UB ~85× with program halving.

**Generalization**

| Item | Promote to generic? |
|------|---------------------|
| Host dispatch ladder (generic → specialized) | **Yes** — pattern for any resize with detectable exact ratio |
| Tight scale-aware slab (③) | **Yes** — **promoted**; loose formula was bicubic-up-specific mistake |
| Phase table without max padding | **Yes** for any periodic weight index (see scalar-latency-traps) |
| Fixed downsample taps | **Bicubic + this coordinate law only** |
| `N×block` slab for down | **Generalizes** to any N× downsample stencil |

---

### Bicubic (exact integer scale, corners aligned)

**Mode-specific behavior**

- Scale is `(H_in−1)/(H_out−1)`, not `1/N`.
- Rational integer coords; phase period `(H_out−1)/gcd(H_in−1,H_out−1)` (e.g. 341 or 85, not 4).
- Still on generic kernel today: float floor + Keys eval.

**Optimizations applied (planned / partial)**

- Same ①–⑤ memory IR as generic path.
- Planned: rational `//`/`%` coords + compact phase table — **memory unchanged**, scalar path only.

**Generalization**

| Item | Promote to generic? |
|------|---------------------|
| ①–⑤ on generic path | **Yes** |
| Rational phase table sizing | **Yes** for **any** align-corners resize with integer dimension ratio |
| Keys on device | **Mode-specific** until table replaces it |
| Period 341 vs 4 | **Not generalizable** as fixed constant — host must compute period per shape |

---

### Area (integer downsample factor)

**Mode-specific behavior**

- Output = average over `factor×factor` input cell; no interpolating kernel.
- Sum/count in UB over one input tile per output block.

**Optimizations applied**

- Dedicated integer-factor kernel (equivalent to box filter).
- One GM→UB tile per output block; accumulate in UB; single write.

**Generalization**

| Item | Promote to generic? |
|------|---------------------|
| GM→UB tile + on-chip reduce | **Yes** — template for **pooling / reduce-over-window** ops |
| ①② aligned stride | **Yes** when tile is flattened or multi-row |
| Launch picker | **Yes** |
| Integer-factor dispatch | **Specific** to area / pool-like modes |
| Separable gather | **No** — replace with sum loop or vector reduce |

---

## Cross-Mode Promotion Summary

| Optimization | Nearest | Bilinear | Bicubic | Area | **Universal?** |
|--------------|---------|----------|---------|------|----------------|
| 2D tile + launch scoring | ✓ | ✓ | ✓ | ✓ | **Yes** |
| GM→UB slab staging | partial | ✓ | ✓ | ✓ | **Yes** (stencil radius ≥ 2) |
| UB stride align (①) | if slab | ✓ | ✓ | ✓ | **Yes** |
| Eliminate densify (②) | if slab | ✓ | ✓ | ✓ | **Yes** |
| Tight slab bounds (③) | if slab | ✓ | ✓ | ✓ | **Yes** |
| Large tile / fewer programs (④) | ✓ | ✓ | ✓ | ✓ | **Yes** |
| Warp heuristic (⑤) | ✓ | ✓ | ✓ | ✓ | **Yes** |
| Periodic weight precompute (⑥) | — | rare | ✓ | — | **When period ≪ output size** |
| Integer-ratio dispatch | — | optional | ✓ | ✓ | **When shape predicate provable** |
| msprof: fix ST not VGATHER | ✓ | ✓ | ✓ | ✓ | **Yes** |

**Bottom line:** Items ①–⑤ and launch/msprof reading are **fully generic** for stencil-like resize; they were **validated most strongly** on wide-stencil (4-tap separable) paths but apply whenever the IR shows GM discrete loads or UB densify. Mode sections add **dispatch predicates**, **slab margin formulas**, and **weight periodicity** — promote formulas to generic when the predicate is shape-derived, not mode-named.
