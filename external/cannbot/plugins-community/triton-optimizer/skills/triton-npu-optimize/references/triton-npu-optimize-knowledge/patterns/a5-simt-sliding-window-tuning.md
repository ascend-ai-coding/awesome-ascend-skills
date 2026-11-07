---
id: a5-simt-sliding-window-tuning
priority: normal
---

# A5 SIMT Fixed-Window Reduction — Feature-Driven Tuning

## Summary

Tuning methodology for kernels that map **each output point** to a **fixed-size input window** over an **affine N-D layout** (e.g. NCDHW/NCHW), then **reduce** (sum, max, etc.) under **A5 `force_simt_only=True`**. Teaches how to **derive** dispatch, inner paths, and launch params from **structural features** — not from a specific op name. Pool-style ops are one common instance; the same signals apply to any op matching the pattern signature below.

## Pattern Signature (when this card applies)

| Structural feature | Typical signal in code |
|--------------------|-------------------------|
| Fixed window size | `KERNEL_D/H/W` constexpr loops |
| Affine output→input map | `start = out * STRIDE - PAD`; dilation optional |
| Per-output reduction | accumulate / compare / optional index write |
| Discrete loads | masked or unmasked `tl.load` per window tap |
| Layout | innermost spatial dim contiguous (often W) |

**Not in scope:** Cube/matmul, irregular gather indices, value-dependent windows.

## Use When

- Kernel matches the pattern signature above.
- A5 confirmed; hot path is scalar/index-heavy (`a5-force-simt-only-discrete-access`).
- Multi-shape harness available; accept/reject by **geomean**.

## Avoid When

- Not A5 SIMT, or compute-bound dense vector math dominates.
- Mixing **SIMT** and **non-SIMT scalar row** kernels in the **same module/process**.

---

## 1. Derive per-case features (host)

| Feature | How to compute |
|---------|----------------|
| Output spatial sizes | from stride, pad, ceil, dilation rules |
| `n_out` | product of output tensor shape |
| `window_vol` | product of fixed kernel sizes |
| `eff_span_*` | `dilation * (k - 1) + 1` per axis |
| `row_parallelism` | outputs sharing outer dims (for 2D grid) |
| `pad_any`, `ceil`, `dilated`, `half` | semantic / dtype flags |
| `window_fully_inside` | last window fits in input (use eff_span) |
| `reduce_kind` | cheap acc vs heavy compare+where+extra state |
| `norm_semantics` | e.g. include-pad in normalizer vs clipped volume only |

**Stack-pressure proxy:**

```text
stack_pressure ≈ window_vol × (mask_branches + extra_state_per_tap)
```

High when: boundary path, clip nest, include-pad normalizer, compare+index chain, half dtype, 2D vector tile.

Derive routing from features; do not copy thresholds from another op without re-benchmarking.

---

## 2. Tuning workflow (one variable class per round)

```text
P0 Launch      → force_simt_only + single launch
P1 Dispatch    → flat 1D vs row×col 2D grid
P2 Inner path  → full interior vs boundary (mask / clip / include-pad norm)
P3 Geometry    → BLOCK, num_warps, static unroll depth
P4 Validation  → fresh process, full harness geomean
```

| Stage | Pass evidence | Revert if |
|-------|---------------|-----------|
| P0 | geomean up vs multi-launch | correctness fail |
| P1 | target shape bucket improves | small shapes regress |
| P2 | semantics match reference | boundary half-dtype regress |
| P3 | 507035 cleared | geomean drops |
| P4 | stable across two runs | single-case ×10 spike (co-JIT) |

**Rules:** architecture before micro-params; **one launch kernel** + `constexpr` branches; inlined `@triton.jit` callees OK — **multiple launch entry points** not OK.

---

## 3. P1 — flat vs row×col (derive)

**Row×col helps when:** wide inner spatial dim + large `window_vol` + enough row parallelism + **cheap per-tap reduce**.

**Flat helps when:** narrow inner dim, low outer parallelism, or **heavy per-tap state** (compare + index + where chains).

**Routing:** start flat-only; enable row×col for a **named feature bucket**; bisect thresholds on harness geomean.

---

## 4. P2 — interior vs boundary inner path

### Interior fast path (`USE_FULL_INTERIOR` / host equivalent)

```text
interior iff no pad AND no ceil edge AND unit dilation
           AND window_fully_inside (eff_span)
```

Kernel: fixed `window_vol` scan, no per-tap masks.

### Boundary path selection

| Normalizer / semantics | Preferred inner | See also |
|------------------------|-----------------|----------|
| Include virtual pad in norm volume | full nest + **coord masks** + **one-shot norm** | §4 below |
| Clipped volume only (exclude pad from norm) | **clip-window** + closed-form volume | `simt-clip-window-closed-reduction` |
| Dilation > 1 | coord masks only | no clip shortcut |
| Heavy compare + half + pad | coord masks over clip | stack depth |
| Compare + arg index | masks + fp32 compare + linear index | |

**Include-pad normalizer (once per lane):**

```python
norm_axis = max(min(start + K, in + PAD) - start, 0)
normalizer = product(norm_axes)  # fp32 when dividing
```

**Empty window:** output sentinel (e.g. 0); clamp normalizer to avoid div-by-zero.

Choose mask vs clip by **semantics first**, then A/B on harness if 507035 or geomean regression.

---

## 5. P3 — BLOCK, warps, unroll

- **BLOCK** tiers with `n_out`; cap lower when `stack_pressure` high (often 512).
- **num_warps:** default 8; try 4 when stack_pressure high (boundary + half).
- **Row×col W tile:** pow2 BLOCK on Ascend; tier row tile with row_parallelism; cap rows on boundary paths.
- **Static unroll:** innermost kernel width only for small fixed k; never full N-D static unroll at large `window_vol`.
- **fp32** for accumulate/compare; store native dtype.

---

## 6. Reduce-kind matrix (re-derive per op)

| Feature | Cheap reduce (sum/mean) | Heavy reduce (max + index) |
|---------|-------------------------|----------------------------|
| Row×col | often wins on wide W + large vol | often loses |
| Boundary warps | usually 4 when not interior | 4 when half + boundary |
| Inner path | include-pad → mask; closed → clip | mask over clip on half pad |
| Scalar-row fallback in SIMT module | validate per module | high co-codegen risk |

---

## 7. Optimization points (source → evidence → failure)

| Point | Why | Verify | Failure |
|-------|-----|--------|---------|
| Single SIMT launch | launch tax | one kernel / forward | still bound on inner |
| force_simt_only | discrete path on A5 | profile + geomean | no gain |
| Interior fast path | skip masks | unpad cases faster | memory-bound |
| flat vs row×col | match parallelism | bucket geomean | small-shape regress |
| One-shot normalizer | no per-tap count | correct semantics | wrong output |
| mask vs clip | stack vs mask count | A/B boundary half | 507035 / geomean drop |
| Monolithic kernel | avoid co-JIT | stable fresh bench | one-case spike |
| No SIMT + scalar-row mix | codegen clash | never embed/import row | geomean collapse |

---

## 8. Anti-patterns (by trigger feature)

| Trigger | Anti-pattern |
|---------|--------------|
| A5 SIMT + overlapping W loads | inner-W slab gather (`sliding-window-inner-w-slab-gather`) without re-proof |
| Any | global interior/boundary **multi-launch** |
| Small window_vol + low parallelism | forced row×col |
| Any | in-kernel `tl.where(interior, …)` split |
| Any | separate launch kernels per interior/boundary/clip |
| Large window_vol | full N-D static unroll → 507035 |
| SIMT module | scalar-row fallback |
| Bench session | mixed kernel architectures → co-JIT |

---

## 9. Related Patterns

- `a5-force-simt-only-discrete-access` — enable SIMT
- `flat-index-decode-tiling` — outer flat index → layout tile
- `simt-clip-window-closed-reduction` — closed normalizer inner loop
- `sliding-window-inner-w-slab-gather` — inner-dim slab (usually not A5 SIMT)
