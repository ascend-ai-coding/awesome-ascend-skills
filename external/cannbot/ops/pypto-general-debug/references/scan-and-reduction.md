# Large-axis scan / cumulative & UB-exceeding reduction (DEBUG_GUIDEBOOK.md §9.21)

*Block-the-axis + carry pattern. Open this when a native scan/cumulative op (or a
large-axis reduction) blows the UB budget, or when you are tempted to scan an
axis element-by-element.*

**Scope:** native scan/cumulative (`pypto.cumsum`, `pypto.cumprod`, `cumsum_reverse`,
`masked_cumsum`, …) along a long axis, or a native reduction whose single-op UB
exceeds the budget. For ops that already fit UB, this machinery is not needed.

## Symptoms

Compile time — UB allocation fails:

```
ErrCode F40005! TENSOR_MEMORY_ALLOCATION. Alloc tensor size [256000] exceeds MEM_UB size [196608]!
```

Runtime — host crash or hang after "fixing" the above with an element-by-element loop:

```
SLAB_ADD_CACHE_FAILED  →  segfault / process killed
```

or the run simply never finishes (4 h benchmark timeout, missing artifacts).

## Root cause

A native scan op's internal workspace scales with the axis length. `pypto.cumsum`
uses ~64 B per dim-axis element (≈16× FP32, a parallel tree-scan keeping
intermediate levels), so a 4000-long axis needs ~256 KB > the 192 KB UB budget
(hard limit ≈ 3072 FP32 elements). Applying the op to the whole axis at once
therefore fails at compile.

The naive escape — looping the axis one element at a time (`pypto.view([1, 1])`
per position) — is numerically correct but spawns `axis_len × batch` tasks
(e.g. 4000 × 128 ≈ 512K), which exhausts host-side scheduling resources
(`SLAB_ADD_CACHE_FAILED` + segfault) or times out.

## Fix: block the axis + propagate a carry

Split the large axis into blocks of length `T` chosen so the op's **per-block**
UB fits the budget (cumsum: `T=1000` → ~64 KB < 192 KB). Loop over the *few*
blocks, `view` a real `[.., T]` block, apply the native op on the block, and
carry the running accumulator across blocks:

```python
# ✅ block the scan axis; native op per block; carry across blocks
def _op_kernel_impl(x, out):                 # x, out: [B, D]  (D large, e.g. 4000)
    B = x.shape[0]                           # SymbolicScalar (dynamic batch)
    D = 4000; T = 1000; NB = D // T          # T fits per-block UB; NB small (4)
    for b in pypto.loop(B, name="batch"):
        carry = pypto.full([1, 1], 0.0, pypto.DT_FP32)            # running accumulator
        # block loop carries a cross-block dependency -> submit_before_loop=True (only NB iters)
        for t in pypto.loop(NB, name="block", unroll_list=[1], submit_before_loop=True):
            blk  = pypto.view(x, [1, T], [b, t * T])              # real [1, T] block (NOT [1,1])
            scan = pypto.cumsum(blk, dim=1)                       # native op on the block
            scan[:] = scan + carry                                # fold in running carry ([1,1]->[1,T])
            pypto.assemble(scan, [b, t * T], out)
            carry[:] = pypto.view(scan, [1, 1], [0, T - 1])       # block tail = new running total
```

Choosing `T`: pick the largest `T` that keeps per-block UB within budget and
divides the axis length (cumsum FP32 → `T ≲ 3000`, e.g. 1000 / 800 / 500).

## block length ≠ vec tile

`T` is a **view** shape, not the vec tile. The kernel above passes with a small
vec tile (e.g. `set_vec_tile_shapes(16, 16)`) left unchanged — correctness comes
from the blocking structure, not the tile value. Do not couple the tile size to
`T`, and do not change the normal vec-tile rules ([16, 64] per axis, rank match,
single-op UB fit).

## Do NOT

- Apply the native op to the whole axis at once → UB OOM (F40005).
- Loop the axis element-by-element (`pypto.view([1, 1])`) → task explosion /
  host crash / timeout.

## See also

- Implementation skeleton (canonical): `pypto-op-develop/references/pypto-kernel-design-format.md` §9c.
- Design-level principle: `pypto-op-design/SKILL.md` §3.0 (4th data-flow principle).
