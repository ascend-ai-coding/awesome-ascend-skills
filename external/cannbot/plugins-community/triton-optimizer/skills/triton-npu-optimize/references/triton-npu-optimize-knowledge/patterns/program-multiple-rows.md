# Program Multiple Rows Pattern

## Summary

Amortize per-program fixed costs and improve vector-friendly batching for **row-reduction or row-wise fused kernels** by mapping **multiple rows** to one Triton `program_id` via `BLOCK_M > 1`, instead of one row per program.

## Use When

- The kernel is **naturally row-wise**: each output row depends mainly on one row of input (e.g. row-wise LogSumExp, row norms, row softmax statistics).
- Profiling or timeline views suggest **high scalar/control overhead**, **under-filled vector work per program**, or **many tiny programs** relative to problem size `B` (batch / number of rows).
- The row-wise math already uses **tile loops along `N`** (`BLOCK_N`); increasing **`BLOCK_M`** does not force an extra full pass over global memory if you keep a **single streaming pass** over `N` per program.

## Signals

### Code

- `program_id(0)` indexes **rows 1:1** (`pid_m` is the row index), and the inner loop only tiles **`N`**.
- Scalar helpers (`program_id`, pointer arithmetic per row) run once **per row**; vector units see **narrow** tensors (e.g. `(1, BLOCK_N)` loads).

### Profile

- **`aiv_scalar_ratio`** or scalar-related time is **disproportionately high** compared to useful vector math, for workloads where `B` is large enough that vector throughput should dominate.
- **`op_statistic`** (per-kernel): **Avg** latency improves when the same logical work uses **fewer launches** (compare with care: **Count** and input shapes must be comparable across runs).
- If **`aiv_mte2_ratio`** is **not** the sole dominant bucket, pure “double-buffer the loads” may be the wrong first lever; **program batching** can still help by making each program’s inner loop **wider** along rows.
- Frequent **barrier / wait** patterns tied to **many short programs** or **thin** vector blocks.
- **Note:** High **`BAR`** cycle counts alone are **not** a success metric; correlate with **wall time**, **op_statistic Avg**, and correctness.

## Implementation sketch (Triton)

1. Add **`BLOCK_M: tl.constexpr`** and treat **`pid_m`** as a **block of rows**:
   - `rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)`
   - `row_mask = rows < B`
2. Build **2D tiles**: `vals` shape **`(BLOCK_M, BLOCK_N)`**; row-wise reductions use **`axis=1`** (max, sum) so running state **`m`, `s`** are **`(BLOCK_M,)`**.
3. Adjust **grid**: `grid = (triton.cdiv(B, BLOCK_M),)`.
4. **Stores**: one scalar per row → `tl.store(y_ptr + rows * stride_ym, x, mask=row_mask)`.
5. **Tune `BLOCK_M`**: start with a small power of two (e.g. 4–16). Too large **`BLOCK_M * BLOCK_N`** may hurt UB/register pressure; validate with benchmark + profile.

### Example reference (row-wise LSE + fused activations)

See the operator workspace pattern: row pointers `row_ptrs = x_ptr + rows[:, None] * stride_xm`, masked load to `(BLOCK_M, BLOCK_N)`, streaming LSE with `tl.max` / `tl.sum` on `axis=1`, then fused elementwise ops and masked store.

## Avoid When

- **Second full pass** over `x` for the same row (e.g. two-pass LSE) usually **increases global reads**; msprof often shows **more MTE / wait** unless the algorithm truly requires it. Prefer **single-pass streaming LSE** when numerically stable.
- **Ping-pong / multibuffer** without evidence of **MTE–vector overlap** can add **sync and UB** cost; treat as a **separate hypothesis** to validate.
- Do not conclude from **one** metric (e.g. `BAR` cycles) without **end-to-end** timing and comparable workload.

## What To Verify After Applying

1. **Correctness**: same dtypes, masks for `rows >= B`, and numerically stable reductions (e.g. LSE max-shift) unchanged in meaning.
2. **Benchmark**: compare **mean / geomean** with the same harness; use project **`compare-perf`** flow when available—avoid hand-computed speedups from raw logs.
3. **Profiler**: compare **`op_statistic` Avg** for the same op; note **Count** and tensor shapes. Optionally re-check **`op_summary`** vector/scalar/MTE mix.
4. **Sanity**: if `B` is tiny, **`BLOCK_M > 1`** may help little; if `B` is huge, launching **`cdiv(B, BLOCK_M)`** programs should visibly reduce launch/program overhead.

## Related Patterns

- Complements **`parallel`**: `BLOCK_M` widens work **within** one program; `tl.parallel` splits **independent** subgraphs across vector cores—orthogonal when dependencies allow.
- Differs from **`software-pipeline`**: multibuffer targets **load/compute overlap** along a tile loop; **`program-multiple-rows`** targets **program granularity** and **row batching**.
