# Symptom Index

Use this file after structured profile or IR evidence already exists.

Read this generated index first. Then read only the one or two most relevant detailed symptom cards before returning to detailed pattern references.

## Generated Symptom Summaries

### `high-scalar-overhead`

- Summary: The round spends too much time on per-program fixed work, scalar control flow, or bookkeeping relative to the amount of vector or cube work each program performs.
- Source: [high-scalar-overhead.md](symptoms/high-scalar-overhead.md)
- Evidence To Confirm:
  - Many tiny launches or very small per-program work dominate the profile.
  - Timeline or summary views suggest under-filled vector execution.
  - Code-mapping outputs show `SCALAR` or control-heavy execution in regions that should mostly feed Vector or MTE work.
  - Code inspection shows one-row-per-program structure, heavy scalar masking, explicit compare-heavy control logic, or a **flat 1D** pad/copy kernel with expensive coordinate decode on the last axis.
- Candidate Pattern Directions:
  - `program-multiple-rows`
  - `padded_row_col_copy`
  - `vec-cmp`
  - `classic-matmul`
  - `simt-clip-window-closed-reduction`
- Common Non-Matches:
  - Scalar-looking code at the edges does not matter if the hot loop is actually cube-bound.
  - Small-shape kernels can show scalar overhead even when the better answer is dispatch or specialization rather than a local rewrite.

### `high-transfer-pressure`

- Summary: The round looks dominated by data movement, staging cost, or transfer-heavy execution rather than useful cube or vector work.
- Source: [high-transfer-pressure.md](symptoms/high-transfer-pressure.md)
- Evidence To Confirm:
  - Profile summaries show transfer-heavy ratios, low compute saturation, or wait tied to memory movement.
  - Measured movement time is far above a rough moved-bytes / bandwidth lower bound, especially when the working set should fit on chip comfortably.
  - IR summaries show many transfer-dense stages or repeated data reshaping around the hot path.
  - Code structure repeatedly reloads tensors, stages many intermediates, or performs gather/scatter-like movement.
- Candidate Pattern Directions:
  - `tiling`
  - `effective-extent-tiling`
  - `cache-use`
  - `algebraic-optimization`
  - `discrete_memory_access`
  - `gather-load`
  - `sliding-window-inner-w-slab-gather`
  - `slice-coalesce`
  - `slice-intermediate`
- Common Non-Matches:
  - High transfer alone does not prove the best fix is software pipelining.
  - A kernel can be transfer-heavy because its structure is still scalarized or under-batched, not because the transfer order itself is wrong.

### `poor-locality`

- Summary: The kernel revisits data in an order that weakens reuse, causes repeated reloads, or creates cache-bank contention instead of feeding contiguous tiles efficiently.
- Source: [poor-locality.md](symptoms/poor-locality.md)
- Evidence To Confirm:
  - Repeated access to the same regions still yields weak cache behavior or surprising reload pressure.
  - IR or profile evidence suggests the working set could fit better than current performance indicates.
  - Code structure uses traversal order or scatter/gather layout that fights the hardware memory hierarchy.
- Candidate Pattern Directions:
  - `cache-use`
  - `diagonal`
  - `slice-coalesce`
  - `discrete_memory_access`
- Common Non-Matches:
  - Poor locality is not the same as pure UB overflow; if the main issue is footprint size, prefer footprint-reduction patterns first.
  - Not every gather or scatter pattern is fixable through traversal order alone.

### `weak-pipeline-overlap`

- Summary: The round appears to leave memory movement and compute insufficiently overlapped, so the kernel pays avoidable wait between loading data and using it.
- Source: [weak-pipeline-overlap.md](symptoms/weak-pipeline-overlap.md)
- Evidence To Confirm:
  - Timeline or wait-oriented profile summaries point to load-then-compute serialization.
  - IR summaries show sync-heavy or transfer-dense stages near the hot loop.
  - Code structure already looks tiled, but each loop iteration still loads, computes, and stores in a mostly serial rhythm.
- Candidate Pattern Directions:
  - `software-pipeline`
  - `reorder-load`
  - `classic-matmul`
- Common Non-Matches:
  - Weak overlap is not a license to add pipeline machinery before basic kernel structure is fixed.
  - If the kernel is still a manual reduction or scalar-heavy matmul shape, first normalize structure with a more foundational rewrite.
