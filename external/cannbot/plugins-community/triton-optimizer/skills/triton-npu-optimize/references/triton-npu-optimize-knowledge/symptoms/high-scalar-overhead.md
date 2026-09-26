# high-scalar-overhead

## Summary

The round spends too much time on per-program fixed work, scalar control flow, or bookkeeping relative to the amount of vector or cube work each program performs.

## Evidence To Confirm

- Many tiny launches or very small per-program work dominate the profile.
- Timeline or summary views suggest under-filled vector execution.
- Code-mapping outputs show `SCALAR` or control-heavy execution in regions that should mostly feed Vector or MTE work.
- Code inspection shows one-row-per-program structure, heavy scalar masking, explicit compare-heavy control logic, or a **flat 1D** pad/copy kernel with expensive coordinate decode on the last axis.

## Candidate Pattern Directions

- `program-multiple-rows`
- `padded_row_col_copy`
- `vec-cmp`
- `classic-matmul`
- `simt-clip-window-closed-reduction`

## Common Non-Matches

- Scalar-looking code at the edges does not matter if the hot loop is actually cube-bound.
- Small-shape kernels can show scalar overhead even when the better answer is dispatch or specialization rather than a local rewrite.
