# weak-pipeline-overlap

## Summary

The round appears to leave memory movement and compute insufficiently overlapped, so the kernel pays avoidable wait between loading data and using it.

## Evidence To Confirm

- Timeline or wait-oriented profile summaries point to load-then-compute serialization.
- IR summaries show sync-heavy or transfer-dense stages near the hot loop.
- Code structure already looks tiled, but each loop iteration still loads, computes, and stores in a mostly serial rhythm.

## Candidate Pattern Directions

- `software-pipeline`
- `reorder-load`
- `classic-matmul`

## Common Non-Matches

- Weak overlap is not a license to add pipeline machinery before basic kernel structure is fixed.
- If the kernel is still a manual reduction or scalar-heavy matmul shape, first normalize structure with a more foundational rewrite.
