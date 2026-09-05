# poor-locality

## Summary

The kernel revisits data in an order that weakens reuse, causes repeated reloads, or creates cache-bank contention instead of feeding contiguous tiles efficiently.

## Evidence To Confirm

- Repeated access to the same regions still yields weak cache behavior or surprising reload pressure.
- IR or profile evidence suggests the working set could fit better than current performance indicates.
- Code structure uses traversal order or scatter/gather layout that fights the hardware memory hierarchy.

## Candidate Pattern Directions

- `cache-use`
- `diagonal`
- `slice-coalesce`
- `discrete_memory_access`

## Common Non-Matches

- Poor locality is not the same as pure UB overflow; if the main issue is footprint size, prefer footprint-reduction patterns first.
- Not every gather or scatter pattern is fixable through traversal order alone.
