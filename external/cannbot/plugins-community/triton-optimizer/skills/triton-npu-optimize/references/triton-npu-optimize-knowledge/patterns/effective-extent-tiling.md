# Effective-Extent Tiling

## Summary

Choose tile widths from the live logical extent on each axis instead of a legacy maximum or blanket power-of-two rule, so masked lanes do not dominate loop trip counts, transfer work, or vector-path work.

## Use When

- A **`BLOCK_*`** tile is much larger than the **valid extent** protected by a mask, so the kernel does visibly more padded lane work than useful work.
- The hot path is either **indexed / masked access** or a **copy-like contiguous axis** whose width does not participate in **`tl.dot`**, cube alignment, or reduction-tree structure.
- Profiling or IR suggests execution cost scales with the **tile width** more than with the **live element count**.
- The host already has shape information that could choose a smaller tile or a different launch branch.

## Signals

### Code

- **`BLOCK_SIZE`**, **`BLOCK_M`**, **`BLOCK_N`**, or **`BLOCK_H`** is chosen from a cap or unconditional **`next_power_of_2()`** while the mask protects a much smaller live extent.
- A copy-like path uses **`tl.max_contiguous(ptr, BLOCK)`** or similar contiguous-lane hints together with **masked `tl.load` / `tl.store`** on the final or inner block.
- Structured and flat launches reuse one tile heuristic even though their effective extents differ.
- Indexed globals use **`tl.load` / `tl.store` / `tl.atomic_*`** with per-lane masks, and the tile looks larger than the number of useful lanes.

### Profile

- **LD/ST counts**, scalar loop cost, or transfer-side metrics track the **full tile** rather than the **valid lane count**.
- Switching from a padded tile to a smaller exact-width tile lowers **MTE / DMA** time on the same logical workload.
- Small or narrow shapes pay unexpectedly high per-kernel latency.

### IR

- **`scf.for`** bounds stay tied to a tile constant even when the logical extent is smaller.
- Masked semantics show up as **`arith.select`** or zero-filled fallback values across the full tile.
- Indexed paths still show **`DiscreteMemAccess`** or extracted load/store structure after lowering.

## Strategy

1. Identify the **live extent** on each tiled axis, not just the configured cap.
2. Pick tile widths from a **small measured candidate set** and compare padded-lane waste against grid growth.
3. Recompute **grid** and any program-budget logic together with the tile; do not tune tile width in isolation.
4. For **copy-only contiguous axes**, treat **exact extent** or **`min(cap, extent)`** as the default candidate set, and treat **`next_power_of_2()`** as a measured alternative rather than the baseline.
5. If one tile policy is only correct or profitable for certain shapes, encode that split at the **host dispatch** layer.
6. If a small set of extents, such as `K == 64` or `K == 128`, dominates production shapes, consider dispatching to extent-specialized kernels so inactive padded branches disappear from the hot path.

## Avoid When

- The axis participates in **`tl.dot`**, cube alignment, fixed reduction-tree structure, or another backend-sensitive alignment contract.
- The real bottleneck is **atomic contention**, unrelated host overhead, or another phase outside the oversized-tile region.
- A smaller tile would explode program count past a launch or synchronization knee.
- The target rejects the needed non-power-of-two tile shapes, or the resulting specialization count is unacceptable.
- Runtime extents vary too widely for shape-specialized dispatch to stay maintainable.

## What To Verify After Applying

- **IR**: loop bounds or static tile sizes shrink with the live extent where expected.
- **Profile**: LD/ST, scalar, or transfer-heavy metrics move with the new tile instead of the old padded width.
- **Correctness**: tail blocks, masks, dtypes, and all dispatch branches still behave identically.
- **Compilation / caching**: the target accepts the chosen tile shapes and JIT specialization does not become a new problem.
- **Dispatch coverage**: specialized extent branches and the generic fallback are all validated.

## Related Patterns

- [discrete_memory_access.md](discrete_memory_access.md)
- [tiling.md](tiling.md)

## Example: Copy-Only Tail Width

For a copy-like hidden-dimension path, suppose the live width is **80** elements and the host picks:

- **padded tile**: `BLOCK_H = next_power_of_2(80) = 128`
- **exact tile**: `BLOCK_H = 80`

If the kernel executes one final masked block with contiguous load/store organization tied to **`BLOCK_H`**, the padded version keeps **48 masked lanes** on that block. On Ascend, those masked lanes can still contribute to transfer and vector-path work even though the mask preserves correctness. In this regime, switching to the exact-width tile often lowers MTE-heavy time because the backend does less padded lane movement.

The same idea also applies to indexed paths: even if **`DiscreteMemAccess`** remains in IR, reducing the tile to the live extent can still cut wasted masked iterations and padded memory-op counts.
