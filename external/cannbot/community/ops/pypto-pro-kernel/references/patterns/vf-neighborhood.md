# VF neighborhoods: loading a region does not redefine output coordinates

Use for sliding windows, local filters, stencils and other operations whose outputs depend on relative positions. Define the mathematical neighborhood and boundary policy before blocking, transfers and gathers; do not derive the window center from a clipped load origin.

## State the three coordinate systems

Let an output block own global interval `[o, o+B)`, where B is its actual output count; neighborhood offsets range from `d_min` to `d_max`, and the input axis has length N.

| Object | Definition | Purpose |
| --- | --- | --- |
| Global output coordinate | `t_out = o + lane` | Defines the neighborhood center/anchor; clipping a load cannot move it |
| Required source interval | `[o+d_min, o+B+d_max)` | Union all output dependencies first, then intersect with `[0,N)` |
| Actual loaded interval | `[l,u) = [max(0,o+d_min), min(N,o+B+d_max))` | GM transfer origin and valid length; handle an empty intersection by contract without out-of-bounds transfers |
| Global coordinate of one dependency | `t = t_out + d` | Determines membership in the input domain and this output's neighborhood |
| In-tile element index | `i = t - l` | Converts coordinates only; it does not redefine the neighborhood |

After clipping the left endpoint to zero, taking the original full window length from zero shifts the window right. The right boundary has the same issue. Derive both unclipped endpoints, then intersect with the input domain. Load clipping and output ownership are separate decisions. Physical tile capacity, row pitch and byte addresses still follow [data movement](../api/datamove.md) and the [memory plan](../api/memory-allocation.md).

## Align addresses, participation and values

Logical indices across a boundary can be negative. Preserve signed semantics or retain a validity test before unsigned subtraction; underflow is not a valid large source index. Decide participation first, then constrain actual accesses to allocated, initialized storage. A final store mask cannot repair an earlier illegal read or contaminated result.

A loop counter only enumerates work; derive logical offsets separately. VF `pl.range` has restrictions on nonnegative ranges and increment overflow, so a negative mathematical interval cannot be copied directly. See [for/range constraints](../api/constraints.md#for-must-use-plrange) for version scope and rewrites.

`vf.select(a,b,p)` selects a when p is true and b otherwise. Write truth-table rows for valid and out-of-bounds predicates and check that safe-index selection and value selection agree. Valid sources use the original index/value; invalid sources use a safe placeholder address and the contracted identity or retained state. Do not fix the value mask while sending indices down the opposite branch. See [coupled state and boundaries](../api/vf.md#grouped-state-and-boundaries-in-lane-exchange).

Memory and register gathers have different access domains. Memory gathers need valid in-tile addresses; register-index wrapping does not authorize logical windows to cross boundaries. Across multiple registers, map both the register number and its lane. When the full UB block already exists, also evaluate direct tile gather instead of simulating it with register selection; see the [two gather forms](../api/vf.md#31-gather-has-two-forms-out-of-bounds-behavior-is-defined-for-only-one).

The forms also repeat different work: reading each neighbor from UB and reusing already-loaded registers cannot be compared by Python line count. A shorter tile-gather expression is not necessarily faster. Measure that choice with identical transfer blocks, workers and boundary semantics before separately changing transfer-block size.

Replicated boundaries, zero padding, extreme-value padding and ignoring invalid values need not be equivalent. Choose by the operation and contract, not because index clamping is easy. Even finite-input max/min must cover extreme values; an arbitrary finite sentinel is not automatically an identity.

## Blocking and work distribution

- A transfer block is not one register. Load a larger output block's neighborhood union once, then let one VF call consume multiple register blocks. Capacity includes live inputs, outputs, metadata, temporaries and slots. Use [transfer granularity and state lifetime](vf-outer-reduce.md#choose-transfer-and-register-blocks-separately) to compare transfer counts, repeated halo reads and VF-call boundary costs.
- Output blocks may share read-only neighborhoods without sharing a worker. For independent outputs, assign disjoint output blocks and allow overlapping reads. With few rows, compare row ownership with further output-axis blocking while preserving one writer per output.
- If a block's global range proves every neighborhood valid, boundary checks can move to block-level or specialized edge paths instead of rebuilding constant boundary state for every neighbor/lane. Preserve the full dependency-range proof and measure complete native calls at fixed workers.

## Verification gate

Start with increasing and decreasing inputs, which select opposite endpoints. Add all-negative/extreme values, impulses exactly at window/block edges, and boundaries across registers and transfer blocks. Within the supported domain, cover singleton axes, empty offset sets or zero radius, neighborhoods wider than the axis, small final output blocks, workers and repeated calls after refilling outputs. The CPU reference intersects mathematical neighborhoods directly rather than copying the kernel's load origin or register-index formulas.

Runnable composition: [sliding_max](../../examples/algorithms/sliding_max/). It separates output blocks from clipped input intervals, reuses one neighborhood transfer across register segments, and keeps source/output buffers separate to preserve unread neighbors. Its block width is a measured example choice, not a universal optimum.
