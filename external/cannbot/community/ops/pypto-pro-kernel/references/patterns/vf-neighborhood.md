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

<!-- kernel-query: {"id":"access-set-layout-cost","aliases":["layout from actual access sets","layout conversion amortization","真实访问集合与布局","重排成本摊销"],"when":"Choosing a neighborhood load region, merging axes, transposing blocks or preparing a contiguous consumer layout.","missing_facts":["Derive actual strides and dependency offsets, touched physical blocks, live metadata, preparation/restoration traffic and complete-call work waves."],"example":"algorithms/simt_tiled_transpose/kernel.py","term_groups":[["layout","布局","重排"],["access set","amortization","amortize","访问集合","摊销"]]} -->
## Compare layouts from actual access sets

Derive every candidate's source addresses before estimating bytes. In `[T,B,D]`, flattening the first two axes gives row `r=t*B+b`; one time-step lag is B rows, so a three-point causal window reaches `r-2*B`, not `r-2`. Preserve the batch coordinate and temporal validity at the boundary. Stride/dilation can likewise create holes in a window: its bounding span determines capacity, while the actual access set, repeated loads and touched physical blocks determine different parts of the cost.

Merge adjacent source axes only when their actual strides establish contiguity, and prove that the destination permutation preserves the merged unit. Some permutations only rearrange whole contiguous blocks; others require an internal transpose. Equal logical byte counts can touch different numbers of UB blocks or GM transactions. [Strided transpose](../../examples/algorithms/strided_transpose/) and [shared-UB transpose](../../examples/algorithms/simt_tiled_transpose/) provide distinct native controls; their different contracts and layouts must be aligned before comparing them.

When changing consumer layout, include `T_prepare + T_consume + T_restore` for the public contract, plus launches, scratch and handovers. A channels-last consumer must repay every required input/output reorder. A precomputed coordinate/weight table occupies space at the same time as payload buffers and can reduce batch width or increase work waves. Compare those effects together with [plan amortization](stable-placement.md#separate-reusable-metadata-from-payload-execution), retaining low-reuse and capacity-boundary controls. Logical bytes or a fitting allocation alone do not establish a gain.

## Verification gate

Start with increasing and decreasing inputs, which select opposite endpoints. Add all-negative/extreme values, impulses exactly at window/block edges, and boundaries across registers and transfer blocks. Within the supported domain, cover singleton axes, empty offset sets or zero radius, neighborhoods wider than the axis, small final output blocks, workers and repeated calls after refilling outputs. The CPU reference intersects mathematical neighborhoods directly rather than copying the kernel's load origin or register-index formulas.

Runnable composition: [sliding_max](../../examples/algorithms/sliding_max/). It separates output blocks from clipped input intervals, reuses one neighborhood transfer across register segments, and keeps source/output buffers separate to preserve unread neighbors. Its block width is a measured example choice, not a universal optimum.

For a direct-GM SIMT correctness anchor, [grouped_conv1d](../../examples/algorithms/grouped_conv1d/) combines stride/dilation coordinates with separate input/output channel groups. Its independent tests preserve group-isolation input invariants, cover unequal channel widths and update inputs at reused addresses. The scalar reduction mapping does not establish a performance advantage over a blocked VF/Cube implementation.
