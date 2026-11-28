# Explicit Tile Placement and Automatic Mutex IDs

Declare native groups with `pl.make_tile_group(type=pl.TileType(...), addrs=..., mutex_ids="auto", depth=N)`. The caller chooses addresses and slot depth; automatic mutex planning assigns synchronization IDs without changing placement. The API is `mutex_ids` in the plural. `pl.mem_space` and `pl.allocate_tile_group` are unavailable in the current development branch.

```python
buffers = pl.make_tile_group(
    type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
    addrs=0x0000,
    mutex_ids="auto",
    depth=2,
)
```

The native implementation is `language/parser/_buffer_parser.py` plus `framework/src/interface/ir/transforms/auto_mutex_id_manager.cpp`. Check the installed Python and native-library identities together; the package version alone does not identify this capability. The framework make_tile_group API (`pypto/docs/zh/api/pro_api/SIMD-API/resource_management/make_tile_group.md`) defines explicit placement.

<!-- kernel-query: {"id":"tile-group-depth","aliases":["pl.make_tile_group","tile-group-depth","make_tile_group depth","auto mutex depth","自动分组 depth"],"symbols":["pl.make_tile_group"],"missing_facts":["Resolve the group's slot count, per-slot placement and synchronization ownership on the installed toolchain."],"rules":["tile-group-depth-required"]} -->
## Scope and Responsibilities

- `addrs=<integer>` lays out slots contiguously using the framework's tile slot size; `addrs=[...]` specifies one address per slot. Every slot still passes alignment and capacity checks; see [memory limits](memory-limits.md).
- `mutex_ids="auto"` requires a positive compile-time `depth` and enabled `auto_mutex`, which is the JIT default. Each group chooses automatic IDs or a manual sequence; they cannot be mixed inside one group.
- Manual groups can coexist with automatic groups. `mutex_ids=None` or an empty sequence creates no automatic synchronization metadata and requires `depth`.
- Automatic planning runs independently for Cube and Vector target parses. There are 32 IDs per core; the planner can reuse IDs when there are more slots, so automatic allocation does not promise one globally unique ID per tile.
- Addresses and physical shapes remain compile-time values. Change the effective runtime extent with `pl.set_validshape`; it does not change the allocation footprint.
- Cross-core handovers still require explicit `fwd_ids` / `bwd_ids`, or GM `pl.bind_crosscore_sync`, together with the pipeline stage structure. Automatic mutex IDs do not allocate event IDs or replace cross-core synchronization; see [cross-core sync](crosscore-sync.md).

The source checker reports unavailable allocation calls before remote parsing. It does not infer a replacement memory layout. Plan every space, slot and lifetime before converting a kernel to explicit addresses.

## Aliasing and Mutual Exclusion

Aliases must preserve the physical allocation and its synchronization ownership. `pl.reinterpret(group, ...)` inherits the original slots, addresses, sizes, depth and IDs. Declare the larger allocation first, then reinterpret it as a smaller view.

When an overlap component contains an automatic group, the planner assigns shared IDs to aliases. A manual tile in that component must pin exactly one ID, and all manual pins in the component must agree. Crossing address ranges where neither contains the other are rejected. Manual-only groups retain their explicit IDs; this is not a general race detector. The source overlap check and pipeline address-reuse validation cover separate, limited cases; see [validation traps](../diagnostics/validation-traps.md).

When configuration predicates choose addresses or physical tile families, verify the predicate that actually ships. For a finite declared configuration set, enumerate it and check each space and slot's alignment, occupied byte extent, capacity and pairwise overlap of simultaneously live buffers. Distinguish intentional aliases with an ordering/lifetime proof from independent storage. For a larger domain, derive the bounds and test their transitions; a representative table alone is not a full-domain argument. Strided storage can occupy more than `numel * element_bytes`; use the [positive-stride physical-span calculation](simt.md#13-scalar-indices-and-custom-physical-strides) for nonempty element-addressed views, accounting separately for empty views, packed formats and transfer padding. Show that the check detects a deliberately overlapping or overflowing configuration before trusting its sweep.

<!-- kernel-query: {"id":"manual-mutex-phase-reuse","aliases":["MutexMismatch","mutex overlap","phase buffer reuse","mutex 重叠","阶段缓冲复用"],"example":"algorithms/attn_epilogue/kernel.py","rules":["overlapping-groups-need-shared-mutex"],"missing_facts":["Resolve physical slot ownership, access lifetimes, generated drains and cross-core handovers before treating an overlap warning as a race."]} -->

### Reuse an address range across differently typed phases

`MutexMismatch` checks overlapping manual declarations with disjoint IDs; it does not analyze when their consumers execute. The source checker also compares slots, with bounded advisories for direct accesses separated by unconditional `bar_all` or confined to distinct core domains. Neither diagnostic proves a race, and a clean result does not prove synchronization. Inspect the generated acquire/release operations, full `PIPE_ALL` drain, prefetch completion and all consumers. Partial or conditional barriers do not establish the same phase boundary; cross-core publication still needs its own event or collective protocol.

For two phases sharing a ring, reserve each role's slot pitch and scale backing for the maximum physical extent across both phases, retain the corresponding slot IDs, and verify every declared tiling key against memory capacities. Otherwise a narrow phase's second slot can overlap a wide phase's first slot under a different ID. Preserve the completion protocol when unifying IDs: shared mutex ownership cannot replace the drains required by accumulator `phase=`/`unit_flag` or a GM handover. The [AttnEpilogue kernel](../../examples/algorithms/attn_epilogue/kernel.py) uses this layout for two projection widths, with the second projection's first weight prefetched through a view that inherits the first phase's ownership.

Serial phases may require incompatible physical tile families, such as BF16 versus FP16 operands or different narrow/wide panels. One candidate is to declare separately typed groups over an intentionally shared address range, if the installed declaration and alias planner support that overlap. This is storage reuse, not a numerical conversion: each phase must produce or reload values in its own required representation. Budget the full backing range for every phase, and prove that all earlier readers and writers, including delayed stages and asynchronous stores, finish before the next typed use. A whole-core barrier can be part of that completion protocol only with its required drains and participants; do not infer [C-layer event lifetime across that barrier](crosscore-sync.md#2-for-reusing-a-buffer-in-a-loop-forward-synchronization-is-not-enough) from the address alias. If the overlap cannot be expressed or its completion proved, use disjoint storage or a supported ordered boundary.

Count simultaneously live values inside each helper, not just named groups. A helper consuming two RHS values together needs two physical slots: taking `.next()` twice from a depth-one group still names the same storage, and automatic mutexes cannot preserve a value overwritten before its use. Inspect the generated typed declarations, addresses, slot selectors and mutex ownership together, then test phase transitions, ring wrap and return to earlier shapes/dtypes with distinguishable values. This is a candidate to validate on the matching toolchain, not a new target-validated aliasing template.

## Independent Buffers and Slot Depth

Independent buffers need disjoint byte ranges. Copying `mutex_ids=[0]` into every group can serialize unrelated work; automatic planning avoids manual bookkeeping but still requires inspecting generated synchronization when diagnosing performance. Compare ID assignment at fixed shapes and depth.

Trace input, retained state, transformed keys, record intermediates and output as separate roles before minimizing UB bytes. Reusing the input as the output makes its next load wait for the output's final store; carrying a residual through normalization/apply or using the input again as a sort payload extends its read lifetime too. When capacity permits, compare independent input and state/output storage at fixed batch, slots and workers. Distinct storage and IDs can permit overlap, while physically aliased views must retain shared ownership. The [bounded-index sorter](../../examples/algorithms/sort_sparse_indices/) uses independent input/output and merge buffers and recovers its result from an invertible key. A switch from full barriers to automatic mutexes alone does not remove dependencies created by an unchanged storage graph.

For independent chunks in a load→VF→store loop, compare a single slot with a ring of disjoint slots when capacity permits. The last store must finish before its slot is overwritten; VF completion alone does not release an updated tile. Advance related groups consistently and prove that GM chunks are independent too. In-place neighboring chunks may need additional ordering.

Keep long-lived reduction state separate from rotating input slots when an in-place aggregate holds an input slot past its last input consumer and blocks the next prefetch. A distinct state tile or legal register state can shorten that input slot's lifetime; compare the added storage or copy cost rather than inserting an unconditional UB copy. The aggregate's dependent updates remain ordered.

Local slot rotation is not a proof of a GM read-after-write dependency. If a later pass reloads GM bytes written through another local slot, establish completion on that actual producer/consumer path; different local addresses and mutex IDs do not describe the shared GM bytes. Verify the supported cross-pipe completion edge for a same-core path, the [GM handover protocol](crosscore-sync.md) for cross-core work, or an ordered native-call boundary when the dataflow uses separate launches.

Tune transfer width and depth together, accounting for every operand, padding and slot. A larger ring may require smaller transfers or add control work. Test tails, short runs and enough iterations to wrap the ring, then compare complete calls. Register lane count does not set the DMA block size.

## L0 Reuse

Left and Right tiles may share one base address across sequential blocks when synchronization orders the reuse. Do not allocate a separate L0 copy for every software lane without accounting for the total capacity. `pl.gemm` receives caller-owned `l0a=` / `l0b=` groups; it does not allocate them.

Multiple pipeline stages may share one local L0 group. A one-step `pl.gemm` advances the supplied group; depth 2 can overlap MTE1 transfers with matrix work. For calls with different shapes, declare the L1, L0 and accumulator chain at the largest shape, use `compact=1` on narrowed tiles, and pass effective extents. Declaring a smaller L1 tile against a larger L0 tile fails the move shape contract.

Two `pl.gemm` calls on different weights in the same Cube stage can reuse a depth-1 L0A/L0B set when their transfers and matrix work are ordered. Calls in independently scheduled stages require their own lifetime analysis; see [GEMM](gemm.md).

## MX Scale Planes

Declare ScaleLeft/ScaleRight groups with explicit per-slot addresses equal to the paired Left/Right data address `>> 4`, and the same depth. Do not derive scale-plane spacing from the plane's own size. L1 scale views remain in Mat, with `layout=pl.ZZ` on side A and `pl.NN` on side B. Scale planes retain separate mutex ownership; see [GEMM MX](gemm-mx.md).

## Regression Entry Points

Framework `test_auto_mutex_ids.py` and `test_mutex_id_planner.py` cover automatic planning, manual pins, aliases, independent target pools and invalid declarations. `test_auto_mutex_slot_count.py` covers pipeline depth without a manual ID list. Device tests under `frontend/bufferid/test_auto_mutex_ids.py` cover runtime selection, helpers, aliases and more than 32 slots. `test_pipelined_crosscore_handover.py` preserves preload and local-slot diagnostics with explicit declarations.
