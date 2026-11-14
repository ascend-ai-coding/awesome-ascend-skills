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

## Independent Buffers and Slot Depth

Independent buffers need disjoint byte ranges. Copying `mutex_ids=[0]` into every group can serialize unrelated work; automatic planning avoids manual bookkeeping but still requires inspecting generated synchronization when diagnosing performance. Compare ID assignment at fixed shapes and depth.

For independent chunks in a load→VF→store loop, compare a single slot with a ring of disjoint slots when capacity permits. The last store must finish before its slot is overwritten; VF completion alone does not release an updated tile. Advance related groups consistently and prove that GM chunks are independent too. In-place neighboring chunks may need additional ordering.

Tune transfer width and depth together, accounting for every operand, padding and slot. A larger ring may require smaller transfers or add control work. Test tails, short runs and enough iterations to wrap the ring, then compare complete calls. Register lane count does not set the DMA block size.

## L0 Reuse

Left and Right tiles may share one base address across sequential blocks when synchronization orders the reuse. Do not allocate a separate L0 copy for every software lane without accounting for the total capacity. `pl.gemm` receives caller-owned `l0a=` / `l0b=` groups; it does not allocate them.

Multiple pipeline stages may share one local L0 group. A one-step `pl.gemm` advances the supplied group; depth 2 can overlap MTE1 transfers with matrix work. For calls with different shapes, declare the L1, L0 and accumulator chain at the largest shape, use `compact=1` on narrowed tiles, and pass effective extents. Declaring a smaller L1 tile against a larger L0 tile fails the move shape contract.

Two `pl.gemm` calls on different weights in the same Cube stage can reuse a depth-1 L0A/L0B set when their transfers and matrix work are ordered. Calls in independently scheduled stages require their own lifetime analysis; see [GEMM](gemm.md).

## MX Scale Planes

Declare ScaleLeft/ScaleRight groups with explicit per-slot addresses equal to the paired Left/Right data address `>> 4`, and the same depth. Do not derive scale-plane spacing from the plane's own size. L1 scale views remain in Mat, with `layout=pl.ZZ` on side A and `pl.NN` on side B. Scale planes retain separate mutex ownership; see [GEMM MX](gemm-mx.md).

## Regression Entry Points

Framework `test_auto_mutex_ids.py` and `test_mutex_id_planner.py` cover automatic planning, manual pins, aliases, independent target pools and invalid declarations. `test_auto_mutex_slot_count.py` covers pipeline depth without a manual ID list. Device tests under `frontend/bufferid/test_auto_mutex_ids.py` cover runtime selection, helpers, aliases and more than 32 slots. `test_pipelined_crosscore_handover.py` preserves preload and local-slot diagnostics with explicit declarations.
