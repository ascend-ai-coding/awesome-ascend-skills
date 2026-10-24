# Cross-core Handover: Three Layers and How to Choose

This page explains which mechanism to use when data passes from cube to vector or vice versa. **Pro has three layers, not one**—the most common mistake is not knowing they exist, falling back to hand-written events, and then using the wrong pipe.

## First Separate the Two Axes

These three layers operate on different axes:

```
Intra-core synchronization ─── A: auto_mutex + mutex_ids
Cross-core handover ────────── B: pipeline + @stage + cross-core buffer declaration   (automatic)
                               └── falls back to ──┐
Cross-core primitives ──────── C: set_cross_core / wait_cross_core
```

**B does not replace C; it automates C**—`_cross_core_scanner.py` scans which buffers each stage touches, their roles (R/W/RW), and the pipes involved, then inserts `set_cross_core` / `wait_cross_core` at stage boundaries.

| Layer | Entry point | What it handles | What it does **not** handle |
| --- | --- | --- | --- |
| A | `pl.jit(auto_mutex=True)` + `mutex_ids` | Within one core: pipeline synchronization and address reuse for Tile slots. | Nothing across cores. |
| B | `pl.jit(pipeline=PipelineConfig(preload=N))` + `@pl.pipeline.stage` + cross-core buffer declarations | **Automatic** cross-core handover: `preload=0` inserts synchronization only; `>0` also rewrites the pipeline. | Recognizes only buffers shared by stages on both sides and explicitly marked cross-core. |
| C | `pl.system.set_cross_core` / `wait_cross_core` | Nothing beyond emitting exactly what you specify. | Does not check whether the pipe is correct, IDs collide, or reverse synchronization is missing. |

## Why Layer A Cannot Cross Cores

It is not missing implementation; the structure makes this impossible. Cube and vector compile into **two independent binaries** (`jit.py`'s `_assemble_cv_source` splits at `__DAV_CUBE__` / `__DAV_VEC__`), and `mutex_lock` lowers to per-core `get_buf` / `rls_buf`. Each side has its own independent space of 32 IDs—**an ID collision across the two sides is not a conflict**, because they never meet.

The official documentation states this up front:

> The **source order** of `section_cube()` and `section_vector()` cannot replace inter-core synchronization.

## How to Choose

```
Can the loop body be refactored as @pl.pipeline.stage -> Use layer B; start with preload=0, verify, then increase it.
  ├─ Handover medium is make_tile_group                 -> Layer B works directly.
  └─ Handover medium is a GM workspace                  -> Layer B + pl.bind_crosscore_sync (see below).
One-shot handover                                        -> Layer B also works with a one-iteration driver loop; choose C only if no loop is used.
The same buffer is reused in a loop                     -> Layer B; if writing by hand, both forward and reverse IDs are required.
Like cores across blocks (not AIC↔AIV)                   -> Layer C, with sync_mode=INTER_BLOCK.
```

**`preload=0` is the most overlooked setting.** It performs no pipeline rewrite; it only inserts cross-core synchronization into the original serial loop (`config.py:33`, `_transformer.py:368`). Stop at this setting when you want “automatic synchronization” without “pipeline reordering.”

### Layer B Still Needs a Driver Loop, Even for One Handover

`pipeline=`, `@pl.pipeline.stage`, and explicit cross-core event declarations do not automatically form a pipeline between the two top-level sections. Put both stage calls in the same driver loop, `for step in pl.range(...)`; one handover needs only `pl.range(1)`, with no artificial extra task or stage. Put exactly one direct stage call in each stage-containing `with pl.section_*()`; place loads, computation, and conditions in the stage function. Verify synchronization with `preload=0` first; if you see `pipeline: no pipeline loop found`, inspect this structure instead of changing synchronization layers or adding slots.

This structural requirement comes from `runtime/pipeline/_analyzer.py::_find_pipeline_loops` / `_stage_call_in_section`; a single-iteration loop was compiled and run on board under behavior version `872df763f`, CANN 9.2.0, Ascend950PR. A loop with one iteration is still a loop, but it is not evidence of pipeline performance.

The installed pipeline validator also requires the stage chain to strictly alternate Cube and Vector, including at `preload=0` (`runtime/pipeline/_validate.py::_check_alternating_sections`). Combine consecutive same-core work, such as a Vector GM load and ND→NZ conversion, inside one producer stage before the Cube consumer. Adding another `section_vector()` around a second stage does not create a legal overlap. Derive the producer's local scratch lifetimes separately from the shared Mat slot lifetime.

### Audit every handover edge

For each shared operand, score panel and returned product, identify the producing/consuming stage, actual writing/first-reading pipe, group/slot and release before reuse. A Cube→Vector→Cube chain needs both directions accounted for. Declare `fwd_ids` / `bwd_ids` on each shared `make_tile_group`, then inspect transformed IR or CCE for the corresponding forward wait and reverse release. Seeing events for K/P/V does not establish completion of a separate score or product buffer. `auto_mutex` and `preload=0` alone do not declare these edges.

The source checker reports `stage-handover-unverified` for directly traceable unmarked groups referenced by both Cube and Vector stages. This is an advisory: intentionally sharing another group's events can be valid, but requires proving matching producer/consumer pipes and the full lifetime. Closure captures, aliases and helpers are outside that precheck. Use the existing framework scanner and emitted program for the actual synchronization analysis; a clean source report is not a race-freedom proof.

### Compute tiling is not a new synchronization stage

When a larger output requires several local matrix panels, first keep that tile loop inside the existing consumer stage if the next role only needs the completed output. Receive the shared operand once, perform the panel loads/products/drains in that stage, and hand over the completed regions with their original ownership. A stage names a synchronized role, not every GEMM call. Duplicating a stage per panel can turn one producer/consumer edge into unsupported fan-out or a non-adjacent handover; renaming the duplicate functions or aliasing the buffer does not repair that graph. Introduce extra stages only for an intended overlap, after deriving the new producer/consumer pairs, adjacency and slot lifetimes.

### Keep slot selection visible to the stage scanner

Pass invariant tile groups to stages and select their slots inside the owning stage with a separate simple assignment, for example `slot = group.current()` followed by `consume(slot)`. The same form applies to `.next()`, `.previous()` and `group[index]`. A traced group accessor nested in another call, a chained assignment or tuple unpacking is rejected by `_cross_core_scanner.py::_validate_slot_accessors`; this includes traced same-core state groups, not only a group with explicit `fwd_ids` / `bwd_ids`.

An outer-loop slot passed to a delayed stage requires statically resolved slot counts in `_analyzer.py::_collect_outer_slots`, which snapshots the selector for that task. Automatic mutex groups provide this structural count through `depth`; manual ID lists can provide it through their length. Prefer moving the selection into the stage and passing the invariant group. Keep per-item initialization and finalization in their owning stages too, or establish the completion boundary described in [online state lifetime](../patterns/online-softmax.md#build-the-serial-recurrence-before-overlap). These parser boundaries were reproduced on behavior `872df763f`; they do not imply that every tile access outside a stage is invalid.

### Postprocess a shared input without changing its writer

For the first fused slice, keep a received cross-core buffer read-only in its consumer. Load its values into VF registers and apply a mask, scale or bias there; if another tile is needed, write a separate local tile. Editing the received slot with `pl.setval` or a VF store adds a writer and can change the stage's pipe dependencies, even for a single element. Moving that edit to the producing stage can still add a different writer pipe. Inspect those accesses when a new postprocess introduces a zero-distance cycle; reducing preload changes scheduling distance, not the read/write topology.

For score masking, use the [register selection form](vf.md#25-combining-masks-with-existing-comparisons-and-selection) and pass the masked register into both the reference/max and coefficient computations. A load already supplies a private register value; no writeback to the shared score tile is required. Designs that intentionally update a handover buffer need their own supported access/ownership proof rather than inheriting the read-only anchor's synchronization argument.

### Handover Medium: Prefer On-chip; GM Is the Fallback

This same-Block transfer preference assumes ownership has already been chosen. It does not choose the grids or kernel boundaries of the complete algorithm; first use the [joint residency/ownership decision](gemm.md#transformed-operands-format-residency-and-repeated-work).

The decision tree above is about “which synchronization layer”; this section is about “which data path.” For a cube → vector result within the same Block, **prefer direct on-chip transfer**: `pl.move(ub_tile, acc_tile, acc_to_vec_mode=…)` (Acc→Vec, FIX pipe; `DualModeSplitM/N` can split an accumulator across two AIVs in one step, but dual modes support same-dtype direct transfer only—narrowing such as FP32→FP16, `scale`, and DN layout are rejected; use Single mode or `vf.astype` after move to convert; see the contract in `_api.py::move`). **Legitimate GM-workspace cases:** the whole output does not fit on one side, handover crosses Blocks, or the Host must read an intermediate value. Measurement (2026-09-28): a single-Block GM round trip costs two HBM transfers and exposed a nondeterministic wrong-value case from reusing the output tensor as a relay (see the silent wrong-result list); direct on-chip transfer avoids both. Board-verified reference: [matmul_add_chain](../../examples/matmul/fundamental_usage/matmul_add_chain/kernel.py) (`DualModeSplitM` + C layer).

### Dual-destination Transfer: Derive Each AIV's Ownership from the Effective Extent

First determine the actual effective M/N in Acc before `pl.move` (including narrowing by `pl.gemm(m=..., n=...)`), then compute the Vector global offset, UB valid extent, and VF mask. **The buffer's declared size gives capacity; it does not determine where a tail block splits by row/column.** Computing each subblock offset as half the declared size can leave unwritten regions or read invalid data; correct synchronization cannot fix this misalignment.

An ordinary same-dtype `DualModeSplitM/N` transfer with no offset splits as follows. Let `extent` be the effective size of the split axis before transfer, and `s` be the AIV subblock index, 0 or 1:

| Mode | Split step `half` | Subblock offset on that axis | Valid elements in subblock |
| --- | --- | --- | --- |
| `DualModeSplitM` | `(extent + 1) // 2` | `s * half` | `max(0, min(half, extent - s * half))` |
| `DualModeSplitN` | `((extent + 31) // 32 * 32) // 2` | `s * half` | `max(0, min(half, extent - s * half))` |

Reserve UB capacity using the aligned size first; the consumer handles only the valid subregion above, and uses the true valid size on the other axis too. An AIV with no valid elements must still enter its paired stage and advance the handover group slots; skip only data processing and GM writeback, not automatic synchronization. Verify that the union of the two valid regions covers the output exactly with no overlap. Acceptance should include an extent smaller than a split unit, an extent that does not divide evenly, and a tail crossing the split-unit boundary; use guards before and after outputs to detect overruns. Checking only the four output corners does not verify memory bounds.

The rule is in the official AccToVecMode (`pypto/docs/zh/api/pro_api/SIMD-API/basic_data_structures/AccToVecMode.md`) documentation; `backend_cce_block_out_ops.cpp::MakeBlockOutMoveCodegenCCE` rounds effective Acc M up to 2 and effective N up to 32 in `872df763f`. The `move` Python docstring describes dual-mode type and rider restrictions but does not give the tail-split formulas; the geometry cannot be inferred from that docstring alone. Offset extraction, other layouts, and paths with additional control parameters require checking their own contracts separately.

### Logical validity is not transfer capacity

Keep three quantities distinct when adapting a padded kernel: the rows used by the mathematical result, the effective Acc extent actually passed to FIX, and each AIV's receiving allocation. An example may use one constant for two of these because they happen to coincide at its original shape. Changing that constant everywhere can shrink a destination while leaving its producer unchanged.

For an ND `DualModeSplitM` handover with effective Acc shape `[M, N]`, each AIV needs room for the aligned split footprint: `ceil(M / 2)` rows at the destination's physical row pitch, retaining the column/pitch padding required by that transfer. The destination declaration or its valid shape does **not** clip the source-driven write. For example, a producer still computing 32 padded rows writes 16 rows per AIV even when only 7 rows are mathematically useful. Allocating only 7 receiving rows is insufficient; ignoring the extra rows in VF or skipping an AIV's final store does not remove those writes. Apply the split-N alignment above when that axis is split instead.

Choose one consistent adaptation: retain the producer extent and full receiving capacity while limiting VF/output consumers to useful coordinates; or narrow the producer's effective extent and rederive both AIVs' offsets, valid regions and coefficient handback. Narrowing the source can move useful rows onto the second AIV, so retaining an old “only AIV0 writes” branch is not automatically correct. Trace this decision through every contraction and handover in the chain, not only its first score tile.

Review the emitted source's Acc valid shape, destination type, address and slot spacing together. Numerical agreement can coexist with writes into an adjacent slot whose values are overwritten later. To probe the transfer footprint safely, reserve a full-sized backing, establish distinguishable contents, transfer through a narrower view, then read the entire backing; do not deliberately underallocate physical storage. The source-controlled footprint was rechecked with such a probe on `75a6c2600`, CANN 9.2.0; the installed `TMov.hpp::TMOV_IMPL` obtains M/N from the source before `TMovCcToUb`. Static-rule success and a sanitizer result do not replace this access calculation; see [check coverage](../diagnostics/check-coverage.md#5-three-distinct-criteria-for-out-of-bounds-access-each-with-its-own-blind-spot-board-evidence-2026-09-29).

The [tile-view receiving-capacity test](../../examples/basic_api/datamove/tile_view/test.py) exercises narrow views over a complete backing, both AIVs, output guards and changed inputs on repeated calls.

### v→c Return Path: NZ Packing and UB Bank Conflicts; Row+1 Pitch Convention (2026-09-29)

When converting vector results to NZ for cube, **packing NZ with a compact pitch (physical row count = logical row count) causes UB bank conflicts**. The performance convention is to add one physical row to the UB NZ tile (pitch = rows+1, making it odd to offset banks). Two precedents: ascriptor's `ub_to_l1_nz` (v8_allhif8 packs P with a 33-row stride and **requires exactly +1**—aligning to 48 instead caused the second P slab to be read incorrectly on board); and PyPTO's own FA (`python/tests/st/pypto_pro/frontend/auto_pipeline/test_fa_serial_dn_auto.py`), whose three-part recipe is `make_tile_group(TileType(shape=[rows+1, cols], valid_shape=[rows, cols], layout=pl.NZ))` (valid shape declared at **declaration time**) + `vf.store_align(..., block_stride=(rows*2>>1)|0x1, repeat_stride=1, data_copy_mode=DATA_BLOCK_COPY, post_update=True)` (linear cursor advance, data traversed along columns) + `pl.insert` into L1 (TINSERT adapts to the source NZ declaration's pitch).

**Feasible form (board-verified 2026-09-29; also applies when data runs along rows):** the cursor form of vsstb: `store_align(nz, reg, preg, data_copy_mode=DATA_BLOCK_COPY, block_stride=physical_column_height, repeat_stride=1, post_update=True)`, with **no explicit offset**. Semantics (probe disproved alternatives bitwise): the initial cursor is the tile base; each store advances by `repeat_stride`×32B (= one 16-element row slice); the register's eight DataBlocks land at `cursor + j*block_stride*32B`. The row loop naturally yields the `r*16` address sequence, and one instruction writes a row slice across eight fractal columns. One 128-element FP16 register exactly covers one 128-column NZ tile; **for wider rows, create one independent NZ tile per column group** (the cursor advances linearly and cannot express the large jump across column groups; the upstream FA ST uses the same two-tile pattern for `x_exp`), then insert each group at its column offset. The in-repository implementation is `examples/algorithms/mm_softmax_mm`: it removes the ND staging tile and full `pl.move`, reducing device time from 9.64 to 9.34us and `aiv_vec` from 0.34 to 0.275 with bitwise-identical results. With matching orientation, this also applies to a skinny tile smaller than one 16-row fractal: calculate pitch as rows+1 (board-verified with 8 rows / pitch 9 in [decode_attention_gqa](../../examples/algorithms/decode_attention_gqa/kernel.py), using a `[9,128]` NZ group); do not round up to 16.

**Three framework gaps (board evidence + generated code identify the cause).** All three have been rejected during parsing on the PyPTO dev branch: (1)(2) since `f2c693e27` (hint points to the cursor form), (3) since `b77a246fc`; only Vec→Vec is rejected for the align16 write pitch in TMOV/TEXTRACT (Acc→Vec uses exact declared Rows, source review found no defect to reject it), and the hint points to the same-address, dual-declaration workaround below. Earlier versions silently accepted these forms. The fix belongs in the CANN toolchain PTO headers (`TMov.hpp`/`TExtract.hpp` should use exact declared Rows instead of align16); evidence supports exact sizing, so this should be reported through the CANN channel (this repository does not fix CANN headers). **Known remaining blind spot (not statically checkable):** pitch is based on runtime `GetValidRow()`—if the declaration is aligned to 16 and `set_validshape` narrows across a 16-row boundary, it is still misaligned; the narrowing semantics of the dual-declaration workaround happen to satisfy `align16(valid)==declared rows`, avoiding this case.

The first two issues share one root cause: **the vsstb instruction has no address operand**—the destination address is tile base + hardware cursor, and the third parameter is the compound stride field `(block_stride << 16u) | (repeat_stride & 0xFFFF)`. Correct generated code is `vsstb(reg, ptr, (33 << 16u) | (1 & 0xFFFFU), preg, POST_UPDATE)`. Codegen nevertheless accepts explicit offsets and puts them into the upper 16 bits of this field: (1) with a list offset, it emits an **empty expression** such as `( << 16u)`, which bisheng fails to compile; (2) with an integer offset, it encodes the offset expression as block_stride—generated `((_expr_tmp << 16u) | ...)`—so the user's offset becomes the block distance, the actual block_stride is discarded, and **a legal instruction has entirely wrong semantics, silently passing compilation** before an out-of-bounds 507015 at runtime. The fix must either reject this combination during compilation or encode offset as a base-address adjustment; it must not place it in the stride bits.

(3) **`pl.move` to a UB NZ destination wrongly applies the L1/L0 `align16(rows)` column-height rule to UB** (UB NZ column height must equal the exact row count): after moving to a `[33,256]` declaration, a physical-order dump decoded with a 48 pitch reported bad=0, while decoding with 33/32 reported bad=7680/8192—move wrote an align16(33)=48 pitch but `pl.insert` read using the declared pitch, so two components in the same framework disagree about UB NZ pitch rules and the full chain is misaligned after the first chunk.

Alternative template (bit-exact probe of ascriptor's PyPTO Pro printer, `backends/pypto_pro/emit.py`, `dma.ub_to_l1.nz` branch): **two declarations at the same address**—pack bytes on the vector side, then before insert declare a second NZ Tile at the same UB address (`pl.make_tile`/`make_tile_group(addrs=)` with physical shape `[rows+1, cols]`), narrow with `pl.set_validshape(alias, [rows, cols])`, then `pl.insert` and restore. Two hard constraints: a compact-NZ window with row offset cannot be expressed with an alias (rows interleave within 32B blocks), and column offsets must be 32B-aligned. Use this for hand-laid addresses; the sugar path can declare a `[rows+1, cols]` NZ group directly (`pl.insert` source must be a registered group for the B-layer crosscore scanner; `pl.reinterpret` outputs are not registered).

**Tested and rejected (board evidence, cause established):** (a) Masked scattered writes—simulate scattering after narrowing with masked stores over eight 16-lane windows. Results were bitwise correct but device time doubled (9.64→19.89us, `aiv_vec` 0.34→0.72): each store writes only 1/8 of the lanes, giving an 8× instruction cost on the VF port and landing on the critical path. Masks select continuous prefixes; scatter belongs in the store stride (see “separation of responsibilities” in vf.md). (b) Multi-step construction of row+1 (for example, the self-consistent 48-pitch `pl.move` declaration, 9.83us ≈ compact pitch at 9.64us)—**pitch gaps in an insert source are nearly free** (TINSERT burst count depends only on valid rows), and compact-pitch UB bank conflicts were not measurably costly on the move+insert path. The net gain from row+1 is only “one-step scattered register store, saving a move”; pitch itself is not the main cost.

### Preload scheduling and slot lifetime

For `preload>0`, the whole pipeline loop is **replaced** (`_transformer.py::_transform_body`): each beat in the main loop advances one task_id, and each stage runs its own iteration `t - delay` under an `is_valid` guard; before the loop, a ctx ring buffer is declared (`pl.struct_array(depth)` holding stage arguments and slot snapshots for each beat, `depth = max_delay + 1`—this is the depth of **argument snapshots**, not data buffers). After the loop, drain beats let stages with high delay finish, then reverse set/wait events are balanced. `preload=0` takes another path, `_transform_serial`: keep the serial loop and insert synchronization only.

Delay allocation rules (`_compute_delays`) are independent of data flow and depend only on source order and core ownership:

- First stage globally: `delay = 0`.
- First stage on a core: delay of the preceding (cross-core) stage + **1**.
- Later stages on the same core: delay of the previous stage on that core + **preload**.

In a 1+1 topology the first cross-core handover has distance 1 at any positive preload. In a longer alternating chain, return hops can instead have distance `preload - 1`; same-core stage distance is `preload`. Derive each edge from its two stage delays, as in the table below. Increasing preload also grows the argument ring and fill/drain span; the rewritten schedule alone does not establish a hardware speedup.

### Slot counts, identity and check coverage

**Slot-count formula** (`_validate.py::_check_slot_counts`; transform-level evidence from 2026-09-30): required slots = **distance in beats between producer and consumer stages + 1**. Calculate distance from the stage delays, not directly from the preload setting. The `N` in “at this preload its producer runs N iterations ahead” is the beat distance, not the slot count. For a 1+1 topology, `depth=2` passed at preload 0/1/2/5; `depth=1` was rejected at preload=2/5 with distance 1. The longer-chain table below distinguishes return and same-core hops. At `preload=0`, the serial path has distance 0 and one slot suffices.

**Pipeline inspection must use the transform path.** Inspect `pipeline_generated.py`, generated CCE (`kernel.cpp`) or IR parsed after `jit.py::_prepare_codegen_inputs`. Calling `to_kernel_def().parse_target_program(...)` on a fresh definition bypasses the pipeline transform and its checks. For a parser-only probe, use `probe_kernel_facts` + `transform_pipeline` as in framework regression `test_gm_handover._generated`; check these internal entry points against the installed version before using them. Structural inspection requires no simulator execution.

**The same formula also applies to same-core inter-stage buffers** (board-verified 2026-09-30 in flash_attention mutation tests): any group that survives across stages needs `distance in beats + 1` slots—same-core distance includes the `+preload` term, so same-core handover buffers such as `alpha` need `preload + 1` slots. One slot too few for a cross-core buffer gives a clear compile-time error (above); **one too few for a same-core buffer silently gives wrong values, not a deadlock** (board evidence: after `alpha` was reduced by one slot, 110,000 elements were out of tolerance with no error). Since PyPTO dev `4ad55d3fb`, a convicted shape (two stages, write then read, slot count < distance+1) emits a compile-time `SlotShortage` warning. Its three blind spots (kernel has no cross-core buffer, >2 users, unlisted VF op prevents complete analysis) stopped being silent since `80a45377a`: an uncovered case emits a separate `SlotCheckIncomplete` warning (can be muted independently from `SlotShortage`; seeing it means the checker did not cover your form, so manually verify slots using this formula). Upstream versions neither check nor report this (`_check_slot_counts` checks cross-core buffers only). The defense is unchanged: test “preload=0 is bitwise identical to the shipping setting” (serial has distance 0, so one slot suffices; the error appears only in pipeline mode). Treat both warnings as supplemental signals.

**Slot count and slot identity need separate proofs.** For manually indexed state crossing delayed stages, enough slots do not protect a selector that aliases live tasks. If the pipeline continues across items, an item-local `j % depth` resets while the previous item's consumer may still be pending. Select by the continuous logical task position, passed as a scalar to the owning stages, or prove an explicit completion boundary before resetting the selector. Producer and consumer must name the same logical task's slot, not the current physical pipeline beat. Cover adjacent items with different reduction lengths/ring phases and distinguishable state; an even-length item can accidentally hide a reset collision that an odd-length one exposes. Apply this to same-core statistics as well as cross-core handovers.

Pruning can also break a rectangular index such as `tick = outer_iteration * full_count + part`: with `full_count=2` and one executed iteration per item, consecutive tasks select `tick=0,2,4` and reuse slot 0 under `tick % 2`. In a c-v-c-v pipeline at `preload=1`, softmax writes the next task's state before update reads the previous task's state. Advance a user-owned logical task counter once per executed pipeline iteration and pass it to the owning stages, or prove another lifetime-safe mapping. Matching trip-count parity only addresses this two-slot mapping; deeper pipelines still require sufficient slots for their stage delays. A passing `preload=0` control alone does not establish a restriction on runtime or block-dependent loop bounds.

**Beat-distance table for a multi-handover chain (c-v-c-v; transform-level evidence from 2026-09-30; cross-validated on single-head flash_attention and flash_attention_mha—the chain shape matches, slot counts differ, and results agree at every setting):** In an alternating chain, “the first stage on the other core = previous stage + 1” is used exactly once (for the second stage); all later delays are “previous same-core stage + preload,” so `delay_i = P·⌊i/2⌋ + (i mod 2)`. The starting core does not affect the values. Distance per hop (P ≥ 1):

| Hop | Distance in beats | Slots | fa shipping P=1 | mha shipping P=4 | Check signal (measured) |
| --- | --- | --- | --- | --- | --- |
| Hop 1 qk→softmax (`shared_s`, c→v) | 1, constant | 2 | 2 | 2 | Compile-time error reports N=1 (P=1/2/4). |
| Hop 2 softmax→pv (`shared_p`, v→c return) | max(P−1, 0) | max(P−1, 0)+1 | 1 | 4 | Compile-time error reports N=P−1 (P=2: 1; P=4: 3). |
| Hop 3 pv→update (`shared_pv`, c→v) | 1, constant | 2 | 2 | 2 | Compile-time error reports N=1 (P=1/4). |
| Same-core softmax→update (`alpha_g`/`st_g`) | P | P+1 | 2 | 5 | `SlotShortage` warning reports dist=P (P=1/2/4, regardless of the slot count actually configured). |

- The distances of two adjacent hops always sum to P (alternating 1 and P−1); chain head/tail do not change the distance of any hop, only the fill/drain count. At P=0, all distances are 0 and one slot per group suffices; at P=1, hop 2 has distance 0, so softmax and pv share a beat and one slot is valid (measured in both examples).
- **A non-adjacent cross-core handover does not exist:** `_check_cross_core_users` rejects it at compile time (`“with N stage(s) in between”`; measured for qk producer and update consumer). The distance formula only needs adjacent hops. Multi-stage data flow must use a same-core local group (warning scope above) or relay it hop by hop.
- Both examples use the formula's **minimum slot count** for each group in their board-verified shipping configurations; there is no excess. Larger slot counts may parse but were slower in board evidence (mha kernel docstring: a third `shared_s`/`shared_pv` slot cost +1.7%/+6.5%, P-slot overprovisioning |Δ|≤0.04%, widening `st_g` cost +0.5%). Shipping configurations have no warnings on the transform path.
- Evidence scope: “check signal” is the framework's check model (distance read from schedule), not a claim about the hardware minimum. Insufficient cross-core slots are rejected at compile time (so no board behavior is available); insufficient same-core slots silently corrupt board results (evidence above). The known blind spots for same-core warnings were not triggered in this chain shape (exactly 2 users, all VF ops are in the role table).

**Generated set/wait counts need not match statically** (counted in generated code for gemm_bias_silu_banded preload=2, 2026-09-30): pre-fire (set only) and drain (wait only) pair in aggregate; in the drain region, only stages with delay are replayed, so when a replayed stage's own set/wait counts differ (consumer on multiple pipes = one wait + N sets, see release rules under “one ID per slot”), the static difference is N−1—9 sets / 8 waits in that example. Dynamic accounting per counter balances exactly (bwd 12=12, fwd 4=4, cross-launch counters reset), consistent with all board consistency tests passing. Judge defects by dynamic accounting, not static text counts.

**The ctx ring snapshots scalar arguments only** (transformed source plus board compile-time rejection, 2026-09-30, flash_attention_mha): in the steady region, a delayed stage call resembles `pv_block(_pl_ctx_neg2.j, ..., vv, ...)`—scalars such as `j` / `g` are read from the snapshot, but non-scalar arguments such as Tile groups, Tensors, and views **directly reference the current value of a same-named variable**. Therefore, non-scalar arguments to delayed stages must be loop-invariant. A view created inside the loop (such as a per-item 2D GM view) cannot be passed to a delayed stage: the drain region is outside the loop and the variable is out of scope, so compilation rejects it with a `NameNotFound` diagnostic: `NameNotFound: [_expression_parser.py:305][PRO_PARSER]:ErrCode: F00004! ... Use of potentially undefined variable 'vv'` (the same argument to a non-delayed stage compiles, so the error appears only at the first delayed stage call; since PyPTO dev `6c71d8c5a`, lowering reports the cause as `NotSupported` during transformation). Workaround: create the group-level view once outside the loop (`pl.reinterpret(group, ...)`) or create a 2D view outside, and pass only per-item scalars.

### Layer B Gateways

- A stage body cannot call another stage, directly or through a helper, or return a value (`_check_no_nested_stage` / `_check_stage_returns`). This concerns nested calls, not defining stage functions inside a Host factory; the [rectangular handover factory](../../examples/matmul/transpose_reinterpret/kernel.py) uses local stage definitions. Ordinary functions called by a stage cannot return Tile or TileGroup.
- Struct arguments must use `pl.struct()`; `pl.struct_array()` is unsupported.
- If `preload` is a list, its length must equal the number of pipeline loops inside the kernel.

### Measured Facts about Event IDs and Concurrency (Board Evidence: 2026-09-29)

- `set/wait_cross_core` are **within-block** primitives (1 cube ↔ 2 vec); event counters are private to each block. Multiple blocks may concurrently use the same event ID pair legally, and ID values do not affect correctness (0/1, 10/11, and 14/15 were all board-verified).
- `MAX_EVENT_ID=15` comes from the INTRA_BLOCK +16 mapping: hardware event space is 0–31, and one cube-side set/wait expands to `id` and `id+16` (one slot per AIV), so the user's half is 0–15. A single UNICAST_BLOCK event is not offset; the official “0~31” likely refers to it—Python checks every mode against 15, which may be too strict for UNICAST (verify the installed implementation before relying on the wider range).
- Mutex and event pools are independent at both Python (`MAX_MUTEX_ID=31` / `MAX_EVENT_ID=15`) and hardware (`get_buf/rls_buf` vs `set/wait_intra_block`) layers. The B-layer allocator avoids only IDs declared by crosscore groups; it cannot see hand-written set/wait. When mixing both layers in one kernel, avoid collisions yourself.

### A Mixed Kernel's `block_dim` Is a Group Count (Launch Semantics ≠ Index Semantics)

For a mixed kernel, `X` in `kernel[None, X]` is the **number of execution groups** (one AIC + two AIVs per group), equal to the number of physical cores to use. Three sources agree: official `kernel_function.md` “Meaning of blockDim” (block_dim=20 means 20 execution groups); `pypto_launch.h::LimitLaunchBlocks` does not divide the request by 2 and instead caps it at `min(request, budget core count / cores per group)`; and board evidence from 2026-09-29 (group-count inversion plus out-of-bounds address accounting).

For a shared Cube/Vector driver, derive the item owner identically on both sides:

```python
group_id = pl.get_block_idx() // pl.get_subblock_num()
for item in pl.range(group_id, item_count, pl.get_block_num()):
    # Both sections enter the stages for this item; split local rows inside them.
    ...
```

The runtime block count is the group stride; only the flattened Vector index needs division. Recheck this mapping when fusing previously separate kernels: retaining the pure Vector `get_block_idx()` owner can give the paired AIVs different items or loop counts from their AIC. That can mix intermediate data or leave unmatched waits even at `preload=0`; increasing slots or changing memory layout cannot restore matching participation. The installed backend's `get_block_idx` / `get_block_num` lowering and the [subblock capability probe](support.md#5-get_subblock_num-depends-on-whether-the-kernel-has-a-cube-section) establish these distinct index/count semantics.

**Wrong form:** infer “launch is also thread-based” from “Vector-side `get_block_idx()` returns the flattened AIV thread index,” then pass `2×core_count`. It does not report an error; it really launches twice as many groups. Extra ghost groups run the same code with `core_id ≥ core_count` and store past `out` into GM rows `[M, 2M)`:

- The first call's output may **still be correct** (real groups fill `out`), while the ghost groups write into someone else's memory.
- The out-of-bounds destination depends on allocator layout and drifts with `out` size: it may appear “harmless” with small outputs, then overlap an adjacent tensor (if it is an input, later calls read dirty data and fail deterministically) or an unmapped page (507015). Therefore, the shape boundary between “works” and “fails” is not a protocol threshold.
- Symptoms are independent of kernel form, synchronization layer, or artifact switch and **look like a concurrency/synchronization defect**. When the first call is correct but later calls deterministically fail, check group count first, then diff inputs, and only then revisit synchronization theory.

### Layer B with a GM Relay: Declare It with `pl.bind_crosscore_sync` (Since 2026-09-26)

The GM buffer is a `pl.Tensor` argument, **not a TileGroup, so there is nowhere to put `fwd_ids`** (`move` also has no DDR path, and a DDR TileGroup cannot be used). Declare it with a separate statement:

```python
@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=2))
def kernel(a, b, ws: pl.Tensor[[M, N], pl.DT_FP32], out):
    pl.bind_crosscore_sync(ws, ids=[3, 4])        # written outside all sections
    ...
```

Write this inside the kernel body but outside every section—the same place as a shared `make_tile_group`. Then pass `ws` to both stages that share it.

There is only one parameter, `ids`: **one ID per slot**, each in 0..15, and **the same IDs are used for both forward and reverse directions**. **The number of IDs expresses depth**—it is also the number of concurrently live workspace buffers and the count `_check_slot_counts` uses to validate preload. There is no separate `depth=`.

Why one ID cannot guard multiple slots is explained below in “one ID per slot.”

**Pipes on both sides are still inferred and are different:** producer `store` reads from Acc → **FIX**; consumer `load` reads from GM → **MTE2**. You do not need and cannot specify them manually.

**One Tensor represents one handover**—the scanner matches the whole buffer, not a sliced region chosen by offset. For two handovers, pass two workspace parameters.

**Before `bind`, the handover was silent** (measured 2026-09-25; this was also why the API was added): a two-stage kernel (cube stores to GM, vector loads the same region, no cross-core buffer declared) compiled with both `preload=0` and `preload=1`, but inserted **zero** `set_cross_core` / `wait_cross_core` calls; `preload=1` still staggered the stages. The cause was `if not cross_buffers: return None` in `_analyzer.py`—“no cross-core buffer declared” was treated as “no cross-core handover.” The regression pins both forms: add or remove that one line in the same kernel and it goes from eight synchronization calls to zero (`pypto/python/tests/st/pypto_pro/frontend/auto_pipeline/test_gm_handover.py`).

See [api/crosscore-contract.md](crosscore-contract.md) for design and tradeoffs.

### Layer B with an Explicit Tile Group

Declare the physical addresses, automatic mutex IDs and cross-core event IDs separately:

```python
shared = pl.make_tile_group(
    type=pl.TileType(shape=[HALF_M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
    addrs=[UB_SLOT_0, UB_SLOT_1], mutex_ids="auto", depth=2,
    fwd_ids=[0, 1], bwd_ids=[0, 1],
)
```

Choose event IDs from the kernel's available pool and check simultaneous handovers for conflicts. Use one event ID per slot in both directions. The mutex planner governs within-core dependencies; the pipeline transform reads the explicit event lists to generate cross-core synchronization.

### Part / `sync_all` / Part (Two-phase Mega-kernel; Board Evidence: 2026-09-30)

A flash-decoding-style “pipeline stage → all-core merge stage” can be written **inside one `@pl.jit` body**: phase 1 is a normal B-layer pipeline loop, followed by a whole-core `pl.system.sync_all` barrier and a plain phase 2. Board evidence: `decode_attention_gqa_fd` in [decode_attention_gqa](../../examples/algorithms/decode_attention_gqa/kernel.py), the first in-repository board-verified two-phase `sync_all` / GM partial-state merge. Five rules:

1. **Both sides of the MIX barrier must arrive** (framework ST call contract): in a mixed kernel, call `pl.system.sync_all(core_type=pl.SyncCoreType.MIX)` once in `section_cube` and once in `section_vector`; do not call it on one side only.
2. **Call `bar_all` on the producer side before entering the barrier**: `sync_all` synchronizes core arrival but does not drain the pipeline. Data written to GM by phase 1 via MTE3 (partial-state workspace) must first drain that core's MTE3 with `pl.system.bar_all()` for the visibility guarantee after the barrier to hold.
3. **The B-layer transform preserves code after the pipeline loop**: put phase 2 sections after the loop; the transform leaves them after drain and executes them normally. No second kernel is needed.
4. **A GM buffer with “write first, then read” ordering needs no initialization when separated by the barrier**: phase 1 writes only and phase 2 reads only; every record read has a writer before the barrier.
5. **Writing a crosscore buffer inside runtime control flow is a safe form**: an actual write to a cross-core group inside a stage may be conditional (for example, only subblock 0 calls `pl.insert` under `if pl.get_subblock_idx() == 0:`). Set/wait are inserted at stage-call boundaries, outside the branch; slot accessors (`.next()`) are called unconditionally, so synchronization accounting is independent of whether the branch runs.

## One ID per Slot: One ID Cannot Guard Multiple Slots

`make_tile_group` accepts `fwd_ids=[0]` with three slots—the framework accepts it, and the docstring says this “trades parallelism for IDs.” **The statement about parallelism is correct; the statement that it does not affect correctness has a counterexample** (board-verified 2026-09-26):

| ID configuration | Number of consumer pipes touching the buffer | Result |
| --- | --- | --- |
| One ID per slot | 1 or 2 | Correct. |
| **One ID for N slots** | **2** | **Silently wrong**, `max\|diff\|` 0.345 (the signal's upper bound is 0.333); the failing task differs every run. |
| One ID for N slots | 1 | Correct. |

**The criterion is “how many pipes touch the buffer,” not “whether it is updated in place.”** The consumer below never modifies `slot`; it reads it once on V and once on MTE3, and sharing the ID still fails:

```python
slot = shared.next()
_scale_vf(tmp, slot, 2.0)        # @pl.vector_function helper: V reads slot
pl.store(out2, slot, [...])      # MTE3 reads slot
```

Reason: if the consumer touches N pipes, it emits N releases (`_gating_readers` adds one edge per pipe). On the same counter, these N releases can be exchanged with releases for the **next task**. The producer's N waits may then see one release from each of two tasks and overwrite a slot while the preceding consumer's other pipe is still reading it.

**The framework now catches this** (`_check_shared_id_covers_one_release`) with a message that says “give each slot its own ID.” So the ID-saving form remains usable when the consumer touches one pipe, and the compiler checks it for you.

## Three Rules for Hand-written C-layer Synchronization

### 1. The Pipes May Differ; Use the Pipe of the Actual Data-touching Op at Each End

- Producer `set`: use the pipe of the **last op that writes the data**.
- Consumer `wait`: use the pipe of the **first op that reads the data**.

`wait_cross_core` **blocks only the queue named in the call**; specifying the wrong one fails **silently**.

Common combinations:

| Handover medium | Producer side | Consumer side |
| --- | --- | --- |
| GM workspace (cube writes, vector reads) | `store` source is in Acc → **FIX** | `load` from GM → **MTE2** |
| UB (cube hands off from L0C using `move`) | Source is in Acc → **FIX** | First consumer op → usually **V** |
| L1 (vector hands back to cube using `insert`) | UB → L1 → **MTE3** | `move` Mat → Left → **MTE1** |
| L1 (cube then `move`s into **Scaling/FB**, such as fixpipe per-channel payload) | UB → L1 → **MTE3** | Mat → Scaling move → **FIX** (not MTE1! A wrong pipe has a treacherous symptom: the first launch reads an unready payload; later launches see the correct stale value left by the previous launch and a one-off check may pass. Board evidence 2026-09-29, quantization.md). |

These are common cases only; see the complete table of **data-movement op pipelines** (`pl.move` branches for every source/destination space, store/load/insert, and descriptor-only exceptions) in [api/datamove.md](datamove.md) section 15. Always look up the pipe for a C-layer call; do not guess.

Pipe responsibilities (official `PipeType.md`):

| Pipe | Responsibility | Side |
| --- | --- | --- |
| MTE1 | L1 → L0A / L0B / BiasTable | Cube |
| MTE2 | GM → L1 / UB | Both sides |
| MTE3 | UB → GM, UB → L1 | Vector |
| M | Cube / MAD matmul | Cube |
| V | Elementwise / reduce / cast / quant | Vector |
| S | getval / setval | Both sides |
| FIX | L0C → GM / UB / L1, on-path quantization | Cube |

### 2. For Reusing a Buffer in a Loop, Forward Synchronization Is Not Enough

The official documentation says:

> When the same Buffer is reused in a loop, forward synchronization for “data ready” alone is insufficient; reverse synchronization for “data consumed” is also needed, or neighboring iterations must use different Buffer slots.

**The requirement is explicit, but the primitives do not implement it for you**—you must pair the reverse direction with another event ID.

A one-shot handover that writes once and reads once, outside a loop, needs only forward synchronization.

### 3. Choose `sync_mode` from the Actual Producer and Consumer; Do Not Ask Only “Is It Cross-core?”

| Mode | Value | Between |
| --- | --- | --- |
| `INTRA_BLOCK` | 2 (default) | AIC ↔ **both** AIVs in the same block |
| `UNICAST_BLOCK` | 3 | AIC ↔ one AIV |
| `INTER_SUBBLOCK` | 1 | AIV0 ↔ AIV1 |
| `INTER_BLOCK` | 0 | Like cores across blocks; **does not create an AIC↔AIV dependency** |

The default `INTRA_BLOCK` means **the vector-section code runs on both AIVs**—use `pl.get_subblock_idx()` to split data, or both AIVs compute the same work.

## Silent Wrong-result Patterns

None of these forms reports an error:

| Form | Symptom |
| --- | --- |
| `wait` pipe is set to the transfer queue, but the consumer's first op runs on V. | That op reads bytes before they have arrived. Measured: the maximum difference is exactly the omitted item; IR still contains the op and all mutexes. |
| Reuse a buffer in a loop with only forward synchronization. | Producer overwrites data while the consumer is still reading it. |
| Rely on `section_cube()` appearing before `section_vector()` in source to guarantee order. | The two cores run concurrently; source order is not synchronization. |
| **Reuse the declared output for a one-shot C-layer GM relay followed by VF computation** (cube writes to `out`, vector reads it, computes, then writes back to `out` in the same kernel). | The recorded probe produces extensive **nondeterministic** errors (`exp(1)` collapses, `inf`). A pure `vf.move` passthrough does not trigger it. The conditions and evidence limits are given below. |

Using the same ID in forward and reverse directions is **safe** (board-verified): a buffer's `fwd_ids` and `bwd_ids` may use the same ID because direction is part of the hardware counter key. The dangerous cases are the other two, both now caught by the framework: **same ID and direction across buffers**, and **one ID guarding multiple slots** (above).

### Output reuse across handover boundaries

**Board evidence for the output-tensor reuse case** (2026-09-28, Ascend950PR, pypto board build `8faa853`): the failing probe combined (1) one-shot C-layer handover, (2) relay GM equal to the declared output Tensor, and (3) real VF computation in the consumer. Cube gemm+bias alone, VF silu alone without cross-core handover, and a `vf.move` passthrough with the same handover structure each matched bitwise; the combined computation with reused `out` produced different wrong-value patterns on two runs. **Use an independent `ws` parameter for this relay**: the same vector computation with separate same-shape scratch matched bitwise. The mechanism was not located to a source line; an automatic path for output Tensors remains a hypothesis, not an established framework defect. Preserve these probe conditions when rechecking another toolchain.

This observation does not prohibit using the output as an intermediate across separate ordered native launches. [gemm_rmsnorm](../../examples/algorithms/gemm_rmsnorm/kernel.py) writes the Cube product to `out`, then a second native task on the same Stream reads and overwrites it with the final normalization. That task-order boundary has a different completion argument from an in-kernel cross-core relay. Other in-kernel reuse forms still need their own ownership, completion and numerical evidence; neither this example nor the failing C-layer probe establishes them.

## Examples

C layer (hand-written `set_cross_core` / `wait_cross_core`):

| Example | Coverage |
| --- | --- |
| [`examples/matmul/fundamental_usage/matmul_add_chain/`](../../examples/matmul/fundamental_usage/matmul_add_chain/test.py) | Two round trips, four events and four pipes, including reverse vector → cube handover. |

B layer (`pl.jit(pipeline=)`, with synchronization inferred by the framework). Each example uses two independent kernels for `preload=0` and the shipping setting and compares them bitwise:

| Example | Coverage |
| --- | --- |
| [`examples/algorithms/gemm_bias_silu_banded/`](../../examples/algorithms/gemm_bias_silu_banded/kernel.py) | **Minimal 1+1 topology**; Acc→Vec on-chip transfer (`DualModeSplitM`), explicit per-slot `fwd_ids` / `bwd_ids`. |
| [`examples/algorithms/mm_softmax_mm/`](../../examples/algorithms/mm_softmax_mm/kernel.py) | Three-stage c-v-c pipeline with softmax; two handovers: Acc→Vec direct transfer + Vec→Mat **on-chip L1** (cross-core NZ group, `pl.insert` writes each subblock half, cube consumes it as the left operand to `pl.gemm`). |
| [`examples/algorithms/flash_attention/`](../../examples/algorithms/flash_attention/kernel.py) | Full c-v-c-v chain (including the v→c return hop and same-core state ring); slot counts use the minimums in the multi-handover table above. |

See “Layer B with a GM relay” above for use of the GM workspace route (`pl.bind_crosscore_sync`); board numerical regression is on the framework side in `test_gm_handover.py` (separate kernels for `preload=0` and `preload=2`, each run twice and compared bitwise).

Layer B framework regressions are under `pypto/python/tests/st/pypto_pro/frontend/auto_pipeline/`: `test_gm_handover.py` (GM + `bind_crosscore_sync`), `test_pipelined_crosscore_handover.py` (explicit addresses, event IDs and preload), `test_shared_event_id.py` (one ID per slot rule), and `test_cross_core_id_collisions.py` (same-ID collision across buffers).

First-hand framework sources: `pypto/python/tests/st/pypto_pro/frontend/auto_pipeline/` (Layer B, four files and seven tests, including serial and overlap variants of FA), `.../matmul/test_matmul_softmax.py`, and `.../matmul/test_matmul_add_matmul_add.py` (Layer C).

Official documentation: Cube-Vector Fusion (`pypto/docs/zh/guide/programming_guide/pro/development/cube_vector_fusion.md`), Automatic CV Pipeline Parallelism (`pypto/docs/zh/guide/programming_guide/pro/advanced_programming/auto_parallel_pipeline.md`).

## A Naming Note

The word `pipeline` has two meanings in Pro, and **neither means** “double buffering overlaps transfer and compute”:

- `pl.jit(pipeline=)` — cross-core pipeline parallelism (Layer B).
- `PipeType.MTE2` / `V` / `FIX` — within-core execution queues (a Layer C parameter).

Double-buffer overlap occurs **within one core**, because two slots carry different mutex IDs. Call it `double_buffer`.
