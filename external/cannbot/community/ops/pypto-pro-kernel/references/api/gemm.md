# `pl.gemm`: Responsibilities and Maintenance Contract

Implemented. Use the framework gemm.md (`pypto/docs/zh/api/pro_api/SIMD-API/cube_computation/gemm.md`) and `language/_api.py::gemm` as the interface authorities. This page preserves tradeoffs that affect correctness and future maintenance.

## Division of Responsibilities Between the Caller and the Sugar

`pl.gemm` expands at the AST layer into ordinary `move`, `set_validshape`, and `matmul` / `matmul_acc` operations; it adds no IR op. Expansion must happen before automatic mutex analysis, not in an op handler that emits IR directly and bypasses the synchronization path.

- The caller manages L1 loads, L0 staging addresses/IDs/depth, and output stores.
- `acc` is both the destination and the accumulator; `init=False` accumulates in place and does not represent independent dst and acc operands.
- L0 arguments may be Tiles or TileGroups. A group's `.next()` advances inside each generated loop iteration.
- Non-named operands are evaluated only once. Transpose and splitting/extent forms need named matrix operands to inspect their shapes: select `a = a_group.current()` and `b = b_group.next()` before `pl.gemm(acc, a, b, ...)`. Re-evaluating an inline accessor to discover its shape could rotate a group twice. This does not change the `l0a=` / `l0b=` group convention above.
- `bias` is used only on the initialization step. When combined with `splitk`, do not add bias on every step.
- The sugar does not take ownership of L0/BT or generate `phase` automatically; see [gemm_mx](gemm-mx.md) for MX.

Plan the final drain from the consumer's dtype and unfinished arithmetic. Before adding an FP32 GM result and a separate output-cast kernel, check the [direct output conversion path](fixpipe.md#choose-the-final-output-path-before-adding-a-conversion-kernel); FP32 accumulation does not require FP32 storage.

## Inputs split along the reduction axis

Distinguish mathematical concatenation from required storage. If matching operand components partition the reduced feature axis, `concat(A0, A1) @ concat(B0, B1).T = A0 @ B0.T + A1 @ B1.T`. A bounded output tile can therefore accumulate the component products directly: initialize its FP32 Acc on the first `pl.gemm`, then use `init=False` for the remaining products. Choose legal L1/L0 shapes and K steps for each component, including a component shorter than the preferred split. Apply a common scale, normalization or other nonlinear consumer only after the complete sum, preserving any task-required intermediate rounding.

This identity does not apply to concatenation along a retained output axis, nor does it remove component-specific weights from the contract. Explicit packing is still a performance candidate when it improves reuse, but it adds a transport and resource obligation. An API named `insert` does not imply arbitrary memory-space copies: establish the supported source/destination layout pair before choosing a concatenation buffer. Compare complete calls, including packing, rather than treating a reference's `concat` operation as a mandatory native kernel or L1 copy.

## Effective Dimensions and Splitting

`m/n/k` narrow the Tiles used directly by matmul; they do not narrow L1 tiles that have already been loaded. The caller must decide the L1 valid region before loading. Narrowing is not undone later, so the subsequent store depends on the accumulator's valid region.

| Axis | Objects narrowed |
| --- | --- |
| `m` | L0A rows and acc rows |
| `n` | L0B columns and acc columns |
| `k` | L0A columns and L0B rows |

**Only Tiles that are narrowed require `compact=1`.** This keeps `move`, `matmul`, and `store` consistent about physical strides. Do not extend the requirement to Tiles that are not narrowed, or apply it to a hand-written route that narrows acc only after matmul and uses the extent only to limit store.

First distinguish which layer owns each parameter: put `compact=1` in declarations such as `TileType`; put `m/n/k` at the `pl.gemm` call. Do not read the error phrase “declared with compact=1” as a request to add a keyword to `gemm`. In behavior version `872df763f`, neither the `gemm/gemm_mx` signatures nor `_gemm_lowering.py::_gemm_keywords` accept `compact`; passing it raises `unexpected keyword 'compact'` during parsing. C++ `type_converter.cpp::ConvertCompactModeToPTOValue` derives the physical type from the tile's `HardwareInfo.compact`; inspect that layer when checking the behavior. An allocated tile also does not automatically change its physical layout because one call needs a tail block.

| Combination | Contract |
| --- | --- |
| `splitk` | Split from L1; on the first step, `init` selects initialization or accumulation, and later steps accumulate; the final step narrows L0. L1 declared capacity and L0 split shape must still match; see below. |
| `splitn` | Emit strips along acc's N fractal and inherit the parent acc mutex ID; the parent acc must identify its slot. |
| `splitn` tail strip | `splitn % 16 == 0`; tail width is `align16(tail)`, and a full-width slot that extends past the parent buffer must not be reused. |
| `m` + `splitn` | Rejected. Narrowing M changes the fractal spacing and invalidates strip addresses computed from the declared M. |
| `n` + `splitn`, `k` + `splitk` | The extent becomes the loop bound and must be known at compile time; other extents may be runtime scalars. |
| `splitn` + `splitk` | Mutually exclusive; write the loop in the caller when two-level splitting is needed. |

**A whole range shorter than one split and a tail in the final split are separate tests.** Narrowing the last `splitk` step does not guarantee that an L1 tile declared smaller than L0 can be moved into that L0. A parameter factory must jointly choose physical L1/L0 sizes and split step, or explicitly plan enough L1 capacity and handle padding; changing only valid-shape is insufficient. In behavior version `872df763f`, a failure with a small L1 tile and large L0 may appear as `insert: unsupported layout/dtype combination src=Mat(NZ,DT_FP16) -> dst=Left(NZ,DT_FP16)` even when the dtype itself is correct; check the declared source and destination sizes first. Acceptance must separately include an axis shorter than a split, exactly equal to a split, and a non-divisible tail after one full split, especially when all public examples are larger than the selected split.

In a parameter factory, distinguish the preferred tuning step from the actual step used this time. If the L1 axis capacity comes directly from the input extent and both capacity and preferred step meet the granularity required by the dtype/path, use `step = min(preferred_step, declared_axis_capacity)` and apply it to both L0 declarations and `splitk`; do not change only the call argument while keeping the old L0 shape. If a larger fixed step is required, declare enough physical L1 capacity and prove the valid range, padding, and compute extent separately; do not mix the two strategies. Alignment requirements still need an independent check: `min` does not make an invalid granularity legal.

**Setting L1 valid shape does not change the default `splitk` loop bound.** The verified lowering takes that bound from the declared K unless the call supplies a compile-time `k=`. A short final outer panel can therefore load only its logical K and still compute the whole physical capacity, consuming stale or uninitialized values. When Host-known shapes determine the final extent, specialize the GEMM factory by that extent and pass it as compile-time `k=`, retaining valid physical L1/L0 sizes. Alternatively, explicitly make every consumed padded reduction lane neutral and verify that path. Removing an unsupported runtime `k=` is not a fix for this mismatch. Test a short final panel after at least one full panel, with nonzero distinguishable earlier data and changed/poisoned storage; exact-divisibility cases and an isolated small GEMM miss the stale-state failure. At behavior version `872df763f`, installed `_gemm_lowering.py::_gemm_splitk` and a device control agreed: changing only the static K bound corrected the failed outer-tail reduction.

Transpose is represented by a call-site alias using `pl.reinterpret(..., layout=pl.ZN)`; it does not modify the caller's declaration or generate an instruction. Therefore comparing only the op sequence cannot verify transpose; inspect the shape/layout read by `move` and verify the numerical result.

## From Capacity Planning to Performance Candidates

Fitting all of K in L0 is a correctness starting point, not a reason to keep all of K there. Jointly choose `M_b/N_b/K_b`: reducing the K staging block may allow a wider N block or deeper slots, but adds accumulation steps; increasing M changes effective parallel work and tail frequency. Budget each resource by dtype bytes and slot count rather than checking only whether one tile fits:

| Space | Main bytes per slot | Consider together |
| --- | --- | --- |
| Left | `M_b × K_b × sizeof(A)` | M reuse, K step count, M tails |
| Right | `K_b × N_b × sizeof(B)` | N strip count, K step count, weight transfers |
| Acc | `M_b × N_b × sizeof(acc)` | Accumulation lifetime, number of valid results, number of drains |

Total capacity also includes alignment, padding, all active groups, and each group's slot count; see [capacity accounting](memory-limits.md). Starting from a baseline that keeps all of K resident but uses a narrow N, compare K splitting with wider N blocks and different M blocks. There is no fixed block size: budget statically, then measure the complete call at a fixed shape and worker count. `splitk` handles K splitting from L1 to L0; the caller still owns reuse from GM to L1 and the outer N/M loops.

Name the resident L1 K panel and the L0 K step separately. With the same M/N compute tile, `K_L1 > K_L0` can reduce GM load calls and change transfer granularity without increasing L0 capacity; it does not by itself reduce total operand bytes. Budget L1 A/B slots with `K_L1`, Left/Right slots with `K_L0`, and Acc with its live output tiles. A failed allocation using one shared K parameter bounds that allocation, not all hierarchical blocking. Test a K shorter than either preferred block and a short outer panel after a full panel using the [static split extent contract](#effective-dimensions-and-splitting). The [grouped GEMM example](../../examples/algorithms/grouped_gemm_swiglu_quant/) varies the two K sizes independently.

When a wider N block forces a smaller L0 K step, compare two independent N accumulators sharing an unchanged L0A operand. Each Right tile can keep the narrower N width and larger K step, while the pair saves repeated Left staging. Budget both Acc tiles and the wider L1 B panel; the pair has fewer logical work items and may worsen the busiest worker's work. This is the [retained-L0 exception](../patterns/gemm-desugar.md#the-only-stated-reasons-to-hand-write-closed-list-source-verified), not permission to bypass mutexes. `paired_gemm_kernel` in the grouped example implements this control, including a missing second N tile and partial M/K/N extents. Re-test paired and unpaired blocks after changing across-group ownership; an earlier losing geometry does not close the new configuration's search.

Also count GM reads and repeated work. An outer M loop with N inside may reuse A while rereading B for each M block; an outer N loop has a different reuse and repetition pattern. Track tile count, the busiest worker's tile count, repeated A/B bytes, GEMM calls, and drains together; avoid reducing K steps only to add more narrow strips or GM rereads. A downstream full-row reduction needs the final full row, but Cube need not use the same physical block width; producer and consumer may tile separately and connect through a defined intermediate tensor and dependency. Preserve the original numerical tolerance for every change, especially when K splitting changes reduction order; test cancellation inputs explicitly.

### Count the complete output grid before assigning workers

For each GEMM, enumerate the independent output tiles across batches/groups and both retained axes M and N. For a uniform group, `M_tiles = ceil(M / M_block)` and `N_tiles = ceil(N / N_block)`; a Cube-only control assigns flattened tile IDs with `pl.range(pl.get_block_idx(), M_tiles * N_tiles, pl.get_block_num())`, then recovers `m_tile = tile_id // N_tiles` and `n_tile = tile_id % N_tiles`. Mixed kernels use [execution-group indexing](crosscore-sync.md#a-mixed-kernels-block_dim-is-a-group-count-launch-semantics--index-semantics). Extend ownership across groups when needed. Reducing K requires a different partial/merge contract; output-axis partitioning does not.

Compare nonempty owners with the worker budget before tuning inner blocks. An M-only assignment that serially visits N can leave workers idle despite a large full grid; increasing launch count alone cannot repair it. Reuse-oriented owners of neighboring tiles remain candidates, but compare their repeated operand bytes and longest worker path against the complete-grid control at fixed workers.

For a small tile grid, compare the busiest worker before splitting merely to reach the worker budget. With `T` equal tiles and `C` workers, that worker executes `ceil(T/C)` tiles. Halving a tile while doubling `T` can create a second iteration without reducing its useful work, while adding loads, synchronization and lost operand reuse. Compare `ceil(T/C) * work_per_tile` and repeated bytes across candidate geometries at fixed workers; some idle workers can be cheaper than a longer critical path. This is a screening estimate, not a hardware timing model: also measure bandwidth/concurrency effects and recheck the final configuration's neighbors. The [bounded MLA factory](../../examples/algorithms/mla_bounded/kernel.py) applies this estimate separately to its matrix and VF phases.

After fixing geometry, choose launch candidates around changes in `ceil(T/C)`, including the nonempty-owner count when `T` fits within the available budget. Adjacent worker counts can leave the same assignment unchanged, so they may miss the relevant boundary. Count each phase separately. If a factory also uses workers to choose its tiles, inspect those realized tiles before attributing the difference to launch count alone.

Retune blocks after choosing ownership and apply this comparison separately to each product in a chain. Use thin-wide, tall-narrow and mixed-tail controls, timing all loads, computation and drains; a valid tile size or a fast square workload does not establish a good mapping for another aspect ratio.

## Thin-row contractions: compare arithmetic and transport paths

A matrix-shaped expression does not settle which core should execute it. For a very small M and a wide retained N, compare a bounded VF dot-product path with the Cube control. Keep the activation in FP32 registers when the contract requires FP32 intermediates, widen BF16 weights, and accumulate along K with explicitly initialized FMA destinations. A Cube path representing one activation with several BF16 components performs several products against the same weight; those extra compute and L0 staging operations belong in the comparison even when GM weight traffic is unchanged. Revalidate reduction order and cancellation against the original evaluator.

For VF, choose the output width, rows sharing each loaded weight, K transfer panel and slot depth jointly. Count live accumulators as `rows × output_width / FP32_lanes`, then include weight conversions, broadcasts and output temporaries. More row reuse reduces repeated weight loads but increases sequential arithmetic and register demand; a slow wide prototype does not reject a narrower VF mapping. Inspect generated code and measure controlled neighbors before attributing a regression to register spills. Keep a correct fallback for other M values and measure a complete call, including any changed packing or downstream fusion. The [MLA prolog example](../../examples/algorithms/mla_prolog/kernel.py) provides FP32 Vector projection and fused per-head consumer paths alongside Cube paths.

Core choice and transfer path are separate decisions. When Cube remains suitable but its GM→Mat path dominates, compare AIV GM→Vec loading, ND→NZ conversion and Vec→Mat insertion followed by the same Cube arithmetic. Count the added conversion, synchronization and on-chip copies; this moves transport work and does not itself remove weight bytes. Establish the [two-stage handover](crosscore-sync.md#layer-b-still-needs-a-driver-loop-even-for-one-handover) at `preload=0`, inspect its emitted forward/reverse events, then measure overlap. A losing implementation or a high MTE2 ratio does not establish an HBM ceiling.

## Transformed operands: format, residency and repeated work

When VF produces a matrix operand, design its format from the Cube consumer backward. Memory space and layout are separate: allocating in Mat does not make an ND tile a legal source for the NZ/ZN L1-to-L0 extraction on the verified A5 FP16 GEMM path. Verify a complete conversion-through-GEMM slice, not just the transformed values in Vec. A supported ND-to-NZ conversion followed by L1 insertion is a useful control before direct VF NZ packing; the latter needs its own container, predicate and physical-pitch proof in [cross-core transfers](crosscore-sync.md). Keep all prescribed rounding at the producer boundary.

Choose the VF transfer block, resident L1 panel and L0 compute step independently. A transformed panel may fit L1 while exceeding Vec capacity: produce bounded Vec fragments and insert them into the resident L1 panel, with a completion boundary before Cube consumes it. This can avoid a full GM intermediate without shrinking every DMA/VF call to one register. Account for both transformed representations during layout conversion, all operand buffers and all handover slots using [capacity preflight](memory-limits.md). Check physical load width and padding separately from the consumer's narrower logical width.

Apply the [complete output-grid comparison](#count-the-complete-output-grid-before-assigning-workers), counting producer work as well as GEMM work. Transforming B separately for each M owner repeats its conversion, metadata loads and synchronization. Compare sharing one produced panel across consumer tiles with separately produced panels, including operand rereads, lost parallel work and the longest owner path.

Choose residency together with ownership. On-chip production removes intermediate GM traffic and launches, but tying producer and consumer to one group can reduce independent work or introduce more handovers. A materialized panel lets phases use different grids and may improve reuse or distribution. Compare active owners, the longest owner's work, live bytes, repeated conversion and transfers for both routes; fitting in L1 alone is not a reason to choose fusion first. Start the complete correctness baseline from the suitable verified interfaces within the task budget, then preserve it while testing the alternative.

When testing fusion, establish a small same-execution-group producer/consumer slice, possibly filling L1 through bounded Vec fragments. Validate one complete panel through Cube before expanding the shape matrix. Assign complete output panels to groups so local conversion and Cube consumption can share storage; use [automatic cross-core stages](crosscore-sync.md) for completion and buffer lifetime. Compare complete calls at the same public contract and recheck available work after fusion. A failed or slower fused attempt does not invalidate the working materialized baseline; scope the result to the actual mapping, rather than treating either residency choice as universally faster.

[On-chip grouped W4A16](../../examples/algorithms/quant_gemm_w4a16_onchip/) provides a complete bounded-fragment conversion and same-group L1 handover. Its contract and resource plan live in the kernel header. Its current M-only ownership is a reuse choice with known thin-M underutilization, not a default mapping for every transformed contraction.

## Distribute tiles across runtime groups

Count work across the complete call, not just within one group. Restarting each group's tile loop at `worker_id` assigns short groups and remainders repeatedly to the same low-numbered workers. Sum each worker's tile count and operand transfers over all groups, including empty groups and mixtures of short and long groups; choosing a smaller M block to fill one group can increase calls and reloads across the whole workload.

Compare a continuous logical tile stream. All workers first derive the same group tile shape/count from runtime metadata; a device-side prefix of preceding groups' tile counts gives each local tile a global number. Assign that global number modulo the worker count, and advance the prefix even when this worker owns no tile in the current group. Empty groups contribute zero. This can rotate short-group ownership without a Host readback or a separate GM worklist. Preserve operand reuse among one worker's tiles of the same group, and prove disjoint, complete output coverage.

For a group with `count` tiles and preceding prefix `prefix`, the first local tile is `(worker + workers - prefix % workers) % workers`, followed by strides of `workers`. The nonnegative expression avoids relying on a language's negative-remainder convention. For a row phase, use the group's starting global row as the prefix. Check each phase independently: balancing Cube tiles does not balance Vector rows. Before preloading per-group operands, check whether this worker has a consumer; use the [empty-consumer lifetime check](../diagnostics/validation-traps.md#empty-consumer-loops-need-their-own-reuse-check), not just a global nonempty-group guard.

Report nonempty workers and the minimum/maximum task counts across the complete call, then examine tail sizes, repeated bytes and the longest worker's work. The [grouped GEMM/SwiGLU/quantization control](../../examples/algorithms/grouped_gemm_swiglu_quant/) keeps arithmetic and geometry fixed between per-group restart and continuous numbering; its benchmark prints both Cube and Vector counts and validates both outputs before paired timing. The source check `nested-worker-restart` points to a possible repeated assignment; it neither knows runtime group sizes nor forbids a reuse-oriented owner.

Retune M/N/K blocks after changing ownership: a large block rejected because most workers were idle in every individual group may become useful when groups share the worker pool. Balanced tile counts alone do not prove equal time when tails or shapes differ; compare the busiest worker's estimated arithmetic, transfers and metadata cost, then validate and time complete calls on uniform, concentrated and mixed-length groups.

## Regression Entry Points

Implementation: `language/parser/_gemm_lowering.py`.

- `python/tests/ut/pypto_pro/language/parser/test_gemm.py`: expansion, rejected boundaries, synchronization, and declarations.
- `python/tests/st/pypto_pro/frontend/matmul/test_gemm_st.py`, `test_gemm_matrix_st.py`: intersections of extents, splits, transpose, bias, and runtime tails.
- This repository's [matmul examples](../../examples/matmul/): complete call patterns.

When comparing IR, normalize SSA names and process-local incrementing memref IDs; compare addresses, layouts, and synchronization rather than erasing them too. The measured board platform was Ascend950PR_957b; verify the target checkout and installation identity before applying these observations. Test entry points alone do not establish a pass on that installation.
